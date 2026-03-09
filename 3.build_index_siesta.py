#!/usr/bin/env python3
"""
build_index_siesta.py

Index a SIESTA run laid out like:
  ROOT/Al/cfg_0000/...
  ROOT/Fe/cfg_0000/...
  ROOT/Ni/cfg_0000/...

Produces:
  - index.jsonl (always)
  - index.parquet (if pandas+pyarrow available)

Parses from *.out (tail scan):
  - scf_converged, scf_iterations
  - final_energy_ev (from "siesta: Final energy (eV):" block)
  - fermi_ev (from "siesta: Fermi = ...")
Also tracks:
  - fatal_error (true abort markers)
  - dm_read_failed + dm_atomic_fallback (benign continuation miss)

Usage:
  python build_index_siesta.py raw_single_elements \
    --out index.jsonl --parquet index.parquet --include-extra
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RE_SCF_ITERS = re.compile(r"SCF cycle converged after\s+(\d+)\s+iterations", re.IGNORECASE)
RE_FERMI = re.compile(r"siesta:\s*Fermi\s*=\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)

RE_FINAL_ENERGY_HDR = re.compile(r"siesta:\s*Final energy\s*\(eV\)\s*:", re.IGNORECASE)

# Try several common SIESTA lines under the "Final energy" header:
RE_FINAL_E_LINE_CANDIDATES = [
    re.compile(r"siesta:\s*(?:Total|Etotal)\s*=\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bTotal\s+energy\b.*?([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bEtotal\b.*?([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
]

# Fallback: any float-looking token
RE_FLOAT = re.compile(r"([-+]?\d+(?:\.\d+)?)")

# "True fatal" patterns (do NOT include generic 'Failed' because DM continuation prints it benignly)
RE_FATAL = re.compile(
    r"("
    r"\bFATAL\b|"
    r"\bERROR\b|"
    r"MPI_Abort|"
    r"SIGSEGV|"
    r"Segmentation fault|"
    r"floating point exception|"
    r"siesta:.*\babort"
    r")",
    re.IGNORECASE,
)

# Benign DM continuation message
RE_DM_READ_FAILED = re.compile(r"Attempting to read DM from file\.\.\.\s*Failed\.\.\.", re.IGNORECASE)
RE_DM_ATOMIC_FALLBACK = re.compile(r"DM filled with atomic data", re.IGNORECASE)

RE_ETOT = re.compile(r"siesta:\s*Etot\s*=\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_FREEENG = re.compile(r"siesta:\s*FreeEng\s*=\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_EKS = re.compile(r"siesta:\s*E_KS\(eV\)\s*=\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)

# SCF table line (captures the Ef(eV) column)
# Example:
# scf:   12    -6820.622674    -6820.622674    -6820.626691  0.000005 -7.354516  0.000069
RE_SCF_LINE_EF = re.compile(
    r"^\s*(?:scf:\s*)?\d+\s+"
    r"[-+]?\d+(?:\.\d+)?\s+"
    r"[-+]?\d+(?:\.\d+)?\s+"
    r"[-+]?\d+(?:\.\d+)?\s+"
    r"[-+]?\d+(?:\.\d+)?\s+"
    r"(?P<ef>[-+]?\d+(?:\.\d+)?)\s+"
    r"[-+]?\d+(?:\.\d+)?\s*$",
    re.IGNORECASE,
)


def parse_out(out_path: Path) -> Tuple[
    Optional[bool], Optional[int], Optional[float], Optional[float],
    Optional[bool], Optional[bool], Optional[bool]
]:
    """
    Returns:
      scf_converged, scf_iterations, final_energy_ev, fermi_ev,
      fatal_error, dm_read_failed, dm_atomic_fallback
    """
    lines = read_tail_lines(out_path, max_lines=4000)
    if not lines:
        return None, None, None, None, None, None, None

    scf_converged = False
    scf_iters: Optional[int] = None
    fermi: Optional[float] = None
    final_e: Optional[float] = None

    fatal_error = False
    dm_read_failed = False
    dm_atomic_fallback = False

    # For fallbacks:
    last_etot: Optional[float] = None
    last_freeeng: Optional[float] = None
    last_eks: Optional[float] = None
    last_ef_from_scf: Optional[float] = None

    # 1) single pass over tail
    for line in lines:
        if RE_FATAL.search(line):
            fatal_error = True
        if RE_DM_READ_FAILED.search(line):
            dm_read_failed = True
        if RE_DM_ATOMIC_FALLBACK.search(line):
            dm_atomic_fallback = True

        m = RE_SCF_ITERS.search(line)
        if m:
            scf_converged = True
            scf_iters = int(m.group(1))

        m = RE_FERMI.search(line)
        if m:
            try:
                fermi = float(m.group(1))
            except ValueError:
                pass

        m = RE_ETOT.search(line)
        if m:
            try:
                last_etot = float(m.group(1))
            except ValueError:
                pass

        m = RE_FREEENG.search(line)
        if m:
            try:
                last_freeeng = float(m.group(1))
            except ValueError:
                pass

        m = RE_EKS.search(line)
        if m:
            try:
                last_eks = float(m.group(1))
            except ValueError:
                pass

        m = RE_SCF_LINE_EF.match(line)
        if m:
            try:
                last_ef_from_scf = float(m.group("ef"))
            except ValueError:
                pass

    # 2) parse final energy block (if complete)
    for i, line in enumerate(lines):
        if RE_FINAL_ENERGY_HDR.search(line):
            window = lines[i + 1 : min(i + 120, len(lines))]

            # Try known candidate lines first
            for wline in window:
                for pat in RE_FINAL_E_LINE_CANDIDATES:
                    mm = pat.search(wline)
                    if mm:
                        try:
                            final_e = float(mm.group(1))
                            break
                        except ValueError:
                            pass
                if final_e is not None:
                    break

            # Fallback: first float in the window
            if final_e is None:
                for wline in window:
                    if not any(ch.isdigit() for ch in wline):
                        continue
                    mm = RE_FLOAT.search(wline)
                    if mm:
                        try:
                            final_e = float(mm.group(1))
                            break
                        except ValueError:
                            pass
            break

    # 3) fallbacks for truncated "Final energy" block
    if final_e is None:
        # Prefer Etot, then FreeEng, then E_KS
        final_e = last_etot if last_etot is not None else (last_freeeng if last_freeeng is not None else last_eks)

    # 4) fermi fallback (if explicit "siesta: Fermi" missing)
    if fermi is None and last_ef_from_scf is not None:
        fermi = last_ef_from_scf

    return scf_converged, scf_iters, final_e, fermi, fatal_error, dm_read_failed, dm_atomic_fallback


# ---------------------------
# File sets
# ---------------------------

CORE_FILES = [
    ("out_path", "*.out"),
    ("fdf_path", "*.fdf"),
    ("xv_path", "*.XV"),
    ("struct_out_path", "*.STRUCT_OUT"),
    ("rho_nc_path", "Rho.grid.nc"),
    ("vh_nc_path", "ElectrostaticPotential.grid.nc"),
]

EXTRA_FILES = [
    ("dm_path", "*.DM"),
    ("hsx_path", "*.HSX"),
    ("eig_path", "*.EIG"),
    ("rho_path", "*.RHO"),
    ("vh_path", "*.VH"),
    ("orb_indx_path", "*.ORB_INDX"),
    ("kp_path", "*.KP"),
    ("bonds_path", "*.BONDS"),
    ("bonds_final_path", "*.BONDS_FINAL"),
    ("ion_path", "*.ion"),
    ("ion_xml_path", "*.ion.xml"),
    ("psml_path", "*.psml"),
]


# ---------------------------
# Data model
# ---------------------------

@dataclass
class Row:
    calc_dir: str
    element: Optional[str]
    calc_id: str

    scf_converged: Optional[bool]
    scf_iterations: Optional[int]
    final_energy_ev: Optional[float]
    fermi_ev: Optional[float]

    fatal_error: Optional[bool]
    dm_read_failed: Optional[bool]
    dm_atomic_fallback: Optional[bool]

    files: Dict[str, Optional[str]]


# ---------------------------
# Helpers
# ---------------------------

def read_tail_lines(path: Path, max_lines: int = 2000, tail_bytes: int = 4_000_000) -> List[str]:
    """
    Read last chunk of the file (up to tail_bytes), return last max_lines lines.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > tail_bytes:
                f.seek(-tail_bytes, os.SEEK_END)
            data = f.read()
        text = data.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        return lines[-max_lines:] if len(lines) > max_lines else lines
    except OSError:
        return []

def first_match(calc_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(calc_dir.glob(pattern))
    return matches[0] if matches else None


def detect_element_from_path(calc_dir: Path, root: Path) -> Optional[str]:
    try:
        rel = calc_dir.relative_to(root)
        return rel.parts[0] if len(rel.parts) >= 2 else None
    except ValueError:
        return None


def make_calc_id(calc_dir: Path, root: Path) -> str:
    try:
        rel = calc_dir.relative_to(root).as_posix()
    except ValueError:
        rel = calc_dir.as_posix()
    return rel.replace("/", "__")


def iter_calc_dirs(root: Path) -> Iterable[Path]:
    """
    Expects root/{Element}/cfg_*/
    """
    for elem_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for cfg_dir in sorted(elem_dir.glob("cfg_*")):
            if cfg_dir.is_dir():
                yield cfg_dir


def build_rows(root: Path, include_extra: bool = False) -> List[Row]:
    rows: List[Row] = []
    wanted = CORE_FILES + (EXTRA_FILES if include_extra else [])

    for calc_dir in iter_calc_dirs(root):
        element = detect_element_from_path(calc_dir, root)
        calc_id = make_calc_id(calc_dir, root)

        files: Dict[str, Optional[str]] = {}
        for key, pat in wanted:
            fp = first_match(calc_dir, pat)
            files[key] = str(fp.relative_to(root)) if fp else None

        scf_converged = scf_iters = final_e = fermi = fatal_error = dm_read_failed = dm_atomic_fallback = None
        out_rel = files.get("out_path")
        if out_rel:
            scf_converged, scf_iters, final_e, fermi, fatal_error, dm_read_failed, dm_atomic_fallback = parse_out(root / out_rel)

        rows.append(Row(
            calc_dir=str(calc_dir.relative_to(root)),
            element=element,
            calc_id=calc_id,
            scf_converged=scf_converged,
            scf_iterations=scf_iters,
            final_energy_ev=final_e,
            fermi_ev=fermi,
            fatal_error=fatal_error,
            dm_read_failed=dm_read_failed,
            dm_atomic_fallback=dm_atomic_fallback,
            files=files,
        ))

    return rows


def write_jsonl(rows: List[Row], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_parquet_if_possible(rows: List[Row], path: Path) -> bool:
    """
    Writes Parquet if pandas+pyarrow are available. Returns True if written.
    """
    try:
        import pandas as pd  # type: ignore
        import pyarrow  # noqa: F401
    except Exception:
        return False

    flat = []
    for r in rows:
        d = asdict(r)
        files = d.pop("files")
        for k, v in files.items():
            d[k] = v
        flat.append(d)

    df = pd.DataFrame(flat)
    df.to_parquet(path, index=False, engine="pyarrow")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Root folder (contains Al/ Fe/ Ni/ folders)")
    ap.add_argument("--out", type=str, default="index.jsonl", help="Output JSONL path")
    ap.add_argument("--parquet", type=str, default="index.parquet", help="Output Parquet path")
    ap.add_argument("--include-extra", action="store_true", help="Include heavy/niche file pointers (DM/HSX/EIG/etc.)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    rows = build_rows(root, include_extra=args.include_extra)

    out_jsonl = Path(args.out).resolve()
    write_jsonl(rows, out_jsonl)
    print(f"Wrote JSONL: {out_jsonl} ({len(rows)} rows)")

    out_parquet = Path(args.parquet).resolve()
    if write_parquet_if_possible(rows, out_parquet):
        print(f"Wrote Parquet: {out_parquet}")
    else:
        print("Parquet not written (missing pandas+pyarrow). JSONL is ready.")

    n = len(rows)
    n_conv = sum(1 for r in rows if r.scf_converged)
    n_e = sum(1 for r in rows if r.final_energy_ev is not None)
    n_f = sum(1 for r in rows if r.fermi_ev is not None)
    n_fatal = sum(1 for r in rows if r.fatal_error)
    n_dm_fail = sum(1 for r in rows if r.dm_read_failed)
    n_dm_atomic = sum(1 for r in rows if r.dm_atomic_fallback)

    print(f"SCF converged: {n_conv}/{n}")
    print(f"Final energy parsed: {n_e}/{n}")
    print(f"Fermi parsed: {n_f}/{n}")
    print(f"Fatal markers: {n_fatal}/{n}")
    print(f"DM read failed (benign): {n_dm_fail}/{n}")
    print(f"DM atomic fallback: {n_dm_atomic}/{n}")


if __name__ == "__main__":
    main()


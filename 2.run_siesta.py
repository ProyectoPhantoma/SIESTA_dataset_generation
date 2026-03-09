#!/usr/bin/env python3
"""
run_siesta.py

Runs SIESTA for single-element configs laid out as:
  raw_single_elements/<El>/cfg_XXXX/<El>.fdf

Produces:
  <El>.out and <El>.RHO inside each cfg_XXXX directory.

Key features:
- Uses the SAME convergence check as build_dataset.py via checks.check.parse_out
- Copies any pseudopotentials referenced by the .fdf (psml/psf) from --pseudo-dir
- SKIPs configs already done (RHO exists + converged)
- Supports --max-per-element to run only first N cfg_* per element
"""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List, Set

from checks.check import parse_out


PSEUDO_RE = re.compile(r"\b([A-Za-z0-9_\-]+)\.(psml|psf)\b", re.IGNORECASE)


def extract_pseudos_from_fdf(fdf_path: Path) -> Set[str]:
    """
    Returns a set of pseudo filenames referenced in the .fdf (e.g. {'Ni.psml'}).
    """
    txt = fdf_path.read_text(errors="ignore")
    found = set()
    for m in PSEUDO_RE.finditer(txt):
        found.add(f"{m.group(1)}.{m.group(2)}")
    return found


def is_converged_by_builder(out_path: Path) -> bool:
    """
    Uses checks.check.parse_out
    """
    try:
        out = parse_out(str(out_path))
        return bool(out.get("converged", False))
    except Exception:
        return False


def already_done(cfg_dir: Path, el: str) -> bool:
    out_path = cfg_dir / f"{el}.out"
    rho_path = cfg_dir / f"{el}.RHO"
    if not (out_path.exists() and rho_path.exists()):
        return False
    return is_converged_by_builder(out_path)


def ensure_pseudos(cfg_dir: Path, fdf_path: Path, pseudo_src_dir: Path, el: str) -> None:

    """
    Copies any pseudo files referenced by the .fdf into cfg_dir.
    If the .fdf references none (rare), fall back to <El>.psml/<El>.psf behavior elsewhere.
    """
    pseudos = extract_pseudos_from_fdf(fdf_path)

    # If none explicitly referenced, copy default <El>.psml if present
    if not pseudos:
        for ext in ("psml", "psf", "vps"):
            src = pseudo_src_dir / f"{el}.{ext}"
            if src.exists():
                dst = cfg_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                return
        return

    for fname in pseudos:
        src = pseudo_src_dir / fname
        dst = cfg_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing pseudopotential source: {src}")
        if not dst.exists():
            shutil.copy2(src, dst)


def run_one(
    cfg_dir: Path,
    el: str,
    pseudo_src_dir: Path,
    siesta_cmd: str,
    timeout_s: Optional[int],
) -> Tuple[str, str]:
    """
    Returns (status, message). status in {"OK","SKIP","FAIL"}.
    """
    try:
        fdf = cfg_dir / f"{el}.fdf"
        outp = cfg_dir / f"{el}.out"

        if not fdf.exists():
            return ("FAIL", f"Missing {fdf.name}")

        if already_done(cfg_dir, el):
            return ("SKIP", "Already done (RHO + converged)")

        # Copy pseudos referenced by the .fdf
        ensure_pseudos(cfg_dir, fdf, pseudo_src_dir, el)

        env = os.environ.copy()
        pdir = str(pseudo_src_dir.resolve())

        env["SIESTA_PP_PATH"] = pdir
        env["SIESTA_PS_PATH"] = pdir

        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
        env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")

        with open(fdf, "rb") as fin, open(outp, "wb") as fout:
            proc = subprocess.run(
                siesta_cmd,              # keep as string (you use shell=True)
                cwd=str(cfg_dir),
                stdin=fin,
                stdout=fout,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
                shell=True,
                env=env,
            )


        rho_path = cfg_dir / f"{el}.RHO"

        if proc.returncode != 0:
            return ("FAIL", f"Return code {proc.returncode}")

        if not rho_path.exists():
            return ("FAIL", "No .RHO produced")

        if not is_converged_by_builder(outp):
            return ("FAIL", "Not converged (parse_out)")

        return ("OK", "Converged")

    except subprocess.TimeoutExpired:
        return ("FAIL", f"Timeout after {timeout_s}s")
    except Exception as e:
        return ("FAIL", str(e))


def discover_tasks(raw_root: Path, elements: List[str], max_per_element: int) -> List[Tuple[Path, str]]:
    tasks: List[Tuple[Path, str]] = []
    for el in elements:
        el_dir = raw_root / el
        if not el_dir.exists():
            continue
        cfg_dirs = sorted([d for d in el_dir.glob("cfg_*") if d.is_dir()])
        if max_per_element > 0:
            cfg_dirs = cfg_dirs[:max_per_element]
        tasks.extend([(d, el) for d in cfg_dirs])

    import random
    random.shuffle(tasks)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, default="raw_single_elements")
    ap.add_argument("--pseudo-dir", type=str, default="build", help="Folder containing *.psml/*.psf files")
    ap.add_argument("--siesta-cmd", type=str, default="siesta", help='e.g. "siesta" or "mpirun -np 4 siesta"')
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--timeout-s", type=int, default=0, help="0 means no timeout")
    ap.add_argument("--elements", type=str, default="Ni,Fe,Al")
    ap.add_argument("--max-per-element", type=int, default=1000, help="0 means no limit")
    args = ap.parse_args()

    siesta_cmd = args.siesta_cmd
    if siesta_cmd.strip() == "siesta":
        p = shutil.which("siesta")
        if p is None:
            raise SystemExit(
                "Could not find 'siesta' in PATH. "
                "Run this script via: micromamba run -n python run_siesta.py "
                "or pass --siesta-cmd /full/path/to/siesta"
            )
        siesta_cmd = p

    raw_root = Path(args.raw_root)
    pseudo_dir = Path(args.pseudo_dir)
    timeout_s = None if args.timeout_s == 0 else int(args.timeout_s)
    elements = [e.strip() for e in args.elements.split(",") if e.strip()]

    tasks = discover_tasks(raw_root, elements, args.max_per_element)
    if not tasks:
        raise SystemExit(f"No cfg_* found under {raw_root} for elements={elements}")

    print(f"Found {len(tasks)} tasks. Running with jobs={args.jobs}.")
    print(f"raw_root={raw_root.resolve()}")
    print(f"pseudo_dir={pseudo_dir.resolve()}")
    print(f"siesta_cmd={args.siesta_cmd}")
    print(f"max_per_element={args.max_per_element}")

    ok = skip = fail = 0
    fail_list = []

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {
            ex.submit(run_one, cfg_dir, el, pseudo_dir, siesta_cmd, timeout_s): (cfg_dir, el)
            for (cfg_dir, el) in tasks
        }
        for fut in as_completed(futs):
            cfg_dir, el = futs[fut]
            status, msg = fut.result()
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                fail += 1
                fail_list.append((el, str(cfg_dir), msg))
            print(f"[{status}] {el}/{cfg_dir.name}: {msg}")

    if fail_list:
        fail_path = raw_root / "failed_jobs.csv"
        with open(fail_path, "w") as f:
            f.write("element,cfg_dir,message\n")
            for el, d, msg in fail_list:
                f.write(f"{el},{d},{msg.replace(',', ';')}\n")
        print(f"[WROTE] {fail_path} ({len(fail_list)} fails)")

    print(f"\nDONE: OK={ok}, SKIP={skip}, FAIL={fail}")
    if fail > 0:
        raise SystemExit(
            "Some jobs failed. Inspect the .out files; re-run the script. "
            "SKIP logic will avoid repeating successful runs."
        )


if __name__ == "__main__":
    import random
    random.seed(0)
    main()
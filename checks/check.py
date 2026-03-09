import os
import json
import numpy as np
from pathlib import Path

import re
import struct

from typing import Tuple, Dict, Any, List


_re_converged = re.compile(r"SCF cycle converged after\s+(\d+)\s+iterations", re.I)
_re_fermi = re.compile(r"siesta:\s+Fermi\s*=\s*([-\d.]+)")
_re_etot = re.compile(r"siesta:\s+Etot\s*=\s*([-\d.]+)")
_re_freeeng = re.compile(r"siesta:\s+FreeEng\s*=\s*([-\d.]+)")
_re_mesh = re.compile(r"InitMesh:\s+MESH\s*=\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", re.I)

def parse_out(path: str):
    txt = Path(path).read_text(errors="ignore")

    m = _re_converged.search(txt)
    converged = m is not None
    scf_steps = int(m.group(1)) if m else None

    mesh = None
    mm = _re_mesh.search(txt)
    if mm:
        mesh = tuple(int(x) for x in mm.groups())

    # Final values appear multiple times; take the last match
    def last_float(regex):
        matches = regex.findall(txt)
        return float(matches[-1]) if matches else None

    return {
        "converged": converged,
        "scf_steps": scf_steps,
        "mesh": mesh,
        "etot_eV": last_float(_re_etot),
        "freeeng_eV": last_float(_re_freeeng),
        "fermi_eV": last_float(_re_fermi),
    }


def _read_fortran_records(path: str) -> List[bytes]:
    b = Path(path).read_bytes()
    i = 0
    n = len(b)
    recs = []
    while i + 8 <= n:
        L = int.from_bytes(b[i:i+4], "little", signed=False)
        i += 4
        if i + L + 4 > n:
            break
        data = b[i:i+L]
        i += L
        L2 = int.from_bytes(b[i:i+4], "little", signed=False)
        i += 4
        if L != L2:
            break
        recs.append(data)
    return recs

def parse_rho(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Parse SIESTA .RHO (binary Fortran unformatted), handling densities stored
    either as a single big record or split across many records (slabs).

    Returns:
      cell: float32 [3,3] Ang
      rho:  float32 [nx,ny,nz] (as stored)
      meta: dict
    """
    recs = _read_fortran_records(path)
    if not recs:
        raise ValueError("No Fortran records found (unexpected .RHO format)")

    meta: Dict[str, Any] = {}

    # ---- Cell record (your file starts with 72 bytes)
    if len(recs[0]) != 72:
        raise ValueError(f"Expected first record = 72 bytes (cell 9*float64), got {len(recs[0])}")
    cell = np.frombuffer(recs[0], dtype=np.float64).reshape(3, 3)

    # ---- Grid dims record: search next few records for 3 int32
    nx = ny = nz = None
    grid_rec_idx = None
    for k in range(1, min(10, len(recs))):
        r = recs[k]
        if len(r) >= 12:
            dims = np.frombuffer(r[:12], dtype=np.int32)
            if (dims > 0).all() and (dims < 10000).all():
                nx, ny, nz = map(int, dims[:3])
                grid_rec_idx = k
                break
    if grid_rec_idx is None:
        raise ValueError("Could not find grid dimensions record (3 int32)")

    ngrid = nx * ny * nz
    meta["grid_shape"] = (nx, ny, nz)
    meta["grid_record_index"] = grid_rec_idx

    # ---- Remaining records: density is usually float32 or float64, possibly split.
    data_recs = recs[grid_rec_idx + 1:]
    if not data_recs:
        raise ValueError("No density records found after grid record")

    total_bytes = sum(len(r) for r in data_recs)

    # Decide dtype by feasibility:
    # - if total_bytes >= ngrid*8 => could be float64
    # - if total_bytes >= ngrid*4 => could be float32
    # Prefer float32 if float64 seems too large / unlikely.
    dtype = None
    if total_bytes >= ngrid * 8:
        # Ambiguous: could still be float32 with extra stuff. We'll test both quickly.
        dtype = "float64"
    if total_bytes >= ngrid * 4:
        # float32 is very common for .RHO; prefer it unless float64 is clearly needed.
        dtype = "float32"

    if dtype is None:
        raise ValueError(f"Not enough density bytes: have {total_bytes}, need {ngrid*4}")

    # Helper: collect exactly ngrid values from concatenated records
    def collect_as(dt: np.dtype, bytes_per: int) -> np.ndarray:
        needed_bytes = ngrid * bytes_per
        buf = bytearray()
        for r in data_recs:
            if len(buf) >= needed_bytes:
                break
            buf.extend(r)
        if len(buf) < needed_bytes:
            raise ValueError(f"Truncated density: got {len(buf)} bytes, need {needed_bytes}")
        arr = np.frombuffer(bytes(buf[:needed_bytes]), dtype=dt)
        return arr

    # Try float32 first (most likely), and sanity-check values
    rho_flat = None
    tried = []
    for dt, bpp, name in [(np.float32, 4, "float32"), (np.float64, 8, "float64")]:
        if total_bytes < ngrid * bpp:
            continue
        try:
            arr = collect_as(dt, bpp)
            # Basic sanity: finite and not all ~0
            if not np.isfinite(arr).all():
                raise ValueError("non-finite values")
            # If it’s basically all zeros, reject
            if float(np.max(np.abs(arr))) < 1e-20:
                raise ValueError("values too close to zero")
            rho_flat = arr
            meta["rho_dtype"] = name
            break
        except Exception as e:
            tried.append(f"{name}:{e}")

    if rho_flat is None:
        raise ValueError("Could not decode density from records; tried " + ", ".join(tried))

    rho = rho_flat.reshape((nx, ny, nz), order="F")

    meta["cell_from_rho"] = cell.astype(np.float32)
    return cell.astype(np.float32), rho, meta

# ----------------------------
# Basic FDF parsing utilities
# ----------------------------

def parse_lattice_vectors(lines):
    lv = []
    in_block = False
    for ln in lines:
        s = ln.strip().lower()
        if s.startswith("%block latticevectors"):
            in_block = True
            continue
        if s.startswith("%endblock latticevectors"):
            break
        if in_block:
            parts = ln.split()
            if len(parts) >= 3:
                lv.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if len(lv) != 3:
        raise ValueError("Invalid LatticeVectors block")
    return np.array(lv, dtype=float)


def parse_atomic_coords(lines) -> np.ndarray:
    coords = []
    in_block = False
    for ln in lines:
        s = ln.strip().lower()
        if s.startswith("%block atomiccoordinatesandatomicspecies"):
            in_block = True
            continue
        if s.startswith("%endblock atomiccoordinatesandatomicspecies"):
            break
        if in_block:
            parts = ln.split()
            if len(parts) >= 3:
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not coords:
        raise ValueError("No atomic coordinates found")
    return np.array(coords, dtype=float)


def parse_species(lines) -> Tuple[int, str]:
    in_block = False
    for ln in lines:
        s = ln.strip().lower()
        if s.startswith("%block chemicalspecieslabel"):
            in_block = True
            continue
        if s.startswith("%endblock chemicalspecieslabel"):
            break
        if in_block:
            parts = ln.split()
            if len(parts) >= 3:
                Z = int(parts[1])
                el = parts[2]
                return Z, el
    raise ValueError("ChemicalSpeciesLabel block not found")

# ----------------------------
# Geometry sanity checks
# ----------------------------

def cell_volume(cell: np.ndarray) -> float:
    return abs(np.linalg.det(cell))


def min_interatomic_distance(R: np.ndarray) -> float:
    dmin = np.inf
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            d = np.linalg.norm(R[i] - R[j])
            dmin = min(dmin, d)
    return dmin


# ----------------------------
# Main checker
# ----------------------------

def check_cfg(cfg_dir: Path) -> None:
    el = cfg_dir.parent.name
    fdf = cfg_dir / f"{el}.fdf"
    meta = cfg_dir / "meta.json"

    if not fdf.exists():
        raise FileNotFoundError(f"Missing {fdf.name}")
    if not meta.exists():
        raise FileNotFoundError("Missing meta.json")

    lines = fdf.read_text(errors="ignore").splitlines()

    # Check no placeholders remain
    for key in ["__EL__", "__Z__", "__LV", "__COORDS__", "__NATOMS__"]:
        if any(key in ln for ln in lines):
            raise ValueError(f"Unreplaced placeholder {key}")

    cell = parse_lattice_vectors(lines)
    R = parse_atomic_coords(lines)
    Z, el_from_fdf = parse_species(lines)

    # Meta consistency
    meta_j = json.loads(meta.read_text())
    if meta_j["element"] != el:
        raise ValueError("meta.json element mismatch")
    if meta_j["Z"] != Z:
        raise ValueError("Atomic number mismatch")

    # Geometry checks
    vol = cell_volume(cell)
    if vol < 5.0:
        raise ValueError(f"Unphysically small cell volume: {vol:.2f} Å^3")

    dmin = min_interatomic_distance(R)
    if dmin < 1.5:
        raise ValueError(f"Atoms too close: min distance = {dmin:.2f} Å")

    # Perturbation sanity
    if "sigma_A" in meta_j and meta_j["sigma_A"] > 0.15:
        raise ValueError("Displacement sigma too large")

    if "iso_strain" in meta_j and abs(meta_j["iso_strain"]) > 0.10:
        raise ValueError("Isotropic strain too large")

# ----------------------------
# Batch runner
# ----------------------------

def main():
    root = Path("raw_v2_single_elements")
    ok = 0
    failed = 0

    for el_dir in sorted(root.iterdir()):
        if not el_dir.is_dir():
            continue
        for cfg in sorted(el_dir.glob("cfg_*")):
            try:
                check_cfg(cfg)
                ok += 1
            except Exception as e:
                failed += 1
                print(f"[FAIL] {el_dir.name}/{cfg.name}: {e}")

    print(f"\nCHECK DONE: OK={ok}, FAILED={failed}")

    if failed > 0:
        raise SystemExit("Some configurations failed checks")

if __name__ == "__main__":
    main()
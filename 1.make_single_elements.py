#!/usr/bin/env python3
"""
make_single_elements.py
==========================

Generator for the **single-element SIESTA configuration dataset**.

This script programmatically generates atomic structures and corresponding
SIESTA input files (.fdf) for large batches of self-consistent field (SCF)
electronic structure calculations. The resulting dataset is designed for
machine-learning and electronic structure research applications.

The workflow constructs perturbed crystal configurations for a set of
single-element systems (Al, Fe, Ni), writes SIESTA input files, and
organizes them into a standardized directory structure suitable for
high-throughput execution.

------------------------------------------------------------
Dataset generation pipeline
------------------------------------------------------------

For each selected element:

1. A base two-atom primitive cell is defined.
   - lattice vectors (`cell`)
   - atomic coordinates (`R`)
   - atomic number (`Z`)
   - SIESTA input template

2. A set of configuration types is scheduled according to user-specified
   mixture counts. These types define the structural perturbations applied
   to the base crystal.

3. For each configuration:
   - lattice vectors and/or atomic coordinates are perturbed
   - a SIESTA `.fdf` input file is generated from a template
   - metadata describing the perturbation is written to `meta.json`

4. Each configuration is stored in a dedicated directory.

------------------------------------------------------------
Output directory layout
------------------------------------------------------------

The generated dataset follows the structure:

    <out_root>/
        Al/
            cfg_0000/
                Al.fdf
                meta.json
            cfg_0001/
                Al.fdf
                meta.json
            ...
        Fe/
            cfg_0000/
                Fe.fdf
                meta.json
            ...
        Ni/
            cfg_0000/
                Ni.fdf
                meta.json

Each configuration directory contains:

    <element>.fdf
        SIESTA input file

    meta.json
        Metadata describing the perturbation parameters used
        to generate the configuration.

These directories are later processed by the execution pipeline
(e.g. `run_siesta.py`) which runs SIESTA and produces the
electronic structure outputs.

------------------------------------------------------------
Perturbation model
------------------------------------------------------------

Five types of structural perturbations are used:

1. disp  (atomic displacement)
   Gaussian noise added to atomic coordinates.

2. iso   (isotropic strain)
   Uniform scaling of the lattice vectors.

3. aniso (anisotropic strain)
   Independent scaling of lattice axes.

4. shear
   Shear deformation applied to lattice vectors.

5. combo
   Combination of isotropic strain and atomic displacement,
   optionally including shear.

Default perturbation ranges:

    atomic displacement σ ∈ {0.02, 0.05, 0.08} Å
    isotropic strain     ε ∈ [-0.02, +0.02]
    anisotropic strain   εx,εy,εz ∈ [-0.02, +0.02]
    shear                s ∈ [-0.01, +0.01]

------------------------------------------------------------
Command-line arguments
------------------------------------------------------------

--out-root
    Output directory where configuration folders are created.

--seed
    Global random seed controlling perturbation generation.

--n-per-element
    Number of configurations generated for each element.

--elements
    Comma-separated list of elements to generate.
    Default: Ni,Fe,Al

--n-disp
    Number of displacement-only configurations.

--n-iso
    Number of isotropic strain configurations.

--n-aniso
    Number of anisotropic strain configurations.

--n-shear
    Number of shear configurations.

--n-combo
    Number of combined perturbation configurations.

The mixture counts must sum to `n-per-element`.

------------------------------------------------------------
Example usage
------------------------------------------------------------

Default dataset generation:

    python make_single_elements.py

Custom dataset:

    python make_single_elements.py \
        --out-root test_dataset \
        --seed 42 \
        --n-per-element 1000 \
        --n-disp 300 \
        --n-iso 300 \
        --n-aniso 250 \
        --n-shear 100 \
        --n-combo 50

------------------------------------------------------------
Reproducibility
------------------------------------------------------------

The dataset is reproducible given:

    - the generator script version
    - the random seed
    - the perturbation parameters
    - the base crystal definitions

Each configuration records a per-run seed in `meta.json`
to enable traceability of individual perturbations.

------------------------------------------------------------
Intended use
------------------------------------------------------------

This generator is part of the pipeline used to construct the
single-element electronic density dataset used for
machine-learning models predicting electron density fields
from atomic configurations.

The generated inputs are executed with SIESTA and the resulting
density grids (`Rho.grid.nc`) and potentials
(`ElectrostaticPotential.grid.nc`) are later indexed and
processed into ML-ready datasets.
"""

import os
import json
import numpy as np
import argparse


BASE = {
    "Ni": {
        "Z": 28,
        "cell": np.array([
            [ 2.47959800,  0.00000000,  0.00000000],
            [-1.23979900,  2.14739500,  0.00000000],
            [ 0.00000000,  0.00000000,  4.06426300],
        ], dtype=float),
        "R": np.array([
            [-0.000001,  1.431597,  1.016066],
            [ 1.239800,  0.715798,  3.048197],
        ], dtype=float),
        "structure": "fcc_2atom",
        "template": "template_scf_EL.fdf",
    },
    "Fe": {
        "Z": 26,
        "cell": np.array([
            [ 2.47959800,  0.00000000,  0.00000000],
            [-1.23979900,  2.14739500,  0.00000000],
            [ 0.00000000,  0.00000000,  4.06426300],
        ], dtype=float),
        "R": np.array([
            [-0.000001,  1.431597,  1.016066],
            [ 1.239800,  0.715798,  3.048197],
        ], dtype=float),
        "structure": "fcc_2atom",
        "template": "template_scf_EL.fdf",
    },
    "Al": {
        "Z": 13,
        "cell": np.array([
            [ 2.47959800,  0.00000000,  0.00000000],
            [-1.23979900,  2.14739500,  0.00000000],
            [ 0.00000000,  0.00000000,  4.06426300],
        ], dtype=float),
        "R": np.array([
            [-0.000001,  1.431597,  1.016066],
            [ 1.239800,  0.715798,  3.048197],
        ], dtype=float),
        "structure": "fcc_2atom",
        "template": "template_scf_EL.fdf",
    },
}

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_fdf_from_template(template_path: str, out_path: str, cell: np.ndarray, R: np.ndarray, Z: int, element: str):
    tpl = open(template_path, "r", encoding="utf-8").read()

    lv = ["%12.8f  %12.8f  %12.8f" % tuple(cell[i]) for i in range(3)]
    coords = []
    for (x, y, z) in R:
        coords.append("%12.6f  %12.6f  %12.6f  1" % (x, y, z))  # species index = 1
    coords_txt = "\n".join(coords)

    txt = (tpl
        .replace("__SYSTEM_LABEL__", element)
        .replace("__NATOMS__", str(len(R)))
        .replace("__Z__", str(Z))
        .replace("__EL__", element)
        .replace("__LV1__", lv[0])
        .replace("__LV2__", lv[1])
        .replace("__LV3__", lv[2])
        .replace("__COORDS__", coords_txt)
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)

def make_iso_strain(cell: np.ndarray, eps: float) -> np.ndarray:
    return cell * (1.0 + eps)

def make_aniso_strain(cell: np.ndarray, ex: float, ey: float, ez: float) -> np.ndarray:
    # Apply diagonal strain matrix in lattice-vector space
    S = np.diag([1.0 + ex, 1.0 + ey, 1.0 + ez])
    return cell @ S

def make_shear(cell: np.ndarray, sh_xy: float, sh_xz: float, sh_yz: float) -> np.ndarray:
    """
    Small shear in lattice-vector space (upper-triangular).
    Values are dimensionless (e.g. 0.01 = 1% shear).
    """
    M = np.array([
        [1.0,   sh_xy, sh_xz],
        [0.0,   1.0,   sh_yz],
        [0.0,   0.0,   1.0],
    ], dtype=float)
    return cell @ M

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate single-element SIESTA inputs.")

    ap.add_argument(
        "--out-root",
        type=str,
        default="raw_single_elements",
        help="Output root directory.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed.",
    )
    ap.add_argument(
        "--n-per-element",
        type=int,
        default=10000,
        help="Number of configurations per element.",
    )
    ap.add_argument(
        "--elements",
        type=str,
        default="Ni,Fe,Al",
        help="Comma-separated list of elements to generate, e.g. Ni,Fe,Al",
    )

    # Mixture counts
    ap.add_argument("--n-disp", type=int, default=3000, help="Number of displacement-only configs.")
    ap.add_argument("--n-iso", type=int, default=3000, help="Number of isotropic-strain configs.")
    ap.add_argument("--n-aniso", type=int, default=2500, help="Number of anisotropic-strain configs.")
    ap.add_argument("--n-shear", type=int, default=1000, help="Number of shear configs.")
    ap.add_argument("--n-combo", type=int, default=500, help="Number of combo configs.")

    return ap

def main():
    args = build_arg_parser().parse_args()

    out_root = args.out_root
    seed = args.seed
    n_per_element = args.n_per_element
    elements = [e.strip() for e in args.elements.split(",") if e.strip()]

    ensure_dir(out_root)

    np.random.seed(seed)

    n_disp = args.n_disp
    n_iso = args.n_iso
    n_aniso = args.n_aniso
    n_shear = args.n_shear
    n_combo = args.n_combo

    total = n_disp + n_iso + n_aniso + n_shear + n_combo
    if total != n_per_element:
        raise ValueError(
            f"Mixture counts must sum to --n-per-element.\n"
            f"Got: disp={n_disp}, iso={n_iso}, aniso={n_aniso}, shear={n_shear}, combo={n_combo}\n"
            f"Sum={total}, n_per_element={n_per_element}"
        )

    disp_sigmas = [0.02, 0.05, 0.08]   # Ang
    iso_min, iso_max = -0.02, 0.02
    aniso_min, aniso_max = -0.02, 0.02
    shear_min, shear_max = -0.01, 0.01


    for el in elements:
        if el not in BASE:
            raise ValueError(f"Unknown element '{el}'. Allowed: {list(BASE.keys())}")

        spec = BASE[el]
        Z = spec["Z"]
        base_cell = spec["cell"]
        base_R = spec["R"]
        template = spec["template"]
        structure = spec["structure"]

        el_root = os.path.join(out_root, el)
        ensure_dir(el_root)

        types = (
            ["disp"]  * n_disp +
            ["iso"]   * n_iso +
            ["aniso"] * n_aniso +
            ["shear"] * n_shear +
            ["combo"] * n_combo
        )
        np.random.shuffle(types)

        for i, t in enumerate(types):
            run_dir = os.path.join(el_root, f"cfg_{i:04d}")
            ensure_dir(run_dir)

            cell = base_cell.copy()
            R = base_R.copy()

            meta = {
                "dataset": "single_elements",
                "element": el,
                "Z": int(Z),
                "structure": structure,
                "cfg_type": t,
                "seed": int(np.random.randint(0, 2**31 - 1)),
                "generator_version": "1.0",
                "notes": "ranges: sigma in {0.02,0.05,0.08}A; iso/aniso up to 2%; includes shear.",
            }

            if t == "disp":
                sigma = float(np.random.choice(disp_sigmas))
                disp = np.random.normal(scale=sigma, size=R.shape)
                R = R + disp
                meta["sigma_A"] = sigma

            elif t == "iso":
                eps = float(np.random.uniform(iso_min, iso_max))
                cell = make_iso_strain(cell, eps)
                meta["iso_strain"] = eps

            elif t == "aniso":
                ex, ey, ez = [float(np.random.uniform(aniso_min, aniso_max)) for _ in range(3)]
                cell = make_aniso_strain(cell, ex, ey, ez)
                meta["aniso_strain"] = [ex, ey, ez]

            elif t == "shear":
                sh_xy, sh_xz, sh_yz = [float(np.random.uniform(shear_min, shear_max)) for _ in range(3)]
                cell = make_shear(cell, sh_xy, sh_xz, sh_yz)
                meta["shear"] = [sh_xy, sh_xz, sh_yz]

            elif t == "combo":
                eps = float(np.random.uniform(iso_min, iso_max))
                cell = make_iso_strain(cell, eps)

                sigma = float(np.random.choice(disp_sigmas))
                disp = np.random.normal(scale=sigma, size=R.shape)
                R = R + disp

                meta["iso_strain"] = eps
                meta["sigma_A"] = sigma

                if np.random.rand() < 0.5:
                    sh_xy, sh_xz, sh_yz = [float(np.random.uniform(shear_min, shear_max)) for _ in range(3)]
                    cell = make_shear(cell, sh_xy, sh_xz, sh_yz)
                    meta["shear"] = [sh_xy, sh_xz, sh_yz]

            else:
                raise ValueError(f"Unknown cfg type: {t}")

            fdf_path = os.path.join(run_dir, f"{el}.fdf")
            write_fdf_from_template(template, fdf_path, cell, R, Z, el)

            with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        print(f"[OK] {el}: wrote {n_per_element} configs to {el_root}/cfg_*/{el}.fdf")


if __name__ == "__main__":
    main()


###### Example usage: ######
# python make_single_elements.py \
#  --out-root raw_single_elements \
#  --n-per-element 1000 \
#  --n-disp 300 \
#  --n-iso 300 \
#  --n-aniso 250 \
#  --n-shear 100 \
#  --n-combo 50
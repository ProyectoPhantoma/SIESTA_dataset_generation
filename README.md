
# SIESTA Single-Elements Dataset — Generation Pipeline

This repository contains the **pipeline used to generate and validate the SIESTA Single-Elements dataset**, a collection of **30,000 self-consistent DFT calculations** for Al, Fe, and Ni.

The repository provides scripts to:

- generate perturbed atomic configurations
- run SIESTA calculations
- build dataset indices
- construct the public dataset release
- validate dataset integrity
- demonstrate the pipeline in a Jupyter notebook

The dataset itself is released separately on Zenodo.

---

# Repository Structure

```
.
├── make_single_elements.py
├── run_siesta.py
├── build_index_siesta.py
├── slim_public_core.py
│
├── checks/
│   ├── checks.py
│   ├── count_total.py
│   ├── element_distribution.py
│   ├── points.py
│   ├── readability.py
│   ├── size_consistency.py
│   ├── scf_convergence.py
│   └── final_check.py
│
├── siesta_v3_pipeline_and_checks.ipynb
│
└── README.md
```

---

# Dataset Overview

The dataset contains **self-consistent density functional theory (DFT) calculations** for three elemental systems:

| Element | Configurations |
|-------|------|
| Al | 10,000 |
| Fe | 10,000 |
| Ni | 10,000 |

Total:

```
30,000 DFT calculations
```

Each configuration contains:

```
.fdf                          SIESTA input file
.out                          SIESTA output log
.XV                           final atomic positions
.STRUCT_OUT                   structure summary
Rho.grid.nc                   electron density grid
ElectrostaticPotential.grid.nc electrostatic potential grid
meta.json                     perturbation metadata
```

---

# Pipeline Overview

The dataset is generated using a **four-stage workflow**.

```
Structure generation
        ↓
SIESTA calculations
        ↓
Index construction
        ↓
Public dataset slimming
```

---

# 1. Structure Generation

```
make_single_elements.py
```

This script generates **perturbed crystal structures and corresponding SIESTA input files**.

Perturbations applied:

| Type | Description |
|----|----|
| disp | random atomic displacements |
| iso | isotropic lattice strain |
| aniso | anisotropic strain |
| shear | lattice shear deformation |
| combo | combination of strain and displacement |

Example:

```bash
python make_single_elements.py   --out-root raw_single_elements   --seed 0   --n-per-element 10000   --n-disp 3000   --n-iso 3000   --n-aniso 2500   --n-shear 1000   --n-combo 500
```

Output directory structure:

```
raw_single_elements/
    Al/
        cfg_0000/
        cfg_0001/
        ...
    Fe/
    Ni/
```

Each configuration directory contains:

```
Element.fdf
meta.json
```

---

# 2. Running SIESTA Calculations

```
run_siesta.py
```

This script executes **self-consistent SIESTA calculations** for each generated configuration.

Example:

```bash
python run_siesta.py   --raw-root raw_single_elements   --pseudo-dir pseudos   --jobs 16
```

Outputs generated per configuration:

```
.out
.XV
.STRUCT_OUT
Rho.grid.nc
ElectrostaticPotential.grid.nc
```

---

# 3. Dataset Index Construction

```
build_index_siesta.py
```

This script parses SIESTA outputs and builds a **structured dataset index**.

Example:

```bash
python build_index_siesta.py raw_single_elements   --include-extra   --out index_full.jsonl   --parquet index_full.parquet
```

The index stores:

```
calc_id
element
calc_dir
scf_converged
scf_iterations
final_energy_ev
fermi_ev
dm_read_failed
dm_atomic_fallback
```

---

# 4. Public Dataset Construction

```
slim_public_core.py
```

This script creates the **public dataset release** by copying only the essential files.

Example:

```bash
python slim_public_core.py   --root raw_single_elements   --index index_full.parquet   --out public_core   --mode copy
```

The resulting dataset structure:

```
public_core/
    index_core.jsonl
    index_core.parquet
    data/
        Al/cfg_0000/
        Fe/cfg_0000/
        Ni/cfg_0000/
```

---

# Dataset Validation

The repository includes several scripts used to verify dataset integrity.

Main checks:

```
checks.py
```

Verifies:

- missing directories
- missing density files
- index consistency

Additional validation scripts:

| Script | Purpose |
|------|------|
| count_total.py | verify total calculations |
| element_distribution.py | verify per-element counts |
| points.py | verify index paths |
| readability.py | verify NetCDF readability |
| size_consistency.py | detect corrupted files |
| scf_convergence.py | verify SCF convergence |
| final_check.py | quick sanity check |

Example:

```bash
python checks.py
```

Expected output:

```
Missing directories: 0
Missing Rho.grid.nc: 0
Missing ElectrostaticPotential.grid.nc: 0
```

---

# Notebook Example

The notebook

```
notebooks/siesta_pipeline_and_checks.ipynb
```

demonstrates:

- the dataset generation pipeline
- loading the dataset index
- validating the dataset
- reading density grids

---

# Software Requirements

Required Python packages:

```
numpy
pandas
pyarrow
netCDF4
```

Optional:

```
zstandard
tqdm
```

---

# Reproducibility

The dataset generation pipeline is **fully deterministic given the random seed** used in `make_single_elements.py`.

The following parameters control reproducibility:

```
seed
number of configurations per element
perturbation mixture counts
perturbation ranges
```

---

# Dataset Availability

The dataset itself is available at:

```
Zenodo DOI: 10.5281/zenodo.18925343
```

Release archive:

```
siesta-single-elements-30k-public-core.tar.zst
```

Size:

```
~37 GB uncompressed
```

---

# License

Code in this repository is released under:

```
MIT License
```

The dataset is released for **academic and research use only**.

---

# Citation

If you use this dataset or code, please cite:

```
Irina Arevalo, Pablo Olleros

"A dataset of SIESTA density functional theory calculations for elemental metals under structural perturbations"

(Under preparation)
```

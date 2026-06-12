# porecdft tutorials — external validation against Stierle & Gross 2024

This directory contains worked examples that reproduce the benchmark systems
from the supplementary material of

> Stierle, R.; Gross, J.  *Automatically differentiable classical density
> functional theory of inhomogeneous fluids with PC-SAFT*.
> **Chemical Engineering Science** **2024**, *299*, 120437.
> DOI: 10.1016/j.ces.2024.120437

The reference paper publishes a JAX cDFT engine with PC-SAFT (their
`dft_ad_jax` package).  The supplementary data set provides:

* `data/structures/` — host atomic coordinates for ZIF-8, MFI (silicalite-1),
  LTA zeolite, and the COF systems Dha and TpPA, plus a metadata index
  (`solid_database.json`).
* `data/forcefields/` — DREIDING and PASCUAL Lennard-Jones parameters used
  by Stierle & Gross to build *V*~ext~.
* `data/fluids/` — fluid PC-SAFT parameters (`pcsaft_gross2001.json` from
  Gross & Sadowski 2001, and `noble_gases.json` for Ar/Kr/Xe).

Each tutorial reproduces a published result using the exact same input data,
loaded through porecdft's standard interface.  This gives an **independent
cross-validation** against an external reference implementation.

## Tutorials

| # | Folder | Benchmark | Reference figure |
|---|--------|-----------|------------------|
| 1 | `01_argon_in_zif8/`    | Ar adsorption isotherm in ZIF-8 at 77 K  | Stierle 2024 Fig. 5 |
| 2 | `02_methane_in_mfi/`   | CH₄ adsorption in silicalite-1 at 300 K  | Stierle 2024 Fig. 7 |
| 3 | `03_co2_in_dha_cof/`   | CO₂ adsorption in Dha COF at 298 K       | Stierle 2024 Fig. 9 |
| 4 | `04_xe_kr_in_zif8/`    | Xe/Kr selectivity in ZIF-8 at 273 K      | Stierle 2024 Fig. 6 |
| 5 | `05_co2_n2_in_zif8/`   | **Binary CO₂/N₂ mixture in ZIF-8 at 298 K** (IAST selectivity) | Stierle 2024 Fig. 8 |

## Common helper

`tutorials/data_loader.py` provides three helpers:

* `load_dat_structure(path)` — reads the custom `.dat` / `.nml` atomic-position
  format used by Stierle's SI and returns a porecdft `HostAtoms`.
* `load_dreiding_ff(path)` — reads `DREIDING.dat` / `PASCUAL.dat` and
  returns the `{element: FFEntry}` dict porecdft expects.
* `load_pcsaft_fluid(name, path)` — looks up a fluid by name in any of the
  JSON parameter files (`noble_gases.json`, `pcsaft_gross2001.json`) and
  returns its PC-SAFT triplet `(m, σ, ε/k_B)`.

## Quick run

```bash
cd /path/to/porecdft
/opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \
    tutorials/01_argon_in_zif8/run.py
```

Each tutorial folder has a `run.py` that performs:

1. Load host + force field via the helpers above.
2. Build the orientation-averaged composite *V*~ext~ on a 3D grid (cached).
3. Solve the FMT-aWBII + PC-SAFT self-consistency for each (P, T).
4. Save a figure under `tutorials/figures/<tutorial>.png` and compare to
   the published Stierle & Gross 2024 numbers stored alongside.

## Citation

If you use any of these benchmarks in your research, please cite the original
Stierle & Gross 2024 paper for the structures, force fields, and PC-SAFT
parameters; cite porecdft for the cDFT engine.

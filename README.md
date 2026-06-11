# porecdft

**porecdft** is an open-source Python package for three-dimensional classical density functional theory (cDFT) of gas adsorption in nanoporous materials — metal–organic frameworks (MOFs), covalent organic frameworks (COFs), and zeolites.

The package is host-agnostic and fluid-agnostic. A pluggable `Potential` interface accepts any analytic external-field source (Lennard–Jones, Morse, Coulomb, quadrupole–EFG) and is designed to accommodate machine-learning interatomic potentials (MACE, NequIP, Allegro) without modifying the solver or functional layers.

The CO₂/aluminum-formate (ALF) system is the primary validation benchmark; H₂ adsorption in metalated COFs is the second. Both are reproduced in `applications/`.

---

## Installation

### With uv (recommended — fast dependency resolution)

```bash
git clone https://github.com/Feugmo-Group/porecdft.git
cd porecdft

# CPU machine (core: NumPy, SciPy, pymatgen, matplotlib)
uv sync

# With JAX for GPU-accelerated Vext computation
uv sync --extra jax
```

### With pip

```bash
pip install -e .           # minimal (CPU)
pip install -e ".[dev]"    # + pytest, ruff
pip install -e ".[jax]"    # + JAX (CPU or GPU JAX)
```

### Conda environment (used in development)

```bash
conda activate jax
pip install -e ".[dev]"
pytest -m "not slow"
```

### Requirements

| Package         | Minimum | Tested  |
|-----------------|---------|---------|
| Python          | 3.10    | 3.12    |
| NumPy           | 2.0     | 2.3     |
| SciPy           | 1.10    | 1.15    |
| pymatgen        | 2024.1  | 2025.6  |
| matplotlib      | 3.8     | 3.10    |
| JAX *(opt: `jax`)* | 0.4 | 0.6    |

> **Note:** CIF files are read with `pymatgen.io.cif`. The `ase` package is **not** required.

---

## Reproducing paper figures

All scripts must be run **from the repository root** with the editable install active (`uv pip install -e .`). Generated figures are written to `applications/alf_co2/figures/` and `applications/h2_cof/figures/`.

Scripts cache intermediate results under `applications/*/results/` (not tracked by git). On a first run each script computes its own cache and subsequent reruns are fast.

### CO₂ / ALF figures

#### Phase 1 — Force-field validation (not paper figures)

These scripts tune and validate the external potential against DFT binding energies.
They do not produce paper figures but their outputs are referenced in the Methods section.

```bash
uv run python applications/alf_co2/notebooks/phase0_evans_check.py        # digitised data check
uv run python applications/alf_co2/notebooks/phase1_vext_validation.py    # SC/LC site probe (figs 01–04)
uv run python applications/alf_co2/notebooks/phase1d_lj_tuning.py         # LJ ε scaling
uv run python applications/alf_co2/notebooks/phase1e_smeared_coulomb_tuning.py  # Coulomb σ tuning
```

#### Paper figures — minimal pipeline

Run the scripts in the order shown. Each step depends on CSV files produced by earlier ones.
Vext caches under `results/vext_cache_flex/` are built on first run (~2–4 h) and reused thereafter.

```bash
# Step 1 — Production isotherm: K_eff × ε_assoc × T sweep + FMT-aWBII baseline
# Runtime: ~2–4 h first run (builds vext_cache_flex/).
# Writes: results/phase3_production_isotherms.csv
#         results/phase2_2_fmt_isotherms.csv  (FMT baseline, used by summary figures)
uv run python applications/alf_co2/notebooks/phase3_production_isotherm.py
# Output: figures/24_phase3_param_sweep.png
#         figures/25_phase3_best_model.png
#         figures/26_phase3_parity.png

# Step 2 — Isosteric heat Q_st (Clausius–Clapeyron from step 1 CSV)
# Runtime: ~2 min. Writes: results/phase3_qst.csv
uv run python applications/alf_co2/notebooks/phase3_qst.py
# Output: figures/27_phase3_qst.png

# Step 3 — Paper summary figures (read steps 1–2 CSVs; no new cDFT runs)
uv run python applications/alf_co2/notebooks/phase3_final_summary.py
# Output: figures/31_phase3_final_summary.png

uv run python applications/alf_co2/henry_crosscheck.py
# Output: figures/32_henry_crosscheck.png

uv run python applications/alf_co2/notebooks/n2_isotherm_selectivity.py
# Output: figures/33_n2_isotherm_298K.png
#         figures/34_co2_n2_selectivity.png

uv run python applications/alf_co2/notebooks/phase_summary_figure.py
# Output: figures/35_co2_vs_experiment_final.png
```

### H₂ / COF figures

```bash
# ── Morse potential validation ────────────────────────────────────────────────
# Runtime: <5 s. No prior results needed.
uv run python applications/alf_co2/notebooks/phase_morse_validation.py
# Output: applications/h2_cof/figures/morse_validation.png

# ── Full COF benchmark: 4 frameworks × 5 metals at 77 K and 298 K ────────────
# Builds Vext caches per framework/metal on first run (~1–2 h total).
uv run python applications/h2_cof/notebooks/make_h2_cof_benchmark.py
# Output: figures/h2_cof_benchmark.png

# ── COF-333-CoCl2 H₂ isotherm (Henry regime + full-pressure cached data) ─────
# Runtime: ~1 min. Requires COF-333-CoCl2 CIF.
uv run python applications/h2_cof/notebooks/make_h2_isotherm.py
# Output: figures/h2_isotherm_cof333.png
```

---

## Package layout

```
porecdft/
  io/           CIF, force-field, and partial-charge readers
  structure/    HostAtoms, supercell builder, pore-volume probes, site finders
  forcefield/   Potential ABC + LJ, Morse, Coulomb, quadrupole-EFG,
                composite, and MLIP adapter implementations
  fluid/        Fluid ABC + CO₂ (EPM2/TraPPE), N₂, CH₄, H₂, generic single-site
  vext/         Fibonacci-sphere orientation sampler + 3D Vext grid builder
                with on-disk caching
  eos/          Bulk equations of state (ideal gas; LJ-MBWR; PC-SAFT)
  functional/   Free-energy functionals: aWBII FMT, Wertheim TPT-1 association,
                elastic framework penalty
  solver/       Picard iteration, Anderson mixing, FIRE minimiser
  diagnostics/  Binding-site probe, Henry constant, isosteric heat
  plotting/     Standardised diagnostic figures
```

---

## Quick start

### CO₂ in a MOF

```python
from porecdft.io import read_cif, read_charges_csv
from porecdft.fluid import EPM2_CO2
from porecdft.forcefield import CompositePotential, LJPotential, CoulombPotential, QuadrupoleEFGPotential
from porecdft.vext import fibonacci_rotations, build_vext_on_grid
import csv

# 1. Load host and assign partial charges
host = read_cif("applications/alf_co2/structures/alf.cif")
charges = read_charges_csv("applications/alf_co2/parameters/charges.csv")
host = host.assign_charges(charges, source="CP2K_Hirshfeld")

# 2. Define force field
def read_ff(path):
    from porecdft.io.forcefield import FFEntry
    ff = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ff[row["element"]] = FFEntry(row["element"], float(row["sigma_A"]),
                                          float(row["epsilon_K"]), row["source"])
    return ff

host_ff = read_ff("applications/alf_co2/parameters/forcefield.csv")
fluid   = EPM2_CO2

potential = CompositePotential([
    LJPotential(host_ff=host_ff, fluid_ff=fluid.ff, cutoff=15.0),
    CoulombPotential(fluid_charges=fluid.charges, cutoff=15.0),
    QuadrupoleEFGPotential(theta_zz=fluid.theta_zz, cutoff=15.0),
])

# 3. Build orientation-averaged Vext (20 orientations, cached)
from porecdft.structure import build_supercell
from dataclasses import replace
host_super = build_supercell(host, 3, 3, 3)
host_super = replace(host_super, positions=host_super.positions
                     - host.lattice[0] - host.lattice[1] - host.lattice[2])

vext = build_vext_on_grid(
    host_super, fluid, potential,
    orientations=fibonacci_rotations(20),
    spacing=0.7,
    temperature_K=298.0,
    cache_path="vext_co2_298K.npy",
)

# 4. Langmuir isotherm (simple, no FMT)
from porecdft.diagnostics.isotherm import compute_isotherm_langmuir
import numpy as np
pressures = np.logspace(-3, 0, 20)   # bar
fw_mass = sum({"Al":26.98,"C":12.01,"O":16.00,"H":1.008}[s] for s in host.species)
iso = compute_isotherm_langmuir(
    vext_avg_grid_K=vext["vext_avg"],
    dV_A3=vext["dV"],
    pressures_bar=pressures,
    temperature_K=298.0,
    framework_mass_amu=fw_mass,
)
print(f"CO₂ @ 1 bar, 298 K: {np.interp(1.0, iso.pressures_bar, iso.loading_mmol_per_g_abs):.2f} mmol/g")
```

---

## The `Potential` interface

Any external-field source can be used by subclassing `porecdft.forcefield.base.Potential`:

```python
class MyPotential(Potential):
    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels):
        return PotentialEnergy(total=..., parts={...})

    def energy_grid(self, grid_xyz, rot, host, fluid_sites, fluid_site_labels):
        # vectorised version — override for speed
        ...
```

Energy units are **Kelvin** (ε/k_B convention) throughout; conversion to kJ/mol only at reporting boundaries.

| Class | Description |
|-------|-------------|
| `LJPotential` | 12-6 Lennard–Jones, Lorentz–Berthelot mixing, 15 Å cutoff |
| `CoulombPotential` | Direct, Wolf-damped, or Gaussian-smeared Coulomb |
| `QuadrupoleEFGPotential` | CO₂ quadrupole – framework electric-field-gradient coupling |
| `MorsePotential` | Morse well for transition-metal binding sites in COFs |
| `CompositePotential` | Sum of any set of Potential instances |

---

## Benchmarks

### CO₂ in aluminum formate (ALF)

ALF (Al(HCOO)₃, cubic Im-3m, Evans et al. *Sci. Adv.* 2022) simultaneously exhibits cooperative pore filling, framework gate-opening, and kinetic molecular sieving.

| Quantity | Experiment (Evans 2022) | porecdft |
|----------|------------------------|---------|
| SC binding energy | −18.4 kJ/mol | −18.2 kJ/mol (< 1%) |
| LC binding energy | −8.1 kJ/mol | −8.0 kJ/mol (< 1%) |
| 298 K isotherm RMSE | — | 0.33 mmol/g |
| Isosteric heat | 25–32 kJ/mol | 25–32 kJ/mol |
| IAST CO₂/N₂ selectivity (thermodynamic) | ~4 | ~4 |

The experimental separation factor of 350–600 is a transport (kinetic) property; the cDFT thermodynamic IAST value of ~4 confirms ALF is a kinetic molecular sieve.

### H₂ in metalated COFs

Morse external potentials are applied to H₂ adsorption in COF-301, COF-322, COF-330, and COF-333 with five first-row transition metals. Cobalt gives the highest Henry-regime uptake in every framework owing to its broad, soft Morse well.

---

## Citation

If you use porecdft in your research, please cite:

> Roy, A.; Tetsassi Feugmo, C. G. *A modular classical density-functional framework for gas adsorption in nanoporous materials: from first-principles binding energies to kinetic molecular sieving.* 2026, in preparation.

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Conrard Giresse Tetsassi Feugmo  
Department of Chemistry, University of Waterloo  
cgtetsas@uwaterloo.ca

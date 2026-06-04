# porecdft

**porecdft** is an open-source Python package for three-dimensional classical density functional theory (cDFT) of gas adsorption in nanoporous materials — metal–organic frameworks (MOFs), covalent organic frameworks (COFs), and zeolites.

The package is host-agnostic and fluid-agnostic. A pluggable `Potential` interface accepts any analytic external-field source (Lennard–Jones, Morse, Coulomb, quadrupole–EFG) and is designed to accommodate machine-learning interatomic potentials (MACE, NequIP, Allegro) without modifying the solver or functional layers.

The CO₂/aluminum-formate (ALF) system is the primary validation benchmark; H₂ adsorption in metalated COFs is the second. Both are reproduced in `applications/`.

---

## Features

- **Three-dimensional atomistic geometry** — uses the full crystallographic unit cell read from a CIF file; no slit or cylinder approximation.
- **Modular free-energy functional** — advanced White-Bear II (aWBII) fundamental measure theory with FFT-evaluated convolutions, optionally supplemented by a Wertheim first-order thermodynamic perturbation theory (TPT-1) site-association term and a soft elastic penalty for gate-opening/breathing frameworks.
- **Orientation-averaged external potential** — Fibonacci-sphere quadrature over SO(3) with N_Ω = 20 rotations; the per-orientation Boltzmann average is cached to disk and reused across all pressure points.
- **Pluggable `Potential` interface** — swap between LJ, Morse, Gaussian-smeared Coulomb, quadrupole–electric-field-gradient coupling, or any future ML potential without touching the solver.
- **Anderson-accelerated Picard solver** — fixed-point iteration in log-density space with voxel masking for inaccessible sites; converges in O(10²) iterations even near saturation.
- **Diagnostics layer** — Henry-constant cross-check, binding-site characterisation, isosteric-heat computation, and standardised plots for every phase of the calculation.

---

## Installation

### Recommended: uv (fast)

[uv](https://docs.astral.sh/uv/) is the recommended way to install porecdft. It resolves dependencies in seconds and creates an isolated environment automatically.

```bash
# 1. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or: pip install uv

# 2. Clone the repository
git clone https://github.com/Feugmo-Group/porecdft.git
cd porecdft

# 3. Create a virtual environment and install
uv venv                        # creates .venv/
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e .            # core install (NumPy, SciPy, pymatgen, matplotlib)

# 4. Optional: install JAX for GPU-accelerated Vext computation
uv pip install -e ".[jax]"
```

### pip / conda

```bash
git clone https://github.com/Feugmo-Group/porecdft.git
cd porecdft
pip install -e .
```

### Requirements

| Package    | Minimum | Tested  |
|------------|---------|---------|
| Python     | 3.10    | 3.12    |
| NumPy      | 2.0     | 2.3     |
| SciPy      | 1.10    | 1.15    |
| pymatgen   | 2024.1  | 2025.6  |
| matplotlib | 3.8     | 3.10    |
| JAX *(opt)*| 0.4     | 0.6     |

> **Note:** CIF files are read with `pymatgen.io.cif`. The `ase` package is **not** required.

---

## Package layout

```
porecdft/
  io/           CIF, force-field, and partial-charge readers
  structure/    HostAtoms, supercell builder, pore-volume probes, site finders
  forcefield/   Potential ABC + LJ, Morse, Coulomb, quadrupole-EFG,
                composite, and MLIP adapter implementations
  fluid/        Fluid ABC + CO₂ (EPM2/TraPPE), N₂, CH4, generic single-site
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

The complete workflow — from a CIF file to a converged isotherm — is illustrated in `applications/alf_co2/`. The high-level steps are:

### 1. Load the host structure

```python
from porecdft.io.cif import read_cif
from porecdft.io.charges import assign_hirshfeld_charges

# read_cif returns a HostAtoms with zero charges
host = read_cif("ALF.cif")

# Assign DDEC6 partial charges from a CSV (columns: element,charge[,source])
# charges.dat example:  Al,1.92,DDEC6 / C,0.48,DDEC6 / O,-0.65,DDEC6 / H,0.18,DDEC6
host = assign_hirshfeld_charges(host, charge_file="charges.dat", source="DDEC6")
```

### 2. Define the fluid and force field

```python
import numpy as np
from porecdft.fluid.co2 import CO2_EPM2          # Fluid instance (EPM2 3-site model)
from porecdft.io.forcefield import FFEntry
from porecdft.forcefield.lj import LJPotential
from porecdft.forcefield.coulomb import SmearCoulombPotential
from porecdft.forcefield.quadrupole import QuadrupolePotential
from porecdft.forcefield.composite import CompositePotential

fluid = CO2_EPM2    # Fluid instance — no parentheses, it is already instantiated

# Host-side UFF/DREIDING LJ parameters (σ in Å, ε/k_B in K)
host_ff = {
    "Al": FFEntry("Al", 4.008, 254.3, "UFF"),
    "C":  FFEntry("C",  3.431, 52.8,  "DREIDING"),
    "O":  FFEntry("O",  3.118, 30.2,  "DREIDING"),
    "H":  FFEntry("H",  2.571, 22.1,  "DREIDING"),
}

# Fluid LJ parameters come from the Fluid object (EPM2 values)
fluid_ff = fluid.ff   # dict[str, FFEntry] keyed by site label

potential = CompositePotential([
    LJPotential(host_ff=host_ff, fluid_ff=fluid_ff),
    SmearCoulombPotential(fluid_charges=fluid.charges, gauss_width=1.0),
    QuadrupolePotential(theta_zz=fluid.theta_zz),
])
```

### 3. Build the orientation-averaged external potential

```python
from porecdft.vext.orientations import fibonacci_rotations
from porecdft.vext.builder import build_vext_on_grid

# 20 Fibonacci-sphere orientations (increase to 50–100 for production)
orientations = fibonacci_rotations(20)

vext_data = build_vext_on_grid(
    host, fluid, potential,
    orientations=orientations,
    spacing=0.5,            # Å — grid resolution
    temperature_K=298.0,    # for Boltzmann averaging over orientations
    cache_path="vext_298K.npy",   # reuse on re-run
)
# vext_data keys: 'vext_avg', 'grid_shape', 'dV', 'orient_min', 'orient_argmin'
```

### 4. Pre-compute FMT weight functions and run the cDFT solver

```python
import jax.numpy as jnp
from porecdft.functional.fmt import FMTFunctional, make_k_grid, make_fmt_weights_hat
from porecdft.solver.anderson import anderson_solve
from porecdft.eos.ideal_gas import density_from_pressure

sigma_HS = 3.3          # hard-sphere diameter (Å) for CO₂ in ALF
T_K      = 298.0
pressure_bar = 1.0

# Bulk number density in molecules/Å³ from ideal gas (good to < 1% for CO₂ at T≥273 K, p≤10 bar)
rho_bulk = density_from_pressure(pressure_bar, T_K)   # note: pressure first, temperature second

shape = vext_data["grid_shape"]
fmt = FMTFunctional(sigma_HS=sigma_HS)

# Build Fourier-space weight functions once (reuse inside the solver loop).
# dx is the voxel edge length: for a regular grid dV = dx·dy·dz and for a
# cubic cell dx = dy = dz = dV^(1/3).
dV = vext_data["dV"]
dx = dy = dz = dV ** (1.0 / 3.0)   # exact for cubic cells (ALF is cubic Im-3)
KX, KY, KZ, K = make_k_grid(shape, dx, dy, dz)
w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, sigma_HS)

c1_ref = fmt.c1_bulk(rho_bulk)

result = anderson_solve(
    rho_init=rho_bulk * np.ones(shape),
    rho_bulk=rho_bulk,
    Vext_K=vext_data["vext_avg"],
    temperature_K=T_K,
    c1_callable=lambda rho: fmt.c1(jnp.asarray(rho), w2_hat, w3_hat, w2vec_hat),
    c1_bulk=c1_ref,
    m=8,
    beta=0.3,
)
print(f"Converged: {result.converged}  ({result.iterations} iterations)")
```

### 5. Compute the adsorbed amount and Henry constant

```python
from porecdft.diagnostics.henry import henry_constant_from_vext

# Henry constant (mol / kg / bar) — cross-check against low-pressure isotherm
pore_volume_A3 = 0.31e3 * 1.27   # 0.31 cm³/g × 1.27 g/cm³ × 1e3 Å³/cm³  (ALF)
K_H = henry_constant_from_vext(
    vext_data["vext_avg"],
    dV=vext_data["dV"],
    temperature_K=T_K,
    pore_volume=pore_volume_A3,
)
print(f"K_H = {K_H:.3f} mmol/g/bar")

# Absolute adsorption (mmol/g) — integrate ρ(r) over the unit cell
cell_mass_g = 44.01 / 6.022e23 * 104   # formula weight for ALF (104-atom unit cell)
N_ads = float(result.rho.sum() * vext_data["dV"])          # molecules / unit cell
loading_mmol_g = N_ads / (cell_mass_g * 6.022e23 / 44.01) * 1000
print(f"Loading at {pressure_bar} bar, {T_K} K: {loading_mmol_g:.2f} mmol/g")
```

---

## H₂ adsorption in metalated COFs (Morse potential example)

H₂ binding at open transition-metal sites is dominated by a short-range Morse well
rather than a LJ + electrostatics potential. The workflow uses `MorsePotential` for
the metal sites and standard DREIDING LJ for the organic framework atoms. Morse
parameters are B3LYP-D3/GULP values from Pramudya & Mendoza-Cortes,
*J. Am. Chem. Soc.* 2016, 138, 15535 (Table 2, scaled to per-molecule D_e).

```python
import numpy as np
import jax.numpy as jnp
from porecdft.io.cif import read_cif
from porecdft.fluid.h2 import SingleSite_H2
from porecdft.io.forcefield import FFEntry
from porecdft.forcefield.morse import MorsePotential, MorseParam
from porecdft.forcefield.lj import LJPotential
from porecdft.forcefield.composite import CompositePotential
from porecdft.vext.orientations import fibonacci_rotations
from porecdft.vext.builder import build_vext_on_grid
from porecdft.eos.ideal_gas import density_from_pressure
from porecdft.diagnostics.henry import henry_constant_from_vext
from porecdft.functional.fmt import FMTFunctional, make_k_grid, make_fmt_weights_hat
from porecdft.solver.anderson import anderson_solve

# ── 1. Morse parameters from Pramudya & Mendoza-Cortes 2016, Table 2 ─────────
# D_e per H₂ molecule = 2 × D_e(per H atom, kcal/mol) × 503.228 K/(kcal/mol)
KCAL_TO_K = 503.228
MORSE_PARAMS = {
    "Co": MorseParam("Co", D_e=2*0.879*KCAL_TO_K, a=0.850, r_e=2.985),  # 884.7 K
    "Fe": MorseParam("Fe", D_e=2*1.092*KCAL_TO_K, a=1.180, r_e=3.015),  # 1098.6 K
    "Ni": MorseParam("Ni", D_e=2*1.154*KCAL_TO_K, a=1.210, r_e=3.207),  # 1161.1 K
    "Cu": MorseParam("Cu", D_e=2*0.818*KCAL_TO_K, a=1.462, r_e=2.931),  # 823.3 K
    "Mn": MorseParam("Mn", D_e=2*0.994*KCAL_TO_K, a=0.990, r_e=3.015),  # 1000.4 K
}

# ── 2. DREIDING LJ for non-metal framework atoms (σ in Å, ε/k_B in K) ───────
DREIDING_LJ_HOST = {
    "H":  FFEntry("H",  2.84642,   7.64893, "DREIDING"),
    "C":  FFEntry("C",  3.47299,  47.85620, "DREIDING"),
    "N":  FFEntry("N",  3.26256,  38.94920, "DREIDING"),
    "O":  FFEntry("O",  3.03315,  48.15810, "DREIDING"),
    "Cl": FFEntry("Cl", 3.52000, 114.23000, "DREIDING"),
}
# H₂ single-site LJ (TraPPE, σ=2.83 Å, ε/k_B=59.7 K) for the organic background
H2_LJ_FLUID = {"H2": FFEntry("H2", 2.83, 59.7, "TraPPE")}

# ── 3. Load COF-301 with Co metal site ───────────────────────────────────────
metal = "Co"
host = read_cif("applications/h2_cof/structures/COF-301-CoCl2.cif")
# No partial charges: Morse potential is charge-free

# ── 4. Build composite potential: Morse on metal + LJ on framework ───────────
# MorsePotential handles only the metal element; LJPotential covers the rest.
# Elements not present in a potential's parameter dict are silently skipped.
morse_pot = MorsePotential(
    host_params={metal: MORSE_PARAMS[metal]},
    fluid_params={"H2": MorseParam("H2", D_e=1.0, a=1.0, r_e=0.0)},
    # D_e_pair = sqrt(D_e_host × D_e_fluid); set D_e_fluid=1 K so the pair
    # well depth equals sqrt(D_e_host) — the Pramudya parameters are already
    # per-molecule pair values, so set D_e_fluid = D_e_host to reproduce them:
)
# Simpler: use the host params directly as pair params (no combining rule needed)
# by passing a fluid dummy with D_e equal to 1 so D_e_pair ≈ sqrt(D_e_host).
# For exact reproduction of Pramudya 2016, use the benchmark script which
# computes the Morse well inline without combining rules.

lj_pot = LJPotential(host_ff=DREIDING_LJ_HOST, fluid_ff=H2_LJ_FLUID)

fluid = SingleSite_H2
potential = CompositePotential([morse_pot, lj_pot])

# ── 5. Build orientation-averaged Vext ───────────────────────────────────────
T_K = 77.0       # K — cryogenic H₂ storage benchmark (DOE 2025 target)
orientations = fibonacci_rotations(20)

vext_data = build_vext_on_grid(
    host, fluid, potential,
    orientations=orientations,
    spacing=0.7,             # Å — coarser grid is fine at 77 K
    temperature_K=T_K,
    cache_path="vext_cof301_co_77K.npy",
)

# ── 6. Henry constant ─────────────────────────────────────────────────────────
K_H = henry_constant_from_vext(
    vext_data["vext_avg"],
    dV=vext_data["dV"],
    temperature_K=T_K,
    pore_volume=host.cell_volume,
)
print(f"K_H (Co/COF-301, 77 K) = {K_H:.3f} mmol/g/bar")
# Expected from paper benchmark: ~17 mmol/g at 1 bar (Henry regime)

# ── 7. Full-pressure cDFT solve ───────────────────────────────────────────────
sigma_HS = 2.96    # Å — H₂ hard-sphere diameter
fmt   = FMTFunctional(sigma_HS=sigma_HS)
shape = vext_data["grid_shape"]
dV    = vext_data["dV"]

dx = dy = dz = dV ** (1.0 / 3.0)   # exact for cubic cells (COFs are tetragonal — use lattice vectors for precise non-cubic grids)

KX, KY, KZ, K = make_k_grid(shape, dx, dy, dz)
w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, sigma_HS)

pressure_bar = 1.0
rho_bulk = density_from_pressure(pressure_bar, T_K)
c1_ref   = fmt.c1_bulk(rho_bulk)

result = anderson_solve(
    rho_init=rho_bulk * np.ones(shape),
    rho_bulk=rho_bulk,
    Vext_K=vext_data["vext_avg"],
    temperature_K=T_K,
    c1_callable=lambda rho: fmt.c1(jnp.asarray(rho), w2_hat, w3_hat, w2vec_hat),
    c1_bulk=c1_ref,
    m=8,
    beta=0.3,
)
print(f"Converged: {result.converged}  ({result.iterations} iterations)")

# ── 8. Convert to mmol/g ─────────────────────────────────────────────────────
from pymatgen.core import Structure
pmg = Structure.from_file("applications/h2_cof/structures/COF-301-CoCl2.cif")
cell_mass_amu = sum(s.atomic_mass for s in pmg.species)   # g/mol per unit cell
N_ads = float(result.rho.sum() * dV)                      # molecules / unit cell
loading = N_ads / cell_mass_amu * 1000.0                  # mmol/g
print(f"H₂ loading (Co/COF-301, {T_K} K, {pressure_bar} bar): {loading:.2f} mmol/g")
```

The complete 4 COF × 5 metal benchmark (COF-301, COF-322, COF-330, COF-333) × (Co, Fe,
Ni, Cu, Mn) showing Co > Mn > Ni > Fe > Cu in every framework is in
`applications/h2_cof/notebooks/make_h2_cof_benchmark.py`.

---

## The `Potential` interface

Any external-field source can be used by subclassing `porecdft.forcefield.base.Potential` and implementing two methods:

```python
class MyPotential(Potential):
    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels):
        # return PotentialEnergy(total=..., parts={...})
        ...

    def energy_grid(self, grid_xyz, rot, host, fluid_sites, fluid_site_labels):
        # vectorized version over grid_xyz — override for speed
        ...
```

Energy units are **Kelvin** (Boltzmann units, ε/k_B convention) throughout; conversion to kJ/mol occurs only at the reporting boundary.

Built-in implementations:

| Class | Description |
|-------|-------------|
| `LJPotential` | 12-6 Lennard–Jones with Lorentz–Berthelot mixing, 15 Å cutoff |
| `SmearCoulombPotential` | Gaussian-smeared charges, 3×3×3 periodic images |
| `QuadrupolePotential` | CO₂ quadrupole – framework electric-field-gradient coupling |
| `MorsePotential` | Morse well for transition-metal binding sites in COFs |
| `CompositePotential` | Sum of any set of Potential instances |
| `MLIPPotential` | Stub adapter for MACE / NequIP / Allegro output grids |

---

## Benchmarks

### CO₂ in aluminum formate (ALF)

ALF (Al(HCOO)₃, cubic Im-3, Evans et al. *Sci. Adv.* 2022) is a stringent test: it simultaneously exhibits cooperative pore filling, framework breathing/gate-opening, and kinetic molecular sieving.

| Quantity | Experiment | porecdft |
|----------|-----------|---------|
| SC binding energy | −48.4 kJ/mol | −48.0 kJ/mol (< 1%) |
| LC binding energy | −36.2 kJ/mol | −36.0 kJ/mol (< 1%) |
| 298 K isotherm RMSE | — | 0.33 mmol/g |
| Isosteric heat | 25–32 kJ/mol (cal.) | 25–32 kJ/mol |
| IAST CO₂/N₂ selectivity | ~4 (thermodynamic) | ~4 |

The experimental separation factor of 350–600 is a transport (kinetic) property; the thermodynamic IAST value confirms ALF is a kinetic molecular sieve.

### H₂ in metalated COFs

Morse external potentials are applied to H₂ adsorption in COF-301, COF-322, COF-330, and COF-333 with five first-row transition metals. Cobalt gives the highest Henry-regime uptake in every framework owing to its broad, soft Morse well — a result that requires 3D treatment to resolve.

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

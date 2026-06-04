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
from porecdft.eos.ideal_gas import ideal_gas_density

sigma_HS = 3.3          # hard-sphere diameter (Å) for CO₂ in ALF
T_K      = 298.0
pressure_bar = 1.0

# Bulk density from equation of state (mol/L → molecules/Å³)
rho_bulk = ideal_gas_density(T_K, pressure_bar)

shape = vext_data["grid_shape"]
fmt = FMTFunctional(sigma_HS=sigma_HS)

# Build Fourier-space weight functions once (reuse inside the solver loop)
# grid spacing dV and shape are needed to derive dx, dy, dz
dV = vext_data["dV"]
cell_volume = dV * shape[0] * shape[1] * shape[2]
dx = (cell_volume / (shape[1] * shape[2])) ** (1/3)   # approximate; exact for cubic cells
dy = dz = dx
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
rather than a LJ + electrostatics potential. The workflow is identical to the CO₂
example above, but replaces the `CompositePotential` with a single `MorsePotential`.
Morse parameters below are from Pramudya & Mendoza-Cortes, *J. Am. Chem. Soc.* 2016.

```python
import numpy as np
from porecdft.io.cif import read_cif
from porecdft.fluid.h2 import SingleSite_H2
from porecdft.forcefield.morse import MorsePotential, MorseParam
from porecdft.vext.orientations import fibonacci_rotations
from porecdft.vext.builder import build_vext_on_grid
from porecdft.eos.ideal_gas import density_from_pressure
from porecdft.diagnostics.henry import henry_constant_from_vext

# --- 1. Load metalated COF structure -----------------------------------------
# CIF should have the metal atom already in place (e.g. Co in COF-301-CoCl2.cif)
host = read_cif("applications/h2_cof/structures/COF-301-CoCl2.cif")
# No partial charges needed for Morse-only potential

# --- 2. Morse potential at the cobalt site ------------------------------------
# Pramudya 2016 B3LYP-D3 parameters (D_e in K, a in 1/Å, r_e in Å)
MORSE_HOST = {
    "Co": MorseParam("Co", D_e=884.7,  a=0.850, r_e=2.20),
}
MORSE_FLUID = {
    "H2": MorseParam("H2", D_e=1.0,    a=1.0,   r_e=0.0),   # fluid param cancels via geometric mean
}
# For a single-site fluid the combining rule gives an effective pair potential
# D_e_pair = sqrt(D_e_host * D_e_fluid_ref); in practice use host params directly:
MORSE_HOST_EFF = {
    "Co": MorseParam("Co", D_e=884.7, a=0.850, r_e=2.20),
    # Add non-metal framework atoms if needed (LJ fallback or small Morse well)
    "C":  MorseParam("C",  D_e=29.0,  a=1.10,  r_e=3.80),
    "N":  MorseParam("N",  D_e=35.0,  a=1.10,  r_e=3.60),
    "H":  MorseParam("H",  D_e=12.0,  a=1.10,  r_e=3.20),
    "Cl": MorseParam("Cl", D_e=50.0,  a=1.10,  r_e=3.60),
}

fluid = SingleSite_H2
potential = MorsePotential(
    host_params=MORSE_HOST_EFF,
    fluid_params={"H2": MorseParam("H2", D_e=1.0, a=1.0, r_e=0.0)},
)

# --- 3. Build orientation-averaged Vext --------------------------------------
# Single-site fluid: orientation averaging is trivial (1 orientation)
# but fibonacci_rotations(20) averages the angular part of the Morse potential.
orientations = fibonacci_rotations(20)
T_K = 77.0      # K (cryogenic H₂ storage benchmark)

vext_data = build_vext_on_grid(
    host, fluid, potential,
    orientations=orientations,
    spacing=0.5,
    temperature_K=T_K,
    cache_path="vext_cof301_77K.npy",
)

# --- 4. Henry constant (mmol/g/bar) ------------------------------------------
# For single-site H₂ at low pressure, K_H alone characterises uptake
# (Henry regime applies at 77 K, 1 bar for most COFs with D_e < 2000 K)
pore_volume_A3 = host.cell_volume  # rough; replace with He-probe volume if available
K_H = henry_constant_from_vext(
    vext_data["vext_avg"],
    dV=vext_data["dV"],
    temperature_K=T_K,
    pore_volume=pore_volume_A3,
)
print(f"K_H (Co, 77 K) = {K_H:.3f} mmol/g/bar")

# --- 5. Run the cDFT solver for a full isotherm point ------------------------
import jax.numpy as jnp
from porecdft.functional.fmt import FMTFunctional, make_k_grid, make_fmt_weights_hat
from porecdft.solver.anderson import anderson_solve

sigma_HS = 2.96   # Å — H₂ hard-sphere diameter
fmt = FMTFunctional(sigma_HS=sigma_HS)

shape = vext_data["grid_shape"]
dV    = vext_data["dV"]
cell_volume = dV * shape[0] * shape[1] * shape[2]
dx = dy = dz = (cell_volume / (shape[0] * shape[1] * shape[2])) ** (1/3)
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

# Convert to mmol/g
cell_mass_g = fluid.molar_mass / 6.022e23 * host.n_atoms  # approximate; use true MW
N_ads = float(result.rho.sum() * dV)
loading = N_ads / (6.022e20 * cell_mass_g / fluid.molar_mass)   # mmol/g
print(f"H₂ loading (Co, {T_K} K, {pressure_bar} bar): {loading:.2f} mmol/g")
```

The complete 4 COF × 5 metal benchmark (Co, Fe, Ni, Cu, Mn) is in
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

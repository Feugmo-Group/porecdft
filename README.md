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

```bash
git clone https://github.com/Feugmo-Group/porecdft.git
cd porecdft
pip install -e .
```

### Requirements

| Package | Version tested |
|---------|----------------|
| Python  | 3.10+          |
| NumPy   | 2.3            |
| JAX     | 0.6            |
| SciPy   | 1.15           |
| pymatgen | 2025.6+       |
| matplotlib | 3.10        |

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

host = read_cif("ALF.cif")
host = assign_hirshfeld_charges(host, charge_file="charges.dat")
```

### 2. Define the fluid and force field

```python
from porecdft.fluid.co2 import CO2_EPM2
from porecdft.forcefield.composite import CompositePotential
from porecdft.forcefield.lj import LJPotential
from porecdft.forcefield.coulomb import SmearCoulombPotential
from porecdft.forcefield.quadrupole import QuadrupolePotential

fluid = CO2_EPM2()
potential = CompositePotential([
    LJPotential(ff_params),
    SmearCoulombPotential(sigma_smear=2.0),
    QuadrupolePotential(),
])
```

### 3. Build the orientation-averaged external potential

```python
from porecdft.vext.orientations import fibonacci_rotations
from porecdft.vext.builder import build_vext_on_grid

orientations = fibonacci_rotations(N=20)
vext_data = build_vext_on_grid(
    host, fluid, potential,
    orientations=orientations,
    spacing=0.5,          # Å
    temperature_K=298.0,
    cache_path="vext_298K.npy",
)
```

### 4. Run the cDFT solver

```python
from porecdft.functional.fmt import FMTFunctional
from porecdft.functional.association import WertheimAssociation
from porecdft.solver.anderson import anderson_solve

fmt = FMTFunctional(sigma_HS=3.3)
assoc = WertheimAssociation(n_sites=7, eps_assoc=400.0, kappa=119.0)

result = anderson_solve(
    rho_init=rho_bulk * np.ones(vext_data["grid_shape"]),
    rho_bulk=rho_bulk,
    Vext_K=vext_data["vext_avg"],
    temperature_K=298.0,
    c1_callable=lambda rho: fmt.c1(rho) + assoc.c1(rho),
    c1_bulk=fmt.c1_bulk(rho_bulk) + assoc.c1_bulk(rho_bulk),
    m=8,
    beta=0.3,
)
```

### 5. Compute the Henry constant (cross-check)

```python
from porecdft.diagnostics.henry import henry_constant_from_vext

K_H = henry_constant_from_vext(
    vext_data["vext_avg"],
    dV=vext_data["dV"],
    temperature_K=298.0,
    pore_volume=pore_volume_A3,
)
print(f"K_H = {K_H:.2f} mmol/g/bar")
```

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

The package is archived on Zenodo: [10.5281/zenodo.19008858](https://zenodo.org/records/19008858)

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Conrard Giresse Tetsassi Feugmo  
Department of Chemistry, University of Waterloo  
cgtetsas@uwaterloo.ca

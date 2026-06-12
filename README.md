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

# CPU machine
uv sync

# GPU machine — CUDA JAX + Warp in one step
uv sync --extra gpu
```

### With pip

```bash
pip install -e .           # minimal (CPU)
pip install -e ".[dev]"    # + pytest, ruff
pip install -e ".[jax]"    # + JAX + equinox (CPU JAX)
pip install -e ".[gpu]"    # + CUDA JAX + equinox + optax + NVIDIA Warp (GPU machine)
pip install -e ".[warp]"   # + Warp only (manage JAX separately)
```

### Conda environment (used in development)

The `jax` conda environment contains all dependencies:

```bash
conda activate jax
pip install -e ".[dev]"
uv sync --group dev   # alternative: uv manages the dev group
pytest -m "not slow"
```

### Requirements

| Package             | Minimum | Tested  |
|---------------------|---------|---------|
| Python              | 3.10    | 3.12    |
| NumPy               | 2.0     | 2.3     |
| SciPy               | 1.10    | 1.15    |
| pymatgen            | 2024.1  | 2025.6  |
| matplotlib          | 3.8     | 3.10    |
| JAX *(opt: `jax`)*  | 0.4     | 0.6     |
| equinox *(opt: `jax`)* | 0.13 | 0.13   |
| optax *(opt: `gpu`)*| 0.2     | 0.2     |
| warp-lang *(opt: `gpu`, `warp`)* | 1.4 | 1.4 |

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

# ── Solver comparison: Picard / Anderson / Adam / FIRE2 ──────────────────────
# Runs all four porecdft solvers on COF-333-CoCl2 (ideal-gas c1=0) and
# compares isotherms + convergence curves.  Runtime: ~1 min.
uv run python applications/h2_cof/notebooks/make_h2_solver_comparison.py
# Output: figures/h2_solver_comparison.png
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
  eos/          Bulk equations of state — see "Equations of state" below
  functional/   Free-energy functionals: aWBII FMT, Wertheim TPT-1 association,
                elastic framework penalty
  solver/       Picard iteration, Anderson mixing, FIRE minimiser
  diagnostics/  Binding-site probe, Henry constant, isosteric heat
  plotting/     Standardised diagnostic figures
  warp_backend/ Optional NVIDIA Warp GPU kernels for 3D hot paths
```

---

## What's implemented — full equation reference

This section catalogues every formula coded in `porecdft` so users can map module → physics directly.

### 1. Grand-potential functional (the equation everything else minimises)

```
Ω[ρ, L] = F_id[ρ] + F_ex[ρ] + ∫ [V_ext(r; L, T) − μ] ρ(r) dr + ½ K_eff (L − L₀)²
```

* `F_id[ρ] = k_B T ∫ ρ(r) [ln(Λ³ ρ(r)) − 1] dr` — ideal-gas free energy (exact).
* `F_ex[ρ]` — sum of FMT-aWBII, Wertheim TPT-1, and WDA contributions (below).
* `V_ext(r; L, T)` — composite, orientation-averaged external potential (below).
* `½ K_eff (L − L₀)²` — affine-elastic framework penalty; `L` is a global lattice scale.

Self-consistency (Euler–Lagrange):
```
ρ(r) = ρ_bulk · exp[−β V_ext(r; L, T) + c⁽¹⁾(r) − c⁽¹⁾_b]
c⁽¹⁾(r) = −β · δF_ex/δρ(r)
```

### 2. Composite external potential `V_ext`

`porecdft.forcefield/`.

**Lennard-Jones 12-6** (`forcefield/lj.py`):
```
V_LJ(r) = 4 ε_ij [(σ_ij/r)¹² − (σ_ij/r)⁶]
```
Lorentz–Berthelot mixing: `σ_ij = (σ_i+σ_j)/2`, `ε_ij = √(ε_i ε_j)`.

**Gaussian-smeared Coulomb** (`forcefield/coulomb.py`):
```
V_Coul(r) = (q_i q_j) / (4π ε₀ r) · erf(r / (√2 σ_eff))
σ_eff² = σ_i² + σ_j²
```
Cures `1/r` divergence + FFT grid aliasing. Default `σ_smear = 2.0` Å, 3³ PBC.

**Quadrupole–electric-field-gradient** (`forcefield/quadrupole.py`):
```
V_Q-EFG(r, Ω) = −⅓ · Θ_αβ^mol(Ω) · V_αβ^host(r)
V_αβ^host(r) = ∂²Φ_host / (∂r_α ∂r_β)  (computed via FFT Hessian of smeared Φ_host)
```
For CO₂: `Θ_zz = −4.30 × 10⁻²⁶ esu cm²`.

**Morse** (`forcefield/morse.py`) — for transition-metal sites in COFs:
```
V_Morse(r) = D_e · [(1 − e^(−α(r − r_e)))² − 1]
```

### 3. Orientation averaging — the rotational free energy

`vext/builder.py`. For polyatomic adsorbates with body-frame site positions `{s_α}`:
```
V_ext(r; T) = −k_B T · ln [ (1/N_Ω) Σ_i exp(−β Σ_α V_α(r + R(Ω_i) s_α)) ]
```
The result is a **rotational free energy**, not a bare potential. Orientations `{Ω_i}` sampled by **Fibonacci-sphere quadrature** on SO(3) with `N_Ω = 20`. Cached as `.npy`; reused across all `(P, T)` points.

### 4. Free-energy functionals (`functional/`)

**FMT-aWBII** (`functional/fmt.py`) — Hansen-Goos & Roth (2006):
```
F_ex^aWBII = k_B T · ∫ Φ^aWBII(n_α(r)) dr
n_α(r)    = ∫ ρ(r′) ω^α(r − r′) dr′,   α ∈ {0, 1, 2, 3, V1, V2}
```
Six weight functions ω^α (scalar + vector). FFT convolutions under periodic boundary conditions; **Lanczos anti-aliasing** filter applied to FFT weights (stabilises the BH-small / coarse-grid limit).

**WDA attractive `c⁽¹⁾`** (`functional/lj_wda.py`) — Weighted-Density Approximation for the long-range LJ/Morse tail.

**Wertheim TPT-1 association** (`functional/association.py`) — Henderson 2021:
```
F_assoc[ρ] = n_SC · ∫ ρ(r) [ln X(r) − X(r)/2 + ½] dr
X(r) = (−1 + √(1 + 4 ρ κ exp(ε_assoc/T))) / (2 ρ κ exp(ε_assoc/T))
```
For CO₂/ALF: `n_SC = 7`, `κ = 119` Å³, `ε_assoc = 400 K` (= DFT SC–LC binding-energy difference).

**Elastic framework response** (`functional/elastic.py`):
```
Ω_tot(L; ρ) = Ω[ρ; V_ext(r; L)] + ½ K_eff (L − L₀)²
```
Reduction of the formal `F_tot[ρ, u] = F_fluid + F_elastic` framework to a single affine
parameter `u(r) = (L/L₀ − 1) r`. For CO₂/ALF: `K_eff = 0.7 GPa` (~20× softer than bulk).
Iterative ρ ↔ L loop:
1. Fix `L` → rebuild `V_ext(r; L)` on strained grid.
2. Solve Euler–Lagrange for `ρ(r)`.
3. Compute strain `f(L) = −∂_L ∫ V_ext(r; L) ρ(r) dr`.
4. Update `L` via `K_eff (L − L₀) = f(L)`. Repeat until `|ΔL| < ε`.

### 5. Solvers (`solver/`)

**Picard fixed-point** (log-density variant prevents `n_3 > 1` overshoots):
```
ρ_{k+1} = (1 − α) ρ_k + α · T[ρ_k]
ln ρ_{k+1} = ln ρ_k + α · (ln T[ρ_k] − ln ρ_k)
```
Default `α = 0.02` with pressure continuation (warm-start from previous (P, T)).

**Anderson acceleration**:
```
ρ_{k+1} = ρ_k + β · Σ_{j=0}^{m−1} c_j · g_{k−j},     g_k = T[ρ_k] − ρ_k
min Σ c_j g_{k−j} ²  s.t.  Σ c_j = 1
```
Defaults: history depth `m = 8`, damping `β = 0.1`, safeguard fallback to Picard at α=0.01 if residual rises.

**Adam (optax)**: Minimises `Ω[ρ]` by gradient descent. Gradient `∇_ρ Ω` from JAX autodiff. lr = 2e-3, 5000 steps. Reparametrise `ρ = ρ_b · exp(η(r))` to enforce positivity.

**FIRE2 / NonlinearCG (optimistix)**: Inertial relaxation. JIT-compiled in JAX, runs on CPU or GPU. Fastest per iteration on the correct basin.

### 6. Bulk equations of state (`eos/`) — v0.2

`bulk_density(P_bar, T_K) → molecules/Å³` (gas branch).

| Class | Physics | When to use | JIT_SAFE |
|-------|---------|-------------|----------|
| `density_from_pressure` | Ideal gas: `ρ = P/(k_B T)` | low-P / high-T limit | ✅ |
| `PengRobinsonEOS` | Cubic, Peng-Robinson 1976: `P = RT/(V−b) − a(T)/(V²+2bV−b²)` | H₂ 0–500 bar, light gases | runs on JAX |
| `SRKEOS` | Cubic, Soave 1972: `P = RT/(V−b) − a(T)/(V(V+b))` with `κ_SRK(ω) = 0.480 + 1.574ω − 0.176ω²` | hydrocarbons VLE | runs on JAX |
| `SpanWagnerCO2EOS` | Reference Helmholtz: `α^r(δ,τ) = Σ N_i δ^d_i τ^t_i` (7-term truncation of S&W 1996 Table 31) | CO₂ near critical | **✅ jax.jit-able** |
| `PCSAFTEOS` | Gross & Sadowski 2001: `ã^res = ã^hc + ã^disp`, `Z = 1 + ρ ∂ã^res/∂ρ` (autodiff) | mixtures, chains | runs on JAX |
| `CPAEOS` | SRK + Wertheim TPT-1: `P = P_SRK + P_assoc`, `X = 2/(1+√(1+8ρΔ))` for 4C scheme | water, alcohols, amines | NumPy |
| `SAFTVRMieEOS` | Carnahan-Starling HS + leading-order Mie dispersion (Lafitte 2013, simplified) | CO₂ alternative | NumPy |
| `LJEOS` (MBWR) | Modified Benedict-Webb-Rubin for Lennard-Jones (Johnson 1993) | FMT-bulk-limit consistency check | NumPy |
| `FeynmanHibbsEOS` | Quantum-corrected wrapper: `ρ_FH(P, T) = ρ_classical · f_Q(T)`, `f_Q = 1/(1 + Λ*²/12)`, `Λ* = h/(σ √(2π m k_B T))` | cryogenic H₂ (77 K) | NumPy float64 |

All EOS subclass `EOSBase` and expose `JIT_SAFE` / `GPU_READY` class attributes.

### 7. NVIDIA Warp GPU kernels (`warp_backend/`)

Optional GPU acceleration of the three biggest 3D hot paths. All kernels degrade gracefully when `warp-lang` is not installed.

| Kernel | Purpose | Replaces |
|--------|---------|----------|
| `rho_bar_sphere_kernel` | Wertheim ρ̄_s = (1/κ_s) ∫_{ \|r−r_s\| < r_κ} ρ(r) dV for all M sites in parallel; one thread per voxel, atomic-add per site | `WertheimAssociation._rho_bar_all` Python loop (O(M × N_g)) |
| `lj_vext_grid_kernel` | LJ V_ext on (N_g,) grid for one orientation, multi-site fluid: `Σ_{s,a} 4 ε_ij[(σ_ij/r)¹² − (σ_ij/r)⁶]` | `LJPotential.energy_grid` per-orientation Python loop |
| `morse_vext_grid_kernel` | Morse-well V for transition-metal sites: `D_e[(1−e^(−α(r−r_e)))²−1]` (with `D_e` cap below) | `MorsePotential.energy_grid` per-orientation loop |
| `smeared_coulomb_grid_kernel` | erf-smeared Coulomb per (S, N_a) pair: `q_i q_j / r · erf(r/(√2 σ_eff)) · k_e e²/k_B` | `CoulombPotential.energy_grid` per-orientation loop |
| `boltzmann_orient_avg_kernel` | Final orientation reduction `V(r;T) = −k_B T ln (1/N_Ω Σ_i exp(−β V_i))` with min-shift LSE | NumPy reduction in `vext/builder.py` |

Each Vext kernel does **one orientation per launch**; the outer 20-orientation loop becomes 20 kernel launches. Per-thread work: 1 voxel × N_sites × N_atoms inner-loop with cutoff masking. Expected speedup on CUDA: **10–100× per orientation** over NumPy `energy_grid`; cuts `build_vext_on_grid` from minutes to seconds.

Python wrappers `lj_vext_grid_warp` and `boltzmann_orient_avg_warp` include a CPU fallback for correctness testing (`tests/test_vext_warp.py`).

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

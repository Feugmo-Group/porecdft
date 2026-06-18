# porecdft

**porecdft** is an open-source Python package for three-dimensional classical density functional theory (cDFT) of gas adsorption in nanoporous materials — metal–organic frameworks (MOFs), covalent organic frameworks (COFs), and zeolites.

The package is host-agnostic and fluid-agnostic. Any analytic or machine-learning external potential plugs in through a single `Potential` interface without touching the solver or functional layers.

> **Paper figures and benchmarks:** see [`applications/README.md`](applications/README.md).
> The manuscript LaTeX source is at [`Submission/paper/`](../Submission/paper/) (outside this repo) — its README lists which script generates each figure.

---

## Contents

- [Installation](#installation)
- [Key physics: the grand-potential functional](#key-physics-the-grand-potential-functional)
- [Solvers](#solvers)
- [Quick start](#quick-start)
- [Configuration system](#configuration-system)
- [Package layout](#package-layout)
- [Equations of state](#equations-of-state)
- [Systems tested](#systems-tested)
- [GPU acceleration](#gpu-acceleration)
- [Citation](#citation)

---

## Installation

```bash
git clone https://github.com/Feugmo-Group/porecdft.git
cd porecdft

# CPU (recommended for development)
uv sync

# GPU (CUDA JAX + NVIDIA Warp in one step)
uv sync --extra gpu
```

Or with pip:

```bash
pip install -e .            # minimal CPU
pip install -e ".[dev]"     # + pytest, ruff
pip install -e ".[gpu]"     # + CUDA JAX + optax + Warp
```

The development conda environment is `jax` (`conda activate jax`).

---

## Key physics: the grand-potential functional

Everything in porecdft minimises the same functional:

```
Ω[ρ] = F_id[ρ] + F_exc[ρ] + ∫ [V_ext(r) − μ] ρ(r) dr
```

| Term | Expression | Module |
|------|-----------|--------|
| `F_id` | `k_B T ∫ ρ [ln(Λ³ρ) − 1] dr` — ideal gas (exact) | — |
| `F_exc` | FMT-aWBII + WDA-LJ + Wertheim TPT-1 association | `functional/` |
| `V_ext` | orientation-averaged composite potential | `vext/`, `forcefield/` |
| `μ` | bulk chemical potential from EOS | `eos/` |

The minimum satisfies the **Euler–Lagrange fixed-point condition**:

```
ρ*(r) = ρ_bulk · exp[ c⁽¹⁾[ρ*](r) − c⁽¹⁾_bulk − β V_ext(r) ]

c⁽¹⁾(r) = −δF_exc / δρ(r) / (k_B T)
```

### F_exc: the excess free energy

The excess free energy combines three contributions:

**FMT-aWBII** (Hansen-Goos & Roth 2006) — hard-sphere repulsion via weighted densities:
```
F_exc^FMT = k_B T ∫ Φ^aWBII(n₀, n₁, n₂, n₃, n_V1, n_V2) dr
nα(r) = ∫ ρ(r′) ωα(r − r′) dr′,   α ∈ {0,1,2,3,V1,V2}
```
Six weight functions (scalar + vector); FFT convolutions with Lanczos anti-aliasing.

**WDA-LJ** (`functional/lj_wda.py`) — Weighted Density Approximation for the long-range LJ/Morse attractive tail.

**Wertheim TPT-1 association** (`functional/association.py`) — for directional interactions (CO₂/ALF gate-opening):
```
F_assoc[ρ] = n_SC · ∫ ρ(r) [ln X(r) − X(r)/2 + ½] dr
X(r) = (−1 + √(1 + 4 ρ κ exp(ε_assoc / T))) / (2 ρ κ exp(ε_assoc / T))
```

### External potential V_ext

For a polyatomic fluid the orientation-averaged potential is a **rotational free energy**, not a bare sum:

```
V_ext(r; T) = −k_B T · ln [ (1/N_Ω) Σ_i exp(−β Σ_α V_α(r + R(Ω_i) s_α)) ]
```

Orientations sampled by Fibonacci-sphere quadrature (`N_Ω = 20`). Cached as `.npy` and reused across all `(P, T)` points.

Supported pair potentials (`forcefield/`):

| Class | Physics |
|-------|---------|
| `LJPotential` | 12-6 Lennard–Jones, Lorentz–Berthelot mixing |
| `CoulombPotential` | Gaussian-smeared: `V = q_i q_j / r · erf(r / √2 σ_eff)` |
| `QuadrupoleEFGPotential` | CO₂ quadrupole × framework EFG: `V = −⅓ Θ_αβ V_αβ^host` |
| `MorsePotential` | Morse well for transition-metal sites: `V = D_e[(1−e^{−α(r−r_e)})²−1]` |
| `CompositePotential` | Sum of any combination of the above |

---

## Solvers

Four solvers are available in `solver/`. All minimise the same grand-potential functional and produce identical converged density profiles when starting from the same basin.

| Solver | Function | Backend | When to use |
|--------|----------|---------|-------------|
| Picard | `picard_solve` | NumPy | Dilute / low-pressure; safe warm-start seed |
| Anderson | `anderson_solve` | NumPy | **Production isotherms** — most robust at high packing |
| Adam (optax) | `jax_solve` | JAX / GPU | Gradient-based minimisation; differentiable pipeline |
| FIRE2 / NonlinearCG | `fire2_solve` | JAX / GPU | Inertial relaxation; fastest per-iteration on GPU |

### Picard fixed-point (`picard_solve`)

Log-density update prevents packing-fraction overshoot:

```
ρ_{k+1} = ρ_bulk · exp[ c⁽¹⁾[ρ_k] − c⁽¹⁾_bulk − β V_ext ]
```

Default step `α = 0.02`. Use for dilute systems or as the initial seed for Anderson.

### Anderson mixing (`anderson_solve`)

Constrained least-squares acceleration over a history of `m` residuals:

```
min ‖Σⱼ cⱼ (T[ρ_{k−j}] − ρ_{k−j})‖²  s.t.  Σⱼ cⱼ = 1
```

History depth `m = 8`, damping `β = 0.1`. Falls back to Picard if the residual rises. Recommended for production isotherms with pressure-continuation warm-start.

```python
from porecdft.solver import anderson_solve

result = anderson_solve(
    rho0, rho_bulk, vext3d, T_K, c1_fn, c1_bulk,
    m=8, beta=0.1, max_iter=5000, tol=1e-6,
    accessibility_mask=access, rho_max=rho_max,
)
# result.rho        — converged density (numpy array, shape = grid)
# result.converged  — bool
# result.iterations — int
```

### Adam / optax (`jax_solve`)

Minimises `Ω[ρ]` by gradient descent. Density reparametrised as `ρ = ρ_bulk · exp(η)` to enforce positivity. Gradient `∇_η Ω` from JAX autodiff. Supports all three `f_exc_mode` options (see below).

```python
from porecdft.solver import jax_solve
import optax

result = jax_solve(
    rho0, rho_bulk, vext3d, T_K, c1_fn, c1_bulk, dV,
    optimizer=optax.adam(2e-3),
    n_steps=5000, tol=1e-8,
    accessibility_mask=access,
    f_exc_mode="endpoint",   # "endpoint" | "rpa" | "quadrature"
)
```

### FIRE2 / NonlinearCG (`fire2_solve`)

Inertial relaxation via `optimistix`. JIT-compiled in JAX; runs on CPU or GPU without code changes.

```python
from porecdft.solver import fire2_solve

result = fire2_solve(
    rho0, rho_bulk, vext3d, T_K, c1_fn, c1_bulk, dV,
    n_steps=3000,
    f_exc_mode="endpoint",   # "endpoint" | "rpa" | "quadrature"
    n_quad=4,                # GL quadrature order (only for mode="quadrature")
)
```

### F_exc mode (`f_exc_mode`) — for Adam and FIRE2

Adam and FIRE2 expose three approximations for the excess free energy integral:

| Mode | Formula | Cost | Notes |
|------|---------|------|-------|
| `"endpoint"` | `F_exc ≈ −k_BT ∫(−c⁽¹⁾[ρ]+c⁽¹⁾_b)ρ dr` | 1 × c⁽¹⁾ | Default; gradient consistent with EL condition |
| `"rpa"` | `F_exc ≈ ½ × endpoint` | 1 × c⁽¹⁾ | Exact only for quadratic F_exc; over-estimates N for FMT/WDA |
| `"quadrature"` | `F_exc ≈ −k_BT Σᵢ wᵢ ∫(−c⁽¹⁾[λᵢρ]+c⁽¹⁾_b)ρ dr` | n × c⁽¹⁾ | 4-pt Gauss–Legendre; loop unrolled at JAX trace time → GPU-ready |

These stem from the exact **adiabatic connection**:
`F_exc[ρ] = −k_BT ∫₀¹ dλ ∫ c⁽¹⁾[λρ] ρ dr`

---

## Quick start

H₂ adsorption isotherm in COF-333-CoCl₂ at T = 298 K.

```python
import numpy as np
from porecdft.eos import H2_PR
from porecdft.functional import LJWDAFunctional
from porecdft.solver import anderson_solve

# ── Load pre-built Vext cache ─────────────────────────────────────────────────
data     = np.load("applications/h2_cof/results/vext_cache_COF-333-CoCl2.npy",
                   allow_pickle=True).item()
vext3d   = data["vext_3d"]
dV       = float(data["dV"])
dx, dy, dz = [float(data["spacings"][i]) for i in range(3)]

T_K = 298.0

# ── Functional ────────────────────────────────────────────────────────────────
wda     = LJWDAFunctional(sigma=2.83, epsilon=59.7, temperature_K=T_K)
rho_max = float(0.45 * 6.0 / (np.pi * wda.d**3))
access  = (vext3d < 50.0 * T_K) & np.isfinite(vext3d)

import jax.numpy as jnp
def c1_fn(rho): return np.asarray(wda.c1(jnp.asarray(rho), dx, dy, dz))

# ── Pressure-continuation isotherm ────────────────────────────────────────────
pressures = [1, 5, 10, 20, 40, 60, 80, 100]   # bar
rho_prev, rho_prev_b = None, None

for P in pressures:
    rho_b  = float(H2_PR.bulk_density(P, T_K))
    c1_b   = float(wda.c1_bulk(rho_b))

    if rho_prev is None:
        # First point: Boltzmann initial guess
        exp  = np.clip(-vext3d / T_K, -50.0, 20.0)
        rho0 = np.where(access, np.clip(rho_b * np.exp(exp), 1e-16, rho_max), 1e-16)
    else:
        # Warm-start: rescale previous solution
        rho0 = np.where(access,
                        np.clip(rho_prev * (rho_b / max(rho_prev_b, 1e-30)), 1e-16, rho_max),
                        1e-16)

    res = anderson_solve(
        rho0, rho_b, vext3d, T_K, c1_fn, c1_b,
        m=8, beta=0.1, max_iter=5000, tol=1e-6,
        accessibility_mask=access, rho_max=rho_max,
    )
    N = float(res.rho.sum() * dV)
    print(f"P = {P:4d} bar   N = {N:.3f} mol/u.c.   conv = {res.converged}")
    rho_prev, rho_prev_b = np.asarray(res.rho).copy(), rho_b
```

---

## Configuration system

All run-time parameters — grid resolution, solver settings, temperature, pressure range, fluid, host, and compute backend — are declared in typed dataclasses (`conf_schema.py`) and stored in YAML files under `conf/`. [Hydra](https://hydra.cc) composes these at startup, validates every field, and saves a complete config snapshot alongside each run's outputs.

### Config tree

```
conf/
  config.yaml          ← top-level composer
  compute/
    cpu_float64.yaml   ← default: CPU, float64, no Warp
    cuda_float32.yaml  ← GPU: CUDA, float32, Warp on cuda:0
    cuda_float64.yaml  ← GPU: CUDA, float64
  vext/
    default.yaml       ← 0.5 Å grid, 20 orientations
    fast.yaml          ← 1.0 Å grid, 10 orientations (for parameter sweeps)
  solver/
    anderson.yaml      ← Anderson mixing (production default)
    fire2.yaml         ← FIRE2 inertial relaxation
  run/
    default.yaml       ← 298 K, 0.1–50 bar
    cryogenic.yaml     ← 77 K, 10⁻⁴–1 bar
  fluid/
    methane.yaml  ethane.yaml  co2.yaml  argon.yaml  n2.yaml
  host/
    dha_tph.yaml  zif8.yaml  dha_cof.yaml
```

### Experiment tracking

Every run is written to an isolated, timestamped directory:

```
outputs/<experiment>/<YYYY-MM-DD>/<HH-MM-SS>/
  .hydra/config.yaml      ← full resolved config (reproducible)
  .hydra/overrides.yaml   ← only what changed from defaults
  vext_methane_298K.npy   ← Vext grid checkpoint (reused on re-run)
  isotherm.npz            ← N(P) arrays + converged density fields
  06_ch4_c2h6_in_dha_tph.png
```

The `.hydra/` folder is Hydra's built-in reproducibility record. Re-running with the same `config.yaml` and `overrides.yaml` reproduces the exact result. The Vext `.npy` checkpoint is reused across runs at the same conditions, so only the solver step is repeated when you change solver parameters.

### CLI usage

Hydra-enabled run scripts (e.g. `run_hydra.py`) accept any config key as a CLI override — no file editing required.

```bash
# defaults (CPU, 298 K, methane, Dha-Tph)
python tutorials/06_ch4_c2h6_in_dha_tph/run_hydra.py

# switch to GPU with one flag
python run_hydra.py compute=cuda_float32

# coarse grid + FIRE2 solver for a quick parameter scan
python run_hydra.py vext=fast solver=fire2 \
                    run.experiment=quick_scan

# cryogenic Ar/ZIF-8
python run_hydra.py run=cryogenic fluid=argon host=zif8 \
                    run.experiment=ar_zif8_77K

# change single values without swapping the whole group
python run_hydra.py run.temperature_K=350 vext.n_orient=200 \
                    solver.tol=0.05 run.experiment=T350_fine

# Hydra multirun — one subfolder per temperature
python run_hydra.py --multirun \
    run.temperature_K=298,350,400 \
    run.experiment=T_sweep
```

### `ComputeConfig` — global backend object

`ComputeConfig` is built from the `compute` config group and passed to every compute function so that one YAML flag switches the entire pipeline — Vext Warp kernels, JAX FFT device, and NumPy array dtype simultaneously.

```python
from porecdft.compute_config import ComputeConfig

# From a Hydra DictConfig
compute = ComputeConfig.from_omegaconf(cfg.compute)
compute.apply_jax_device()   # sets JAX global device once (FMT/aWBII FFTs + solver)
compute.enable_jax_x64()     # optional: float64 in JAX (requires JAX_ENABLE_X64)

# Convenience factories
compute = ComputeConfig.cpu_float64()    # development default
compute = ComputeConfig.cuda_float32()  # production GPU

# Pass to any compute function
vext = build_vext_on_grid(..., compute=compute)
```

| Attribute | Effect |
|-----------|--------|
| `use_warp` | Enable Warp GPU kernels for Vext (LJ, Coulomb, Morse, Boltzmann avg) |
| `warp_device` | `"cpu"` or `"cuda:0"` — passed to every `wp.launch` call |
| `jax_device` | `"cpu"` / `"gpu"` / `"tpu"` — set via `apply_jax_device()` once at startup |
| `dtype` | `"float64"` (default) or `"float32"` — NumPy accumulation + Warp output cast |

---

## Package layout

```
porecdft/
  conf/           Hydra YAML config groups (compute, vext, solver, run, fluid, host)
  conf_schema.py  Typed dataclasses for all config groups + ConfigStore registration
  compute_config.py  ComputeConfig — global backend object (Warp + JAX + dtype)
  io/             CIF, force-field CSV, and partial-charge readers
  structure/      HostAtoms, supercell builder, pore-volume probes, site finders
  forcefield/     Potential ABC + LJ, Morse, Coulomb, quadrupole-EFG,
                  CompositePotential, MLIP adapter
  fluid/          Fluid ABC + CO₂ (EPM2/TraPPE), N₂, CH₄, H₂, generic
  vext/           Fibonacci-sphere orientation sampler + 3D Vext grid builder
                  with on-disk caching
  eos/            Bulk equations of state (see table below)
  functional/     F_exc: FMT-aWBII, WDA-LJ, Wertheim TPT-1, elastic penalty
  solver/         picard.py · anderson.py · jax_solver.py · fire2.py
  diagnostics/    Binding-site probe, Henry constant, isosteric heat
  plotting/       Standardised diagnostic figures
  warp_backend/   Optional NVIDIA Warp GPU kernels (hot paths)

applications/
  alf_co2/        CO₂ in aluminum formate (ALF) — paper figures
  h2_cof/         H₂ in metalated COFs
  eos_compare/    Multi-EOS bulk-density comparison

tutorials/
  01_argon_in_zif8/        run.py · run_hydra.py
  02_methane_in_mfi/
  03_co2_in_dha_cof/
  04_xe_kr_in_zif8/
  05_co2_n2_in_zif8/
  06_ch4_c2h6_in_dha_tph/  run.py · run_hydra.py
```

---

## Equations of state

All EOS subclass `EOSBase` and expose `bulk_density(P_bar, T_K) → float` (molecules/Å³, gas branch).

| Singleton | Class | Physics | `JIT_SAFE` |
|-----------|-------|---------|------------|
| `H2_PR`, `N2_PR`, `CH4_PR` | `PengRobinsonEOS` | Peng-Robinson 1976 | ✓ |
| `CO2_SRK`, `CH4_SRK`, `N2_SRK` | `SRKEOS` | Soave 1972 | ✓ |
| `CO2_SW` | `SpanWagnerCO2EOS` | Reference Helmholtz (Span & Wagner 1996) | ✓ |
| `CO2_PCSAFT`, `N2_PCSAFT`, `CH4_PCSAFT` | `PCSAFTEOS` | Gross & Sadowski 2001 | ✓ |
| `H2O_CPA` | `CPAEOS` | SRK + Wertheim association | NumPy |
| `CO2_SAFT_VR_Mie` | `SAFTVRMieEOS` | Lafitte 2013 (leading-order) | NumPy |
| `H2_FH` | `FeynmanHibbsEOS` | Quantum-corrected H₂ (77 K) | NumPy |
| — | `LJEOS` (MBWR) | Johnson 1993 LJ reference | NumPy |

```python
from porecdft.eos import H2_PR, CO2_SW, H2_FH

print(H2_PR.bulk_density(10.0, 298.0))   # 2.42e-4 molecules/Å³
print(CO2_SW.bulk_density(10.0, 298.0))  # CO₂ near-critical
print(H2_FH.bulk_density(1.0,  77.0))   # quantum-corrected H₂ at 77 K
```

---

## Systems tested

| System | Fluid | Host | T (K) | P range | Functional | Solver | Script |
|--------|-------|------|--------|---------|------------|--------|--------|
| CO₂ / ALF | EPM2 CO₂ (LJ + Coulomb + quadrupole) | Al(HCOO)₃ cubic Im-3m | 278, 298, 318 K | 0–1 bar | FMT-aWBII + WDA + Wertheim TPT-1 + elastic | Anderson | `applications/alf_co2/` |
| N₂ / ALF | TraPPE N₂ | Al(HCOO)₃ | 298 K | 0–1 bar | FMT-aWBII + WDA | Anderson | `applications/alf_co2/` |
| H₂ / COF-301 | LJ H₂ + Morse (Co, Ni, Cu, Zn, Mn) | COF-301 | 77, 298 K | 0–100 bar | WDA-LJ + Morse | Anderson | `applications/h2_cof/` |
| H₂ / COF-322 | LJ H₂ + Morse | COF-322 | 77, 298 K | 0–100 bar | WDA-LJ + Morse | Anderson | `applications/h2_cof/` |
| H₂ / COF-330 | LJ H₂ + Morse | COF-330 | 77, 298 K | 0–100 bar | WDA-LJ + Morse | Anderson | `applications/h2_cof/` |
| H₂ / COF-333-CoCl₂ | LJ H₂ + Morse | COF-333-CoCl₂ | 298 K | 0–500 bar | WDA-LJ + Morse | Anderson / Adam / FIRE2 | `applications/h2_cof/` |
| Ar / ZIF-8 | LJ Ar | ZIF-8 | 87, 273 K | 0–1 bar | FMT-aWBII + WDA | Anderson | `tutorials/01_argon_in_zif8/` |
| CH₄ / MFI zeolite | LJ CH₄ | MFI (silicalite) | 300 K | 0–10 bar | FMT-aWBII + WDA | Anderson | `tutorials/02_methane_in_mfi/` |
| CO₂ / Dha-COF | EPM2 CO₂ | Dha-COF | 298 K | 0–1 bar | FMT-aWBII + WDA + Wertheim | Anderson | `tutorials/03_co2_in_dha_cof/` |
| Xe, Kr / ZIF-8 | LJ Xe, LJ Kr | ZIF-8 | 273 K | 0–1 bar | FMT-aWBII + WDA | Anderson | `tutorials/04_xe_kr_in_zif8/` |
| CO₂ + N₂ / ZIF-8 | EPM2 CO₂ + TraPPE N₂ | ZIF-8 | 298 K | 0–1 bar | FMT-aWBII + WDA | Anderson | `tutorials/05_co2_n2_in_zif8/` |
| CH₄ + C₂H₆ / Dha-Tph COF | LJ CH₄ + LJ C₂H₆ | Dha-Tph | 298 K | 0–20 bar | FMT-aWBII + WDA | Anderson | `tutorials/06_ch4_c2h6_in_dha_tph/` |

**Fluids available** (`fluid/`): CO₂ (EPM2), CO₂ (TraPPE), N₂ (TraPPE), CH₄ (TraPPE), C₂H₆ (TraPPE), H₂ (LJ), Ar (LJ), Kr (LJ), Xe (LJ), generic single-site LJ.

---

## GPU acceleration

Install the GPU extras once:

```bash
uv sync --extra gpu   # CUDA JAX + optax + warp-lang
```

Everything falls back to CPU NumPy / JAX-CPU when the GPU extras are absent — no code changes needed.

### What runs where

| Layer | Technology | Device control |
|-------|-----------|----------------|
| Vext LJ / Coulomb / Morse kernels | NVIDIA Warp | `compute.warp_device` |
| Boltzmann orientation average | NVIDIA Warp | `compute.warp_device` |
| FMT / aWBII FFT convolutions | JAX / XLA (cuFFT) | `compute.jax_device` via `apply_jax_device()` |
| PC-SAFT functional | JAX / XLA | same |
| Anderson / FIRE2 / Adam solver | JAX / XLA | same |

### Warp kernels (`warp_backend/`)

| Kernel | Replaces | Expected speedup |
|--------|---------|-----------------|
| `lj_vext_grid_kernel` | `LJPotential.energy_grid` per-orientation | 10–100× |
| `morse_vext_grid_kernel` | `MorsePotential.energy_grid` per-orientation | 10–100× |
| `smeared_coulomb_grid_kernel` | `CoulombPotential.energy_grid` per-orientation | 10–100× |
| `boltzmann_orient_avg_kernel` | orientation reduction in `vext/builder.py` | 5–20× |
| `rho_bar_sphere_kernel` | `WertheimAssociation._rho_bar_all` | 20–50× |

### Switching to GPU

The entire pipeline — Vext Warp kernels + JAX FFTs + solver — moves to GPU with a single CLI flag:

```bash
python run_hydra.py compute=cuda_float32
```

Or in code:

```python
from porecdft.compute_config import ComputeConfig

compute = ComputeConfig.cuda_float32()
compute.apply_jax_device()   # JAX FFTs now use cuFFT

vext = build_vext_on_grid(..., compute=compute)
```

See the [Configuration system](#configuration-system) section for the full `ComputeConfig` API and YAML layout.

---

## Citation

If you use porecdft in your research, please cite:

> Roy, A.; Tetsassi Feugmo, C. G. *A modular classical density-functional framework for gas adsorption in nanoporous materials.* 2026, in preparation.

---

## License

MIT License. See `LICENSE` for details.

## Contact

Conrard Giresse Tetsassi Feugmo — cgtetsas@uwaterloo.ca  
Department of Chemistry, University of Waterloo

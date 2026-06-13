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
- [Package layout](#package-layout)
- [Equations of state](#equations-of-state)
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

Two production solvers are available in `solver/`:

### Picard fixed-point (`picard_solve`)

```
ρ_{k+1} = ρ_bulk · exp[ c⁽¹⁾[ρ_k] − c⁽¹⁾_bulk − β V_ext ]   (log-density update)
```

Default step `α = 0.02`. Log-density update prevents packing-fraction overshoot. Suitable for dilute systems and warm-starting from a nearby pressure point.

### Anderson mixing (`anderson_solve`)

Solves the constrained least-squares acceleration:

```
min ‖Σⱼ cⱼ (T[ρ_{k−j}] − ρ_{k−j})‖²  s.t.  Σⱼ cⱼ = 1
```

History depth `m = 8`, damping `β = 0.1`. Falls back to Picard (`α = 0.01`) if the residual rises. **Recommended for production isotherms** with pressure-continuation warm-start.

```python
from porecdft.solver import anderson_solve

result = anderson_solve(
    rho0, rho_bulk, vext3d, T_K, c1_fn, c1_bulk,
    m=8, beta=0.1, max_iter=5000, tol=1e-6,
    accessibility_mask=access, rho_max=rho_max,
)
# result.rho       — converged density profile (numpy array)
# result.converged — bool
# result.iterations
```

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

## Package layout

```
porecdft/
  io/             CIF, force-field CSV, and partial-charge readers
  structure/      HostAtoms, supercell builder, pore-volume probes, site finders
  forcefield/     Potential ABC + LJ, Morse, Coulomb, quadrupole-EFG,
                  CompositePotential, MLIP adapter
  fluid/          Fluid ABC + CO₂ (EPM2/TraPPE), N₂, CH₄, H₂, generic
  vext/           Fibonacci-sphere orientation sampler + 3D Vext grid builder
                  with on-disk caching
  eos/            Bulk equations of state (see table below)
  functional/     F_exc: FMT-aWBII, WDA-LJ, Wertheim TPT-1, elastic penalty
  solver/         picard.py · anderson.py
  diagnostics/    Binding-site probe, Henry constant, isosteric heat
  plotting/       Standardised diagnostic figures
  warp_backend/   Optional NVIDIA Warp GPU kernels (hot paths)

applications/
  alf_co2/        CO₂ in aluminum formate (ALF) — paper figures
  h2_cof/         H₂ in metalated COFs
  eos_compare/    Multi-EOS bulk-density comparison
  tutorials/      Step-by-step notebooks
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

## GPU acceleration

Optional NVIDIA Warp kernels for the three largest 3D hot paths. Install with `uv sync --extra gpu`. Falls back to CPU NumPy when `warp-lang` is absent.

| Kernel | Replaces | Expected speedup |
|--------|---------|-----------------|
| `lj_vext_grid_kernel` | `LJPotential.energy_grid` per-orientation | 10–100× |
| `morse_vext_grid_kernel` | `MorsePotential.energy_grid` per-orientation | 10–100× |
| `smeared_coulomb_grid_kernel` | `CoulombPotential.energy_grid` per-orientation | 10–100× |
| `boltzmann_orient_avg_kernel` | orientation reduction in `vext/builder.py` | 5–20× |
| `rho_bar_sphere_kernel` | `WertheimAssociation._rho_bar_all` | 20–50× |

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

"""Run Adam isotherms without Picard polish for 3k and 20k steps.

Compares four configurations over the full COF-333 pressure range:
  1. Anderson          (cached → isotherm_h2_cof333_anderson.npy)
  2. Adam + polish     (cached → isotherm_h2_cof333_adam.npy)
  3. Adam  3k no-pol   (new)
  4. Adam 20k no-pol   (new)

Uses pressure-continuation warm-start so each point is fast.
Saves results and produces the comparison figure.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARENT    = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

from porecdft.eos import H2_PR
from porecdft.functional import LJWDAFunctional
from porecdft.solver import jax_solve

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

T_K        = 298.0
SIGMA_H2   = 2.83
EPSILON_H2 = 59.7
ADAM_LR    = 2e-3

P_ISO = np.array([1, 5, 10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 450, 500],
                 dtype=float)

data     = np.load(RESULTS_DIR / "vext_cache_COF-333-CoCl2.npy", allow_pickle=True).item()
vext3d   = data["vext_3d"]
dV       = float(data["dV"])
spacings = data["spacings"]
dx, dy, dz = float(spacings[0]), float(spacings[1]), float(spacings[2])

wda     = LJWDAFunctional(sigma=SIGMA_H2, epsilon=EPSILON_H2, temperature_K=T_K)
rho_max = float(0.45 * 6.0 / (np.pi * wda.d**3))
access  = (vext3d < 50.0 * T_K) & np.isfinite(vext3d)

# Pre-warm weight cache outside JIT
_ = wda._get_weights(vext3d.shape, dx, dy, dz)

def c1_fn(rho): return wda.c1(jnp.asarray(rho), dx, dy, dz)

def boltzmann_init(rho_b):
    exp = np.clip(-vext3d / T_K, -50.0, 20.0)
    return np.where(access, np.clip(rho_b * np.exp(exp), 1e-16, rho_max), 1e-16)

def run_adam_isotherm(n_steps: int, label: str) -> np.ndarray:
    N_arr, rho_prev, rho_prev_b = [], None, None
    t0 = time.perf_counter()
    for P in P_ISO:
        rho_b = float(H2_PR.bulk_density(P, T_K))
        c1_b  = float(wda.c1_bulk(rho_b))
        if rho_prev is not None:
            rho0 = np.where(access,
                            np.clip(rho_prev * (rho_b / max(rho_prev_b, 1e-30)),
                                    1e-16, rho_max), 1e-16)
        else:
            rho0 = boltzmann_init(rho_b)
        res = jax_solve(rho0, rho_b, vext3d, T_K, c1_fn, c1_b, dV,
                        optimizer=optax.adam(ADAM_LR),
                        n_steps=n_steps, tol=1e-8,
                        accessibility_mask=access,
                        f_exc_mode="endpoint")
        rho_prev, rho_prev_b = np.asarray(res.rho).copy(), rho_b
        N = float(res.rho.sum() * dV)
        N_arr.append(N)
        print(f"  [{label}] P={P:5.0f} bar  N={N:.2f}  iters={res.iterations}  conv={res.converged}",
              flush=True)
    total = time.perf_counter() - t0
    print(f"  [{label}] Total: {total:.1f} s\n")
    return np.array(N_arr)

print("Running Adam 3k (no polish) isotherm...")
N_adam3k = run_adam_isotherm(3000, "Adam-3k")
np.save(RESULTS_DIR / "isotherm_h2_cof333_adam3k_nopol.npy", N_adam3k)

print("Running Adam 20k (no polish) isotherm...")
N_adam20k = run_adam_isotherm(20000, "Adam-20k")
np.save(RESULTS_DIR / "isotherm_h2_cof333_adam20k_nopol.npy", N_adam20k)

# Load cached reference isotherms
N_anderson  = np.load(RESULTS_DIR / "isotherm_h2_cof333_anderson.npy")
N_adam_pol  = np.load(RESULTS_DIR / "isotherm_h2_cof333_adam.npy")

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
fig.suptitle(
    r"H$_2$ adsorption isotherm in COF-333-CoCl$_2$"
    "\nAdam solver variants vs Anderson reference, T = 298 K",
    fontsize=13, fontweight="bold"
)

# ── Left: full isotherm ───────────────────────────────────────────────────────
ax = axes[0]
ax.plot(P_ISO, N_anderson,  "k-",  lw=2.2, label="Anderson (reference)",  zorder=5)
ax.plot(P_ISO, N_adam_pol,  "b--", lw=1.8, label="Adam 3k + Picard polish", zorder=4)
ax.plot(P_ISO, N_adam3k,   "r:",  lw=1.8, label="Adam 3k (no polish)",    zorder=3)
ax.plot(P_ISO, N_adam20k,  "g-.", lw=1.8, label="Adam 20k (no polish)",   zorder=3)

ax.set_xlabel(r"$P$ / bar", fontsize=12)
ax.set_ylabel(r"$N_\mathrm{ads}$ / mol u.c.$^{-1}$", fontsize=12)
ax.set_xlim(0, 520)
ax.set_ylim(0)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(alpha=0.25, ls=":")
ax.set_title("Full isotherm (0–500 bar)", fontsize=11)

# Shade the gap between Anderson and Adam no-polish
ax.fill_between(P_ISO, N_adam20k, N_anderson, alpha=0.12, color="red",
                label=r"Adam plateau gap ($\Delta N$)")

# ── Right: zoomed residual panel ──────────────────────────────────────────────
ax2 = axes[1]
delta3k  = N_anderson - N_adam3k
delta20k = N_anderson - N_adam20k
delta_pol = N_anderson - N_adam_pol

ax2.bar(P_ISO - 12, delta3k,  width=22, label="Adam 3k gap",    color="#d62728", alpha=0.8)
ax2.bar(P_ISO + 12, delta20k, width=22, label="Adam 20k gap",   color="#2ca02c", alpha=0.8)
ax2.plot(P_ISO, delta_pol, "bs", ms=6, label="Adam+polish gap", zorder=5)
ax2.axhline(0, color="black", lw=1.2)

ax2.set_xlabel(r"$P$ / bar", fontsize=12)
ax2.set_ylabel(r"$\Delta N = N_\mathrm{Anderson} - N_\mathrm{Adam}$ / mol u.c.$^{-1}$",
               fontsize=11)
ax2.set_xlim(0, 520)
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(alpha=0.25, ls=":")
ax2.set_title("Residual gap vs Anderson", fontsize=11)

out = FIGURES_DIR / "isotherm_adam_variants.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {out}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'P (bar)':>8}  {'Anderson':>10}  {'Adam-3k':>9}  {'Adam-20k':>9}"
      f"  {'Adam+pol':>9}  {'Δ3k%':>7}  {'Δ20k%':>7}")
print("-"*70)
for i, P in enumerate(P_ISO):
    d3k  = 100*(N_anderson[i]-N_adam3k[i])/max(N_anderson[i],1e-6)
    d20k = 100*(N_anderson[i]-N_adam20k[i])/max(N_anderson[i],1e-6)
    print(f"  {P:6.0f}    {N_anderson[i]:9.3f}  {N_adam3k[i]:9.3f}  "
          f"{N_adam20k[i]:9.3f}  {N_adam_pol[i]:9.3f}  {d3k:+6.2f}%  {d20k:+6.2f}%")

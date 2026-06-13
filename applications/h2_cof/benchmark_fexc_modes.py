"""Benchmark: compare F_exc approximations for H₂/COF-333 at T=298 K, P=10 bar.

Tests all three excess free-energy modes in both the Adam (optax) and FIRE2
(NonlinearCG) solvers against Anderson as the reference:

  Modes
  -----
  endpoint   : F_ex ≈ ∫(-c¹[ρ] + c¹_bulk)ρ dV             (λ=1, current default)
  rpa        : F_ex ≈ ½ ∫(-c¹[ρ] + c¹_bulk)ρ dV            (Bao 2025 suggestion)
  quadrature : F_ex ≈ ∑_i w_i ∫(-c¹[λ_iρ] + c¹_bulk)ρ dV  (4-pt GL, this work)

  Solvers
  -------
  Anderson   : reference fixed-point solver (not affected by f_exc_mode)
  Adam       : optax Adam (lr=2e-3), each mode independently
  FIRE2      : optimistix NonlinearCG, each mode independently

Outputs
-------
  results printed to stdout (N_ads, Ω, iters, wall-clock, peak RSS)
  benchmark_fexc_modes.png — bar chart + convergence loss curves
"""
from __future__ import annotations

import os
import sys
import time
import tracemalloc
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

from porecdft.eos import H2_PR
from porecdft.functional import LJWDAFunctional
from porecdft.solver import anderson_solve, jax_solve, fire2_solve

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
T_K        = 298.0
P_BAR      = 10.0
KCAL_TO_K  = 503.228
SIGMA_H2   = 2.83
EPSILON_H2 = 59.7
RCUT_H2    = 5.0 * SIGMA_H2

# Adam / FIRE2 hyperparameters
ADAM_LR    = 2e-3
ADAM_STEPS = 3000
FIRE2_STEPS = 3000

# ── load cached Vext ─────────────────────────────────────────────────────────
VEXT_CACHE = RESULTS_DIR / "vext_cache_COF-333-CoCl2.npy"
if not VEXT_CACHE.exists():
    VEXT_CACHE = RESULTS_DIR / "vext_cache_COF-333-CoCl2_LJonly.npy"

if not VEXT_CACHE.exists():
    raise FileNotFoundError(
        f"Vext cache not found at {RESULTS_DIR}.\n"
        "Run make_h2_isotherm_cdft.py first to build the cache."
    )

data   = np.load(VEXT_CACHE, allow_pickle=True).item()
vext3d = data["vext_3d"]
dV     = float(data["dV"])
n_pts  = tuple(data["n_pts"])
print(f"Loaded Vext: {VEXT_CACHE.name}  grid {n_pts}  dV={dV:.4f} Å³")

# ── WDA functional ───────────────────────────────────────────────────────────
wda      = LJWDAFunctional(sigma=SIGMA_H2, epsilon=EPSILON_H2, temperature_K=T_K)
rho_bulk = float(H2_PR.bulk_density(P_BAR, T_K))
rho_max  = float(0.45 * 6.0 / (np.pi * wda.d**3))
c1_bulk  = float(wda.c1_bulk(rho_bulk))
access   = (vext3d < 50.0 * T_K) & np.isfinite(vext3d)

# Grid spacings (Å)
n_pts = vext3d.shape
spacings = data.get("spacings", None)
if spacings is not None:
    dx, dy, dz = float(spacings[0]), float(spacings[1]), float(spacings[2])
else:
    dx = dy = dz = dV ** (1.0 / 3.0)

print(f"T={T_K} K  P={P_BAR} bar  rho_bulk={rho_bulk:.4e} Å⁻³  rho_max={rho_max:.4e}")
print(f"BH d={wda.d:.4f} Å  dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} Å")

def c1_fn(rho):
    return wda.c1(jnp.asarray(rho), dx, dy, dz)

def boltzmann_init():
    exp = np.clip(-vext3d / T_K, -50.0, 20.0)
    return np.where(access, np.clip(rho_bulk * np.exp(exp), 1e-16, rho_max), 1e-16)

# ── Anderson reference ───────────────────────────────────────────────────────
print("\n── Anderson (reference) ──")
rho0 = boltzmann_init()
t0   = time.perf_counter()
tracemalloc.start()
res_and = anderson_solve(
    rho0, rho_bulk, vext3d, T_K, c1_fn, c1_bulk,
    m=8, beta=0.1, tol=1e-6, max_iter=500,
    accessibility_mask=access, rho_max=rho_max,
)
mem_and = tracemalloc.get_traced_memory()[1] / 1e6
tracemalloc.stop()
t_and   = time.perf_counter() - t0
N_and   = float(res_and.rho.sum() * dV)
print(f"  N={N_and:.4f} Å⁻³·Å³  conv={res_and.converged}  "
      f"iters={res_and.iterations}  t={t_and:.2f}s  mem={mem_and:.1f} MB")

# ── Run all (solver, mode) combos ────────────────────────────────────────────
MODES   = ["endpoint", "rpa", "quadrature"]
SOLVERS = ["Adam", "FIRE2"]
RESULTS: dict[str, dict] = {}

import optax

for solver_name in SOLVERS:
    for mode in MODES:
        key = f"{solver_name}_{mode}"
        print(f"\n── {solver_name} / {mode} ──")
        rho0 = boltzmann_init()
        t0   = time.perf_counter()
        tracemalloc.start()

        if solver_name == "Adam":
            res = jax_solve(
                rho0, rho_bulk, vext3d, T_K, c1_fn, c1_bulk, dV,
                optimizer=optax.adam(ADAM_LR),
                n_steps=ADAM_STEPS, tol=1e-6,
                accessibility_mask=access,
                f_exc_mode=mode,
                n_quad=4,
            )
            omega_hist = res.omega_history
            n_iters    = res.iterations
            converged  = res.converged
        else:  # FIRE2
            res = fire2_solve(
                rho0, rho_bulk, vext3d, T_K, c1_fn, c1_bulk, dV,
                max_steps=FIRE2_STEPS, rtol=1e-7, atol=1e-9,
                accessibility_mask=access,
                collect_history=True,
                collect_max_steps=FIRE2_STEPS,
                f_exc_mode=mode,
                n_quad=4,
            )
            omega_hist = None   # FIRE2 history path tracks grad norm not Ω
            n_iters    = res.iterations
            converged  = res.converged

        mem_peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        t_elapsed = time.perf_counter() - t0
        N_val = float(res.rho.sum() * dV)
        print(f"  N={N_val:.4f}  conv={converged}  "
              f"iters={n_iters}  t={t_elapsed:.2f}s  mem={mem_peak:.1f} MB")

        RESULTS[key] = {
            "N": N_val, "converged": converged, "iters": n_iters,
            "t": t_elapsed, "mem": mem_peak,
            "omega_hist": omega_hist,
            "error_hist": res.error_history if hasattr(res, "error_history") else None,
        }

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig.suptitle(
    f"F_exc mode comparison — H₂/COF-333-CoCl₂, T={T_K:.0f} K, P={P_BAR:.0f} bar",
    fontsize=12, fontweight="bold"
)

# ── Left: N_ads bar chart ──
ax = axes[0]
colors    = {"endpoint": "#1f77b4", "rpa": "#d62728", "quadrature": "#2ca02c"}
hatches   = {"Adam": "//", "FIRE2": ""}
bar_width = 0.28
x_pos     = np.arange(len(MODES))

for j, solver_name in enumerate(SOLVERS):
    Ns = [RESULTS[f"{solver_name}_{m}"]["N"] for m in MODES]
    bars = ax.bar(
        x_pos + (j - 0.5) * bar_width, Ns,
        bar_width, label=solver_name,
        color=[colors[m] for m in MODES],
        hatch=hatches[solver_name], edgecolor="black", linewidth=0.8,
        alpha=0.85,
    )

ax.axhline(N_and, color="black", ls="--", lw=1.8, label=f"Anderson ref ({N_and:.3f})")
ax.set_xticks(x_pos)
ax.set_xticklabels(["endpoint\n(λ=1)", "rpa\n(½ factor)", "quadrature\n(4-pt GL)"],
                   fontsize=10)
ax.set_ylabel("N_ads (Å⁻³·Å³)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, ls=":")
ax.set_title("Adsorption vs mode", fontsize=10)

# ── Right: resource table (time + memory) ──
ax2 = axes[1]
ax2.axis("off")
rows = [["Solver", "Mode", "N_ads", "Δ_N%", "iters", "time (s)", "mem (MB)"]]
for solver_name in SOLVERS:
    for mode in MODES:
        r = RESULTS[f"{solver_name}_{mode}"]
        delta_pct = 100.0 * (r["N"] - N_and) / max(N_and, 1e-10)
        rows.append([
            solver_name, mode,
            f"{r['N']:.4f}", f"{delta_pct:+.2f}%",
            str(r["iters"]), f"{r['t']:.2f}", f"{r['mem']:.1f}",
        ])
# Add Anderson reference row
rows.append(["Anderson", "ref",
             f"{N_and:.4f}", "0.00%",
             str(res_and.iterations), f"{t_and:.2f}", f"{mem_and:.1f}"])

col_labels = rows[0]
cell_text  = rows[1:]
tbl = ax2.table(
    cellText=cell_text, colLabels=col_labels,
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.5)
ax2.set_title("Resource summary", fontsize=10)

out = FIGURES_DIR / "benchmark_fexc_modes.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {out}")

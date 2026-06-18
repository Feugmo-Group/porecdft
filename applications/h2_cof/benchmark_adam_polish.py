"""Quick benchmark: Adam convergence variants for endpoint mode only.

Measures 4 Adam configurations at T=298K, P=10 bar, endpoint f_exc_mode:
  1. Adam  3 000 steps          (already known: N=16.63)
  2. Adam  9 000 steps  (3×)
  3. Adam  3 000 steps + Picard polish (Anderson finalize, warm-start)
  4. Anderson reference          (already known: N=16.94)

Prints results as a compact table and saves them to benchmark_adam_polish.npy.
"""
from __future__ import annotations
import sys, time, tracemalloc
from pathlib import Path

import numpy as np

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
from porecdft.solver import anderson_solve, jax_solve

RESULTS_DIR = Path(__file__).parent / "results"
T_K      = 298.0
P_BAR    = 10.0
SIGMA_H2 = 2.83
EPSILON_H2 = 59.7
ADAM_LR  = 2e-3

data     = np.load(RESULTS_DIR / "vext_cache_COF-333-CoCl2.npy", allow_pickle=True).item()
vext3d   = data["vext_3d"]
dV       = float(data["dV"])
spacings = data["spacings"]
dx, dy, dz = float(spacings[0]), float(spacings[1]), float(spacings[2])

wda      = LJWDAFunctional(sigma=SIGMA_H2, epsilon=EPSILON_H2, temperature_K=T_K)
rho_bulk = float(H2_PR.bulk_density(P_BAR, T_K))
rho_max  = float(0.45 * 6.0 / (np.pi * wda.d**3))
c1_bulk  = float(wda.c1_bulk(rho_bulk))
access   = (vext3d < 50.0 * T_K) & np.isfinite(vext3d)

def c1_fn(rho):     return wda.c1(jnp.asarray(rho), dx, dy, dz)
def c1_fn_np(rho):  return np.asarray(wda.c1(jnp.asarray(rho), dx, dy, dz))

# Pre-warm _get_weights cache with concrete (non-traced) JAX arrays so that
# subsequent jax.jit calls don't cache leaked tracers across compilations.
_ = wda._get_weights(vext3d.shape, dx, dy, dz)

def boltzmann_init():
    exp = np.clip(-vext3d / T_K, -50.0, 20.0)
    return np.where(access, np.clip(rho_bulk * np.exp(exp), 1e-16, rho_max), 1e-16)

def run_adam(n_steps, label):
    rho0 = boltzmann_init()
    tracemalloc.start()
    t0 = time.perf_counter()
    res = jax_solve(rho0, rho_bulk, vext3d, T_K, c1_fn, c1_bulk, dV,
                    optimizer=optax.adam(ADAM_LR), n_steps=n_steps, tol=1e-8,
                    accessibility_mask=access, f_exc_mode="endpoint")
    mem = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    t = time.perf_counter() - t0
    N = float(res.rho.sum() * dV)
    print(f"  {label:35s}  N={N:.4f}  conv={res.converged}  "
          f"iters={res.iterations}  t={t:.1f}s  mem={mem:.1f}MB")
    return res.rho.copy(), N, res.iterations, t, mem, res.converged

def run_polish(rho_adam, label_prefix):
    rho0 = np.asarray(rho_adam)
    tracemalloc.start()
    t0 = time.perf_counter()
    res = anderson_solve(rho0, rho_bulk, vext3d, T_K, c1_fn_np, c1_bulk,
                         m=8, beta=0.1, max_iter=5000, tol=1e-6,
                         accessibility_mask=access, rho_max=rho_max,
                         safeguard_alpha=0.01, picard_warmup=50)
    mem = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    t = time.perf_counter() - t0
    N = float(res.rho.sum() * dV)
    label = f"{label_prefix} + Picard polish"
    print(f"  {label:35s}  N={N:.4f}  conv={res.converged}  "
          f"iters={res.iterations}  t={t:.1f}s  mem={mem:.1f}MB")
    return N, res.iterations, t, mem, res.converged

print(f"\nAdam convergence variants  T={T_K}K  P={P_BAR}bar  (endpoint mode)")
print(f"  rho_bulk={rho_bulk:.4e}  rho_max={rho_max:.4e}  BH d={wda.d:.4f} Å")
print()

# 1. Adam 3 000 steps
rho_3k, N_3k, it_3k, t_3k, mem_3k, conv_3k = run_adam(3000,  "Adam 3 000 steps")
# 2. Adam 9 000 steps
rho_9k, N_9k, it_9k, t_9k, mem_9k, conv_9k = run_adam(9000,  "Adam 9 000 steps (3×)")
# 3. Adam 3 000 + polish
N_p, it_p, t_p, mem_p, conv_p = run_polish(rho_3k, "Adam 3 000 steps")

out = RESULTS_DIR / "benchmark_adam_polish.npy"
np.save(out, {
    "adam_3k":     (N_3k,  it_3k,  t_3k,  mem_3k,  conv_3k),
    "adam_9k":     (N_9k,  it_9k,  t_9k,  mem_9k,  conv_9k),
    "adam_polish": (N_p,   it_p,   t_p+t_3k,  mem_p, conv_p),
})
print(f"\nSaved → {out}")

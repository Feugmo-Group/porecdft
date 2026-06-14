"""H₂/COF-333-CoCl₂ isotherm: Anderson vs Adam (endpoint) vs Adam (GL-8pt quadrature).

Compares three solution strategies for the cDFT grand potential at T=298 K:

  Anderson    — log-density Walker-Ni fixed-point iteration (reference)
  Adam-EP     — Adam on Ω with endpoint F_ex approximation (λ=1)
  Adam-GL8    — Adam on Ω with 8-pt Gauss-Legendre F_ex quadrature

Physics (Songhao Wu, 2025):
  Exact:     F_ex[ρ] = -k_BT ∫₀¹ dλ ∫ c¹[λρ] ρ dr
  Endpoint:  F_ex ≈ -k_BT ∫ (c¹[ρ] - c¹_b) ρ dr          (λ→1)
  GL-8:      F_ex ≈ -k_BT Σᵢ wᵢ ∫ (c¹[λᵢρ] - c¹_b) ρ dr  (Gauss-Legendre)

Output: figures/h2_fexc_isotherm_comparison.png
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[3]
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
from porecdft.solver import anderson_solve, jax_solve, OPTAX_AVAILABLE, EQX_AVAILABLE

RESULTS_DIR = Path(__file__).parents[1] / "results"
FIGURES_DIR = Path(__file__).parents[1] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

T_K        = 298.0
SIGMA_H2   = 2.83
EPSILON_H2 = 59.7
P_ISO = np.array([1, 5, 10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 450, 500], dtype=float)

ADAM_STEPS = 6000
ADAM_LR    = 2e-3
N_QUAD     = 8

# GCMC reference from Pramudya & Mendoza-Cortes 2016 (Fig. 3, COF-333-CoCl2, 298 K)
GCMC_P   = np.array([1, 5, 10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500])
GCMC_wt  = np.array([0.14, 0.42, 0.63, 0.97, 1.49, 1.87, 2.16, 2.40, 2.58, 2.79, 3.07, 3.27, 3.43, 3.62, 3.72])
H2_MW = 2.016e-3   # kg/mol
NA    = 6.022140857e23


def wt_to_mol(wt_pct, M_host_kg=None):
    """wt% H2 → mol/kg_host (not used here — we plot mol/u.c.)"""
    return wt_pct / (100.0 * H2_MW)


def load_vext():
    cache = RESULTS_DIR / "vext_cache_COF-333-CoCl2.npy"
    if not cache.exists():
        raise FileNotFoundError(f"Run make_h2_isotherm_cdft.py first to build {cache}")
    raw  = np.load(cache, allow_pickle=True)
    data = raw.item() if raw.ndim == 0 else dict(raw)
    vext3d   = np.array(data.get("vext_3d", data.get("vext")), dtype=float)
    dV       = float(data["dV"])
    spacings = data.get("spacings", None)
    if spacings is not None:
        dx, dy, dz = float(spacings[0]), float(spacings[1]), float(spacings[2])
    else:
        dx = dy = dz = dV ** (1.0/3.0)
    return vext3d, dV, dx, dy, dz


def boltzmann_init(vext3d, rho_b, rho_max, access):
    exp = np.clip(-vext3d / T_K, -50.0, 20.0)
    return np.where(access, np.clip(rho_b * np.exp(exp), 1e-16, rho_max), 1e-16)


def run_anderson(vext3d, dV, wda, access, rho_max, dx, dy, dz):
    # Load from cache if available
    iso_cache = RESULTS_DIR / "isotherm_h2_cof333_298K.npz"
    if iso_cache.exists():
        data = dict(np.load(iso_cache))
        N_cached = np.interp(P_ISO, np.array(data["P"], dtype=float),
                             np.array(data["N_abs"], dtype=float))
        print("  Anderson loaded from cache", flush=True)
        return N_cached, np.zeros(len(P_ISO))

    c1_fn = lambda rho: np.asarray(wda.c1(jnp.asarray(rho), dx, dy, dz))
    N_arr, T_arr, rho_prev, rho_prev_b = [], [], None, None
    for P in P_ISO:
        rho_b = H2_PR.bulk_density(P, T_K)
        c1_b  = wda.c1_bulk(rho_b)
        rho0  = (np.where(access, np.clip(rho_prev*(rho_b/rho_prev_b), 1e-16, rho_max), 1e-16)
                 if rho_prev is not None else boltzmann_init(vext3d, rho_b, rho_max, access))
        t0 = time.perf_counter()
        res = anderson_solve(rho0, rho_b, vext3d, T_K, c1_fn, c1_b,
                             m=8, beta=0.1, max_iter=8000, tol=1e-5,
                             accessibility_mask=access, safeguard_alpha=0.01,
                             picard_warmup=100)
        T_arr.append(time.perf_counter() - t0)
        rho_prev, rho_prev_b = res.rho.copy(), rho_b
        N = float(res.rho.sum() * dV)
        N_arr.append(N)
        print(f"  Anderson  P={P:5.0f} bar  N={N:.2f}  conv={res.converged}  "
              f"t={T_arr[-1]:.1f}s", flush=True)
    return np.array(N_arr), np.array(T_arr)


def run_adam(vext3d, dV, wda, access, rho_max, dx, dy, dz, mode, n_quad=4):
    import optax
    # Pre-warm the weight cache outside JIT so the cache holds concrete arrays,
    # not DynamicJaxprTracers (which would leak across pressure-point iterations).
    wda._get_weights(vext3d.shape, dx, dy, dz)
    c1_jax = lambda rho: wda.c1(rho, dx, dy, dz)
    N_arr, T_arr, rho_prev, rho_prev_b = [], [], None, None
    for P in P_ISO:
        rho_b = H2_PR.bulk_density(P, T_K)
        c1_b  = float(wda.c1_bulk(rho_b))
        rho0  = (np.where(access, np.clip(rho_prev*(rho_b/rho_prev_b), 1e-16, rho_max), 1e-16)
                 if rho_prev is not None else boltzmann_init(vext3d, rho_b, rho_max, access))
        t0 = time.perf_counter()
        res = jax_solve(rho0, rho_b, vext3d, T_K, c1_jax, c1_b, dV=dV,
                        optimizer=optax.adam(ADAM_LR), n_steps=ADAM_STEPS, tol=1e-8,
                        accessibility_mask=access, f_exc_mode=mode, n_quad=n_quad)
        T_arr.append(time.perf_counter() - t0)
        rho_prev, rho_prev_b = np.asarray(res.rho).copy(), rho_b
        N = float(res.rho.sum() * dV)
        N_arr.append(N)
        print(f"  Adam-{mode:8s}  P={P:5.0f} bar  N={N:.2f}  "
              f"conv={res.converged}  iters={res.iterations}  t={T_arr[-1]:.1f}s",
              flush=True)
    return np.array(N_arr), np.array(T_arr)


def mol_uc_to_wt(N_arr, V_cell_A3, n_uc=1):
    """mol/u.c. → wt% H2, using unit-cell mass from COF-333 CIF."""
    # COF-333-CoCl2: ~5000 Da / u.c. (approximate; from Pramudya SI)
    M_host_g = 5000.0 * n_uc
    m_H2_g   = N_arr * H2_MW * 1e3 * NA / NA  # N mol × 2.016 g/mol
    return 100.0 * m_H2_g / (m_H2_g + M_host_g)


def main():
    if not (OPTAX_AVAILABLE and EQX_AVAILABLE):
        raise ImportError("Install optax and equinox: pip install optax equinox")

    vext3d, dV, dx, dy, dz = load_vext()
    wda    = LJWDAFunctional(sigma=SIGMA_H2, epsilon=EPSILON_H2, temperature_K=T_K)
    rho_b0 = H2_PR.bulk_density(1.0, T_K)
    c1_b0  = float(wda.c1_bulk(rho_b0))
    rho_max = float(np.exp(-c1_b0) * rho_b0 * 10)
    access  = (vext3d < 50.0 * T_K)

    print("\n── Anderson (reference) ──", flush=True)
    N_and, T_and = run_anderson(vext3d, dV, wda, access, rho_max, dx, dy, dz)

    print(f"\n── Adam / endpoint (lr={ADAM_LR}, n_steps={ADAM_STEPS}) ──", flush=True)
    N_ep, T_ep = run_adam(vext3d, dV, wda, access, rho_max, dx, dy, dz,
                          mode="endpoint")

    print(f"\n── Adam / GL-{N_QUAD}pt quadrature ──", flush=True)
    N_gl, T_gl = run_adam(vext3d, dV, wda, access, rho_max, dx, dy, dz,
                          mode="quadrature", n_quad=N_QUAD)

    # ── Save results ──────────────────────────────────────────────────────
    np.savez(RESULTS_DIR / "fexc_isotherm_comparison.npz",
             P=P_ISO, N_and=N_and, N_ep=N_ep, N_gl=N_gl,
             T_and=T_and, T_ep=T_ep, T_gl=T_gl)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.suptitle(
        r"H$_2$/COF-333-CoCl$_2$, $T=298\,$K: $F_{\rm ex}$ approximation comparison",
        fontsize=12, fontweight="bold"
    )

    COLORS = {"Anderson": "#444444", "Adam-EP": "#1f77b4", "Adam-GL8": "#d62728"}
    LS     = {"Anderson": "--",       "Adam-EP": "-",       "Adam-GL8": "-"}
    MS     = {"Anderson": "^",        "Adam-EP": "o",       "Adam-GL8": "s"}

    t_and_tot = T_and.sum()
    t_ep_tot  = T_ep.sum()
    t_gl_tot  = T_gl.sum()
    labels = {
        "Anderson": f"Anderson (ref., {t_and_tot:.0f} s)",
        "Adam-EP":  f"Adam, endpoint (${ADAM_STEPS//1000}$k steps, {t_ep_tot:.0f} s)",
        "Adam-GL8": f"Adam, GL-{N_QUAD}pt quad. ({t_gl_tot:.0f} s)",
    }

    ax = axes[0]
    ax.scatter(GCMC_P, GCMC_wt, marker="*", s=80, color="#2ca02c",
               zorder=5, label="GCMC (Pramudya 2016)")
    for key, N_arr in [("Anderson", N_and), ("Adam-EP", N_ep), ("Adam-GL8", N_gl)]:
        wt = mol_uc_to_wt(N_arr)
        ax.plot(P_ISO, wt, color=COLORS[key], ls=LS[key], marker=MS[key],
                ms=5, label=labels[key])
    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("H$_2$ uptake (wt%)")
    ax.set_xlim(0, 510)
    ax.legend(fontsize=8)
    ax.set_title("Isotherm comparison")

    ax = axes[1]
    err_ep = 100.0 * (N_ep - N_and) / (N_and + 1e-10)
    err_gl = 100.0 * (N_gl - N_and) / (N_and + 1e-10)
    ax.axhline(0, color="#444444", ls="--", lw=1)
    ax.plot(P_ISO, err_ep, color=COLORS["Adam-EP"], marker="o", ms=5, label=labels["Adam-EP"])
    ax.plot(P_ISO, err_gl, color=COLORS["Adam-GL8"], marker="s", ms=5, label=labels["Adam-GL8"])
    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel(r"$\Delta N / N_{\rm Anderson}$ (%)")
    ax.set_xlim(0, 510)
    ax.legend(fontsize=8)
    ax.set_title("Relative deviation from Anderson reference")

    out = FIGURES_DIR / "h2_fexc_isotherm_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}", flush=True)

    # Print summary table
    print(f"\n{'P':>6}  {'N_And':>7}  {'N_EP':>7}  {'ΔEP%':>6}  "
          f"{'N_GL8':>7}  {'ΔGL%':>6}")
    for i, P in enumerate(P_ISO):
        print(f"{P:6.0f}  {N_and[i]:7.3f}  {N_ep[i]:7.3f}  {err_ep[i]:+6.1f}  "
              f"{N_gl[i]:7.3f}  {err_gl[i]:+6.1f}")


if __name__ == "__main__":
    main()

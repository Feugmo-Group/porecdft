"""Comparing excess free energy functional obtained from the endpoint and quadrature method for H₂/COF-333-CoCl₂ — Morse+LJ Vext + aWBII+WDA c₁.

Runs four porecdft solvers on the full self-consistent grand-potential problem
at T=298 K (same physics as make_h2_isotherm_cdft.py):

The exact expression of Fex should be calculated through adiabatic connection
F_ex = -k_B T ∫ dλ ∫ dr [ c¹(λρ) - c¹_bulk ]

quadrature approximation:
F_ex ≈ -k_B T Σ_{i=1}^N w_i ∫ dr [ c¹(λ_i ρ) - c¹_bulk ]

end point approximation:
F_ex ≈ -k_B T ∫ dr [ c¹(ρ) - c¹_bulk ]

Outputs (two separate files):
  h2_solver_comparison_isotherm.png — N_ads vs P for all four solvers
  h2_solver_comparison_loss.png     — convergence history at P=10 bar
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

ROOT = str(_REPO_ROOT)

import jax
# jax.config.update("jax_enable_x64", False)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pymatgen.core import Structure

from porecdft.structure.host import HostAtoms
from porecdft.structure.supercell import build_supercell
from porecdft.eos import H2_PR
from porecdft.functional import LJWDAFunctional
from porecdft.solver import (
    picard_solve, anderson_solve,
    jax_solve, OPTAX_AVAILABLE, EQX_AVAILABLE,
    fire2_solve, OPTX_AVAILABLE,
)

# ── constants (identical to make_h2_isotherm_cdft.py) ────────────────────────
KCAL_TO_K  = 503.228
SIGMA_H2   = 2.83
EPSILON_H2 = 59.7
RCUT_H2    = 5.0 * SIGMA_H2
NA         = 6.022140857e23

MORSE_METALS = {"Co", "Fe", "Ni", "Cu", "Mn"}
MORSE_PARAMS = {
    "Co": dict(D_e=2*0.879*KCAL_TO_K, a=0.850, r_e=2.985, cutoff=12.0),
    "Fe": dict(D_e=2*1.092*KCAL_TO_K, a=1.180, r_e=3.015, cutoff=12.0),
    "Ni": dict(D_e=2*1.154*KCAL_TO_K, a=1.210, r_e=3.207, cutoff=12.0),
    "Cu": dict(D_e=2*0.818*KCAL_TO_K, a=1.462, r_e=2.931, cutoff=12.0),
    "Mn": dict(D_e=2*0.994*KCAL_TO_K, a=0.990, r_e=3.015, cutoff=12.0),
}
DREIDING = {
    "H":  (2.84642,   7.64893),
    "C":  (3.47299,  47.85620),
    "N":  (3.26256,  38.94920),
    "O":  (3.03315,  48.15810),
    "F":  (3.09320,  36.48345),
    "Al": (3.91104, 155.99820),
    "Si": (3.80414, 155.99820),
    "Br": (3.51905, 186.19140),
    "Cu": (3.11369,   2.51610),
    "Zn": (4.04468,  27.67710),
    "Co": (2.55800,   7.05000),
    "Cl": (3.52000, 114.23000),
}
MASS_MAP = {
    "H": 1.00784, "C": 12.0107, "N": 14.0067, "O": 15.999,
    "Co": 58.933,  "Cl": 35.45, "F": 18.998,  "Al": 26.9815,
    "Si": 28.0855, "Br": 79.904,"Cu": 63.546,  "Zn": 65.38,
    "Fe": 55.845,  "Ni": 58.693,"Mn": 54.938,
}

STRUCTURES_DIR = os.path.join(ROOT, "applications/h2_cof/structures")
RESULTS_DIR    = os.path.join(ROOT, "applications/h2_cof/results")
FIGURES_DIR    = os.path.join(ROOT, "applications/h2_cof/figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

T_K = 298.0
# Same pressure array as make_h2_isotherm_cdft.py — ensures Picard curve matches
# h2_isotherm_cof333.png exactly.
P_ISO = np.array([1, 5, 10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 450, 500], dtype=float)
P_CONV = 10.0   # state point for convergence demo


# ── structure ────────────────────────────────────────────────────────────────

def load_host(name: str) -> HostAtoms:
    cif = os.path.join(STRUCTURES_DIR, name + ".cif")
    pmg = Structure.from_file(cif)
    return HostAtoms(
        positions=pmg.cart_coords.copy(),
        species=[str(s) for s in pmg.species],
        charges=np.zeros(pmg.num_sites),
        lattice=pmg.lattice.matrix.copy(),
        source=cif,
    )


# ── Vext 3D  (reuses cache from make_h2_isotherm_cdft.py) ───────────────────

def build_vext_3d(host, grid_spacing=0.25*SIGMA_H2, supercell=(3,3,3),
                  cache_path=None):
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        print(f"  Loaded Vext from cache: {cache_path}", flush=True)
        return data["vext_3d"], tuple(data["n_pts"]), data["spacings"], float(data["dV"])

    nx, ny, nz = supercell
    host_sc = build_supercell(host, nx, ny, nz)
    shift = (-(nx//2)*host.lattice[0] - (ny//2)*host.lattice[1]
             - (nz//2)*host.lattice[2])
    pos_sc  = host_sc.positions + shift
    spec_sc = host_sc.species

    lengths = np.linalg.norm(host.lattice, axis=1)
    n_pts   = tuple(max(2, int(np.ceil(l / grid_spacing))) for l in lengths)
    spacings = np.array([lengths[i] / n_pts[i] for i in range(3)])
    dV       = float(spacings.prod())

    grid_xyz = (np.stack(np.meshgrid(
        np.linspace(0, 1, n_pts[0], endpoint=False),
        np.linspace(0, 1, n_pts[1], endpoint=False),
        np.linspace(0, 1, n_pts[2], endpoint=False),
        indexing="ij"), axis=-1).reshape(-1, 3) @ host.lattice)

    lj_params = {el: (0.5*(SIGMA_H2+s), float(np.sqrt(EPSILON_H2*e)))
                 for el,(s,e) in DREIDING.items()
                 if el not in MORSE_METALS}

    vext = np.zeros(len(grid_xyz))
    for el, pos_i in zip(spec_sc, pos_sc):
        dr = grid_xyz - pos_i
        r  = np.sqrt(np.einsum("gi,gi->g", dr, dr).clip(1e-8))
        if el in MORSE_METALS:
            mp = MORSE_PARAMS[el]
            mask = r < mp["cutoff"]
            if mask.any():
                x = np.exp(-mp["a"] * (r[mask] - mp["r_e"]))
                vext[mask] += np.clip(mp["D_e"]*((1-x)**2-1), -mp["D_e"], 1e5)
        elif el in lj_params:
            sig, eps = lj_params[el]
            mask = r < RCUT_H2
            if mask.any():
                sr6 = (sig / r[mask])**6
                vext[mask] += 4*eps*(sr6**2 - sr6)

    vext_3d = vext.reshape(n_pts)
    if cache_path:
        np.save(cache_path, {"vext_3d": vext_3d, "n_pts": np.array(n_pts),
                              "spacings": spacings, "dV": dV})
        print(f"  Cached Vext → {cache_path}", flush=True)
    return vext_3d, n_pts, spacings, dV


# ── solver runners ───────────────────────────────────────────────────────────

def boltzmann_init(vext_3d, rho_b, rho_max, access):
    exp = np.clip(-vext_3d / T_K, -50.0, 20.0)
    return np.where(access, np.clip(rho_b * np.exp(exp), 1e-16, rho_max), 1e-16)


def _linear_picard(rho_init, rho_b, vext_3d, T_K, c1_fn, c1_b,
                   rho_max, access, alpha=0.02, max_iter=50000, tol=1e-5):
    """Identical to make_h2_isotherm_cdft._linear_picard — guaranteed correct branch."""
    rho = rho_init.copy()
    converged = False
    for _ in range(max_iter):
        c1      = c1_fn(rho)
        exp_arg = np.clip(-vext_3d / T_K + c1 - c1_b, -50.0, 50.0)
        rho_tgt = np.where(access, np.clip(rho_b * np.exp(exp_arg), 0.0, rho_max), 0.0)
        rho_new = np.where(access,
                           np.clip((1-alpha)*rho + alpha*rho_tgt, 1e-16, rho_max),
                           1e-16)
        err = float(np.max(np.abs(rho_tgt - rho)))
        rho = rho_new
        if err < tol:
            converged = True
            break
    return rho, converged

def run_anderson(vext_3d, dV, wda, access, rho_max, dx, dy, dz):
    # pre-compute weights in Fourier space
    shape = vext_3d.shape
    w2_hat, w3_hat, w2vec_hat, w_hat = wda._get_weights(shape, dx, dy, dz)
    # define one-body correlation function
    c1_fn  = lambda rho: np.asarray(wda.c1(jnp.asarray(rho), dx, dy, dz, w2_hat, w3_hat, w2vec_hat, w_hat))

    N_arr, rho_prev, rho_prev_b = [], None, None
    for P in P_ISO:
        rho_b = H2_PR.bulk_density(P, T_K)
        c1_b  = wda.c1_bulk(rho_b)
        if rho_prev is not None:
            rho0 = np.where(access, np.clip(rho_prev*(rho_b/rho_prev_b), 1e-16, rho_max), 1e-16)
        else:
            rho0 = boltzmann_init(vext_3d, rho_b, rho_max, access)
        res = anderson_solve(rho0, rho_b, vext_3d, T_K, c1_fn, c1_b,
                             m=8, beta=0.1, max_iter=8000, tol=1e-5,
                             accessibility_mask=access,
                             safeguard_alpha=0.01, picard_warmup=100)
        rho_prev, rho_prev_b = res.rho.copy(), rho_b
        N_arr.append(float(res.rho.sum() * dV))
        print(f"  Anderson P={P:5.0f} bar  N={N_arr[-1]:.1f} mol/uc  "
              f"conv={res.converged}", flush=True)
    return np.array(N_arr)

def run_adam(vext_3d, dV, wda, access, rho_max, dx, dy, dz, quadrature=True):
    import optax
    # pre-compute weights in Fourier space
    shape = vext_3d.shape
    w2_hat, w3_hat, w2vec_hat, w_hat = wda._get_weights(shape, dx, dy, dz)
    # define one-body correlation function
    c1_jax = lambda rho: wda.c1(rho, dx, dy, dz, w2_hat, w3_hat, w2vec_hat, w_hat)

    N_arr, rho_prev, rho_prev_b = [], None, None
    for P in P_ISO:
        rho_b = H2_PR.bulk_density(P, T_K)
        c1_b  = wda.c1_bulk(rho_b)
        if rho_prev is not None:
            rho0 = np.where(access, np.clip(rho_prev*(rho_b/rho_prev_b), 1e-16, rho_max), 1e-16)
        else:
            rho0 = boltzmann_init(vext_3d, rho_b, rho_max, access)
        # 1) Adam minimisation with tight *relative* stopping (n_steps and tol
        # bumped; the jax_solver patch makes tol a relative |ΔΩ|/|Ω| test with
        # a streak filter so high-P landscapes are no longer truncated by
        # float32 precision).
        res = jax_solve(rho0, rho_b, vext_3d, T_K, c1_jax, c1_b, dV=dV,
                        optimizer=optax.adam(2e-3), n_steps=8000, tol=1e-8, quadrature=quadrature)
        rho_solved = np.asarray(res.rho).copy()

        rho_prev, rho_prev_b = rho_solved, rho_b
        N_arr.append(float(rho_solved.sum() * dV))
        print(f"  Adam    P={P:5.0f} bar  N={N_arr[-1]:.1f} mol/uc  "
              f"conv={res.converged}", flush=True)
    return np.array(N_arr)


# ── convergence demo at P_CONV ───────────────────────────────────────────────

def run_convergence_demo(vext_3d, dV, wda, access, rho_max, dx, dy, dz):
    rho_b = H2_PR.bulk_density(P_CONV, T_K)
    c1_b  = wda.c1_bulk(rho_b)
    rho0  = boltzmann_init(vext_3d, rho_b, rho_max, access)

    # pre-compute weights in Fourier space
    w2_hat, w3_hat, w2vec_hat, w_hat = wda._get_weights(rho0.shape, dx, dy, dz)

    c1_np  = lambda rho: np.asarray(wda.c1(jnp.asarray(rho), dx, dy, dz, w2_hat, w3_hat, w2vec_hat, w_hat))
    c1_jax = lambda rho: wda.c1(rho, dx, dy, dz, w2_hat, w3_hat, w2vec_hat, w_hat)

    conv = {}

    t0 = time.perf_counter()
    r = anderson_solve(rho0, rho_b, vext_3d, T_K, c1_np, c1_b,
                       m=8, beta=0.1, max_iter=8000, tol=1e-5,
                       accessibility_mask=access,
                       safeguard_alpha=0.01, picard_warmup=100)

    conv["Anderson"] = {"history": r.error_history, "iters": r.iterations,
                        "converged": r.converged, "time_s": time.perf_counter()-t0,
                        "ylabel": r"$\max|F(\rho)|$",
                        "adsorption": jnp.sum(r.rho*dV)}

    if OPTAX_AVAILABLE and EQX_AVAILABLE:
        import optax
        t0 = time.perf_counter()
        r = jax_solve(rho0, rho_b, vext_3d, T_K, c1_jax, c1_b, dV=dV,
                      optimizer=optax.adam(2e-3), n_steps=5000, tol=1e-5, quadrature=False)
        conv["Adam_end_point"] = {"history": r.error_history, "iters": r.iterations,
                        "converged": r.converged, "time_s": time.perf_counter()-t0,
                        "ylabel": r"$|\Delta\Omega|$",
                        "adsorption": jnp.sum(r.rho*dV)}
        t0 = time.perf_counter()
        r = jax_solve(rho0, rho_b, vext_3d, T_K, c1_jax, c1_b, dV=dV,
                      optimizer=optax.adam(2e-3), n_steps=5000, tol=1e-5, quadrature=True)
        conv["Adam_quadrature"] = {"history": r.error_history, "iters": r.iterations,
                        "converged": r.converged, "time_s": time.perf_counter()-t0,
                        "ylabel": r"$|\Delta\Omega|$",
                        "adsorption": jnp.sum(r.rho*dV)}
    else:
        print("  Adam skipped (optax/equinox not available)")
    return conv


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading COF-333-CoCl2...", flush=True)
    host   = load_host("COF-333-CoCl2")
    V_cell = host.cell_volume
    mss    = sum(MASS_MAP.get(el, 0.0) for el in host.species)
    mass_frame_g = mss * 1.66054e-24
    print(f"  V_cell={V_cell:.1f} Å³   mass={mss:.1f} u")

    # Reuse Vext cache from make_h2_isotherm_cdft.py
    vext_cache = os.path.join(RESULTS_DIR, "vext_cache_h2_cof333.npy")
    print("\nBuilding/loading Vext grid...", flush=True)
    vext_3d, n_pts, spacings, dV = build_vext_3d(
        host, grid_spacing=0.25*SIGMA_H2, supercell=(3,3,3),
        cache_path=vext_cache,
    )
    dx, dy, dz = float(spacings[0]), float(spacings[1]), float(spacings[2])
    print(f"  Grid {n_pts}  dV={dV:.4f} Å³  "
          f"Vext ∈ [{vext_3d.min():.0f}, {np.percentile(vext_3d,99.9):.0f}] K")

    wda     = LJWDAFunctional(sigma=SIGMA_H2, epsilon=EPSILON_H2, temperature_K=T_K)
    access  = vext_3d < 50.0 * T_K
    rho_max = float(0.45 * 6.0 / (np.pi * wda.d**3))
    print(f"  BH diameter d={wda.d:.4f} Å  rho_max={rho_max:.4f} Å⁻³")
    print("  Compiling WDA (first call)...", flush=True)
    _ = wda.c1_bulk(1e-5)

    # ── Isotherm: one run per solver ─────────────────────────────────────────
    isotherms = {}

    print(f"\n── Anderson isotherm ──")
    t0 = time.perf_counter()
    isotherms["Anderson"] = run_anderson(vext_3d, dV, wda, access, rho_max, dx, dy, dz)
    print(f"  Total: {time.perf_counter()-t0:.1f}s")

    print(f"\n── Adam end point isotherm ──")
    t0 = time.perf_counter()
    isotherms["Adam_end_point"] = run_adam(vext_3d, dV, wda, access, rho_max, dx, dy, dz, quadrature=False)
    print(f"  Total: {time.perf_counter()-t0:.1f}s")

    print(f"\n── Adam quadrature isotherm ──")
    t0 = time.perf_counter()
    isotherms["Adam_quadrature"] = run_adam(vext_3d, dV, wda, access, rho_max, dx, dy, dz, quadrature=True)
    print(f"  Total: {time.perf_counter()-t0:.1f}s")

    MASS_H2 = 2.016  # g/mol

    def to_wt_pct(N_arr):
        """Convert molecules/uc → gravimetric wt% — same formula as make_h2_isotherm_cdft.py."""
        mass_h2 = N_arr * MASS_H2 / NA   # g per uc
        return mass_h2 / (mass_h2 + mass_frame_g) * 100.0

    def to_mmol_g(N_arr):
        return N_arr / NA * 1000.0 / mass_frame_g

    # ── Convergence demo ─────────────────────────────────────────────────────
    print(f"\n── Convergence demo at P={P_CONV} bar, T={T_K} K ──")
    conv = run_convergence_demo(vext_3d, dV, wda, access, rho_max, dx, dy, dz)
    for name, info in conv.items():
        status = "converged" if info["converged"] else "NOT converged"
        print(f"  {name:10s}: {info['iters']:5d} iters  {info['time_s']:.2f}s  {status} adsorption {info['adsorption']:.4f} mol/cell")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Isotherm comparison  (wt% — same axes as h2_isotherm_cof333.png left panel)
    # ══════════════════════════════════════════════════════════════════════════
    COLORS  = {"Adam_quadrature": "#d6604d", "Adam_end_point": "#ff7f0e", "Anderson": "#2ca02c",  }
    MARKERS = {"Adam_quadrature": "o", "Adam_end_point": "s", "Anderson": "^"}
    LS      = {"Adam_quadrature": "-", "Adam_end_point": "-", "Anderson": "--"}

    # fig, ax = plt.subplots(figsize=(7, 5))
    # for name, N_arr in isotherms.items():
        # wt = to_wt_pct(N_arr)
        # ax.plot(P_ISO, wt, color=COLORS[name], ls=LS[name], lw=2,
                # marker=MARKERS[name], ms=6, label=name)

    # ax.axhline(0, color="gray", lw=0.5, ls="--")
    # ax.set_xlabel("Pressure (bar)", fontsize=12)
    # ax.set_ylabel("Gravimetric H$_2$ uptake (wt%)", fontsize=12)
    # ax.set_title(
        # "COF-333-CoCl$_2$ — H$_2$ adsorption isotherm at 298 K\n"
        # "Morse+LJ V$_\\mathrm{ext}$,  aWBII+WDA functional — solver comparison",
        # fontsize=10)
    # ax.legend(fontsize=10, framealpha=0.85)
    # ax.grid(alpha=0.25)
    # ax.set_xlim(0, 520)
    # ax.set_ylim(0, None)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    # out_iso = os.path.join(FIGURES_DIR, "h2_solver_comparison_isotherm.png")
    # fig.tight_layout()
    # fig.savefig(out_iso, dpi=150, bbox_inches="tight")
    # plt.close(fig)
    # print(f"\nFigure saved: {out_iso}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Convergence / loss curves + time comparison bar chart
    # ══════════════════════════════════════════════════════════════════════════
    n_loss   = len(conv)
    n_cols   = n_loss + 2          # loss panels + 1 time bar chart + 1 adsorption bar chart
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(4.2 * n_loss + 3.6 + 3.6, 4.2),
                             gridspec_kw={"width_ratios": [1]*n_loss + [0.75] + [0.75]})

    # — loss panels —
    for ax, (name, info) in zip(axes[:n_loss], conv.items()):
        hist  = info["history"]
        xs    = np.arange(1, len(hist) + 1)
        color = COLORS.get(name, "gray")
        ax.semilogy(xs, hist, color=color, lw=2)
        status = "converged" if info["converged"] else "NOT converged"
        ax.set_title(
            f"{name}\n{info['iters']} iters  {info['time_s']:.2f} s  [{status}]",
            fontsize=10)
        ax.set_xlabel("Iteration / step", fontsize=11)
        ax.set_ylabel(info["ylabel"], fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # — time bar chart —
    ax_t   = axes[-2]
    names  = list(conv.keys())
    times  = [conv[n]["time_s"] for n in names]
    colors = [COLORS.get(n, "gray") for n in names]
    bars   = ax_t.barh(names, times, color=colors, height=0.55, edgecolor="white")
    for bar, t in zip(bars, times):
        ax_t.text(t + max(times)*0.02, bar.get_y() + bar.get_height()/2,
                  f"{t:.2f}s", va="center", ha="left", fontsize=9)
    ax_t.set_xlabel("Wall-clock time (s)", fontsize=11)
    ax_t.set_title("Time to convergence\n(P=10 bar, T=298 K)", fontsize=10)
    ax_t.set_xlim(0, max(times) * 1.25)
    ax_t.invert_yaxis()
    ax_t.spines["top"].set_visible(False)
    ax_t.spines["right"].set_visible(False)
    ax_t.grid(axis="x", alpha=0.25)

    # - adsorption bar chart
    ax_a   = axes[-1]
    names  = list(conv.keys())
    adsorp  = [conv[n]["adsorption"] for n in names]
    colors = [COLORS.get(n, "gray") for n in names]
    bars   = ax_a.barh(names, times, color=colors, height=0.55, edgecolor="white")
    for bar, a in zip(bars, adsorp):
        ax_a.text(a + max(adsorp)*0.02, bar.get_y() + bar.get_height()/2,
                  f"{t:.2f}s", va="center", ha="left", fontsize=9)
    ax_a.set_xlabel("adsoption (mol/cell)", fontsize=11)
    ax_a.set_title("Adsorption\n(P=10 bar, T=298 K)", fontsize=10)
    ax_a.set_xlim(0, max(adsorp) * 1.25)
    ax_a.invert_yaxis()
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Solver convergence — COF-333-CoCl₂  H₂  T=298 K  P={P_CONV:.0f} bar\n"
        "Morse+LJ V$_\\mathrm{ext}$,  aWBII+WDA c₁",
        fontsize=11, y=1.02)
    fig.tight_layout()
    out_loss = os.path.join(FIGURES_DIR, "h2_solver_comparison_loss.png")
    fig.savefig(out_loss, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_loss}")


if __name__ == "__main__":
    main()

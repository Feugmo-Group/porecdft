"""Solver comparison for H₂/COF-333-CoCl₂ — Morse+LJ Vext + aWBII+WDA c₁.

Runs four porecdft solvers on the full self-consistent grand-potential problem
at T=298 K (same physics as make_h2_isotherm_cdft.py):

  1. Picard   — linear mixing, α=0.02, pressure continuation
  2. Anderson — m=8, β=0.1, pressure continuation
  3. Adam     — optax Adam (lr=2e-3), Boltzmann warm-start per pressure
  4. FIRE2    — optimistix NonlinearCG, Boltzmann warm-start per pressure

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
P_ISO = np.array([1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 500.0])
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


def run_picard(vext_3d, dV, wda, access, rho_max, dx, dy, dz):
    c1_fn = lambda rho: np.asarray(wda.c1(jnp.asarray(rho), dx, dy, dz))
    N_arr, rho_prev, rho_prev_b = [], None, None
    for P in P_ISO:
        rho_b = H2_PR.bulk_density(P, T_K)
        c1_b  = wda.c1_bulk(rho_b)
        if rho_prev is not None:
            rho0 = np.where(access, np.clip(rho_prev*(rho_b/rho_prev_b), 1e-16, rho_max), 1e-16)
        else:
            rho0 = boltzmann_init(vext_3d, rho_b, rho_max, access)
        res = picard_solve(rho0, rho_b, vext_3d, T_K, c1_fn, c1_b,
                           alpha=0.02, max_iter=50000, tol=1e-5,
                           accessibility_mask=access)
        rho_prev, rho_prev_b = res.rho.copy(), rho_b
        N_arr.append(float(res.rho.sum() * dV))
        print(f"  Picard  P={P:5.0f} bar  N={N_arr[-1]:.1f} mol/uc  "
              f"conv={res.converged}", flush=True)
    return np.array(N_arr)


def run_anderson(vext_3d, dV, wda, access, rho_max, dx, dy, dz):
    c1_fn = lambda rho: np.asarray(wda.c1(jnp.asarray(rho), dx, dy, dz))
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


def run_adam(vext_3d, dV, wda, access, rho_max, dx, dy, dz):
    import optax
    c1_fn = lambda rho: wda.c1(rho, dx, dy, dz)
    N_arr, rho_prev, rho_prev_b = [], None, None
    for P in P_ISO:
        rho_b = H2_PR.bulk_density(P, T_K)
        c1_b  = wda.c1_bulk(rho_b)
        if rho_prev is not None:
            rho0 = np.where(access, np.clip(rho_prev*(rho_b/rho_prev_b), 1e-16, rho_max), 1e-16)
        else:
            rho0 = boltzmann_init(vext_3d, rho_b, rho_max, access)
        res = jax_solve(rho0, rho_b, vext_3d, T_K, c1_fn, c1_b, dV=dV,
                        optimizer=optax.adam(2e-3), n_steps=5000, tol=1e-5)
        rho_prev, rho_prev_b = np.asarray(res.rho).copy(), rho_b
        N_arr.append(float(rho_prev.sum() * dV))
        print(f"  Adam    P={P:5.0f} bar  N={N_arr[-1]:.1f} mol/uc  "
              f"conv={res.converged}", flush=True)
    return np.array(N_arr)


def run_fire2(vext_3d, dV, wda, access, rho_max, dx, dy, dz):
    c1_fn = lambda rho: wda.c1(rho, dx, dy, dz)
    N_arr, rho_prev, rho_prev_b = [], None, None
    for P in P_ISO:
        rho_b = H2_PR.bulk_density(P, T_K)
        c1_b  = wda.c1_bulk(rho_b)
        if rho_prev is not None:
            rho0 = np.where(access, np.clip(rho_prev*(rho_b/rho_prev_b), 1e-16, rho_max), 1e-16)
        else:
            rho0 = boltzmann_init(vext_3d, rho_b, rho_max, access)
        res = fire2_solve(rho0, rho_b, vext_3d, T_K, c1_fn, c1_b, dV=dV,
                          rtol=1e-5, atol=1e-7, max_steps=20000)
        rho_prev, rho_prev_b = np.asarray(res.rho).copy(), rho_b
        N_arr.append(float(rho_prev.sum() * dV))
        print(f"  FIRE2   P={P:5.0f} bar  N={N_arr[-1]:.1f} mol/uc  "
              f"conv={res.converged}", flush=True)
    return np.array(N_arr)


# ── convergence demo at P_CONV ───────────────────────────────────────────────

def run_convergence_demo(vext_3d, dV, wda, access, rho_max, dx, dy, dz):
    rho_b = H2_PR.bulk_density(P_CONV, T_K)
    c1_b  = wda.c1_bulk(rho_b)
    rho0  = boltzmann_init(vext_3d, rho_b, rho_max, access)

    c1_np  = lambda rho: np.asarray(wda.c1(jnp.asarray(rho), dx, dy, dz))
    c1_jax = lambda rho: wda.c1(rho, dx, dy, dz)

    conv = {}

    t0 = time.perf_counter()
    r = picard_solve(rho0, rho_b, vext_3d, T_K, c1_np, c1_b,
                     alpha=0.02, max_iter=50000, tol=1e-5,
                     accessibility_mask=access)
    conv["Picard"] = {"history": r.error_history, "iters": r.iterations,
                      "converged": r.converged, "time_s": time.perf_counter()-t0,
                      "ylabel": r"$\|\Delta\rho\|_\infty / \rho_b$"}

    t0 = time.perf_counter()
    r = anderson_solve(rho0, rho_b, vext_3d, T_K, c1_np, c1_b,
                       m=8, beta=0.1, max_iter=8000, tol=1e-5,
                       accessibility_mask=access,
                       safeguard_alpha=0.01, picard_warmup=100)
    conv["Anderson"] = {"history": r.error_history, "iters": r.iterations,
                        "converged": r.converged, "time_s": time.perf_counter()-t0,
                        "ylabel": r"$\max|F(\rho)|$"}

    if OPTAX_AVAILABLE and EQX_AVAILABLE:
        import optax
        t0 = time.perf_counter()
        r = jax_solve(rho0, rho_b, vext_3d, T_K, c1_jax, c1_b, dV=dV,
                      optimizer=optax.adam(2e-3), n_steps=5000, tol=1e-5)
        conv["Adam"] = {"history": r.error_history, "iters": r.iterations,
                        "converged": r.converged, "time_s": time.perf_counter()-t0,
                        "ylabel": r"$|\Delta\Omega|$"}
    else:
        print("  Adam skipped (optax/equinox not available)")

    if OPTX_AVAILABLE:
        t0 = time.perf_counter()
        r = fire2_solve(rho0, rho_b, vext_3d, T_K, c1_jax, c1_b, dV=dV,
                        rtol=1e-5, atol=1e-7, max_steps=20000)
        hist = getattr(r, "error_history", None) or [r.residual]
        conv["FIRE2"] = {"history": hist,
                         "iters": r.iterations, "converged": r.converged,
                         "time_s": time.perf_counter()-t0,
                         "ylabel": r"$\|\nabla\Omega\|$"}
    else:
        print("  FIRE2 skipped (optimistix not available)")

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

    print(f"\n── Picard isotherm ({len(P_ISO)} pressures) ──")
    t0 = time.perf_counter()
    isotherms["Picard"] = run_picard(vext_3d, dV, wda, access, rho_max, dx, dy, dz)
    print(f"  Total: {time.perf_counter()-t0:.1f}s")

    print(f"\n── Anderson isotherm ──")
    t0 = time.perf_counter()
    isotherms["Anderson"] = run_anderson(vext_3d, dV, wda, access, rho_max, dx, dy, dz)
    print(f"  Total: {time.perf_counter()-t0:.1f}s")

    if OPTAX_AVAILABLE and EQX_AVAILABLE:
        print(f"\n── Adam isotherm ──")
        t0 = time.perf_counter()
        isotherms["Adam"] = run_adam(vext_3d, dV, wda, access, rho_max, dx, dy, dz)
        print(f"  Total: {time.perf_counter()-t0:.1f}s")

    if OPTX_AVAILABLE:
        print(f"\n── FIRE2 isotherm ──")
        t0 = time.perf_counter()
        isotherms["FIRE2"] = run_fire2(vext_3d, dV, wda, access, rho_max, dx, dy, dz)
        print(f"  Total: {time.perf_counter()-t0:.1f}s")

    # Convert mol/uc → mmol/g
    def to_mmol_g(N_arr):
        return N_arr / NA * 1000.0 / mass_frame_g

    # ── Convergence demo ─────────────────────────────────────────────────────
    print(f"\n── Convergence demo at P={P_CONV} bar, T={T_K} K ──")
    conv = run_convergence_demo(vext_3d, dV, wda, access, rho_max, dx, dy, dz)
    for name, info in conv.items():
        status = "converged" if info["converged"] else "NOT converged"
        print(f"  {name:10s}: {info['iters']:5d} iters  {info['time_s']:.2f}s  {status}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Isotherm comparison
    # ══════════════════════════════════════════════════════════════════════════
    COLORS  = {"Picard": "#1f77b4", "Anderson": "#ff7f0e",
               "Adam": "#2ca02c",  "FIRE2": "#d62728"}
    MARKERS = {"Picard": "o", "Anderson": "s", "Adam": "^", "FIRE2": "D"}
    LS      = {"Picard": "-", "Anderson": "-", "Adam": "--", "FIRE2": "--"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, N_arr in isotherms.items():
        mmol = to_mmol_g(N_arr)
        ax.plot(P_ISO, mmol, color=COLORS[name], ls=LS[name], lw=2,
                marker=MARKERS[name], ms=6, label=name)

    ax.set_xlabel("Pressure (bar)", fontsize=12)
    ax.set_ylabel("H₂ adsorbed (mmol g⁻¹)", fontsize=12)
    ax.set_title(
        "COF-333-CoCl₂  H₂ isotherm at 298 K\n"
        "Morse+LJ V$_\\mathrm{ext}$,  aWBII+WDA functional — solver comparison",
        fontsize=10)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, P_ISO.max() * 1.05)
    ax.set_ylim(0, None)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_iso = os.path.join(FIGURES_DIR, "h2_solver_comparison_isotherm.png")
    fig.tight_layout()
    fig.savefig(out_iso, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_iso}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Convergence / loss curves + time comparison bar chart
    # ══════════════════════════════════════════════════════════════════════════
    n_loss   = len(conv)
    n_cols   = n_loss + 1          # loss panels + 1 time bar chart
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(4.2 * n_loss + 3.6, 4.2),
                             gridspec_kw={"width_ratios": [1]*n_loss + [0.75]})

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
    ax_t   = axes[-1]
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

"""Solver comparison for H₂ adsorption in COF-333-CoCl₂.

Runs four solvers on the ideal-gas (c1=0) grand-potential problem:
  1. Picard  (classical mixing, α=0.1)
  2. Anderson (accelerated fixed-point, m=8)
  3. Adam    (JAX gradient descent via optax, lr=5e-3)
  4. FIRE2   (NonlinearCG via optimistix)

Functional: c1=0 (ideal gas).  The EL equation is then:
  ρ*(r) = ρ_bulk · exp(−Vext(r)/T)
which is solvable exactly — this is the Henry formula.  All four solvers
should converge to the identical density profile, with the comparison showing
purely the convergence *behaviour* differences.

Using c1=0 avoids the inconsistency between the Picard fixed-point iteration
and the gradient-based Ω minimisation (which differs when the excess free
energy functional F_exc is linearized as ∫(−c1+c1_b)ρ dV).

Two temperatures are shown:
  T=77 K  — cryogenic H₂ storage; large Boltzmann contrasts → slow Picard.
  T=298 K — room temperature; moderate contrasts → fast convergence.

Figure panels:
  A. H₂ isotherm (N_ads vs P, 0.1–100 bar) — both T, all solvers overlay
  B. Convergence curves at P=10 bar, T=77 K — iterations to tolerance

Outputs:
  applications/h2_cof/figures/h2_solver_comparison.png
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

ROOT = str(_REPO_ROOT)

from pymatgen.core import Structure

from porecdft.structure.host import HostAtoms
from porecdft.structure.supercell import build_supercell
from porecdft.solver import (
    picard_solve, anderson_solve,
    jax_solve, OPTAX_AVAILABLE, EQX_AVAILABLE,
    fire2_solve, OPTX_AVAILABLE,
)
import jax.numpy as jnp

# ─── shared constants ────────────────────────────────────────────────────────
STRUCTURES_DIR = os.path.join(ROOT, "applications/h2_cof/structures")
FIGURES_DIR    = os.path.join(ROOT, "applications/h2_cof/figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

SIGMA_H2   = 2.83    # Å  TraPPE single-site
EPSILON_H2 = 59.7    # K
RCUT_H2    = 5.0 * SIGMA_H2
KCAL_TO_K  = 503.228

# Morse parameters (same as make_h2_isotherm.py)
from porecdft.forcefield.morse import MorseParam
MORSE_PARAMS = {
    "Co": MorseParam("Co", D_e=2 * 0.879 * KCAL_TO_K, a=0.850, r_e=2.985),
    "Fe": MorseParam("Fe", D_e=2 * 1.092 * KCAL_TO_K, a=1.180, r_e=3.015),
    "Ni": MorseParam("Ni", D_e=2 * 1.154 * KCAL_TO_K, a=1.210, r_e=3.207),
    "Cu": MorseParam("Cu", D_e=2 * 0.818 * KCAL_TO_K, a=1.462, r_e=2.931),
    "Mn": MorseParam("Mn", D_e=2 * 0.994 * KCAL_TO_K, a=0.990, r_e=3.015),
}
MORSE_METALS = set(MORSE_PARAMS.keys())

from porecdft.io.forcefield import FFEntry
DREIDING_LJ = {
    "H":  FFEntry("H",  2.84642,   7.64893, "DREIDING"),
    "C":  FFEntry("C",  3.47299,  47.85620, "DREIDING"),
    "N":  FFEntry("N",  3.26256,  38.94920, "DREIDING"),
    "O":  FFEntry("O",  3.03315,  48.15810, "DREIDING"),
    "F":  FFEntry("F",  3.09320,  36.48345, "DREIDING"),
    "Al": FFEntry("Al", 3.91104, 155.99820, "DREIDING"),
    "Si": FFEntry("Si", 3.80414, 155.99820, "DREIDING"),
    "Br": FFEntry("Br", 3.51905, 186.19140, "DREIDING"),
    "Cu": FFEntry("Cu", 3.11369,   2.51610, "DREIDING"),
    "Zn": FFEntry("Zn", 4.04468,  27.67710, "DREIDING"),
    "Co": FFEntry("Co", 2.55800,   7.05000, "DREIDING"),
    "Cl": FFEntry("Cl", 3.52000, 114.23000, "DREIDING"),
}
H2_FF_SITE = {"H2": FFEntry("H2", SIGMA_H2, EPSILON_H2, "TraPPE")}

kB_Pa_A3 = 1.380649e-23 * 1e30   # Pa·Å³/K


# ─── structure & Vext ────────────────────────────────────────────────────────

def load_host(name: str) -> HostAtoms:
    cif_path = os.path.join(STRUCTURES_DIR, name + ".cif")
    pmg = Structure.from_file(cif_path)
    return HostAtoms(
        positions=pmg.cart_coords.copy(),
        species=[str(s) for s in pmg.species],
        charges=np.zeros(pmg.num_sites),
        lattice=pmg.lattice.matrix.copy(),
        source=cif_path,
    )


def compute_vext_flat(host: HostAtoms, grid_spacing: float = 0.7,
                      supercell=(3, 3, 3)):
    nx, ny, nz = supercell
    host_sc = build_supercell(host, nx, ny, nz)
    shift = (-(nx // 2) * host.lattice[0]
             - (ny // 2) * host.lattice[1]
             - (nz // 2) * host.lattice[2])
    pos_sc  = host_sc.positions + shift
    spec_sc = host_sc.species

    lengths = np.linalg.norm(host.lattice, axis=1)
    n_pts   = tuple(max(2, int(np.ceil(l / grid_spacing))) for l in lengths)
    fx = np.linspace(0, 1, n_pts[0], endpoint=False)
    fy = np.linspace(0, 1, n_pts[1], endpoint=False)
    fz = np.linspace(0, 1, n_pts[2], endpoint=False)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")
    frac = np.stack([Fx, Fy, Fz], axis=-1).reshape(-1, 3)
    grid = frac @ host.lattice

    lj_params = {}
    for el in set(spec_sc):
        if el in MORSE_METALS or el not in DREIDING_LJ:
            continue
        ff = DREIDING_LJ[el]
        lj_params[el] = (0.5 * (SIGMA_H2 + ff.sigma),
                         float(np.sqrt(EPSILON_H2 * ff.epsilon)))

    vext = np.zeros(len(grid))
    for el, pos_i in zip(spec_sc, pos_sc):
        dr = grid - pos_i
        r  = np.sqrt(np.maximum(np.einsum("gi,gi->g", dr, dr), 1e-8))
        if el in MORSE_METALS:
            mp = MORSE_PARAMS[el]
            mask = r < 12.0
            if np.any(mask):
                x = np.exp(-mp.a * (r[mask] - mp.r_e))
                v = mp.D_e * ((1 - x)**2 - 1)
                vext[mask] += np.clip(v, -mp.D_e, 1e5)
        elif el in lj_params:
            sig, eps = lj_params[el]
            mask = r < RCUT_H2
            if np.any(mask):
                sr6 = (sig / r[mask])**6
                vext[mask] += 4 * eps * (sr6**2 - sr6)

    return vext, n_pts


# ─── physics helpers ─────────────────────────────────────────────────────────

def ideal_gas_density(P_bar: float, T_K: float) -> float:
    """ρ_bulk in molecules/Å³ from ideal gas law."""
    return P_bar * 1e5 / (kB_Pa_A3 * T_K)


# ── Ideal-gas functional (c1=0) ──────────────────────────────────────────────
# EL: ρ*(r) = ρ_bulk · exp(−Vext/T) — Henry regime, exact analytic solution.
# All four solvers converge to the same answer; the comparison shows convergence.

def cs_c1_np(rho_arr):
    """c1=0 (ideal gas) for numpy arrays."""
    return np.zeros_like(rho_arr)

def cs_c1_jax(rho_arr):
    """c1=0 (ideal gas), JAX-traceable."""
    return jnp.zeros_like(rho_arr)

def cs_c1_bulk(rho_b: float) -> float:
    return 0.0


def henry_exact(vext_flat, T_K, V_cell, P_bar, mss_u):
    """Exact Henry-regime N_ads in mmol/g (c1=0, analytical reference)."""
    rho_b = ideal_gas_density(P_bar, T_K)
    acc   = vext_flat < 5.0 * T_K
    v     = np.clip(vext_flat[acc], -5.0 * T_K, None)
    boltz = np.exp(-v / T_K)
    dV    = V_cell / len(vext_flat)
    N_ads = rho_b * boltz.sum() * dV
    NA    = 6.022140857e23
    mass_g = mss_u * 1.66054e-24
    return N_ads / NA * 1000.0 / mass_g


def solver_isotherm(vext_flat_clip, dV, V_cell, mss_u, T_K,
                    P_bar_arr, solver_name: str, **kw):
    """Return N_ads (mmol/g) for each P using Carnahan-Starling c1."""
    NA     = 6.022140857e23
    mass_g = mss_u * 1.66054e-24
    results = []
    for P in P_bar_arr:
        rho_b       = ideal_gas_density(P, T_K)
        c1_bulk_val = cs_c1_bulk(rho_b)
        rho_init    = np.full(len(vext_flat_clip), rho_b)

        if solver_name == "picard":
            res = picard_solve(rho_init, rho_b, vext_flat_clip, T_K,
                               cs_c1_np, c1_bulk_val, **kw)
            rho_out = res.rho
        elif solver_name == "anderson":
            res = anderson_solve(rho_init, rho_b, vext_flat_clip, T_K,
                                 cs_c1_np, c1_bulk_val, **kw)
            rho_out = res.rho
        elif solver_name == "adam":
            res = jax_solve(rho_init, rho_b, vext_flat_clip, T_K,
                            cs_c1_jax, c1_bulk_val, dV=dV, **kw)
            rho_out = res.rho
        elif solver_name == "fire2":
            res = fire2_solve(rho_init, rho_b, vext_flat_clip, T_K,
                              cs_c1_jax, c1_bulk_val, dV=dV, **kw)
            rho_out = res.rho
        else:
            raise ValueError(solver_name)

        N_ads = rho_out.sum() * dV
        results.append(N_ads / NA * 1000.0 / mass_g)
    return np.array(results)


# ─── convergence demo at a single state point ────────────────────────────────

def run_convergence_demo(vext_flat_clip, dV, T_K, P_bar):
    rho_b = ideal_gas_density(P_bar, T_K)
    c1_b  = cs_c1_bulk(rho_b)
    rho_init = np.full(len(vext_flat_clip), rho_b)
    conv = {}

    # Picard
    t0 = time.perf_counter()
    r = picard_solve(rho_init, rho_b, vext_flat_clip, T_K,
                     cs_c1_np, c1_b, alpha=0.1, max_iter=600, tol=1e-5)
    conv["Picard"] = {
        "history": r.error_history, "converged": r.converged,
        "iters": r.iterations, "label": "‖Δρ‖/ρ_b",
        "time_s": time.perf_counter() - t0,
    }

    # Anderson
    t0 = time.perf_counter()
    r = anderson_solve(rho_init, rho_b, vext_flat_clip, T_K,
                       cs_c1_np, c1_b, m=8, beta=0.5, max_iter=300, tol=1e-6)
    conv["Anderson"] = {
        "history": r.error_history, "converged": r.converged,
        "iters": r.iterations, "label": "max|F|",
        "time_s": time.perf_counter() - t0,
    }

    if OPTAX_AVAILABLE and EQX_AVAILABLE:
        import optax
        t0 = time.perf_counter()
        r = jax_solve(rho_init, rho_b, vext_flat_clip, T_K,
                      cs_c1_jax, c1_b, dV=dV,
                      optimizer=optax.adam(5e-3), n_steps=1000, tol=1e-6)
        conv["Adam"] = {
            "history": r.error_history, "converged": r.converged,
            "iters": r.iterations, "label": "|ΔΩ|",
            "time_s": time.perf_counter() - t0,
        }

    if OPTX_AVAILABLE:
        t0 = time.perf_counter()
        r = fire2_solve(rho_init, rho_b, vext_flat_clip, T_K,
                        cs_c1_jax, c1_b, dV=dV, rtol=1e-6, atol=1e-8)
        conv["FIRE2"] = {
            "converged": r.converged, "iters": r.iterations,
            "history": [r.residual],  "label": "‖∇Ω‖",
            "time_s": time.perf_counter() - t0,
        }

    return conv


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading COF-333-CoCl2...", flush=True)
    host   = load_host("COF-333-CoCl2")
    V_cell = host.cell_volume
    mass_map = {
        "H": 1.00784, "C": 12.0107, "N": 14.0067, "O": 15.999,
        "Co": 58.933, "Cl": 35.45, "Fe": 55.845, "Ni": 58.693,
        "Cu": 63.546, "Mn": 54.938, "Al": 26.9815, "Zn": 65.38,
    }
    mss = sum(mass_map.get(el, 0.0) for el in host.species)
    print(f"  V_cell={V_cell:.1f} Å³   mass={mss:.1f} u")

    print("Building Vext grid (0.7 Å, 3×3×3)...", flush=True)
    vext_flat, n_pts = compute_vext_flat(host, grid_spacing=0.7, supercell=(3, 3, 3))
    dV = V_cell / len(vext_flat)
    print(f"  grid {n_pts}, Ng={len(vext_flat)}, dV={dV:.4f} Å³")
    print(f"  Vext raw [{vext_flat.min():.1f}, {np.percentile(vext_flat,99.9):.1f}] K")

    # Clip Vext at ±5·T to remove hard-wall and deep-overlap voxels.
    # T=77K: clip at ±385K;  T=298K: clip at ±1490K
    T_conv = 77.0    # K — cryogenic; non-trivial convergence (large Boltzmann contrast)
    T_iso  = 298.0   # K — room temperature isotherm

    Vclip_77  = np.clip(vext_flat, -5.0 * T_conv, 5.0 * T_conv)
    Vclip_298 = np.clip(vext_flat, -5.0 * T_iso,  5.0 * T_iso)
    print(f"  Vext clipped: "
          f"{100*(Vclip_77 != vext_flat).mean():.1f}% at T=77K, "
          f"{100*(Vclip_298 != vext_flat).mean():.1f}% at T=298K")

    # Pressures for isotherm
    P_iso = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])

    # ── Exact Henry reference ────────────────────────────────────────────────
    print("\nHenry reference (exact, c1=0)...", flush=True)
    N_henry_77  = np.array([henry_exact(vext_flat, T_conv, V_cell, P, mss) for P in P_iso])
    N_henry_298 = np.array([henry_exact(vext_flat, T_iso,  V_cell, P, mss) for P in P_iso])

    # ── Solver isotherms at T=298K ────────────────────────────────────────────
    print("Picard (T=298K)...", flush=True)
    N_picard = solver_isotherm(Vclip_298, dV, V_cell, mss, T_iso, P_iso,
                               "picard", alpha=0.1, max_iter=600, tol=1e-5)

    print("Anderson (T=298K)...", flush=True)
    N_anderson = solver_isotherm(Vclip_298, dV, V_cell, mss, T_iso, P_iso,
                                 "anderson", m=8, beta=0.5, max_iter=300, tol=1e-6)

    N_adam  = None
    N_fire2 = None

    if OPTAX_AVAILABLE and EQX_AVAILABLE:
        import optax
        print("Adam (T=298K)...", flush=True)
        N_adam = solver_isotherm(Vclip_298, dV, V_cell, mss, T_iso, P_iso,
                                 "adam", optimizer=optax.adam(2e-3),
                                 n_steps=3000, tol=1e-6)
    else:
        print("  Adam skipped (optax/equinox not available)")

    if OPTX_AVAILABLE:
        print("FIRE2 (T=298K)...", flush=True)
        N_fire2 = solver_isotherm(Vclip_298, dV, V_cell, mss, T_iso, P_iso,
                                  "fire2", rtol=1e-6, atol=1e-8, max_steps=20_000)
    else:
        print("  FIRE2 skipped (optimistix not available)")

    # ── Convergence demo at P=10 bar, T=77K ──────────────────────────────────
    print("\nConvergence demo at T=77K, P=10 bar...", flush=True)
    conv = run_convergence_demo(Vclip_77, dV, T_conv, 10.0)
    for name, info in conv.items():
        status = "converged" if info["converged"] else "NOT converged"
        print(f"  {name:10s}: {info['iters']:4d} iters  {info['time_s']:.2f}s  {status}")

    # Print loading comparison at P=10 bar, T=298K
    idx10 = np.argmin(np.abs(P_iso - 10.0))
    print(f"\n  Loadings at P=10 bar, T=298K (mmol/g):")
    print(f"    Henry    = {N_henry_298[idx10]:.4f}")
    print(f"    Picard   = {N_picard[idx10]:.4f}")
    print(f"    Anderson = {N_anderson[idx10]:.4f}")
    if N_adam  is not None: print(f"    Adam     = {N_adam[idx10]:.4f}")
    if N_fire2 is not None: print(f"    FIRE2    = {N_fire2[idx10]:.4f}")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE
    # ═════════════════════════════════════════════════════════════════════════
    colors = {"Henry (exact)": "#555555", "Picard": "#1f77b4",
              "Anderson": "#ff7f0e", "Adam": "#2ca02c", "FIRE2": "#d62728"}
    markers = {"Henry (exact)": "x", "Picard": "o", "Anderson": "s",
               "Adam": "^", "FIRE2": "D"}

    fig = plt.figure(figsize=(14, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.35)
    ax_iso = fig.add_subplot(gs[0])
    ax_cv  = fig.add_subplot(gs[1])

    # Panel A — isotherm at T=298K (all solvers + Henry reference)
    ax_iso.plot(P_iso, N_henry_298, color=colors["Henry (exact)"],
                ls="--", lw=2.0, label="Henry (exact, c1=0, T=298K)")
    ax_iso.plot(P_iso, N_henry_77,  color=colors["Henry (exact)"],
                ls=":", lw=1.5, label="Henry (exact, c1=0, T=77K)")
    ax_iso.plot(P_iso, N_picard, color=colors["Picard"],
                lw=2, marker=markers["Picard"], ms=5, label="Picard (T=298K)")
    ax_iso.plot(P_iso, N_anderson, color=colors["Anderson"],
                lw=2, marker=markers["Anderson"], ms=5, label="Anderson (T=298K)")
    if N_adam is not None:
        ax_iso.plot(P_iso, N_adam, color=colors["Adam"],
                    lw=2, marker=markers["Adam"], ms=5, label="Adam (T=298K)")
    if N_fire2 is not None:
        ax_iso.plot(P_iso, N_fire2, color=colors["FIRE2"],
                    lw=2, marker=markers["FIRE2"], ms=5, label="FIRE2 (T=298K)")

    ax_iso.set_xlabel("Pressure (bar)", fontsize=12)
    ax_iso.set_ylabel("H₂ adsorbed (mmol g⁻¹)", fontsize=12)
    ax_iso.set_title(
        "COF-333-CoCl₂ H₂ isotherm — solver comparison\n"
        "c1=0 (ideal gas, Henry regime),  Vext capped at ±5T", fontsize=10)
    ax_iso.legend(fontsize=8.5, framealpha=0.85)
    ax_iso.grid(alpha=0.25)
    ax_iso.set_xlim(0, P_iso.max() * 1.05)
    ax_iso.set_ylim(0, None)
    ax_iso.spines["top"].set_visible(False)
    ax_iso.spines["right"].set_visible(False)

    # Panel B — convergence at T=77K, P=10 bar
    for name, info in conv.items():
        hist = info["history"]
        if len(hist) <= 1:
            continue
        t_s = info["time_s"]
        lbl = f"{name} ({info['iters']} iters, {t_s:.2f}s)"
        ax_cv.semilogy(range(1, len(hist) + 1), hist,
                       color=colors.get(name, "gray"), lw=2, label=lbl)

    ax_cv.set_xlabel("Iteration / step", fontsize=12)
    ax_cv.set_ylabel("Convergence metric (log scale)", fontsize=12)
    ax_cv.set_title(
        "Convergence at T=77 K, P=10 bar  (c1=0)\n"
        "Picard: ‖Δρ‖/ρ_b;   Anderson: max|F|;   Adam: |ΔΩ|", fontsize=10)
    ax_cv.legend(fontsize=9, framealpha=0.85)
    ax_cv.grid(alpha=0.25, which="both")
    ax_cv.spines["top"].set_visible(False)
    ax_cv.spines["right"].set_visible(False)

    fig.suptitle(
        "porecdft solver comparison — H₂/COF-333-CoCl₂  "
        "(Morse+LJ Vext,  ideal-gas functional,  c1=0)",
        fontsize=11, y=1.02)

    out = os.path.join(FIGURES_DIR, "h2_solver_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()

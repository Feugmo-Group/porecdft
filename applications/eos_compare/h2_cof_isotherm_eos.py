"""H2/COF-333-CoCl2 isotherm at 298 K computed with several bulk EOS.

Solves the LJ-WDA self-consistency

    ρ(r) = ρ_bulk(P, T) · exp[ −β V_ext(r) + c¹_WDA(r) − c¹_WDA(ρ_bulk) ]

for the standard Morse + LJ Vext (the one used in `make_h2_isotherm.py`).
For each EOS we swap the bulk-density mapping P → ρ_bulk(P, T):

  * Ideal gas
  * Peng-Robinson (H2_PR)
  * Feynman-Hibbs   (H2_FH) — quantum correction at low T

Run with:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \
        applications/eos_compare/h2_cof_isotherm_eos.py
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARENT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pymatgen.core import Structure

from porecdft.structure.host import HostAtoms
from porecdft.structure.supercell import build_supercell
from porecdft.eos import H2_PR, H2_FH, density_from_pressure
from porecdft.functional import LJWDAFunctional
from porecdft.solver import anderson_solve, picard_solve


# ── constants copied from make_h2_isotherm_cdft.py ─────────────────────────
KCAL_TO_K  = 503.228
SIGMA_H2   = 2.83
EPSILON_H2 = 59.7
RCUT_H2    = 5.0 * SIGMA_H2
NA         = 6.022e23
MASS_H2    = 2.016

MORSE_METALS = {"Co", "Fe", "Ni", "Cu", "Mn"}
MORSE_PARAMS = {
    "Co": dict(D_e=2*0.879*KCAL_TO_K, a=0.850, r_e=2.985, cutoff=12.0),
}
DREIDING = {
    "H":  (2.84642,   7.64893), "C":  (3.47299, 47.85620),
    "N":  (3.26256,  38.94920), "O":  (3.03315, 48.15810),
    "Cl": (3.52000, 114.23000),
}
MASS_MAP = {"H":1.00784,"C":12.0107,"N":14.0067,"O":15.999,
            "Co":58.933,"Cl":35.45}

ROOT = str(_REPO_ROOT)
STRUCTURES_DIR = os.path.join(ROOT, "applications/h2_cof/structures")
RESULTS_DIR    = os.path.join(ROOT, "applications/eos_compare/results")
FIGURES_DIR    = os.path.join(ROOT, "applications/eos_compare/figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

T_K = 298.0
P_ISO = np.array([1, 5, 10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500],
                  dtype=float)


# ── Vext ────────────────────────────────────────────────────────────────────

def load_host(name):
    cif = os.path.join(STRUCTURES_DIR, name + ".cif")
    pmg = Structure.from_file(cif)
    return HostAtoms(
        positions=pmg.cart_coords.copy(),
        species=[str(s) for s in pmg.species],
        charges=np.zeros(pmg.num_sites),
        lattice=pmg.lattice.matrix.copy(),
        source=cif,
    )


def build_vext_3d(host, grid_spacing=0.25*SIGMA_H2, supercell=(3,3,3),
                  cache_path=None):
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        print(f"  Loaded Vext cache: {cache_path}", flush=True)
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
    return vext_3d, n_pts, spacings, dV


# ── isotherm runner ────────────────────────────────────────────────────────

def _linear_picard(rho_init, rho_b, vext_3d, T_K, c1_fn, c1_b,
                   rho_max, access, alpha=0.02, max_iter=50000, tol=1e-5):
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


def run_isotherm(vext_3d, dV, wda, access, rho_max, dx, dy, dz,
                 P_arr, T_K, rho_bulk_fn, label):
    """Anderson + linear-Picard fallback isotherm sweep."""
    c1_fn = lambda rho: np.asarray(wda.c1(jnp.asarray(rho), dx, dy, dz))
    V_cell = float(vext_3d.size * dV)
    max_possible = V_cell / (SIGMA_H2**3 * 0.5)
    N_arr, rho_prev, rho_prev_b, N_prev = [], None, None, None

    for P in P_arr:
        rho_b = float(rho_bulk_fn(P))
        c1_b  = float(wda.c1_bulk(rho_b))
        if rho_prev is not None:
            rho0 = np.where(access, np.clip(rho_prev*(rho_b/rho_prev_b),
                                            1e-16, rho_max), 1e-16)
        else:
            exp_arg = np.clip(-vext_3d / T_K, -50.0, 20.0)
            rho0 = np.where(access,
                            np.clip(rho_b * np.exp(exp_arg), 1e-16, rho_max),
                            1e-16)

        res = anderson_solve(rho0, rho_b, vext_3d, T_K, c1_fn, c1_b,
                             m=8, beta=0.1, max_iter=8000, tol=1e-5,
                             accessibility_mask=access,
                             safeguard_alpha=0.01, picard_warmup=100)
        N = float(res.rho.sum() * dV)
        rho_sol = res.rho

        if N_prev is not None and N > 2.5 * N_prev:
            N = np.inf
        if not np.isfinite(N) or N > max_possible or N < 0:
            rho_sol, _ = _linear_picard(rho0, rho_b, vext_3d, T_K,
                                        c1_fn, c1_b, rho_max, access,
                                        alpha=0.005, max_iter=200000, tol=1e-5)
            N = float(rho_sol.sum() * dV)
        rho_prev, rho_prev_b, N_prev = rho_sol.copy(), rho_b, N
        N_arr.append(N)
    return np.array(N_arr)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"H2/COF-333-CoCl2 isotherm — EOS comparison at T = {T_K:.0f} K")
    print("=" * 70)

    host   = load_host("COF-333-CoCl2")
    V_cell = host.cell_volume
    mss    = sum(MASS_MAP.get(el, 0.0) for el in host.species)
    mass_frame_g = mss * 1.66054e-24
    print(f"  V_cell={V_cell:.1f} Å³  mass={mss:.1f} u", flush=True)

    vext_cache = os.path.join(ROOT, "applications/h2_cof/results/vext_cache_h2_cof333.npy")
    print("\nLoading Vext...", flush=True)
    vext_3d, n_pts, spacings, dV = build_vext_3d(
        host, cache_path=vext_cache,
    )
    dx, dy, dz = float(spacings[0]), float(spacings[1]), float(spacings[2])
    wda = LJWDAFunctional(sigma=SIGMA_H2, epsilon=EPSILON_H2, temperature_K=T_K)
    access = vext_3d < 50.0 * T_K
    rho_max = float(0.45 * 6.0 / (np.pi * wda.d**3))
    _ = wda.c1_bulk(1e-5)
    print(f"  Grid {n_pts}  dV={dV:.4f}  d_BH={wda.d:.3f}", flush=True)

    EOS_CASES = [
        ("Ideal gas", lambda P: density_from_pressure(P, T_K),         "#7f7f7f", "--"),
        ("PR (H2_PR)", lambda P: H2_PR.bulk_density(P, T_K),            "#1f77b4", "-"),
        ("FH (H2_FH)", lambda P: H2_FH.bulk_density(P, T_K),            "#9467bd", "-"),
    ]

    print(f"\nRunning {len(EOS_CASES)} isotherms × {len(P_ISO)} pressures...\n", flush=True)
    curves = {}
    for name, fn, color, ls in EOS_CASES:
        t0 = time.time()
        N_arr = run_isotherm(vext_3d, dV, wda, access, rho_max, dx, dy, dz,
                             P_ISO, T_K, fn, name)
        dt = time.time() - t0
        wt_pct = N_arr * MASS_H2 / NA / (N_arr * MASS_H2 / NA + mass_frame_g) * 100
        mmol_g = N_arr / NA * 1000.0 / mass_frame_g
        curves[name] = (mmol_g, wt_pct, color, ls)
        print(f"  {name:12s}  {dt:5.1f}s  N(500 bar) = {N_arr[-1]:.1f} mol/uc, "
              f"{wt_pct[-1]:.2f} wt%", flush=True)

    # ── plot: two panels (mmol/g + wt%) ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for name, (mmol, wt, color, ls) in curves.items():
        axes[0].plot(P_ISO, mmol, color=color, ls=ls, lw=2.0, marker="o", ms=5,
                     label=name)
        axes[1].plot(P_ISO, wt,   color=color, ls=ls, lw=2.0, marker="o", ms=5,
                     label=name)
    for ax in axes:
        ax.set_xlabel("Pressure (bar)", fontsize=11)
        ax.grid(alpha=0.3, ls=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, P_ISO.max()*1.05)
        ax.legend(fontsize=9, framealpha=0.9)
    axes[0].set_ylabel(r"H$_2$ uptake (mmol g$^{-1}$)", fontsize=11)
    axes[1].set_ylabel(r"H$_2$ uptake (wt %)", fontsize=11)
    axes[0].set_title("Gravimetric loading", fontsize=12, fontweight="bold")
    axes[1].set_title("Weight fraction", fontsize=12, fontweight="bold")
    fig.suptitle(f"H$_2$/COF-333-CoCl$_2$ isotherm at {T_K:.0f} K — bulk-EOS comparison",
                 fontsize=12, fontweight="bold")

    out_png = os.path.join(FIGURES_DIR, "h2_cof_isotherm_eos.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote: {out_png}")


if __name__ == "__main__":
    main()

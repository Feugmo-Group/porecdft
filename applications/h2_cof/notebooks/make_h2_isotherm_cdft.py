"""
Produce h2_isotherm_cof333.png using real self-consistent cDFT.

Replaces make_h2_isotherm.py — same two-panel figure, no hardcoded data.

Panel A: Full-pressure isotherm at 298 K
         COF-333-CoCl2, Morse + LJ external field, aWBII+WDA functional,
         Peng-Robinson bulk density, Anderson solver + Picard fallback,
         pressure continuation.

Panel B: Henry-regime isotherm at 77 / 195 / 298 K (ideal-gas bulk)
         Same Morse+LJ Vext grid, same structure.

Intermediate results cached to applications/h2_cof/results/:
  vext_cache_h2_cof333.npy   — 3D Vext grid (reused on reruns)
  isotherm_h2_cof333_298K.npz — converged isotherm (reused on reruns)
"""
from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path

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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure

# ── porecdft imports ────────────────────────────────────────────────────────
from porecdft.structure.host import HostAtoms
from porecdft.structure.supercell import build_supercell
from porecdft.eos import H2_PR
from porecdft.functional import LJWDAFunctional
from porecdft.solver import anderson_solve, picard_solve

# ════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
KCAL_TO_K  = 503.228
SIGMA_H2   = 2.83       # Å
EPSILON_H2 = 59.7       # K (ε/kB)
MASS_H2    = 2.016      # g/mol
RCUT_H2    = 5.0 * SIGMA_H2
NA         = 6.022e23

MORSE_METALS = {"Co", "Fe", "Ni", "Cu", "Mn"}

# Direct Morse params for host-metal / H2 pairs (no combining rule)
MORSE_PARAMS = {
    "Co": dict(D_e=2*0.879*KCAL_TO_K, a=0.850, r_e=2.985, cutoff=12.0),
    "Fe": dict(D_e=2*1.092*KCAL_TO_K, a=1.180, r_e=3.015, cutoff=12.0),
    "Ni": dict(D_e=2*1.154*KCAL_TO_K, a=1.210, r_e=3.207, cutoff=12.0),
    "Cu": dict(D_e=2*0.818*KCAL_TO_K, a=1.462, r_e=2.931, cutoff=12.0),
    "Mn": dict(D_e=2*0.994*KCAL_TO_K, a=0.990, r_e=3.015, cutoff=12.0),
}

# DREIDING LJ params for organic elements
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


# ════════════════════════════════════════════════════════════════════════════
# 2. STRUCTURE LOADING
# ════════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════════
# 3. VEXT GRID  (3D)
# ════════════════════════════════════════════════════════════════════════════

def build_vext_3d(
    host: HostAtoms,
    grid_spacing: float = 0.25 * SIGMA_H2,
    supercell: tuple = (3, 3, 3),
    cache_path: str | None = None,
) -> tuple[np.ndarray, tuple, np.ndarray, float]:
    """Build Vext on a 3D grid (Nx, Ny, Nz) using Morse+LJ mixed FF.

    Returns
    -------
    vext_3d  : (Nx, Ny, Nz) ndarray in K
    n_pts    : (Nx, Ny, Nz) int tuple
    spacings : (dx, dy, dz) float ndarray in Å
    dV       : float voxel volume in Å³
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        print(f"  Loaded Vext from cache: {cache_path}", flush=True)
        return data["vext_3d"], tuple(data["n_pts"]), data["spacings"], float(data["dV"])

    nx, ny, nz = supercell
    host_sc = build_supercell(host, nx, ny, nz)
    shift = (
        -(nx // 2) * host.lattice[0]
        - (ny // 2) * host.lattice[1]
        - (nz // 2) * host.lattice[2]
    )
    pos_sc  = host_sc.positions + shift
    spec_sc = host_sc.species

    # Grid over ONE unit cell
    lengths = np.linalg.norm(host.lattice, axis=1)
    n_pts   = tuple(max(2, int(np.ceil(l / grid_spacing))) for l in lengths)
    spacings = np.array([lengths[i] / n_pts[i] for i in range(3)])
    dV       = float(spacings.prod())

    fx = np.linspace(0, 1, n_pts[0], endpoint=False)
    fy = np.linspace(0, 1, n_pts[1], endpoint=False)
    fz = np.linspace(0, 1, n_pts[2], endpoint=False)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")
    grid_xyz = np.stack([Fx, Fy, Fz], axis=-1).reshape(-1, 3) @ host.lattice

    Ng   = grid_xyz.shape[0]
    vext = np.zeros(Ng, dtype=float)

    lj_params = {}
    for el in set(spec_sc):
        if el in MORSE_METALS or el not in DREIDING:
            continue
        sigma_s, epsilon_s = DREIDING[el]
        lj_params[el] = (
            0.5 * (SIGMA_H2 + sigma_s),
            float(np.sqrt(EPSILON_H2 * epsilon_s)),
        )

    print(f"  Computing Vext on {n_pts[0]}×{n_pts[1]}×{n_pts[2]} grid "
          f"({len(spec_sc)} atoms in {nx}×{ny}×{nz} supercell)...", flush=True)

    for i, (el, pos_i) in enumerate(zip(spec_sc, pos_sc)):
        dr = grid_xyz - pos_i[None, :]
        r  = np.sqrt(np.einsum("gi,gi->g", dr, dr).clip(1e-8))

        if el in MORSE_METALS:
            mp = MORSE_PARAMS[el]
            mask = r < mp["cutoff"]
            if mask.any():
                x = np.exp(-mp["a"] * (r[mask] - mp["r_e"]))
                v = mp["D_e"] * ((1.0 - x)**2 - 1.0)
                vext[mask] += np.clip(v, -mp["D_e"], 1e5)
        elif el in lj_params:
            sigma_sf, epsilon_sf = lj_params[el]
            mask = r < RCUT_H2
            if mask.any():
                sr6 = (sigma_sf / r[mask])**6
                vext[mask] += 4.0 * epsilon_sf * (sr6**2 - sr6)

    vext_3d = vext.reshape(n_pts)

    if cache_path:
        np.save(cache_path, {
            "vext_3d": vext_3d, "n_pts": np.array(n_pts),
            "spacings": spacings, "dV": dV,
        })
        print(f"  Cached Vext → {cache_path}", flush=True)

    return vext_3d, n_pts, spacings, dV


# ════════════════════════════════════════════════════════════════════════════
# 4. CDFT ISOTHERM  (aWBII+WDA, Peng-Robinson bulk, Anderson + Picard fallback)
# ════════════════════════════════════════════════════════════════════════════

def run_isotherm_cdft(
    vext_3d: np.ndarray,
    spacings: np.ndarray,
    dV: float,
    V_cell: float,
    mss_u: float,
    T_K: float,
    pressures_bar: list[float],
    cache_path: str | None = None,
) -> dict:
    """Self-consistent cDFT isotherm using aWBII+WDA functional.

    Returns dict with keys:
      P, N_abs, extra_gL, wt_pct, mmol_g, converged
    """
    if cache_path and os.path.exists(cache_path):
        data = dict(np.load(cache_path))
        print(f"  Loaded isotherm from cache: {cache_path}", flush=True)
        return data

    dx, dy, dz = float(spacings[0]), float(spacings[1]), float(spacings[2])

    wda = LJWDAFunctional(sigma=SIGMA_H2, epsilon=EPSILON_H2, temperature_K=T_K)
    print(f"  WDA: BH diameter d={wda.d:.4f} Å, r_wda={wda._r_wda:.4f} Å", flush=True)

    # Pre-compute WDA weights (triggers JAX compilation)
    print("  Compiling WDA weights + FMT (first pressure point may be slow)...", flush=True)
    _ = wda.c1_bulk(1e-5)

    # Accessibility mask: voxels where Vext > 50*T are hard walls
    access_mask = (vext_3d < 50.0 * T_K)

    # c1 wrapper: numpy in → numpy out (safe for numpy-based solver)
    def c1_fn(rho_np: np.ndarray) -> np.ndarray:
        return np.asarray(wda.c1(jnp.asarray(rho_np), dx, dy, dz))

    # FMT packing-fraction ceiling for initialisation clamp
    d = wda.d
    rho_max = float(0.45 * 6.0 / (np.pi * d**3))

    results = {k: [] for k in ("P", "N_abs", "extra_gL", "wt_pct", "mmol_g", "converged")}
    mass_frame_g = mss_u * 1.66054e-24   # g per unit cell

    rho_prev     = None
    rho_prev_b   = None
    max_possible = V_cell / (SIGMA_H2**3 * 0.5)

    for P in pressures_bar:
        print(f"\n  P = {P:6.1f} bar", end="  ", flush=True)
        rho_b = H2_PR.bulk_density(P, T_K)
        c1_b  = wda.c1_bulk(rho_b)

        # Initial density
        if rho_prev is not None:
            rho_init = np.clip(rho_prev * (rho_b / rho_prev_b), 1e-16, rho_max)
            rho_init = np.where(access_mask, rho_init, 1e-16)
        else:
            exponent = np.clip(-vext_3d / T_K, -50.0, 20.0)
            rho_init = np.where(access_mask,
                                np.clip(rho_b * np.exp(exponent), 1e-16, rho_max),
                                1e-16)

        # Anderson solve
        res = anderson_solve(
            rho_init=rho_init, rho_bulk=rho_b,
            Vext_K=vext_3d, temperature_K=T_K,
            c1_callable=c1_fn, c1_bulk=c1_b,
            m=8, beta=0.3, max_iter=2000, tol=1e-5,
            accessibility_mask=access_mask,
            safeguard_alpha=0.02, picard_warmup=50,
        )

        rho_sol = res.rho
        N_abs   = float(rho_sol.sum() * dV)

        # Fallback to Picard if Anderson diverged
        if not np.isfinite(N_abs) or N_abs > max_possible or N_abs < 0:
            print("Anderson diverged → Picard...", end="  ", flush=True)
            res = picard_solve(
                rho_init=rho_init, rho_bulk=rho_b,
                Vext_K=vext_3d, temperature_K=T_K,
                c1_callable=c1_fn, c1_bulk=c1_b,
                alpha=0.005, max_iter=20000, tol=1e-5,
                accessibility_mask=access_mask,
            )
            rho_sol = res.rho
            N_abs   = float(rho_sol.sum() * dV)

        if not np.isfinite(N_abs) or N_abs > max_possible or N_abs < 0:
            print(f"FAILED (Nabs={N_abs:.2e})", flush=True)
            rho_prev = rho_prev_b = None
            continue

        rho_prev, rho_prev_b = rho_sol.copy(), rho_b

        N_bulk     = rho_b * V_cell
        total_gL   = N_abs  * MASS_H2 / NA * 1e27 / V_cell
        bulk_gL    = N_bulk * MASS_H2 / NA * 1e27 / V_cell
        extra_gL   = total_gL - bulk_gL
        mass_h2_g  = N_abs  * MASS_H2 / NA
        wt_pct     = mass_h2_g / (mass_h2_g + mass_frame_g) * 100.0
        mmol_g     = (N_abs / NA * 1000.0) / mass_frame_g

        results["P"].append(P)
        results["N_abs"].append(N_abs)
        results["extra_gL"].append(extra_gL)
        results["wt_pct"].append(wt_pct)
        results["mmol_g"].append(mmol_g)
        results["converged"].append(res.converged)

        status = "OK" if res.converged else "not-converged"
        print(f"Nabs={N_abs:7.1f} mol/uc  wt%={wt_pct:.2f}  [{status}]", flush=True)

    # Convert to arrays
    results = {k: np.array(v) for k, v in results.items()}

    if cache_path:
        np.savez(cache_path, **results)
        print(f"\n  Cached isotherm → {cache_path}", flush=True)

    return results


# ════════════════════════════════════════════════════════════════════════════
# 5. HENRY-REGIME ISOTHERM  (identical to original script)
# ════════════════════════════════════════════════════════════════════════════

def henry_constant_kh(vext_flat: np.ndarray, T_K: float, V_cell: float) -> float:
    accessible = vext_flat < 5.0 * T_K
    v = np.clip(vext_flat[accessible], -5.0 * T_K, None)
    boltz = np.exp(-v / T_K)
    dV_vox = V_cell / len(vext_flat)
    kB_Pa_A3 = 1.380649e-23 * 1e30
    return float(boltz.sum() * dV_vox / (kB_Pa_A3 * T_K * V_cell))


def henry_isotherm(vext_flat, T_K, V_cell, mss_u, P_bar_arr):
    kH = henry_constant_kh(vext_flat, T_K, V_cell)
    mass_frame_g = mss_u * 1.66054e-24
    mmol_g = np.array([
        (kH * V_cell * P * 1e5 / NA * 1000.0) / mass_frame_g
        for P in P_bar_arr
    ])
    return mmol_g, kH


# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("H₂ isotherm — COF-333-CoCl2 — porecdft aWBII+WDA cDFT")
print("=" * 65)

print("\nLoading structure...", flush=True)
host   = load_host("COF-333-CoCl2")
mss    = sum(MASS_MAP.get(el, 0.0) for el in host.species)
V_cell = host.cell_volume
print(f"  V_cell = {V_cell:.1f} Å³   mss = {mss:.1f} u", flush=True)

print("\nBuilding Vext grid...", flush=True)
vext_cache = os.path.join(RESULTS_DIR, "vext_cache_h2_cof333.npy")
vext_3d, n_pts, spacings, dV = build_vext_3d(
    host,
    grid_spacing=0.25 * SIGMA_H2,
    supercell=(3, 3, 3),
    cache_path=vext_cache,
)
print(f"  Grid: {n_pts[0]}×{n_pts[1]}×{n_pts[2]}  dV={dV:.4f} Å³  "
      f"Vext ∈ [{vext_3d.min():.0f}, {np.percentile(vext_3d, 99.9):.0f}] K", flush=True)

# ── Panel A: full-pressure cDFT at 298 K ────────────────────────────────────
T_cdft   = 298.0
P_cdft   = [1, 5, 10, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500]
iso_cache = os.path.join(RESULTS_DIR, "isotherm_h2_cof333_298K.npz")

print(f"\nRunning aWBII+WDA isotherm at {T_cdft} K...", flush=True)
iso = run_isotherm_cdft(
    vext_3d=vext_3d,
    spacings=spacings,
    dV=dV,
    V_cell=V_cell,
    mss_u=mss,
    T_K=T_cdft,
    pressures_bar=P_cdft,
    cache_path=iso_cache,
)

# ── Panel B: Henry regime at multiple T ─────────────────────────────────────
vext_flat  = vext_3d.ravel()
P_henry    = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
temps      = [77, 195, 298]
colors_T   = {77: "#2166ac", 195: "#4dac26", 298: "#d6604d"}
labels_T   = {77: "77 K", 195: "195 K", 298: "298 K"}

henry_results = {}
for T in temps:
    mmol_g, kH = henry_isotherm(vext_flat, T, V_cell, mss, P_henry)
    henry_results[T] = mmol_g
    p1 = P_henry[np.argmin(np.abs(P_henry - 1.0))]
    print(f"  Henry T={T:3d} K: KH={kH:.3e} mol/(kg·Pa)  "
          f"N(1 bar)={mmol_g[np.argmin(np.abs(P_henry-1))]:.3f} mmol/g", flush=True)

# mmol/g for cDFT curve on Panel B overlay
mmol_g_cdft_low = iso["mmol_g"][iso["P"] <= 20]
P_cdft_low      = iso["P"][iso["P"] <= 20]

# ════════════════════════════════════════════════════════════════════════════
# 7. FIGURE
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Panel A ─────────────────────────────────────────────────────────────────
ax = axes[0]
l1, = ax.plot(iso["P"], iso["wt_pct"],
              color="#d6604d", lw=2.0, marker="o", ms=5,
              label="Gravimetric uptake (wt%)")
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlabel("Pressure (bar)", fontsize=12)
ax.set_ylabel("Gravimetric H$_2$ uptake (wt%)", fontsize=12, color="#d6604d")
ax.set_title("COF-333-CoCl$_2$ — H$_2$ adsorption isotherm at 298 K\n"
             "porecdft (Morse + LJ external field, aWBII+WDA functional)", fontsize=10)
ax.set_xlim(0, 520)
ax.set_ylim(0, None)
ax.tick_params(labelsize=10, axis="y", colors="#d6604d")
ax.spines["top"].set_visible(False)

ax2 = ax.twinx()
ax2.set_ylabel("Excess H$_2$ uptake (g L$^{-1}$)", fontsize=12, color="#8c564b")
ax2.tick_params(labelsize=10, colors="#8c564b")
if len(iso["extra_gL"]):
    ax2.set_ylim(0, iso["extra_gL"].max() * 1.15)
l2, = ax2.plot(iso["P"], iso["extra_gL"],
               color="#8c564b", lw=1.5, ls="--", marker="s", ms=4, alpha=0.85,
               label="Excess uptake (g L$^{-1}$)")
ax2.spines["top"].set_visible(False)

ax.legend([l1, l2], [l1.get_label(), l2.get_label()],
          fontsize=9, framealpha=0.85, loc="upper left")

# ── Panel B ─────────────────────────────────────────────────────────────────
ax = axes[1]
for T in temps:
    ax.plot(P_henry, henry_results[T],
            color=colors_T[T], lw=2.0, marker="o", ms=5, label=labels_T[T])

# Overlay low-P cDFT points at 298 K
if len(mmol_g_cdft_low):
    ax.plot(P_cdft_low, mmol_g_cdft_low,
            color="#d6604d", lw=0, marker="^", ms=7, alpha=0.7,
            label="298 K (aWBII+WDA cDFT)")

ax.set_xlabel("Pressure (bar)", fontsize=12)
ax.set_ylabel("H$_2$ adsorbed (mmol g$^{-1}$)", fontsize=12)
ax.set_title("COF-333-CoCl$_2$ — Henry-regime isotherm\n"
             "(porecdft Morse+LJ, ideal-gas bulk)", fontsize=10)
ax.set_xlim(0, P_henry.max() * 1.05)
ax.set_ylim(0, None)
ax.legend(fontsize=9, framealpha=0.8)
ax.tick_params(labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(pad=1.5)
out = os.path.join(FIGURES_DIR, "h2_isotherm_cof333.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nFigure saved → {out}")
plt.close()

print("\n" + "=" * 65)
print("STRUCTURE SUMMARY")
print("=" * 65)
warnings.filterwarnings("ignore")
for name in ["COF-301-CoCl2", "COF-333-CoCl2"]:
    cif_path = os.path.join(STRUCTURES_DIR, name + ".cif")
    pmg = Structure.from_file(cif_path)
    sg_sym, sg_num = pmg.get_space_group_info()
    lat = pmg.lattice
    print(f"\n  {name}: {pmg.formula} | {sg_sym} #{sg_num}")
    print(f"    a={lat.a:.4f}  b={lat.b:.4f}  c={lat.c:.4f} Å")
    print(f"    V={pmg.volume:.2f} Å³   N_sites={pmg.num_sites}")

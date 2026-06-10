"""
Produce morse_h2_isotherm.png

Three panels / one combined figure in the style of Evans 2022 Fig. 2A:
  - Cached DFT isotherm from Morse notebook (COF-333-CoCl2, T=298 K)
  - Cached DFT isotherm from LJ notebook   (COF-333-CoCl2, T=298 K)
    (identical outputs — both notebooks ran the same simulation)
  - Henry-regime porecdft prediction (COF-333-CoCl2, Morse+LJ mixed ff)
    at multiple T using KH × P ideal gas approximation

The cached isotherm data was read from notebook output cells:
  data_for_morse/gas_adsorption_jax_Morse.ipynb  (aWBII+WDA cDFT)
  data_for_morse/gas_adsorption_jax_LJ.ipynb     (identical run)
"""

import sys
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

ROOT = str(_REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pymatgen.core import Structure

# ── porecdft imports ────────────────────────────────────────────────────────
from porecdft.structure.host import HostAtoms
from porecdft.structure.supercell import build_supercell
from porecdft.forcefield.morse import MorseParam, MorsePotential
from porecdft.forcefield.lj import LJPotential
from porecdft.forcefield.composite import CompositePotential
from porecdft.forcefield.base import PotentialEnergy
from porecdft.io.forcefield import FFEntry


# ════════════════════════════════════════════════════════════════════════════
# 1. CACHED ISOTHERM DATA  (read from notebook output cells)
# ════════════════════════════════════════════════════════════════════════════
# Both Morse and LJ notebooks produced identical results for COF-333-CoCl2
# at T = 298 K, pressure range 1–500 bar (600 & 700 bar failed to converge).

# P (bar), Nabs (molec/uc), Extra_H2 (g/L), wt%
NOTEBOOK_DATA = np.array([
    [  1,   2.02,  0.37, 0.08],
    [  5,   9.45,  1.71, 0.38],
    [ 10,  17.56,  3.13, 0.70],
    [ 20,  31.07,  5.36, 1.24],
    [ 40,  51.85,  8.42, 2.04],
    [ 60,  67.93, 10.46, 2.66],
    [ 80,  81.19, 11.88, 3.16],
    [100,  92.51, 12.90, 3.59],
    [120, 102.40, 13.63, 3.96],
    [150, 115.23, 14.32, 4.43],
    [200, 132.80, 14.76, 5.07],
    [250, 147.11, 14.65, 5.59],
    [300, 159.13, 14.21, 6.02],
    [400, 178.54, 12.80, 6.70],
    [500, 193.83, 11.08, 7.24],
])

P_bar_cached = NOTEBOOK_DATA[:, 0]
wt_pct_cached = NOTEBOOK_DATA[:, 3]       # gravimetric wt%
extra_gL_cached = NOTEBOOK_DATA[:, 2]     # volumetric excess g/L
nabs_uc_cached = NOTEBOOK_DATA[:, 1]      # molecules / unit cell


# ════════════════════════════════════════════════════════════════════════════
# 2. STRUCTURE INFO
# ════════════════════════════════════════════════════════════════════════════
STRUCTURES_DIR = os.path.join(ROOT, "applications/h2_cof/structures")

def load_host(name: str) -> HostAtoms:
    """Load a CIF file into a HostAtoms object (zero charges — not needed for Morse/LJ)."""
    cif_path = os.path.join(STRUCTURES_DIR, name + ".cif")
    pmg = Structure.from_file(cif_path)
    lattice = pmg.lattice.matrix.copy()   # (3, 3) Å, rows = a, b, c vectors
    positions = pmg.cart_coords.copy()
    species = [str(s) for s in pmg.species]
    charges = np.zeros(len(species))
    return HostAtoms(
        positions=positions,
        species=species,
        charges=charges,
        lattice=lattice,
        source=cif_path,
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. HENRY-REGIME ISOTHERM via porecdft  (COF-333-CoCl2, Morse+LJ mixed ff)
# ════════════════════════════════════════════════════════════════════════════
# For an ideal-gas bulk (Henry regime):
#   N_ads / V_cell = K_H × P_bulk
# where K_H = (1/kB T) × <exp(-Vext/kBT)> is the Henry constant.
# In Kelvin units: K_H [molecules/(Å³·K)] = (1/T) × mean(exp(-Vext/T)) × Ng/V
# Then N_ads = K_H × V_cell × (ρ_bulk from ideal gas: n = P/kB T)

KCAL_TO_K = 503.228   # 1 kcal/mol → K

# H2 TraPPE single-site (same as notebook)
SIGMA_H2 = 2.83   # Å
EPSILON_H2 = 59.7  # K
RCUT_H2 = 5.0 * SIGMA_H2

# Morse parameters from notebook (DREIDING for transition metals)
# D in K (notebook: D = 2 * D_kcal * KCAL_TO_K, factor 2 for two H atoms)
MORSE_PARAMS = {
    "Co": MorseParam("Co", D_e=2 * 0.879 * KCAL_TO_K, a=0.850, r_e=2.985),
    "Fe": MorseParam("Fe", D_e=2 * 1.092 * KCAL_TO_K, a=1.180, r_e=3.015),
    "Ni": MorseParam("Ni", D_e=2 * 1.154 * KCAL_TO_K, a=1.210, r_e=3.207),
    "Cu": MorseParam("Cu", D_e=2 * 0.818 * KCAL_TO_K, a=1.462, r_e=2.931),
    "Mn": MorseParam("Mn", D_e=2 * 0.994 * KCAL_TO_K, a=0.990, r_e=3.015),
}

# H2 as a single Morse-compatible fluid "site" — use a very shallow well
# (representing the fluid self-interaction parameter for cross-combination).
# For H2 single-site: D_e ~ EPSILON_H2, a and r_e fitted so cross-pair via
# geometric-mean D_e and arithmetic-mean a/r_e reproduces notebook Morse params.
# Since the notebook uses DIRECT host-element Morse params (no combining rule
# for fluid), we implement the Vext grid ourselves here for full fidelity.

# DREIDING LJ params for organic elements (same as notebook forcefield.dat)
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
# H2 LJ params (single-site TraPPE)
H2_FF_SITE = {"H2": FFEntry("H2", SIGMA_H2, EPSILON_H2, "TraPPE")}

# Metals that use Morse (not LJ)
MORSE_METALS = set(MORSE_PARAMS.keys())


def compute_vext_grid_morse_lj(host: HostAtoms, grid_spacing: float = 0.5,
                                 supercell: tuple = (3, 3, 3)) -> np.ndarray:
    """
    Compute Vext on a 3D grid using:
      - Morse for Co/Fe/Ni/Cu/Mn host atoms (direct params, no combining rule)
      - LJ (Lorentz-Berthelot) for all other elements vs H2 single site

    Returns vext_flat (Ng,) in Kelvin.
    """
    # Build supercell for PBC
    nx, ny, nz = supercell
    host_sc = build_supercell(host, nx, ny, nz)
    # Centre shift so original cell is in middle
    shift = (
        -(nx // 2) * host.lattice[0]
        - (ny // 2) * host.lattice[1]
        - (nz // 2) * host.lattice[2]
    )
    positions_sc = host_sc.positions + shift
    species_sc = host_sc.species

    # Build grid over original unit cell
    lengths = np.linalg.norm(host.lattice, axis=1)
    n_pts = tuple(max(2, int(np.ceil(l / grid_spacing))) for l in lengths)
    fx = np.linspace(0.0, 1.0, n_pts[0], endpoint=False)
    fy = np.linspace(0.0, 1.0, n_pts[1], endpoint=False)
    fz = np.linspace(0.0, 1.0, n_pts[2], endpoint=False)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")
    frac = np.stack([Fx, Fy, Fz], axis=-1).reshape(-1, 3)
    grid_xyz = frac @ host.lattice   # (Ng, 3)

    Ng = grid_xyz.shape[0]
    vext = np.zeros(Ng, dtype=float)

    # Precompute pair params for non-metal elements
    lj_params = {}
    for el in set(species_sc):
        if el in MORSE_METALS:
            continue
        if el not in DREIDING_LJ:
            print(f"  WARNING: no FF params for {el}, skipping")
            continue
        ff = DREIDING_LJ[el]
        sigma_sf = 0.5 * (SIGMA_H2 + ff.sigma)
        epsilon_sf = float(np.sqrt(EPSILON_H2 * ff.epsilon))
        lj_params[el] = (sigma_sf, epsilon_sf)

    # Loop over host atoms
    total_atoms = len(species_sc)
    for i, (el, pos_i) in enumerate(zip(species_sc, positions_sc)):
        dr = grid_xyz - pos_i[None, :]   # (Ng, 3)
        r2 = np.einsum("gi,gi->g", dr, dr)
        r = np.sqrt(np.maximum(r2, 1e-8))

        if el in MORSE_METALS:
            mp = MORSE_PARAMS[el]
            # Direct Morse (same as notebook): no combining rule — notebook uses
            # pre-set D, alpha, r0 directly for the host-atom/H2 pair.
            cutoff = 12.0
            mask = r < cutoff
            if np.any(mask):
                x = np.exp(-mp.a * (r[mask] - mp.r_e))
                v = mp.D_e * ((1.0 - x) ** 2 - 1.0)
                v = np.clip(v, -mp.D_e, 1e5)
                vext[mask] += v
        else:
            if el not in lj_params:
                continue
            sigma_sf, epsilon_sf = lj_params[el]
            cutoff = RCUT_H2
            mask = r < cutoff
            if np.any(mask):
                sr6 = (sigma_sf / r[mask]) ** 6
                vext[mask] += 4.0 * epsilon_sf * (sr6 ** 2 - sr6)

    return vext, n_pts, grid_xyz.shape[0]


def henry_constant_kh(vext_flat: np.ndarray, T_K: float, V_cell: float) -> float:
    """
    K_H in units of (molecules / Å³ / Pa) via:
      K_H = (1 / (kB T V_cell)) × sum_i exp(-V_i / T) × dV
    where V_i is in Kelvin and T in Kelvin.

    Returns K_H in molecules/(Å³ Pa).

    Voxels with Vext > +5*T are inaccessible (Boltzmann weight < e^-5 ≈ 0.007)
    and are excluded. Deep Morse wells are capped at -10*T to prevent divergence
    from numerical grid artefacts (point-contact singularities).

    In the ideal-gas (Henry) regime:
      N_ads = K_H × kB × T × V_cell × rho_bulk
            = K_H × V_cell × P   [molecules/uc per Pascal]
    """
    # Mark hard-wall (unphysically repulsive) voxels as inaccessible
    # Threshold: 5 * T in Kelvin = Boltzmann factor < e^-5
    accessible = vext_flat < 5.0 * T_K
    v = vext_flat[accessible]
    # Cap deep wells at -5*T to guard against grid-overlap artefacts
    # (Boltzmann factor at -5T = e^5 ≈ 148, a reasonable maximum)
    v = np.clip(v, -5.0 * T_K, None)
    boltz = np.exp(-v / T_K)
    dV = V_cell / len(vext_flat)  # Å³ per voxel (full grid for correct volume)
    kB_Pa_A3 = 1.380649e-23 * 1e30  # Pa·Å³/K
    return boltz.sum() * dV / (kB_Pa_A3 * T_K * V_cell)


def henry_isotherm(vext_flat, T_K, V_cell, mss_u, P_bar_arr):
    """
    Return N_ads in mmol/g for each pressure in P_bar_arr at temperature T_K.
    Uses ideal gas bulk (Henry regime): rho_bulk = P / (kB T) [molecules/Å³].
    """
    kB_PA3 = 1.380649e-23 * 1e30  # Pa·Å³/K
    NA = 6.022e23
    MASS_H2 = 2.016  # g/mol
    mass_frame_g = mss_u * 1.66054e-24  # g per unit cell

    kH = henry_constant_kh(vext_flat, T_K, V_cell)
    result = []
    for P in P_bar_arr:
        P_Pa = P * 1e5
        N_ads = kH * V_cell * P_Pa  # molecules / uc
        # mmol/g = (N_ads / NA × 1000 mmol/mol) / (mass_frame in g)
        mmol_g = (N_ads / NA * 1000.0) / mass_frame_g
        result.append(mmol_g)
    return np.array(result), kH


# ════════════════════════════════════════════════════════════════════════════
# 4. RUN HENRY CALCULATION for COF-333-CoCl2
# ════════════════════════════════════════════════════════════════════════════
print("Loading COF-333-CoCl2 structure...", flush=True)
host = load_host("COF-333-CoCl2")

# Crystal mass (u) — from notebook: 5008.643 u
# Recompute from DREIDING masses
mass_map = {
    "H": 1.00784, "C": 12.0107, "N": 14.0067, "O": 15.999,
    "Co": 58.933, "Cl": 35.45, "F": 18.998, "Al": 26.9815,
    "Si": 28.0855, "Br": 79.904, "Cu": 63.546, "Zn": 65.38,
    "Fe": 55.845, "Ni": 58.693, "Mn": 54.938,
}
mss = sum(mass_map.get(el, 0.0) for el in host.species)
V_cell = host.cell_volume
print(f"  V_cell = {V_cell:.1f} Å³,  mss = {mss:.1f} u", flush=True)

print("Building Vext grid (grid spacing 0.7 Å, 3×3×3 supercell)...", flush=True)
vext_flat, grid_shape, Ng = compute_vext_grid_morse_lj(
    host, grid_spacing=0.7, supercell=(3, 3, 3)
)
print(f"  Grid shape: {grid_shape}, Ng={Ng}", flush=True)
print(f"  Vext range: [{vext_flat.min():.1f}, {np.percentile(vext_flat, 99.9):.1f}] K "
      f"(min, 99.9th pct)", flush=True)

# Henry isotherm at multiple temperatures (low-pressure regime: 0.01–5 bar)
P_henry = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
temperatures = [77, 195, 298]  # K — typical H2 storage temperatures
colors_T = {77: "#2166ac", 195: "#4dac26", 298: "#d6604d"}
labels_T = {77: "77 K", 195: "195 K", 298: "298 K"}

henry_results = {}
for T in temperatures:
    mmol_g, kH = henry_isotherm(vext_flat, T, V_cell, mss, P_henry)
    henry_results[T] = mmol_g
    print(f"  T={T} K: K_H = {kH:.3e} mol/(kg Pa), "
          f"N(1 bar) = {mmol_g[np.argmin(np.abs(P_henry - 1.0))]:.3f} mmol/g", flush=True)

# Also compute mmol/g for cached isotherm (to show on a common y-axis)
NA = 6.022e23
mass_frame_g_cof333 = mss * 1.66054e-24  # g per uc
mmol_g_cached = (nabs_uc_cached / NA * 1000.0) / mass_frame_g_cof333


# ════════════════════════════════════════════════════════════════════════════
# 5. FIGURE — Evans 2022 Fig. 2A style
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── Panel A: cached full-pressure isotherm (298 K, Morse+LJ cDFT, aWBII+WDA) ──
ax = axes[0]
l1, = ax.plot(P_bar_cached, wt_pct_cached,
        color="#d6604d", lw=2.0, marker="o", ms=5,
        label="Gravimetric uptake (wt%)")
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlabel("Pressure (bar)", fontsize=12)
ax.set_ylabel("Gravimetric H$_2$ uptake (wt%)", fontsize=12, color="#d6604d")
ax.set_title("COF-333-CoCl$_2$ — H$_2$ adsorption isotherm at 298 K\n"
             "porecdft (Morse + LJ external potential, aWBII+WDA functional)", fontsize=10)
ax.set_xlim(0, 520)
ax.set_ylim(0, None)
ax.tick_params(labelsize=10, axis="y", colors="#d6604d")
ax.spines["top"].set_visible(False)

# Add secondary y-axis: excess g/L
ax2 = ax.twinx()
ax2.set_ylabel("Excess H$_2$ uptake (g L$^{-1}$)", fontsize=12, color="#8c564b")
ax2.tick_params(labelsize=10, colors="#8c564b")
ax2.set_ylim(0, extra_gL_cached.max() * 1.15)
l2, = ax2.plot(P_bar_cached, extra_gL_cached,
         color="#8c564b", lw=1.5, ls="--", marker="s", ms=4, alpha=0.85,
         label="Excess uptake (g L$^{-1}$)")
ax2.spines["top"].set_visible(False)

# Combined legend with both curves clearly labelled
ax.legend([l1, l2],
          [l1.get_label(), l2.get_label()],
          fontsize=9, framealpha=0.85, loc="upper left")

# ── Panel B: Henry-regime isotherm from porecdft (multi-T) ──
ax = axes[1]
for T in temperatures:
    ax.plot(P_henry, henry_results[T],
            color=colors_T[T], lw=2.0, marker="o", ms=5,
            label=labels_T[T])

# Overlay cached 298 K result in low-P range for comparison
low_P_mask = P_bar_cached <= 20
ax.plot(P_bar_cached[low_P_mask], mmol_g_cached[low_P_mask],
        color="#d6604d", lw=0, marker="^", ms=7, alpha=0.7,
        label="298 K (notebook cDFT)")

ax.set_xlabel("Pressure (bar)", fontsize=12)
ax.set_ylabel("H$_2$ adsorbed (mmol g$^{-1}$)", fontsize=12)
ax.set_title("COF-333-CoCl$_2$ — Henry-regime isotherm\n"
             "(porecdft MorseScalar+LJ, ideal-gas bulk)", fontsize=10)
ax.set_xlim(0, P_henry.max() * 1.05)
ax.set_ylim(0, None)
ax.legend(fontsize=9, framealpha=0.8)
ax.tick_params(labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(pad=1.5)

out_path = os.path.join(ROOT, "applications/h2_cof/figures/h2_isotherm_cof333.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nSaved figure to:\n  {out_path}", flush=True)
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 6. STRUCTURE SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRUCTURE SUMMARY")
print("=" * 70)
import warnings
warnings.filterwarnings("ignore")
for name in ["COF-301-CoCl2", "COF-333-CoCl2"]:
    cif_path = os.path.join(STRUCTURES_DIR, name + ".cif")
    pmg = Structure.from_file(cif_path)
    sg_sym, sg_num = pmg.get_space_group_info()
    lat = pmg.lattice
    print(f"\n  Structure: {name}")
    print(f"    Formula:      {pmg.formula}")
    print(f"    Num sites:    {pmg.num_sites}")
    print(f"    Space group:  {sg_sym} (#{sg_num})")
    print(f"    a = {lat.a:.4f} Å,  b = {lat.b:.4f} Å,  c = {lat.c:.4f} Å")
    print(f"    α = {lat.alpha:.2f}°, β = {lat.beta:.2f}°, γ = {lat.gamma:.2f}°")
    print(f"    V_cell = {pmg.volume:.2f} Å³")
print()

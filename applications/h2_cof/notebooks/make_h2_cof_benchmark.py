"""H2/COF multi-structure benchmark — Morse potential extension of porecdft.

Produces  applications/h2_cof/figures/h2_cof_benchmark.png  with three panels:

  (a) Morse potential profiles V(r) for Co, Fe, Ni, Cu, Mn from
      Pramudya & Mendoza-Cortes (J. Am. Chem. Soc. 2016, 138, 15535).

  (b) Henry-regime H2 uptake N(1 bar, 77 K) for 4 COF frameworks
      (COF-301, COF-322, COF-330, COF-333) × 5 metal dopants.
      Metal-substitution is performed in-silico; the organic framework
      geometry is held fixed from the DFT-optimised structure.

  (c) Full-pressure H2 isotherm in COF-333-CoCl2 at 298 K (best case).
      Both gravimetric (wt%) and excess (g/L); DOE 2025 target (5.5 wt%).

Run from repo root:
    python applications/h2_cof/notebooks/make_h2_cof_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[3]
_PARENT = ROOT.parent
for _p in [str(_PARENT), str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from porecdft.structure.host import HostAtoms
from porecdft.structure.supercell import build_supercell
from porecdft.forcefield.morse import MorseParam
from porecdft.io.forcefield import FFEntry

STRUCTURES_DIR = ROOT / "applications/h2_cof/structures"
OUT = ROOT / "applications/h2_cof/figures/h2_cof_benchmark.png"

# ══════════════════════════════════════════════════════════════════════════
# 1. Morse parameters (Pramudya & Mendoza-Cortes 2016, Table 2)
#    D_e per H2 molecule = 2 × D_e(per H atom, kcal/mol) × 503.228 K/(kcal/mol)
# ══════════════════════════════════════════════════════════════════════════
KCAL_TO_K = 503.228
MORSE_PARAMS = {
    "Co": MorseParam("Co", D_e=2 * 0.879 * KCAL_TO_K, a=0.850, r_e=2.985),
    "Fe": MorseParam("Fe", D_e=2 * 1.092 * KCAL_TO_K, a=1.180, r_e=3.015),
    "Ni": MorseParam("Ni", D_e=2 * 1.154 * KCAL_TO_K, a=1.210, r_e=3.207),
    "Cu": MorseParam("Cu", D_e=2 * 0.818 * KCAL_TO_K, a=1.462, r_e=2.931),
    "Mn": MorseParam("Mn", D_e=2 * 0.994 * KCAL_TO_K, a=0.990, r_e=3.015),
}
METALS = ["Co", "Fe", "Ni", "Cu", "Mn"]
METAL_COLOR = {
    "Co": "#1f77b4",
    "Fe": "#d62728",
    "Ni": "#2ca02c",
    "Cu": "#8c564b",
    "Mn": "#9467bd",
}

# H2 single-site TraPPE LJ parameters
SIGMA_H2, EPSILON_H2 = 2.83, 59.7
RCUT_H2 = 5.0 * SIGMA_H2

# DREIDING LJ parameters for non-metal framework atoms
DREIDING_LJ = {
    "H":  FFEntry("H",  2.84642,   7.64893, "DREIDING"),
    "C":  FFEntry("C",  3.47299,  47.85620, "DREIDING"),
    "N":  FFEntry("N",  3.26256,  38.94920, "DREIDING"),
    "O":  FFEntry("O",  3.03315,  48.15810, "DREIDING"),
    "Cl": FFEntry("Cl", 3.52000, 114.23000, "DREIDING"),
}

mass_map = {
    "H": 1.00784, "C": 12.0107, "N": 14.0067, "O": 15.999,
    "Co": 58.933, "Cl": 35.45, "Fe": 55.845, "Ni": 58.693,
    "Cu": 63.546, "Mn": 54.938, "Pd": 106.42,
}

# COF frameworks: (cif_stem, display_label, base_metal_in_CIF)
COF_LIST = [
    ("COF-301-CoCl2", "COF-301", "Co"),
    ("COF-322-PdCl2", "COF-322", "Pd"),
    ("COF-330-PdCl2", "COF-330", "Pd"),
    ("COF-333-CoCl2", "COF-333", "Co"),
]

COF_MARKER = {
    "COF-301": "o",
    "COF-322": "s",
    "COF-330": "^",
    "COF-333": "D",
}

COF_HATCH = {
    "COF-301": "",
    "COF-322": "//",
    "COF-330": "xx",
    "COF-333": "..",
}


def load_host_with_metal_swap(cif_stem: str, base_metal: str,
                              target_metal: str) -> HostAtoms:
    cif_path = STRUCTURES_DIR / f"{cif_stem}.cif"
    pmg = Structure.from_file(str(cif_path))
    lattice = pmg.lattice.matrix.copy()
    positions = pmg.cart_coords.copy()
    species = [target_metal if str(s) == base_metal else str(s)
               for s in pmg.species]
    charges = np.zeros(len(species))
    return HostAtoms(
        positions=positions, species=species, charges=charges,
        lattice=lattice, source=str(cif_path),
    )


def vext_morse_plus_lj(host: HostAtoms, metal: str, spacing: float = 0.7,
                       supercell=(3, 3, 3)) -> tuple[np.ndarray, float]:
    mp = MORSE_PARAMS[metal]
    nx, ny, nz = supercell
    host_sc = build_supercell(host, nx, ny, nz)
    shift = (-(nx // 2) * host.lattice[0]
             - (ny // 2) * host.lattice[1]
             - (nz // 2) * host.lattice[2])
    positions_sc = host_sc.positions + shift
    species_sc = host_sc.species

    lengths = np.linalg.norm(host.lattice, axis=1)
    n_pts = tuple(max(2, int(np.ceil(l / spacing))) for l in lengths)
    fx = np.linspace(0.0, 1.0, n_pts[0], endpoint=False)
    fy = np.linspace(0.0, 1.0, n_pts[1], endpoint=False)
    fz = np.linspace(0.0, 1.0, n_pts[2], endpoint=False)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")
    grid_xyz = (np.stack([Fx, Fy, Fz], axis=-1).reshape(-1, 3)) @ host.lattice

    lj_params = {}
    for el in set(species_sc):
        if el == metal:
            continue
        if el not in DREIDING_LJ:
            continue
        ff = DREIDING_LJ[el]
        lj_params[el] = (0.5 * (SIGMA_H2 + ff.sigma),
                         float(np.sqrt(EPSILON_H2 * ff.epsilon)))

    vext = np.zeros(grid_xyz.shape[0], dtype=float)
    for pos_i, el in zip(positions_sc, species_sc):
        dr = grid_xyz - pos_i[None, :]
        r2 = np.einsum("gi,gi->g", dr, dr)
        r = np.sqrt(np.maximum(r2, 1e-8))
        if el == metal:
            mask = r < 12.0
            if mask.any():
                x = np.exp(-mp.a * (r[mask] - mp.r_e))
                v = mp.D_e * ((1.0 - x) ** 2 - 1.0)
                vext[mask] += np.clip(v, -mp.D_e, 1e5)
        elif el in lj_params:
            sig_sf, eps_sf = lj_params[el]
            mask = r < RCUT_H2
            if mask.any():
                sr6 = (sig_sf / r[mask]) ** 6
                vext[mask] += 4.0 * eps_sf * (sr6 ** 2 - sr6)

    return vext, host.cell_volume


def henry_n_at_1bar(vext_flat, T_K, V_cell, mass_uc_u):
    """Henry-regime N(1 bar, T_K) in mmol/g."""
    accessible = vext_flat < 5.0 * T_K
    v = np.clip(vext_flat[accessible], -5.0 * T_K, None)
    boltz = np.exp(-v / T_K)
    dV = V_cell / len(vext_flat)
    kB_Pa_A3 = 1.380649e-23 * 1e30
    kH = boltz.sum() * dV / (kB_Pa_A3 * T_K * V_cell)
    mass_g = mass_uc_u * 1.66054e-24
    N_A = 6.022e23
    return kH * V_cell * 1e5 / N_A * 1000.0 / mass_g, kH


# ══════════════════════════════════════════════════════════════════════════
# 2. Compute Henry uptakes: all COFs × all metals at 77 K
# ══════════════════════════════════════════════════════════════════════════
print("Computing H2 Henry uptakes at 77 K, 1 bar...")
results = {}   # {(cof_label, metal): N_mmol_g}
for cif_stem, cof_label, base_metal in COF_LIST:
    print(f"  {cof_label}")
    for metal in METALS:
        host = load_host_with_metal_swap(cif_stem, base_metal, metal)
        mass_uc = sum(mass_map.get(el, 12.0) for el in host.species)
        vext, V = vext_morse_plus_lj(host, metal, spacing=0.7, supercell=(3, 3, 3))
        N, kH = henry_n_at_1bar(vext, 77.0, V, mass_uc)
        results[(cof_label, metal)] = N
        print(f"    {metal}: N(1 bar) = {N:.3f} mmol/g  K_H = {kH:.2e}")

# ══════════════════════════════════════════════════════════════════════════
# 3. Cached COF-333-CoCl2 full-pressure data at 298 K
# ══════════════════════════════════════════════════════════════════════════
NOTEBOOK_DATA = np.array([
    [  1,   2.02,  0.37, 0.08], [  5,   9.45,  1.71, 0.38],
    [ 10,  17.56,  3.13, 0.70], [ 20,  31.07,  5.36, 1.24],
    [ 40,  51.85,  8.42, 2.04], [ 60,  67.93, 10.46, 2.66],
    [ 80,  81.19, 11.88, 3.16], [100,  92.51, 12.90, 3.59],
    [120, 102.40, 13.63, 3.96], [150, 115.23, 14.32, 4.43],
    [200, 132.80, 14.76, 5.07], [250, 147.11, 14.65, 5.59],
    [300, 159.13, 14.21, 6.02], [400, 178.54, 12.80, 6.70],
    [500, 193.83, 11.08, 7.24],
])

# ══════════════════════════════════════════════════════════════════════════
# 4. Figure
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 5.2))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
ax_morse = fig.add_subplot(gs[0, 0])
ax_bar   = fig.add_subplot(gs[0, 1])
ax_iso   = fig.add_subplot(gs[0, 2])

# ── (a) Morse potential profiles ──────────────────────────────────────────
r = np.linspace(2.0, 8.0, 400)
for metal in METALS:
    mp = MORSE_PARAMS[metal]
    x = np.exp(-mp.a * (r - mp.r_e))
    v = mp.D_e * ((1.0 - x) ** 2 - 1.0)
    ax_morse.plot(r, v / 1000.0, lw=2, color=METAL_COLOR[metal],
                  label=fr"{metal}  $D_e$={mp.D_e/1000:.2f} kK"
                        fr"  $\alpha$={mp.a:.2f} Å$^{{-1}}$")
    ax_morse.scatter([mp.r_e], [-mp.D_e / 1000.0],
                     marker="o", s=40, color=METAL_COLOR[metal], zorder=5)
ax_morse.axhline(0, color="gray", lw=0.5, ls="--")
ax_morse.set_xlabel(r"M$\cdots$H$_2$ separation $r$ (Å)", fontsize=11)
ax_morse.set_ylabel(r"$V_\mathrm{Morse}(r)$ ($10^3$ K)", fontsize=11)
ax_morse.set_title("(a) Morse interaction potentials\nPramudya & Mendoza-Cortes 2016",
                   fontsize=10)
ax_morse.set_xlim(2.0, 7.0)
ax_morse.set_ylim(-1.5, 2.5)
ax_morse.legend(fontsize=7.5, loc="upper right", framealpha=0.88)
ax_morse.grid(alpha=0.25)

# ── (b) Grouped bar chart: N(1 bar, 77 K) ────────────────────────────────
cof_labels = [c[1] for c in COF_LIST]
n_cofs = len(cof_labels)
n_metals = len(METALS)
bar_w = 0.15
x = np.arange(n_metals)

for i, (cif_stem, cof_label, _) in enumerate(COF_LIST):
    N_vals = [results[(cof_label, m)] for m in METALS]
    offset = (i - (n_cofs - 1) / 2.0) * bar_w
    bars = ax_bar.bar(x + offset, N_vals, bar_w,
                      label=cof_label,
                      hatch=COF_HATCH[cof_label],
                      color=[METAL_COLOR[m] for m in METALS],
                      edgecolor="white", linewidth=0.5, alpha=0.85)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(METALS, fontsize=11)
ax_bar.set_xlabel("Metal dopant (MCl$_2$)", fontsize=11)
ax_bar.set_ylabel(r"H$_2$ adsorbed at 77 K, 1 bar (mmol g$^{-1}$)", fontsize=10)
ax_bar.set_title("(b) Henry-regime uptake across 4 COFs\n"
                 "porecdft Morse + LJ (3D cDFT, 77 K, 1 bar)", fontsize=10)
ax_bar.grid(axis="y", alpha=0.25)
ax_bar.legend(title="COF framework", fontsize=8, title_fontsize=8,
              loc="upper right", framealpha=0.88)

# add value labels on top bars
for i, (cif_stem, cof_label, _) in enumerate(COF_LIST):
    N_vals = [results[(cof_label, m)] for m in METALS]
    offset = (i - (n_cofs - 1) / 2.0) * bar_w
    for j, N in enumerate(N_vals):
        ax_bar.text(x[j] + offset, N + 0.05, f"{N:.1f}",
                    ha="center", va="bottom", fontsize=6, rotation=90)

# ── (c) Full-pressure COF-333-CoCl2 at 298 K ─────────────────────────────
P_full = NOTEBOOK_DATA[:, 0]
wt     = NOTEBOOK_DATA[:, 3]
ex     = NOTEBOOK_DATA[:, 2]

l1, = ax_iso.plot(P_full, wt, "o-", color="#d6604d", lw=2.0, ms=5,
                  label="Gravimetric (wt%)")
ax_iso.axhline(5.5, color="black", lw=1.0, ls=":", alpha=0.55)
ax_iso.text(460, 5.70, "DOE 2025\n(5.5 wt%)", fontsize=8, color="black",
            ha="right", va="bottom")
ax_iso.set_xlabel("Pressure (bar)", fontsize=11)
ax_iso.set_ylabel("Gravimetric H$_2$ uptake (wt%)", fontsize=11, color="#d6604d")
ax_iso.set_title("(c) Full-pressure isotherm — COF-333-CoCl$_2$, 298 K\n"
                 "porecdft Morse + LJ + aWBII + WDA", fontsize=10)
ax_iso.set_xlim(0, 520)
ax_iso.set_ylim(0, None)
ax_iso.tick_params(axis="y", colors="#d6604d", labelsize=10)
ax_iso.spines["top"].set_visible(False)

ax2 = ax_iso.twinx()
l2, = ax2.plot(P_full, ex, "s--", color="#8c564b", lw=1.5, ms=4, alpha=0.85,
               label=r"Excess (g L$^{-1}$)")
ax2.set_ylabel(r"Excess H$_2$ uptake (g L$^{-1}$)", fontsize=11, color="#8c564b")
ax2.tick_params(axis="y", colors="#8c564b", labelsize=10)
ax2.set_ylim(0, ex.max() * 1.18)
ax2.spines["top"].set_visible(False)
ax_iso.legend([l1, l2], [l1.get_label(), l2.get_label()],
              loc="lower right", fontsize=9, framealpha=0.85)

fig.suptitle(r"H$_2$ adsorption in metalated COFs: porecdft Morse potential benchmark"
             "\n(Pramudya & Mendoza-Cortes 2016 COF series, 5 transition metals)",
             fontsize=11, fontweight="bold")

plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"\nSaved: {OUT}")
plt.close()

# Print summary table
print("\nHenry uptakes N(1 bar, 77 K) in mmol/g:")
header = f"{'COF':<12}" + "".join(f"{m:>10}" for m in METALS)
print(header)
print("-" * (12 + 10 * len(METALS)))
for _, cof_label, _ in COF_LIST:
    row = f"{cof_label:<12}" + "".join(f"{results[(cof_label, m)]:>10.3f}"
                                        for m in METALS)
    print(row)

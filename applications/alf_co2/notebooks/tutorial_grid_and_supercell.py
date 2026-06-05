"""Tutorial: why the cDFT grid lives on the unit cell, not the supercell.

Key conceptual question from the code review:

    grid_xyz, shape, dV = build_grid(host, spacing)       # ← original cell
    host_super = build_supercell(host, 3, 3, 3)           # ← atom positions only

Why not build the grid on host_super?

Answer in one sentence: V_ext is periodic with the lattice, so the supercell
grid is just 27 identical copies of the unit-cell grid — at 27× higher cost.

This script demonstrates that quantitatively with four plots:

  (a) V_ext on the unit-cell grid (computed correctly with 3×3×3 atom set)
  (b) the supercell grid — obtained by tiling (a) 3×3×3 times analytically
  (c) 1D profile showing the tiling
  (d) Henry-constant and cost comparison

Run from the repo root (porecdft/):
    python applications/alf_co2/notebooks/tutorial_grid_and_supercell.py
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO.parent))

from porecdft.io.cif import read_cif
from porecdft.io.charges import assign_hirshfeld_charges
from porecdft.io.forcefield import FFEntry
from porecdft.fluid.co2 import SingleSiteLJ_CO2
from porecdft.forcefield.lj import LJPotential
from porecdft.structure.supercell import build_supercell

STRUCTURES = REPO / "applications/alf_co2/structures"
PARAMS     = REPO / "applications/alf_co2/parameters"
OUT        = REPO / "applications/alf_co2/figures/tutorial_grid_supercell.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── 1. Load ALF ──────────────────────────────────────────────────────────────
print("Loading ALF …")
host = read_cif(str(STRUCTURES / "alf.cif"))
host = assign_hirshfeld_charges(host, str(PARAMS / "charges.csv"))
L  = host.lattice
a  = np.linalg.norm(L[0])
print(f"  cubic a = {a:.3f} Å,  V = {host.cell_volume:.1f} Å³,  {host.n_atoms} atoms")

fluid = SingleSiteLJ_CO2
host_ff = {
    "Al": FFEntry("Al", 4.008, 254.1, "UFF"),
    "C":  FFEntry("C",  3.473, 47.86, "DREIDING"),
    "O":  FFEntry("O",  3.033, 48.16, "DREIDING"),
    "H":  FFEntry("H",  2.846,  7.649, "DREIDING"),
}
potential = LJPotential(host_ff=host_ff, fluid_ff=fluid.ff)
rot = np.eye(3)

# ── 2. Build unit-cell grid ──────────────────────────────────────────────────
spacing = 0.7  # Å

def make_grid(lattice, sp):
    lengths = np.linalg.norm(lattice, axis=1)
    n = tuple(max(2, int(np.ceil(l / sp))) for l in lengths)
    fx = np.linspace(0, 1, n[0], endpoint=False)
    fy = np.linspace(0, 1, n[1], endpoint=False)
    fz = np.linspace(0, 1, n[2], endpoint=False)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")
    cart = (np.stack([Fx, Fy, Fz], axis=-1).reshape(-1, 3)) @ lattice
    dV   = abs(np.linalg.det(lattice)) / np.prod(n)
    return cart, n, dV

grid_uc, shape_uc, dV_uc = make_grid(L, spacing)
Nx, Ny, Nz = shape_uc
print(f"\nUnit-cell grid: {shape_uc} = {np.prod(shape_uc):,} points")

# ── 3. Compute V_ext on unit-cell grid (correct workflow) ────────────────────
#    Use the 3×3×3 centred supercell for atom positions so the 15 Å cutoff
#    is satisfied for every grid point in the original cell.
nx, ny, nz = 3, 3, 3
host_super = build_supercell(host, nx, ny, nz)
from dataclasses import replace
shift = -(nx//2)*L[0] - (ny//2)*L[1] - (nz//2)*L[2]
host_super_c = replace(host_super, positions=host_super.positions + shift,
                       lattice=host_super.lattice)

print("Evaluating V_ext on unit-cell grid …", flush=True)
t0 = time.time()
vext_uc = potential.energy_grid(
    grid_uc, rot, host_super_c, fluid.body_sites, fluid.site_labels
)
t_uc = time.time() - t0
V_uc = vext_uc.reshape(shape_uc)
print(f"  {t_uc:.2f} s   min={vext_uc.min():.0f} K   max(accessible)="
      f"{vext_uc[vext_uc < 1e5].max():.0f} K")

# ── 4. Tile the result 3×3×3 to get what a supercell grid WOULD give ─────────
#    This is what you would compute if you built grid_xyz on the supercell:
#    exactly the same pattern repeated 27 times.
V_tiled = np.tile(V_uc, (nx, ny, nz))          # shape (3Nx, 3Ny, 3Nz)
shape_tiled = V_tiled.shape
n_tiled = np.prod(shape_tiled)
dV_tiled = dV_uc                                # same voxel size

# Timing model: supercell grid would need a 5×5×5 atom set to satisfy cutoffs
# at all points in [0,3a); that's 125x more atoms per grid point.
n_atoms_uc  = host_super_c.n_atoms             # 3³ × 104 = 2808 atoms used for unit-cell grid
n_atoms_sc5 = 5**3 * host.n_atoms             # 5³ × 104 = 13000 atoms needed for supercell grid
# Time would scale as: n_tiled / n_uc * n_atoms_sc5 / n_atoms_uc
t_sc_estimate = t_uc * (n_tiled / np.prod(shape_uc)) * (n_atoms_sc5 / n_atoms_uc)

print(f"\nSupercell tiled: {shape_tiled} = {n_tiled:,} points  ({n_tiled//np.prod(shape_uc)}× more)")
print(f"  Estimated compute time on supercell grid: {t_sc_estimate:.0f} s")

# ── 5. Henry constants — must be identical ───────────────────────────────────
T_K = 298.0
kB_Pa_A3   = 1.380649e-23 * 1e30
cell_mass_g = sum(
    {"Al":26.982,"C":12.011,"O":15.999,"H":1.008}.get(el,12) / 6.022e23
    for el in host.species
)

def henry(vext_flat, dV, T, total_mass_g):
    """K_H in mmol/(g·bar).  total_mass_g = mass of the domain represented by vext_flat."""
    acc = vext_flat < 5.0 * T
    v   = np.clip(vext_flat[acc], -5.0*T, None)
    integral = np.exp(-v / T).sum() * dV          # Å³ (Boltzmann-weighted volume)
    return integral / (kB_Pa_A3 * T) * 1e5 / 6.022e23 * 1000 / total_mass_g

K_H_uc     = henry(vext_uc,         dV_uc,    T_K, cell_mass_g)
K_H_tiled  = henry(V_tiled.ravel(), dV_tiled, T_K, 27 * cell_mass_g)
rel_err    = abs(K_H_tiled - K_H_uc) / max(K_H_uc, 1e-12) * 100

print(f"\nK_H (unit-cell grid): {K_H_uc:.6f} mmol/g/bar")
print(f"K_H (tiled 3×3×3):    {K_H_tiled:.6f} mmol/g/bar  (Δ = {rel_err:.2e}%)")
print("→ Identical to machine precision: the supercell adds zero new information.")

# ── 6. Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 4, figure=fig, wspace=0.38, hspace=0.50)
V_CLIP = 2500.0
norm   = TwoSlopeNorm(vmin=-V_CLIP, vcenter=0, vmax=V_CLIP)
cmap   = "RdBu_r"
iz     = Nz // 2   # mid-z slice

# (a) unit-cell Vext
ax_a = fig.add_subplot(gs[0, 0])
im_a = ax_a.imshow(V_uc[:, :, iz].clip(-V_CLIP, V_CLIP).T,
                   origin="lower", cmap=cmap, norm=norm,
                   extent=[0, 1, 0, 1], aspect="auto")
ax_a.set_title(f"(a)  Unit-cell grid\n"
               f"{Nx}×{Ny}×{Nz} = {np.prod(shape_uc):,} pts — {t_uc:.2f} s", fontsize=9)
ax_a.set_xlabel("Fractional x"); ax_a.set_ylabel("Fractional y")
plt.colorbar(im_a, ax=ax_a, label="V$_{ext}$ (K)", shrink=0.85)

# (b) tiled (supercell) Vext
ax_b = fig.add_subplot(gs[0, 1])
im_b = ax_b.imshow(V_tiled[:, :, iz*nx].clip(-V_CLIP, V_CLIP).T,
                   origin="lower", cmap=cmap, norm=norm,
                   extent=[0, 3, 0, 3], aspect="auto")
ax_b.set_title(f"(b)  Supercell grid (analytical tiling of (a))\n"
               f"{shape_tiled[0]}×{shape_tiled[1]}×{shape_tiled[2]} = {n_tiled:,} pts"
               f" — est. {t_sc_estimate:.0f} s", fontsize=9)
ax_b.set_xlabel("Fractional x (supercell)"); ax_b.set_ylabel("Fractional y (supercell)")
for i in range(1, 3):
    ax_b.axvline(i, color="k", lw=0.8, ls="--")
    ax_b.axhline(i, color="k", lw=0.8, ls="--")
ax_b.add_patch(plt.Rectangle((0,0), 1, 1, fill=False,
                               edgecolor="gold", lw=2.5, label="Original unit cell"))
ax_b.legend(fontsize=8, loc="upper right")
plt.colorbar(im_b, ax=ax_b, label="V$_{ext}$ (K)", shrink=0.85)

# (c) Difference between tile (1,1) from tiled array and unit-cell
tile_11 = V_tiled[Nx:2*Nx, Ny:2*Ny, Nz:2*Nz]
diff    = tile_11 - V_uc
finite  = np.abs(V_uc) < 1e5
vd = max(np.abs(diff[finite]).max(), 1.0)
ax_c = fig.add_subplot(gs[0, 2])
im_c = ax_c.imshow(np.clip(diff[:, :, iz], -vd, vd).T,
                   origin="lower", cmap="bwr", vmin=-vd, vmax=vd,
                   extent=[0, 1, 0, 1], aspect="auto")
ax_c.set_title(f"(c)  V$_{{tiled}}$[tile (1,1)] − V$_{{uc}}$\n"
               f"max |diff| = {vd:.1e} K  (exact zero by construction)", fontsize=9)
ax_c.set_xlabel("Fractional x"); ax_c.set_ylabel("Fractional y")
plt.colorbar(im_c, ax=ax_c, label="ΔV (K)", shrink=0.85)

# (d) Scaling table
ax_d = fig.add_subplot(gs[0, 3])
ax_d.axis("off")
rows = [
    ["Metric", "Unit-cell\ngrid", "3×3×3\nsupercell grid"],
    ["Grid pts",      f"{np.prod(shape_uc):,}", f"{n_tiled:,}"],
    ["RAM (float64)", f"{np.prod(shape_uc)*8/1e6:.1f} MB", f"{n_tiled*8/1e6:.1f} MB"],
    ["Compute time",  f"{t_uc:.2f} s", f"~{t_sc_estimate:.0f} s"],
    ["K$_H$ (mmol/g/bar)", f"{K_H_uc:.5f}", f"{K_H_tiled:.5f}"],
    ["Δ K$_H$",       "—", f"{rel_err:.1e}%"],
    ["New physics?",  "—", "None"],
]
tbl = ax_d.table(cellText=rows[1:], colLabels=rows[0],
                 loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.0)
tbl.scale(1.15, 1.75)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c7bb6")
        cell.set_text_props(color="white", fontweight="bold")
ax_d.set_title("(d)  Cost vs. information", fontsize=9, pad=18)

# (e) 1D profile showing tiling
ax_e = fig.add_subplot(gs[1, :2])
iy_1d = Ny // 2
x_uc  = np.linspace(0, 1, Nx, endpoint=False)
x_sc  = np.linspace(0, 3, shape_tiled[0], endpoint=False)
vline = np.clip(V_uc[:, iy_1d, iz],    -V_CLIP, V_CLIP)
vtile = np.clip(V_tiled[:, iy_1d * nx, iz * nz], -V_CLIP, V_CLIP)
ax_e.plot(x_uc, vline,  "steelblue", lw=2.5, label="Unit-cell grid (correct)")
ax_e.plot(x_sc, vtile,  "r--",       lw=1.5, alpha=0.8,
          label="Supercell grid (tile of unit-cell result)")
for i in range(1, 3):
    ax_e.axvline(i, color="gray", lw=0.8, ls=":", zorder=0)
ax_e.axvspan(0, 1, alpha=0.07, color="steelblue", zorder=0)
ax_e.set_xlabel("Fractional coordinate along a-axis", fontsize=9)
ax_e.set_ylabel("V$_{ext}$ (K)", fontsize=9)
ax_e.set_title(
    "(e)  1D profile along [100]: V(r + L) = V(r)  — the supercell grid is a periodic repetition",
    fontsize=9)
ax_e.legend(fontsize=8)
ax_e.set_ylim(-V_CLIP * 1.05, V_CLIP * 0.6)

# (f) histogram
ax_f = fig.add_subplot(gs[1, 2:])
bins = np.linspace(-V_CLIP, V_CLIP, 80)
ax_f.hist(np.clip(vext_uc, -V_CLIP, V_CLIP), bins=bins, density=True,
          alpha=0.7, color="steelblue",
          label=f"Unit cell   K$_H$ = {K_H_uc:.5f}")
ax_f.hist(np.clip(V_tiled.ravel(), -V_CLIP, V_CLIP), bins=bins,
          density=True, histtype="step", lw=2.0, color="firebrick",
          label=f"Supercell   K$_H$ = {K_H_tiled:.5f}  (Δ = {rel_err:.1e}%)")
ax_f.set_xlabel("V$_{ext}$ (K)", fontsize=9)
ax_f.set_ylabel("Probability density", fontsize=9)
ax_f.set_yscale("log")
ax_f.set_title(
    "(f)  V$_{ext}$ distributions are identical — tiling changes nothing physically",
    fontsize=9)
ax_f.legend(fontsize=8)

fig.suptitle(
    "porecdft tutorial: the cDFT grid belongs on the unit cell\n"
    "The density field ρ(r) is periodic — solving on the supercell\n"
    "gives 27 identical copies at 27× the cost.",
    fontsize=10, fontweight="bold",
)
fig.savefig(str(OUT), dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT}")

print("\n" + "="*60)
print("TAKE-AWAY")
print("="*60)
print(f"  Unit-cell grid : {np.prod(shape_uc):>8,} pts  ({t_uc:.2f} s)")
print(f"  Supercell grid : {n_tiled:>8,} pts  (~{t_sc_estimate:.0f} s  est.)")
print(f"  ΔK_H           : {rel_err:.2e}%  — identical to machine precision")
print()
print("V_ext is periodic: V(r + nL) = V(r) for any integer n.")
print("The unit-cell grid captures all the physics.")
print("The 3×3×3 supercell provides host-atom positions for")
print("the LJ/Coulomb cutoff — NOT extra grid evaluation points.")

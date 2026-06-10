"""Phase 2.5 — Flexible host + Wertheim TPT-1 (combined model).

Combines the multi-strain Ω-minimisation from Phase 2 final with the
self-consistent Wertheim association from Phase 2.4.

For each (K_eff, ε_assoc) pair:
  1. For each strain ε and T: compute Langmuir+Wertheim isotherm using
     strain-scaled SC pore centers.
  2. For each (T, p): find ε*(p,T) = argmin Ω(ε, p, T)
         Ω = −T · N_wertheim(ε, p, T) + ½ K_eff · V₀ · (V(ε)/V₀ − 1)²
  3. Read off the loading at ε*.

Grid of models:
  K_eff ∈ {0.5, 5.0} GPa
  ε_assoc ∈ {0, 300, 500, 800} K

Main output: 3-panel (T) isotherm comparison showing the sweet spot where
flexible + association together reproduce Evans across all three temperatures.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial.distance import cdist

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.alf_co2 import ALF_CIF_DFT as ALF_CIF, CHARGES_CSV, DATA_DIR, EXP_TARGETS
from porecdft.diagnostics.isotherm import (
    AVOGADRO,
    IsothermResult,
    density_from_pressure,
)
from porecdft.functional.association import WertheimiAssociation
from porecdft.io import read_cif, read_charges_csv
from porecdft.structure import build_supercell
from porecdft.vext import build_grid

OUT_FIG = DATA_DIR / "figures"
OUT_CACHE = DATA_DIR / "results" / "vext_cache_flex"

ATOMIC_MASS = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}

# EXP_TARGETS imported from applications.alf_co2

# Model grid
K_EFF_LIST   = [0.5, 5.0]          # GPa
EPS_ASSOC_LIST = [0.0, 300.0, 500.0, 800.0]   # K

STRAINS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
TEMPERATURES = [298.0, 273.0, 323.0]
KAPPA_A3 = 119.0                   # association volume per SC site (Å³)
V_EXCL_A3 = 57.0
V_MIN_CLIP_K = -4000.0
BOLTZ_CAP = 50.0
N_PICARD = 4
PA_TO_K_PER_A3 = 1.0e9 / 1.380649e-23 * 1.0e-30


def find_sc_centers(host, vext_grid, grid_3d, access):
    """Deepest accessible Vext minima in the unit cell (filter size 11)."""
    V_masked = np.where(access & np.isfinite(vext_grid), vext_grid, np.inf)
    V_min_f = minimum_filter(V_masked, size=11, mode="wrap")
    mask = (V_masked == V_min_f) & access & np.isfinite(V_masked)
    vals, pos = V_masked[mask], grid_3d[mask]

    inv_lat = np.linalg.inv(host.lattice.T)
    frac = pos @ inv_lat.T
    in_cell = np.all((frac >= -0.01) & (frac <= 1.01), axis=1)
    pos, vals = pos[in_cell], vals[in_cell]
    order = np.argsort(vals)
    pos, vals = pos[order], vals[order]

    D = cdist(pos, pos)
    visited = np.zeros(len(pos), bool)
    idx = []
    for i in range(len(pos)):
        if not visited[i]:
            visited[np.where(D[i] < 5.0)[0]] = True
            idx.append(i)
    return pos[idx]


def langmuir_wertheim_loading(vext, dV, grid_3d, access, rb, T, assoc):
    """Loading (N/cell) from Langmuir + self-consistent Wertheim, one pressure."""
    beta = 1.0 / T
    V = np.maximum(np.where(np.isfinite(vext), vext, +1e6), V_MIN_CLIP_K)
    boltz = np.exp(-np.clip(beta * V, -BOLTZ_CAP, BOLTZ_CAP)) * access
    rho_capped = rb * boltz / (1.0 + rb * boltz * V_EXCL_A3)

    if assoc is not None:
        for _ in range(N_PICARD):
            V_eff = assoc.effective_vext(V, rho_capped, grid_3d, dV, T)
            be = np.exp(-np.clip(beta * V_eff, -BOLTZ_CAP, BOLTZ_CAP)) * access
            rho_capped = rb * be / (1.0 + rb * be * V_EXCL_A3)

    return float(rho_capped.sum() * dV)


def main():
    host0 = read_cif(ALF_CIF)
    charges = read_charges_csv(CHARGES_CSV)
    host0 = host0.assign_charges(charges, source="Hirshfeld CP2K")
    framework_mass_amu = sum(ATOMIC_MASS[s] for s in host0.species)
    framework_mass_g = framework_mass_amu / AVOGADRO
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_g)
    V0 = host0.cell_volume
    a1, a2, a3 = host0.lattice

    pressures_bar = np.logspace(-3, 0, 30)

    # Build base grid + accessibility at ε=0
    grid_xyz, shape, dV = build_grid(host0, 0.7)
    grid_3d_0 = grid_xyz.reshape(*shape, 3)

    host_super0 = build_supercell(host0, 3, 3, 3)
    host_super0 = replace(host_super0, positions=host_super0.positions - a1 - a2 - a3)
    nn0 = np.full(shape, np.inf)
    for h in host_super0.positions:
        dr = grid_3d_0 - h
        nn0 = np.minimum(nn0, np.sqrt(np.einsum("ijkd,ijkd->ijk", dr, dr)))
    access0 = nn0 >= 2.0

    # SC centers from ε=0 Vext
    vext_ref = np.asarray(np.load(
        OUT_CACHE / "vext_avg_strain0.000_T298K.npy", allow_pickle=True
    ).item()["vext_avg"])
    sc0 = find_sc_centers(host0, vext_ref, grid_3d_0, access0)
    print(f"SC sites: {len(sc0)}   κ={KAPPA_A3:.0f} Å³")

    # Pre-cache: N_per_cell[strain, T, pressure] with and without association
    # Using separate per-strain grids would be ideal, but all strains share the
    # same cached Vext on the unit-cell grid (build_grid scales with the host).
    # Approximate: scale SC centers by (1+strain); grid positions scale the same way.

    print("Computing loading table (strain × T × ε_assoc × p)...")
    # loading_table[eps_assoc][strain][T][ip] = N/cell
    loading_table: dict = {}

    for eps_assoc in EPS_ASSOC_LIST:
        loading_table[eps_assoc] = {}
        for strain in STRAINS:
            scale = 1.0 + strain
            V_L = V0 * scale ** 3
            # SC centers scaled for this strain
            sc_scaled = sc0 * scale
            assoc = (WertheimiAssociation.from_positions(sc_scaled, eps_assoc, KAPPA_A3)
                     if eps_assoc > 0 else None)

            loading_table[eps_assoc][strain] = {}
            for T in TEMPERATURES:
                cache = OUT_CACHE / f"vext_avg_strain{strain:.3f}_T{T:.0f}K.npy"
                vext = np.asarray(np.load(cache, allow_pickle=True).item()["vext_avg"])

                # Rebuild grid for scaled cell
                host_s = replace(
                    host0,
                    positions=host0.positions * scale,
                    lattice=host0.lattice * scale,
                )
                gxyz, shp, dV_s = build_grid(host_s, 0.7)
                g3d_s = gxyz.reshape(*shp, 3)

                # Accessibility at this strain
                hs = build_supercell(host_s, 3, 3, 3)
                hs = replace(hs, positions=hs.positions
                             - host_s.lattice[0] - host_s.lattice[1] - host_s.lattice[2])
                nn_s = np.full(shp, np.inf)
                for h in hs.positions:
                    dr = g3d_s - h
                    nn_s = np.minimum(nn_s, np.sqrt(np.einsum("ijkd,ijkd->ijk", dr, dr)))
                acc_s = nn_s >= 2.0

                rho_bulk = np.array([density_from_pressure(p, T) for p in pressures_bar])
                N = np.array([
                    langmuir_wertheim_loading(vext, dV_s, g3d_s, acc_s, rb, T, assoc)
                    for rb in rho_bulk
                ])
                loading_table[eps_assoc][strain][T] = N
            print(f"  ε_assoc={eps_assoc:.0f} K  strain={strain*100:.1f}%  done")

    print("Ω-minimisation...")
    # results[K_eff][eps_assoc][T] = loading_mmol_per_g array (len pressures)
    omega_results: dict = {}
    for K_eff_GPa in K_EFF_LIST:
        K_K_per_A3 = K_eff_GPa * PA_TO_K_PER_A3
        omega_results[K_eff_GPa] = {}
        for eps_assoc in EPS_ASSOC_LIST:
            omega_results[K_eff_GPa][eps_assoc] = {}
            for T in TEMPERATURES:
                loading_min = np.empty(len(pressures_bar))
                for ip in range(len(pressures_bar)):
                    Omegas = []
                    for strain in STRAINS:
                        N = loading_table[eps_assoc][strain][T][ip]
                        V_L = V0 * (1 + strain) ** 3
                        F_el = 0.5 * K_K_per_A3 * V0 * ((V_L / V0) - 1.0) ** 2
                        Omegas.append(-T * N + F_el)
                    idx = int(np.argmin(Omegas))
                    loading_min[ip] = (loading_table[eps_assoc][STRAINS[idx]][T][ip]
                                       * to_mmol_per_g)
                omega_results[K_eff_GPa][eps_assoc][T] = loading_min

    # ---- Plot: 3×2 grid — one column per K_eff, rows by (K_eff, ε_assoc) ----
    # Main figure: one row per K_eff (2 rows), 3 columns per T
    assoc_cmap = plt.get_cmap("plasma")
    assoc_colors = {e: assoc_cmap(i / max(len(EPS_ASSOC_LIST) - 1, 1))
                    for i, e in enumerate(EPS_ASSOC_LIST)}

    fig, axes = plt.subplots(len(K_EFF_LIST), 3, figsize=(16, 4.5 * len(K_EFF_LIST)),
                              sharex=True, sharey=True)
    if len(K_EFF_LIST) == 1:
        axes = axes[np.newaxis, :]

    for row, K_eff in enumerate(K_EFF_LIST):
        for col, T in enumerate(TEMPERATURES):
            ax = axes[row][col]
            for eps in EPS_ASSOC_LIST:
                load = omega_results[K_eff][eps][T]
                lbl = (f"ε={eps:.0f} K ({eps*8.314e-3:.1f} kJ/mol)"
                       if eps > 0 else "no assoc.")
                ax.plot(pressures_bar, load, color=assoc_colors[eps], lw=2.0, label=lbl)
            if T in EXP_TARGETS:
                p_exp, n_exp = zip(*EXP_TARGETS[T])
                ax.plot(p_exp, n_exp, "ko", ms=8, label="Evans Fig 2A")
            ax.set_xscale("log")
            ax.set_xlim(1e-3, 1.2)
            ax.set_ylim(0, 7)
            ax.set_xlabel("Pressure (bar)")
            ax.set_ylabel("CO₂ loading (mmol / g)")
            ax.set_title(f"K_eff={K_eff} GPa   T={T:.0f} K")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Phase 2.5 — Flexible host + Wertheim TPT-1 (combined model)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "21_phase2_5_flex_wertheim.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure: {OUT_FIG}/21_phase2_5_flex_wertheim.png")

    # ---- Best model: K_eff=0.5 GPa, single panel per ε_assoc ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    K_best = 0.5
    for ax, T in zip(axes, TEMPERATURES):
        for eps in EPS_ASSOC_LIST:
            load = omega_results[K_best][eps][T]
            lbl = f"ε={eps:.0f} K" if eps > 0 else "no assoc."
            ax.plot(pressures_bar, load, color=assoc_colors[eps], lw=2.5, label=lbl)
        if T in EXP_TARGETS:
            p_exp, n_exp = zip(*EXP_TARGETS[T])
            ax.plot(p_exp, n_exp, "ko", ms=8, label="Evans Fig 2A")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.2)
        ax.set_ylim(0, 7)
        ax.set_xlabel("Pressure (bar)")
        ax.set_ylabel("CO₂ loading (mmol / g)")
        ax.set_title(f"T = {T:.0f} K")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Phase 2.5 — Flexible (K_eff={K_best} GPa) + Wertheim ε sweep")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "22_phase2_5_best_model.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {OUT_FIG}/22_phase2_5_best_model.png")

    # ---- Summary table ----
    print("\n=== Phase 2.5 @ 1 bar ===")
    for K_eff in K_EFF_LIST:
        print(f"\nK_eff = {K_eff} GPa")
        for T in TEMPERATURES:
            evans = next((n for p, n in EXP_TARGETS.get(T, []) if p > 0.9), None)
            row = f"  T={T:.0f} K (Evans {evans:.2f}): "
            for eps in EPS_ASSOC_LIST:
                v = omega_results[K_eff][eps][T][-1]
                row += f"  ε={eps:.0f}→{v:.2f}"
            print(row)


if __name__ == "__main__":
    main()

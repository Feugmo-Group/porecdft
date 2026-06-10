"""Phase 2.4 — Wertheim TPT-1 association at SC pore centers (self-consistent).

Physical picture
----------------
The orientation-averaged Vext captures the *mean* host-fluid binding.  The
Wertheim term adds the *directed* H-bond bonus ε_assoc that is lost when
averaging over orientations.  Rather than adding extra molecules on top of the
Langmuir baseline (double-counting), we apply Wertheim as an effective-Vext
deepening:

    V_eff(r) = V_ext(r) − T · Δc¹_assoc(r; ρ)

and iterate the Langmuir density self-consistently (4 Picard steps).  The SC
pore centers are identified from the deepest Vext minima in accessible space.

Parameter sweep
---------------
ε_HB ∈ {500, 800, 1100, 1500} K  ≡  {4.2, 6.7, 9.1, 12.5} kJ/mol
κ  = 119 Å³  →  r_κ = 3.1 Å  (association sphere just beyond 2 Å exclusion zone)
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
from applications.alf_co2.notebooks.phase1_vext_validation import _read_forcefield_csv
from porecdft.diagnostics import (
    compute_isotherm_langmuir,
    compute_isotherm_langmuir_assoc_sc,
)
from porecdft.diagnostics.isotherm import AVOGADRO
from porecdft.functional.association import WertheimiAssociation
from porecdft.io import read_cif, read_charges_csv
from porecdft.structure import build_supercell
from porecdft.vext import build_grid

OUT_FIG = DATA_DIR / "figures"
OUT_RES = DATA_DIR / "results"
OUT_CACHE = DATA_DIR / "results" / "vext_cache_flex"

ATOMIC_MASS = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}

# EXP_TARGETS imported from applications.alf_co2

EPS_HB_K_LIST = [500.0, 800.0, 1100.0, 1500.0]
# κ = 119 Å³ → r_κ ≈ 3.1 Å (just past the 2 Å accessibility cutoff)
KAPPA_A3 = 119.0


def find_sc_centers(host, vext_grid, grid_3d, shape, access, dV) -> np.ndarray:
    """Find SC pore centers as deepest accessible Vext minima in the unit cell."""
    V_masked = np.where(access & np.isfinite(vext_grid), vext_grid, np.inf)
    V_min_filter = minimum_filter(V_masked, size=11, mode="wrap")
    local_min_mask = (V_masked == V_min_filter) & access & np.isfinite(V_masked)

    min_vals = V_masked[local_min_mask]
    min_pos = grid_3d[local_min_mask]

    # Keep only positions inside the unit cell (fractional coords in [0,1])
    inv_lat = np.linalg.inv(host.lattice.T)
    frac = min_pos @ inv_lat.T
    in_cell = np.all((frac >= -0.01) & (frac <= 1.01), axis=1)
    min_pos, min_vals = min_pos[in_cell], min_vals[in_cell]

    order = np.argsort(min_vals)
    min_pos, min_vals = min_pos[order], min_vals[order]

    # Cluster sites within 5 Å to keep unique basin minima
    D = cdist(min_pos, min_pos)
    visited = np.zeros(len(min_pos), bool)
    clusters = []
    for i in range(len(min_pos)):
        if not visited[i]:
            visited[np.where(D[i] < 5.0)[0]] = True
            clusters.append(i)
    return min_pos[clusters], min_vals[clusters]


def main():
    host = read_cif(ALF_CIF)
    charges = read_charges_csv(CHARGES_CSV)
    host = host.assign_charges(charges, source="Hirshfeld CP2K")
    framework_mass_amu = sum(ATOMIC_MASS[s] for s in host.species)
    a1, a2, a3 = host.lattice

    temperatures = [298.0, 273.0, 323.0]
    pressures_bar = np.logspace(-3, 0, 30)
    spacing = 0.7

    # Build grid and accessibility
    grid_xyz, shape, dV = build_grid(host, spacing)
    grid_3d = grid_xyz.reshape(*shape, 3)

    host_super = build_supercell(host, 3, 3, 3)
    host_super = replace(host_super, positions=host_super.positions - a1 - a2 - a3)
    nn_dist = np.full(shape, np.inf)
    for h in host_super.positions:
        dr = grid_3d - h
        r = np.sqrt(np.einsum("ijkd,ijkd->ijk", dr, dr))
        nn_dist = np.minimum(nn_dist, r)
    access = nn_dist >= 2.0

    # Load reference Vext (ε=0%, T=298K — geometry only for site finding)
    cache_ref = OUT_CACHE / "vext_avg_strain0.000_T298K.npy"
    vext_ref = np.asarray(np.load(cache_ref, allow_pickle=True).item()["vext_avg"])

    # Find SC pore centers
    sc_centers, sc_V = find_sc_centers(host, vext_ref, grid_3d, shape, access, dV)
    print(f"SC pore centers found: {len(sc_centers)}")
    for i, (pos, v) in enumerate(zip(sc_centers, sc_V)):
        print(f"  [{i}] V={v*8.314e-3:.2f} kJ/mol   "
              f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")

    # Count accessible voxels within r_κ of site 0
    r_kappa = (3 * KAPPA_A3 / (4 * np.pi)) ** (1 / 3)
    dr0 = grid_3d - sc_centers[0]
    d0 = np.sqrt(np.einsum("ijkd,ijkd->ijk", dr0, dr0))
    in_sphere = d0 < r_kappa
    n_acc = (in_sphere & access).sum()
    print(f"\nκ = {KAPPA_A3:.0f} Å³  →  r_κ = {r_kappa:.2f} Å")
    print(f"Accessible voxels in sphere of SC[0]: {n_acc}  (V_acc = {n_acc*dV:.1f} Å³)")

    def load_vext(T: float) -> np.ndarray:
        f = OUT_CACHE / f"vext_avg_strain0.000_T{T:.0f}K.npy"
        return np.asarray(np.load(f, allow_pickle=True).item()["vext_avg"])

    # Baseline (no association)
    baseline = {}
    for T in temperatures:
        vext = load_vext(T)
        baseline[T] = compute_isotherm_langmuir(
            vext_avg_grid_K=vext, dV_A3=dV,
            pressures_bar=pressures_bar, temperature_K=T,
            framework_mass_amu=framework_mass_amu,
            accessibility_mask=access, v_excl_A3=57.0,
        )

    # Association sweeps — self-consistent Vext deepening
    assoc_isos: dict[float, dict[float, object]] = {}
    for eps in EPS_HB_K_LIST:
        assoc = WertheimiAssociation.from_positions(
            sc_centers, energy_K=eps, kappa_A3=KAPPA_A3
        )
        assoc_isos[eps] = {}
        for T in temperatures:
            vext = load_vext(T)
            iso = compute_isotherm_langmuir_assoc_sc(
                vext_avg_grid_K=vext, dV_A3=dV,
                grid_xyz=grid_3d,
                pressures_bar=pressures_bar, temperature_K=T,
                framework_mass_amu=framework_mass_amu,
                assoc=assoc,
                accessibility_mask=access, v_excl_A3=57.0,
                n_picard=4,
            )
            assoc_isos[eps][T] = iso
            delta = iso.loading_mmol_per_g_abs[-1] - baseline[T].loading_mmol_per_g_abs[-1]
            print(f"  ε={eps:.0f} K  T={T:.0f} K  @1bar: "
                  f"{iso.loading_mmol_per_g_abs[-1]:.3f}  (Δ={delta:+.3f})")

    # ---- Plot: 3 panels × T ----
    eps_cmap = plt.get_cmap("plasma")
    eps_colors = {e: eps_cmap(i / max(len(EPS_HB_K_LIST) - 1, 1))
                  for i, e in enumerate(EPS_HB_K_LIST)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, T in zip(axes, temperatures):
        ax.plot(baseline[T].pressures_bar, baseline[T].loading_mmol_per_g_abs,
                "k--", lw=1.8, alpha=0.7, label="Langmuir (no assoc.)")
        for eps in EPS_HB_K_LIST:
            iso = assoc_isos[eps][T]
            ax.plot(iso.pressures_bar, iso.loading_mmol_per_g_abs,
                    color=eps_colors[eps], lw=2.0,
                    label=f"ε_assoc={eps:.0f} K ({eps*8.314e-3:.1f} kJ/mol)")
        if T in EXP_TARGETS:
            p_exp, n_exp = zip(*EXP_TARGETS[T])
            ax.plot(p_exp, n_exp, "ko", ms=8, label="Evans Fig 2A")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.2)
        ax.set_ylim(0, 6)
        ax.set_xlabel("Pressure (bar)")
        ax.set_ylabel("CO₂ loading (mmol / g)")
        ax.set_title(f"T = {T:.0f} K")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(
        f"Phase 2.4 — Wertheim TPT-1 (SC pore centers, κ={KAPPA_A3:.0f} Å³)\n"
        f"{len(sc_centers)} sites · self-consistent Vext deepening (4 Picard)"
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG / "19_phase2_4_wertheim_assoc.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure: {OUT_FIG}/19_phase2_4_wertheim_assoc.png")

    # ---- Δ-loading (association contribution) ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, T in zip(axes, temperatures):
        for eps in EPS_HB_K_LIST:
            iso = assoc_isos[eps][T]
            delta = iso.loading_mmol_per_g_abs - baseline[T].loading_mmol_per_g_abs
            ax.plot(iso.pressures_bar, delta,
                    color=eps_colors[eps], lw=2.0, label=f"ε_assoc={eps:.0f} K")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.2)
        ax.set_xlabel("Pressure (bar)")
        ax.set_ylabel("ΔN_assoc (mmol / g)")
        ax.set_title(f"T = {T:.0f} K — association contribution")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "20_phase2_4_assoc_delta.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {OUT_FIG}/20_phase2_4_assoc_delta.png")

    # ---- Summary ----
    print("\n=== Phase 2.4 summary @ 1 bar ===")
    for T in temperatures:
        evans = next((n for p, n in EXP_TARGETS.get(T, []) if p > 0.9), None)
        print(f"\n  T = {T:.0f} K  (Evans ≈ {evans} mmol/g)")
        print(f"    Langmuir baseline: {baseline[T].loading_mmol_per_g_abs[-1]:.3f}")
        for eps in EPS_HB_K_LIST:
            iso = assoc_isos[eps][T]
            delta = iso.loading_mmol_per_g_abs[-1] - baseline[T].loading_mmol_per_g_abs[-1]
            print(f"    + ε_assoc={eps:.0f} K: {iso.loading_mmol_per_g_abs[-1]:.3f} "
                  f"  (Δ={delta:+.3f} mmol/g,  err vs Evans: "
                  f"{iso.loading_mmol_per_g_abs[-1]-evans:+.3f})")


if __name__ == "__main__":
    main()

"""Phase 2.1 — Henry / ideal-gas-in-external-field baseline isotherm for CO2/ALF.

Strategy:
1. Build the DFT-correct ALF host + Hirshfeld charges.
2. Build the Phase-1-tuned composite potential:
     LJ(Al ε×0, others ε×0.7) + Coulomb(smeared σ=3.0 Å) + Quadrupole-EFG.
3. At each temperature T in {273, 298, 323} K:
     a. Build orientation-averaged Vext on a 3D real-space grid (Boltzmann average
        over 80 Fibonacci orientations) — ONE expensive computation per T.
     b. Sweep pressures p ∈ [1e-3, 1] bar, 25 log-spaced points.
     c. Compute Henry-regime ρ(r) = ρ_bulk · exp(-βV) and integrate to get loading.
4. Plot pycdft isotherms vs experimental targets from Evans paper.

Outputs:
    figures/13_phase2_baseline_isotherms.png    — 3-panel isotherm comparison
    figures/14_phase2_vext_average_T298.png      — 2D slice of orientation-avg Vext
    results/phase2_baseline_isotherms.csv         — raw data
"""
from __future__ import annotations

import csv
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.alf_co2 import (
    ALF_CIF, CHARGES_CSV, FORCEFIELD_CSV, DATA_DIR, EXP_TARGETS,
)
from applications.alf_co2.notebooks.phase1_vext_validation import _read_forcefield_csv
from applications.alf_co2.notebooks.phase1d_lj_tuning import make_lj_variant
from porecdft.diagnostics import compute_isotherm_henry, compute_isotherm_langmuir
from porecdft.diagnostics.isotherm import K_TO_KJ_PER_MOL
from porecdft.fluid import EPM2_CO2
from porecdft.forcefield import (
    CompositePotential, CoulombPotential, LJPotential, QuadrupoleEFGPotential,
)
from porecdft.io import read_cif, read_charges_csv
from porecdft.structure import build_supercell
from porecdft.vext import build_vext_on_grid, build_grid, fibonacci_rotations
from porecdft.plotting import plot_vext_slice_2d

OUT_FIG = DATA_DIR / "figures"
OUT_RES = DATA_DIR / "results"
OUT_CACHE = DATA_DIR / "results" / "vext_cache"
OUT_CACHE.mkdir(parents=True, exist_ok=True)

# Atomic masses (g/mol) for unit-cell mass calculation
ATOMIC_MASS = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}

# Experimental targets (Evans et al. 2022, hand-picked from Fig. 2A description)
# Replace with digitised CSV when available.
# EXP_TARGETS imported from applications.alf_co2


def main():
    # ----- Build host + potentials -----
    host = read_cif(ALF_CIF)
    host_ff = _read_forcefield_csv(FORCEFIELD_CSV)
    charges = read_charges_csv(CHARGES_CSV)
    host = host.assign_charges(charges, source="Hirshfeld CP2K")
    print(host.summary())
    a = float(np.linalg.norm(host.lattice[0]))

    # Framework mass in amu
    framework_mass_amu = sum(ATOMIC_MASS[s] for s in host.species)
    print(f"Framework mass per unit cell: {framework_mass_amu:.1f} amu")

    # Build PBC-replicated host
    host_super = build_supercell(host, 3, 3, 3)
    shift = -host.lattice[0] - host.lattice[1] - host.lattice[2]
    host_super = replace(host_super, positions=host_super.positions + shift)

    # === Phase 2.1b — EPM2 + smeared Coulomb + Quadrupole-EFG ===
    # The single-site LJ baseline under-predicted by 50× because it lacks the
    # H-bond electrostatic attraction (the actual physics in ALF). Now use the
    # full EPM2 3-site CO2 with the v_reject_below_K filter that rejects rigid-
    # geometry artefacts (rigid EPM2 sites accidentally colliding with framework
    # H atoms). Keep full UFF Al + DREIDING C/O/H LJ parameters (no scaling).
    co2 = EPM2_CO2
    lj = LJPotential(host_ff=host_ff, fluid_ff=co2.ff, cutoff=15.0)
    coul = CoulombPotential(
        fluid_charges=co2.charges, cutoff=15.0,
        method="smeared", gauss_width=2.0,        # ~mid of Phase-1.5 scan
    )
    quad = QuadrupoleEFGPotential(theta_zz=co2.theta_zz, cutoff=15.0)
    vtot = CompositePotential([lj, coul, quad])
    print(f"Composite: {vtot.name}  (EPM2 + smeared Coul σ=2.0 + Quad)")

    # Orientation set: 20 Fibonacci rotations — sufficient for linear-molecule
    # orientation averaging at this resolution; cuts Vext build time ~2.5× vs 50.
    n_orient = 20
    rots = fibonacci_rotations(n_orient)
    print(f"Orientations: {n_orient} Fibonacci rotations")

    temperatures = [298.0, 273.0, 323.0]   # 298 first (closest to T_room)
    pressures_bar = np.logspace(-3, 0, 25)

    isotherms = {}
    vext_grids = {}
    for T in temperatures:
        cache = OUT_CACHE / f"vext_avg_T{T:.0f}K.npy"
        print(f"\n=== T = {T:.0f} K ===")
        if cache.exists():
            data = np.load(cache, allow_pickle=True).item()
            vext_avg = data["vext_avg"]
            shape = data["grid_shape"]
            dV = float(data["dV"])
            print(f"  Loaded cached Vext (shape {shape}, dV={dV:.4f} Å³)")
        else:
            print(f"  Building Vext on grid (50 orientations × ~30³ grid)...")
            data = build_vext_on_grid(
                host, co2, vtot,
                orientations=rots,
                spacing=0.7,                  # 0.5 → 0.7 Å cuts grid by ~2.7×
                pbc_supercell=(3, 3, 3),
                centre_supercell=True,
                temperature_K=T,
                cache_path=cache,
                # Reject orientations more attractive than the strongest Phase-1
                # validated binding (~ -25 kJ/mol). Without this, rigid EPM2
                # finds artefactual -80 to -150 kJ/mol orientations that
                # dominate the Boltzmann average everywhere in the pore.
                v_reject_below_K=-3000.0,
            )
            vext_avg = data["vext_avg"]
            shape = data["grid_shape"]
            dV = float(data["dV"])
            print(f"  Done. grid {shape}, dV={dV:.4f} Å³, "
                  f"Vmin={vext_avg.min()*K_TO_KJ_PER_MOL:+.2f} "
                  f"Vmax={vext_avg.max()*K_TO_KJ_PER_MOL:+.2f} kJ/mol")
        vext_grids[T] = (vext_avg, shape, dV)

        # Build accessibility mask once (CO2 centre must be ≥ 2.0 Å from every
        # host atom — otherwise we're inside a wall and Vext is artefactual).
        grid_xyz, _, _ = build_grid(host, spacing=0.7)
        grid_3d = grid_xyz.reshape(*shape, 3)
        nn_dist = np.full(shape, np.inf, dtype=float)
        for h in host_super.positions:
            dr = grid_3d - h
            r = np.sqrt(np.einsum("ijkd,ijkd->ijk", dr, dr))
            nn_dist = np.minimum(nn_dist, r)
        access = nn_dist >= 2.0
        print(f"  accessibility: {access.sum()}/{access.size} voxels "
              f"({100*access.mean():.1f}%) are ≥ 2.0 Å from any atom")
        # Also report Vext stats inside the accessible region
        v_in = vext_avg[access]
        if len(v_in):
            print(f"  Vext over accessible voxels (kJ/mol): "
                  f"min={v_in.min()*K_TO_KJ_PER_MOL:+7.2f} "
                  f"max={v_in.max()*K_TO_KJ_PER_MOL:+7.2f} "
                  f"median={np.median(v_in)*K_TO_KJ_PER_MOL:+6.2f}")

        iso_henry = compute_isotherm_henry(
            vext_avg_grid_K=vext_avg,
            dV_A3=dV,
            pressures_bar=pressures_bar,
            temperature_K=T,
            framework_mass_amu=framework_mass_amu,
            accessibility_mask=access,
            v_min_clip_K=-4000.0,
        )
        iso_lang = compute_isotherm_langmuir(
            vext_avg_grid_K=vext_avg,
            dV_A3=dV,
            pressures_bar=pressures_bar,
            temperature_K=T,
            framework_mass_amu=framework_mass_amu,
            accessibility_mask=access,
            v_excl_A3=57.0,            # ~ 4·(4π/3)(σ_CO2/2)³  with σ=3.017 Å
            v_min_clip_K=-4000.0,
        )
        isotherms[T] = {"henry": iso_henry, "langmuir": iso_lang}
        print(f"  Henry    @ 1 bar (mmol/g):  abs={iso_henry.loading_mmol_per_g_abs[-1]:.3f}  "
              f"exc={iso_henry.loading_mmol_per_g_exc[-1]:.3f}")
        print(f"  Langmuir @ 1 bar (mmol/g):  abs={iso_lang.loading_mmol_per_g_abs[-1]:.3f}  "
              f"exc={iso_lang.loading_mmol_per_g_exc[-1]:.3f}")

    # ----- Plot isotherms vs experiment -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, T in zip(axes, temperatures):
        h = isotherms[T]["henry"]
        l = isotherms[T]["langmuir"]
        ax.plot(h.pressures_bar, h.loading_mmol_per_g_abs,
                "b-", label="Henry (no saturation)", linewidth=2, alpha=0.8)
        ax.plot(l.pressures_bar, l.loading_mmol_per_g_abs,
                "g-", label="Langmuir cap", linewidth=2)
        if T in EXP_TARGETS:
            p_exp, n_exp = zip(*EXP_TARGETS[T])
            ax.plot(p_exp, n_exp, "ro", markersize=8, label="Evans 2022 (approx)")
        ax.set_xlabel("Pressure (bar)")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.2)
        ax.set_ylim(0, max(6.0, l.loading_mmol_per_g_abs[-1] * 1.2))
        ax.set_title(f"T = {T:.0f} K")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylabel("CO₂ loading (mmol / g framework)")
    fig.suptitle("Phase 2.1b — Henry vs Langmuir isotherm vs Evans 2022")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "13_phase2_baseline_isotherms.png", dpi=150)
    plt.close(fig)

    # 2D slice of Vext at T=298 K (skip if 298 not in temperatures)
    if 298.0 not in vext_grids:
        return
    vext_298, shape_298, _ = vext_grids[298.0]
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_vext_slice_2d(
        vext_298, axis="x", index=shape_298[0] // 2,
        extent=(0, a, 0, a),
        vmin_kJ_per_mol=-30, vmax_kJ_per_mol=+30,
        title=f"Orientation-averaged Vext slice ⟂ x at T=298 K", ax=ax,
    )
    ax.set_xlabel("y (Å)"); ax.set_ylabel("z (Å)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "14_phase2_vext_average_T298.png", dpi=150)
    plt.close(fig)

    # ----- Write CSV -----
    csv_path = OUT_RES / "phase2_baseline_isotherms.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["T_K", "p_bar", "model",
                    "N_per_cell_abs", "N_per_cell_exc",
                    "mmol_per_g_abs", "mmol_per_g_exc"])
        for T, ivar in isotherms.items():
            for model_name in ("henry", "langmuir"):
                iso = ivar[model_name]
                for i in range(len(iso.pressures_bar)):
                    w.writerow([
                        T, f"{iso.pressures_bar[i]:.5e}", model_name,
                        f"{iso.loading_N_per_cell_abs[i]:.5e}",
                        f"{iso.loading_N_per_cell_exc[i]:.5e}",
                        f"{iso.loading_mmol_per_g_abs[i]:.5f}",
                        f"{iso.loading_mmol_per_g_exc[i]:.5f}",
                    ])
    print(f"\nFigures: {OUT_FIG}/13_*.png, 14_*.png")
    print(f"Results: {csv_path}")
    print(f"\nPhase 2.1 baseline complete. Henry-regime CO2 isotherms saved.")
    print("Next: add hard-sphere FMT excluded volume to handle saturation,")
    print("then mean-field LJ for fluid-fluid attraction.")


if __name__ == "__main__":
    main()

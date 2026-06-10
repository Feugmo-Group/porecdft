"""Phase 2 final REPLOT — reuses cached Vext, sweeps K_eff for the soft-mode penalty,
and uses Fig.2A-faithful experimental targets.

The original Phase-2 final used the isotropic bulk modulus K=14 GPa (Evans).
That penalty is much too large to allow gate-opening at sub-1-bar CO2 loadings,
so Ω-minimisation picked ε*=0 everywhere. Real ALF gate-opening involves a
soft anisotropic mode with effective modulus ~10×–100× smaller than K_bulk.
Here we sweep K_eff ∈ {0.05, 0.5, 5, 14} GPa to show how the gate-opening
crossover emerges as the framework gets softer.
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

from applications.alf_co2 import ALF_CIF, CHARGES_CSV, FORCEFIELD_CSV, DATA_DIR
from applications.alf_co2.notebooks.phase1_vext_validation import _read_forcefield_csv
from porecdft.diagnostics import compute_isotherm_langmuir
from porecdft.diagnostics.isotherm import AVOGADRO
from porecdft.io import read_cif, read_charges_csv

OUT_FIG = DATA_DIR / "figures"
OUT_RES = DATA_DIR / "results"
OUT_CACHE = DATA_DIR / "results" / "vext_cache_flex"

ATOMIC_MASS = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}

# Updated EXP_TARGETS — faithful to Fig 2A of sciadv.ade1473 (digitised by eye
# from the published figure; values in (p_bar, mmol/g)).
# EXP_TARGETS imported from applications.alf_co2


def main():
    host0 = read_cif(ALF_CIF)
    host_ff = _read_forcefield_csv(FORCEFIELD_CSV)
    charges = read_charges_csv(CHARGES_CSV)
    host0 = host0.assign_charges(charges, source="Hirshfeld CP2K")
    framework_mass_amu = sum(ATOMIC_MASS[s] for s in host0.species)
    framework_mass_g = framework_mass_amu / AVOGADRO
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_g)
    V0 = host0.cell_volume

    strains = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    temperatures = [298.0, 273.0, 323.0]
    pressures_bar = np.logspace(-3, 0, 25)

    # Recover Vext + isotherms per (strain, T) from cache (built by phase2_final_summary.py)
    # Each cached file contains the orientation-averaged Vext on the grid.
    results = {T: {} for T in temperatures}
    for strain in strains:
        scale = 1.0 + strain
        # cell volume at this scale (isotropic expansion)
        V_L = V0 * scale ** 3
        # accessibility derived from atom positions at this scale
        from porecdft.vext import build_grid
        from porecdft.structure import build_supercell
        host_s = replace(
            host0, positions=host0.positions * scale, lattice=host0.lattice * scale,
        )
        host_super = build_supercell(host_s, 3, 3, 3)
        host_super = replace(
            host_super,
            positions=host_super.positions - host_s.lattice[0] - host_s.lattice[1] - host_s.lattice[2],
        )
        spacing = 0.7
        grid_xyz, shape, dV = build_grid(host_s, spacing)
        grid_3d = grid_xyz.reshape(*shape, 3)
        nn_dist = np.full(shape, np.inf)
        for h in host_super.positions:
            dr = grid_3d - h
            r = np.sqrt(np.einsum("ijkd,ijkd->ijk", dr, dr))
            nn_dist = np.minimum(nn_dist, r)
        access = nn_dist >= 2.0

        for T in temperatures:
            cache = OUT_CACHE / f"vext_avg_strain{strain:.3f}_T{T:.0f}K.npy"
            data = np.load(cache, allow_pickle=True).item()
            vext_avg = np.asarray(data["vext_avg"])
            iso = compute_isotherm_langmuir(
                vext_avg_grid_K=vext_avg, dV_A3=dV,
                pressures_bar=pressures_bar, temperature_K=T,
                framework_mass_amu=framework_mass_amu,
                accessibility_mask=access, v_excl_A3=57.0,
            )
            results[T][strain] = {"iso": iso, "V_L": V_L}

    # Sweep effective bulk modulus to show the gate-opening crossover
    K_eff_GPa_list = [0.05, 0.5, 5.0, 14.0]
    pa_to_K_per_A3 = 1.0e9 / 1.380649e-23 * 1.0e-30
    K_eff_K_per_A3 = {K: K * pa_to_K_per_A3 for K in K_eff_GPa_list}

    # Pre-compute Ω-min isotherm for each K_eff
    omega_min_curves: dict[float, dict[float, np.ndarray]] = {K: {T: None for T in temperatures} for K in K_eff_GPa_list}
    eps_star_curves: dict[float, dict[float, np.ndarray]] = {K: {T: None for T in temperatures} for K in K_eff_GPa_list}
    for K in K_eff_GPa_list:
        K_K_per_A3 = K_eff_K_per_A3[K]
        for T in temperatures:
            load = []
            eps_star = []
            for ip, p in enumerate(pressures_bar):
                Omegas = []
                for strain in strains:
                    N = results[T][strain]["iso"].loading_N_per_cell_abs[ip]
                    V_L = results[T][strain]["V_L"]
                    F_elast = 0.5 * K_K_per_A3 * V0 * ((V_L / V0) - 1.0) ** 2
                    Omega = -T * N + F_elast
                    Omegas.append(Omega)
                idx = int(np.argmin(Omegas))
                eps_star.append(strains[idx] * 100)
                load.append(results[T][strains[idx]]["iso"].loading_mmol_per_g_abs[ip])
            omega_min_curves[K][T] = np.array(load)
            eps_star_curves[K][T] = np.array(eps_star)

    # ---- Plot: 3 panels (T) × isotherm-vs-pressure with multiple K_eff overlaid ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    rigid_cmap = plt.get_cmap("Blues")
    rigid_colors = {s: rigid_cmap(0.3 + 0.6 * i / max(len(strains) - 1, 1)) for i, s in enumerate(strains)}
    keff_cmap = plt.get_cmap("autumn_r")
    keff_colors = {K: keff_cmap(i / max(len(K_eff_GPa_list) - 1, 1)) for i, K in enumerate(K_eff_GPa_list)}

    for ax, T in zip(axes, temperatures):
        for strain in strains:
            iso = results[T][strain]["iso"]
            ax.plot(iso.pressures_bar, iso.loading_mmol_per_g_abs,
                    color=rigid_colors[strain], alpha=0.5, lw=1.0,
                    label=f"rigid ε={strain * 100:.1f}%" if strain in (0.0, 0.05) else None)
        for K in K_eff_GPa_list:
            ax.plot(pressures_bar, omega_min_curves[K][T],
                    color=keff_colors[K], lw=2.5,
                    label=f"Ω-min K={K} GPa")
        if T in EXP_TARGETS:
            p_exp, n_exp = zip(*EXP_TARGETS[T])
            ax.plot(p_exp, n_exp, "ko", markersize=8, label="Evans Fig 2A")
        ax.set_xlabel("Pressure (bar)")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.2)
        ax.set_ylim(0, 5)
        ax.set_title(f"T = {T:.0f} K")
        ax.set_ylabel("CO₂ loading (mmol / g)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Phase 2 final (replot) — K_eff sweep + corrected Fig-2A targets")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "17_phase2_final_K_eff_sweep.png", dpi=150)
    plt.close(fig)

    # ---- Plot: ε*(p, T) for the SOFT K_eff = 0.05 GPa case ----
    fig, ax = plt.subplots(figsize=(8, 5))
    K_soft = 0.05
    for T in temperatures:
        ax.plot(pressures_bar, eps_star_curves[K_soft][T], "o-",
                label=f"T = {T:.0f} K", markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("Equilibrium lattice strain ε* (%)")
    ax.set_title(f"Ω-minimised strain vs pressure  (K_eff = {K_soft} GPa, soft-mode proxy)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "18_phase2_strain_vs_p_soft.png", dpi=150)
    plt.close(fig)

    # ---- Summary print + CSV ----
    print("\n=== Phase 2 final replot @ 1 bar ===")
    for T in temperatures:
        evans = next((n for p, n in EXP_TARGETS.get(T, []) if p > 0.9), None)
        print(f"\nT = {T:.0f} K   (Evans @1 bar ≈ {evans:.2f} mmol/g)")
        print(f"  rigid closed (ε=0%):     {results[T][0.0]['iso'].loading_mmol_per_g_abs[-1]:.3f}")
        for K in K_eff_GPa_list:
            eps_last = eps_star_curves[K][T][-1]
            load_last = omega_min_curves[K][T][-1]
            print(f"  Ω-min K={K:5.2f} GPa:    ε*={eps_last:4.1f}%  →  loading {load_last:.3f}")

    print(f"\nFigures: {OUT_FIG}/17_phase2_final_K_eff_sweep.png, 18_phase2_strain_vs_p_soft.png")


if __name__ == "__main__":
    main()

"""Phase 2 final summary — multi-state flexible host + joint Ω minimisation.

Scans lattice expansion strain ε ∈ {0, 0.5, 1, 2, 3, 5}% and computes the
Langmuir-capped isotherm at each (T, p, L). Then for each (T, p) finds the
L that minimises the grand potential

    Ω(L, p, T) = -k_B T · ln[ V_eff(L, p, T) ]  +  ½ · K_bulk · V₀ · (V(L)/V₀ - 1)²

with V_eff(L, p, T) = ∫ exp(-β V_ext(r; L)) ρ_bulk dV  + Langmuir saturation.

The first term favours expansion (more attractive volume), the second penalises
deformation (elastic energy). K_bulk = 14 GPa (Evans 2022 mechanical data).

Final figure (Fig. 17): 3 panels (T = 273, 298, 323 K), each showing all
five rigid branches + the Ω-minimised flexible result + Evans 2022.
"""
from __future__ import annotations

import csv
import sys
import time
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
from porecdft.diagnostics.isotherm import K_TO_KJ_PER_MOL, AVOGADRO
from porecdft.eos import density_from_pressure
from porecdft.fluid import EPM2_CO2
from porecdft.forcefield import (
    CompositePotential, CoulombPotential, LJPotential, QuadrupoleEFGPotential,
)
from porecdft.io import read_cif, read_charges_csv
from porecdft.structure import build_supercell
from porecdft.vext import build_vext_on_grid, build_grid, fibonacci_rotations

OUT_FIG = DATA_DIR / "figures"
OUT_RES = DATA_DIR / "results"
OUT_CACHE = DATA_DIR / "results" / "vext_cache_flex"
OUT_CACHE.mkdir(parents=True, exist_ok=True)

ATOMIC_MASS = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}

# EXP_TARGETS imported from applications.alf_co2

# ALF bulk modulus from Evans 2022 mechanical data — convert to K/Å³
K_BULK_GPA = 14.0
K_BULK_K_PER_A3 = K_BULK_GPA * 1.0e9 / 1.380649e-23 * 1.0e-30   # K/Å³


def make_scaled_host(host, scale: float):
    new_lat = host.lattice * scale
    new_pos = host.positions * scale
    return replace(host, positions=new_pos, lattice=new_lat,
                   source=host.source + f" [scale {scale:.4f}]")


def main():
    host0 = read_cif(ALF_CIF)
    host_ff = _read_forcefield_csv(FORCEFIELD_CSV)
    charges = read_charges_csv(CHARGES_CSV)
    host0 = host0.assign_charges(charges, source="Hirshfeld CP2K")
    framework_mass_amu = sum(ATOMIC_MASS[s] for s in host0.species)
    framework_mass_g = framework_mass_amu / AVOGADRO
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_g)
    V0 = host0.cell_volume                                # Å³ at L = L₀

    print(f"Framework: {host0.summary()}")
    print(f"V₀ = {V0:.2f} Å³, K_bulk = {K_BULK_GPA} GPa = {K_BULK_K_PER_A3:.2f} K/Å³")

    co2 = EPM2_CO2
    lj = LJPotential(host_ff=host_ff, fluid_ff=co2.ff, cutoff=15.0)
    coul = CoulombPotential(fluid_charges=co2.charges, cutoff=15.0,
                            method="smeared", gauss_width=2.0)
    quad = QuadrupoleEFGPotential(theta_zz=co2.theta_zz, cutoff=15.0)
    vtot = CompositePotential([lj, coul, quad])
    rots = fibonacci_rotations(20)
    spacing = 0.7

    strains = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    temperatures = [298.0, 273.0, 323.0]
    pressures_bar = np.logspace(-3, 0, 25)

    # results[T][strain] = {"iso": IsothermResult, "Vext_int": V_eff, ...}
    results: dict[float, dict[float, dict]] = {T: {} for T in temperatures}

    for strain in strains:
        scale = 1.0 + strain
        host_s = make_scaled_host(host0, scale)
        host_super = build_supercell(host_s, 3, 3, 3)
        shift = -host_s.lattice[0] - host_s.lattice[1] - host_s.lattice[2]
        host_super = replace(host_super, positions=host_super.positions + shift)
        V_L = host_s.cell_volume                          # Å³ at this L

        # Build accessibility mask
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
            print(f"\n--- strain ε={strain*100:.1f}%, T={T:.0f} K ---")
            if cache.exists():
                data = np.load(cache, allow_pickle=True).item()
                vext_avg = np.asarray(data["vext_avg"])
                print(f"  cached Vext (shape {vext_avg.shape})")
            else:
                t0 = time.time()
                data = build_vext_on_grid(
                    host_s, co2, vtot, orientations=rots, spacing=spacing,
                    pbc_supercell=(3, 3, 3), centre_supercell=True,
                    temperature_K=T, cache_path=cache,
                    v_reject_below_K=-3000.0,
                )
                vext_avg = np.asarray(data["vext_avg"])
                print(f"  built in {time.time() - t0:.1f}s,  "
                      f"Vmin={float(np.nanmin(vext_avg)) * K_TO_KJ_PER_MOL:+.2f} kJ/mol")
            iso = compute_isotherm_langmuir(
                vext_avg_grid_K=vext_avg, dV_A3=dV,
                pressures_bar=pressures_bar, temperature_K=T,
                framework_mass_amu=framework_mass_amu,
                accessibility_mask=access, v_excl_A3=57.0,
            )
            # Effective volume V_eff = ∫ exp(-βV) dV (low-p Henry integral) — used for Ω
            beta = 1.0 / T
            V = np.where(np.isfinite(vext_avg), vext_avg, +1e6)
            V = np.maximum(V, -4000.0)
            bV = np.clip(beta * V, -50.0, +50.0)
            V_eff = float(np.sum(np.exp(-bV) * access) * dV)
            results[T][strain] = {"iso": iso, "V_eff": V_eff, "V_L": V_L}

    # ---- Ω minimisation over L for each (T, p) ----
    # Ω(L, p, T) per unit cell:
    #   gas chemical-potential side: -k_B T · N(L, p, T)    (lower for more N)
    #   elastic side:                 ½ K_bulk · V₀ · (V_L/V₀ - 1)²
    # (units: K)
    for T in temperatures:
        omega_min_idx = []                # selected strain index per pressure
        loading_min: list[float] = []
        for ip, p in enumerate(pressures_bar):
            Omegas = []
            for strain in strains:
                r = results[T][strain]
                N = r["iso"].loading_N_per_cell_abs[ip]
                V_L = r["V_L"]
                F_elast = 0.5 * K_BULK_K_PER_A3 * V0 * ((V_L / V0) - 1.0) ** 2
                Omega = -T * N + F_elast
                Omegas.append(Omega)
            idx = int(np.argmin(Omegas))
            omega_min_idx.append(idx)
            loading_min.append(results[T][strains[idx]]["iso"].loading_mmol_per_g_abs[ip])
        results[T]["omega_min_strain_idx"] = omega_min_idx
        results[T]["loading_min"] = np.array(loading_min)

    # ---- Final summary plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.get_cmap("viridis")
    strain_colors = {s: cmap(i / max(len(strains) - 1, 1)) for i, s in enumerate(strains)}
    for ax, T in zip(axes, temperatures):
        for strain in strains:
            iso = results[T][strain]["iso"]
            ax.plot(iso.pressures_bar, iso.loading_mmol_per_g_abs,
                    color=strain_colors[strain], alpha=0.6, lw=1.5,
                    label=f"L=L₀×{1 + strain:.3f}")
        ax.plot(pressures_bar, results[T]["loading_min"],
                "k-", lw=3.0, label="Ω-min (flexible)")
        if T in EXP_TARGETS:
            p_exp, n_exp = zip(*EXP_TARGETS[T])
            ax.plot(p_exp, n_exp, "ro", markersize=10, label="Evans 2022")
        ax.set_xlabel("Pressure (bar)")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.2)
        ax.set_ylim(0, 6)
        ax.set_title(f"T = {T:.0f} K")
        ax.set_ylabel("CO₂ loading (mmol / g)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Phase 2 final — multi-strain rigid branches + Ω-minimised flexible host vs Evans 2022")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "17_phase2_final_multistrain.png", dpi=150)
    plt.close(fig)

    # ---- L*(p, T) plot — selected strain vs pressure ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for T in temperatures:
        eps_select = [strains[i] * 100 for i in results[T]["omega_min_strain_idx"]]
        ax.plot(pressures_bar, eps_select, "o-", label=f"T = {T:.0f} K", markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("Equilibrium lattice strain ε* (%)")
    ax.set_title("Ω-minimised lattice expansion vs pressure (flexible host)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "18_phase2_strain_vs_p.png", dpi=150)
    plt.close(fig)

    # ---- Save consolidated CSV ----
    csv_path = OUT_RES / "phase2_final_summary.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["T_K", "strain_pct", "p_bar",
                    "mmol_per_g_abs", "N_per_cell_abs",
                    "is_omega_min"])
        for T in temperatures:
            for strain in strains:
                iso = results[T][strain]["iso"]
                for ip, p in enumerate(pressures_bar):
                    is_min = int(strains[results[T]["omega_min_strain_idx"][ip]] == strain)
                    w.writerow([T, strain * 100, f"{p:.5e}",
                                f"{iso.loading_mmol_per_g_abs[ip]:.5f}",
                                f"{iso.loading_N_per_cell_abs[ip]:.5e}",
                                is_min])

    # ---- Summary print ----
    print("\n=== Phase 2 final summary @ 1 bar ===")
    print(f"{'T':>5}  {'strain(rigid)':>15}  {'loading(rigid)':>16}  "
          f"{'loading(Ω-min)':>16}  {'ε*':>8}  {'Evans':>7}")
    for T in temperatures:
        Nlast = results[T]["loading_min"][-1]
        eps_last = strains[results[T]["omega_min_strain_idx"][-1]]
        N0 = results[T][0.0]["iso"].loading_mmol_per_g_abs[-1]
        exp = EXP_TARGETS[T][-1][1]
        print(f"{T:>5.0f}  {'0% (closed)':>15}  {N0:>16.3f}  "
              f"{Nlast:>16.3f}  {eps_last*100:>7.1f}%  {exp:>7.2f}")
    print(f"\nFigures: {OUT_FIG}/17_*, 18_*")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

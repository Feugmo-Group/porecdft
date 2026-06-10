"""Phase 2.3 — Flexible host (gate-opening) extension.

The Evans 2022 isotherms show two anomalous features that a rigid framework
cannot capture:
  (1) sigmoidal step at 273 K — cooperative pore filling
  (2) at 1 bar, 323 K loading exceeds 273 K — endothermic gate-opening

Both are signatures of framework deformation under CO2 loading. The minimal
extension is **two-state framework**: closed (equilibrium L₀) and open (slightly
expanded L₁ ≈ L₀·1.01). Build Vext for each, integrate to get two isotherm
branches. At each (p, T) the system picks the branch with lower grand
potential:

    Ω(L, p, T) = F_id + F_HS + ∫ V_ext(r; L) ρ(r) dr − μ ∫ ρ dr + ½·K_bulk·(L − L₀)²

For now we show both branches side-by-side; the joint Ω minimisation is
straightforward to bolt on once Anderson convergence is in place.

Cell scaling: stretch all three lattice vectors isotropically by (1 + δ).
δ = 0.01 corresponds to ~1% volume expansion (typical for MOF gate-opening).
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
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.alf_co2 import ALF_CIF, CHARGES_CSV, FORCEFIELD_CSV, DATA_DIR
from applications.alf_co2.notebooks.phase1_vext_validation import _read_forcefield_csv
from porecdft.diagnostics import compute_isotherm_langmuir
from porecdft.diagnostics.isotherm import K_TO_KJ_PER_MOL, AVOGADRO
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


def make_scaled_host(host, scale: float):
    """Return a new HostAtoms with lattice (and atom positions) scaled by `scale`."""
    new_lat = host.lattice * scale
    # Fractional coords stay the same → Cartesian positions scale by `scale`
    new_pos = host.positions * scale
    return replace(host, positions=new_pos, lattice=new_lat,
                   source=host.source + f" [scale {scale:.4f}]")


def main():
    host_closed = read_cif(ALF_CIF)
    host_ff = _read_forcefield_csv(FORCEFIELD_CSV)
    charges = read_charges_csv(CHARGES_CSV)
    host_closed = host_closed.assign_charges(charges, source="Hirshfeld CP2K")
    framework_mass_amu = sum(ATOMIC_MASS[s] for s in host_closed.species)
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_amu / AVOGADRO)
    framework_mass_g = framework_mass_amu / AVOGADRO
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_g)
    print(f"Framework: {host_closed.summary()}")

    # Open-state lattice (isotropic expansion by 1%)
    open_scale = 1.01
    host_open = make_scaled_host(host_closed, open_scale)
    print(f"Open state: lattice × {open_scale} = "
          f"{open_scale * float(np.linalg.norm(host_closed.lattice[0])):.3f} Å")

    co2 = EPM2_CO2
    lj = LJPotential(host_ff=host_ff, fluid_ff=co2.ff, cutoff=15.0)
    coul = CoulombPotential(fluid_charges=co2.charges, cutoff=15.0,
                            method="smeared", gauss_width=2.0)
    quad = QuadrupoleEFGPotential(theta_zz=co2.theta_zz, cutoff=15.0)
    vtot = CompositePotential([lj, coul, quad])
    rots = fibonacci_rotations(20)
    spacing = 0.7

    temperatures = [298.0, 273.0, 323.0]
    pressures_bar = np.logspace(-3, 0, 25)

    results = {}  # results[T] = {"closed": iso, "open": iso}
    for T in temperatures:
        print(f"\n=== T = {T:.0f} K ===")
        T_results = {}
        for state_name, host_st in (("closed", host_closed), ("open", host_open)):
            cache = OUT_CACHE / f"vext_avg_{state_name}_T{T:.0f}K.npy"
            host_super = build_supercell(host_st, 3, 3, 3)
            shift = -host_st.lattice[0] - host_st.lattice[1] - host_st.lattice[2]
            host_super = replace(host_super, positions=host_super.positions + shift)
            if cache.exists():
                data = np.load(cache, allow_pickle=True).item()
                vext_avg = np.asarray(data["vext_avg"])
                shape = tuple(data["grid_shape"])
                dV = float(data["dV"])
                print(f"  {state_name}: cached Vext (shape {shape})")
            else:
                print(f"  {state_name}: building Vext...")
                t0 = time.time()
                data = build_vext_on_grid(
                    host_st, co2, vtot,
                    orientations=rots, spacing=spacing,
                    pbc_supercell=(3, 3, 3), centre_supercell=True,
                    temperature_K=T, cache_path=cache,
                    v_reject_below_K=-3000.0,
                )
                vext_avg = np.asarray(data["vext_avg"])
                shape = tuple(data["grid_shape"])
                dV = float(data["dV"])
                print(f"    done in {time.time() - t0:.1f}s, "
                      f"Vmin={float(np.nanmin(vext_avg)) * K_TO_KJ_PER_MOL:+.2f} kJ/mol")
            # Accessibility mask
            grid_xyz, _, _ = build_grid(host_st, spacing)
            grid_3d = grid_xyz.reshape(*shape, 3)
            nn_dist = np.full(shape, np.inf)
            for h in host_super.positions:
                dr = grid_3d - h
                r = np.sqrt(np.einsum("ijkd,ijkd->ijk", dr, dr))
                nn_dist = np.minimum(nn_dist, r)
            access = nn_dist >= 2.0

            iso = compute_isotherm_langmuir(
                vext_avg_grid_K=vext_avg, dV_A3=dV,
                pressures_bar=pressures_bar, temperature_K=T,
                framework_mass_amu=framework_mass_amu,
                accessibility_mask=access, v_excl_A3=57.0,
            )
            T_results[state_name] = iso
            print(f"  {state_name} @ 1 bar: {iso.loading_mmol_per_g_abs[-1]:.3f} mmol/g")
        results[T] = T_results

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, T in zip(axes, temperatures):
        iso_c = results[T]["closed"]
        iso_o = results[T]["open"]
        ax.plot(iso_c.pressures_bar, iso_c.loading_mmol_per_g_abs, "b-",
                label="closed (L₀)", linewidth=2)
        ax.plot(iso_o.pressures_bar, iso_o.loading_mmol_per_g_abs, "g-",
                label=f"open (×{open_scale})", linewidth=2)
        if T in EXP_TARGETS:
            p_exp, n_exp = zip(*EXP_TARGETS[T])
            ax.plot(p_exp, n_exp, "ro", markersize=8, label="Evans 2022")
        ax.set_xlabel("Pressure (bar)")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.2)
        ax.set_ylim(0, 6)
        ax.set_title(f"T = {T:.0f} K")
        ax.set_ylabel("CO₂ loading (mmol / g)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Phase 2.3 — Two-state flexible host (closed vs +1% open) vs Evans 2022")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "16_phase2_3_flexible_host.png", dpi=150)
    plt.close(fig)

    csv_path = OUT_RES / "phase2_3_flexible_host.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["T_K", "p_bar", "state", "mmol_per_g_abs", "mmol_per_g_exc"])
        for T, r in results.items():
            for state, iso in r.items():
                for i in range(len(iso.pressures_bar)):
                    w.writerow([T, f"{iso.pressures_bar[i]:.5e}", state,
                                f"{iso.loading_mmol_per_g_abs[i]:.5f}",
                                f"{iso.loading_mmol_per_g_exc[i]:.5f}"])
    print(f"\nFigure: {OUT_FIG}/16_phase2_3_flexible_host.png")
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()

"""Phase 2.2 — Self-consistent FMT (aWBII) isotherm for CO2/ALF.

Replaces the Langmuir cap from Phase 2.1b with rigorous fundamental-measure-theory
excluded volume. Each (p, T) point is now a self-consistent Picard iteration:

    ρ(r) = ρ_bulk · exp[ −β V_ext(r) + c¹_HS(r) − c¹_HS(ρ_bulk) ]

with c¹_HS computed from the aWBII functional via FFT convolutions of the weight
functions w2, w3, w2vec against ρ.

The Vext is the same as Phase 2.1b: EPM2 + smeared Coulomb σ=2.0 + Quadrupole-EFG,
reused from cache.
"""
from __future__ import annotations

import csv
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
import time

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
from porecdft.diagnostics.isotherm import K_TO_KJ_PER_MOL, AVOGADRO
from porecdft.eos import density_from_pressure
from porecdft.fluid import EPM2_CO2
from porecdft.forcefield import (
    CompositePotential, CoulombPotential, LJPotential, QuadrupoleEFGPotential,
)
from porecdft.io.forcefield import FFEntry
from porecdft.io import read_cif, read_charges_csv
from porecdft.structure import build_supercell
from porecdft.vext import build_vext_on_grid, build_grid, fibonacci_rotations
from porecdft.compute_config import ComputeConfig

OUT_FIG = DATA_DIR / "figures"
OUT_RES = DATA_DIR / "results"
OUT_CACHE = DATA_DIR / "results" / "vext_cache"

ATOMIC_MASS = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}

# EXP_TARGETS imported from applications.alf_co2

# Hard-sphere diameter for FMT. Using single-site equivalent σ_CO2 = 3.017 Å.
SIGMA_HS = 3.017


def main():
    host = read_cif(ALF_CIF)
    host_ff = _read_forcefield_csv(FORCEFIELD_CSV)
    charges = read_charges_csv(CHARGES_CSV)
    host = host.assign_charges(charges, source="GCS2009 REPEAT")
    print(host.summary())
    framework_mass_amu = sum(ATOMIC_MASS[s] for s in host.species)
    framework_mass_g = framework_mass_amu / AVOGADRO
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_g)
    print(f"Framework mass per unit cell: {framework_mass_amu:.1f} amu")

    host_super = build_supercell(host, 3, 3, 3)
    shift = -host.lattice[0] - host.lattice[1] - host.lattice[2]
    host_super = replace(host_super, positions=host_super.positions + shift)

    co2 = EPM2_CO2
    # LJ: single-site at CO2 center of mass (mTraFF σ=3.017 Å, ε=85.671 K).
    # This matches co2_cdft_final_v2 exactly: LJ uses one spherical site at COM,
    # so it has no orientation dependence and avoids repulsive O-site contacts.
    # The EPM2 C-site IS at body position (0,0,0), so mapping "C"→single-site FF
    # achieves this without any geometry change; the two "O" labels are simply
    # absent from fluid_ff and silently skipped by LJPotential.energy_grid.
    # β_sf = 1.41: fitted empirical ε scaling from the original notebook.
    beta_sf = 1.41
    _ss_ff = {"C": FFEntry("C", 3.017, 85.671, "mTraFF")}
    lj = LJPotential(host_ff=host_ff, fluid_ff=_ss_ff, cutoff=15.0,
                     epsilon_scale=beta_sf)
    # Direct Coulomb with MIC (minimum image convention) using the original 104-atom
    # unit cell. Passing the 3×3×3 supercell (2808 atoms) with cutoff=15 Å > a/2=5.68 Å
    # double-counts periodic images, breaking charge-neutral cancellation and giving
    # median accessible Vext ~ +33000 K. MIC + original cell gives each physical atom
    # exactly once at the nearest image, restoring ⟨V_Coul⟩_orient ≈ 0.
    coul = CoulombPotential(fluid_charges=co2.charges, cutoff=15.0,
                            method="direct",
                            host_override=host,      # original 104-atom unit cell
                            mic_lattice=host.lattice)
    quad = QuadrupoleEFGPotential(theta_zz=co2.theta_zz, cutoff=15.0)
    vtot = CompositePotential([lj, coul, quad])

    n_orient = 50                  # 50 orientations matches original notebook
    rots = fibonacci_rotations(n_orient)
    spacing = 0.7

    temperatures = [298.0, 273.0, 323.0]
    pressures_bar = np.logspace(-3, 0, 20)

    # Pre-grid setup
    grid_xyz, shape, dV = build_grid(host, spacing)
    grid_3d = grid_xyz.reshape(*shape, 3)
    nn_dist = np.full(shape, np.inf)
    for h in host_super.positions:
        dr = grid_3d - h
        r = np.sqrt(np.einsum("ijkd,ijkd->ijk", dr, dr))
        nn_dist = np.minimum(nn_dist, r)
    access = nn_dist >= 2.0
    print(f"Accessibility mask: {access.sum()}/{access.size} voxels ({100*access.mean():.1f}%)")
    V_cell = float(np.prod(shape) * dV)
    vext_runtime = {}
    for T in temperatures:
        print(f"\n=== T = {T:.0f} K ===")

        runtime = time.time()
        print(f"  Building Vext on grid from cpu...")
        cache = OUT_CACHE / f"vext_avg_T{T:.0f}K_cpu.npy"
        compute_config = ComputeConfig(use_warp    = False,
                                       warp_device = "cpu",
                                       jax_device  = "cpu",
                                       dtype       = "float64")
        data = build_vext_on_grid(
            host, co2, vtot,
            orientations=rots, spacing=spacing,
            pbc_supercell=(3, 3, 3), centre_supercell=True,
            temperature_K=T, cache_path=cache,
            v_reject_below_K=-10000.0,
            v_cap_above_K=5000,          # no upper cap — let +/- contributions cancel
            averaging="boltzmann",      # stable with O(50) orientations; Boltzmann needs O(500+)
            compute = compute_config
        )
        elapsed = time.time() - runtime 
        elapsed = time.time() - runtime 
        vext_runtime[f"cpu_T{T:.0f}K"] = [elapsed]
        vext_avg_cpu = np.asarray(data["vext_avg"])

        runtime = time.time()
        print(f"  Building Vext on grid from gpu...")
        cache = OUT_CACHE / f"vext_avg_T{T:.0f}K_gpu.npy"
        compute_config = ComputeConfig(use_warp    = True,
                                       warp_device = "gpu",
                                       jax_device  = "gpu",
                                       dtype       = "float64")
        data = build_vext_on_grid(
            host, co2, vtot,
            orientations=rots, spacing=spacing,
            pbc_supercell=(3, 3, 3), centre_supercell=True,
            temperature_K=T, cache_path=cache,
            v_reject_below_K=-10000.0,
            v_cap_above_K=5000,          # no upper cap — let +/- contributions cancel
            averaging="boltzmann",      # stable with O(50) orientations; Boltzmann needs O(500+)
            compute = compute_config
        )
        elapsed = time.time() - runtime 
        vext_runtime[f"gpu_T{T:.0f}K"] = [elapsed]
        vext_avg_gpu = np.asarray(data["vext_avg"])

        # Diagnostics — float32 (warp) vs float64 (numpy) precision difference
        # is expected to be O(0.01–1 K) for Vext values of O(100–5000 K).
        abs_diff = np.abs(vext_avg_cpu - vext_avg_gpu)
        print(f"  CPU vs GPU Vext comparison at T={T:.0f} K:")
        print(f"    max |diff|  = {abs_diff.max():.4f} K")
        print(f"    mean |diff| = {abs_diff.mean():.4f} K")
        print(f"    max |cpu|   = {np.abs(vext_avg_cpu).max():.1f} K")
        # Physically meaningful tolerance: 1 K absolute OR 0.1% relative.
        # Float32 introduces ~0.001–0.1 K errors; larger discrepancies indicate a bug.
        ok = np.allclose(vext_avg_cpu, vext_avg_gpu, atol=1.0, rtol=1e-3)
        print(f"    PASS (atol=1 K, rtol=0.1%): {ok}")
        assert ok, (
            f"CPU vs GPU Vext mismatch at T={T:.0f} K: "
            f"max |diff| = {abs_diff.max():.3f} K"
        )

    # store data 
    import pandas as pd
    print(vext_runtime)
    df = pd.DataFrame(vext_runtime)
    df.to_csv(OUT_RES / "vext_runtime.csv", index=False)
    
    # plot data
    df = pd.read_csv(OUT_RES / "vext_runtime.csv")
    COLORS = {"cpu": "blue", "gpu": "orange"}
    plt.figure(figsize=(8, 4))
    plt.bar(df.columns, df.iloc[0], color=[COLORS["cpu"] if "cpu" in c else COLORS["gpu"] for c in df.columns])
    plt.ylabel("Vext build time (s)")
    plt.title("Vext build time comparison (CPU vs GPU)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "vext_runtime_comparison.png", dpi=300)


if __name__ == "__main__":
    main()

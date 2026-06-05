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

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
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
from porecdft.functional import (
    make_k_grid, make_fmt_weights_hat,
    compute_weighted_densities, compute_c1, bulk_c1,
)
from porecdft.io import read_cif, read_charges_csv
from porecdft.solver import picard_solve, anderson_solve
from porecdft.structure import build_supercell
from porecdft.vext import build_vext_on_grid, build_grid, fibonacci_rotations

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
    host = host.assign_charges(charges, source="Hirshfeld CP2K")
    print(host.summary())
    framework_mass_amu = sum(ATOMIC_MASS[s] for s in host.species)
    framework_mass_g = framework_mass_amu / AVOGADRO
    to_mmol_per_g = 1000.0 / (AVOGADRO * framework_mass_g)
    print(f"Framework mass per unit cell: {framework_mass_amu:.1f} amu")

    host_super = build_supercell(host, 3, 3, 3)
    shift = -host.lattice[0] - host.lattice[1] - host.lattice[2]
    host_super = replace(host_super, positions=host_super.positions + shift)

    co2 = EPM2_CO2
    lj = LJPotential(host_ff=host_ff, fluid_ff=co2.ff, cutoff=15.0)
    coul = CoulombPotential(fluid_charges=co2.charges, cutoff=15.0,
                            method="smeared", gauss_width=2.0)
    quad = QuadrupoleEFGPotential(theta_zz=co2.theta_zz, cutoff=15.0)
    vtot = CompositePotential([lj, coul, quad])

    n_orient = 20
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

    # FFT k-grid + FMT weights (built once for all T, p)
    # Cell axes a,b,c with grid spacing a/Nx etc; for a non-orthorhombic cell we'd need a
    # more careful treatment, but ALF is orthorhombic ✓.
    lat = host.lattice
    dx = float(np.linalg.norm(lat[0])) / shape[0]
    dy = float(np.linalg.norm(lat[1])) / shape[1]
    dz = float(np.linalg.norm(lat[2])) / shape[2]
    KX, KY, KZ, K = make_k_grid(shape, dx, dy, dz)
    w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, SIGMA_HS)
    print(f"FMT setup: grid {shape}, dx={dx:.3f} dy={dy:.3f} dz={dz:.3f} Å, σ_HS={SIGMA_HS} Å")

    isotherms = {}
    for T in temperatures:
        cache = OUT_CACHE / f"vext_avg_T{T:.0f}K.npy"
        print(f"\n=== T = {T:.0f} K ===")
        if cache.exists():
            data = np.load(cache, allow_pickle=True).item()
            vext_avg = np.asarray(data["vext_avg"])
            print(f"  Loaded cached Vext (shape {vext_avg.shape})")
        else:
            print(f"  Building Vext on grid...")
            data = build_vext_on_grid(
                host, co2, vtot,
                orientations=rots, spacing=spacing,
                pbc_supercell=(3, 3, 3), centre_supercell=True,
                temperature_K=T, cache_path=cache,
                v_reject_below_K=-3000.0,
            )
            vext_avg = np.asarray(data["vext_avg"])

        # Clip Vext to physical range
        Vext_K = np.where(np.isfinite(vext_avg), vext_avg, +1e6)
        Vext_K = np.maximum(Vext_K, -4000.0)

        N_abs_arr = np.empty(len(pressures_bar))
        N_exc_arr = np.empty(len(pressures_bar))
        iter_arr = np.empty(len(pressures_bar), dtype=int)
        conv_arr = np.empty(len(pressures_bar), dtype=bool)
        t0 = time.time()
        def boltzmann_init(rho_b, beta):
            ri = rho_b * np.exp(np.clip(-beta * Vext_K, -50.0, 20.0)) * access
            rho_max = 0.45 * 6.0 / (np.pi * SIGMA_HS ** 3)
            return np.minimum(ri, rho_max)

        def c1_callable(rho_arr):
            wd = compute_weighted_densities(rho_arr, w2_hat, w3_hat, w2vec_hat, SIGMA_HS)
            return np.asarray(compute_c1(rho_arr, wd, w2_hat, w3_hat, w2vec_hat,
                                          SIGMA_HS, model="aWBII"))

        # Warm start: ρ from previous (lower) pressure
        rho = None
        for i, p in enumerate(pressures_bar):
            rho_bulk = density_from_pressure(p, T)
            c1_b = bulk_c1(rho_bulk, SIGMA_HS, model="aWBII")
            beta = 1.0 / T
            rho_init = rho if rho is not None else boltzmann_init(rho_bulk, beta)
            # Fallback to Boltzmann if warm-start looks unhealthy
            if rho is not None and (not np.isfinite(rho).all() or rho.sum() * dV * to_mmol_per_g > 50.0):
                rho_init = boltzmann_init(rho_bulk, beta)

            res = anderson_solve(
                rho_init=rho_init, rho_bulk=rho_bulk,
                Vext_K=Vext_K, temperature_K=T,
                c1_callable=c1_callable, c1_bulk=c1_b,
                m=6, beta=0.3, max_iter=800, tol=1e-4,
                accessibility_mask=access, log_clip=25.0,
                safeguard_alpha=0.02, picard_warmup=30, step_clip=2.0,
            )
            last_err = res.error_history[-1] if res.error_history else np.inf
            # If Anderson oscillated, stabilise with slow Picard from its last iterate
            if not res.converged and (not np.isfinite(last_err) or last_err > 0.1):
                res = picard_solve(
                    rho_init=res.rho if np.isfinite(last_err) else boltzmann_init(rho_bulk, beta),
                    rho_bulk=rho_bulk, Vext_K=Vext_K, temperature_K=T,
                    c1_callable=c1_callable, c1_bulk=c1_b,
                    alpha=0.005, max_iter=2000, tol=1e-3,
                    accessibility_mask=access, log_clip=25.0,
                )
                last_err = res.error_history[-1] if res.error_history else np.inf
            rho_solved = res.rho
            rho = rho_solved if (np.isfinite(last_err) and last_err < 0.5) else None
            N_abs = float(rho_solved.sum() * dV)
            N_exc = float((rho_solved - rho_bulk * access).sum() * dV)
            N_abs_arr[i] = N_abs; N_exc_arr[i] = N_exc
            iter_arr[i] = res.iterations; conv_arr[i] = res.converged
            if i in (0, len(pressures_bar) // 2, len(pressures_bar) - 1):
                print(f"  p={p:.4f} bar  it={res.iterations:3d}  "
                      f"conv={res.converged}  err={res.error_history[-1]:.2e}  "
                      f"N_abs={N_abs * to_mmol_per_g:.3f} mmol/g")
        elapsed = time.time() - t0
        print(f"  T={T:.0f} K done in {elapsed:.1f}s  ({iter_arr.mean():.0f} iters/point avg)")
        isotherms[T] = {
            "p": pressures_bar,
            "N_abs": N_abs_arr,
            "N_exc": N_exc_arr,
            "mmol_per_g_abs": N_abs_arr * to_mmol_per_g,
            "mmol_per_g_exc": N_exc_arr * to_mmol_per_g,
            "iters": iter_arr,
            "converged": conv_arr,
        }

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, T in zip(axes, temperatures):
        iso = isotherms[T]
        ax.plot(iso["p"], iso["mmol_per_g_abs"], "m-", label="FMT-aWBII (abs)", linewidth=2)
        ax.plot(iso["p"], iso["mmol_per_g_exc"], "m--", label="FMT-aWBII (excess)", alpha=0.7)
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
    fig.suptitle("Phase 2.2 — Self-consistent FMT-aWBII isotherm vs Evans 2022")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "15_phase2_2_fmt_isotherms.png", dpi=150)
    plt.close(fig)

    csv_path = OUT_RES / "phase2_2_fmt_isotherms.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["T_K", "p_bar", "N_per_cell_abs", "N_per_cell_exc",
                    "mmol_per_g_abs", "mmol_per_g_exc", "picard_iters", "converged"])
        for T, iso in isotherms.items():
            for i in range(len(iso["p"])):
                w.writerow([T, f"{iso['p'][i]:.5e}",
                            f"{iso['N_abs'][i]:.5e}", f"{iso['N_exc'][i]:.5e}",
                            f"{iso['mmol_per_g_abs'][i]:.5f}",
                            f"{iso['mmol_per_g_exc'][i]:.5f}",
                            iso["iters"][i], iso["converged"][i]])
    print(f"\nFigure: {OUT_FIG}/15_phase2_2_fmt_isotherms.png")
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()

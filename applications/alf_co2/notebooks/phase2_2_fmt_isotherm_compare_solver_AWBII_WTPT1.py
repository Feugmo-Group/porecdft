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
import itertools as it

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
from porecdft.functional import (
    make_k_grid, make_fmt_weights_hat,
    compute_weighted_densities, compute_c1, bulk_c1,
)
from porecdft.functional.association import WertheimAssociation
from porecdft.io import read_cif, read_charges_csv
from porecdft.solver import picard_solve, anderson_solve, fire2_solve, jax_solve
from porecdft.structure import build_supercell
from porecdft.vext import build_vext_on_grid, build_grid, fibonacci_rotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

OUT_FIG = DATA_DIR / "figures"
OUT_RES = DATA_DIR / "results"
OUT_CACHE = DATA_DIR / "results" / "vext_cache"

ATOMIC_MASS = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}

# EXP_TARGETS imported from applications.alf_co2

# Hard-sphere diameter for FMT. Using single-site equivalent σ_CO2 = 3.017 Å.
SIGMA_HS = 3.017

# parameters for WTPT1 functional
ENERGY_K = 400.0 # binding energy
KAPPA_A3 = 119.0 # SC volume

def compute_mass_per_cell(host):
    """
    compute mass of the unit cell
    atoms on the corner: Shared by 8 unit cells, contributing 1/8 per atom
    atoms on the face of the cell: Shared by 2 unit cells, contributing 1/2 per atom
    atoms on the edge: Shared by 4 unit cells, contributing 1/4 per atom
    atoms inside the cell: Contained entirely inside, contributing 1 full atom
    """

    def on_edge(x):
        return np.allclose(x, 0.) or np.allclose(x, 1.)

    frac_coord = []
    M = np.linalg.inv(host.lattice)
    for i, atom in enumerate(host.species):
         frac_coord.append((atom, np.dot(M, host.positions[i,:])))
    mass = 0.
    for atom, pos in frac_coord:
        edge = sum(on_edge(x) for x in pos)
        if edge == 0:
            weight = 1
        elif edge == 1:
            weight = 1./2.
        elif edge == 2:
            weight = 1./4.
        else:
            weight = 1./8.

        mass += weight * ATOMIC_MASS[atom]
    return mass

def main():
    host = read_cif(ALF_CIF)
    host_ff = _read_forcefield_csv(FORCEFIELD_CSV)
    charges = read_charges_csv(CHARGES_CSV)
    host = host.assign_charges(charges, source="GCS2009 REPEAT")
    print(host.summary())

    framework_mass_amu = compute_mass_per_cell(host)
    print(f"atomic mass {framework_mass_amu} amu")

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
    lj = LJPotential(host_ff=host_ff, fluid_ff=_ss_ff, cutoff=15.0 ,epsilon_scale=beta_sf)
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

    # initialize WTPT1 functional
    assoc = WertheimAssociation.from_host_element(host, "H", energy_K=ENERGY_K, kappa_A3=KAPPA_A3)

    data_iso = {}
    runtime ={}
    solvers = ["Anderson", "Adam", "Fire2"]
    if False:
        for method, T in it.product(solvers, temperatures):
            cache = OUT_CACHE / f"vext_avg_T{T:.0f}K.npy"
            print(f"\n=== T = {T:.0f} K === {method} method ===")
            vext_avg = None
            if cache.exists():
                data = np.load(cache, allow_pickle=True).item()
                vext_avg = np.asarray(data["vext_avg"])
                if vext_avg.shape != shape:
                    print(f"  Cache shape {vext_avg.shape} != grid {shape} — discarding stale cache")
                    cache.unlink()
                    data = None
                    vext_avg = None
                else:
                    print(f"  Loaded cached Vext (shape {vext_avg.shape})")
            if vext_avg is None:
                print(f"  Building Vext on grid...")
                data = build_vext_on_grid(
                     host, co2, vtot,
                     orientations=rots, spacing=spacing,
                     pbc_supercell=(3, 3, 3), centre_supercell=True,
                     temperature_K=T, cache_path=cache,
                     v_reject_below_K=-10000.0,
                     v_cap_above_K=None,          # no upper cap — let +/- contributions cancel
                     averaging="arithmetic",      # stable with O(50) orientations; Boltzmann needs O(500+)
                 )
                vext_avg = np.asarray(data["vext_avg"])

            # Clip Vext to physical range
            Vext_K = np.where(np.isfinite(vext_avg), vext_avg, +1e6)
            Vext_K = np.maximum(Vext_K, -4000.0)

            # Henry constant cross-check (Boltzmann integral, no FMT)
            kB_Pa_A3 = 1.380649e-23 * 1e30
            beta_T = 1.0 / T
            boltz_sum = np.exp(np.clip(-beta_T * Vext_K, -700, 700)).sum() * dV
            K_H_boltz = boltz_sum / (kB_Pa_A3 * T) * 1e5 / 6.022e23 * 1000 / framework_mass_g
            print(f"  K_H (Boltzmann integral, no FMT): {K_H_boltz:.5f} mmol/g/bar")
            accessible_vals = Vext_K[np.isfinite(Vext_K) & (Vext_K < 1e4)]
            vmax_acc = accessible_vals.max() if accessible_vals.size > 0 else float("nan")
            print(f"  Vext min={Vext_K.min():.0f} K  max(accessible)={vmax_acc:.0f} K")

            N_abs_arr = np.empty(len(pressures_bar))
            N_exc_arr = np.empty(len(pressures_bar))
            iter_arr = np.empty(len(pressures_bar), dtype=int)
            conv_arr = np.empty(len(pressures_bar), dtype=bool)
            t0 = time.time()
            def boltzmann_init(rho_b, beta):
                ri = rho_b * np.exp(np.clip(-beta * Vext_K, -50.0, 20.0)) * access
                rho_max = 0.45 * 6.0 / (np.pi * SIGMA_HS ** 3)
                return np.minimum(ri, rho_max)

            def c1_callable(rho_arr, output_type="jax"):
                wd = compute_weighted_densities(rho_arr, w2_hat, w3_hat, w2vec_hat, SIGMA_HS)
                c1_HS = compute_c1(rho_arr, wd, w2_hat, w3_hat, w2vec_hat,
                                              SIGMA_HS, model="aWBII")
                c1_assoc = assoc.c1_correction(rho_arr, grid_xyz, dV, T, use_warp=True)
                if output_type == "jax":
                    return c1_HS + c1_assoc
                elif output_type == "numpy":
                    return np.asarray(c1_HS + c1_assoc)
                else:
                    raise ValueError("Warning: output type unknown!")
            # Warm start: ρ from previous (lower) pressure
            rho = None
            for i, p in enumerate(pressures_bar):
                rho_bulk = density_from_pressure(p, T)
                c1_b = bulk_c1(rho_bulk, SIGMA_HS, model="aWBII")
                beta = 1.0 / T
                rho_init = rho if rho is not None else boltzmann_init(rho_bulk, beta)
                # Fallback to Boltzmann if warm-start looks unhealthy
                if rho is not None and (not np.isfinite(rho).all() or rho.sum() * dV * to_mmol_per_g > 50.0) or i == 0:
                    rho_init = boltzmann_init(rho_bulk, beta)

                # adam solver
                if method == "Adam":
                    res = jax_solve(
                     rho_init = rho_init,
                     rho_bulk = rho_bulk,
                     Vext_K = Vext_K,
                     temperature_K = T,
                     c1_callable = lambda rho: c1_callable(rho, output_type="jax"),
                     c1_bulk = c1_b,
                     dV = dV,
                     optimizer=optax.adam(1e-3),
                     n_steps = 4000,
                     tol = 1e-5,
                     accessibility_mask=access,
                     log_clip = 25.0,
                     print_every  = 0,
                     f_exc_mode = "quadrature",
                     n_quad = 4,
                 )

                elif method == "Fire2":
                    res = fire2_solve(
                        rho_init = rho_init,
                        rho_bulk = rho_bulk,
                        Vext_K = Vext_K,
                        temperature_K = T,
                        c1_callable = lambda rho: c1_callable(rho, "jax"),
                        c1_bulk = c1_b,
                        dV = dV,
                        accessibility_mask = access,
                        collect_history = False,
                        collect_max_steps = 1000,
                        max_steps = 2000,
                        f_exc_mode = "quadrature",
                        n_quad = 4,
                        rtol = 1e-6,
                        atol = 1e-4,
                    )

                elif method == "Anderson":
                    res = anderson_solve(
                        rho_init=rho_init, rho_bulk=rho_bulk,
                        Vext_K=Vext_K, temperature_K=T,
                        c1_callable = lambda rho: c1_callable(rho, output_type="numpy"),
                        c1_bulk=c1_b,
                        m=6, beta=0.3, max_iter=2000, tol=1e-6,
                        accessibility_mask=access, log_clip=25.0,
                        safeguard_alpha=0.02, picard_warmup=30, step_clip=2.0,
                        # Physical cap on local density — η < 0.45 packing fraction.
                        # Without this the deep SC well drives ρ above the
                        # hard-sphere close-packing limit → FMT log(1−n_3) → NaN
                        # → isotherm diverges at high P.  See solver/anderson.py.
                        rho_max=0.45 * 6.0 / (np.pi * SIGMA_HS ** 3),
                    )
                else:
                    raise ValueError("Solver method note implemented!")


                last_err = res.error_history[-1] if res.error_history else np.inf
                # If other oscillated, stabilise with slow Picard from its last iterate
                # if not res.converged:
                    # res = picard_solve(
                     # rho_init=res.rho if np.isfinite(last_err) else boltzmann_init(rho_bulk, beta),
                     # rho_bulk=rho_bulk, Vext_K=Vext_K, temperature_K=T,
                     # c1_callable=lambda rho: c1_callable(rho, output_type="numpy"),
                     # c1_bulk=c1_b,
                     # alpha=0.005, max_iter=2000, tol=1e-3,
                     # accessibility_mask=access, log_clip=25.0,
                     # rho_max=0.45 * 6.0 / (np.pi * SIGMA_HS ** 3),
                    # )
                    # last_err = res.error_history[-1] if res.error_history else np.inf
                rho_solved = res.rho
                rho = rho_solved if (np.isfinite(last_err) and last_err < 0.5) else None
                N_abs = float(rho_solved.sum() * dV)
                N_exc = float((rho_solved - rho_bulk * access).sum() * dV)
                N_abs_arr[i] = N_abs; N_exc_arr[i] = N_exc
                iter_arr[i] = res.iterations; conv_arr[i] = res.converged
                # if i in (0, len(pressures_bar) // 2, len(pressures_bar) - 1):

                print(f"  p={p:.4f} bar  it={res.iterations:3d}  "
                      f"conv={res.converged} "
                      # " err={res.error_history[-1]:.2e}  "
                      f"N_abs={N_abs * to_mmol_per_g:.3f} mmol/g")
            elapsed = time.time() - t0
            print(f"  T={T:.0f} K done in {elapsed:.1f}s  ({iter_arr.mean():.0f} iters/point avg)")
            isotherms = {
            "T_K": [T] * len(pressures_bar),
            "p_bar": pressures_bar,
            "N_abs": N_abs_arr,
            "N_exc": N_exc_arr,
            "mmol_per_g_abs": N_abs_arr * to_mmol_per_g,
            "mmol_per_g_exc": N_exc_arr * to_mmol_per_g,
            "iters": iter_arr,
            "converged": conv_arr,
            "time": [elapsed] * len(pressures_bar)
            }
            key = f"{T}K {method}"
            runtime[key] = elapsed
            data_iso[(method, T)] = isotherms

    # post pocess data (exclude divergege points)
    if True:
        import pandas as pd
        for method in solvers:
            tmp = []
            output =  OUT_RES / f"phase2_2_fmt_wtpt1_isotherms_{method}.csv"
            for T in  temperatures:
                if True:
                    df = pd.read_csv(output, index_col=0)
                    df = df[df["T_K"]==T]
                else:
                    df = pd.DataFrame(data_iso[(method, T)])
                data_iso[(method, T)] = df[df["converged"]]
                tmp.append(df)
            result = pd.concat(tmp, ignore_index=True)
            result.to_csv(output)

    # ---- Plot ----
    COLORS  = {"Picard": "#d6604d", "Anderson": "#ff7f0e",
                "Adam": "#2ca02c",  "Fire2": "#8073ac"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, T in zip(axes[:4], temperatures):
        if T in EXP_TARGETS:
            p_exp, n_exp = zip(*EXP_TARGETS[T])
            ax.plot(p_exp, n_exp, "ro", markersize=8, label="Evans 2022")
    times = []
    names = []
    for method in solvers:
        for idx, T in enumerate(temperatures):
            axes[idx].plot(data_iso[(method, T)]["p_bar"], data_iso[(method, T)]["mmol_per_g_abs"], "-", label=f"FMT-aWBII+WTPT1 (abs) {method}", alpha=0.7, color=COLORS[method])
            axes[idx].plot(data_iso[(method, T)]["p_bar"], data_iso[(method, T)]["mmol_per_g_exc"], "--", label=f"FMT-aWBII+WTPT1 (excess) {method}", alpha=0.7, color=COLORS[method])

            axes[idx].set_xlabel("Pressure (bar)")
            # es[idx]ax.set_xscale("log")
            axes[idx].set_xlim(1e-3, 1.2)
            axes[idx].set_ylim(0, 6)
            axes[idx].set_title(f"T = {T:.0f} K")
            axes[idx].set_ylabel("CO₂ loading (mmol / g)")
            axes[idx].grid(alpha=0.3)
            axes[idx].legend(fontsize=8, loc="upper left")
            print(data_iso[(method,T)].keys())
            print(data_iso[(method, T)]["time"].iloc[0])
            times.append(data_iso[(method, T)]["time"].iloc[0])
            names.append(f"{method} at {T:.1f} K")


    # COLORS = {"Picard": (0.1, 0.1, 0.1), "Anderson": (0.2, 0.2, 0.2), "Adam":(0.4, 0.4, 0.4), "Fire2":(0.8, 0.8, 0.8)}

    axes[3].bar(names, times,  color = [COLORS[method] for method, _ in it.product(solvers, temperatures)] )
    # axes[3].set_xlabel("Method")
    axes[3].set_ylabel("Run time (s)")
    axes[3].set_xticklabels(rotation=45, ha='right', labels=names)
    # , color=["red", "red", "blue", "blue", "green", "green"])

    fig.suptitle("Phase 2.2 — Self-consistent FMT-aWBII+WTPT1 isotherm benchmark")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "15_phase2_2_fmt_wtpt1_isotherms_compare.png", dpi=150)
    plt.close(fig)

    print(f"\nFigure: {OUT_FIG}/15_phase2_2_fmt_wtpt1_isotherms_comparison.png")


if __name__ == "__main__":
    main()

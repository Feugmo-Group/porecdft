"""Phase 1.3 — Validate Vext at the small (SC) and large (LC) cavities of ALF
against the Evans et al. DFT binding energies (-18.4 / -8.1 kJ/mol).

Pipeline:
    1. Load alf.cif, assign DDEC6-ish partial charges, verify neutrality.
    2. Build the EPM2 CO₂ fluid and a CompositePotential (LJ + Coulomb + Quadrupole-EFG).
    3. Use a single-site LJ probe on a coarse grid to locate pore centres
       (low-Vext local minima → candidate SC and LC).
    4. Probe each candidate with full EPM2 CO₂ over 100 Fibonacci orientations.
    5. Save diagnostic plots: 2D Vext slices, orientation roses, decomposition bars,
       orientation histograms, plus a summary CSV.

All output goes to ``applications/alf_co2/figures/`` and ``applications/alf_co2/results/``.
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

# Make `porecdft` and `applications` importable when running the script directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

# silence pymatgen's noisy CIF warning
warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.alf_co2 import (
    ALF_CIF, CHARGES_CSV, FORCEFIELD_CSV, DFT_BINDING_KJ_PER_MOL,
    DATA_DIR,
)
from porecdft.diagnostics import probe_binding_site
from porecdft.diagnostics.binding_site import K_TO_KJ_PER_MOL
from porecdft.fluid import EPM2_CO2, SingleSiteLJ_CO2
from porecdft.forcefield import (
    CompositePotential, CoulombPotential, LJPotential, QuadrupoleEFGPotential,
)
from porecdft.io import read_charges_csv, read_cif, read_forcefield_dat
from porecdft.io.forcefield import FFEntry
from porecdft.plotting import (
    plot_binding_rose, plot_orientation_histogram, plot_part_decomposition,
    plot_vext_slice_2d,
)
from porecdft.structure import build_supercell
from porecdft.structure.sites import find_local_minima
from porecdft.vext import build_grid, fibonacci_rotations

OUT_FIG = DATA_DIR / "figures"
OUT_RES = DATA_DIR / "results"
OUT_FIG.mkdir(exist_ok=True)
OUT_RES.mkdir(exist_ok=True)


def _read_forcefield_csv(path: Path) -> dict[str, FFEntry]:
    """Parse the CSV forcefield.csv we ship with the application."""
    ff: dict[str, FFEntry] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            el = row["element"].strip()
            ff[el] = FFEntry(
                element=el,
                sigma=float(row["sigma_A"]),
                epsilon=float(row["epsilon_K"]),
                source=row.get("source", ""),
            )
    return ff


def main() -> None:
    # ----- 1. Load host -----
    host = read_cif(ALF_CIF)
    host_ff = _read_forcefield_csv(FORCEFIELD_CSV)
    charges = read_charges_csv(CHARGES_CSV)
    host = host.assign_charges(charges, source="charges.csv (LJ-cDFT placeholder)")
    print(host.summary())
    print(f"  Net charge: {host.neutrality_residual():.2e} e (should be ≪ 1e-6)")
    print(f"  Charge source: {host.charge_source}")
    print(f"  Lattice (Å):\n{host.lattice}")
    a = float(np.linalg.norm(host.lattice[0]))

    # Replicate host once so PBC images sit inside the cutoff; centre on origin cell.
    host_super = build_supercell(host, 3, 3, 3)
    shift = -host.lattice[0] - host.lattice[1] - host.lattice[2]
    from dataclasses import replace
    host_super = replace(host_super, positions=host_super.positions + shift)

    # ----- 2. Build CO₂ EPM2 + composite potential -----
    co2 = EPM2_CO2
    lj = LJPotential(host_ff=host_ff, fluid_ff=co2.ff, cutoff=15.0)
    coul = CoulombPotential(fluid_charges=co2.charges, cutoff=15.0)
    quad = QuadrupoleEFGPotential(theta_zz=co2.theta_zz, cutoff=15.0)
    vtot = CompositePotential([lj, coul, quad])
    print(f"\nPotential: {vtot.name}")
    print(f"Fluid: {co2.name} with {co2.n_sites} sites, Θ_zz = {co2.theta_zz:+.4f} e·Å²")

    # ----- 3. Find pore centres via single-site LJ probe -----
    print("\nScanning unit cell with single-site LJ CO₂ probe to locate pores...")
    probe = LJPotential(host_ff=host_ff, fluid_ff=SingleSiteLJ_CO2.ff, cutoff=15.0)
    grid_xyz, shape, dV = build_grid(host, spacing=0.4)
    # one rotation suffices for spherical probe
    R0 = np.eye(3)
    v_probe = probe.energy_grid(
        grid_xyz, R0, host_super,
        SingleSiteLJ_CO2.body_sites, SingleSiteLJ_CO2.site_labels,
    ).reshape(shape)
    print(f"  Grid {shape}, dV={dV:.3f} Å³, "
          f"Vmin={v_probe.min()*K_TO_KJ_PER_MOL:+.2f} kJ/mol, "
          f"Vmax={v_probe.max()*K_TO_KJ_PER_MOL:+.2f} kJ/mol")

    # Multi-slice plot at four z values to see ALL pores in the cell
    extent = (0.0, a, 0.0, a)
    z_fracs = [0.0, 0.25, 0.5, 0.75]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, zf in zip(axes, z_fracs):
        idx = int(zf * shape[2]) % shape[2]
        plot_vext_slice_2d(
            v_probe, axis="z", index=idx, extent=extent,
            vmin_kJ_per_mol=-30.0, vmax_kJ_per_mol=+30.0,
            title=f"z = {zf:.2f}·a ≈ {zf*a:.2f} Å", ax=ax,
        )
        ax.set_xlabel("x (Å)")
        if ax is axes[0]:
            ax.set_ylabel("y (Å)")
    fig.suptitle("Single-site LJ probe Vext (kJ/mol) at four z-planes — pores in blue")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "01_vext_probe_multi_slice.png", dpi=150)
    plt.close(fig)

    # ----- 4. Pick candidate sites -----
    # Distance-to-nearest-atom for every grid point — we need this to filter out
    # grid minima that are sitting next to a framework atom (artefacts of the
    # LJ tail).
    print("\nComputing nearest-atom distance for each grid point...")
    grid_3d = grid_xyz.reshape(*shape, 3)
    nn_dist = np.full(shape, np.inf, dtype=float)
    for h in host_super.positions:
        dr = grid_3d - h
        r = np.sqrt(np.einsum("ijkd,ijkd->ijk", dr, dr))
        nn_dist = np.minimum(nn_dist, r)
    print(f"  nearest-atom distance: min={nn_dist.min():.2f} Å, "
          f"median={np.median(nn_dist):.2f} Å, max={nn_dist.max():.2f} Å")
    min_pore_radius = 2.5  # Å — must be at least this far from any atom

    # grid local minima, then filter
    mins = find_local_minima(v_probe)
    energies = v_probe[tuple(mins.T)]
    dists = nn_dist[tuple(mins.T)]
    keep = dists >= min_pore_radius
    mins = mins[keep]
    energies = energies[keep]
    dists = dists[keep]
    order = np.argsort(energies)
    mins, energies, dists = mins[order], energies[order], dists[order]

    print(f"\nLocal minima ≥ {min_pore_radius} Å from any atom "
          f"({len(mins)} after filtering):")
    cand_positions: list[np.ndarray] = []
    cand_labels: list[str] = []
    seen_positions: list[np.ndarray] = []
    for idx_ijk, e, d in zip(mins[:50], energies[:50], dists[:50]):
        frac = idx_ijk / np.array(shape)
        cart = frac @ host.lattice
        # de-dup symmetry-equivalent sites that are close in Å
        is_new = True
        for sp in seen_positions:
            dr = cart - sp
            # min image
            frac_dr = dr @ np.linalg.inv(host.lattice)
            frac_dr -= np.round(frac_dr)
            if np.linalg.norm(frac_dr @ host.lattice) < 2.0:
                is_new = False
                break
        if not is_new:
            continue
        seen_positions.append(cart)
        print(f"  frac={np.round(frac, 3)}  cart={cart.round(3)}  "
              f"E={e * K_TO_KJ_PER_MOL:+6.2f} kJ/mol  nn_dist={d:.2f} Å")
        if len(seen_positions) >= 8:
            break

    # Hand-picked high-symmetry candidates for Im-3m ALF. We probe these in
    # addition to the grid minima — sometimes the cavity centre falls between
    # grid points.
    high_sym = {
        "(1/2,1/2,1/2)": np.array([0.5, 0.5, 0.5]) @ host.lattice,
        "(1/2,1/2,0)":   np.array([0.5, 0.5, 0.0]) @ host.lattice,
        "(1/2,0,0)":     np.array([0.5, 0.0, 0.0]) @ host.lattice,
        "(1/4,1/4,3/4)": np.array([0.25, 0.25, 0.75]) @ host.lattice,
        "(1/4,1/2,1/2)": np.array([0.25, 0.5, 0.5]) @ host.lattice,
        "(0,1/4,1/2)":   np.array([0.0, 0.25, 0.5]) @ host.lattice,
    }
    print("\nProbing high-symmetry candidates with single-site LJ:")
    for label, r in high_sym.items():
        e_K = float(probe.energy_at(
            r, R0, host_super, SingleSiteLJ_CO2.body_sites, SingleSiteLJ_CO2.site_labels,
        ).total)
        # nearest atom distance
        dr = host_super.positions - r
        r_nn = float(np.sqrt(np.einsum("ad,ad->a", dr, dr)).min())
        print(f"  {label:>15s}  cart={r.round(3)}  "
              f"E_LJ={e_K * K_TO_KJ_PER_MOL:+6.2f} kJ/mol  nn={r_nn:.2f} Å")

    # Combine grid-minima candidates + high-sym candidates and rank
    all_candidates: list[tuple[str, np.ndarray]] = []
    for k, (sp, e) in enumerate(zip(seen_positions, energies[:len(seen_positions)])):
        all_candidates.append((f"grid_{k}", sp))
    for label, r in high_sym.items():
        all_candidates.append((label, r))
    print(f"\n→ {len(all_candidates)} candidate sites total. "
          f"Ranking by single-site LJ to pick top 2 distinct (≥ 3 Å apart)...")
    ranked = []
    for label, r in all_candidates:
        e_K = float(probe.energy_at(
            r, R0, host_super,
            SingleSiteLJ_CO2.body_sites, SingleSiteLJ_CO2.site_labels,
        ).total)
        dr = host_super.positions - r
        r_nn = float(np.sqrt(np.einsum("ad,ad->a", dr, dr)).min())
        if r_nn < min_pore_radius:
            continue
        ranked.append((e_K, label, r))
    ranked.sort()  # most negative first

    cand_positions = []
    cand_labels = []
    for e_K, label, r in ranked:
        # de-dup
        is_new = True
        for r0 in cand_positions:
            frac_dr = (r - r0) @ np.linalg.inv(host.lattice)
            frac_dr -= np.round(frac_dr)
            if np.linalg.norm(frac_dr @ host.lattice) < 3.0:
                is_new = False; break
        if not is_new:
            continue
        new_label = "SC" if len(cand_positions) == 0 else "LC"
        print(f"  Chose {new_label} = {label}  cart={r.round(3)}  "
              f"E_LJ={e_K * K_TO_KJ_PER_MOL:+6.2f} kJ/mol")
        cand_positions.append(r)
        cand_labels.append(new_label)
        if len(cand_positions) == 2:
            break

    # ----- 5. Probe each candidate with full EPM2 + (LJ + Coul + Quad) -----
    n_orient = 100
    rots = fibonacci_rotations(n_orient)

    # 5a. Diagnostic — probe each candidate with sub-potentials separately so
    # we can see whether LJ, Coulomb, or Quad is responsible for any anomaly.
    print("\n=== Sub-potential diagnostic at each candidate ===")
    # Variant: zero-out Al LJ (common GCMC convention for metal centres in oxide MOFs)
    host_ff_noAl = dict(host_ff)
    host_ff_noAl["Al"] = FFEntry("Al", host_ff["Al"].sigma, 0.0, host_ff["Al"].source + " (ε=0)")
    lj_noAl = LJPotential(host_ff=host_ff_noAl, fluid_ff=co2.ff, cutoff=15.0)
    coul_wolf = CoulombPotential(fluid_charges=co2.charges, cutoff=15.0, method="wolf")
    sub_pots = {
        "LJ only":              CompositePotential([lj]),
        "LJ (Al ε=0)":          CompositePotential([lj_noAl]),
        "Coul direct":          CompositePotential([coul]),
        "Coul Wolf":            CompositePotential([coul_wolf]),
        "LJ + Coul (direct)":   CompositePotential([lj, coul]),
        "LJ + Coul (Wolf)":     CompositePotential([lj, coul_wolf]),
        "LJ + Quad":            CompositePotential([lj, quad]),
        "LJ + Coul + Quad (W)": CompositePotential([lj, coul_wolf, quad]),
    }
    diag = {label: {} for label in cand_labels}
    for site_label, r0 in zip(cand_labels, cand_positions):
        print(f"-- {site_label} at {r0.round(3)} Å --")
        for sub_name, sub_pot in sub_pots.items():
            res_sub = probe_binding_site(
                host_super, co2, sub_pot, r0, rots,
                site_label=site_label, temperature_K=298.15,
            )
            diag[site_label][sub_name] = res_sub.E_min_kJ_per_mol
            print(f"    {sub_name:>18s}  E_min = {res_sub.E_min_kJ_per_mol:+8.2f} kJ/mol")

    # 5a-2. Save the diagnostic as a model-comparison bar chart per site.
    fig, axes = plt.subplots(1, len(cand_labels), figsize=(5*len(cand_labels), 4), squeeze=False)
    for ax, site_label in zip(axes[0], cand_labels):
        names = list(sub_pots.keys())
        vals = [diag[site_label][n] for n in names]
        bars = ax.bar(range(len(names)), vals, color="steelblue")
        ax.axhline(DFT_BINDING_KJ_PER_MOL[site_label], color="red", lw=2, ls="--",
                   label=f"DFT ref {DFT_BINDING_KJ_PER_MOL[site_label]:+.1f}")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("E_min (kJ/mol)")
        ax.set_title(f"{site_label} model sweep")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:+.1f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "05_model_sweep.png", dpi=150)
    plt.close(fig)

    # Save the sub-potential diagnostic to CSV
    sub_csv = OUT_RES / "phase1_subpotential_diagnostic.csv"
    with sub_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["site"] + list(sub_pots.keys()) + ["DFT_ref"])
        for s in cand_labels:
            w.writerow(
                [s] + [f"{diag[s][k]:.3f}" for k in sub_pots] + [DFT_BINDING_KJ_PER_MOL[s]]
            )
    print(f"\nSub-potential diagnostic written: {sub_csv}")

    # 5b. Full probe with composite potential (kept for downstream plots)
    results = []
    for label, r0 in zip(cand_labels, cand_positions):
        print(f"\nProbing {label} with EPM2 CO₂ over {n_orient} orientations...")
        res = probe_binding_site(
            host_super, co2, vtot, r0, rots,
            site_label=label, temperature_K=298.15,
        )
        results.append(res)
        print(f"  E_min  = {res.E_min_kJ_per_mol:+.2f} kJ/mol "
              f"(reference {DFT_BINDING_KJ_PER_MOL[label]:+.1f}; "
              f"Δ = {res.E_min_kJ_per_mol - DFT_BINDING_KJ_PER_MOL[label]:+.2f})")
        print(f"  E_mean = {res.E_mean_K * K_TO_KJ_PER_MOL:+.2f} kJ/mol")
        print(f"  E_max  = {res.E_max_K * K_TO_KJ_PER_MOL:+.2f} kJ/mol")
        print(f"  Boltzmann avg @ 298 K = "
              f"{res.boltzmann_average_K(298.15) * K_TO_KJ_PER_MOL:+.2f} kJ/mol")
        print(f"  Decomposition at min orientation (kJ/mol):")
        for k, v in res.parts_at_min.items():
            print(f"    {k:>10s} = {v * K_TO_KJ_PER_MOL:+.2f}")

    # ----- 6. Plots -----
    # 6a. orientation roses
    for res in results:
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(111, projection="polar")
        plot_binding_rose(
            res, dft_reference_kJ_per_mol=DFT_BINDING_KJ_PER_MOL[res.site_label],
            title=f"Vext orientations @ {res.site_label}", ax=ax,
        )
        fig.tight_layout()
        fig.savefig(OUT_FIG / f"02_rose_{res.site_label}.png", dpi=150)
        plt.close(fig)

    # 6b. histograms
    for res in results:
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_orientation_histogram(res, n_bins=25, ax=ax)
        ax.axvline(DFT_BINDING_KJ_PER_MOL[res.site_label], color="red", lw=2,
                   label=f"DFT ref {DFT_BINDING_KJ_PER_MOL[res.site_label]:+.1f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_FIG / f"03_hist_{res.site_label}.png", dpi=150)
        plt.close(fig)

    # 6c. decomposition
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_part_decomposition(
        results,
        dft_references_kJ_per_mol={"SC": -18.4, "LC": -8.1},
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG / "04_decomposition.png", dpi=150)
    plt.close(fig)

    # ----- 7. Summary CSV -----
    summary_path = OUT_RES / "phase1_vext_validation.csv"
    with summary_path.open("w") as f:
        w = csv.writer(f)
        w.writerow([
            "site", "x_A", "y_A", "z_A",
            "E_min_kJ_per_mol", "E_mean_kJ_per_mol", "E_max_kJ_per_mol",
            "E_boltz298_kJ_per_mol", "DFT_ref_kJ_per_mol", "delta_kJ_per_mol",
            "LJ_kJ_per_mol", "Coulomb_kJ_per_mol", "Quad_kJ_per_mol",
        ])
        for res in results:
            x, y, z = res.r_center
            parts = {k.split("#")[0]: v * K_TO_KJ_PER_MOL for k, v in res.parts_at_min.items()}
            ref = DFT_BINDING_KJ_PER_MOL[res.site_label]
            w.writerow([
                res.site_label, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",
                f"{res.E_min_kJ_per_mol:.3f}",
                f"{res.E_mean_K * K_TO_KJ_PER_MOL:.3f}",
                f"{res.E_max_K * K_TO_KJ_PER_MOL:.3f}",
                f"{res.boltzmann_average_K(298.15) * K_TO_KJ_PER_MOL:.3f}",
                f"{ref:.3f}",
                f"{res.E_min_kJ_per_mol - ref:+.3f}",
                f"{parts.get('LJ', 0.0):.3f}",
                f"{parts.get('Coulomb', 0.0):.3f}",
                f"{parts.get('Quad', 0.0):.3f}",
            ])
    print(f"\nSummary written: {summary_path}")
    print(f"Figures written to: {OUT_FIG}")


if __name__ == "__main__":
    main()

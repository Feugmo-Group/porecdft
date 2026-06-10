"""Phase 1.3c — Extract CO2 positions from the CP2K-optimised + thermalised
single snapshots in 2_NVT_abinitio_MD/ALF-S{1,2,3}/ALF-S{1,2,3}-{n}CO2_NVT_10ps.xyz.

For each loading n = 1, 2, 3, 4 CO2 we get the DFT-equilibrated positions of
all n molecules. These are *the* ground-truth binding sites: where DFT-D3
chooses to put a CO2 molecule at given loading. Comparing them to our
single-site LJ probe minima validates (or corrects) our SC/LC identifications.

We also probe each DFT-given CO2 centre with our porecdft potential to compare
the binding energy at the "true" site vs at our LJ-probe candidates.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.alf_co2 import (
    ALF_CIF, CHARGES_CSV, FORCEFIELD_CSV, DFT_BINDING_KJ_PER_MOL, DATA_DIR,
)
from porecdft.diagnostics import probe_binding_site
from porecdft.diagnostics.binding_site import K_TO_KJ_PER_MOL
from porecdft.fluid import EPM2_CO2
from porecdft.forcefield import (
    CompositePotential, CoulombPotential, LJPotential, QuadrupoleEFGPotential,
)
from porecdft.io import read_cif, read_charges_csv
from porecdft.io.forcefield import FFEntry
from porecdft.structure import build_supercell
from porecdft.vext import fibonacci_rotations

import csv

OUT_FIG = DATA_DIR / "figures"
OUT_RES = DATA_DIR / "results"
OUT_FIG.mkdir(exist_ok=True)
OUT_RES.mkdir(exist_ok=True)

AIMD_BASE = Path(
    "/Users/cgtetsas/Library/CloudStorage/OneDrive-UniversityofWaterloo/UWaterloo/"
    "Manuscript/2026/ALF/cDFT/ALF-CO2-N2_Manuscript_data/CP2K_calculations/"
    "3_abinitio_MD/2_NVT_abinitio_MD"
)


def read_xyz(path: Path) -> tuple[list[str], np.ndarray, float]:
    """Return (species, positions(N,3) in Å, energy_Ha if found in comment).
    CP2K comment line: "i = N, time = T, E = E_au"
    """
    lines = path.read_text().splitlines()
    natoms = int(lines[0].strip())
    comment = lines[1]
    e_au = float("nan")
    if "E =" in comment:
        e_au = float(comment.split("E =")[1].strip().split()[0])
    species, positions = [], []
    for ln in lines[2:2 + natoms]:
        toks = ln.split()
        species.append(toks[0])
        positions.append([float(toks[1]), float(toks[2]), float(toks[3])])
    return species, np.array(positions), e_au


def _read_forcefield_csv(path: Path) -> dict[str, FFEntry]:
    ff: dict[str, FFEntry] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            ff[row["element"].strip()] = FFEntry(
                row["element"].strip(),
                float(row["sigma_A"]),
                float(row["epsilon_K"]),
                row.get("source", ""),
            )
    return ff


def _make_aimd_host(framework_pos, framework_sp, lattice, charges_per_el):
    """Build a HostAtoms from AIMD framework positions, then 3x3x3 centred supercell."""
    from porecdft.structure.host import HostAtoms
    import numpy as np
    from dataclasses import replace
    n = len(framework_sp)
    qs = np.array([charges_per_el[s] for s in framework_sp])
    host = HostAtoms(
        positions=framework_pos, species=list(framework_sp), charges=qs,
        lattice=lattice, source="AIMD framework", charge_source="Hirshfeld CP2K",
    )
    host_super = build_supercell(host, 3, 3, 3)
    shift = -lattice[0] - lattice[1] - lattice[2]
    host_super = replace(host_super, positions=host_super.positions + shift)
    return host, host_super


def main():
    # --- 1. Load porecdft setup ---
    host = read_cif(ALF_CIF)
    host_ff = _read_forcefield_csv(FORCEFIELD_CSV)
    charges = read_charges_csv(CHARGES_CSV)
    host = host.assign_charges(charges, source="Hirshfeld CP2K")
    print(host.summary())

    host_super = build_supercell(host, 3, 3, 3)
    from dataclasses import replace
    shift = -host.lattice[0] - host.lattice[1] - host.lattice[2]
    host_super = replace(host_super, positions=host_super.positions + shift)

    # Cell from CP2K input — used when we rebuild host from each AIMD frame
    AIMD_LATTICE = np.diag([22.861, 11.4305, 11.4305])
    AIMD_CHARGES = charges  # element → charge dict (Hirshfeld)

    co2 = EPM2_CO2
    lj = LJPotential(host_ff=host_ff, fluid_ff=co2.ff, cutoff=15.0)
    coul = CoulombPotential(fluid_charges=co2.charges, cutoff=15.0, method="direct")
    quad = QuadrupoleEFGPotential(theta_zz=co2.theta_zz, cutoff=15.0)
    vtot = CompositePotential([lj, coul, quad])
    rots = fibonacci_rotations(80)

    a = float(np.linalg.norm(host.lattice[0]))
    L = host.lattice
    Linv = np.linalg.inv(L)

    # --- 2. Scan all loadings of CO2 (no N2 mixed) across ALF-S1/2/3 ---
    rows = []
    co2_centres_all: list[tuple[str, int, np.ndarray]] = []  # (sample, loading, center xyz)
    for sample in ("ALF-S1", "ALF-S2", "ALF-S3"):
        for n_co2 in (1, 2, 3, 4):
            xyz_path = AIMD_BASE / sample / f"{sample}-{n_co2}CO2_NVT_10ps.xyz"
            if not xyz_path.exists():
                continue
            species, positions, e_au = read_xyz(xyz_path)
            n_total = len(species)
            n_frame = 104  # ALF framework
            # extract the n_co2 CO2 molecules
            adsorbates = positions[n_frame:]
            ads_species = species[n_frame:]
            # each CO2 = 3 atoms: O C O or C O O (CP2K xyz typically lists by element)
            # Group into molecules of 3 consecutive atoms
            n_ads_atoms = len(ads_species)
            if n_ads_atoms != 3 * n_co2:
                # different ALF supercell for S2/S3 — skip until we extend lattice handling
                print(f"  WARNING: {sample} n_co2={n_co2}: framework size != 104 — skipping")
                continue

            # Build a fresh host from THIS frame's framework — every loading has
            # its own equilibrated framework, and mixing them causes clashes.
            this_host, this_host_super = _make_aimd_host(
                np.array(positions[:n_frame]), species[:n_frame],
                AIMD_LATTICE, AIMD_CHARGES,
            )
            print(f"\n{sample} | n_CO2={n_co2} | E={e_au:.4f} Ha | adsorbate species: {ads_species}")

            for k in range(n_co2):
                mol = adsorbates[3*k:3*k+3]
                mol_sp = ads_species[3*k:3*k+3]
                # The C is the centre — find it
                if "C" in mol_sp:
                    c_idx = mol_sp.index("C")
                    c_pos = mol[c_idx]
                else:
                    c_pos = mol.mean(axis=0)
                # wrap into the original unit cell
                frac = c_pos @ Linv
                frac_wrapped = frac - np.floor(frac)
                cart_wrapped = frac_wrapped @ L
                co2_centres_all.append((sample, n_co2, cart_wrapped))

                # Probe this position with porecdft against THIS frame's framework
                res = probe_binding_site(
                    this_host_super, co2, vtot, cart_wrapped, rots,
                    site_label=f"{sample}-{n_co2}#{k+1}", temperature_K=298.15,
                )
                rows.append({
                    "sample": sample,
                    "n_co2": n_co2,
                    "molecule_idx": k + 1,
                    "x_A": cart_wrapped[0], "y_A": cart_wrapped[1], "z_A": cart_wrapped[2],
                    "frac_x": frac_wrapped[0], "frac_y": frac_wrapped[1], "frac_z": frac_wrapped[2],
                    "porecdft_E_min_kJ_per_mol": res.E_min_kJ_per_mol,
                    "porecdft_E_LJ_kJ_per_mol": res.parts_at_min.get("LJ", 0.0) * K_TO_KJ_PER_MOL,
                    "porecdft_E_Coul_kJ_per_mol": res.parts_at_min.get("Coulomb", 0.0) * K_TO_KJ_PER_MOL,
                    "porecdft_E_Quad_kJ_per_mol": res.parts_at_min.get("Quad", 0.0) * K_TO_KJ_PER_MOL,
                })
                print(f"  CO2 #{k+1}: cart={cart_wrapped.round(3)}  "
                      f"frac={frac_wrapped.round(3)}  "
                      f"E_porecdft={res.E_min_kJ_per_mol:+6.2f} kJ/mol")

    # --- 3. Save CSV ---
    csv_path = OUT_RES / "phase1c_aimd_co2_positions.csv"
    if rows:
        with csv_path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWritten {csv_path}")

    # --- 4. Plot CO2 centres in the unit cell — overlay on ALF atoms ---
    if co2_centres_all:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        # ALF framework atoms
        colors = {"Al": "slategray", "O": "red", "C": "black", "H": "lightgrey"}
        for el, c in colors.items():
            mask = host.select(el)
            ax.scatter(host.positions[mask, 0], host.positions[mask, 1], host.positions[mask, 2],
                       color=c, s=30 if el == "Al" else 12, alpha=0.6, label=el)
        # CO2 centres color-coded by loading
        loadings = sorted({n for _, n, _ in co2_centres_all})
        cmap = plt.get_cmap("plasma")
        for n in loadings:
            pts = np.array([p for _, ln, p in co2_centres_all if ln == n])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       color=cmap(0.2 + 0.8*loadings.index(n)/max(1, len(loadings)-1)),
                       s=80, marker="*", label=f"CO2 @ n={n}", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
        ax.set_title("DFT-equilibrated CO₂ centres in ALF (Phase 1.3c)")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.05, 1))
        fig.tight_layout()
        fig.savefig(OUT_FIG / "06_aimd_co2_positions_3d.png", dpi=150)
        plt.close(fig)

        # 2D projections — XY at varying z, then YZ
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        proj_axes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        labels = ["XY", "XZ", "YZ"]
        for ax, (i, j, _), lab in zip(axes, proj_axes, labels):
            for el, c in colors.items():
                mask = host.select(el)
                ax.scatter(host.positions[mask, i], host.positions[mask, j],
                           color=c, s=30 if el == "Al" else 10, alpha=0.5)
            for n in loadings:
                pts = np.array([p for _, ln, p in co2_centres_all if ln == n])
                ax.scatter(pts[:, i], pts[:, j],
                           color=cmap(0.2 + 0.8*loadings.index(n)/max(1, len(loadings)-1)),
                           s=120, marker="*", edgecolor="white", linewidth=0.5,
                           label=f"n={n}" if ax is axes[0] else None)
            ax.set_xlabel(f"{lab[0]} (Å)"); ax.set_ylabel(f"{lab[1]} (Å)")
            ax.set_title(f"{lab} projection")
            ax.set_aspect("equal")
        axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=2)
        fig.suptitle("DFT-equilibrated CO₂ positions overlaid on ALF framework")
        fig.tight_layout()
        fig.savefig(OUT_FIG / "07_aimd_co2_positions_proj.png", dpi=150)
        plt.close(fig)

    # --- 5. Histogram of binding energies vs DFT references ---
    if rows:
        es = [r["porecdft_E_min_kJ_per_mol"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(es, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(DFT_BINDING_KJ_PER_MOL["SC"], color="red", lw=2,
                   label=f"DFT SC ref {DFT_BINDING_KJ_PER_MOL['SC']:+.1f}")
        ax.axvline(DFT_BINDING_KJ_PER_MOL["LC"], color="orange", lw=2,
                   label=f"DFT LC ref {DFT_BINDING_KJ_PER_MOL['LC']:+.1f}")
        ax.set_xlabel("porecdft binding energy at DFT-equilibrated CO₂ centre (kJ/mol)")
        ax.set_ylabel("# CO₂ molecules")
        ax.set_title("porecdft Vext at DFT-given CO₂ positions vs Evans DFT references")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "08_aimd_porecdft_energies_hist.png", dpi=150)
        plt.close(fig)

    print(f"\nFigures: {OUT_FIG}/06_*, 07_*, 08_*")


if __name__ == "__main__":
    main()

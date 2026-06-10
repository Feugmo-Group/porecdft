"""Phase 1.4 — LJ force-field tuning against DFT binding energies.

Strategy:
1. Build a panel of FF variants (Al ε scaling × global ε scaling × mixing rule).
2. For each variant, probe at every AIMD-equilibrated CO2 centre across loadings
   1..4 in ALF-S1, with both the LJ-only sub-potential and the full
   LJ+Coul+Quad composite.
3. Score each variant by RMSE against the DFT references:
     - DFT_SC = -18.4 kJ/mol (most-attractive single-CO2 binding site)
     - DFT_LC = -8.1 kJ/mol (large cavity)
   We assign each AIMD CO2 to SC or LC by clustering on its current porecdft
   energy, then compute the RMSE.
4. Diagnostic plots: per-variant decomposition; parity plot of porecdft vs DFT
   reference; heatmap of (alpha_Al, alpha_others) RMSE.
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
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.alf_co2 import (
    CHARGES_CSV, FORCEFIELD_CSV, DFT_BINDING_KJ_PER_MOL, DATA_DIR,
)
from applications.alf_co2.notebooks.phase1c_aimd_co2_positions import (
    AIMD_BASE, read_xyz, _make_aimd_host, _read_forcefield_csv,
)
from porecdft.diagnostics import probe_binding_site
from porecdft.diagnostics.binding_site import K_TO_KJ_PER_MOL
from porecdft.fluid import EPM2_CO2
from porecdft.forcefield import (
    CompositePotential, CoulombPotential, LJPotential, QuadrupoleEFGPotential,
)
from porecdft.io import read_charges_csv
from porecdft.io.forcefield import FFEntry
from porecdft.vext import fibonacci_rotations

OUT_FIG = DATA_DIR / "figures"
OUT_RES = DATA_DIR / "results"

DFT_SC = DFT_BINDING_KJ_PER_MOL["SC"]
DFT_LC = DFT_BINDING_KJ_PER_MOL["LC"]


def collect_co2_positions() -> list[dict]:
    """Return all valid (104-framework, n_CO2 in {1..4}) AIMD frames from ALF-S1
    as a list of dicts with framework atoms + per-CO2 centres."""
    out = []
    sample = "ALF-S1"
    AIMD_LATTICE = np.diag([22.861, 11.4305, 11.4305])
    Linv = np.linalg.inv(AIMD_LATTICE)
    for n in (1, 2, 3, 4):
        f = AIMD_BASE / sample / f"{sample}-{n}CO2_NVT_10ps.xyz"
        sp, pos, e = read_xyz(f)
        if len(sp) - 104 != 3 * n:
            continue
        frame = {
            "sample": sample, "n_co2": n,
            "framework_pos": np.array(pos[:104]),
            "framework_sp": sp[:104],
            "lattice": AIMD_LATTICE,
            "co2_centres": [],
        }
        ads_pos = pos[104:]
        ads_sp = sp[104:]
        for k in range(n):
            mol = np.array(ads_pos[3*k:3*k+3])
            mol_sp = ads_sp[3*k:3*k+3]
            ci = mol_sp.index("C") if "C" in mol_sp else 0
            c = mol[ci]
            f_frac = c @ Linv
            f_frac -= np.floor(f_frac)
            frame["co2_centres"].append(f_frac @ AIMD_LATTICE)
        out.append(frame)
    return out


def make_lj_variant(host_ff_base, fluid_ff, alpha_Al, alpha_other, mixing="lorentz-berthelot"):
    """Return an LJPotential with scaled ε for Al (alpha_Al) and others (alpha_other)."""
    scaled: dict[str, FFEntry] = {}
    for el, e in host_ff_base.items():
        f = alpha_Al if el == "Al" else alpha_other
        scaled[el] = FFEntry(el, e.sigma, e.epsilon * f, e.source + f" (×{f:.2f})")
    return LJPotential(host_ff=scaled, fluid_ff=fluid_ff, cutoff=15.0, mixing=mixing)


def evaluate_variant(name, lj, coul, quad, frames, rots, charges, max_orient_skip_threshold=200.0):
    """For each CO2 centre in `frames`, probe with LJ+Coul+Quad and LJ-only.
    Return list of records.
    """
    records = []
    for fr in frames:
        _, hsuper = _make_aimd_host(fr["framework_pos"], fr["framework_sp"], fr["lattice"], charges)
        vtot = CompositePotential([lj, coul, quad])
        vlj  = CompositePotential([lj])
        for k, r0 in enumerate(fr["co2_centres"]):
            res_full = probe_binding_site(hsuper, EPM2_CO2, vtot, r0, rots, site_label=f"n{fr['n_co2']}#{k+1}")
            res_lj   = probe_binding_site(hsuper, EPM2_CO2, vlj,  r0, rots, site_label=f"n{fr['n_co2']}#{k+1}")
            records.append({
                "variant": name,
                "n_co2": fr["n_co2"],
                "k": k + 1,
                "E_full": res_full.E_min_kJ_per_mol,
                "E_LJ":   res_lj.E_min_kJ_per_mol,
                "is_clash": res_full.E_min_kJ_per_mol > max_orient_skip_threshold,
            })
    return records


def assign_sc_lc(energies_kJ):
    """Heuristic: deepest 30% are SC, rest are LC; clashes excluded."""
    e = np.array(energies_kJ)
    finite = e < 50.0
    if finite.sum() < 2:
        return np.array(["?"] * len(e))
    threshold = np.percentile(e[finite], 30)
    label = np.where(finite & (e <= threshold), "SC",
            np.where(finite, "LC", "clash"))
    return label


def compute_rmse(energies, labels):
    refs = np.array([DFT_SC if l == "SC" else DFT_LC if l == "LC" else np.nan for l in labels])
    e = np.array(energies)
    m = np.isfinite(refs) & (np.array(labels) != "clash")
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((e[m] - refs[m]) ** 2)))


def main():
    host_ff = _read_forcefield_csv(FORCEFIELD_CSV)
    charges = read_charges_csv(CHARGES_CSV)
    co2 = EPM2_CO2
    coul = CoulombPotential(fluid_charges=co2.charges, cutoff=15.0)
    quad = QuadrupoleEFGPotential(theta_zz=co2.theta_zz, cutoff=15.0)
    rots = fibonacci_rotations(80)

    frames = collect_co2_positions()
    print(f"Collected {sum(len(f['co2_centres']) for f in frames)} CO2 centres "
          f"from {len(frames)} AIMD frames")

    # FF variants — coarse 2D scan over (alpha_Al, alpha_other)
    alpha_Al_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha_other_grid = [0.3, 0.5, 0.7, 1.0]

    print("\nScanning (alpha_Al, alpha_other) parameter grid...")
    all_records = []
    rmse_grid = np.full((len(alpha_Al_grid), len(alpha_other_grid)), np.nan)
    for i, aAl in enumerate(alpha_Al_grid):
        for j, aOth in enumerate(alpha_other_grid):
            name = f"Al×{aAl:.2f} other×{aOth:.2f}"
            lj = make_lj_variant(host_ff, co2.ff, aAl, aOth)
            recs = evaluate_variant(name, lj, coul, quad, frames, rots, charges)
            for r in recs:
                r["alpha_Al"] = aAl
                r["alpha_other"] = aOth
            energies = [r["E_full"] for r in recs]
            labels = assign_sc_lc(energies)
            rmse = compute_rmse(energies, labels)
            rmse_grid[i, j] = rmse
            print(f"  {name}: RMSE = {rmse:6.2f} kJ/mol "
                  f"(n_clash = {sum(1 for r in recs if r['is_clash'])})")
            all_records.extend(recs)

    # find best variant by RMSE
    best_ij = np.unravel_index(np.nanargmin(rmse_grid), rmse_grid.shape)
    best_aAl = alpha_Al_grid[best_ij[0]]
    best_aOth = alpha_other_grid[best_ij[1]]
    print(f"\nBest FF: Al ε×{best_aAl:.2f}, other ε×{best_aOth:.2f}, "
          f"RMSE = {rmse_grid[best_ij]:.2f} kJ/mol")

    # --- Plot 1: heatmap of RMSE over the (aAl, aOth) grid ---
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(rmse_grid, origin="lower", cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(alpha_other_grid)))
    ax.set_xticklabels([f"{a:.2f}" for a in alpha_other_grid])
    ax.set_yticks(range(len(alpha_Al_grid)))
    ax.set_yticklabels([f"{a:.2f}" for a in alpha_Al_grid])
    ax.set_xlabel("ε scaling for O/C/H")
    ax.set_ylabel("ε scaling for Al")
    ax.set_title("RMSE vs DFT references (kJ/mol)")
    for i, aAl in enumerate(alpha_Al_grid):
        for j, aOth in enumerate(alpha_other_grid):
            ax.text(j, i, f"{rmse_grid[i, j]:.1f}", ha="center", va="center",
                    color="white" if rmse_grid[i, j] > np.nanmean(rmse_grid) else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="RMSE (kJ/mol)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "09_lj_tuning_rmse.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: parity (DFT vs porecdft) for the best variant ---
    best_recs = [r for r in all_records
                 if r["alpha_Al"] == best_aAl and r["alpha_other"] == best_aOth]
    energies = [r["E_full"] for r in best_recs]
    labels = assign_sc_lc(energies)

    fig, ax = plt.subplots(figsize=(6, 6))
    for r, lab in zip(best_recs, labels):
        if lab == "clash":
            continue
        ref = DFT_SC if lab == "SC" else DFT_LC
        ax.scatter(ref, r["E_full"],
                   color="red" if lab == "SC" else "orange",
                   s=70, label=lab if lab not in [t.get_text() for t in ax.legend().get_texts()] else None)
    ax.plot([-25, 0], [-25, 0], "k--", label="parity")
    ax.set_xlabel("DFT reference (kJ/mol)")
    ax.set_ylabel(f"porecdft (Al ε×{best_aAl:.2f}, other ε×{best_aOth:.2f}) (kJ/mol)")
    ax.set_title("Best-variant parity plot")
    ax.set_xlim(-25, 0); ax.set_ylim(-25, 5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "10_lj_tuning_parity.png", dpi=150)
    plt.close(fig)

    # --- Save tuning results CSV ---
    csv_path = OUT_RES / "phase1d_lj_tuning.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        w.writeheader()
        for r in all_records:
            w.writerow(r)
    print(f"\nResults: {csv_path}")
    print(f"Figures: {OUT_FIG}/09_*, 10_*")

    # --- Print per-site results for the best variant ---
    print(f"\n=== Best variant detail (Al ε×{best_aAl:.2f}, other ε×{best_aOth:.2f}) ===")
    print(f"{'n_co2':>5}  {'k':>3}  {'E_full':>10}  {'E_LJ':>10}  {'label':>6}")
    for r, lab in zip(best_recs, labels):
        print(f"  {r['n_co2']:>3}   {r['k']:>3}  {r['E_full']:>+8.2f}    "
              f"{r['E_LJ']:>+8.2f}    {lab:>6}")


if __name__ == "__main__":
    main()

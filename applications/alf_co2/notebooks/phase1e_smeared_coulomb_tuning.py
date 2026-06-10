"""Phase 1.5 — Tune the Gaussian-smeared-Coulomb width against DFT binding
energies at the AIMD-equilibrated CO2 centres.

Sweep gauss_width over [0.5, 0.7, 1.0, 1.3, 1.6, 2.0] Å. For each width, also
combine with the best LJ variant from Phase 1.4 (Al ε×0, others ε×0.70). For
every CO2 position in ALF-S1 n=1..4 (10 CO2 total), probe with porecdft and
compute RMSE against the DFT reference range [-18.4, -8.1] kJ/mol.
"""
from __future__ import annotations

import csv
import sys
import warnings
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
    CHARGES_CSV, FORCEFIELD_CSV, DFT_BINDING_KJ_PER_MOL, DATA_DIR,
)
from applications.alf_co2.notebooks.phase1c_aimd_co2_positions import (
    AIMD_BASE, read_xyz, _read_forcefield_csv, _make_aimd_host,
)
from applications.alf_co2.notebooks.phase1d_lj_tuning import (
    collect_co2_positions, make_lj_variant, assign_sc_lc, compute_rmse,
)
from porecdft.diagnostics import probe_binding_site
from porecdft.fluid import EPM2_CO2
from porecdft.forcefield import (
    CompositePotential, CoulombPotential, LJPotential, QuadrupoleEFGPotential,
)
from porecdft.io import read_charges_csv
from porecdft.vext import fibonacci_rotations

OUT_FIG = DATA_DIR / "figures"
OUT_RES = DATA_DIR / "results"


def main():
    host_ff = _read_forcefield_csv(FORCEFIELD_CSV)
    charges = read_charges_csv(CHARGES_CSV)
    co2 = EPM2_CO2
    quad = QuadrupoleEFGPotential(theta_zz=co2.theta_zz, cutoff=15.0)
    rots = fibonacci_rotations(80)

    frames = collect_co2_positions()
    print(f"Collected {sum(len(f['co2_centres']) for f in frames)} CO2 centres "
          f"from {len(frames)} AIMD frames\n")

    # Best LJ variant from Phase 1.4: Al ε×0, others ε×0.70
    alpha_Al = 0.0
    alpha_other = 0.70
    lj = make_lj_variant(host_ff, co2.ff, alpha_Al, alpha_other)
    print(f"Fixing LJ variant: Al ε×{alpha_Al}, others ε×{alpha_other}\n")

    gauss_widths = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
    print(f"Scanning gauss_width over {gauss_widths} Å (+ direct sum as ref)\n")

    # 1) Baseline: direct sum (current method)
    coul_direct = CoulombPotential(fluid_charges=co2.charges, cutoff=15.0, method="direct")
    # 2) Set of smeared variants
    all_records: list[dict] = []
    width_summary: list[dict] = []

    for label, coul in [("direct", coul_direct)] + [
        (f"smeared σ={w:.1f}",
         CoulombPotential(fluid_charges=co2.charges, cutoff=15.0,
                          method="smeared", gauss_width=w))
        for w in gauss_widths
    ]:
        recs = []
        for fr in frames:
            _, hsuper = _make_aimd_host(fr["framework_pos"], fr["framework_sp"], fr["lattice"], charges)
            vtot = CompositePotential([lj, coul, quad])
            for k, r0 in enumerate(fr["co2_centres"]):
                res = probe_binding_site(hsuper, EPM2_CO2, vtot, r0, rots,
                                          site_label=f"n{fr['n_co2']}#{k+1}")
                rec = {
                    "variant": label,
                    "n_co2": fr["n_co2"],
                    "k": k + 1,
                    "E_full": res.E_min_kJ_per_mol,
                }
                recs.append(rec)
                all_records.append(rec)
        energies = [r["E_full"] for r in recs]
        labels_sclc = assign_sc_lc(energies)
        rmse = compute_rmse(energies, labels_sclc)
        n_clash = sum(1 for e in energies if e > 50)
        n_finite = sum(1 for e in energies if e < 50)
        e_arr = np.array([e for e in energies if e < 50])
        width_summary.append({
            "variant": label,
            "rmse": rmse,
            "n_clash": n_clash,
            "n_finite": n_finite,
            "E_min": float(e_arr.min()) if len(e_arr) else np.nan,
            "E_max": float(e_arr.max()) if len(e_arr) else np.nan,
            "E_mean": float(e_arr.mean()) if len(e_arr) else np.nan,
        })
        print(f"  {label:>18s}  RMSE = {rmse:6.2f}  n_clash = {n_clash}  "
              f"E∈[{width_summary[-1]['E_min']:+7.1f}, "
              f"{width_summary[-1]['E_max']:+7.1f}] kJ/mol")

    # Pick best variant by RMSE
    best = min(width_summary, key=lambda r: r["rmse"] if not np.isnan(r["rmse"]) else np.inf)
    print(f"\nBest: {best['variant']}  RMSE = {best['rmse']:.2f} kJ/mol")

    # --- Plot: RMSE vs gauss_width, with horizontal line for direct ---
    sm = [w for w in width_summary if w["variant"].startswith("smeared")]
    widths = [float(w["variant"].split("=")[1]) for w in sm]
    rmses = [w["rmse"] for w in sm]
    direct_rmse = next(w["rmse"] for w in width_summary if w["variant"] == "direct")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(widths, rmses, "o-", color="steelblue", label="smeared Coulomb", markersize=8)
    ax.axhline(direct_rmse, color="red", ls="--", label=f"direct sum (RMSE={direct_rmse:.1f})")
    ax.set_xlabel("Gaussian smearing width σ (Å)")
    ax.set_ylabel("RMSE vs DFT references (kJ/mol)")
    ax.set_title("Phase 1.5 — RMSE vs Gaussian-smearing width\n"
                 "(LJ fixed at Al ε×0, others ε×0.70)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "11_smeared_coulomb_rmse.png", dpi=150)
    plt.close(fig)

    # --- Plot: per-site energy panels for direct vs best smeared ---
    direct_recs = [r for r in all_records if r["variant"] == "direct"]
    best_label = best["variant"]
    best_recs = [r for r in all_records if r["variant"] == best_label]
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.arange(len(direct_recs))
    site_labels = [f"n{r['n_co2']}#{r['k']}" for r in direct_recs]
    ax.bar(idx - 0.2, [r["E_full"] for r in direct_recs], width=0.4,
           color="lightcoral", label="direct sum")
    ax.bar(idx + 0.2, [r["E_full"] for r in best_recs], width=0.4,
           color="steelblue", label=best_label)
    ax.axhline(DFT_BINDING_KJ_PER_MOL["SC"], color="red", ls=":", label="DFT SC −18.4")
    ax.axhline(DFT_BINDING_KJ_PER_MOL["LC"], color="orange", ls=":", label="DFT LC −8.1")
    ax.set_xticks(idx)
    ax.set_xticklabels(site_labels, fontsize=8)
    ax.set_ylabel("E_min (kJ/mol)")
    ax.set_title(f"Per-site binding energy: direct vs {best_label}")
    ax.legend(fontsize=9)
    ax.set_ylim(-180, 180)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "12_smeared_per_site.png", dpi=150)
    plt.close(fig)

    # CSV
    csv_path = OUT_RES / "phase1e_smeared_tuning.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        w.writeheader()
        for r in all_records:
            w.writerow(r)
    print(f"\nFigures: {OUT_FIG}/11_*.png, 12_*.png")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()

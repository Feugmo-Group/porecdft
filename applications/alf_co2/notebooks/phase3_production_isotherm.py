"""Phase 3 — Production CO₂/ALF adsorption isotherm.

Full (K_eff, ε_assoc) parameter sweep against the corrected Evans 2022 data,
followed by RMSE-optimal model figure. This is the paper's main result.

Model:
  - Langmuir-on-grid with excluded-volume cap (v_excl = 57 Å³)
  - Ω-minimisation over strains {0, 0.5, 1, 2, 3, 5}% at K_eff
  - Self-consistent Wertheim TPT-1 at 7 SC pore centres (κ=119 Å³, ε_assoc)

Sweep:
  K_eff    ∈ {0.3, 0.5, 0.7, 1.0, 2.0} GPa
  ε_assoc  ∈ {0, 100, 200, 300, 400, 500, 600, 700, 800} K

Output:
  24_phase3_param_sweep.png   — RMSE heat-map over (K_eff, ε_assoc)
  25_phase3_best_model.png    — best-model isotherm vs Evans (3 panels)
  26_phase3_parity.png        — parity plot + residuals (all T, all p)
  phase3_production_isotherms.csv
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter
from scipy.spatial.distance import cdist

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.alf_co2 import ALF_CIF_DFT as ALF_CIF, CHARGES_CSV, DATA_DIR, EXP_TARGETS
from porecdft.diagnostics.isotherm import AVOGADRO, density_from_pressure
from porecdft.functional.association import WertheimiAssociation
from porecdft.io import read_cif, read_charges_csv
from porecdft.structure import build_supercell
from porecdft.vext import build_grid

OUT_FIG   = DATA_DIR / "figures"
RES_DIR   = DATA_DIR / "results"
CACHE     = RES_DIR / "vext_cache_flex"
OUT_CSV   = RES_DIR / "phase3_production_isotherms.csv"

ATOMIC_MASS = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}
TEMPERATURES = [273.0, 298.0, 323.0]
STRAINS      = [0.000, 0.005, 0.010, 0.020, 0.030, 0.050]
KAPPA_A3     = 119.0
V_EXCL_A3    = 57.0
V_MIN_CLIP   = -4000.0
BOLTZ_CAP    = 50.0
N_PICARD     = 4
PA_TO_K_A3   = 1.0e9 / 1.380649e-23 * 1.0e-30

# Fine sweep grid
K_EFF_GPa_LIST  = [0.3, 0.5, 0.7, 1.0, 2.0]
EPS_ASSOC_LIST  = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]

# Pressure axis matching experimental range
PRESSURES = np.concatenate([
    np.logspace(-3, np.log10(0.02), 12),
    np.logspace(np.log10(0.02), np.log10(1.2), 30),
])
PRESSURES = np.unique(np.sort(PRESSURES))


# ── helpers ──────────────────────────────────────────────────────────────────

def build_access(host, g3d, shp):
    a1, a2, a3 = host.lattice
    hs = build_supercell(host, 3, 3, 3)
    hs = replace(hs, positions=hs.positions - a1 - a2 - a3)
    nn = np.full(shp, np.inf)
    for h in hs.positions:
        dr = g3d - h
        nn = np.minimum(nn, np.sqrt(np.einsum("ijkd,ijkd->ijk", dr, dr)))
    return nn >= 2.0


def find_sc_centers(host, vext, g3d, access):
    V = np.where(access & np.isfinite(vext), vext, np.inf)
    Vf = minimum_filter(V, size=11, mode="wrap")
    mask = (V == Vf) & access & np.isfinite(V)
    pos, vals = g3d[mask], V[mask]
    inv = np.linalg.inv(host.lattice.T)
    frac = pos @ inv.T
    ok = np.all((frac >= -0.01) & (frac <= 1.01), axis=1)
    pos, vals = pos[ok], vals[ok]
    order = np.argsort(vals)
    pos, vals = pos[order], vals[order]
    D = cdist(pos, pos)
    vis = np.zeros(len(pos), bool)
    keep = []
    for i in range(len(pos)):
        if not vis[i]:
            vis[np.where(D[i] < 5.0)[0]] = True
            keep.append(i)
    return pos[keep]


def langmuir_wertheim(vext, dV, g3d, access, rb, T, assoc):
    beta = 1.0 / T
    V = np.maximum(np.where(np.isfinite(vext), vext, 1e6), V_MIN_CLIP)
    be = np.exp(-np.clip(beta * V, -BOLTZ_CAP, BOLTZ_CAP)) * access
    rho = rb * be / (1.0 + rb * be * V_EXCL_A3)
    if assoc is not None:
        for _ in range(N_PICARD):
            Veff = assoc.effective_vext(V, rho, g3d, dV, T)
            be2 = np.exp(-np.clip(beta * Veff, -BOLTZ_CAP, BOLTZ_CAP)) * access
            rho = rb * be2 / (1.0 + rb * be2 * V_EXCL_A3)
    return float(rho.sum() * dV)


def omega_min(loading_by_strain, V0, K_K_per_A3, T, ip):
    best_om, best_N = np.inf, 0.0
    for s in STRAINS:
        N = loading_by_strain[s][ip]
        VL = V0 * (1 + s) ** 3
        Fel = 0.5 * K_K_per_A3 * V0 * ((VL / V0) - 1.0) ** 2
        Om = -T * N + Fel
        if Om < best_om:
            best_om, best_N = Om, N
    return best_N


def rmse_vs_evans(curve_mmol, T):
    """RMSE of model against all Evans points for temperature T."""
    pts = EXP_TARGETS.get(int(T), [])
    if not pts:
        return np.nan
    errs = []
    for p_exp, n_exp in pts:
        n_model = float(np.interp(p_exp, PRESSURES, curve_mmol))
        errs.append((n_model - n_exp) ** 2)
    return float(np.sqrt(np.mean(errs)))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    host0 = read_cif(ALF_CIF)
    charges = read_charges_csv(CHARGES_CSV)
    host0 = host0.assign_charges(charges, source="Hirshfeld CP2K")
    fw_mass = sum(ATOMIC_MASS[s] for s in host0.species)
    to_mmol = 1000.0 / (AVOGADRO * fw_mass / AVOGADRO)
    V0 = host0.cell_volume

    gxyz0, shp0, dV0 = build_grid(host0, 0.7)
    g3d0 = gxyz0.reshape(*shp0, 3)
    acc0 = build_access(host0, g3d0, shp0)

    vext_ref = np.asarray(np.load(
        CACHE / "vext_avg_strain0.000_T298K.npy", allow_pickle=True
    ).item()["vext_avg"])
    sc0 = find_sc_centers(host0, vext_ref, g3d0, acc0)
    print(f"SC sites: {len(sc0)}   κ={KAPPA_A3:.0f} Å³")

    # ── precompute loading table [eps_assoc][strain][T][ip] ──────────────────
    print("Building loading table (strain × T × ε_assoc × p) …")
    # loading_raw[eps_assoc][strain][T] = N_per_cell array over PRESSURES
    loading_raw: dict = {}
    for eps in EPS_ASSOC_LIST:
        loading_raw[eps] = {}
        for strain in STRAINS:
            scale = 1.0 + strain
            host_s = replace(host0,
                             positions=host0.positions * scale,
                             lattice=host0.lattice * scale)
            gxyz_s, shp_s, dV_s = build_grid(host_s, 0.7)
            g3d_s = gxyz_s.reshape(*shp_s, 3)
            acc_s = build_access(host_s, g3d_s, shp_s)
            sc_s  = sc0 * scale
            assoc = (WertheimiAssociation.from_positions(sc_s, eps, KAPPA_A3)
                     if eps > 0 else None)
            loading_raw[eps][strain] = {}
            for T in TEMPERATURES:
                vext = np.asarray(np.load(
                    CACHE / f"vext_avg_strain{strain:.3f}_T{T:.0f}K.npy",
                    allow_pickle=True
                ).item()["vext_avg"])
                rb_arr = np.array([density_from_pressure(p, T) for p in PRESSURES])
                N_arr = np.array([
                    langmuir_wertheim(vext, dV_s, g3d_s, acc_s, rb, T, assoc)
                    for rb in rb_arr
                ])
                loading_raw[eps][strain][T] = N_arr
        print(f"  ε={eps:.0f} K done")

    # ── Ω-minimisation + RMSE sweep ──────────────────────────────────────────
    print("\nΩ-minimisation + RMSE sweep …")
    rmse_table = {}   # (K_eff, eps) → {T: rmse, "total": rmse}
    curves     = {}   # (K_eff, eps) → {T: mmol_per_g array}

    for K_eff in K_EFF_GPa_LIST:
        K_K_A3 = K_eff * PA_TO_K_A3
        for eps in EPS_ASSOC_LIST:
            key = (K_eff, eps)
            curves[key] = {}
            rmse_table[key] = {}
            rmse_sum = 0.0
            for T in TEMPERATURES:
                N_opt = np.array([
                    omega_min(
                        {s: loading_raw[eps][s][T] for s in STRAINS},
                        V0, K_K_A3, T, ip
                    )
                    for ip in range(len(PRESSURES))
                ])
                mmol = N_opt * to_mmol
                curves[key][T] = mmol
                r = rmse_vs_evans(mmol, T)
                rmse_table[key][T] = r
                rmse_sum += r
            rmse_table[key]["total"] = rmse_sum / len(TEMPERATURES)

    # ── find best overall ────────────────────────────────────────────────────
    best_key = min(curves.keys(), key=lambda k: rmse_table[k]["total"])
    print(f"\nBest (K_eff={best_key[0]} GPa, ε={best_key[1]:.0f} K)  "
          f"total RMSE={rmse_table[best_key]['total']:.4f} mmol/g")
    for T in TEMPERATURES:
        exp_1bar = next((n for p, n in EXP_TARGETS.get(int(T), []) if p > 0.9), None)
        model_1bar = float(np.interp(1.0, PRESSURES, curves[best_key][T]))
        err = (model_1bar - exp_1bar) / exp_1bar * 100 if exp_1bar else float("nan")
        print(f"  {T:.0f} K:  model={model_1bar:.3f}  Evans={exp_1bar:.3f}  "
              f"err={err:+.1f}%  RMSE={rmse_table[best_key][T]:.4f}")

    # ── save CSV ─────────────────────────────────────────────────────────────
    rows = []
    for (K_eff, eps), Tcurves in curves.items():
        for T, mmol in Tcurves.items():
            for p, n in zip(PRESSURES, mmol):
                rows.append({"K_eff_GPa": K_eff, "eps_assoc_K": eps,
                              "T_K": T, "p_bar": p, "N_mmol_g": n})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nCSV: {OUT_CSV}")

    # ── Figure 24: RMSE heat-map ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(K_EFF_GPa_LIST),
                             figsize=(3.5 * len(K_EFF_GPa_LIST), 4.5),
                             sharey=True)
    for ax, K_eff in zip(axes, K_EFF_GPa_LIST):
        Z = np.array([[rmse_table[(K_eff, eps)]["total"]
                       for eps in EPS_ASSOC_LIST]])
        ax.bar(range(len(EPS_ASSOC_LIST)),
               [rmse_table[(K_eff, eps)]["total"] for eps in EPS_ASSOC_LIST],
               color="steelblue", alpha=0.8)
        best_i = int(np.argmin([rmse_table[(K_eff, eps)]["total"]
                                for eps in EPS_ASSOC_LIST]))
        ax.bar(best_i, rmse_table[(K_eff, EPS_ASSOC_LIST[best_i])]["total"],
               color="crimson", alpha=0.9, label="best")
        ax.set_xticks(range(len(EPS_ASSOC_LIST)))
        ax.set_xticklabels([f"{e:.0f}" for e in EPS_ASSOC_LIST], rotation=45, fontsize=8)
        ax.set_xlabel("ε_assoc (K)", fontsize=10)
        ax.set_title(f"K_eff={K_eff} GPa", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("RMSE (mmol/g)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Phase 3 — RMSE vs Evans 2022 (mean over 273/298/323 K)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "24_phase3_param_sweep.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {OUT_FIG}/24_phase3_param_sweep.png")

    # ── Figure 25: best-model isotherm ───────────────────────────────────────
    K_best, eps_best = best_key
    T_COLORS = {273.0: "#1f77b4", 298.0: "#ff7f0e", 323.0: "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, T in zip(axes, TEMPERATURES):
        color = T_COLORS[T]
        ax.plot(PRESSURES, curves[best_key][T],
                color=color, lw=2.5, label=f"cDFT model")
        if int(T) in EXP_TARGETS:
            p_e, n_e = zip(*EXP_TARGETS[int(T)])
            ax.plot(p_e, n_e, "o", color=color, ms=8, mec="k", mew=0.8,
                    zorder=5, label="Evans 2022")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.3)
        ax.set_ylim(0, 6.5)
        ax.set_xlabel("Pressure (bar)", fontsize=11)
        ax.set_ylabel("CO₂ loading (mmol / g)", fontsize=11)
        ax.set_title(f"$T$ = {T:.0f} K", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", alpha=0.2, lw=0.5)
        ax.legend(fontsize=9)
        rmse_T = rmse_table[best_key][T]
        ax.text(0.97, 0.06, f"RMSE = {rmse_T:.3f} mmol/g",
                transform=ax.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.suptitle(
        f"Phase 3 — Best model: K_eff={K_best} GPa, ε_assoc={eps_best:.0f} K\n"
        f"(Ω-min flexible host + Wertheim TPT-1)  Total RMSE={rmse_table[best_key]['total']:.3f} mmol/g",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG / "25_phase3_best_model.png", dpi=180)
    plt.close(fig)
    print(f"Figure: {OUT_FIG}/25_phase3_best_model.png")

    # ── Figure 26: parity + residuals ────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax_par = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1])

    all_exp, all_mod, all_T = [], [], []
    for T in TEMPERATURES:
        for p_exp, n_exp in EXP_TARGETS.get(int(T), []):
            n_mod = float(np.interp(p_exp, PRESSURES, curves[best_key][T]))
            all_exp.append(n_exp)
            all_mod.append(n_mod)
            all_T.append(T)

    all_exp = np.array(all_exp)
    all_mod = np.array(all_mod)
    all_T   = np.array(all_T)
    total_rmse = float(np.sqrt(np.mean((all_mod - all_exp) ** 2)))

    for T, color in T_COLORS.items():
        mask = all_T == T
        ax_par.scatter(all_exp[mask], all_mod[mask], color=color,
                       s=60, zorder=3, label=f"{T:.0f} K", edgecolors="k", lw=0.5)
    lim = max(all_exp.max(), all_mod.max()) * 1.08
    ax_par.plot([0, lim], [0, lim], "k--", lw=1.2, alpha=0.6)
    ax_par.set_xlim(0, lim); ax_par.set_ylim(0, lim)
    ax_par.set_xlabel("Experimental (mmol / g)", fontsize=11)
    ax_par.set_ylabel("Model (mmol / g)", fontsize=11)
    ax_par.set_title("Parity plot", fontsize=12)
    ax_par.legend(fontsize=9)
    ax_par.text(0.05, 0.95, f"RMSE = {total_rmse:.3f} mmol/g",
                transform=ax_par.transAxes, ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
    ax_par.grid(alpha=0.3)

    for T, color in T_COLORS.items():
        mask = all_T == T
        resid = all_mod[mask] - all_exp[mask]
        ax_res.scatter(all_exp[mask], resid, color=color,
                       s=60, zorder=3, label=f"{T:.0f} K", edgecolors="k", lw=0.5)
    ax_res.axhline(0, color="k", lw=1.2, ls="--", alpha=0.6)
    ax_res.set_xlabel("Experimental (mmol / g)", fontsize=11)
    ax_res.set_ylabel("Residual: model − exp (mmol / g)", fontsize=11)
    ax_res.set_title("Residuals", fontsize=12)
    ax_res.legend(fontsize=9)
    ax_res.grid(alpha=0.3)

    fig.suptitle(f"Phase 3 production isotherm — parity & residuals\n"
                 f"K_eff={K_best} GPa, ε_assoc={eps_best:.0f} K",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "26_phase3_parity.png", dpi=180)
    plt.close(fig)
    print(f"Figure: {OUT_FIG}/26_phase3_parity.png")


if __name__ == "__main__":
    main()

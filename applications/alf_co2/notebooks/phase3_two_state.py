"""Phase 3 — Discrete two-state gate-opening model.

ALF undergoes a first-order Im-3̄m (closed) → Pm-3̄m (open) transition.
This notebook models it as a strict two-state system:

  State C (closed): strain=0%  Vext_closed
  State O (open):   strain=5%  Vext_open

For each (T, p):
  Ω_C = −T · N_C(T,p)
  Ω_O = −T · N_O(T,p)  +  ΔF₀          [ΔF₀ > 0: opening costs energy]

  Framework adopts argmin(Ω_C, Ω_O).

Single free-energy parameter ΔF₀ (K per unit cell) controls the gate-opening
pressure and explains the Evans anomaly N(323 K) > N(273 K) at 1 bar.

Sweep:
  ΔF₀      ∈ {100, 200, 500, 1000, 2000, 3000, 5000, 8000, 10000} K
  ε_assoc  ∈ {0, 100, 200, 300, 400, 500} K

Outputs:
  28_phase3_twostate_rmse.png     — RMSE heat-map (ΔF₀ × ε_assoc)
  29_phase3_twostate_best.png     — best-model isotherms vs Evans
  30_phase3_twostate_transition.png — gate-opening pressure vs T
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter
from scipy.spatial.distance import cdist

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.alf_co2 import ALF_CIF, CHARGES_CSV, DATA_DIR, EXP_TARGETS
from porecdft.diagnostics.isotherm import AVOGADRO, density_from_pressure
from porecdft.functional.association import WertheimiAssociation
from porecdft.io import read_cif, read_charges_csv
from porecdft.structure import build_supercell
from porecdft.vext import build_grid

OUT_FIG  = DATA_DIR / "figures"
CACHE    = DATA_DIR / "results" / "vext_cache_flex"
OUT_CSV  = DATA_DIR / "results" / "phase3_twostate_isotherms.csv"

ATOMIC_MASS  = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}
TEMPERATURES = [273.0, 298.0, 323.0]
STRAIN_C     = 0.000   # closed Im-3̄m
STRAIN_O     = 0.050   # open  (approximate Pm-3̄m via 5% expansion)
KAPPA_A3     = 119.0
V_EXCL_A3    = 57.0
V_MIN_CLIP   = -4000.0
BOLTZ_CAP    = 50.0
N_PICARD     = 4

DF0_LIST    = [100., 200., 500., 1000., 2000., 3000., 5000., 8000., 10000.]
EPS_LIST    = [0., 100., 200., 300., 400., 500.]

PRESSURES = np.concatenate([
    np.logspace(-3, np.log10(0.02), 12),
    np.logspace(np.log10(0.02), np.log10(1.3), 32),
])
PRESSURES = np.unique(np.sort(PRESSURES))

T_COLORS = {273.0: "#1f77b4", 298.0: "#ff7f0e", 323.0: "#2ca02c"}


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
    pos = pos[order]
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


def rmse_vs_evans(curve_mmol, T):
    pts = EXP_TARGETS.get(int(T), [])
    if not pts:
        return np.nan
    errs = [(float(np.interp(p, PRESSURES, curve_mmol)) - n) ** 2 for p, n in pts]
    return float(np.sqrt(np.mean(errs)))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    host0 = read_cif(ALF_CIF)
    charges = read_charges_csv(CHARGES_CSV)
    host0 = host0.assign_charges(charges, source="Hirshfeld CP2K")
    fw_mass = sum(ATOMIC_MASS[s] for s in host0.species)
    to_mmol = 1000.0 / (AVOGADRO * fw_mass / AVOGADRO)
    V0 = host0.cell_volume

    # Base grid (strain=0)
    gxyz0, shp0, dV0 = build_grid(host0, 0.7)
    g3d0 = gxyz0.reshape(*shp0, 3)
    acc0 = build_access(host0, g3d0, shp0)
    vext_ref = np.asarray(np.load(
        CACHE / "vext_avg_strain0.000_T298K.npy", allow_pickle=True
    ).item()["vext_avg"])
    sc0 = find_sc_centers(host0, vext_ref, g3d0, acc0)
    print(f"SC sites: {len(sc0)}")

    # ── precompute N_closed and N_open for each (eps, T, p) ──────────────────
    print("Precomputing closed and open loadings …")
    # N_state[eps][T] = array over PRESSURES (molecules/cell)
    N_state: dict = {"closed": {}, "open": {}}

    for state_label, strain in [("closed", STRAIN_C), ("open", STRAIN_O)]:
        scale = 1.0 + strain
        host_s = replace(host0,
                         positions=host0.positions * scale,
                         lattice=host0.lattice * scale)
        gxyz_s, shp_s, dV_s = build_grid(host_s, 0.7)
        g3d_s = gxyz_s.reshape(*shp_s, 3)
        acc_s = build_access(host_s, g3d_s, shp_s)
        sc_s  = sc0 * scale

        N_state[state_label] = {}
        for eps in EPS_LIST:
            assoc = (WertheimiAssociation.from_positions(sc_s, eps, KAPPA_A3)
                     if eps > 0 else None)
            N_state[state_label][eps] = {}
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
                N_state[state_label][eps][T] = N_arr
            print(f"  {state_label} ε={eps:.0f} K done")

    # ── sweep (ΔF₀, ε_assoc) ────────────────────────────────────────────────
    print("\nSweeping (ΔF₀, ε_assoc) …")
    rmse_map = {}   # (dF0, eps) → {T: rmse, "total": rmse}
    curves   = {}   # (dF0, eps) → {T: mmol array}
    p_gate   = {}   # (dF0, eps) → {T: gate-opening pressure or None}

    for dF0 in DF0_LIST:
        for eps in EPS_LIST:
            key = (dF0, eps)
            curves[key] = {}
            rmse_map[key] = {}
            p_gate[key] = {}
            rmse_sum = 0.0
            for T in TEMPERATURES:
                N_C = N_state["closed"][eps][T]
                N_O = N_state["open"][eps][T]
                Om_C = -T * N_C
                Om_O = -T * N_O + dF0
                # Discrete two-state selection
                N_sel = np.where(Om_C <= Om_O, N_C, N_O)
                mmol  = N_sel * to_mmol
                curves[key][T] = mmol
                r = rmse_vs_evans(mmol, T)
                rmse_map[key][T] = r
                rmse_sum += r
                # Find gate-opening pressure (first index where open wins)
                open_wins = Om_O < Om_C
                p_gate[key][T] = float(PRESSURES[np.argmax(open_wins)]) if open_wins.any() else None
            rmse_map[key]["total"] = rmse_sum / len(TEMPERATURES)

    best_key = min(curves.keys(), key=lambda k: rmse_map[k]["total"])
    dF0_best, eps_best = best_key
    print(f"\nBest: ΔF₀={dF0_best:.0f} K, ε={eps_best:.0f} K  "
          f"total RMSE={rmse_map[best_key]['total']:.4f} mmol/g")
    for T in TEMPERATURES:
        exp_1bar = next((n for p, n in EXP_TARGETS.get(int(T), []) if p > 0.9), None)
        mod_1bar = float(np.interp(1.0, PRESSURES, curves[best_key][T]))
        err = (mod_1bar - exp_1bar) / exp_1bar * 100 if exp_1bar else float("nan")
        pg = p_gate[best_key][T]
        pg_str = f"{pg:.3f} bar" if pg else "no gate-opening"
        print(f"  {T:.0f} K: model={mod_1bar:.3f}  Evans={exp_1bar:.3f}  "
              f"err={err:+.1f}%  RMSE={rmse_map[best_key][T]:.4f}  gate@{pg_str}")

    # ── save CSV ─────────────────────────────────────────────────────────────
    rows = []
    for (dF0, eps), Tcurves in curves.items():
        for T, mmol in Tcurves.items():
            for p, n in zip(PRESSURES, mmol):
                rows.append({"dF0_K": dF0, "eps_assoc_K": eps,
                              "T_K": T, "p_bar": p, "N_mmol_g": n})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nCSV: {OUT_CSV}")

    # ── Figure 28: RMSE heat-map ─────────────────────────────────────────────
    Z = np.array([[rmse_map[(dF0, eps)]["total"]
                   for eps in EPS_LIST] for dF0 in DF0_LIST])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, aspect="auto", cmap="viridis_r", origin="lower",
                   vmin=Z.min(), vmax=np.percentile(Z, 80))
    ax.set_xticks(range(len(EPS_LIST)))
    ax.set_xticklabels([f"{e:.0f}" for e in EPS_LIST])
    ax.set_yticks(range(len(DF0_LIST)))
    ax.set_yticklabels([f"{d:.0f}" for d in DF0_LIST])
    ax.set_xlabel("ε_assoc (K)", fontsize=11)
    ax.set_ylabel("ΔF₀ (K / unit cell)", fontsize=11)
    ax.set_title("Two-state model — total RMSE (mmol/g)\nmean over 273 / 298 / 323 K",
                 fontsize=11)
    plt.colorbar(im, ax=ax, label="RMSE (mmol/g)")
    # mark best
    bi = DF0_LIST.index(dF0_best)
    bj = EPS_LIST.index(eps_best)
    ax.plot(bj, bi, "r*", ms=18, label=f"best: ΔF₀={dF0_best:.0f}, ε={eps_best:.0f} K")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "28_phase3_twostate_rmse.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {OUT_FIG}/28_phase3_twostate_rmse.png")

    # ── Figure 29: best-model isotherms ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, T in zip(axes, TEMPERATURES):
        color = T_COLORS[T]
        ax.plot(PRESSURES, curves[best_key][T],
                color=color, lw=2.5, label="Two-state model")
        # also show the individual closed/open curves faintly
        ax.plot(PRESSURES, N_state["closed"][eps_best][T] * to_mmol,
                color=color, lw=1.2, ls=":", alpha=0.5, label="Closed only")
        ax.plot(PRESSURES, N_state["open"][eps_best][T] * to_mmol,
                color=color, lw=1.2, ls="--", alpha=0.5, label="Open only")
        # gate-opening line
        pg = p_gate[best_key][T]
        if pg:
            ax.axvline(pg, color=color, lw=1.0, ls=":", alpha=0.7)
            ax.text(pg * 1.1, 0.3, f"p* = {pg:.3f} bar",
                    color=color, fontsize=7, rotation=90, va="bottom")
        # Evans
        if int(T) in EXP_TARGETS:
            p_e, n_e = zip(*EXP_TARGETS[int(T)])
            ax.plot(p_e, n_e, "o", color=color, ms=8, mec="k", mew=0.8,
                    zorder=5, label="Evans 2022")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.4)
        ax.set_ylim(0, 7)
        ax.set_xlabel("Pressure (bar)", fontsize=11)
        ax.set_ylabel("CO₂ loading (mmol / g)", fontsize=11)
        ax.set_title(f"$T$ = {T:.0f} K", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", alpha=0.2, lw=0.5)
        ax.legend(fontsize=7, loc="upper left")
        ax.text(0.97, 0.06, f"RMSE={rmse_map[best_key][T]:.3f}",
                transform=ax.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.suptitle(
        f"Phase 3 — Discrete two-state gate-opening model\n"
        f"ΔF₀={dF0_best:.0f} K/cell, ε_assoc={eps_best:.0f} K  "
        f"(total RMSE={rmse_map[best_key]['total']:.3f} mmol/g)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG / "29_phase3_twostate_best.png", dpi=180)
    plt.close(fig)
    print(f"Figure: {OUT_FIG}/29_phase3_twostate_best.png")

    # ── Figure 30: gate-opening pressure vs T ───────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dF0 in [500., 1000., 2000., 3000., 5000.]:
        pgs = [p_gate[(dF0, eps_best)][T] for T in TEMPERATURES]
        pgs_plot = [p if p else np.nan for p in pgs]
        ax.plot(TEMPERATURES, pgs_plot, "o-", lw=1.8, ms=7,
                label=f"ΔF₀={dF0:.0f} K")
    ax.set_xlabel("Temperature (K)", fontsize=11)
    ax.set_ylabel("Gate-opening pressure p* (bar)", fontsize=11)
    ax.set_title("Gate-opening pressure vs temperature\n"
                 f"(ε_assoc={eps_best:.0f} K)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "30_phase3_twostate_transition.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {OUT_FIG}/30_phase3_twostate_transition.png")


if __name__ == "__main__":
    main()

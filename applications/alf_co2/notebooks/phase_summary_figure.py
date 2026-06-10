"""All-phases summary figure.

Shows the progressive improvement from Phase 2.1 through 2.5 against Evans 2022.
Each panel is one temperature.  Six curves per panel:

  1. Rigid Langmuir (Phase 2.1b) — EPM2 + Hirshfeld, no flexibility, no association
  2. FMT hard-sphere (Phase 2.2) — aWBII excluded-volume correction
  3. Flexible Ω-min K=0.5 GPa  (Phase 2 final) — soft-mode gate-opening
  4. Flexible + Wertheim ε=300 K (Phase 2.5) — gate-opening + association
  5. Evans Fig 2A experimental data

The FMT result and Langmuir baseline come from saved CSVs.
The flex and flex+Wertheim curves are recomputed from cached Vext so the
pressure axis matches exactly.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter
from scipy.spatial.distance import cdist

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.alf_co2 import ALF_CIF, CHARGES_CSV, DATA_DIR, EXP_TARGETS
from porecdft.diagnostics.isotherm import (
    AVOGADRO, density_from_pressure,
)
from porecdft.functional.association import WertheimiAssociation
from porecdft.io import read_cif, read_charges_csv
from porecdft.structure import build_supercell
from porecdft.vext import build_grid

OUT_FIG = DATA_DIR / "figures"
RES_DIR = DATA_DIR / "results"
CACHE   = DATA_DIR / "results" / "vext_cache_flex"

ATOMIC_MASS = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}

# EXP_TARGETS imported from applications.alf_co2

TEMPERATURES  = [273.0, 298.0, 323.0]
STRAINS       = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
V_EXCL_A3     = 57.0
V_MIN_CLIP_K  = -4000.0
BOLTZ_CAP     = 50.0
N_PICARD      = 4
PA_TO_K_A3    = 1.0e9 / 1.380649e-23 * 1.0e-30


# ── helpers ──────────────────────────────────────────────────────────────────

def build_access(host, grid_3d, shape, spacing=0.7):
    a1, a2, a3 = host.lattice
    hs = build_supercell(host, 3, 3, 3)
    hs = replace(hs, positions=hs.positions - a1 - a2 - a3)
    nn = np.full(shape, np.inf)
    for h in hs.positions:
        dr = grid_3d - h
        nn = np.minimum(nn, np.sqrt(np.einsum("ijkd,ijkd->ijk", dr, dr)))
    return nn >= 2.0


def sc_centers(host, vext_ref, grid_3d, access):
    V = np.where(access & np.isfinite(vext_ref), vext_ref, np.inf)
    Vf = minimum_filter(V, size=11, mode="wrap")
    mask = (V == Vf) & access & np.isfinite(V)
    vals, pos = V[mask], grid_3d[mask]
    inv = np.linalg.inv(host.lattice.T)
    frac = pos @ inv.T
    ok = np.all((frac >= -0.01) & (frac <= 1.01), axis=1)
    pos, vals = pos[ok], vals[ok]
    idx = np.argsort(vals)
    pos, vals = pos[idx], vals[idx]
    D = cdist(pos, pos)
    vis = np.zeros(len(pos), bool)
    keep = []
    for i in range(len(pos)):
        if not vis[i]:
            vis[np.where(D[i] < 5.0)[0]] = True
            keep.append(i)
    return pos[keep]


def langmuir_loading(vext, dV, access, rb, T, assoc=None):
    beta = 1.0 / T
    V = np.maximum(np.where(np.isfinite(vext), vext, 1e6), V_MIN_CLIP_K)
    be = np.exp(-np.clip(beta * V, -BOLTZ_CAP, BOLTZ_CAP)) * access
    rho = rb * be / (1.0 + rb * be * V_EXCL_A3)
    if assoc is not None:
        grid_3d = None   # injected via closure — see caller
        raise RuntimeError("use langmuir_wertheim_loading for association")
    return float(rho.sum() * dV)


def langmuir_wertheim(vext, dV, grid_3d, access, rb, T, assoc):
    beta = 1.0 / T
    V = np.maximum(np.where(np.isfinite(vext), vext, 1e6), V_MIN_CLIP_K)
    be = np.exp(-np.clip(beta * V, -BOLTZ_CAP, BOLTZ_CAP)) * access
    rho = rb * be / (1.0 + rb * be * V_EXCL_A3)
    if assoc is not None:
        for _ in range(N_PICARD):
            Veff = assoc.effective_vext(V, rho, grid_3d, dV, T)
            be2 = np.exp(-np.clip(beta * Veff, -BOLTZ_CAP, BOLTZ_CAP)) * access
            rho = rb * be2 / (1.0 + rb * be2 * V_EXCL_A3)
    return float(rho.sum() * dV)


def omega_min_curve(loading_by_strain, strains, V0, K_K_per_A3, T, pressures):
    """Return Ω-minimised loading (mmol/g) for one T."""
    to_mmol = loading_by_strain[0.0]["to_mmol"]
    out = np.empty(len(pressures))
    for ip in range(len(pressures)):
        best = np.inf
        best_N = 0.0
        for s in strains:
            N = loading_by_strain[s]["N"][ip]
            VL = V0 * (1 + s) ** 3
            Fel = 0.5 * K_K_per_A3 * V0 * ((VL / V0) - 1.0) ** 2
            Om = -T * N + Fel
            if Om < best:
                best, best_N = Om, N
        out[ip] = best_N * to_mmol
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    host0 = read_cif(ALF_CIF)
    charges = read_charges_csv(CHARGES_CSV)
    host0 = host0.assign_charges(charges, source="Hirshfeld CP2K")
    fw_mass = sum(ATOMIC_MASS[s] for s in host0.species)
    fw_g    = fw_mass / AVOGADRO
    to_mmol = 1000.0 / (AVOGADRO * fw_g)
    V0 = host0.cell_volume

    pressures = np.logspace(-3, 0, 40)

    # ── build base grid ──────────────────────────────────────────────────────
    gxyz0, shp0, dV0 = build_grid(host0, 0.7)
    g3d0 = gxyz0.reshape(*shp0, 3)
    acc0 = build_access(host0, g3d0, shp0)

    vext_ref = np.asarray(np.load(
        CACHE / "vext_avg_strain0.000_T298K.npy", allow_pickle=True
    ).item()["vext_avg"])
    sc = sc_centers(host0, vext_ref, g3d0, acc0)
    print(f"SC sites: {len(sc)}")

    # ── load CSV baselines ────────────────────────────────────────────────────
    df_base = pd.read_csv(RES_DIR / "phase2_baseline_isotherms.csv")
    df_fmt  = pd.read_csv(RES_DIR / "phase2_2_fmt_isotherms.csv")

    def csv_curve(df, T, model_col=None, model_val=None, col="mmol_per_g_abs"):
        sub = df[df["T_K"].round(0) == T]
        if model_col and model_val:
            sub = sub[sub[model_col] == model_val]
        sub = sub.sort_values("p_bar")
        return sub["p_bar"].values, sub[col].values

    # ── compute flex and flex+Wertheim for each strain/T ─────────────────────
    K_eff_GPa   = 0.5
    K_K_per_A3  = K_eff_GPa * PA_TO_K_A3
    eps_assoc   = 300.0   # K

    loading_rigid  = {T: {} for T in TEMPERATURES}   # strain → {"N": array}
    loading_w300   = {T: {} for T in TEMPERATURES}

    for strain in STRAINS:
        scale = 1.0 + strain
        host_s = replace(host0, positions=host0.positions*scale, lattice=host0.lattice*scale)
        gxyz_s, shp_s, dV_s = build_grid(host_s, 0.7)
        g3d_s = gxyz_s.reshape(*shp_s, 3)
        acc_s = build_access(host_s, g3d_s, shp_s)

        sc_s   = sc * scale
        assoc  = WertheimiAssociation.from_positions(sc_s, eps_assoc, 119.0)

        for T in TEMPERATURES:
            vext = np.asarray(np.load(
                CACHE / f"vext_avg_strain{strain:.3f}_T{T:.0f}K.npy",
                allow_pickle=True
            ).item()["vext_avg"])
            rho_bulk = np.array([density_from_pressure(p, T) for p in pressures])

            N_rigid = np.array([langmuir_wertheim(vext, dV_s, g3d_s, acc_s, rb, T, None)
                                 for rb in rho_bulk])
            N_w300  = np.array([langmuir_wertheim(vext, dV_s, g3d_s, acc_s, rb, T, assoc)
                                 for rb in rho_bulk])
            loading_rigid[T][strain] = {"N": N_rigid, "to_mmol": to_mmol}
            loading_w300 [T][strain] = {"N": N_w300,  "to_mmol": to_mmol}

        print(f"  strain={strain*100:.1f}% done")

    # ── Ω-minimised curves ────────────────────────────────────────────────────
    flex_curve  = {T: omega_min_curve(loading_rigid[T], STRAINS, V0, K_K_per_A3, T, pressures)
                   for T in TEMPERATURES}
    flexw_curve = {T: omega_min_curve(loading_w300[T],  STRAINS, V0, K_K_per_A3, T, pressures)
                   for T in TEMPERATURES}

    # ── plot ─────────────────────────────────────────────────────────────────
    COLORS = {
        "baseline": "#4878CF",   # blue
        "fmt":      "#6ACC65",   # green
        "flex":     "#D65F5F",   # red
        "combined": "#B47CC7",   # purple
        "evans":    "black",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax, T in zip(axes, TEMPERATURES):
        # 1 — Rigid Langmuir
        pb, nb = csv_curve(df_base, T, "model", "langmuir")
        ax.plot(pb, nb, color=COLORS["baseline"], lw=2.0, ls="--",
                label="Rigid Langmuir (Ph 2.1b)")

        # 2 — FMT (skip if all unconverged — e.g. 273 K needs Anderson acc.)
        sub_fmt = df_fmt[df_fmt["T_K"].round(0) == T]
        fmt_converged = sub_fmt[sub_fmt.get("converged", pd.Series(True, index=sub_fmt.index))]
        if len(fmt_converged) == 0:
            fmt_converged = sub_fmt  # fall back to all rows
        pf = fmt_converged.sort_values("p_bar")["p_bar"].values
        nf = fmt_converged.sort_values("p_bar")["mmol_per_g_abs"].values
        valid_fmt = nf < 20.0   # exclude clearly diverged points
        if valid_fmt.sum() >= 3:
            ax.plot(pf[valid_fmt], nf[valid_fmt], color=COLORS["fmt"], lw=2.0, ls="-.",
                    label="FMT aWBII (Ph 2.2)")
        else:
            ax.plot([], [], color=COLORS["fmt"], lw=2.0, ls="-.",
                    label="FMT aWBII (Ph 2.2) [not converged]")

        # 3 — Flexible K=0.5 GPa
        ax.plot(pressures, flex_curve[T], color=COLORS["flex"], lw=2.5, ls="-",
                label="Flex Ω-min K=0.5 GPa (Ph 2 final)")

        # 4 — Flexible + Wertheim ε=300 K
        ax.plot(pressures, flexw_curve[T], color=COLORS["combined"], lw=2.5, ls="-",
                label="Flex + Wertheim ε=300 K (Ph 2.5)")

        # Evans
        p_e, n_e = zip(*EXP_TARGETS[T])
        ax.plot(p_e, n_e, "o", color=COLORS["evans"], ms=8, zorder=5,
                label="Evans 2022 (Fig 2A)")

        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.2)
        ax.set_ylim(0, 7)
        ax.set_xlabel("Pressure (bar)", fontsize=11)
        ax.set_ylabel("CO₂ loading (mmol / g)", fontsize=11)
        ax.set_title(f"$T$ = {T:.0f} K", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", alpha=0.2, lw=0.5)
        ax.tick_params(labelsize=10)

        # error annotation at 1 bar
        evans_1bar = next((n for p, n in EXP_TARGETS[T] if p > 0.9), None)
        models_at_1bar = {
            "rigid":    float(np.interp(1.0, pb, nb)),
            "flex":     flex_curve[T][-1],
            "combined": flexw_curve[T][-1],
        }
        err_strs = []
        for tag, val in models_at_1bar.items():
            err = (val - evans_1bar) / evans_1bar * 100
            err_strs.append(f"{tag}: {err:+.0f}%")
        ax.text(0.98, 0.04, "\n".join(err_strs), transform=ax.transAxes,
                fontsize=7.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # single shared legend under the panels
    handles = [
        mlines.Line2D([], [], color=COLORS["baseline"], lw=2, ls="--",
                      label="Rigid Langmuir (Ph 2.1b)"),
        mlines.Line2D([], [], color=COLORS["fmt"],      lw=2, ls="-.",
                      label="FMT aWBII (Ph 2.2)"),
        mlines.Line2D([], [], color=COLORS["flex"],     lw=2.5,
                      label="Flexible Ω-min K=0.5 GPa (Ph 2 final)"),
        mlines.Line2D([], [], color=COLORS["combined"], lw=2.5,
                      label="Flexible + Wertheim ε=300 K (Ph 2.5)"),
        mlines.Line2D([], [], color=COLORS["evans"],  lw=0, marker="o", ms=8,
                      label="Evans 2022 (Fig 2A)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "cDFT model progression — CO₂ / ALF adsorption isotherms",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    outpath = OUT_FIG / "23_all_phases_summary.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")

    # ── print table ───────────────────────────────────────────────────────────
    print("\n=== @ 1 bar  (Evans / rigid / FMT / flex / flex+W) ===")
    for T in TEMPERATURES:
        evans_1bar = next((n for p, n in EXP_TARGETS[T] if p > 0.9), None)
        _, nb = csv_curve(df_base, T, "model", "langmuir")
        pf_all, nf_all = csv_curve(df_fmt, T)
        rigid_v = float(nb[-1])
        valid_f = nf_all < 20.0
        fmt_v   = float(nf_all[valid_f][-1]) if valid_f.sum() >= 3 else float("nan")
        flex_v  = flex_curve[T][-1]
        comb_v  = flexw_curve[T][-1]
        fmt_str = (f"FMT {fmt_v:.2f} ({(fmt_v-evans_1bar)/evans_1bar*100:+.0f}%)"
                   if not np.isnan(fmt_v) else "FMT n/a (not converged)")
        print(f"  {T:.0f} K | Evans {evans_1bar:.2f} | "
              f"Rigid {rigid_v:.2f} ({(rigid_v-evans_1bar)/evans_1bar*100:+.0f}%) | "
              f"{fmt_str} | "
              f"Flex {flex_v:.2f} ({(flex_v-evans_1bar)/evans_1bar*100:+.0f}%) | "
              f"Flex+W {comb_v:.2f} ({(comb_v-evans_1bar)/evans_1bar*100:+.0f}%)")


if __name__ == "__main__":
    main()

"""Shared runner for the 4 per-system H2 cDFT scripts.

Loads the helpers from make_h2_isotherm_cdft.py (without running its main
block), exposes ``run_isotherm(name, mode)`` that produces the isotherm
NPZ + a single-system figure with the GCMC peak marker.

mode = "LJ-Morse"  → Morse on {Co,Fe,Ni,Cu,Mn}, LJ DREIDING elsewhere
mode = "LJ-only"   → LJ DREIDING on every host atom (including Co)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT    = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

import jax
jax.config.update("jax_enable_x64", True)

# Pull helpers from the production script (everything before "# 6. MAIN").
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE / "make_h2_isotherm_cdft.py"
_helper_src = _SRC.read_text().split("# 6. MAIN")[0]
_NS: dict = {"__name__": "_helpers", "__file__": str(_SRC)}
exec(compile(_helper_src, str(_SRC), "exec"), _NS)

T_K = 298.0
P_BAR = [1, 5, 10, 20, 40, 60, 80, 100, 120, 150, 180, 220, 260,
         300, 350, 400, 450, 500, 600, 700]

GCMC_REF = {
    "COF-301-CoCl2": {"P_opt_bar": 120.0, "peak_gL": 22.2},
    "COF-333-CoCl2": {"P_opt_bar": 220.0, "peak_gL": 13.3},
}


def run_isotherm(name: str, mode: str) -> dict:
    assert mode in ("LJ-Morse", "LJ-only"), f"bad mode: {mode}"
    # In LJ-only mode, empty the Morse-metal set so build_vext_3d goes through
    # the LJ branch for Co as well.
    if mode == "LJ-only":
        _NS["MORSE_METALS"] = set()
    else:
        _NS["MORSE_METALS"] = {"Co", "Fe", "Ni", "Cu", "Mn"}

    load_host         = _NS["load_host"]
    build_vext_3d     = _NS["build_vext_3d"]
    run_isotherm_cdft = _NS["run_isotherm_cdft"]
    MASS_MAP          = _NS["MASS_MAP"]
    SIGMA_H2          = _NS["SIGMA_H2"]
    RESULTS_DIR       = _NS["RESULTS_DIR"]
    FIGURES_DIR       = _NS["FIGURES_DIR"]

    tag = "" if mode == "LJ-Morse" else "_LJonly"

    print("=" * 72)
    print(f"H2 cDFT — {name} — {mode}, aWBII+WDA, T = {T_K} K")
    print("=" * 72)
    host   = load_host(name)
    mss    = sum(MASS_MAP.get(el, 0.0) for el in host.species)
    V_cell = host.cell_volume
    print(f"  V_cell = {V_cell:.1f} Å³   mss = {mss:.1f} u")

    vext_cache = os.path.join(RESULTS_DIR, f"vext_cache_{name}{tag}.npy")
    vext_3d, n_pts, spacings, dV = build_vext_3d(
        host, grid_spacing=0.25 * SIGMA_H2, supercell=(3, 3, 3),
        cache_path=vext_cache,
    )
    print(f"  Grid: {n_pts[0]}x{n_pts[1]}x{n_pts[2]}  dV={dV:.4f}  "
          f"Vext min={vext_3d.min():.0f} K", flush=True)

    iso_cache = os.path.join(
        RESULTS_DIR, f"isotherm_{name}{tag}_{int(T_K)}K_0-700bar.npz")
    iso = run_isotherm_cdft(
        vext_3d=vext_3d, spacings=spacings, dV=dV, V_cell=V_cell,
        mss_u=mss, T_K=T_K, pressures_bar=P_BAR, cache_path=iso_cache,
    )

    P  = np.asarray(iso["P"])
    eg = np.asarray(iso["extra_gL"])
    ref = GCMC_REF[name]
    color = "#d6604d" if mode == "LJ-Morse" else "#1f77b4"
    label_cdft = f"porecdft cDFT ({mode})"

    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    ax.plot(P, eg, "o-", color=color, lw=2.0, ms=5, label=label_cdft)
    ax.scatter([ref["P_opt_bar"]], [ref["peak_gL"]],
               s=160, c="navy", marker="*", zorder=5,
               label=f"GCMC peak ({ref['P_opt_bar']:.0f} bar, "
                     f"{ref['peak_gL']:.1f} g/L)")
    ax.axvline(ref["P_opt_bar"], color="navy", lw=0.8, ls=":")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel(r"Extra H$_2$ uptake (g L$^{-1}$)")
    ax.set_title(f"{name} — H$_2$ at {int(T_K)} K ({mode})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3, ls=":")

    p_peak = float(P[int(np.argmax(eg))])
    print(f"\n  Peak: {eg.max():.1f} g/L at {p_peak:.0f} bar   "
          f"|  GCMC: {ref['peak_gL']:.1f} g/L at {ref['P_opt_bar']:.0f} bar")

    out = os.path.join(FIGURES_DIR,
                       f"h2_{name}_{mode.replace('-', '_')}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out}")
    return iso

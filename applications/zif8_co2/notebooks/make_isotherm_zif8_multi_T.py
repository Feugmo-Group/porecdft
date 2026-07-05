"""Multi-temperature CO2/ZIF-8 adsorption isotherms for SI figure.

Runs the FMT-aWBII isotherm at 273, 298, 323 K using the Vext caches produced
by build_vext_mace.py (298 K) + rebuild_vext_multi_T.py (273, 323 K), then
overlays the three curves on a single log-P plot.
"""
from __future__ import annotations

import csv
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT    = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try:    sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.zif8_co2 import CACHE_DIR, FIG_DIR

from porecdft.eos import CO2_SW
from porecdft.functional import (
    make_k_grid, make_fmt_weights_hat,
    compute_weighted_densities, compute_c1, bulk_c1,
)
from porecdft.solver import picard_solve, anderson_solve

AVOGADRO   = 6.02214076e23
K_B_J      = 1.380649e-23
ZIF8_MW_G_PER_MOL = 1856.1
ZIF8_MW_G_PER_UC  = ZIF8_MW_G_PER_MOL / AVOGADRO

SIGMA_HS = 3.017                   # Å
T_LIST   = [273.0, 298.0, 323.0]   # K — matches Vext caches
# Cap at 25 bar until the Span-Wagner root-picking bug is fixed
PRESSURES = np.concatenate([
    np.linspace(0.05, 0.9, 6),
    np.linspace(1.0,  9.0, 9),
    np.linspace(10.0, 25.0, 6),
])

COLORS = {273.0: "#1f77b4", 298.0: "#2ca02c", 323.0: "#d62728"}


def _load_vext(T: float) -> tuple[np.ndarray, tuple, float, np.ndarray]:
    cache = CACHE_DIR / f"vext_mace_T{T:.0f}K.npy"
    if not cache.exists():
        raise FileNotFoundError(f"Missing: {cache}")
    data    = np.load(cache, allow_pickle=True).item()
    vext_3d = np.asarray(data["vext_avg"], dtype=np.float64)
    shape   = tuple(data["shape"])
    lattice = np.asarray(data["lattice"])
    dV = float(np.prod([np.linalg.norm(lattice[i]) for i in range(3)])) / np.prod(shape)
    return vext_3d, shape, dV, lattice


def isotherm_at(T_K: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (N_abs_mmol_g, N_exc_mmol_g) arrays over PRESSURES."""
    vext_3d, shape, dV, lattice = _load_vext(T_K)
    Vext_K = np.where(np.isfinite(vext_3d), vext_3d, +1e6)
    Vext_K = np.maximum(Vext_K, -4000.0)
    access = Vext_K < 5000.0

    ax, ay, az = [np.linalg.norm(lattice[i]) for i in range(3)]
    dx, dy, dz = ax / shape[0], ay / shape[1], az / shape[2]
    KX, KY, KZ, K = make_k_grid(shape, dx, dy, dz)
    w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, SIGMA_HS)
    rho_max = 0.45 * 6.0 / (np.pi * SIGMA_HS ** 3)

    def c1_callable(rho_arr):
        wd = compute_weighted_densities(rho_arr, w2_hat, w3_hat, w2vec_hat, SIGMA_HS)
        return np.asarray(compute_c1(rho_arr, wd, w2_hat, w3_hat, w2vec_hat,
                                     SIGMA_HS, model="aWBII"))

    def boltzmann_init(rho_b):
        ri = rho_b * np.exp(np.clip(-Vext_K / T_K, -50.0, 20.0)) * access
        return np.minimum(ri, rho_max)

    N_abs_arr = np.empty(len(PRESSURES))
    N_exc_arr = np.empty(len(PRESSURES))
    rho = None
    to_mmol_g = 1000.0 / (AVOGADRO * ZIF8_MW_G_PER_UC)

    for i, p in enumerate(PRESSURES):
        rho_bulk = CO2_SW.bulk_density(p, T_K)
        c1_b     = bulk_c1(rho_bulk, SIGMA_HS, model="aWBII")
        rho_init = rho if rho is not None else boltzmann_init(rho_bulk)
        if rho is not None and (not np.isfinite(rho).all() or rho.sum() * dV > 1e4):
            rho_init = boltzmann_init(rho_bulk)

        res = anderson_solve(
            rho_init=rho_init, rho_bulk=rho_bulk,
            Vext_K=Vext_K, temperature_K=T_K,
            c1_callable=c1_callable, c1_bulk=c1_b,
            m=6, beta=0.3, max_iter=800, tol=1e-4,
            accessibility_mask=access, log_clip=25.0,
            safeguard_alpha=0.02, picard_warmup=30, step_clip=2.0,
            rho_max=rho_max,
        )
        last_err = res.error_history[-1] if res.error_history else np.inf

        if not res.converged and (not np.isfinite(last_err) or last_err > 0.1):
            res = picard_solve(
                rho_init=res.rho if np.isfinite(last_err) else boltzmann_init(rho_bulk),
                rho_bulk=rho_bulk, Vext_K=Vext_K, temperature_K=T_K,
                c1_callable=c1_callable, c1_bulk=c1_b,
                alpha=0.005, max_iter=2000, tol=1e-3,
                accessibility_mask=access, log_clip=25.0,
                rho_max=rho_max,
            )
            last_err = res.error_history[-1] if res.error_history else np.inf

        rho = res.rho if (np.isfinite(last_err) and last_err < 0.5) else None
        N_abs_arr[i] = float(res.rho.sum() * dV)
        N_exc_arr[i] = float((res.rho - rho_bulk * access).sum() * dV)

    return N_abs_arr * to_mmol_g, N_exc_arr * to_mmol_g


def main() -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    csv_path = CACHE_DIR.parent / "zif8_co2_isotherm_multi_T.csv"

    rows = []
    for T in T_LIST:
        print(f"\n=== T = {T:.0f} K ===")
        t0 = time.time()
        N_abs, N_exc = isotherm_at(T)
        print(f"  done in {time.time()-t0:.1f} s")
        for p, na, ne in zip(PRESSURES, N_abs, N_exc):
            rows.append([T, p, na, ne])
        ax.plot(PRESSURES, N_abs, "o-", color=COLORS[T],
                markersize=4, label=f"T = {T:.0f} K")

    ax.set_xlabel("Pressure (bar)", fontsize=13)
    ax.set_ylabel("CO₂ loading (mmol / g)", fontsize=13)
    ax.set_title("CO₂ / ZIF-8 adsorption isotherms  [MACE-MP-0 + FMT-aWBII]",
                 fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(PRESSURES.min() * 0.8, PRESSURES.max() * 1.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, title="Temperature")
    fig.tight_layout()

    out = FIG_DIR / "zif8_co2_isotherm_mace_multi_T.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nFigure: {out}")

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_K", "P_bar", "N_abs_mmol_g", "N_exc_mmol_g"])
        w.writerows(rows)
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()

"""Compute the CO2 adsorption isotherm in ZIF-8 using FMT-aWBII + MACE Vext.

Prerequisites
-------------
Run build_vext_mace.py first to generate::

    applications/zif8_co2/results/vext_cache/vext_mace_T298K.npy

Pipeline
--------
1. Load the MACE-averaged Vext grid from the .npy cache.
2. Set up the FMT-aWBII hard-sphere functional (σ_HS = 3.017 Å for CO2).
3. Use Span-Wagner CO2 EOS for accurate bulk density ρ(P, T).
4. Iterate the self-consistent density equation via Anderson mixing:

       ρ(r) = ρ_bulk · exp[−β V_ext(r) + c¹_HS(r) − c¹_HS(ρ_bulk)]

5. Integrate to get absolute N_abs and excess N_exc loading per unit cell.
6. Plot N(P) at 298 K and save to figures/zif8_co2_isotherm_mace.png.

Usage::

    cd porecdft
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \\
        applications/zif8_co2/notebooks/make_isotherm_zif8.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── sys.path ──────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT    = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try:    sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore", message=".*symmetry_equiv_pos_as_xyz.*")

from applications.zif8_co2 import CACHE_DIR, FIG_DIR

from porecdft.eos import CO2_SW            # Span-Wagner reference EOS for CO2
from porecdft.functional import (
    make_k_grid, make_fmt_weights_hat,
    compute_weighted_densities, compute_c1, bulk_c1,
)
from porecdft.solver import picard_solve, anderson_solve

# ── constants ──────────────────────────────────────────────────────────────────
AVOGADRO   = 6.02214076e23
K_B_J      = 1.380649e-23           # J / K
K_B_PA_A3  = K_B_J * 1e30          # Pa·Å³ / K

# ZIF-8 molecular formula per unit cell: Zn₈(mIm)₁₆ · a = 16.991 Å, V = 4905.2 Å³
ZIF8_MW_G_PER_MOL = 1856.1          # g / mol  (Zn8 C48 H60 N32, MW from formula)
ZIF8_MW_G_PER_UC  = ZIF8_MW_G_PER_MOL / AVOGADRO   # g / uc

# ── simulation parameters ──────────────────────────────────────────────────────
T_K         = 298.0
SIGMA_HS    = 3.017                  # Å  (single-site spherical CO2 hard-sphere diam.)
PRESSURES   = np.concatenate([
    np.linspace(0.05, 0.9, 6),
    np.linspace(1.0,  9.0, 9),
    np.linspace(10.0, 50.0, 9),
])                                   # bar


def _load_vext(T: float) -> tuple[np.ndarray, tuple, float, np.ndarray]:
    """Load orientation-averaged Vext from MACE cache. Returns (vext_3d, shape, dV, lattice)."""
    cache = CACHE_DIR / f"vext_mace_T{T:.0f}K.npy"
    if not cache.exists():
        raise FileNotFoundError(
            f"Vext cache not found: {cache}\n"
            "Run build_vext_mace.py first."
        )
    data    = np.load(cache, allow_pickle=True).item()
    vext_3d = np.asarray(data["vext_avg"], dtype=np.float64)
    shape   = tuple(data["shape"])
    lattice = np.asarray(data["lattice"])
    dV      = float(np.prod([np.linalg.norm(lattice[i]) for i in range(3)])) / np.prod(shape)
    return vext_3d, shape, dV, lattice


def main() -> None:
    print(f"Loading Vext for T = {T_K:.0f} K …")
    vext_3d, shape, dV, lattice = _load_vext(T_K)
    print(f"  shape {shape}, dV = {dV:.3f} Å³")
    print(f"  Vext: min={vext_3d.min():.0f} K  max={vext_3d.max():.0f} K  mean={vext_3d.mean():.1f} K")

    # ── clip Vext to safe range ────────────────────────────────────────────────
    Vext_K = np.where(np.isfinite(vext_3d), vext_3d, +1e6)
    Vext_K = np.maximum(Vext_K, -4000.0)   # prevent exp overflow in deep wells

    # ── accessibility mask: grid points closer than 1.5 Å to any framework atom ─
    # For the MACE workflow we skip the re-read of the host and simply use the Vext
    # itself: any site with Vext > 5000 K is effectively inaccessible.
    access = Vext_K < 5000.0
    print(f"  Accessible voxels: {access.sum()}/{access.size} ({100*access.mean():.1f}%)")

    # ── FMT k-grid and weight functions ───────────────────────────────────────
    ax = np.linalg.norm(lattice[0])
    ay = np.linalg.norm(lattice[1])
    az = np.linalg.norm(lattice[2])
    dx, dy, dz = ax / shape[0], ay / shape[1], az / shape[2]
    KX, KY, KZ, K = make_k_grid(shape, dx, dy, dz)
    w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, SIGMA_HS)
    print(f"FMT: σ_HS={SIGMA_HS} Å, grid {shape}, dx={dx:.3f} Å")

    rho_max = 0.45 * 6.0 / (np.pi * SIGMA_HS ** 3)

    def c1_callable(rho_arr):
        wd = compute_weighted_densities(rho_arr, w2_hat, w3_hat, w2vec_hat, SIGMA_HS)
        return np.asarray(compute_c1(rho_arr, wd, w2_hat, w3_hat, w2vec_hat,
                                     SIGMA_HS, model="aWBII"))

    def boltzmann_init(rho_b: float) -> np.ndarray:
        ri = rho_b * np.exp(np.clip(-Vext_K / T_K, -50.0, 20.0)) * access
        return np.minimum(ri, rho_max)

    # ── Henry constant (no FMT, fast check) ───────────────────────────────────
    boltz_sum = np.exp(np.clip(-Vext_K / T_K, -700, 700)).sum() * dV
    V_cell_m3 = np.prod([np.linalg.norm(lattice[i]) for i in range(3)]) * 1e-30
    K_H = boltz_sum * 1e-30 / (K_B_PA_A3 * T_K * 1e-30) / 1e5 / AVOGADRO * 1000 / ZIF8_MW_G_PER_UC
    print(f"K_H (Boltzmann integral): {K_H:.4f} mmol / g / bar")

    # ── isotherm loop ──────────────────────────────────────────────────────────
    N_abs_arr = np.empty(len(PRESSURES))
    N_exc_arr = np.empty(len(PRESSURES))
    iters_arr = np.empty(len(PRESSURES), dtype=int)
    conv_arr  = np.empty(len(PRESSURES), dtype=bool)

    rho = None   # warm-start carry-over
    t0  = time.time()

    for i, p in enumerate(PRESSURES):
        rho_bulk = CO2_SW.bulk_density(p, T_K)   # Å⁻³
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
        N_abs = float(res.rho.sum() * dV)
        N_exc = float((res.rho - rho_bulk * access).sum() * dV)
        N_abs_arr[i] = N_abs
        N_exc_arr[i] = N_exc
        iters_arr[i] = res.iterations
        conv_arr[i]  = res.converged

        to_mmol_g = 1000.0 / (AVOGADRO * ZIF8_MW_G_PER_UC)
        print(f"  P={p:6.2f} bar  rho_b={rho_bulk:.3e} Å⁻³  "
              f"it={res.iterations:3d}  conv={res.converged}  "
              f"N_abs={N_abs*to_mmol_g:.3f} mmol/g")

    print(f"\nIsotherm done in {time.time()-t0:.1f} s")

    # ── plot ───────────────────────────────────────────────────────────────────
    to_mmol_g = 1000.0 / (AVOGADRO * ZIF8_MW_G_PER_UC)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(PRESSURES, N_abs_arr * to_mmol_g, "b-o", markersize=4, label="FMT-aWBII (abs)")
    ax.plot(PRESSURES, N_exc_arr * to_mmol_g, "b--", alpha=0.6, label="FMT-aWBII (excess)")
    ax.set_xlabel("Pressure (bar)", fontsize=13)
    ax.set_ylabel("CO₂ loading (mmol / g)", fontsize=13)
    ax.set_title(f"CO₂ / ZIF-8, T = {T_K:.0f} K  [MACE-MP-0 + FMT-aWBII]", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(PRESSURES.min() * 0.8, PRESSURES.max() * 1.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()

    out_fig = FIG_DIR / "zif8_co2_isotherm_mace.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out_fig}")

    # ── save CSV ───────────────────────────────────────────────────────────────
    import csv
    csv_path = CACHE_DIR.parent / "zif8_co2_isotherm_mace.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P_bar", "N_abs_per_uc", "N_exc_per_uc",
                    "N_abs_mmol_g", "N_exc_mmol_g", "iters", "converged"])
        for i in range(len(PRESSURES)):
            w.writerow([
                f"{PRESSURES[i]:.4f}",
                f"{N_abs_arr[i]:.5e}", f"{N_exc_arr[i]:.5e}",
                f"{N_abs_arr[i]*to_mmol_g:.4f}", f"{N_exc_arr[i]*to_mmol_g:.4f}",
                iters_arr[i], conv_arr[i],
            ])
    print(f"CSV saved:   {csv_path}")


if __name__ == "__main__":
    main()

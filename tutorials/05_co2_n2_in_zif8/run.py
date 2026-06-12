"""Tutorial 5 — Binary CO2 / N2 mixture in ZIF-8 at 298 K.

Demonstrates how to combine two single-component porecdft isotherms via
**Ideal Adsorbed Solution Theory (IAST)** to predict the mixture loading
at a chosen feed composition.

The single-component cDFT solver in porecdft handles *one* fluid at a time.
The standard route to a mixture is:

1. Compute single-gas isotherms q_i(p) for CO₂ and N₂.
2. Use IAST (Myers & Prausnitz 1965) to invert q_i(p) → mixture loadings
   at the chosen total pressure and feed composition.

We use a flue-gas-like 15:85 CO₂:N₂ feed and report:

* the individual loadings q_CO2 and q_N2 across total pressure,
* the IAST selectivity S = (q_CO2/q_N2) / (y_CO2/y_N2).

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \
        tutorials/05_co2_n2_in_zif8/run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO.parent), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(_REPO / "tutorials"))

from data_loader import STRUCT_DIR, FF_DIR, load_dreiding_ff
from porecdft.fluid import EPM2_CO2, TraPPE_N2
from porecdft.io import read_cif
from porecdft.forcefield import (
    LJPotential, CoulombPotential, QuadrupoleEFGPotential, CompositePotential,
)
from porecdft.functional import (
    make_k_grid, make_fmt_weights_hat, compute_weighted_densities,
    compute_c1, bulk_c1,
)
from porecdft.eos import CO2_PCSAFT, N2_PCSAFT
from porecdft.solver import anderson_solve
from porecdft.structure import build_supercell
from porecdft.vext import build_vext_on_grid, fibonacci_rotations

T_K = 298.0
P_TOT = np.logspace(-2, 1.5, 12)        # 0.01 … ~32 bar total pressure
Y_CO2_FEED = 0.15                       # flue-gas-like 15:85 CO2:N2 mol-fraction
SIGMA_HS_CO2 = 3.017
SIGMA_HS_N2  = 3.31
AVOGADRO     = 6.022e23
ATOMIC_MASS  = {"Zn": 65.38, "N": 14.007, "C": 12.011, "H": 1.008}

FIG_DIR = _REPO / "tutorials" / "figures"


def single_gas_isotherm(fluid, eos, sigma_hs, host, host_ff, name, P_arr,
                        use_charges=True):
    """Compute a single-component cDFT isotherm at T_K."""
    cache = Path(__file__).parent / f"vext_zif8_{name}_298K.npy"
    if cache.exists():
        vd = np.load(cache, allow_pickle=True).item()
    else:
        pots = [LJPotential(host_ff=host_ff, fluid_ff=fluid.ff, cutoff=15.0)]
        # Note: fluid.charges is a dict {label: charge} — sum is non-zero only
        # if the molecule carries real partial charges (e.g. EPM2 CO2, TraPPE N2).
        has_q = any(abs(q) > 0 for q in fluid.charges.values()) if use_charges else False
        if has_q:
            pots.append(CoulombPotential(fluid_charges=fluid.charges,
                                         sigma_smear=2.0, cutoff=15.0))
            if hasattr(fluid, "theta_zz") and abs(fluid.theta_zz) > 0:
                pots.append(QuadrupoleEFGPotential(theta_zz=fluid.theta_zz,
                                                    cutoff=15.0))
        potential = CompositePotential(pots)
        print(f"  Building Vext for {name}...")
        vd = build_vext_on_grid(host, fluid, potential,
                                orientations=fibonacci_rotations(20),
                                spacing=1.0, pbc_supercell=(2, 2, 2),
                                temperature_K=T_K,
                                cache_path=str(cache),
                                v_reject_below_K=-10000.0,
                                v_cap_above_K=+5000.0,
                                averaging="boltzmann")

    Vext_K = vd["vext_avg"]
    dV     = vd["dV"]
    access = np.isfinite(Vext_K) & (Vext_K < 50.0 * T_K)
    Nx, Ny, Nz = Vext_K.shape
    Lx, Ly, Lz = np.linalg.norm(host.lattice, axis=1)
    KX, KY, KZ, K = make_k_grid((Nx, Ny, Nz),
                                 dx=Lx/Nx, dy=Ly/Ny, dz=Lz/Nz)
    w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, sigma_hs)
    RHO_MAX = 0.45 * 6.0 / (np.pi * sigma_hs ** 3)

    def c1_fn(rho):
        wd = compute_weighted_densities(rho, w2_hat, w3_hat, w2vec_hat, sigma_hs)
        return np.asarray(compute_c1(rho, wd, w2_hat, w3_hat, w2vec_hat,
                                      sigma_hs, model="aWBII"))

    fw_amu = sum(ATOMIC_MASS.get(s, 0.0) for s in host.species)
    fw_g   = fw_amu / AVOGADRO
    to_mmol_per_g = (1.0 / AVOGADRO) * 1000.0 / fw_g

    N_arr = np.empty(len(P_arr))
    rho_prev = None
    rho_prev_b = None
    for i, P in enumerate(P_arr):
        rho_b = float(eos.bulk_density(P, T_K))
        c1_b  = bulk_c1(rho_b, sigma_hs, model="aWBII")
        if rho_prev is not None and rho_prev_b:
            rho0 = np.where(access, np.clip(rho_prev * (rho_b / rho_prev_b),
                                            1e-16, RHO_MAX), 1e-16)
        else:
            rho0 = np.minimum(rho_b * np.exp(np.clip(-Vext_K / T_K, -50, 20)) * access,
                              RHO_MAX)
        res = anderson_solve(rho0, rho_b, Vext_K, T_K, c1_fn, c1_b,
                             m=6, beta=0.3, max_iter=800, tol=1e-4,
                             accessibility_mask=access, log_clip=25.0,
                             safeguard_alpha=0.02, picard_warmup=30,
                             step_clip=2.0, rho_max=RHO_MAX)
        rho_prev, rho_prev_b = res.rho.copy(), rho_b
        N_arr[i] = float(res.rho.sum() * dV) * to_mmol_per_g
    return N_arr


# ── IAST routines ───────────────────────────────────────────────────────────

def _spreading_pressure(p_grid: np.ndarray, q_grid: np.ndarray) -> callable:
    """Return Π(p) = ∫₀ᵖ q(p') / p' dp' as a callable, via cumulative trapezoid
    on log p (handles wide pressure ranges robustly).
    """
    # Sort by p; insert (0, 0) at the front for the integral lower bound.
    order = np.argsort(p_grid)
    p = np.concatenate([[1e-30], p_grid[order]])
    q = np.concatenate([[0.0],   q_grid[order]])
    # piecewise-linear q on log-p → Π(p) = ∫ (q/p) dp = ∫ q d(log p)
    # use simple trapezoidal on (log p, q)
    log_p = np.log(p)
    integrand = q
    cum_Pi = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                              * (log_p[1:] - log_p[:-1]))])
    # Build a 1-D interpolator
    def Pi(p_query):
        lp = np.log(np.maximum(p_query, 1e-30))
        return np.interp(lp, log_p, cum_Pi)
    return Pi


def iast_loading(p_total: float, y_co2: float,
                 P_grid: np.ndarray, q_co2_grid: np.ndarray, q_n2_grid: np.ndarray):
    """Myers-Prausnitz IAST.  Returns (q_co2_mix, q_n2_mix) at (P, y_CO2).

    Strategy: find the equivalent single-component pressures p_i* such that
    Π_CO2(p_co2*) = Π_N2(p_n2*) AND  y_i · P_total = x_i · p_i*  with x_CO2+x_N2=1.
    """
    Pi_co2 = _spreading_pressure(P_grid, q_co2_grid)
    Pi_n2  = _spreading_pressure(P_grid, q_n2_grid)
    P_min, P_max = P_grid.min(), P_grid.max()

    # Solve for x_CO2: spreading-pressure equality at p_i* = y_i P / x_i.
    def f(x_co2):
        x_n2 = 1.0 - x_co2
        if x_co2 <= 0 or x_n2 <= 0:
            return 1e10
        p_co2_star = y_co2 * p_total / x_co2
        p_n2_star  = (1 - y_co2) * p_total / x_n2
        if not (P_min <= p_co2_star <= P_max and P_min <= p_n2_star <= P_max):
            # clip & still return a meaningful sign
            p_co2_star = float(np.clip(p_co2_star, P_min, P_max))
            p_n2_star  = float(np.clip(p_n2_star,  P_min, P_max))
        return Pi_co2(p_co2_star) - Pi_n2(p_n2_star)

    try:
        x_co2 = brentq(f, 1e-6, 1.0 - 1e-6, xtol=1e-6)
    except ValueError:
        # cannot bracket — return single-gas as approximation
        x_co2 = y_co2

    x_n2 = 1.0 - x_co2
    p_co2_star = y_co2 * p_total / x_co2
    p_n2_star  = (1 - y_co2) * p_total / x_n2
    p_co2_star = float(np.clip(p_co2_star, P_min, P_max))
    p_n2_star  = float(np.clip(p_n2_star,  P_min, P_max))
    # Total loading from IAST mixing rule
    q_co2_pure = np.interp(p_co2_star, P_grid, q_co2_grid)
    q_n2_pure  = np.interp(p_n2_star,  P_grid, q_n2_grid)
    q_total = 1.0 / (x_co2 / max(q_co2_pure, 1e-30) + x_n2 / max(q_n2_pure, 1e-30))
    return x_co2 * q_total, x_n2 * q_total


def main():
    host = read_cif(str(STRUCT_DIR / "ZIF-8.cif"))
    host = host.assign_charges({s: 0.0 for s in set(host.species)},
                                source="ZIF-8 framework neutral")
    host_ff = load_dreiding_ff(FF_DIR / "DREIDING.dat")
    print(f"ZIF-8 host: {len(host.species)} atoms")

    print("\n=== Single-gas isotherms ===")
    print("  CO2 ...")
    q_co2 = single_gas_isotherm(EPM2_CO2, CO2_PCSAFT, SIGMA_HS_CO2,
                                 host, host_ff, "CO2", P_TOT)
    print("  N2 ...")
    q_n2  = single_gas_isotherm(TraPPE_N2, N2_PCSAFT, SIGMA_HS_N2,
                                 host, host_ff, "N2",  P_TOT)

    print("\n=== IAST mixture (15:85 CO2:N2 feed) ===")
    q_co2_mix = np.empty(len(P_TOT))
    q_n2_mix  = np.empty(len(P_TOT))
    for i, P in enumerate(P_TOT):
        q_co2_mix[i], q_n2_mix[i] = iast_loading(P, Y_CO2_FEED,
                                                  P_TOT, q_co2, q_n2)
    sel = (q_co2_mix / np.where(q_n2_mix > 0, q_n2_mix, np.nan)) / (Y_CO2_FEED / (1 - Y_CO2_FEED))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].semilogx(P_TOT, q_co2,     "o--", color="#d62728", lw=1.6, ms=4, label="CO₂ pure")
    axes[0].semilogx(P_TOT, q_n2,      "s--", color="#1f77b4", lw=1.6, ms=4, label="N₂ pure")
    axes[0].semilogx(P_TOT, q_co2_mix, "o-",  color="#d62728", lw=2.0, ms=6, label="CO₂ in mix")
    axes[0].semilogx(P_TOT, q_n2_mix,  "s-",  color="#1f77b4", lw=2.0, ms=6, label="N₂ in mix")
    axes[0].set_xlabel("Total pressure (bar)", fontsize=12)
    axes[0].set_ylabel(r"loading (mmol g$^{-1}$)", fontsize=12)
    axes[0].set_title("Pure + mixture isotherms",
                      fontsize=11, fontweight="bold")
    axes[0].grid(alpha=0.3, ls=":")
    axes[0].legend(fontsize=9)

    axes[1].semilogx(P_TOT, sel, "o-", color="#2ca02c", lw=2.0, ms=6)
    axes[1].set_xlabel("Total pressure (bar)", fontsize=12)
    axes[1].set_ylabel(r"S$_{CO_2/N_2}$", fontsize=12)
    axes[1].set_title("IAST selectivity (15 % CO₂)",
                      fontsize=11, fontweight="bold")
    axes[1].grid(alpha=0.3, ls=":")
    axes[1].axhline(1, color="grey", lw=0.8, ls=":")
    fig.suptitle("Tutorial 5 — CO₂ / N₂ binary mixture in ZIF-8 at 298 K (IAST)",
                 fontsize=12, fontweight="bold")
    out = FIG_DIR / "05_co2_n2_in_zif8.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()

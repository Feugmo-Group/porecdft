"""Tutorial 6 — CH4 / C2H6 binary mixture in 2,3-Dha-Tph COF at 298 K.

Reproduces THE adsorption-isotherm figure of Stierle & Gross 2024
(\\emph{Chem.\\ Eng.\\ Sci.} \\textbf{298}, 120380), the only such figure in
the paper.  Original conditions:

    Host             : 2,3-Dha-Tph (QEq charges, LJ-only Vext for this run)
    Fluids           : CH4 and C2H6, PC-SAFT bulk (Gross & Sadowski 2001)
    Temperature      : 298 K
    Total pressure   : 0.1 ... 50 bar
    Vapour comp.     : x_CH4^V = 0.6  (40 % C2H6)

We compute the two single-component cDFT isotherms first, then apply
Myers--Prausnitz IAST to predict the mixture loadings ``N_i / unit cell''
(same y-axis as the paper figure).

Black markers in the figure are values digitised from the paper image
``1-s2.0-S0009250924006808-gr003_lrg.jpg``.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \\
        tutorials/06_ch4_c2h6_in_dha_tph/run.py
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

from data_loader import STRUCT_DIR, FF_DIR, load_dreiding_ff, load_pcsaft_fluid, make_pcsaft_eos
from porecdft.fluid.base import Fluid
from porecdft.io import read_cif
from porecdft.io.forcefield import FFEntry
from porecdft.forcefield import LJPotential, CompositePotential
from porecdft.functional import (
    make_k_grid, make_fmt_weights_hat, compute_weighted_densities,
    compute_c1, bulk_c1,
)
from porecdft.functional.pcsaft_pure import PurePCSAFTFunctional, hsd_pcsaft
from porecdft.solver import anderson_solve
from porecdft.vext import build_vext_on_grid, fibonacci_rotations

T_K       = 298.0
X_CH4_V   = 0.6
P_TOT     = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 7.0,
                      10.0, 15.0, 20.0, 30.0, 40.0, 50.0])
FIG_DIR   = _REPO / "tutorials" / "figures"
CACHE_DIR = Path(__file__).parent

# GCMC reference points digitised from Stierle 2024 gr003.
GCMC_REF = {
    "C2H6": {"p": [0.5, 2.0, 7.0, 19.0, 50.0],
             "N": [0.5, 3.8, 8.5, 15.4, 19.1]},
    "CH4":  {"p": [0.5, 2.0, 7.0, 19.0, 50.0],
             "N": [0.3, 1.0, 2.1, 4.2, 6.7]},
}


def make_simple_fluid(name: str):
    m, sigma, eps, M = load_pcsaft_fluid(name)
    label = name.title()
    fluid = Fluid(
        name=label,
        body_sites=np.zeros((1, 3)),
        site_labels=[label],
        ff={label: FFEntry(label, sigma, eps, source="gross2001.json")},
        charges={label: 0.0},
        molar_mass=M,
    )
    eos = make_pcsaft_eos(name)
    return fluid, eos, float(sigma)


def single_gas_isotherm(name: str, host, host_ff, P_arr):
    fluid, eos, sigma_hs = make_simple_fluid(name)
    # Reload raw PC-SAFT (m, σ, ε/k) for the dispersion + chain functional.
    m_chain, sigma_seg, eps_k, _ = load_pcsaft_fluid(name)
    print(f"\n{name.title()} PC-SAFT:  m={m_chain:.3f}  sigma={sigma_seg:.3f}  eps/k={eps_k:.2f}")
    # Use the temperature-dependent Barker--Henderson HSD (matches Stierle 2024).
    sigma_hs_T = hsd_pcsaft(sigma_seg, eps_k, T_K)
    print(f"  HSD(T={T_K}) = {sigma_hs_T:.3f}  (vs raw σ = {sigma_seg:.3f})")
    sigma_hs = sigma_hs_T
    pcsaft_F = PurePCSAFTFunctional(m=m_chain, sigma=sigma_seg, eps_k=eps_k, T=T_K)

    cache = CACHE_DIR / f"vext_dha_{name}_298K.npy"
    if cache.exists():
        vd = np.load(cache, allow_pickle=True).item()
    else:
        potential = CompositePotential([
            LJPotential(host_ff=host_ff, fluid_ff=fluid.ff, cutoff=15.0),
        ])
        print(f"  Building Vext for {name}...")
        vd = build_vext_on_grid(host, fluid, potential,
                                orientations=fibonacci_rotations(20),
                                spacing=1.2, pbc_supercell=(2, 2, 2),
                                temperature_K=T_K,
                                cache_path=str(cache),
                                v_reject_below_K=-10000.0,
                                v_cap_above_K=+5000.0,
                                averaging="boltzmann")

    Vext_K = vd["vext_avg"]
    dV     = float(vd["dV"])
    # Exclude strongly repulsive voxels (V > 5 kT) from the convergence mask.
    # At these sites ρ ≈ 0 and the hard-chain c¹ ≈ (m−1)·ln(ρ) → −∞,
    # creating huge EL residuals that prevent convergence without affecting N.
    access = np.isfinite(Vext_K) & (Vext_K < 5.0 * T_K)
    Nx, Ny, Nz = Vext_K.shape
    Lx, Ly, Lz = np.linalg.norm(host.lattice, axis=1)
    KX, KY, KZ, K = make_k_grid((Nx, Ny, Nz), dx=Lx/Nx, dy=Ly/Ny, dz=Lz/Nz)
    w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, sigma_hs)
    # PC-SAFT dispersion + chain weights — on the rfft k-grid (jnp.fft.rfftn).
    _, _, _, K_rfft = make_k_grid((Nx, Ny, Nz), dx=Lx/Nx, dy=Ly/Ny, dz=Lz/Nz,
                                    real_fft=True)
    w_disp_hat   = pcsaft_F._weight_disp_hat(K_rfft)
    w_lambd_hat, w_zeta3_hat = pcsaft_F._weight_chain_hat(K_rfft)
    # Packing-fraction cap η = (π/6) m d³ ρ ≤ 0.45  ⇒  ρ_chain_max = packing/(m·v_seg).
    RHO_MAX = 0.45 * 6.0 / (np.pi * sigma_hs ** 3 * m_chain)

    # Stierle 2024 Eq (13a-f): every FMT weight carries a factor m (= chain
    # segment count). Equivalently, compute FMT on the *segment* density
    # ρ_seg = m·ρ_chain and chain-rule back: dF/dρ_chain = m · dF/dρ_seg.
    def c1_fn(rho):
        rho_seg = m_chain * rho
        wd = compute_weighted_densities(rho_seg, w2_hat, w3_hat, w2vec_hat, sigma_hs)
        c1_fmt_seg = np.asarray(compute_c1(rho_seg, wd, w2_hat, w3_hat, w2vec_hat,
                                            sigma_hs, model="aWBII"))
        c1_fmt = m_chain * c1_fmt_seg
        c1_dc  = np.asarray(pcsaft_F.c1(rho, dV, w_disp_hat,
                                         w_lambd_hat, w_zeta3_hat))
        return c1_fmt + c1_dc

    # For chain molecules Stierle 2024 uses segment-based ideal gas entropy,
    # giving EL: ln(ρ/ρ_b) = (c¹ − c¹_b − β·V) / m.  Implemented by passing
    # Vext/m and c¹/m to the standard EL solver (equivalent reformulation).
    # For m=1 (methane), _inv_m=1 so this is a no-op.
    _inv_m = 1.0 / m_chain

    def c1_fn_eff(rho, _fn=c1_fn, _inv=_inv_m):
        return _fn(rho) * _inv

    N_per_uc = np.empty(len(P_arr))
    rho_prev = None
    rho_prev_b = None
    print(f"\n  m={m_chain:.4f}  d={sigma_hs:.4f} A  inv_m={_inv_m:.4f}")
    print(f"  {'P (bar)':>10}  {'N/uc':>8}  {'conv':>5}  {'iters':>6}")
    for i, P in enumerate(P_arr):
        rho_b = float(eos.bulk_density(P, T_K))
        # Bulk c1 with the m-scaled FMT (consistent with c1_fn above).
        c1_b_fmt = m_chain * bulk_c1(m_chain * rho_b, sigma_hs, model="aWBII")
        c1_b     = c1_b_fmt + pcsaft_F.bulk_c1(rho_b)
        if rho_prev is not None and rho_prev_b:
            rho0 = np.where(access, np.clip(rho_prev * (rho_b / rho_prev_b),
                                            1e-16, RHO_MAX), 1e-16)
        else:
            rho0 = np.minimum(rho_b * np.exp(np.clip(-Vext_K / T_K, -50, 20)) * access,
                              RHO_MAX)
        # Conservative parameters work for both m=1 (CH4) and m>1 (C2H6).
        # float32 PC-SAFT caps achievable tolerance at ~0.1 in log-density space;
        # step_clip=0.5 and picard_warmup=100 prevent oscillations at high density.
        res = anderson_solve(
            rho0, rho_b,
            Vext_K * _inv_m, T_K,   # effective Vext/m for segment-based EL
            c1_fn_eff,               # effective c¹/m
            c1_b * _inv_m,           # effective bulk c¹/m
            m=6,
            beta=0.15,
            max_iter=2000,
            tol=0.1,
            accessibility_mask=access,
            log_clip=15.0,
            safeguard_alpha=0.02,
            picard_warmup=100,
            step_clip=0.5,
            rho_max=RHO_MAX)
        rho_prev, rho_prev_b = res.rho.copy(), rho_b
        N_per_uc[i] = float(res.rho.sum() * dV)
        print(f"  {P:10.3f}  {N_per_uc[i]:8.3f}  {str(res.converged):>5}  {res.iterations:>6}")
    return N_per_uc


# ── IAST (Myers-Prausnitz) ─────────────────────────────────────────────────

def _spreading_pressure(p_grid: np.ndarray, q_grid: np.ndarray):
    order = np.argsort(p_grid)
    p = np.concatenate([[1e-30], p_grid[order]])
    q = np.concatenate([[0.0],   q_grid[order]])
    log_p = np.log(p)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1])
                                            * (log_p[1:] - log_p[:-1]))])
    def Pi(pq):
        return np.interp(np.log(np.maximum(pq, 1e-30)), log_p, cum)
    return Pi


def iast_loading(P_total, y_ch4, P_grid, q_ch4_pure, q_c2h6_pure):
    Pi_ch4  = _spreading_pressure(P_grid, q_ch4_pure)
    Pi_c2h6 = _spreading_pressure(P_grid, q_c2h6_pure)
    P_min, P_max = P_grid.min(), P_grid.max()

    def f(x_ch4):
        x_c2h6 = 1.0 - x_ch4
        if x_ch4 <= 0 or x_c2h6 <= 0:
            return 1e10
        p_ch4_star  = y_ch4 * P_total / x_ch4
        p_c2h6_star = (1 - y_ch4) * P_total / x_c2h6
        p_ch4_star  = float(np.clip(p_ch4_star,  P_min, P_max))
        p_c2h6_star = float(np.clip(p_c2h6_star, P_min, P_max))
        return Pi_ch4(p_ch4_star) - Pi_c2h6(p_c2h6_star)

    try:
        x_ch4 = brentq(f, 1e-6, 1.0 - 1e-6, xtol=1e-6)
    except ValueError:
        x_ch4 = y_ch4
    x_c2h6 = 1.0 - x_ch4
    p_ch4_star  = float(np.clip(y_ch4 * P_total / x_ch4,        P_min, P_max))
    p_c2h6_star = float(np.clip((1 - y_ch4) * P_total / x_c2h6, P_min, P_max))
    q_ch4_p  = np.interp(p_ch4_star,  P_grid, q_ch4_pure)
    q_c2h6_p = np.interp(p_c2h6_star, P_grid, q_c2h6_pure)
    q_total = 1.0 / (x_ch4 / max(q_ch4_p, 1e-30)
                     + x_c2h6 / max(q_c2h6_p, 1e-30))
    return x_ch4 * q_total, x_c2h6 * q_total


def main():
    cif = STRUCT_DIR / "Dha_Tph_QEq.cif"
    host = read_cif(str(cif))
    print(f"2,3-Dha-Tph host: {len(host.species)} atoms; cell = {host.lattice.diagonal()}")
    host_ff = load_dreiding_ff(FF_DIR / "DREIDING.dat")

    q_ch4  = single_gas_isotherm("methane", host, host_ff, P_TOT)
    q_c2h6 = single_gas_isotherm("ethane",  host, host_ff, P_TOT)

    print(f"\n=== IAST mixture at x_CH4 = {X_CH4_V} ===")
    q_ch4_mix  = np.empty(len(P_TOT))
    q_c2h6_mix = np.empty(len(P_TOT))
    for i, P in enumerate(P_TOT):
        q_ch4_mix[i], q_c2h6_mix[i] = iast_loading(P, X_CH4_V, P_TOT, q_ch4, q_c2h6)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    ax.plot(P_TOT, q_c2h6_mix, "-",  color="black", lw=2.0,
            label=r"C$_2$H$_6$ (DFT)")
    ax.plot(P_TOT, q_ch4_mix,  "--", color="black", lw=1.6,
            label=r"CH$_4$ (DFT)")
    ax.scatter(GCMC_REF["C2H6"]["p"], GCMC_REF["C2H6"]["N"],
               s=70, c="black", marker="o", label=r"C$_2$H$_6$ (GCMC)")
    ax.scatter(GCMC_REF["CH4"]["p"],  GCMC_REF["CH4"]["N"],
               s=70, facecolors="white", edgecolors="black",
               linewidths=1.4, marker="o", label=r"CH$_4$ (GCMC)")
    ax.set_xlabel(r"$p$ / bar", fontsize=13)
    ax.set_ylabel(r"$N_i$ / unit cell", fontsize=13)
    ax.set_title(r"Tutorial 6 — CH$_4$ / C$_2$H$_6$ in 2,3-Dha-Tph at 298 K, $x_{\rm CH_4}^V = 0.6$",
                 fontsize=11, fontweight="bold")
    ax.text(0.97, 0.95, r"$T$ = 298 K" + "\n" + r"$x_{\rm CH_4}^V = 0.6$",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.set_xlim(0, 50)
    gcmc_max = max(max(GCMC_REF["C2H6"]["N"]), max(GCMC_REF["CH4"]["N"]))
    ax.set_ylim(0, max(q_c2h6_mix.max(), q_c2h6.max(), gcmc_max) * 1.1)

    out = FIG_DIR / "06_ch4_c2h6_in_dha_tph.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()

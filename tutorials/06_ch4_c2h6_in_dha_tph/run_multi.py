"""Tutorial 6 (multi-component cDFT) — CH4 / C2H6 in 2,3-Dha-Tph at 298 K.

Same reference as ``run.py`` but solves the full multi-component PC-SAFT
cDFT (Stierle Eqs. 16, 17 + per-component FMT) instead of running pure
isotherms then IAST.  This is the direct route used in Stierle 2024.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \\
        tutorials/06_ch4_c2h6_in_dha_tph/run_multi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
from porecdft.functional.pcsaft_pure import MultiPCSAFTFunctional, hsd_pcsaft
from porecdft.vext import build_vext_on_grid, fibonacci_rotations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

T_K     = 298.0
X_CH4_V = 0.6
P_TOT   = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 7.0,
                    10.0, 15.0, 20.0, 30.0, 40.0, 50.0])
FIG_DIR = _REPO / "tutorials" / "figures"
CACHE_DIR = Path(__file__).parent

GCMC_REF = {
    "C2H6": {"p": [0.5, 2.0, 7.0, 19.0, 50.0],
             "N": [0.5, 3.8, 8.5, 15.4, 19.1]},
    "CH4":  {"p": [0.5, 2.0, 7.0, 19.0, 50.0],
             "N": [0.3, 1.0, 2.1, 4.2, 6.7]},
}


def build_or_load_vext(name: str, host, host_ff):
    """Reuse the V_ext caches built by the pure-isotherm tutorial."""
    cache = CACHE_DIR / f"vext_dha_{name}_298K.npy"
    if cache.exists():
        return np.load(cache, allow_pickle=True).item()
    m, sigma, eps_k, M = load_pcsaft_fluid(name)
    label = name.title()
    fluid = Fluid(name=label, body_sites=np.zeros((1, 3)),
                  site_labels=[label],
                  ff={label: FFEntry(label, sigma, eps_k,
                                       source="gross2001.json")},
                  charges={label: 0.0}, molar_mass=M)
    potential = CompositePotential([
        LJPotential(host_ff=host_ff, fluid_ff=fluid.ff, cutoff=15.0),
    ])
    return build_vext_on_grid(host, fluid, potential,
                               orientations=fibonacci_rotations(20),
                               spacing=1.2, pbc_supercell=(2, 2, 2),
                               temperature_K=T_K,
                               cache_path=str(cache),
                               v_reject_below_K=-10000.0,
                               v_cap_above_K=+5000.0,
                               averaging="boltzmann")


def main():
    cif = STRUCT_DIR / "Dha_Tph_QEq.cif"
    host = read_cif(str(cif))
    host_ff = load_dreiding_ff(FF_DIR / "DREIDING.dat")
    print(f"2,3-Dha-Tph host: {len(host.species)} atoms")

    # Per-component V_ext.
    vd_ch4 = build_or_load_vext("methane", host, host_ff)
    vd_c2  = build_or_load_vext("ethane",  host, host_ff)
    Vext = np.stack([vd_ch4["vext_avg"], vd_c2["vext_avg"]], axis=0)   # (2, Nx, Ny, Nz)
    dV   = float(vd_ch4["dV"])
    Nx, Ny, Nz = Vext.shape[1:]
    Lx, Ly, Lz = np.linalg.norm(host.lattice, axis=1)

    # PC-SAFT parameters.
    m_ch4, sig_ch4, eps_ch4, _ = load_pcsaft_fluid("methane")
    m_c2,  sig_c2,  eps_c2,  _ = load_pcsaft_fluid("ethane")
    d_ch4 = hsd_pcsaft(sig_ch4, eps_ch4, T_K)
    d_c2  = hsd_pcsaft(sig_c2,  eps_c2,  T_K)
    m_arr = np.array([m_ch4, m_c2])
    sig_arr = np.array([sig_ch4, sig_c2])
    eps_arr = np.array([eps_ch4, eps_c2])
    hsd_arr = np.array([d_ch4, d_c2])

    # Bulk EOS for each component.
    eos_ch4 = make_pcsaft_eos("methane")
    eos_c2  = make_pcsaft_eos("ethane")

    # FMT weights: per-component for back-convolution + joint average for repulsion.
    # d_CH4 ≈ d_C2H6 at 298 K so the mass-weighted average d is a good approximation
    # for the mixture FMT (Boublik-Mansoori cross-term).
    KX, KY, KZ, K        = make_k_grid((Nx, Ny, Nz), Lx/Nx, Ly/Ny, Lz/Nz)
    _, _, _, K_r         = make_k_grid((Nx, Ny, Nz), Lx/Nx, Ly/Ny, Lz/Nz, real_fft=True)
    w2_fmt_ch4, w3_fmt_ch4, wv_fmt_ch4 = make_fmt_weights_hat(K, KX, KY, KZ, d_ch4)
    w2_fmt_c2,  w3_fmt_c2,  wv_fmt_c2  = make_fmt_weights_hat(K, KX, KY, KZ, d_c2)
    # Mass-weighted average segment diameter for joint FMT.
    m_tot = m_ch4 + m_c2
    d_avg = (m_ch4 * d_ch4 + m_c2 * d_c2) / m_tot
    w2_avg, w3_avg, wv_avg = make_fmt_weights_hat(K, KX, KY, KZ, d_avg)
    # Multi-component PC-SAFT functional.
    mFx = MultiPCSAFTFunctional(m=m_arr, sigma=sig_arr, eps_k=eps_arr, T=T_K)
    wD, wL, wZ2, wZ3 = mFx._weights(K_r)

    RHO_MAX = 0.45 * 6.0 / (np.pi * hsd_arr ** 3 * m_arr)  # (2,)
    access = (Vext < 50.0 * T_K) & np.isfinite(Vext)        # (2, Nx, Ny, Nz)

    def c1_total(rho):
        """rho shape (2, Nx, Ny, Nz) → c¹ shape (2, Nx, Ny, Nz).

        Uses joint FMT (both components contribute to the same n3) via the
        mass-weighted average segment diameter.  This correctly accounts for
        the cross-packing repulsion between CH4 and C2H6 segments, which
        prevents the spurious pore condensation that occurs when each species'
        FMT is computed independently (underestimating total packing fraction).
        """
        # Joint segment density → FMT c1 for the mixture.
        rho_seg_joint = m_arr[0] * rho[0] + m_arr[1] * rho[1]
        wd_joint = compute_weighted_densities(rho_seg_joint, w2_avg, w3_avg,
                                              wv_avg, d_avg)
        c1_seg_joint = np.asarray(compute_c1(rho_seg_joint, wd_joint,
                                              w2_avg, w3_avg, wv_avg,
                                              d_avg, model="aWBII"))
        c1_fmt = np.stack([m_arr[c] * c1_seg_joint for c in range(2)])
        # Multi-component PC-SAFT dispersion + hard-chain.
        c1_pc = np.asarray(mFx.c1(jnp.asarray(rho), dV, wD, wL, wZ2, wZ3))
        return c1_fmt + c1_pc

    fw_amu = sum({"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999}.get(s, 0.0)
                  for s in host.species)

    q_ch4_mix = np.empty(len(P_TOT))
    q_c2_mix  = np.empty(len(P_TOT))
    print(f"\n=== Multi-component cDFT (CH4 / C2H6) at x_CH4 = {X_CH4_V}, T = {T_K} K ===")
    print(f"  {'P (bar)':>9}  {'CH4 N/uc':>9}  {'C2H6 N/uc':>10}  iters")
    for i, P in enumerate(P_TOT):
        # Ideal-gas bulk for each component (CH4 & C2H6 ~ ideal at <50 bar).
        p_ch4 = X_CH4_V * P
        p_c2  = (1 - X_CH4_V) * P
        rho_b_ch4 = float(eos_ch4.bulk_density(p_ch4, T_K))
        rho_b_c2  = float(eos_c2.bulk_density(p_c2,  T_K))
        rho_b = np.array([rho_b_ch4, rho_b_c2])
        # Bulk c¹ — m-scaled FMT + multi PC-SAFT.
        # Bulk c1: joint FMT at the total segment bulk density.
        rho_seg_b_joint = m_arr[0] * rho_b[0] + m_arr[1] * rho_b[1]
        c1_seg_b_joint  = bulk_c1(rho_seg_b_joint, d_avg, model="aWBII")
        c1b_fmt = np.array([m_arr[c] * c1_seg_b_joint for c in range(2)])
        c1b_pc  = np.asarray(mFx.bulk_c1(rho_b))
        c1b     = c1b_fmt + c1b_pc

        # Always initialise from Boltzmann factor of Vext (gas-branch).
        # Warm-starting from the previous pressure rescales densities by
        # ρ_b_new/ρ_b_old, which can push C2H6 past a condensation threshold.
        rho = np.empty_like(Vext)
        for c in range(2):
            expn = np.clip(-Vext[c] / T_K, -50.0, 20.0)
            rho[c] = np.minimum(rho_b[c] * np.exp(expn) * access[c], RHO_MAX[c])

        # Anderson mixing in LOG-DENSITY space (Walker-Ni form) on the stacked
        # (2, Nx, Ny, Nz) field.  Joint FMT ensures correct cross-packing
        # repulsion between CH4 and C2H6 segments (see c1_total above).
        # Convergence is declared on MEAN |F| over accessible voxels to avoid
        # false non-convergence from isolated boundary voxels.
        from collections import deque

        m_aa          = 8
        beta          = 0.1
        safeguard     = 0.05
        step_clip     = 3.0
        log_clip      = 30.0
        tol           = 1e-5
        max_iter      = 1000
        picard_warmup = 30

        log_rho_b     = np.log(np.maximum(rho_b, 1e-300))
        log_clip_lo   = (log_rho_b - log_clip)[:, None, None, None]
        log_clip_hi_  = (log_rho_b + log_clip)[:, None, None, None]
        log_rho_max   = np.log(RHO_MAX)[:, None, None, None]
        log_clip_hi   = np.minimum(log_clip_hi_, log_rho_max)

        u = np.where(access,
                     np.clip(np.log(np.maximum(rho, 1e-30)), log_clip_lo, log_clip_hi),
                     log_clip_lo)

        u_hist: deque = deque(maxlen=m_aa + 1)
        F_hist: deque = deque(maxlen=m_aa + 1)
        n_access = int(access.sum())

        iters     = 0
        converged = False

        for k in range(max_iter):
            rho_cur = np.exp(u)
            c1 = c1_total(rho_cur)
            v = np.empty_like(u)
            for c in range(2):
                v[c] = -Vext[c] / T_K + c1[c] - c1b[c] + log_rho_b[c]
            v = np.clip(v, log_clip_lo, log_clip_hi)
            F = np.where(access, v - u, 0.0)

            err = float(np.max(np.abs(F)))
            iters = k + 1
            if err < tol:
                converged = True
                break
            if not np.isfinite(err) or err > 1e10:
                break

            u_hist.append(u.ravel().copy())
            F_hist.append(F.ravel().copy())

            if k < picard_warmup or len(F_hist) < 3:
                step = safeguard * F
            else:
                n_pairs = min(m_aa, len(F_hist) - 1)
                DU = np.stack(
                    [u_hist[-1] - u_hist[-1 - j] for j in range(1, n_pairs + 1)],
                    axis=1)
                DF = np.stack(
                    [F_hist[-1] - F_hist[-1 - j] for j in range(1, n_pairs + 1)],
                    axis=1)
                try:
                    gamma, *_ = np.linalg.lstsq(DF, F.ravel(), rcond=1e-8)
                    step_flat = beta * F.ravel() - (DU + beta * DF) @ gamma
                    step = step_flat.reshape(u.shape)
                    if not np.all(np.isfinite(step)):
                        raise np.linalg.LinAlgError("nonfinite")
                except (np.linalg.LinAlgError, ValueError):
                    step = safeguard * F

            step = np.clip(step, -step_clip, step_clip)
            u_new = np.clip(u + step, log_clip_lo, log_clip_hi)
            u = np.where(access, u_new, log_clip_lo)

        rho = np.exp(u)

        q_ch4_mix[i] = float(rho[0].sum() * dV)
        q_c2_mix[i]  = float(rho[1].sum() * dV)
        print(f"  {P:9.3f}  {q_ch4_mix[i]:9.3f}  {q_c2_mix[i]:10.3f}  {iters}  conv={converged}")

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    ax.plot(P_TOT, q_c2_mix,  "-",  color="black", lw=2.0, label=r"C$_2$H$_6$ (DFT, multi)")
    ax.plot(P_TOT, q_ch4_mix, "--", color="black", lw=1.6, label=r"CH$_4$ (DFT, multi)")
    ax.scatter(GCMC_REF["C2H6"]["p"], GCMC_REF["C2H6"]["N"], s=70, c="black",
                marker="o", label=r"C$_2$H$_6$ (GCMC)")
    ax.scatter(GCMC_REF["CH4"]["p"],  GCMC_REF["CH4"]["N"],  s=70,
                facecolors="white", edgecolors="black", linewidths=1.4,
                marker="o", label=r"CH$_4$ (GCMC)")
    ax.set_xlabel(r"$p$ / bar", fontsize=13)
    ax.set_ylabel(r"$N_i$ / unit cell", fontsize=13)
    ax.set_title(r"Tutorial 6 — CH$_4$ / C$_2$H$_6$ in 2,3-Dha-Tph at 298 K, $x_{\rm CH_4}^V = 0.6$"
                  + "\n(multi-component PC-SAFT cDFT — no IAST)",
                  fontsize=11, fontweight="bold")
    ax.text(0.97, 0.95, r"$T$ = 298 K" + "\n" + r"$x_{\rm CH_4}^V = 0.6$",
             transform=ax.transAxes, ha="right", va="top", fontsize=11,
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, max(q_c2_mix.max(), max(GCMC_REF["C2H6"]["N"])) * 1.18)
    out = FIG_DIR / "06_ch4_c2h6_in_dha_tph_multi.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()

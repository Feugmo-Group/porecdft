"""Tutorial 6 — CH4 / C2H6 in 2,3-Dha-Tph COF, Hydra-configurable version.

Every parameter is specified in conf/config.yaml (or its sub-groups) and can
be overridden from the command line without editing this file.

Run with defaults::

    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \\
        tutorials/06_ch4_c2h6_in_dha_tph/run_hydra.py

GPU run (single flag)::

    python run_hydra.py compute=cuda_float32

Quick test with coarse grid::

    python run_hydra.py vext=fast solver.max_iter=500 \\
                        run.n_pressure=5 run.experiment=quick_test

Parameter sweep (Hydra multirun)::

    python run_hydra.py --multirun \\
        run.temperature_K=298,350,400 run.experiment=T_sweep

Each run writes to::

    outputs/<run.experiment>/<YYYY-MM-DD>/<HH-MM-SS>/
        .hydra/config.yaml      ← full config snapshot (reproducible)
        .hydra/overrides.yaml   ← only what was changed from defaults
        vext_ch4_298K.npy       ← Vext checkpoint (re-used on repeated runs)
        vext_c2h6_298K.npy
        isotherm.npz            ← N(P) arrays
        06_ch4_c2h6_in_dha_tph.png
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO.parent), str(_REPO), str(_REPO / "tutorials")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import hydra
from omegaconf import DictConfig, OmegaConf

from porecdft.conf_schema import register_configs

register_configs()   # must be called before @hydra.main


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged physics from run.py)
# ─────────────────────────────────────────────────────────────────────────────

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
from porecdft.compute_config import ComputeConfig

AVOGADRO = 6.022e23

GCMC_REF = {
    "C2H6": {"p": [0.5, 2.0, 7.0, 19.0, 50.0], "N": [0.5, 3.8, 8.5, 15.4, 19.1]},
    "CH4":  {"p": [0.5, 2.0, 7.0, 19.0, 50.0], "N": [0.3, 1.0, 2.1, 4.2,  6.7]},
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


def single_gas_isotherm(name, host, host_ff, P_arr, T_K, vcfg, scfg, compute,
                        cache_dir: Path):
    fluid, eos, _ = make_simple_fluid(name)
    m_chain, sigma_seg, eps_k, _ = load_pcsaft_fluid(name)
    print(f"\n{name.title()} PC-SAFT:  m={m_chain:.3f}  sigma={sigma_seg:.3f}  eps/k={eps_k:.2f}")
    sigma_hs = hsd_pcsaft(sigma_seg, eps_k, T_K)
    print(f"  HSD(T={T_K}) = {sigma_hs:.3f}  (vs raw σ = {sigma_seg:.3f})")
    pcsaft_F = PurePCSAFTFunctional(m=m_chain, sigma=sigma_seg, eps_k=eps_k, T=T_K)

    cache = cache_dir / f"vext_{name}_{int(T_K)}K.npy"
    if cache.exists():
        vd = np.load(cache, allow_pickle=True).item()
        print(f"  Loaded Vext cache: {cache.name}")
    else:
        potential = CompositePotential([
            LJPotential(host_ff=host_ff, fluid_ff=fluid.ff, cutoff=vcfg.cutoff_lj
                        if hasattr(vcfg, 'cutoff_lj') else 15.0),
        ])
        print(f"  Building Vext for {name}...")
        vd = build_vext_on_grid(
            host, fluid, potential,
            orientations=fibonacci_rotations(vcfg.n_orient),
            spacing=vcfg.spacing,
            pbc_supercell=list(vcfg.pbc_supercell),
            temperature_K=T_K,
            cache_path=str(cache),
            v_reject_below_K=vcfg.v_reject_below_K,
            v_cap_above_K=vcfg.v_cap_above_K,
            averaging=vcfg.averaging,
            compute=compute,
        )

    Vext_K = vd["vext_avg"]
    dV     = float(vd["dV"])
    access = np.isfinite(Vext_K) & (Vext_K < vcfg.access_factor * T_K)
    Nx, Ny, Nz = Vext_K.shape
    Lx, Ly, Lz = np.linalg.norm(host.lattice, axis=1)
    KX, KY, KZ, K = make_k_grid((Nx, Ny, Nz), dx=Lx/Nx, dy=Ly/Ny, dz=Lz/Nz)
    w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, sigma_hs)
    _, _, _, K_rfft = make_k_grid((Nx, Ny, Nz), dx=Lx/Nx, dy=Ly/Ny, dz=Lz/Nz,
                                   real_fft=True)
    w_disp_hat            = pcsaft_F._weight_disp_hat(K_rfft)
    w_lambd_hat, w_zeta3_hat = pcsaft_F._weight_chain_hat(K_rfft)
    RHO_MAX = 0.45 * 6.0 / (np.pi * sigma_hs ** 3 * m_chain)

    def c1_fn(rho):
        rho_seg = m_chain * rho
        wd = compute_weighted_densities(rho_seg, w2_hat, w3_hat, w2vec_hat, sigma_hs)
        c1_fmt = m_chain * np.asarray(
            compute_c1(rho_seg, wd, w2_hat, w3_hat, w2vec_hat, sigma_hs, model="aWBII"))
        c1_dc  = np.asarray(pcsaft_F.c1(rho, dV, w_disp_hat, w_lambd_hat, w_zeta3_hat))
        return c1_fmt + c1_dc

    _inv_m = 1.0 / m_chain
    def c1_fn_eff(rho, _fn=c1_fn, _inv=_inv_m):
        return _fn(rho) * _inv

    N_per_uc = np.empty(len(P_arr))
    rho_prev = rho_prev_b = None
    print(f"\n  m={m_chain:.4f}  d={sigma_hs:.4f} A  inv_m={_inv_m:.4f}")
    print(f"  {'P (bar)':>10}  {'N/uc':>8}  {'conv':>5}  {'iters':>6}")
    for i, P in enumerate(P_arr):
        rho_b = float(eos.bulk_density(P, T_K))
        c1_b  = bulk_c1(rho_b, sigma_hs, model="aWBII")
        beta  = 1.0 / T_K
        if rho_prev is not None:
            rho0 = np.where(access, np.clip(rho_prev * (rho_b / rho_prev_b),
                                            1e-16, RHO_MAX), 1e-16)
        else:
            rho0 = rho_b * np.exp(np.clip(-beta * Vext_K * _inv_m, -50, 20)) * access
            rho0 = np.minimum(rho0, RHO_MAX)

        res = anderson_solve(
            rho_init=rho0, rho_bulk=rho_b,
            Vext_K=Vext_K * _inv_m, temperature_K=T_K,
            c1_callable=c1_fn_eff, c1_bulk=c1_b * _inv_m,
            m=scfg.m, beta=scfg.beta, max_iter=scfg.max_iter, tol=scfg.tol,
            accessibility_mask=access, log_clip=scfg.log_clip,
            safeguard_alpha=scfg.safeguard_alpha,
            picard_warmup=scfg.picard_warmup,
            step_clip=scfg.step_clip,
            rho_max=RHO_MAX,
        )
        rho_prev, rho_prev_b = res.rho.copy(), rho_b
        N_per_uc[i] = float(res.rho.sum() * dV)
        print(f"  {P:10.3f}  {N_per_uc[i]:8.3f}  {str(res.converged):>5}  {res.iterations:6d}")
    return N_per_uc


def iast_mixture(N_ch4, N_c2h6, P_arr, x_ch4=0.6):
    """Myers–Prausnitz IAST for a binary mixture."""
    from scipy.interpolate import interp1d
    P_ch4 = np.interp(np.linspace(0, 1, 200),
                      np.linspace(0, 1, len(P_arr)), P_arr)
    # reuse the pure isotherms directly via interpolation
    f_ch4  = interp1d(P_arr, N_ch4,  kind="linear", fill_value="extrapolate")
    f_c2h6 = interp1d(P_arr, N_c2h6, kind="linear", fill_value="extrapolate")

    q_ch4_mix  = np.empty(len(P_arr))
    q_c2h6_mix = np.empty(len(P_arr))
    x_c2h6 = 1.0 - x_ch4
    for i, P in enumerate(P_arr):
        def spreading_eq(P0_ch4):
            P0_c2h6 = P0_ch4 * (x_c2h6 / x_ch4) if x_ch4 > 0 else 1e-10
            I_ch4  = np.trapz(f_ch4(np.linspace(1e-6, P0_ch4, 200)) /
                               np.linspace(1e-6, P0_ch4, 200), np.linspace(1e-6, P0_ch4, 200))
            I_c2h6 = np.trapz(f_c2h6(np.linspace(1e-6, P0_c2h6, 200)) /
                               np.linspace(1e-6, P0_c2h6, 200), np.linspace(1e-6, P0_c2h6, 200))
            return I_ch4 - I_c2h6
        try:
            P0_ch4 = brentq(spreading_eq, 1e-6, P * 10, xtol=1e-6)
            P0_c2h6 = P0_ch4 * x_c2h6 / x_ch4
            n1 = f_ch4(P0_ch4)
            n2 = f_c2h6(P0_c2h6)
            denom = x_ch4 / n1 + x_c2h6 / n2
            q_ch4_mix[i]  = x_ch4  / denom
            q_c2h6_mix[i] = x_c2h6 / denom
        except Exception:
            q_ch4_mix[i] = q_c2h6_mix[i] = float("nan")
    return q_ch4_mix, q_c2h6_mix


# ─────────────────────────────────────────────────────────────────────────────
# Hydra entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Print the resolved config so it's captured in the .hydra/ run log.
    print(OmegaConf.to_yaml(cfg))

    # ── Build ComputeConfig ──
    compute = ComputeConfig.from_omegaconf(cfg.compute)
    compute.apply_jax_device()
    if cfg.compute.dtype == "float64":
        compute.enable_jax_x64()

    # ── Run directory (Hydra sets cwd to the output dir automatically) ──
    run_dir = Path.cwd()   # already = outputs/<experiment>/<date>/<time>/
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Pressure grid ──
    rcfg = cfg.run
    if rcfg.p_log_space:
        P_arr = np.logspace(np.log10(rcfg.p_min_bar), np.log10(rcfg.p_max_bar),
                            rcfg.n_pressure)
    else:
        P_arr = np.linspace(rcfg.p_min_bar, rcfg.p_max_bar, rcfg.n_pressure)
    T_K = float(rcfg.temperature_K)

    # ── Host ──
    cif_path = STRUCT_DIR / cfg.host.cif
    host = read_cif(str(cif_path))
    host_ff = load_dreiding_ff(FF_DIR / cfg.host.ff_dat)
    host = host.assign_charges({s: 0.0 for s in set(host.species)},
                                source=cfg.host.charge_method)
    print(f"\nHost: {cif_path.stem}  {host.n_atoms} atoms")

    # ── Pure isotherms ──
    N_ch4  = single_gas_isotherm("methane", host, host_ff, P_arr, T_K,
                                  cfg.vext, cfg.solver, compute, run_dir)
    N_c2h6 = single_gas_isotherm("ethane",  host, host_ff, P_arr, T_K,
                                  cfg.vext, cfg.solver, compute, run_dir)

    # ── IAST mixture ──
    x_ch4 = 0.6
    print(f"\n=== IAST mixture at x_CH4 = {x_ch4} ===")
    q_ch4_mix, q_c2h6_mix = iast_mixture(N_ch4, N_c2h6, P_arr, x_ch4)

    # ── Save checkpoint ──
    np.savez(run_dir / "isotherm.npz",
             P_arr=P_arr, N_ch4=N_ch4, N_c2h6=N_c2h6,
             q_ch4_mix=q_ch4_mix, q_c2h6_mix=q_c2h6_mix,
             temperature_K=T_K)

    # ── Figure ──
    if rcfg.save_figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(P_arr, N_ch4,  "s-", color="#1f77b4", label="CH4 pure (cDFT)")
        ax.plot(P_arr, N_c2h6, "^-", color="#ff7f0e", label="C2H6 pure (cDFT)")
        ax.plot(P_arr, q_ch4_mix,  "s--", color="#1f77b4", alpha=0.6,
                label=f"CH4 mix xV={x_ch4:.1f} (IAST)")
        ax.plot(P_arr, q_c2h6_mix, "^--", color="#ff7f0e", alpha=0.6,
                label=f"C2H6 mix (IAST)")
        gcmc_max = max(max(GCMC_REF["C2H6"]["N"]), max(GCMC_REF["CH4"]["N"]))
        ax.plot(GCMC_REF["C2H6"]["p"], GCMC_REF["C2H6"]["N"],
                "^k", ms=8, label="C2H6 GCMC (Stierle 2024)")
        ax.plot(GCMC_REF["CH4"]["p"],  GCMC_REF["CH4"]["N"],
                "sk", ms=8, label="CH4 GCMC (Stierle 2024)")
        ax.set_xlabel("Total pressure (bar)")
        ax.set_ylabel("Loading (mol / unit cell)")
        ax.set_ylim(0, max(q_c2h6_mix[np.isfinite(q_c2h6_mix)].max(),
                           N_c2h6.max(), gcmc_max) * 1.1)
        ax.legend(fontsize=9, ncol=2)
        ax.set_title(f"CH4/C2H6 in Dha-Tph  T={T_K} K  [{rcfg.experiment}]")
        fig.tight_layout()
        figpath = run_dir / "06_ch4_c2h6_in_dha_tph.png"
        fig.savefig(figpath, dpi=rcfg.figure_dpi)
        plt.close(fig)
        print(f"\nFigure saved: {figpath}")

    print(f"\nAll outputs in: {run_dir}")


if __name__ == "__main__":
    main()

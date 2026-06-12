"""Tutorial 4 — Xe / Kr selectivity in ZIF-8 at 273 K.

Compares the two single-gas isotherms then computes the IAST selectivity
S_Xe/Kr = (q_Xe / q_Kr) / (y_Xe / y_Kr) at equimolar feed.

Same machinery as Tutorial 1 (LJ-only DREIDING, PC-SAFT bulk for each gas).
Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \
        tutorials/04_xe_kr_in_zif8/run.py
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

from data_loader import (
    STRUCT_DIR, FF_DIR, load_dreiding_ff,
    load_pcsaft_fluid, make_pcsaft_eos,
)
from porecdft.fluid.base import Fluid
from porecdft.io import read_cif
from porecdft.io.forcefield import FFEntry
from porecdft.forcefield import LJPotential, CompositePotential
from porecdft.functional import (
    make_k_grid, make_fmt_weights_hat, compute_weighted_densities,
    compute_c1, bulk_c1,
)
from porecdft.solver import anderson_solve
from porecdft.structure import build_supercell
from porecdft.vext import build_vext_on_grid, fibonacci_rotations

T_K   = 273.0
P_BAR = np.logspace(-3, 0, 14)
AVOGADRO    = 6.022e23
ATOMIC_MASS = {"Zn": 65.38, "N": 14.007, "C": 12.011, "H": 1.008}

FIG_DIR = _REPO / "tutorials" / "figures"


# ── helpers ────────────────────────────────────────────────────────────────

def single_gas_isotherm(gas_name: str, host, host_ff):
    m, sigma, eps, M = load_pcsaft_fluid(gas_name)
    print(f"\n{gas_name.title()} PC-SAFT:  σ={sigma}  ε/k={eps}  M={M}")
    fluid = Fluid(
        name=gas_name.title(),
        body_sites=np.zeros((1, 3)),
        site_labels=[gas_name.title()],
        ff={gas_name.title(): FFEntry(gas_name.title(), sigma, eps,
                                       source="noble_gases.json")},
        charges={gas_name.title(): 0.0},
        molar_mass=M,
    )
    eos = make_pcsaft_eos(gas_name)

    cache = Path(__file__).parent / f"vext_zif8_{gas_name}_273K.npy"
    if cache.exists():
        vd = np.load(cache, allow_pickle=True).item()
        print(f"  Vext cache loaded: {cache}")
    else:
        potential = CompositePotential([
            LJPotential(host_ff=host_ff, fluid_ff=fluid.ff, cutoff=15.0),
        ])
        vd = build_vext_on_grid(host, fluid, potential,
                                orientations=fibonacci_rotations(20),
                                spacing=1.0, pbc_supercell=(2, 2, 2),
                                temperature_K=T_K,
                                cache_path=str(cache),
                                v_reject_below_K=-5000.0,
                                v_cap_above_K=+10000.0,
                                averaging="boltzmann")

    Vext_K = vd["vext_avg"]
    dV     = vd["dV"]
    access = np.isfinite(Vext_K) & (Vext_K < 50.0 * T_K)

    SIGMA_HS = sigma
    Nx, Ny, Nz = Vext_K.shape
    Lx, Ly, Lz = np.linalg.norm(host.lattice, axis=1)
    KX, KY, KZ, K = make_k_grid((Nx, Ny, Nz),
                                 dx=Lx/Nx, dy=Ly/Ny, dz=Lz/Nz)
    w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, SIGMA_HS)
    RHO_MAX = 0.45 * 6.0 / (np.pi * SIGMA_HS ** 3)

    def c1_fn(rho):
        wd = compute_weighted_densities(rho, w2_hat, w3_hat, w2vec_hat, SIGMA_HS)
        return np.asarray(compute_c1(rho, wd, w2_hat, w3_hat, w2vec_hat,
                                      SIGMA_HS, model="aWBII"))

    fw_amu = sum(ATOMIC_MASS.get(s, 0.0) for s in host.species)
    fw_g   = fw_amu / AVOGADRO
    to_mmol_per_g = (1.0 / AVOGADRO) * 1000.0 / fw_g

    N_arr = np.empty(len(P_BAR))
    rho_prev = None
    rho_prev_b = None
    for i, P in enumerate(P_BAR):
        rho_b = float(eos.bulk_density(P, T_K))
        c1_b  = bulk_c1(rho_b, SIGMA_HS, model="aWBII")
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


def main():
    host = read_cif(str(STRUCT_DIR / "ZIF-8.cif"))
    host = host.assign_charges({s: 0.0 for s in set(host.species)},
                                source="ZIF-8 LJ-only")
    host_ff = load_dreiding_ff(FF_DIR / "DREIDING.dat")
    print(f"ZIF-8 host: {len(host.species)} atoms, cell = {host.lattice.diagonal()}")

    N_xe = single_gas_isotherm("xenon",   host, host_ff)
    N_kr = single_gas_isotherm("krypton", host, host_ff)

    # IAST selectivity at equimolar feed (single-component limit):
    # S = (q_Xe / q_Kr) under identical p_partial.
    S_iast = N_xe / np.where(N_kr > 0, N_kr, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].semilogx(P_BAR, N_xe, "o-", color="#1f77b4", lw=2.0, ms=6, label="Xe")
    axes[0].semilogx(P_BAR, N_kr, "s-", color="#ff7f0e", lw=2.0, ms=6, label="Kr")
    axes[0].set_xlabel("Pressure (bar)", fontsize=12)
    axes[0].set_ylabel(r"uptake (mmol g$^{-1}$)", fontsize=12)
    axes[0].set_title("Single-gas isotherms (T = 273 K)",
                      fontsize=11, fontweight="bold")
    axes[0].grid(alpha=0.3, ls=":")
    axes[0].legend(fontsize=10)

    axes[1].semilogx(P_BAR, S_iast, "o-", color="#2ca02c", lw=2.0, ms=6)
    axes[1].axhline(1, color="grey", lw=0.8, ls=":")
    axes[1].set_xlabel("Pressure (bar)", fontsize=12)
    axes[1].set_ylabel(r"S$_{Xe/Kr}$ = q$_{Xe}$ / q$_{Kr}$", fontsize=12)
    axes[1].set_title("IAST selectivity (equimolar feed)",
                      fontsize=11, fontweight="bold")
    axes[1].grid(alpha=0.3, ls=":")
    fig.suptitle("Tutorial 4 — Xe / Kr in ZIF-8 (validation vs Stierle 2024 Fig. 6)",
                 fontsize=12, fontweight="bold")
    out = FIG_DIR / "04_xe_kr_in_zif8.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()

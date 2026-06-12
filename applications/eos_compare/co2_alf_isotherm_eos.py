"""CO2/ALF isotherms at 273 K and 298 K computed with several bulk EOS.

Solves the FMT-aWBII self-consistency

    ρ(r) = ρ_bulk(P, T) · exp[ −β V_ext(r) + c¹_HS(r) − c¹_HS(ρ_bulk) ]

for every pressure in the Evans 2022 range, swapping the bulk EOS that maps
P → ρ_bulk for each curve.  Plots the resulting absolute-loading isotherms.

EOS swept:

  * Ideal gas      — ρ = P / (k_B T)              (porecdft.eos.density_from_pressure)
  * Peng-Robinson  — porecdft.eos.PengRobinsonEOS using CO2 critical params
  * Span-Wagner    — porecdft.eos.CO2_SW   (truncated 7-term reference EOS)
  * SRK            — porecdft.eos.CO2_SRK
  * PC-SAFT        — porecdft.eos.CO2_PCSAFT (Gross & Sadowski 2001)

Vext is reused from the existing phase-2 cache (built once with the standard
EPM2 + smeared-Coulomb + Quadrupole-EFG composite force field).

Run with:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \
        applications/eos_compare/co2_alf_isotherm_eos.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARENT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore")

from applications.alf_co2 import (
    ALF_CIF, CHARGES_CSV, FORCEFIELD_CSV, DATA_DIR, EXP_TARGETS,
)
from applications.alf_co2.notebooks.phase1_vext_validation import _read_forcefield_csv
from porecdft.eos import (
    density_from_pressure,
    PengRobinsonEOS,
    CO2_SW,
    CO2_SRK,
    CO2_PCSAFT,
)
from porecdft.fluid import EPM2_CO2
from porecdft.forcefield import (
    CompositePotential, CoulombPotential, LJPotential, QuadrupoleEFGPotential,
)
from porecdft.io.forcefield import FFEntry
from porecdft.functional import (
    make_k_grid, make_fmt_weights_hat,
    compute_weighted_densities, compute_c1, bulk_c1,
)
from porecdft.io import read_cif, read_charges_csv
from porecdft.solver import anderson_solve, picard_solve
from porecdft.structure import build_supercell
from porecdft.vext import build_vext_on_grid, fibonacci_rotations

OUT_FIG  = Path(__file__).parent / "figures"
OUT_FIG.mkdir(exist_ok=True)
OUT_RES  = Path(__file__).parent / "results"
OUT_RES.mkdir(exist_ok=True)

CACHE_ROOT = DATA_DIR / "results" / "vext_cache"

# ── physical constants ─────────────────────────────────────────────────────
ATOMIC_MASS = {"Al": 26.9815, "C": 12.011, "O": 15.999, "H": 1.008, "N": 14.007}
AVOGADRO    = 6.022e23
SIGMA_HS    = 3.017     # Å — CO2 hard-sphere diameter

# CO2 critical parameters (for PR)
CO2_PR = PengRobinsonEOS(Tc=304.13, Pc=73.77e5, omega=0.225, molar_mass=44.01, name="CO2_PR")

TEMPERATURES = [273.0, 298.0]


def build_vext(host, fluid, host_ff, n_orient=20, T=298.0):
    """Reuse the existing phase-2 Vext cache.  Available T = 273, 298, 323 K."""
    cache_path = CACHE_ROOT / f"vext_avg_T{int(T)}K.npy"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Vext cache not found: {cache_path}\n"
            "Build it first by running:\n"
            "    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \\\n"
            "        applications/alf_co2/notebooks/phase2_2_fmt_isotherm.py\n"
            "(builds the EPM2 + smeared-Coulomb + Q-EFG Vext for T = 273/298/323 K)."
        )
    data = np.load(cache_path, allow_pickle=True).item()
    print(f"  Loaded Vext from cache: {cache_path}", flush=True)
    return data


# ── isotherm at one (T, EOS) ────────────────────────────────────────────────

def run_isotherm(Vext_K, dV, access, pressures_bar, T_K, rho_bulk_fn,
                 framework_mass_g, label: str, host):
    """Self-consistent FMT-aWBII isotherm.  ``rho_bulk_fn(P)`` returns the
    bulk density in molecules/Å³ at pressure P (bar) and temperature T."""
    Nx, Ny, Nz = Vext_K.shape
    Lx = float(np.linalg.norm(host.lattice[0]))
    Ly = float(np.linalg.norm(host.lattice[1]))
    Lz = float(np.linalg.norm(host.lattice[2]))
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    KX, KY, KZ, K = make_k_grid((Nx, Ny, Nz), dx=dx, dy=dy, dz=dz)
    w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(K, KX, KY, KZ, SIGMA_HS)
    # Physical density cap for FMT-aWBII: η < 0.45 → ρ < 0.45·6/(π σ³).
    # Without this the deep SC well (~−50 kJ/mol ≈ −20000 K) produces
    # Boltzmann factors exp(70) which the relative log_clip lets ρ exceed
    # the hard-sphere close-packing limit → log(1−n_3) overflows to NaN.
    RHO_MAX = 0.45 * 6.0 / (np.pi * SIGMA_HS ** 3)

    def c1_fn(rho):
        wd = compute_weighted_densities(rho, w2_hat, w3_hat, w2vec_hat, SIGMA_HS)
        return np.asarray(compute_c1(rho, wd, w2_hat, w3_hat, w2vec_hat,
                                      SIGMA_HS, model="aWBII"))

    to_mmol_per_g = (1.0 / AVOGADRO) * 1000.0 / framework_mass_g

    def boltzmann_init(rho_b, beta):
        ri = rho_b * np.exp(np.clip(-beta * Vext_K, -50.0, 20.0)) * access
        rho_max = 0.45 * 6.0 / (np.pi * SIGMA_HS ** 3)
        return np.minimum(ri, rho_max)

    N_abs = np.empty(len(pressures_bar))
    rho_prev = None
    rho_prev_bulk = None
    for i, p in enumerate(pressures_bar):
        rho_bulk = float(rho_bulk_fn(p))
        beta = 1.0 / T_K
        c1_b = bulk_c1(rho_bulk, SIGMA_HS, model="aWBII")

        # Warm start
        if rho_prev is not None and rho_prev_bulk and rho_prev_bulk > 0:
            rho_init = np.where(access,
                                np.clip(rho_prev * (rho_bulk / rho_prev_bulk),
                                        1e-16, 0.45 * 6.0 / (np.pi * SIGMA_HS**3)),
                                1e-16)
        else:
            rho_init = boltzmann_init(rho_bulk, beta)

        res = anderson_solve(
            rho_init=rho_init, rho_bulk=rho_bulk,
            Vext_K=Vext_K, temperature_K=T_K,
            c1_callable=c1_fn, c1_bulk=c1_b,
            m=6, beta=0.3, max_iter=800, tol=1e-4,
            accessibility_mask=access, log_clip=25.0,
            safeguard_alpha=0.02, picard_warmup=30, step_clip=2.0,
            rho_max=RHO_MAX,
        )
        last_err = res.error_history[-1] if res.error_history else np.inf
        if not res.converged and (not np.isfinite(last_err) or last_err > 0.1):
            res = picard_solve(
                rho_init=res.rho if np.isfinite(last_err) else boltzmann_init(rho_bulk, beta),
                rho_bulk=rho_bulk, Vext_K=Vext_K, temperature_K=T_K,
                c1_callable=c1_fn, c1_bulk=c1_b,
                alpha=0.005, max_iter=2000, tol=1e-3,
                accessibility_mask=access, log_clip=25.0,
                rho_max=RHO_MAX,
            )
        N_abs[i] = float(res.rho.sum() * dV)
        rho_prev = res.rho.copy()
        rho_prev_bulk = rho_bulk
    return N_abs * to_mmol_per_g


# ── main ───────────────────────────────────────────────────────────────────

def make_eos_cases(T):
    """Return list of (name, rho_bulk(P)->fn, color, linestyle) for given T."""
    return [
        ("Ideal gas",     lambda P, T=T: density_from_pressure(P, T),        "#7f7f7f", "--"),
        ("PR",            lambda P, T=T: CO2_PR.bulk_density(P, T),           "#1f77b4", "-"),
        ("SRK",           lambda P, T=T: CO2_SRK.bulk_density(P, T),          "#ff7f0e", "-"),
        ("Span-Wagner",   lambda P, T=T: float(CO2_SW.bulk_density(P, T)),    "#d62728", "-"),
        ("PC-SAFT",       lambda P, T=T: float(CO2_PCSAFT.bulk_density(P, T)),"#9467bd", "-"),
    ]


def main():
    print("=" * 70)
    print(f"CO2/ALF isotherm — EOS comparison at T = {TEMPERATURES} K")
    print("=" * 70)

    # ── Load host + force field ──
    host = read_cif(str(ALF_CIF))
    charges = read_charges_csv(str(CHARGES_CSV))
    host = host.assign_charges(charges, source="CP2K_Hirshfeld")
    host_ff = _read_forcefield_csv(Path(FORCEFIELD_CSV))
    fluid = EPM2_CO2

    framework_mass_amu = sum(ATOMIC_MASS[s] for s in host.species)
    framework_mass_g   = framework_mass_amu / AVOGADRO

    all_curves = {}      # {T: {eos_name: (N_mmol_g, color, ls)}}
    all_p_grids = {}     # {T: pressures_bar}
    all_p_exp = {}       # {T: (p, n_exp)}

    # ── Per-temperature: load Vext, run all EOS ──
    for T in TEMPERATURES:
        print(f"\n{'─'*70}\n  T = {T:.0f} K\n{'─'*70}", flush=True)
        print("[1/3] Loading Vext...", flush=True)
        vd = build_vext(host, fluid, host_ff, n_orient=20, T=T)
        Vext_K = vd["vext_avg"]
        dV     = vd["dV"]
        access = np.isfinite(Vext_K) & (Vext_K < 50.0 * T)

        p_exp_T, n_exp_T = zip(*EXP_TARGETS[int(T)])
        p_grid = np.array(p_exp_T)
        all_p_grids[T] = p_grid
        all_p_exp[T]   = (p_exp_T, n_exp_T)

        EOS_CASES = make_eos_cases(T)

        print(f"[2/3] Running {len(EOS_CASES)} isotherms (one per EOS)...\n", flush=True)
        curves = {}
        for name, fn, color, ls in EOS_CASES:
            t0 = time.time()
            N_mmol_g = run_isotherm(Vext_K, dV, access, p_grid, T, fn,
                                    framework_mass_g, name, host)
            dt = time.time() - t0
            curves[name] = (N_mmol_g, color, ls)
            print(f"    {name:14s}  {dt:5.1f}s  "
                  f"N(max P) = {N_mmol_g[-1]:.3f} mmol/g", flush=True)
        all_curves[T] = curves

    # ── Plot: one row per temperature, two panels (linear / log) per row ──
    print("\n[3/3] Plotting...", flush=True)
    n_T = len(TEMPERATURES)
    fig, axes = plt.subplots(n_T, 2, figsize=(11, 4.2 * n_T),
                              constrained_layout=True)
    if n_T == 1:
        axes = np.array([axes])

    for r, T in enumerate(TEMPERATURES):
        curves    = all_curves[T]
        p_grid    = all_p_grids[T]
        p_exp_T, n_exp_T = all_p_exp[T]
        for c in (0, 1):
            ax = axes[r, c]
            for name, (N, color, ls) in curves.items():
                ax.plot(p_grid, N, color=color, ls=ls, lw=2.0,
                        marker="o", ms=4, label=name)
            ax.plot(p_exp_T, n_exp_T, "ko", ms=7, mfc="white", mew=1.5,
                    label="Evans 2022 (expt)")
            ax.set_xlabel("Pressure (bar)", fontsize=11)
            ax.set_ylabel(r"$N_\mathrm{abs}$ (mmol g$^{-1}$)", fontsize=11)
            ax.grid(alpha=0.3, ls=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=8, framealpha=0.9)
            if c == 0:
                ax.set_title(f"T = {T:.0f} K — linear", fontsize=11, fontweight="bold")
            else:
                ax.set_xscale("log")
                ax.set_title(f"T = {T:.0f} K — log", fontsize=11, fontweight="bold")

    fig.suptitle("CO$_2$/ALF isotherm — EOS comparison\n"
                 "(FMT-aWBII + smeared Coulomb + Q-EFG, no Wertheim / elastic)",
                 fontsize=12, fontweight="bold")
    out_png = OUT_FIG / "co2_alf_isotherm_eos.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote: {out_png}")

    # Save numeric results
    save_data = {}
    for T in TEMPERATURES:
        save_data[f"p_T{int(T)}"] = all_p_grids[T]
        for name, (N, _, _) in all_curves[T].items():
            slug = name.replace(" ", "_").replace("-", "_")
            save_data[f"{slug}_T{int(T)}"] = N
    np.savez(OUT_RES / "co2_alf_isotherm_eos.npz", **save_data)
    print(f"Saved: {OUT_RES / 'co2_alf_isotherm_eos.npz'}")


if __name__ == "__main__":
    main()

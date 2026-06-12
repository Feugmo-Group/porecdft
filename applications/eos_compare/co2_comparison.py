"""CO2 bulk-density comparison across all porecdft EOS implementations.

Generates ``applications/eos_compare/figures/co2_eos_comparison.png`` with two
panels: ρ(P) and Z(P) at 298 K for CO2 from 0.1 to 100 bar using each EOS.

Run with:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \
        applications/eos_compare/co2_comparison.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT.parent))
sys.path.insert(0, str(_REPO_ROOT))

from porecdft.eos import (
    density_from_pressure,
    PengRobinsonEOS,
    CO2_SW,
    CO2_SRK,
    CO2_SAFT_VR_Mie,
    CO2_PCSAFT,
)

# CO2 Peng-Robinson singleton (paper Tc=304.13, Pc=73.77 bar, omega=0.225)
CO2_PR = PengRobinsonEOS(Tc=304.13, Pc=73.77e5, omega=0.225, molar_mass=44.01, name="CO2_PR")

# k_B (J/K) for Z = P / (rho * k_B * T) when rho is in 1/m³
K_B = 1.380649e-23

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

T_K = 298.0
P_bar = np.logspace(-1, 2, 60)   # 0.1 to 100 bar

eos_panel = [
    ("Ideal gas", lambda P: density_from_pressure(P, T_K), "#7f7f7f", "--"),
    ("PR",        lambda P: CO2_PR.bulk_density(P, T_K),  "#1f77b4", "-"),
    ("SRK",       lambda P: CO2_SRK.bulk_density(P, T_K), "#ff7f0e", "-"),
    ("Span-Wagner", lambda P: CO2_SW.bulk_density(P, T_K), "#d62728", "-"),
    ("SAFT-VR-Mie", lambda P: CO2_SAFT_VR_Mie.bulk_density(P, T_K), "#2ca02c", "-"),
    ("PC-SAFT",     lambda P: float(CO2_PCSAFT.bulk_density(P, T_K)), "#9467bd", "-"),
]

rho_curves = {}
for name, fn, *_ in eos_panel:
    rho_curves[name] = np.array([fn(float(P)) for P in P_bar])  # molecules / Å³

# Compressibility factor Z = P / (rho * k_B * T) where rho in molecules/m³
# rho [mol/Å³] * 1e30 = mol/m³ → P[Pa] = rho_m3 * k_B * T * Z
# ⇒ Z = P_Pa / (rho_m3 * k_B * T)
def Z_from_rho(rho_per_A3, P_bar_value):
    rho_m3 = rho_per_A3 * 1e30
    return (P_bar_value * 1e5) / (rho_m3 * K_B * T_K)

Z_curves = {name: np.array([Z_from_rho(rho_curves[name][i], P_bar[i])
                            for i in range(len(P_bar))])
            for name, *_ in eos_panel}

# ── figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)

for name, _, color, ls in eos_panel:
    axes[0].loglog(P_bar, rho_curves[name], color=color, ls=ls, lw=2.0, label=name)
    axes[1].semilogx(P_bar, Z_curves[name], color=color, ls=ls, lw=2.0, label=name)

for ax in axes:
    ax.grid(alpha=0.3, which="both", linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Pressure (bar)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)

axes[0].set_ylabel(r"$\rho_\mathrm{bulk}$ (molecules / Å³)", fontsize=11)
axes[0].set_title("CO$_2$ bulk density at 298 K", fontsize=12, fontweight="bold")
axes[1].set_ylabel(r"$Z = P / (\rho\,k_B T)$", fontsize=11)
axes[1].set_title("Compressibility factor at 298 K", fontsize=12, fontweight="bold")
axes[1].axhline(1.0, color="gray", lw=0.8, ls=":")

fig.suptitle("porecdft EOS comparison — CO$_2$ at 298 K",
             fontsize=13, fontweight="bold")

out = FIG_DIR / "co2_eos_comparison.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Wrote: {out}")

# Console summary table
print()
print(f"{'EOS':<14} {'ρ(1 bar) [Å⁻³]':<20} {'Z(1 bar)':<14} {'ρ(50 bar) [Å⁻³]':<20} {'Z(50 bar)':<12}")
print("─" * 80)
for name, *_ in eos_panel:
    i1, i50 = np.argmin(np.abs(P_bar - 1.0)), np.argmin(np.abs(P_bar - 50.0))
    print(f"{name:<14} {rho_curves[name][i1]:<20.4e} {Z_curves[name][i1]:<14.4f} "
          f"{rho_curves[name][i50]:<20.4e} {Z_curves[name][i50]:<12.4f}")

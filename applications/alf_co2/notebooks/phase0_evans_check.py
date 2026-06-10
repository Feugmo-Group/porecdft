"""Phase 0 — Verify Evans 2022 experimental isotherm digitisation.

Plots the hand-digitised CO₂/ALF isotherms at 273, 298, 323, and 348 K
against the original figure description from Evans et al. (Sci. Adv. 2022).

Output
------
  figures/00_evans_digitized_check.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARENT = _REPO_ROOT.parent
for _p in (str(_REPO_ROOT), str(_PARENT)):
    try: sys.path.remove(_p)
    except ValueError: pass
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_REPO_ROOT))

from applications.alf_co2 import DATA_DIR, EXP_ISOTHERMS

OUT_FIG = DATA_DIR / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

COLORS = {273: "#1f77b4", 298: "#ff7f0e", 323: "#2ca02c", 348: "#d62728", 398: "#9467bd"}
MARKERS = {273: "o", 298: "s", 323: "^", 348: "D", 398: "v"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for T, data in EXP_ISOTHERMS.items():
    if T == 398:
        continue
    p = np.array(data["p_bar"])
    N = np.array(data["N_mmol_g"])
    c = COLORS[T]
    m = MARKERS[T]
    axes[0].plot(p, N, marker=m, color=c, lw=1.8, ms=6, label=f"{T} K")
    axes[1].semilogx(p, N, marker=m, color=c, lw=1.8, ms=6, label=f"{T} K")

for ax in axes:
    ax.set_xlabel("Pressure (bar)", fontsize=12)
    ax.set_ylabel(r"$N_\mathrm{abs}$ (mmol g$^{-1}$)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 5)

axes[0].set_xlim(0, 0.13)
axes[0].set_title("Linear scale", fontsize=12)
axes[1].set_xlim(1e-3, 0.13)
axes[1].set_title("Log scale", fontsize=12)

fig.suptitle("Evans 2022 CO₂/ALF experimental isotherms (digitised)", fontsize=13)
fig.tight_layout()

out = OUT_FIG / "00_evans_digitized_check.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

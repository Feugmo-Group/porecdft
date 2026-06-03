"""Coulomb electrostatic potential between charged fluid sites and a charged host.

Two summation methods are supported:

- ``method="direct"`` (default) — truncated direct sum with a real-space cutoff.
  Appropriate when the host unit cell has been replicated into a large supercell
  (typically 3×3×3) and the framework partial charges are small enough that the
  cutoff truncation error is acceptable. Used by both legacy codes
  (``co2_pcsaft_v4.py``, ``co2_cdft_final_v2.ipynb``).

- ``method="wolf"`` — damped, shifted potential of Wolf, Keasler, Coker &
  Tildesley (1999).

- ``method="smeared"`` — **Gaussian-smeared point charges**. Each charge q_i is
  represented as a 3D Gaussian of width σ. The pair interaction between two
  Gaussians of widths σ_i and σ_j is

  .. math::
      \\phi_{\\rm smear}(r) = K_E \\, \\frac{\\mathrm{erf}(r/\\sigma_{ij})}{r},
      \\quad \\sigma_{ij} = \\sqrt{\\sigma_i^2 + \\sigma_j^2}.

  As ``r → 0`` this is finite (``2 K_E / (σ_ij √π)``) — no singularity.
  As ``r >> σ_ij`` it recovers bare Coulomb ``1/r``. This is the physically
  motivated fix for point-charge over-attraction at H-bond contact distances,
  where Hirshfeld charges + EPM2 O at ~2.5 Å from formate H give unphysical
  −100 kJ/mol attractions. The width parameter ``gauss_width`` is the *combined*
  σ_ij (or, equivalently, the per-atom σ if all atoms share a single width
  σ_i = gauss_width/√2). Typical value 0.7–1.5 Å.

Energy units: Kelvin. Conversion factor :math:`K_E = k_e \\cdot e^2 / k_B`
with :math:`k_e` in SI (8.987e9 N·m²/C²), :math:`e = 1.602\\times 10^{-19}` C,
:math:`k_B = 1.381\\times 10^{-23}` J/K, giving :math:`K_E \\approx 167101` K·Å
when distances are in Å and charges are in units of e.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf, erfc

from porecdft.forcefield.base import Potential, PotentialEnergy

# k_e * e^2 / k_B in K*Å (for r in Å and q in units of e)
COULOMB_K_KELVIN_ANGSTROM = 167101.0


@dataclass(frozen=True)
class CoulombPotential(Potential):
    """Coulomb between fluid sites and host atoms.

    Parameters
    ----------
    fluid_charges : dict[str, float]
        Charge per fluid-site label, e.g. {"C": +0.6512, "O": -0.3256} for EPM2.
    cutoff : float
        Real-space cutoff in Å.
    method : str
        ``"direct"``, ``"wolf"``, or ``"smeared"``.
    wolf_alpha : float
        Damping parameter (1/Å). Only used when ``method="wolf"``. If left at the
        default 0.0, it is set to ``2.7 / cutoff``.
    gauss_width : float
        Effective Gaussian width σ_ij (Å) when ``method="smeared"``. Default 1.0 Å.
        Pairwise width should account for both atoms; if all atoms are treated as
        identical-width Gaussians of σ_atom, then σ_ij = σ_atom·√2.
    """
    fluid_charges: dict[str, float]
    cutoff: float = 15.0
    method: str = "direct"
    wolf_alpha: float = 0.0
    gauss_width: float = 1.0
    name: str = "Coulomb"

    def __post_init__(self):
        if self.method not in ("direct", "wolf", "smeared"):
            raise ValueError(f"Unknown Coulomb method {self.method!r}")
        if self.method == "wolf" and self.wolf_alpha == 0.0:
            object.__setattr__(self, "wolf_alpha", 2.7 / self.cutoff)
        if self.method == "smeared" and self.gauss_width <= 0.0:
            raise ValueError("smeared Coulomb requires gauss_width > 0")

    def _phi(self, r: np.ndarray) -> np.ndarray:
        """Per-pair Coulomb factor in K·Å (multiply by q_i*q_j to get energy in K)."""
        Rc = self.cutoff
        if self.method == "direct":
            return COULOMB_K_KELVIN_ANGSTROM / r
        if self.method == "wolf":
            a = self.wolf_alpha
            shift = erfc(a * Rc) / Rc
            return COULOMB_K_KELVIN_ANGSTROM * (erfc(a * r) / r - shift)
        # method == "smeared": Gaussian-smeared point charges
        # φ(r) = KE · erf(r/σ) / r — finite at r=0, → KE/r at large r
        sigma = self.gauss_width
        return COULOMB_K_KELVIN_ANGSTROM * erf(r / sigma) / r

    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels) -> PotentialEnergy:
        r_center = np.asarray(r_center)
        sites_lab = r_center + fluid_sites @ rot.T
        host_q = host.charges
        host_pos = host.positions
        total = 0.0
        cutoff2 = self.cutoff * self.cutoff
        for s_idx, label in enumerate(fluid_site_labels):
            q_s = self.fluid_charges[label]
            if q_s == 0.0:
                continue
            dr = host_pos - sites_lab[s_idx]          # (Na, 3)
            r2 = np.einsum("ad,ad->a", dr, dr)
            mask = (r2 < cutoff2) & (r2 > 0.0)
            if not np.any(mask):
                continue
            r_safe = np.sqrt(r2[mask])
            phi = self._phi(r_safe)                   # K·Å * 1/Å = K per unit q_i*q_j (dimensionless e²)
            total += float(q_s * (host_q[mask] * phi).sum())
        return PotentialEnergy(total=total, parts={"Coulomb": total})

    def energy_grid(self, grid_xyz, rot, host, fluid_sites, fluid_site_labels):
        grid = np.asarray(grid_xyz)
        sites_lab = grid[:, None, :] + (fluid_sites @ rot.T)[None, :, :]  # (Ng, S, 3)
        host_pos = host.positions
        host_q = host.charges
        total = np.zeros(len(grid), dtype=float)
        cutoff2 = self.cutoff * self.cutoff
        for s_idx, label in enumerate(fluid_site_labels):
            q_s = self.fluid_charges[label]
            if q_s == 0.0:
                continue
            site = sites_lab[:, s_idx, :]                     # (Ng, 3)
            dr = host_pos[None, :, :] - site[:, None, :]      # (Ng, Na, 3)
            r2 = np.einsum("gad,gad->ga", dr, dr)
            mask = (r2 < cutoff2) & (r2 > 0.0)
            r_safe = np.sqrt(np.where(mask, r2, 1.0))         # avoid div-by-zero outside mask
            phi = np.where(mask, self._phi(r_safe), 0.0)
            total += q_s * (phi * host_q[None, :]).sum(axis=1)
        return total

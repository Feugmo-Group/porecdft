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

from dataclasses import dataclass, field
from typing import Any

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
    host_override : HostAtoms, optional
        If set, use this HostAtoms (e.g. the original 104-atom unit cell) instead of
        the (possibly replicated) host passed to energy_grid/energy_at. Use together
        with ``mic_lattice`` to apply the minimum image convention. This is required
        for charge-neutral frameworks where the unit cell is smaller than the Coulomb
        cutoff: passing a 3×3×3 supercell without MIC double-counts periodic images,
        breaking the charge-neutral cancellation ⟨V_Coul⟩_orient ≈ 0.
    mic_lattice : ndarray (3, 3), optional
        Lattice matrix (rows = lattice vectors, Å) used to apply the minimum image
        convention when computing pair distances. When set, each dr vector is wrapped
        to the nearest periodic image before the cutoff test and energy evaluation.
        Typically pass ``host.lattice`` (original unit cell).
    """
    fluid_charges: dict[str, float]
    cutoff: float = 15.0
    method: str = "direct"
    wolf_alpha: float = 0.0
    gauss_width: float = 1.0
    name: str = "Coulomb"
    host_override: Any = field(default=None, compare=False, hash=False)
    mic_lattice: Any = field(default=None, compare=False, hash=False)

    def __post_init__(self):
        if self.method not in ("direct", "wolf", "smeared"):
            raise ValueError(f"Unknown Coulomb method {self.method!r}")
        if self.method == "wolf" and self.wolf_alpha == 0.0:
            object.__setattr__(self, "wolf_alpha", 2.7 / self.cutoff)
        if self.method == "smeared" and self.gauss_width <= 0.0:
            raise ValueError("smeared Coulomb requires gauss_width > 0")

    def _mic(self, dr: np.ndarray) -> np.ndarray:
        """Wrap displacement vectors to the nearest periodic image (MIC).

        dr shape: (..., 3) in Cartesian Å.
        mic_lattice rows are lattice vectors: mic_lattice[i] = a_i vector.
        """
        L = np.asarray(self.mic_lattice)        # (3, 3)
        invL = np.linalg.inv(L)
        frac = dr @ invL                        # (..., 3) fractional coords
        frac = frac - np.round(frac)            # wrap to [-0.5, 0.5)
        return frac @ L

    def _phi(self, r: np.ndarray) -> np.ndarray:
        """Per-pair Coulomb factor in K·Å (multiply by q_i*q_j to get energy in K)."""
        Rc = self.cutoff
        if self.method == "direct":
            return COULOMB_K_KELVIN_ANGSTROM / r
        elif self.method == "wolf":
            a = self.wolf_alpha
            shift = erfc(a * Rc) / Rc
            return COULOMB_K_KELVIN_ANGSTROM * (erfc(a * r) / r - shift)
        elif self.method == "smeared": # Gaussian-smeared point charges
            # φ(r) = KE · erf(r/σ) / r — finite at r=0, → KE/r at large r
            sigma = self.gauss_width
            return COULOMB_K_KELVIN_ANGSTROM * erf(r / sigma) / r

    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels) -> PotentialEnergy:
        actual_host = self.host_override if self.host_override is not None else host
        r_center = np.asarray(r_center)
        sites_lab = r_center + fluid_sites @ rot.T
        host_q = actual_host.charges
        host_pos = actual_host.positions
        total = 0.0
        cutoff2 = self.cutoff * self.cutoff
        for s_idx, label in enumerate(fluid_site_labels):
            q_s = self.fluid_charges.get(label, 0.0)
            if q_s == 0.0:
                continue
            dr = host_pos - sites_lab[s_idx]          # (Na, 3)
            if self.mic_lattice is not None:
                dr = self._mic(dr)
            r2 = np.einsum("ad,ad->a", dr, dr)
            mask = (r2 < cutoff2) & (r2 > 0.0)
            if not np.any(mask):
                continue
            r_safe = np.sqrt(r2[mask])
            phi = self._phi(r_safe)
            total += float(q_s * (host_q[mask] * phi).sum())
        return PotentialEnergy(total=total, parts={"Coulomb": total})

    def energy_grid(self, grid_xyz, rot, host, fluid_sites, fluid_site_labels, use_warp):
        if use_warp:
            if self.method == "wolf":
                # Wolf summation has no warp kernel; fall back to NumPy
                use_warp = False
            else:
                from porecdft.warp_backend import coulomb_vext_grid_warp
                site_offset = fluid_sites @ rot.T
                actual_host = self.host_override if self.host_override is not None else host
                host_q = actual_host.charges
                host_pos = actual_host.positions
                site_q = np.array([self.fluid_charges.get(label, 0.0)
                                    for label in fluid_site_labels])
                # sigma_eff=0.0 signals the kernel to use bare 1/r (direct Coulomb)
                sigma_eff = self.gauss_width if self.method == "smeared" else 0.0
                prefactor_K = COULOMB_K_KELVIN_ANGSTROM
                return coulomb_vext_grid_warp(
                    grid_xyz,
                    site_offset,
                    site_q,
                    host_pos,
                    host_q,
                    sigma_eff,
                    prefactor_K,
                    self.cutoff,
                    mic_lattice=self.mic_lattice,  # None if not set → no MIC in kernel
                )

        else: # numpy implementation
            actual_host = self.host_override if self.host_override is not None else host
            grid = np.asarray(grid_xyz)
            sites_lab = grid[:, None, :] + (fluid_sites @ rot.T)[None, :, :]  # (Ng, S, 3)
            host_pos = actual_host.positions
            host_q = actual_host.charges
            total = np.zeros(len(grid), dtype=float)
            cutoff2 = self.cutoff * self.cutoff
            use_mic = self.mic_lattice is not None
            if use_mic:
                invL = np.linalg.inv(np.asarray(self.mic_lattice))
                L = np.asarray(self.mic_lattice)
            for s_idx, label in enumerate(fluid_site_labels):
                q_s = self.fluid_charges.get(label, 0.0)
                if q_s == 0.0:
                    continue
                site = sites_lab[:, s_idx, :]                     # (Ng, 3)
                dr = host_pos[None, :, :] - site[:, None, :]      # (Ng, Na, 3)
                if use_mic:
                    frac = dr @ invL                               # (Ng, Na, 3)
                    frac = frac - np.round(frac)
                    dr = frac @ L
                r2 = np.einsum("gad,gad->ga", dr, dr)
                mask = (r2 < cutoff2) & (r2 > 0.0)
                r_safe = np.sqrt(np.where(mask, r2, 1.0))
                phi = np.where(mask, self._phi(r_safe), 0.0)
                total += q_s * (phi * host_q[None, :]).sum(axis=1)
            return total


class SmearCoulombPotential(CoulombPotential):
    """Gaussian-smeared Coulomb potential — a convenience subclass of CoulombPotential.

    Equivalent to ``CoulombPotential(fluid_charges, method="smeared", gauss_width=gauss_width)``.
    The smearing avoids singularities at short host–fluid distances that arise when
    Hirshfeld charges and a close-contact geometry (e.g. ALF formate-H at ~2.5 Å) are
    combined with point-charge Coulomb.
    """

    def __init__(
        self,
        fluid_charges: dict[str, float],
        gauss_width: float = 1.0,
        cutoff: float = 15.0,
    ):
        super().__init__(
            fluid_charges=fluid_charges,
            cutoff=cutoff,
            method="smeared",
            gauss_width=gauss_width,
        )

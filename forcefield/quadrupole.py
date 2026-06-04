"""Linear-molecule quadrupole coupling to the host electric-field-gradient (EFG).

For a linear molecule (CO₂, N₂) with quadrupole moment Θ_zz along its body z-axis,
the electrostatic interaction with the host can be decomposed as

    V = q_C·Φ(r_C) + q_O·Φ(r_O1) + q_O·Φ(r_O2)              ← handled by CoulombPotential
      + ½·Θ_ij · ∂_i ∂_j Φ(r_center)                          ← this module
      + higher multipoles (ignored).

For a linear molecule rotated by R (body-frame z-axis → lab-frame direction u),
the lab-frame quadrupole tensor is

    Θ_ij = Θ_zz · (3 u_i u_j - δ_ij) / 2,

with trace zero. The contracted energy is

    V_quad = ½ · (Θ_zz/2) · (3 (u · ∇)² Φ - ∇² Φ)
           = (Θ_zz/2) · ( 3 (u · ∇)² Φ - 0 ) / 2     [in vacuum where ∇²Φ = 0]
           = (3 Θ_zz / 4) · (u · ∇)² Φ.

We compute ∂_i Φ at r_center from the analytic host-charge field (point charges)
and take the directional second derivative along u = R · ẑ. This gives the
correct orientation dependence without needing a finite-difference Laplacian.

NOTE: the original PC-SAFT code computed only the z-component and dropped the
orientation dependence — see `memory/project_code_state.md` bug #3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from porecdft.forcefield.base import Potential, PotentialEnergy
from porecdft.forcefield.coulomb import COULOMB_K_KELVIN_ANGSTROM
from porecdft.structure.host import HostAtoms


@dataclass(frozen=True)
class QuadrupoleEFGPotential(Potential):
    """Linear-molecule quadrupole coupling to the host EFG.

    Parameters
    ----------
    theta_zz : float
        Body-frame quadrupole moment along the molecular axis, in units of e·Å².
        For CO₂ EPM2: Θ_zz = 2·q_O · d_CO² ≈ -0.8597 e·Å² (with q_O = -0.3256 e
        and d_CO = 1.149 Å). The sign follows the convention Θ_ij = ½⟨3 r_i r_j − r² δ_ij⟩
        for the *charge distribution* (so an oblate / negative-z-end molecule has Θ_zz<0).
    cutoff : float
        Real-space cutoff in Å on the host-charge contribution to ∇∇Φ.
    """
    theta_zz: float
    cutoff: float = 15.0
    name: str = "QuadrupoleEFG"

    def _grad_grad_phi(self, r0: np.ndarray, host: HostAtoms) -> np.ndarray:
        """Analytic ∂_i∂_j Φ(r0) from point charges, in K/Å² (energy units).

        Φ(r) = KE Σ_a q_a / |r - r_a|
        ∂_i∂_j Φ = KE Σ_a q_a (3 (r-r_a)_i (r-r_a)_j - δ_ij r_a²) / |r-r_a|^5
        """
        dr = r0 - host.positions                # (Na, 3)
        r2 = np.einsum("ad,ad->a", dr, dr)
        mask = (r2 < self.cutoff**2) & (r2 > 0.0)
        if not np.any(mask):
            return np.zeros((3, 3))
        dr_m = dr[mask]
        r2_m = r2[mask]
        q_m = host.charges[mask]
        r5 = r2_m**2.5
        eye = np.eye(3)
        # outer: (Na, 3, 3)
        outer = dr_m[:, :, None] * dr_m[:, None, :]
        hess = np.einsum("a,aij,a->ij", q_m, 3.0 * outer, 1.0 / r5) \
             - np.einsum("a,ij,a->ij", q_m, eye, 1.0 / r5 * r2_m)
        return COULOMB_K_KELVIN_ANGSTROM * hess

    def energy_at(self, r_center, rot, host, fluid_sites, fluid_site_labels) -> PotentialEnergy:
        r_center = np.asarray(r_center)
        hess = self._grad_grad_phi(r_center, host)
        # Molecular z-axis direction in lab frame:
        u = rot @ np.array([0.0, 0.0, 1.0])
        directional = float(u @ hess @ u)
        v = 0.75 * self.theta_zz * directional
        return PotentialEnergy(total=v, parts={"Quad": v})


# Short alias for convenience
QuadrupolePotential = QuadrupoleEFGPotential

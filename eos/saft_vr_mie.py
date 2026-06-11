"""SAFT-VR-Mie equation of state — bulk density route only (v0.2).

Implements a simplified single-component SAFT-VR-Mie EOS following Lafitte
et al. (*J. Chem. Phys.* **2013**, 139, 154504) and the CO2 parametrisation
of Avendaño et al. (*J. Phys. Chem. B* **2011**, 115, 11154).

Scope (v0.2)
------------
* Monomer term ``a_M`` — Carnahan-Starling hard-sphere + leading-order
  (mean-field / van-der-Waals limit) Mie dispersion contribution.
* Chain term ``a_chain`` — included formally; vanishes for ``m = 1``
  (CO2). The implementation skips the radial-distribution-function piece,
  which is acceptable for the m=1 single-bead model.
* **No association**, **no cross-species mixing** — deferred to v0.3.

Only :meth:`bulk_density` (and its inverse :meth:`pressure`) are exposed.
Chemical potential, fugacity and VLE are deferred to v0.3.

References
----------
Lafitte, T.; Apostolakou, A.; Avendaño, C.; Galindo, A.; Müller, E. A.;
Jackson, G. *J. Chem. Phys.* **2013**, *139*, 154504.

Avendaño, C.; Lafitte, T.; Galindo, A.; Adjiman, C. S.; Müller, E. A.;
Jackson, G. *J. Phys. Chem. B* **2011**, *115*, 11154.
"""
from __future__ import annotations

import numpy as np

from porecdft.eos.base import EOSBase
from porecdft.eos.lj_mbwr import bh_diameter

# ─── Physical constants ──────────────────────────────────────────────────────
_K_B_J_PER_K = 1.380649e-23
_BAR_PA = 1.0e5
_ANGSTROM3_M3 = 1.0e-30


def _mie_prefactor(lambda_a: float, lambda_r: float) -> float:
    """``C(λ_a, λ_r)`` from Lafitte 2013 Eq. (2)."""
    return (lambda_r / (lambda_r - lambda_a)) * (
        lambda_r / lambda_a
    ) ** (lambda_a / (lambda_r - lambda_a))


class SAFTVRMieEOS(EOSBase):
    """Single-component SAFT-VR-Mie EOS (bulk density route only).

    Parameters
    ----------
    sigma_A : float
        Mie segment diameter ``σ`` in Å.
    epsilon_K : float
        Mie well depth ``ε/k_B`` in K.
    lambda_a : float, optional
        Attractive exponent (default 6).
    lambda_r : float, optional
        Repulsive exponent (default 12 → Lennard-Jones limit).
    m : float, optional
        Number of segments per chain (default 1). Chain term vanishes when
        ``m = 1`` so the simplification is exact for CO2.
    molar_mass : float, optional
        Molar mass in g/mol (stored for convenience).
    name : str, optional
        Short identifier; defaults to ``"SAFT-VR-Mie"``.
    """

    name = "SAFT-VR-Mie"

    def __init__(
        self,
        sigma_A: float,
        epsilon_K: float,
        lambda_a: float = 6.0,
        lambda_r: float = 12.0,
        m: float = 1.0,
        molar_mass: float = 1.0,
        name: str | None = None,
    ):
        self.sigma = float(sigma_A)
        self.epsilon = float(epsilon_K)      # ε/k_B in K
        self.lambda_a = float(lambda_a)
        self.lambda_r = float(lambda_r)
        self.m = float(m)
        self.molar_mass = float(molar_mass)
        if name is not None:
            self.name = name
        self._C = _mie_prefactor(self.lambda_a, self.lambda_r)

    # ------------------------------------------------------------ structure
    def _d(self, T_K: float) -> float:
        """Barker-Henderson effective hard-sphere diameter in Å."""
        # ``bh_diameter`` expects (kT, sigma, epsilon) with kT and epsilon
        # in the same units — here both in K.
        return bh_diameter(T_K, self.sigma, self.epsilon)

    def _eta(self, rho: float, T_K: float) -> float:
        """Packing fraction ``η = (π/6) ρ m d³``.

        ``rho`` is the number density of *molecules* (Å⁻³), so the segment
        density is ``m·ρ``.
        """
        d = self._d(T_K)
        return (np.pi / 6.0) * rho * self.m * d ** 3

    # ------------------------------------------------------------ a_res
    def _a_res(self, rho: float, T_K: float) -> float:
        """Residual Helmholtz energy per molecule, ``a_res / (N k_B T)``.

        Composed of the Carnahan-Starling hard-sphere term and a
        leading-order (mean-field) Mie dispersion contribution. The chain
        term vanishes for ``m = 1`` and is therefore omitted.
        """
        eta = self._eta(rho, T_K)
        # Carnahan-Starling hard-sphere
        a_HS = (4.0 * eta - 3.0 * eta ** 2) / (1.0 - eta) ** 2
        # Leading-order Mie dispersion (van-der-Waals limit, Lafitte 2013
        # Eq. 23 retained to first order in η — integrates Mie tail from
        # σ to ∞ assuming g(r) ≈ 1):
        #   a_disp / NkT = -2π ρ m² C (ε/T) σ³ · [1/(λ_a-3) - 1/(λ_r-3)]
        a_disp = (
            -2.0 * np.pi * rho * self.m ** 2 * self._C
            * (self.epsilon / T_K) * self.sigma ** 3
            * (1.0 / (self.lambda_a - 3.0) - 1.0 / (self.lambda_r - 3.0))
        )
        return self.m * a_HS + a_disp

    def _da_res_drho(self, rho: float, T_K: float) -> float:
        """Analytic ``∂(a_res / NkT) / ∂ρ`` (units Å³).

        Hard-sphere piece differentiated analytically through ``η(ρ)``;
        dispersion piece is linear in ``ρ``.
        """
        d = self._d(T_K)
        deta_drho = (np.pi / 6.0) * self.m * d ** 3
        eta = (np.pi / 6.0) * rho * self.m * d ** 3
        # d/dη [(4η-3η²)/(1-η)²] = (4 - 2η) / (1 - η)³
        da_HS_deta = (4.0 - 2.0 * eta) / (1.0 - eta) ** 3
        da_HS_drho = self.m * da_HS_deta * deta_drho
        # a_disp linear in ρ
        da_disp_drho = (
            -2.0 * np.pi * self.m ** 2 * self._C
            * (self.epsilon / T_K) * self.sigma ** 3
            * (1.0 / (self.lambda_a - 3.0) - 1.0 / (self.lambda_r - 3.0))
        )
        return da_HS_drho + da_disp_drho

    # ------------------------------------------------------------ EOSBase
    def pressure(self, rho: float, T_K: float) -> float:
        """Pressure in **bar** at number density ``rho`` (molecules/Å³)."""
        # Z = 1 + ρ ∂(a_res/NkT)/∂ρ
        Z = 1.0 + rho * self._da_res_drho(rho, T_K)
        # P = ρ k_B T Z  with ρ in molecules/Å³ → convert to Pa
        rho_per_m3 = rho / _ANGSTROM3_M3
        P_Pa = rho_per_m3 * _K_B_J_PER_K * T_K * Z
        return float(P_Pa / _BAR_PA)

    def bulk_density(self, P_bar: float, T_K: float) -> float:
        """Solve P(ρ, T) = P_bar for ρ on the gas branch via Newton iteration.

        Initial guess: ideal gas. The Newton step is safeguarded by a
        bisection-style backtrack that keeps ``η < 0.5`` so we stay on the
        gas-like root.
        """
        P_target_Pa = P_bar * _BAR_PA
        # Ideal-gas initial guess (molecules/Å³)
        rho = (P_target_Pa / (_K_B_J_PER_K * T_K)) * _ANGSTROM3_M3
        d = self._d(T_K)
        eta_cap = 0.5
        rho_cap = eta_cap / ((np.pi / 6.0) * self.m * d ** 3)

        for _ in range(100):
            P_calc_bar = self.pressure(rho, T_K)
            f = P_calc_bar - P_bar
            if abs(f) < 1e-10 * max(P_bar, 1.0):
                return float(rho)
            # dP/dρ via small finite-difference (analytic available but
            # this is cheap and robust)
            drho = max(1e-6 * rho, 1e-18)
            dPdrho = (self.pressure(rho + drho, T_K) - P_calc_bar) / drho
            if dPdrho <= 0:
                # On gas branch dP/dρ > 0; if not, shrink step toward ideal gas
                rho *= 0.5
                continue
            step = f / dPdrho
            rho_new = rho - step
            # Keep on gas branch
            if rho_new <= 0 or rho_new > rho_cap:
                rho_new = 0.5 * (rho + rho_cap) if rho_new > rho_cap else 0.5 * rho
            if abs(rho_new - rho) < 1e-14:
                return float(rho_new)
            rho = rho_new
        return float(rho)


# ─── Pre-built singletons ────────────────────────────────────────────────────

#: CO2 — Avendaño et al. 2011 SAFT-VR-Mie parameters
#: σ = 3.741 Å, ε/k_B = 353.55 K, λ_a = 6, λ_r = 23, m = 1
CO2_SAFT_VR_Mie = SAFTVRMieEOS(
    sigma_A=3.741,
    epsilon_K=353.55,
    lambda_a=6.0,
    lambda_r=23.0,
    m=1.0,
    molar_mass=44.01,
    name="CO2_SAFT_VR_Mie",
)

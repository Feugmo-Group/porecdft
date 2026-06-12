"""Cubic-Plus-Association (CPA) equation of state.

CPA combines a Soave-Redlich-Kwong (SRK) cubic term with a Wertheim
TPT-1 association term, giving a thermodynamically consistent EOS for
hydrogen-bonding fluids (water, alcohols, amines, glycols, carboxylic
acids).

The pressure is

    P = P_SRK + P_assoc,

with

    P_SRK   = R T / (V_m − b) − a(T) / (V_m (V_m + b)),
    P_assoc = − R T / V_m · (1/2) · (1 + ρ ∂ln g/∂ρ) · Σ_sites (1 − X_A),

where the site monomer fractions ``X_A`` solve the mass-action law

    X_A = 1 / (1 + ρ Σ_B X_B Δ_AB),
    Δ_AB = b g(η) (exp(ε_AB / k_B T) − 1) β_AB,
    g(η) = (2 − η) / (2 (1 − η)^3),
    η = b ρ / 4.

For the symmetric *n*-site schemes used in this module (4C for water,
2B for alcohols) every site couples to one symmetric partner, so all
``X_A`` collapse to a single ``X`` with closed-form solution

    X = (−1 + √(1 + 8 ρ Δ)) / (4 ρ Δ).

References
----------
Kontogeorgis, G. M.; Voutsas, E. C.; Yakoumis, I. V.; Tassios, D. P.
*Ind. Eng. Chem. Res.* **1996**, 35, 4310.

Kontogeorgis, G. M.; Yakoumis, I. V.; Meijer, H.; Hendriks, E.;
Moorwood, T. *Ind. Eng. Chem. Res.* **1999**, 38, 4453.
"""
from __future__ import annotations

import math

# JIT-safety note: this module uses Python ``math`` + scalar Python control flow
# in the Newton iteration of :meth:`bulk_density`, so it is **not** wrappable
# in ``jax.jit``. The returned ``float`` is fully usable as a scalar on GPU at
# the porecdft architecture level (cDFT calls ``bulk_density`` once per (P, T)
# state point and uses the scalar density as the bulk reference). A JIT-safe
# rewrite using ``jax.lax.while_loop`` is deferred to v0.3.
from porecdft.eos.base import EOSBase
from porecdft.eos.cubic_utils import (
    R_GAS_J_MOL_K,
    N_A,
    bar_to_Pa,
    number_density_from_Z,
    solve_cubic_gas_root,
)


# ─────────────────────────────────────────────────────────────────────────────


def _vm_from_rho(rho: float) -> float:
    """Convert number density (molecules / Å³) to molar volume (m³/mol)."""
    return N_A / (rho * 1e30)


def _rho_mol_per_m3(rho: float) -> float:
    """Number density in molecules/Å³ → mol/m³ (so it pairs with R, b in SI)."""
    return rho * 1e30 / N_A


class CPAEOS(EOSBase):
    """Cubic-Plus-Association EOS (SRK + Wertheim TPT-1).

    Parameters
    ----------
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    c1 : float
        Soave alpha-function parameter — *fitted* in CPA (not the SRK κ(ω)
        correlation), see Kontogeorgis 1999.
    eps_assoc : float
        Association energy ε/k_B in K.
    beta_assoc : float
        Association volume (dimensionless).
    n_sites : int, optional
        Number of association sites (4 for water 4C, 2 for alcohol 2B).
        Sites are assumed to come in symmetric donor/acceptor pairs so that
        a single monomer fraction ``X`` describes every site (a standard
        4C/2B simplification).
    molar_mass : float, optional
        Molar mass in g/mol.
    name : str, optional
        Short identifier (default ``"CPA"``).
    """

    name = "CPA"
    #: NumPy/math-based Newton iteration — not jax.jit-compatible.
    #: bulk_density returns Python float → safe to use on GPU as a scalar.
    JIT_SAFE = False
    GPU_READY = True

    def __init__(
        self,
        Tc: float,
        Pc: float,
        c1: float,
        eps_assoc: float,
        beta_assoc: float,
        n_sites: int = 4,
        molar_mass: float = 18.015,
        name: str | None = None,
    ):
        self.Tc = Tc
        self.Pc = Pc
        self.c1 = c1
        self.eps_assoc = eps_assoc
        self.beta_assoc = beta_assoc
        self.n_sites = n_sites
        self.molar_mass = molar_mass
        if name is not None:
            self.name = name

        self._a0 = 0.42748 * R_GAS_J_MOL_K**2 * Tc**2 / Pc
        self._b_val = 0.08664 * R_GAS_J_MOL_K * Tc / Pc

    # ───────────────────────────────────────────────── SRK part
    def _a(self, T_K: float) -> float:
        """Attractive parameter a(T) = a₀ α(T) (SRK / CPA convention)."""
        alpha = (1.0 + self.c1 * (1.0 - math.sqrt(T_K / self.Tc))) ** 2
        return self._a0 * alpha

    def _b(self) -> float:
        """Co-volume parameter b."""
        return self._b_val

    # ───────────────────────────────────────────────── association part
    def _Delta(self, rho: float, T_K: float) -> float:
        """Wertheim association strength Δ_AB (m³/mol).

        ``rho`` is in molecules / Å³.
        """
        b = self._b()
        eta = b * _rho_mol_per_m3(rho) / 4.0
        g = (2.0 - eta) / (2.0 * (1.0 - eta) ** 3)
        return (
            b
            * g
            * (math.exp(self.eps_assoc / T_K) - 1.0)
            * self.beta_assoc
        )

    def _X(self, rho: float, T_K: float) -> float:
        """Site monomer fraction ``X`` for the symmetric (4C / 2B) scheme.

        Closed form:

            X = (−1 + √(1 + 8 ρ Δ)) / (4 ρ Δ).

        Returns 1.0 in the ρ → 0 limit (no association).
        """
        Delta = self._Delta(rho, T_K)
        rho_mol = _rho_mol_per_m3(rho)
        x = rho_mol * Delta
        if x <= 0.0:
            return 1.0
        # numerically stable form: X = 2 / (1 + sqrt(1 + 8x))
        return 2.0 / (1.0 + math.sqrt(1.0 + 8.0 * x))

    def _Z_assoc(self, rho: float, T_K: float) -> float:
        """Association contribution to the compressibility factor.

            Z_assoc = − (1/2) (1 + ρ ∂ln g / ∂ρ) Σ_sites (1 − X_A).

        With g(η) = (2 − η)/(2 (1 − η)³) and η = b ρ / 4:

            ρ ∂ln g / ∂ρ = η · [−1/(2 − η) + 3/(1 − η)].
        """
        b = self._b()
        eta = b * _rho_mol_per_m3(rho) / 4.0
        if eta <= 0.0:
            return 0.0
        d_lng = eta * (-1.0 / (2.0 - eta) + 3.0 / (1.0 - eta))
        X = self._X(rho, T_K)
        return -0.5 * (1.0 + d_lng) * self.n_sites * (1.0 - X)

    # ───────────────────────────────────────────────── EOSBase API
    def pressure(self, rho: float, T_K: float) -> float:
        """Pressure (bar) at density ``rho`` (molecules / Å³) and T (K)."""
        Vm = _vm_from_rho(rho)
        a = self._a(T_K)
        b = self._b()

        # SRK: P = RT / (Vm − b) − a / (Vm (Vm + b))
        P_srk = R_GAS_J_MOL_K * T_K / (Vm - b) - a / (Vm * (Vm + b))

        # Association: P_assoc = ρ R T Z_assoc
        rho_mol = _rho_mol_per_m3(rho)
        P_assoc = rho_mol * R_GAS_J_MOL_K * T_K * self._Z_assoc(rho, T_K)

        return float((P_srk + P_assoc) / 1e5)

    def bulk_density(self, P_bar: float, T_K: float) -> float:
        """Gas-phase number density (molecules / Å³) at *(P, T)*.

        Strategy: take the SRK gas-branch density as the initial guess
        (association pulls the density only slightly upward in the dilute
        vapour regime), then Newton-iterate on
        ``f(ρ) = pressure(ρ, T) − P_bar``.
        """
        P_Pa = bar_to_Pa(P_bar)
        a = self._a(T_K)
        b = self._b()

        # ── initial guess from pure SRK cubic ──
        A = a * P_Pa / (R_GAS_J_MOL_K * T_K) ** 2
        B = b * P_Pa / (R_GAS_J_MOL_K * T_K)
        # SRK: Z³ − Z² + (A − B − B²) Z − A B = 0
        c2 = -1.0
        c1 = A - B - B**2
        c0 = -A * B
        Z = solve_cubic_gas_root(c2, c1, c0)
        rho = number_density_from_Z(Z, P_Pa, T_K)

        # Newton iteration on the full CPA pressure.
        for _ in range(80):
            P_curr = self.pressure(rho, T_K)
            f = P_curr - P_bar
            if abs(f) < max(1e-6 * abs(P_bar), 1e-9):
                break
            drho = max(rho * 1e-5, 1e-12)
            dfdrho = (self.pressure(rho + drho, T_K) - P_curr) / drho
            if dfdrho == 0.0 or not math.isfinite(dfdrho):
                break
            step = f / dfdrho
            # damp to keep rho positive and within physical bounds
            rho_new = rho - step
            if rho_new <= 0.0 or rho_new > 1.0:
                rho_new = rho * 0.5 if step > 0 else rho * 1.5
            rho = rho_new

        return float(rho)


# ─── Pre-built instance ───────────────────────────────────────────────────

#: Water in the 4C scheme (Kontogeorgis 1999, Ind. Eng. Chem. Res. 38, 4453).
H2O_CPA = CPAEOS(
    Tc=647.096,
    Pc=22.064e6,
    c1=0.6736,
    eps_assoc=2003.25,
    beta_assoc=0.0692,
    n_sites=4,
    molar_mass=18.015,
    name="H2O_CPA",
)

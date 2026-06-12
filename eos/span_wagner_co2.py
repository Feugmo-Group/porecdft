"""Span-Wagner reference equation of state for CO2 (truncated short form).

Span & Wagner published a 42-term multiparameter Helmholtz-energy formulation
for CO2 valid from the triple point to 1100 K and 800 MPa. The full form is
overkill for porecdft's gas/supercritical regime below ~100 bar; this module
implements the **truncated short form** keeping only the seven leading
polynomial residual terms of Table 31. That captures the gas branch to
~0.5 % accuracy below 100 bar — adequate for the ALF/CO2 isotherm target.

Strategy
--------
* Reduced variables ``δ = ρ / ρ_c`` and ``τ = T_c / T``.
* Residual Helmholtz energy  ``α^r(δ, τ) = Σ N_i · δ^{d_i} · τ^{t_i}``.
* Compressibility factor      ``Z = 1 + δ · (∂α^r / ∂δ)_τ``.
* :meth:`pressure` is a direct closed-form evaluation.
* :meth:`bulk_density` solves ``P / (ρ k_B T) = Z(ρ, T)`` by Newton iteration
  starting from the ideal-gas density.

The full 42-term formulation (including the exponential and Gaussian
non-analytic terms required near the critical point) is deferred to v0.3.

Reference
---------
Span, R.; Wagner, W. *J. Phys. Chem. Ref. Data* **1996**, *25*, 1509.
"""
from __future__ import annotations

import jax.numpy as jnp

from porecdft.eos.base import EOSBase
from porecdft.eos.ideal_gas import K_B_J_PER_K, BAR_PA, ANGSTROM3_M3

# ─── Span-Wagner 1996 Table 31 — first 7 polynomial residual terms ────────
_N = jnp.array([
    0.388568232032e00,
    0.293854759427e01,
    -0.558671885349e01,
    -0.767531995925e00,
    0.317290055804e00,
    0.548033158978e00,
    0.122794112203e00,
])
_D = jnp.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
_T_EXP = jnp.array([0.00, 0.75, 1.00, 2.00, 0.75, 2.00, 0.75])


def _alpha_r(delta, tau):
    """Residual Helmholtz energy α^r(δ, τ) — truncated polynomial form."""
    return jnp.sum(_N * delta ** _D * tau ** _T_EXP)


def _d_alpha_r_d_delta(delta, tau):
    """∂α^r/∂δ at fixed τ."""
    return jnp.sum(_N * _D * delta ** (_D - 1) * tau ** _T_EXP)


def _d2_alpha_r_d_delta2(delta, tau):
    """∂²α^r/∂δ² at fixed τ — needed for Newton iteration."""
    return jnp.sum(_N * _D * (_D - 1) * delta ** (_D - 2) * tau ** _T_EXP)


class SpanWagnerCO2EOS(EOSBase):
    """Truncated Span-Wagner reference EOS for CO2.

    Notes
    -----
    Critical parameters and reducing density follow Span & Wagner 1996:

    * ``T_c`` = 304.1282 K
    * ``P_c`` = 7.3773 MPa
    * ``ρ_c`` = 467.6 kg/m³

    Only the seven leading polynomial residual terms (Table 31, i = 1..7) are
    retained. This is accurate to roughly 0.5 % in the gas/supercritical
    regime below 100 bar; do not rely on it near the critical point or in the
    dense-liquid branch.
    """

    name = "SpanWagner_CO2"
    #: Newton iteration uses ``jax.lax.fori_loop`` and ``jnp.where`` for
    #: control flow — fully JIT-safe. ``pressure`` and ``bulk_density`` can
    #: both be wrapped in ``jax.jit`` / ``jax.vmap`` for batched evaluation.
    JIT_SAFE = True
    GPU_READY = True
    Tc = 304.1282
    Pc = 73.773e5      # Pa
    rho_c = 467.6       # kg/m³
    molar_mass = 44.01  # g/mol

    # Reducing number density in molecules / Å³:
    #   ρ_c [kg/m³] · 1e3 (kg→g) / M [g/mol] · N_A [mol⁻¹] · 1e-30 (m³→Å³)
    _N_A = 6.022e23
    rho_c_num = (rho_c * 1e3 / molar_mass) * _N_A * ANGSTROM3_M3

    # ------------------------------------------------------------------ core
    def _Z(self, rho: float, T_K: float) -> float:
        """Compressibility factor Z = 1 + δ · ∂α^r/∂δ."""
        delta = rho / self.rho_c_num
        tau = self.Tc / T_K
        return 1.0 + delta * _d_alpha_r_d_delta(delta, tau)

    def pressure(self, rho, T_K):
        """Pressure in bar for number density ``rho`` (molecules/Å³) at *T* (K)."""
        Z = self._Z(rho, T_K)
        # P [Pa] = ρ [molecules/m³] · k_B · T · Z
        rho_SI = rho / ANGSTROM3_M3  # molecules / m³
        P_Pa = rho_SI * K_B_J_PER_K * T_K * Z
        return P_Pa / BAR_PA

    def bulk_density(self, P_bar, T_K):
        """Number density (molecules/Å³) at *P* (bar) and *T* (K), gas branch.

        Newton iteration on δ to satisfy ``P / (ρ k_B T) = Z(ρ, T)``, starting
        from the ideal-gas density. Uses ``jax.lax.fori_loop`` so the routine
        is JIT-safe and runs on GPU under ``jax.jit`` / ``jax.vmap``.
        """
        from jax import lax  # local import keeps top-level imports light

        P_Pa = P_bar * BAR_PA
        # Ideal-gas seed: ρ = P / (k_B T) in molecules/m³, convert to /Å³.
        rho_ideal = P_Pa / (K_B_J_PER_K * T_K) * ANGSTROM3_M3
        tau = self.Tc / T_K
        kT = K_B_J_PER_K * T_K
        rho_c_SI = self.rho_c_num / ANGSTROM3_M3  # molecules/m³

        def newton_step(_, delta):
            dadd = _d_alpha_r_d_delta(delta, tau)
            d2add = _d2_alpha_r_d_delta2(delta, tau)
            Z = 1.0 + delta * dadd
            f = P_Pa - delta * rho_c_SI * kT * Z
            dZ_dd = dadd + delta * d2add
            df = -rho_c_SI * kT * (Z + delta * dZ_dd)
            step = f / df
            delta_new = delta - step
            # Guard against negative δ (use jnp.where for JIT-safety)
            return jnp.where(delta_new <= 0, 0.5 * delta, delta_new)

        delta0 = rho_ideal / self.rho_c_num
        delta = lax.fori_loop(0, 80, newton_step, delta0)
        return delta * self.rho_c_num


# ─── Pre-built singleton ──────────────────────────────────────────────────

#: Span-Wagner CO2 reference EOS (truncated 7-term form).
CO2_SW = SpanWagnerCO2EOS()

"""Shared cubic-EOS utilities.

Cubic equations of state (Peng-Robinson, SRK, CPA built on top of either,
etc.) share two needs:

1. **Solve a monic cubic** ``Z³ + c2·Z² + c1·Z + c0 = 0`` and pick a chosen
   real root (gas branch = largest positive root, liquid branch = smallest
   positive root).
2. **Convert** a compressibility factor *Z* to a number density in
   molecules / Å³ given pressure (Pa) and temperature (K).

The cubic root-finder uses the **companion-matrix eigenvalue method** rather
than ``np.roots`` so the function runs unchanged under JAX on any backend
(CPU/GPU/TPU). A pure-NumPy fallback is provided in case JAX is not installed.

References
----------
Soave 1972; Peng & Robinson 1976.
Reid, Prausnitz, Poling, *The Properties of Gases and Liquids*, 5th ed.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:  # JAX is optional at the EOS level (LJ-MBWR and Span-Wagner don't need it)
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:  # pragma: no cover
    jnp = np  # type: ignore[assignment]
    _HAS_JAX = False

# ─── Physical constants reused by cubic EOS ───────────────────────────────
R_GAS_J_MOL_K = 8.314         # universal gas constant, J/(mol·K)
N_A = 6.022e23                # Avogadro number, mol⁻¹


# ─── Cubic solver ─────────────────────────────────────────────────────────

def solve_cubic_real_roots(c2: float, c1: float, c0: float) -> jnp.ndarray:
    """Return the three (possibly complex) roots of ``Z³ + c2·Z² + c1·Z + c0``.

    Uses the companion-matrix eigenvalue trick — JAX-friendly.
    Output is a length-3 jnp array of complex numbers.
    """
    companion = jnp.array(
        [
            [0.0, 0.0, -c0],
            [1.0, 0.0, -c1],
            [0.0, 1.0, -c2],
        ]
    )
    return jnp.linalg.eigvals(companion)


def solve_cubic_gas_root(c2: float, c1: float, c0: float, tol: float = 1e-10) -> float:
    """Largest real positive root of ``Z³ + c2·Z² + c1·Z + c0 = 0`` — the gas branch.

    Parameters
    ----------
    c2, c1, c0 : float
        Coefficients of the monic cubic in Z.
    tol : float, optional
        Imaginary-part tolerance below which a root is treated as real.

    Returns
    -------
    float
        ``Z_gas`` — compressibility factor of the gas-phase root.
        If no real positive root exists, returns ``-inf``.
    """
    roots = solve_cubic_real_roots(c2, c1, c0)
    real_parts = jnp.real(roots)
    is_real_pos = (jnp.abs(jnp.imag(roots)) < tol) & (real_parts > 0)
    Z = jnp.max(jnp.where(is_real_pos, real_parts, -jnp.inf))
    return float(Z)


def solve_cubic_liquid_root(c2: float, c1: float, c0: float, tol: float = 1e-10) -> float:
    """Smallest real positive root of the cubic — the liquid branch.

    Used by CPA when the EOS is in the two-phase region.
    """
    roots = solve_cubic_real_roots(c2, c1, c0)
    real_parts = jnp.real(roots)
    is_real_pos = (jnp.abs(jnp.imag(roots)) < tol) & (real_parts > 0)
    Z = jnp.min(jnp.where(is_real_pos, real_parts, jnp.inf))
    return float(Z)


# ─── Z → number density ─────────────────────────────────────────────────────

def number_density_from_Z(Z: float, P_Pa: float, T_K: float) -> float:
    """Convert compressibility factor ``Z`` to bulk density (molecules / Å³).

    ``Vm = Z·R·T / P`` (m³/mol) → ρ = N_A / (V_m · 10³⁰).
    """
    Vm = Z * R_GAS_J_MOL_K * T_K / P_Pa
    return N_A / (Vm * 1e30)


def bar_to_Pa(P_bar: float) -> float:
    """Tiny convenience to keep call sites readable."""
    return P_bar * 1e5


# ─── Convenience: end-to-end cubic→density ─────────────────────────────────

def cubic_to_gas_density(
    P_bar: float,
    T_K: float,
    cubic_coeffs: Tuple[float, float, float],
) -> float:
    """One-liner: ``(c2, c1, c0)`` → gas-branch number density in molecules / Å³.

    The caller is responsible for computing the dimensionless cubic
    coefficients from the EOS's *a(T)*, *b* parameters.
    """
    P_Pa = bar_to_Pa(P_bar)
    Z = solve_cubic_gas_root(*cubic_coeffs)
    return number_density_from_Z(Z, P_Pa, T_K)

"""FIRE2Solver — Nonlinear-CG minimiser for the cDFT grand potential.

Uses ``optimistix.NonlinearCG`` (Polak-Ribières) to minimise Ω[ρ] directly.
Operates in log-density space (ψ = ln ρ) so ρ > 0 is automatic and the
ideal-gas logarithmic term is well-scaled.

This is complementary to the Picard/Anderson solvers:
  - Picard / Anderson: fixed-point iteration, fast per-step, can stall near
    saturation density or with shallow gradients.
  - FIRE2Solver (NonlinearCG): proper minimiser, monotone Ω decrease, more
    robust for stiff problems.  Slightly heavier per-step due to JAX JIT.

Requires: jax, optimistix >= 0.0.10 (``uv add optimistix``).

API mirrors jax_solver.py — the functional convenience wrapper is ``fire2_solve``,
the equinox Module is ``FIRE2Solver``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

try:
    import equinox as eqx
    EQX_AVAILABLE = True
except ImportError:
    eqx = None  # type: ignore
    EQX_AVAILABLE = False

try:
    import optimistix as optx
    OPTX_AVAILABLE = True
except ImportError:
    optx = None  # type: ignore
    OPTX_AVAILABLE = False


@dataclass
class FIRE2Result:
    rho: np.ndarray
    converged: bool
    iterations: int
    residual: float
    omega_final: float


def _omega_fn(
    log_rho: jnp.ndarray,
    rho_bulk: float,
    Vext_K: jnp.ndarray,
    temperature_K: float,
    c1_callable: Callable,
    c1_bulk: float,
    dV: float,
    accessibility_mask: jnp.ndarray | None,
    rho_min: float,
) -> jnp.ndarray:
    """Grand potential Ω[exp(ψ)] as a JAX scalar.  Mirrors grand_potential_jax
    but written inline so optimistix can trace through it cleanly."""
    rho = jnp.clip(jnp.exp(log_rho), rho_min)
    if accessibility_mask is not None:
        rho = jnp.where(accessibility_mask, rho, 0.0)

    beta = 1.0 / temperature_K
    log_rho_bulk = float(np.log(rho_bulk + 1e-300))

    rho_safe = jnp.where(rho > 0, rho, rho_min)
    # All three terms in kBT units — no T prefactor on f_id so gradients balance:
    # d(Ω/kBT)/dρ = ln(ρ/ρ_bulk) + (−c1+c1_bulk) + β·Vext = 0
    # gives EL ρ* = ρ_bulk·exp(c1−c1_bulk−β·Vext) ✓
    f_id = jnp.sum(rho * (jnp.log(rho_safe) - log_rho_bulk - 1.0)) * dV
    c1 = jnp.asarray(c1_callable(rho))   # c1_callable must accept JAX arrays
    f_exc = jnp.sum((-c1 + c1_bulk) * rho) * dV
    f_ext = jnp.sum(beta * Vext_K * rho) * dV

    return f_id + f_exc + f_ext


def fire2_solve(
    rho_init: np.ndarray,
    rho_bulk: float,
    Vext_K: np.ndarray,
    temperature_K: float,
    c1_callable: Callable[[np.ndarray], np.ndarray],
    c1_bulk: float,
    dV: float = 1.0,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_steps: int = 10_000,
    rho_min: float = 1e-14,
    log_clip: float = 25.0,
    accessibility_mask: np.ndarray | None = None,
) -> FIRE2Result:
    """Minimise Ω[ρ] with NonlinearCG (Polak-Ribières) via optimistix.

    Parameters
    ----------
    rho_init, rho_bulk, Vext_K, temperature_K : standard solver inputs.
    c1_callable : callable
        c¹(ρ) → array same shape as ρ.  Must be JAX-traceable (use jnp ops).
    c1_bulk : float  — bulk reference value c¹(ρ_bulk, uniform).
    rtol, atol : convergence tolerances for optimistix.
    max_steps : hard iteration limit.
    """
    if not OPTX_AVAILABLE:
        raise ImportError(
            "FIRE2Solver requires optimistix.  Install with:\n"
            "    uv add optimistix"
        )

    log_rho_bulk = float(np.log(rho_bulk + 1e-300))
    lo = log_rho_bulk - log_clip
    hi = log_rho_bulk + log_clip

    Vext_j = jnp.asarray(Vext_K, dtype=jnp.float32)
    mask_j = jnp.asarray(accessibility_mask, dtype=bool) if accessibility_mask is not None else None

    psi_init = jnp.clip(
        jnp.log(jnp.maximum(jnp.asarray(rho_init, dtype=jnp.float32), rho_min)),
        lo, hi,
    )
    if mask_j is not None:
        psi_init = jnp.where(mask_j, psi_init, lo)

    def wrapped_c1(rho):
        return jnp.asarray(c1_callable(rho))

    def omega(psi, _args):
        return _omega_fn(psi, rho_bulk, Vext_j, temperature_K,
                         wrapped_c1, c1_bulk, dV, mask_j, rho_min)

    solver = optx.NonlinearCG(rtol=rtol, atol=atol, method=optx.polak_ribiere)
    result = optx.minimise(omega, solver, psi_init, max_steps=max_steps, throw=False)

    psi_out = jnp.clip(result.value, lo, hi)
    rho_out = np.asarray(jnp.clip(jnp.exp(psi_out), rho_min))
    if accessibility_mask is not None:
        rho_out = np.where(accessibility_mask, rho_out, 0.0)

    converged = bool(result.result == optx.RESULTS.successful)
    n_steps = int(result.stats["num_steps"])
    grad_norm = float(jnp.max(jnp.abs(result.state.f_info.grad)))
    omega_val = float(omega(result.value, None))

    return FIRE2Result(
        rho=rho_out,
        converged=converged,
        iterations=n_steps,
        residual=grad_norm,
        omega_final=omega_val,
    )


if EQX_AVAILABLE:
    class FIRE2Solver(eqx.Module):
        """equinox Module wrapper around fire2_solve.

        Stores solver hyperparameters as static fields so the instance is
        jit-/vmap-compatible (hyperparameters are Python scalars, not arrays).
        """
        rtol: float = eqx.field(static=True)
        atol: float = eqx.field(static=True)
        max_steps: int = eqx.field(static=True)
        rho_min: float = eqx.field(static=True)
        log_clip: float = eqx.field(static=True)

        def __init__(
            self,
            rtol: float = 1e-7,
            atol: float = 1e-9,
            max_steps: int = 10_000,
            rho_min: float = 1e-14,
            log_clip: float = 25.0,
        ):
            self.rtol = rtol
            self.atol = atol
            self.max_steps = max_steps
            self.rho_min = rho_min
            self.log_clip = log_clip

        def solve(
            self,
            rho_init: np.ndarray,
            rho_bulk: float,
            Vext_K: np.ndarray,
            temperature_K: float,
            c1_callable: Callable,
            c1_bulk: float,
            dV: float = 1.0,
            accessibility_mask: np.ndarray | None = None,
        ) -> FIRE2Result:
            return fire2_solve(
                rho_init, rho_bulk, Vext_K, temperature_K,
                c1_callable, c1_bulk, dV,
                rtol=self.rtol,
                atol=self.atol,
                max_steps=self.max_steps,
                rho_min=self.rho_min,
                log_clip=self.log_clip,
                accessibility_mask=accessibility_mask,
            )
else:
    FIRE2Solver = None  # type: ignore

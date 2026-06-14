"""JAX-native grand-potential minimiser for cDFT — equinox Module API.

Recasts the self-consistent cDFT problem as direct minimisation of the
grand potential Ω[ρ] using automatic differentiation and any optax
optimizer (Adam, SGD, Yogi, …).

Grand potential (in K × Å³ units, divided by k_B):

    Ω[ρ] = T ∫ ρ ln(ρ/ρ_bulk) dV          ← ideal part
          + ∫ (−c¹(ρ) + c¹_bulk) ρ dV      ← FMT excess (linearised)
          + β ∫ V_ext ρ dV                  ← external field

Setting ∂Ω/∂ρ = 0 recovers the Picard equation ρ = ρ_bulk exp[−βV + c¹ − c¹_bulk],
so gradient descent on Ω is equivalent to Picard but:

  - Ω decreases monotonically → principled convergence criterion
  - `jax.grad` flows through the full pipeline (inverse problems, FF fitting)
  - Any optax optimizer works out of the box
  - `jax.jit` compiles the inner loop to XLA; GPU-ready on CUDA boxes
  - equinox.Module stores optimizer + hyperparameters as a frozen dataclass,
    making the solver `jax.jit`- and `jax.vmap`-compatible

Log-density parametrisation: ψ = log ρ, ρ = exp ψ → ρ > 0 without constraints.

Usage
-----
    import optax, equinox as eqx
    from porecdft.solver import GrandPotentialSolver

    solver = GrandPotentialSolver(
        optimizer=optax.adam(learning_rate=5e-3),
        n_steps=2000,
        tol=1e-5,
    )
    result = solver.solve(
        rho_init, rho_bulk, Vext_K, temperature_K,
        c1_callable, c1_bulk, dV,
        accessibility_mask=access,
    )
    rho = result.rho

Functional convenience wrapper (no equinox dependency at call site):

    from porecdft.solver import jax_solve
    result = jax_solve(rho_init, rho_bulk, Vext_K, T, c1_fn, c1_bulk, dV)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

try:
    import equinox as eqx            # type: ignore[import]
    EQX_AVAILABLE = True
except ImportError:
    eqx = None                       # type: ignore[assignment]
    EQX_AVAILABLE = False

try:
    import optax as _optax           # type: ignore[import]
    OPTAX_AVAILABLE = True
except ImportError:
    _optax = None                    # type: ignore[assignment]
    OPTAX_AVAILABLE = False


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class JaxSolverResult:
    rho: np.ndarray
    converged: bool
    iterations: int
    omega_history: list[float]
    error_history: list[float]

def F_ex_quadrature(rho: jnp.ndarray,
                    c1_callable: Callable[[jnp.ndarray], jnp.ndarray],
                    c1_bulk: float,
                    dV: float,
                    n_quad: int = 8,
                    ):
    """
    calculate excess free energy functional from quadrature rule
    """
    def integrant(l):
        rho_l = l * rho
        # avoid singularites ate rho=0
        rho_l = jax.lax.cond(
                 jnp.allclose(rho_l, jnp.zeros_like(rho)),
                 lambda _: rho_l + 1e-12,
                 lambda _: rho_l,
                operand=None
                )
        c1 = c1_callable(rho_l)
        return jnp.sum((-c1 + c1_bulk) * rho) * dV

    grids, wts = np.polynomial.legendre.leggauss(n_quad) # calcuate Legendre quadrature weights
    # transform to the range [0, 1]
    a, b = 0.0, 1.0
    grids = 0.5 * (b - a) * grids + 0.5 * (a + b)
    wts = 0.5 * (b - a) * wts
    # convert to jnp array
    grids = jnp.asarray(grids)
    wts = jnp.asarray(wts)
    vals = jax.vmap(integrant)(grids)
    return jnp.sum(wts*vals)

# ── Grand potential (pure JAX, differentiable) ───────────────────────────────

def grand_potential_jax(
    log_rho: jnp.ndarray,
    rho_bulk: float,
    Vext_K: jnp.ndarray,
    temperature_K: float,
    c1_callable: Callable[[jnp.ndarray], jnp.ndarray],
    c1_bulk: float,
    dV: float,
    accessibility_mask: jnp.ndarray | None = None,
    quadrature: bool = False
) -> jnp.ndarray:
    """Grand potential Ω[exp(ψ)] as a differentiable JAX scalar (K·Å³ / k_B).

    Parameters
    ----------
    log_rho : jnp.ndarray
        ψ = log ρ, any shape.
    rho_bulk, temperature_K, c1_bulk, dV : float
        Reservoir state and voxel volume.
    c1_callable : callable
        c¹(ρ) → array same shape as ρ.  Must use jnp ops.
    accessibility_mask : bool array, optional
        Inaccessible voxels are excluded from Ω.
    """
    rho = jnp.exp(log_rho)
    beta = 1.0 / temperature_K
    log_rho_bulk = float(np.log(rho_bulk + 1e-300))

    if accessibility_mask is not None:
        rho = jnp.where(accessibility_mask, rho, 0.0)

    # Ideal part in kBT units: ∫ ρ (ψ − log ρ_bulk − 1) dV
    # −1 is essential: d/dρ [ρ(ln ρ − ln ρ_bulk − 1)] = ln ρ − ln ρ_bulk,
    # which (with β·Vext term) gives EL ρ* = ρ_bulk·exp(−β·V_ext).
    # No T factor here — all three terms are in units of kBT so gradients balance.
    f_id = jnp.sum(rho * (log_rho - log_rho_bulk - 1.0)) * dV
    if quadrature:
        f_exc = F_ex_quadrature(rho, c1_callable, c1_bulk, dV)

    else:
        # Excess: ∫ (−c¹(ρ) + c¹_bulk) ρ dV
        c1 = c1_callable(rho)
        f_exc = jnp.sum((-c1 + c1_bulk) * rho) * dV

    # External field: β ∫ V_ext ρ dV
    f_ext = jnp.sum(beta * Vext_K * rho) * dV

    return f_id + f_exc + f_ext


# ── equinox Module solver ────────────────────────────────────────────────────

def _make_solver_class():
    """Return GrandPotentialSolver class (requires equinox + optax)."""

    class GrandPotentialSolver(eqx.Module):
        """Minimise Ω[ρ] with any optax optimizer.

        Stores the optimizer as an equinox field so the whole solver is
        jit-/vmap-compatible.  Pass it by value — equinox Modules are
        immutable pytrees.

        Parameters
        ----------
        optimizer : optax.GradientTransformation
            e.g. ``optax.adam(1e-3)`` or ``optax.sgd(1e-2)``.
        n_steps : int
            Maximum gradient steps.
        tol : float
            Stop when |Ω_n − Ω_{n−1}| < tol.
        log_clip : float
            Clamp |log ρ − log ρ_bulk| ≤ log_clip.
        print_every : int
            Print progress every N steps (0 = silent).
        """
        optimizer: _optax.GradientTransformation
        n_steps: int = eqx.field(static=True)
        tol: float = eqx.field(static=True)
        log_clip: float = eqx.field(static=True)
        print_every: int = eqx.field(static=True)
        quadrature: bool = eqx.field(static=True)

        def __init__(
            self,
            optimizer=None,
            n_steps: int = 2000,
            tol: float = 1e-5,
            log_clip: float = 25.0,
            print_every: int = 0,
            quadrature: bool = False,
        ):
            self.optimizer = optimizer if optimizer is not None else _optax.adam(5e-3)
            self.n_steps = n_steps
            self.tol = tol
            self.log_clip = log_clip
            self.print_every = print_every
            self.quadrature = quadrature

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
        ) -> JaxSolverResult:
            """Run the minimisation and return a JaxSolverResult."""
            quadrature = self.quadrature
            log_rho_bulk = float(np.log(rho_bulk + 1e-300))
            lo = log_rho_bulk - self.log_clip
            hi = log_rho_bulk + self.log_clip

            Vext_j = jnp.asarray(Vext_K, dtype=jnp.float32)
            mask_j = (
                jnp.asarray(accessibility_mask, dtype=bool)
                if accessibility_mask is not None else None
            )

            log_rho = jnp.clip(
                jnp.log(jnp.maximum(jnp.asarray(rho_init, dtype=jnp.float32), 1e-30)),
                lo, hi,
            )
            if mask_j is not None:
                log_rho = jnp.where(mask_j, log_rho, lo)

            opt_state = self.optimizer.init(log_rho)

            def omega_fn(lr):
                return grand_potential_jax(
                    lr, rho_bulk, Vext_j, temperature_K,
                    c1_callable, c1_bulk, dV, mask_j, quadrature
                )

            @jax.jit
            def step(lr, opt_s):
                omega, grad = jax.value_and_grad(omega_fn)(lr)
                if mask_j is not None:
                    grad = jnp.where(mask_j, grad, 0.0)
                updates, new_opt_s = self.optimizer.update(grad, opt_s)
                new_lr = _optax.apply_updates(lr, updates)
                new_lr = jnp.clip(new_lr, lo, hi)
                if mask_j is not None:
                    new_lr = jnp.where(mask_j, new_lr, lo)
                return new_lr, new_opt_s, omega

            omega_history: list[float] = []
            error_history: list[float] = []
            prev_omega = float("inf")
            converged = False
            # Relative tolerance on Ω AND a minimum-iteration floor so the
            # solver never stops on float-precision noise alone (the absolute
            # check used previously tripped on |ΔΩ| < 1e-5 at high pressure,
            # where Ω ~ 10^6 and 1e-5 / 10^6 = 1e-11 is at float32 precision).
            # We additionally require a *streak* of small steps to filter
            # one-off noise blips before declaring convergence.
            min_iters = max(200, int(0.05 * self.n_steps))
            small_step_streak_needed = 5
            small_step_streak = 0

            for i in range(self.n_steps):
                log_rho, opt_state, omega_val = step(log_rho, opt_state)
                omega_f = float(omega_val)
                delta = abs(omega_f - prev_omega)
                rel_delta = delta / max(abs(omega_f), 1.0)
                omega_history.append(omega_f)
                error_history.append(rel_delta)

                if self.print_every > 0 and (
                    i % self.print_every == 0 or i == self.n_steps - 1
                ):
                    print(f"  step {i:4d}  Ω = {omega_f:.6g}  "
                          f"|ΔΩ|/|Ω| = {rel_delta:.2e}")
                if not jnp.isfinite(omega_f):
                    print("Warning: grand potential diverges!")
                    break

                if i >= min_iters and rel_delta < self.tol:
                    small_step_streak += 1
                    if small_step_streak >= small_step_streak_needed:
                        converged = True
                        break
                else:
                    small_step_streak = 0
                prev_omega = omega_f

            rho_final = np.asarray(jnp.exp(log_rho))
            return JaxSolverResult(
                rho=rho_final,
                converged=converged,
                iterations=i + 1,
                omega_history=omega_history,
                error_history=error_history,
            )

    return GrandPotentialSolver


if EQX_AVAILABLE and OPTAX_AVAILABLE:
    GrandPotentialSolver = _make_solver_class()
else:
    GrandPotentialSolver = None  # type: ignore[assignment]


# ── Functional convenience wrapper ───────────────────────────────────────────

def jax_solve(
    rho_init: np.ndarray,
    rho_bulk: float,
    Vext_K: np.ndarray,
    temperature_K: float,
    c1_callable: Callable,
    c1_bulk: float,
    dV: float = 1.0,
    optimizer=None,
    n_steps: int = 2000,
    tol: float = 1e-5,
    accessibility_mask: np.ndarray | None = None,
    log_clip: float = 25.0,
    print_every: int = 0,
    quadrature: bool = False,
) -> JaxSolverResult:
    """Convenience wrapper: create a GrandPotentialSolver and call .solve().

    Requires both ``equinox`` and ``optax``.  Install with::

        pip install equinox optax
    """
    if not EQX_AVAILABLE or not OPTAX_AVAILABLE:
        raise ImportError(
            "jax_solve requires equinox and optax:\n"
            "    pip install equinox optax"
        )
    import optax  # noqa: F811

    if optimizer is None:
        optimizer = optax.adam(learning_rate=5e-3)

    solver = GrandPotentialSolver(
        optimizer=optimizer,
        n_steps=n_steps,
        tol=tol,
        log_clip=log_clip,
        print_every=print_every,
        quadrature = quadrature
    )
    return solver.solve(
        rho_init, rho_bulk, Vext_K, temperature_K,
        c1_callable, c1_bulk, dV,
        accessibility_mask=accessibility_mask,
    )

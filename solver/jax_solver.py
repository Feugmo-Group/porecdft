"""JAX-native grand-potential minimiser for cDFT.

Recasts the self-consistent cDFT problem as direct minimisation of the
grand potential Ω[ρ] using automatic differentiation and any optax
optimizer (Adam, SGD, L-BFGS via optax-contrib, etc.).

Grand potential (in Kelvin × Å³ units, i.e. Ω / k_B):

    Ω[ρ] = ∫ dV [ ρ (ln ρ − 1) − c¹_bulk ρ            ← ideal + linearised excess ref
                 + Φ_exc(ρ)                               ← FMT excess free-energy density
                 + β V_ext ρ                              ← external field
                 − ln(ρ_bulk) ρ ]                         ← chemical potential

    = T ∫ dV ρ ln(ρ / ρ_bulk) + F_exc[ρ] + ∫ dV V_ext ρ − const

The gradient ∂Ω/∂ρ = 0 is the Euler-Lagrange equation:

    ln ρ − ln ρ_bulk + β V_ext − c¹(ρ) + c¹_bulk = 0
    ρ(r) = ρ_bulk · exp[−β V_ext + c¹(ρ) − c¹_bulk]        ← the Picard equation

so gradient descent on Ω is equivalent to (but more robust than) Picard.

Advantages over fixed-point iteration
--------------------------------------
- Principled convergence: Ω decreases monotonically along the gradient.
- Any optax optimizer (Adam, SGD, Adagrad, Yogi, …) works out of the box.
- `jax.grad` flows through the full pipeline → usable for inverse problems
  (fit FF parameters to experimental isotherms, learn neural Φ₃, etc.).
- With `jax.jit` the inner loop compiles to XLA once; on a CUDA box the
  FFT convolutions in c¹ run on GPU automatically.
- The `warp` backend flag (mirrors fluidax) will eventually route the Φ
  evaluation through a fused Warp kernel for further GPU speedup.

Log-density parametrisation
-----------------------------
We minimise over ψ = log ρ so that ρ = exp ψ > 0 always — no positivity
constraint needed.  The accessibility mask is applied by adding a hard
penalty: ψ is clamped to −700 (ρ ≈ 0) at inaccessible voxels.

Usage example
-------------
    from porecdft.solver import jax_solve
    import optax

    result = jax_solve(
        rho_init=rho_boltz,
        rho_bulk=rho_bulk,
        Vext_K=Vext_K,
        temperature_K=T,
        c1_callable=lambda r: compute_c1(r, wd_fn, ...),
        c1_bulk=c1_b,
        optimizer=optax.adam(learning_rate=2e-3),
        n_steps=2000,
        tol=1e-5,
        accessibility_mask=access,
    )
    rho = result.rho
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

try:
    import optax as _optax  # type: ignore[import]
    OPTAX_AVAILABLE = True
except ImportError:
    OPTAX_AVAILABLE = False


@dataclass
class JaxSolverResult:
    rho: np.ndarray
    converged: bool
    iterations: int
    omega_history: list[float]
    error_history: list[float]


def grand_potential_jax(
    log_rho: jnp.ndarray,
    rho_bulk: float,
    Vext_K: jnp.ndarray,
    temperature_K: float,
    c1_callable: Callable[[jnp.ndarray], jnp.ndarray],
    c1_bulk: float,
    dV: float,
    accessibility_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Grand potential Ω[ρ] / (k_B dV) as a differentiable JAX scalar.

    Uses the log-density parametrisation ρ = exp(ψ).

    The result has units of K (energy per k_B), integrated over the grid.
    It is divided by dV so that the gradient ∂Ω/∂ψ_i has units K/Å³
    (intensive), making it grid-spacing-independent.

    Parameters
    ----------
    log_rho : jnp.ndarray
        ψ = log ρ, shape (N,) or (Nx, Ny, Nz).
    rho_bulk, temperature_K, c1_bulk, dV : floats
        Reservoir state and voxel volume.
    c1_callable : callable
        c¹(ρ) → (N,) array.  Must be JAX-compatible (jnp ops only).
    accessibility_mask : bool array, optional
        True where the fluid is admitted.  Inaccessible voxels are not
        included in Ω so their ψ gradient is exactly zero.
    """
    rho = jnp.exp(log_rho)
    beta = 1.0 / temperature_K
    log_rho_bulk = float(jnp.log(rho_bulk + 1e-300))

    if accessibility_mask is not None:
        rho = jnp.where(accessibility_mask, rho, 0.0)

    # Ideal part: T ∫ ρ (log ρ − log ρ_bulk) dV
    # = T ∫ ρ (ψ − log ρ_bulk) dV   (note: ψ = log ρ exactly)
    f_id = temperature_K * jnp.sum(rho * (log_rho - log_rho_bulk)) * dV

    # Excess part: −∫ c¹(ρ) ρ dV  +  c¹_bulk ∫ ρ dV
    # (linearised around bulk; the full FMT free-energy density Φ is
    # recovered by integration by parts but the linearised form is
    # numerically equivalent at fixed point and has cheaper autodiff)
    c1 = c1_callable(rho)
    f_exc = jnp.sum((-c1 + c1_bulk) * rho) * dV

    # External-field part
    f_ext = jnp.sum(beta * Vext_K * rho) * dV

    return f_id + f_exc + f_ext


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
) -> JaxSolverResult:
    """Minimise Ω[ρ] with a JAX-compatible optimizer.

    Parameters
    ----------
    rho_init : ndarray
        Starting density profile (numpy, shape matches Vext_K).
    rho_bulk, Vext_K, temperature_K, c1_callable, c1_bulk, dV :
        Same semantics as ``picard_solve`` / ``anderson_solve``.
    optimizer : optax.GradientTransformation, optional
        Any optax optimizer.  Default: ``optax.adam(learning_rate=5e-3)``.
    n_steps : int
        Maximum gradient steps.
    tol : float
        Stop when |Ω_n − Ω_{n−1}| < tol.
    accessibility_mask : bool ndarray, optional
        Inaccessible voxels are held at ρ ≈ 0 throughout.
    log_clip : float
        Hard clamp on log ρ relative to log ρ_bulk.
    print_every : int
        Print progress every this many steps (0 = never).

    Returns
    -------
    JaxSolverResult
        ``.rho`` is the final density; ``.omega_history`` contains Ω per step.
    """
    if not OPTAX_AVAILABLE:
        raise ImportError(
            "optax is required for jax_solve.  Install with:\n"
            "    pip install optax\n"
            "or add it to your [dependency-groups] dev in pyproject.toml."
        )

    import optax  # noqa: F811  (shadow module-level import)

    if optimizer is None:
        optimizer = optax.adam(learning_rate=5e-3)

    log_rho_bulk = float(np.log(rho_bulk + 1e-300))
    log_clip_lo = log_rho_bulk - log_clip
    log_clip_hi = log_rho_bulk + log_clip

    # Convert inputs to JAX arrays
    Vext_j = jnp.asarray(Vext_K, dtype=jnp.float32)
    mask_j = jnp.asarray(accessibility_mask, dtype=bool) if accessibility_mask is not None else None

    # Initialise log ρ
    log_rho = jnp.log(jnp.maximum(jnp.asarray(rho_init, dtype=jnp.float32), 1e-30))
    log_rho = jnp.clip(log_rho, log_clip_lo, log_clip_hi)
    if mask_j is not None:
        log_rho = jnp.where(mask_j, log_rho, log_clip_lo)

    opt_state = optimizer.init(log_rho)

    def omega_fn(log_r):
        return grand_potential_jax(
            log_r, rho_bulk, Vext_j, temperature_K,
            c1_callable, c1_bulk, dV, mask_j,
        )

    @jax.jit
    def step(log_r, opt_s):
        omega, grad = jax.value_and_grad(omega_fn)(log_r)
        # Zero gradient at inaccessible voxels
        if mask_j is not None:
            grad = jnp.where(mask_j, grad, 0.0)
        updates, new_opt_s = optimizer.update(grad, opt_s)
        new_log_r = optax.apply_updates(log_r, updates)
        new_log_r = jnp.clip(new_log_r, log_clip_lo, log_clip_hi)
        if mask_j is not None:
            new_log_r = jnp.where(mask_j, new_log_r, log_clip_lo)
        return new_log_r, new_opt_s, omega

    omega_history: list[float] = []
    error_history: list[float] = []
    converged = False
    prev_omega = float("inf")

    for i in range(n_steps):
        log_rho, opt_state, omega_val = step(log_rho, opt_state)
        omega_f = float(omega_val)
        delta = abs(omega_f - prev_omega)
        omega_history.append(omega_f)
        error_history.append(delta)

        if print_every > 0 and (i % print_every == 0 or i == n_steps - 1):
            print(f"  step {i:4d}  Ω = {omega_f:.6g}  |ΔΩ| = {delta:.2e}")

        if delta < tol and i > 0:
            converged = True
            break
        prev_omega = omega_f

    rho_final = np.asarray(jnp.exp(log_rho))
    return JaxSolverResult(
        rho=rho_final,
        converged=converged,
        iterations=i + 1,
        omega_history=omega_history,
        error_history=error_history,
    )

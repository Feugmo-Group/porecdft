"""Tests for GrandPotentialSolver (Optax/Adam) and grand_potential_jax."""
import numpy as np
import pytest

from porecdft.solver.jax_solver import (
    EQX_AVAILABLE,
    OPTAX_AVAILABLE,
    JaxSolverResult,
    grand_potential_jax,
    jax_solve,
)

pytestmark = pytest.mark.skipif(
    not (EQX_AVAILABLE and OPTAX_AVAILABLE),
    reason="equinox and optax required",
)


def _zero_c1(rho):
    import jax.numpy as jnp
    return jnp.zeros_like(rho)


@pytest.fixture()
def params():
    N = 40
    rho_bulk = 0.01
    T = 300.0
    return np.full(N, rho_bulk), rho_bulk, np.zeros(N), T


def test_grand_potential_ideal_minimum(params):
    """Ideal gas: Ω is minimised at ρ = ρ_bulk, so dΩ/dρ → 0 there."""
    import jax.numpy as jnp
    rho_init, rho_bulk, Vext, T = params
    log_rho = jnp.log(jnp.array(rho_init, dtype=jnp.float32))
    omega = grand_potential_jax(log_rho, rho_bulk, jnp.array(Vext, dtype=jnp.float32),
                                T, _zero_c1, 0.0, 1.0)
    assert float(omega) < 1e-3   # Ω ≈ 0 at the minimum for uniform ρ = ρ_bulk


def test_jax_solve_ideal_gas_converges(params):
    rho_init, rho_bulk, Vext, T = params
    result = jax_solve(
        rho_init, rho_bulk, Vext, T, _zero_c1, 0.0, dV=1.0,
        n_steps=3000, tol=1e-6,
    )
    assert isinstance(result, JaxSolverResult)
    assert result.converged, f"did not converge in {result.iterations} steps"
    assert np.allclose(result.rho, rho_bulk, rtol=5e-3)


def test_jax_solve_accessibility_mask(params):
    rho_init, rho_bulk, Vext, T = params
    N = len(rho_init)
    mask = np.ones(N, dtype=bool)
    mask[:5] = False   # first 5 voxels inaccessible
    result = jax_solve(
        rho_init, rho_bulk, Vext, T, _zero_c1, 0.0,
        n_steps=2000, accessibility_mask=mask,
    )
    assert np.all(result.rho[:5] == pytest.approx(0.0, abs=1e-6))
    assert np.allclose(result.rho[5:], rho_bulk, rtol=1e-2)


def test_jax_solve_attractive_well(params):
    """Dense packing inside an attractive well — same logic as FIRE2 test."""
    rho_init, rho_bulk, _, T = params
    N = len(rho_init)
    Vext = np.zeros(N)
    Vext[N // 4 : 3 * N // 4] = -300.0
    result = jax_solve(
        rho_init, rho_bulk, Vext, T, _zero_c1, 0.0, n_steps=4000, tol=1e-6,
    )
    rho_in  = result.rho[N // 4 : 3 * N // 4].mean()
    rho_out = np.concatenate([result.rho[:N // 4], result.rho[3 * N // 4:]]).mean()
    assert rho_in > rho_out * 2

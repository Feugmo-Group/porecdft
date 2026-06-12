"""Tests for fire2_solve / FIRE2Solver on a 1D cDFT toy problem.

Uses a flat Vext = 0 box with no excess free energy (ideal gas limit) so
the analytic answer is ρ(r) = ρ_bulk everywhere.
"""
import numpy as np
import pytest

from porecdft.solver.fire2 import FIRE2Result, FIRE2Solver, OPTX_AVAILABLE, fire2_solve

pytestmark = pytest.mark.skipif(
    not OPTX_AVAILABLE, reason="optimistix not installed"
)


def _ideal_c1(rho):
    """c¹ = 0 for ideal gas (no excess free energy). Must use jnp so fire2 can trace it."""
    import jax.numpy as jnp
    return jnp.zeros_like(rho)


@pytest.fixture()
def ideal_params():
    N = 50
    rho_bulk = 0.01
    T = 300.0
    rho_init = np.full(N, rho_bulk)
    Vext = np.zeros(N)
    return rho_init, rho_bulk, Vext, T


def test_fire2_ideal_gas_converges(ideal_params):
    rho_init, rho_bulk, Vext, T = ideal_params
    result = fire2_solve(
        rho_init, rho_bulk, Vext, T, _ideal_c1, c1_bulk=0.0, dV=1.0,
        max_steps=2000,
    )
    assert isinstance(result, FIRE2Result)
    assert result.converged, f"did not converge after {result.iterations} steps"
    assert np.allclose(result.rho, rho_bulk, rtol=1e-3), (
        f"max |ρ − ρ_bulk| = {np.max(np.abs(result.rho - rho_bulk)):.3e}"
    )


def test_fire2_result_fields(ideal_params):
    rho_init, rho_bulk, Vext, T = ideal_params
    result = fire2_solve(rho_init, rho_bulk, Vext, T, _ideal_c1, c1_bulk=0.0)
    assert result.rho.shape == rho_init.shape
    assert isinstance(result.iterations, int) and result.iterations > 0
    assert np.isfinite(result.residual)
    assert np.isfinite(result.omega_final)


def test_fire2solver_module_api(ideal_params):
    rho_init, rho_bulk, Vext, T = ideal_params
    solver = FIRE2Solver(max_steps=2000)
    result = solver.solve(rho_init, rho_bulk, Vext, T, _ideal_c1, c1_bulk=0.0)
    assert result.converged
    assert np.allclose(result.rho, rho_bulk, rtol=1e-3)


def test_fire2_external_field(ideal_params):
    """With a well potential Vext = -ε in the centre, density should pile up there."""
    rho_init, rho_bulk, _, T = ideal_params
    N = len(rho_init)
    Vext = np.zeros(N)
    Vext[N // 4 : 3 * N // 4] = -500.0   # attractive well in K
    result = fire2_solve(
        rho_init, rho_bulk, Vext, T, _ideal_c1, c1_bulk=0.0, max_steps=5000,
    )
    # density inside the well should be higher than outside
    rho_in  = result.rho[N // 4 : 3 * N // 4].mean()
    rho_out = np.concatenate([result.rho[:N // 4], result.rho[3 * N // 4:]]).mean()
    assert rho_in > rho_out * 2, f"expected pile-up: rho_in={rho_in:.4g}, rho_out={rho_out:.4g}"

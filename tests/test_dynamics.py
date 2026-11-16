"""Exact-limit tests for the dynamic cDFT prototype (porecdft.dynamics).

Numbering follows the validation ladder of the proposal (V1 to V4):

V1  equilibrium limit: closed DDFT relaxes to the static solution (analytic for an ideal gas, Anderson solver for FMT);
    the number of molecules is conserved; the grand potential decreases.
V2  thermodynamic factor: decay rate of a long-wavelength mode is D q^2 Gamma, Gamma from the bulk functional.
V3  hopping rate: relaxation rate of a symmetric double well against Kramers' (Smoluchowski) formula.
V4  analytic uptake: slab open on one face, flat landscape, dilute gas, against the Crank series.

Run with:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python -m pytest tests/test_dynamics.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import eigs

import porecdft.dynamics as dyn  # enables jax x64
from porecdft.dynamics import DDFTProblem, bernoulli, build_operator, crank_slab, fractional_uptake, time_to_fraction
from porecdft.dynamics.flux import bernoulli_jax, face_weights, flux_divergence_jax

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from porecdft.functional import (  # noqa: E402
    bulk_c1,
    compute_c1,
    compute_weighted_densities,
    make_fmt_weights_hat,
    make_k_grid,
)
from porecdft.solver import anderson_solve  # noqa: E402

SIGMA = 3.0  # hard-sphere diameter (A)
T_K = 300.0


def fmt_c1(shape, h):
    """JAX-traceable FMT-aWBII c1 for a periodic grid, plus a NumPy wrapper for the static solvers."""
    KX, KY, KZ, K = make_k_grid(shape, dx=h, dy=h, dz=h)
    w2, w3, w2v = make_fmt_weights_hat(K, KX, KY, KZ, SIGMA)

    def c1_jax(rho):
        wd = compute_weighted_densities(rho, w2, w3, w2v, SIGMA)
        return compute_c1(rho, wd, w2, w3, w2v, SIGMA, model="aWBII")

    return c1_jax, lambda rho: np.asarray(c1_jax(rho))


# ----------------------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------------------
def test_bernoulli_properties():
    x = np.array([-800.0, -30.0, -1e-3, 0.0, 1e-7, 1e-3, 30.0, 800.0])
    B = bernoulli(x)
    assert np.all(np.isfinite(B)) and np.all(B >= 0)   # B(800) underflows to exactly 0
    assert bernoulli(0.0) == pytest.approx(1.0)
    np.testing.assert_allclose(B - bernoulli(-x), -x, atol=1e-12, rtol=1e-12)  # B(x) - B(-x) = -x
    np.testing.assert_allclose(np.asarray(bernoulli_jax(jnp.asarray(x))), B, rtol=1e-13, atol=1e-300)


@pytest.mark.parametrize("periodic", [(True, True, True), (False, True, False)])
def test_operator_conserves_mass_is_m_matrix_and_matches_jax(periodic):
    rng = np.random.default_rng(0)
    shape = (7, 5, 4)
    v = 3.0 * rng.normal(size=shape)
    rho = rng.random(shape) + 0.1
    access = rng.random(shape) > 0.1
    spacing = (0.4, 0.5, 0.6)
    A = build_operator(v, 1.3, spacing, periodic, access)
    assert np.max(np.abs(np.asarray(A.sum(axis=0)))) < 1e-12              # zero column sums: N conserved
    off = A.tocoo()
    assert np.all(off.data[off.row != off.col] >= 0.0)                      # non-negative off-diagonals
    w = [None if q is None else jnp.asarray(q) for q in face_weights(shape, 1.3, spacing, periodic, access)]
    d_jax = np.asarray(flux_divergence_jax(jnp.asarray(rho), jnp.asarray(v), w)).ravel()
    np.testing.assert_allclose(A @ rho.ravel(), d_jax, atol=1e-12)          # sparse and matrix-free agree


def test_inaccessible_voxels_are_inert():
    shape = (16, 4, 4)
    access = np.ones(shape, bool)
    access[6:9, :, :] = False                                               # a wall that blocks the periodic box
    V = np.zeros(shape)
    rho0 = np.full(shape, 0.01)
    rho0[:6] *= 3.0
    p = DDFTProblem(V, T_K, (0.5, 0.5, 0.5), 1.0, access=access, periodic=(False, True, True))
    res = p.run(rho0, 50.0, 0.5)
    np.testing.assert_array_equal(res.rho[~access], rho0[~access])         # untouched
    assert abs(res.N[-1] / res.N[0] - 1.0) < 1e-12


# ----------------------------------------------------------------------------------------------------------
# V1: equilibrium limit
# ----------------------------------------------------------------------------------------------------------
def _sinusoidal_landscape(shape, h, depth_K):
    x = (np.arange(shape[0]) + 0.5) * h
    return np.tile((-depth_K * np.cos(2 * np.pi * x / (shape[0] * h)))[:, None, None], (1, shape[1], shape[2]))


def test_ideal_gas_closed_relaxation_is_boltzmann():
    shape, h = (64, 4, 4), 0.25
    V = _sinusoidal_landscape(shape, h, 150.0)
    p = DDFTProblem(V, T_K, (h, h, h), 1.0)
    res = p.relax(np.full(shape, 0.01), dt0=0.01, growth=1.5, tol=1e-12)
    w = np.exp(-V / T_K)
    expected = 0.01 * w / np.mean(w)
    assert res.converged
    np.testing.assert_allclose(res.rho, expected, rtol=1e-9)
    assert abs(res.N[-1] / res.N[0] - 1.0) < 1e-12


def test_open_reservoir_steady_state_is_boltzmann():
    """Grand-canonical limit: a held-density gas voxel and a landscape give rho = rho_g exp(-beta V)."""
    n, h = 60, 0.5
    x = np.arange(n) * h
    V = (-200.0 * np.exp(-(((x - 15.0) / 3.0) ** 2)))[:, None, None]
    V[0] = 0.0
    fixed = np.zeros((n, 1, 1), bool)
    fixed[0] = True
    p = DDFTProblem(V, T_K, (h, 1.0, 1.0), 1.0, periodic=(False, False, False), fixed=fixed)
    rho_g = 0.02
    res = p.relax(np.zeros((n, 1, 1)), rho_fixed=rho_g, dt0=0.1, growth=1.5, tol=1e-11)
    assert res.converged
    np.testing.assert_allclose(res.rho[:, 0, 0], rho_g * np.exp(-V[:, 0, 0] / T_K), rtol=1e-6)


def test_ideal_gas_grand_potential_decreases():
    shape, h = (32, 4, 4), 0.5
    V = _sinusoidal_landscape(shape, h, 200.0)
    p = DDFTProblem(V, T_K, (h, h, h), 1.0)
    rng = np.random.default_rng(1)
    rho = 0.01 * (1.0 + 0.5 * rng.random(shape))
    rho_ref = 0.01

    def omega(r):  # beta*(F - mu N) up to a constant, N conserved
        r = r.reshape(shape)
        return np.sum(r * (np.log(r / rho_ref) - 1.0) + r * V / T_K) * p.dV

    values = [omega(rho)]
    r = rho.ravel()
    for _ in range(40):
        r, _ = p.step(r, 0.5)
        values.append(omega(r))
    assert np.all(np.diff(values) < 1e-12)
    assert values[-1] < values[0]


def test_fmt_closed_relaxation_matches_static_solver():
    """V1 with the production hard-sphere functional: DDFT at fixed N reaches the Anderson solution."""
    shape, h = (32, 4, 4), 0.5
    c1_jax, c1_np = fmt_c1(shape, h)
    V = _sinusoidal_landscape(shape, h, 150.0)
    rho_b = 0.01
    static = anderson_solve(np.full(shape, rho_b), rho_b, V, T_K, c1_np, bulk_c1(rho_b, SIGMA, model="aWBII"),
                            m=6, beta=0.15, max_iter=5000, tol=1e-10, rho_max=0.45 * 6 / (np.pi * SIGMA**3))
    assert static.converged
    p = DDFTProblem(V, T_K, (h, h, h), 1.0, c1_fn=c1_jax, jax_c1=True)
    res = p.relax(np.full(shape, static.rho.mean()), dt0=0.05, growth=1.3, tol=1e-9, dt_max=50.0)
    assert res.converged
    assert np.max(np.abs(res.rho / static.rho - 1.0)) < 1e-6
    assert abs(res.N[-1] / res.N[0] - 1.0) < 1e-10


def test_jax_and_finite_difference_paths_agree():
    shape, h = (32, 4, 4), 0.5
    c1_jax, c1_np = fmt_c1(shape, h)
    V = _sinusoidal_landscape(shape, h, 150.0)
    rho0 = np.full(shape, 0.01) * (1.0 + 0.05 * np.cos(2 * np.pi * np.arange(32) / 32))[:, None, None]
    a, _ = DDFTProblem(V, T_K, (h, h, h), 1.0, c1_fn=c1_np).step(rho0, 2.0)
    b, _ = DDFTProblem(V, T_K, (h, h, h), 1.0, c1_fn=c1_jax, jax_c1=True).step(rho0, 2.0)
    np.testing.assert_allclose(a, b, rtol=1e-6)


# ----------------------------------------------------------------------------------------------------------
# V2: thermodynamic factor
# ----------------------------------------------------------------------------------------------------------
def test_long_wavelength_decay_rate_is_D_q2_Gamma():
    Nx, h = 400, 0.5
    shape = (Nx, 4, 4)
    c1_jax, _ = fmt_c1(shape, h)
    eta = 0.2
    rho_bar = eta / (np.pi / 6 * SIGMA**3)
    d = 1e-5
    dc1 = (bulk_c1(rho_bar * (1 + d), SIGMA, model="aWBII") - bulk_c1(rho_bar * (1 - d), SIGMA, model="aWBII")) / (2 * d * rho_bar)
    gamma = 1.0 - rho_bar * dc1                                            # Gamma = d ln f / d ln rho
    q = 2 * np.pi / (Nx * h)
    xx = (np.arange(Nx) + 0.5) * h
    rho = (rho_bar * (1 + 1e-3 * np.cos(q * xx))[:, None, None] * np.ones(shape)).ravel()
    p = DDFTProblem(np.zeros(shape), T_K, (h, h, h), 1.0, c1_fn=c1_jax, jax_c1=True)

    def amp(r):
        return abs(np.sum(r.reshape(shape).mean(axis=(1, 2)) * np.exp(-1j * q * xx)))

    dt, amps = 4.0, [amp(rho)]
    for _ in range(15):
        rho, _ = p.step(rho, dt)
        amps.append(amp(rho))
    lam = (amps[-2] / amps[-1] - 1.0) / dt                                  # exact for backward Euler on a linear mode
    expected = 2 * (1 - np.cos(q * h)) / h**2 * gamma                       # D = 1, discrete Laplacian symbol
    assert gamma > 3.0                                                      # a stiff case: the explicit scheme would fail
    assert lam / expected == pytest.approx(1.0, abs=0.01)


# ----------------------------------------------------------------------------------------------------------
# V3: hopping rate
# ----------------------------------------------------------------------------------------------------------
def _double_well_rate(beta_dE, n_grid, a=1.5, D=1.0):
    L = 3 * a
    h = 2 * L / n_grid
    x = -L + (np.arange(n_grid) + 0.5) * h
    dE = beta_dE * T_K
    V = (dE * (1 - (x / a) ** 2) ** 2)[:, None, None]
    p = DDFTProblem(V, T_K, (h, 1.0, 1.0), D, periodic=(False, False, False))
    A = p.operator(np.ones(n_grid)).tocsc()
    vals = np.sort(np.real(eigs(A, k=3, sigma=-1e-12, which="LM", return_eigenvectors=False)))[::-1]
    lam = -vals[1]                                                         # slowest relaxation rate = 2 k
    kramers = D / (2 * np.pi * T_K) * np.sqrt((8 * dE / a**2) * (4 * dE / a**2)) * np.exp(-beta_dE)
    return lam, 2 * kramers


@pytest.mark.parametrize("beta_dE", [8.0, 12.0])
def test_double_well_rate_matches_kramers(beta_dE):
    lam, two_k = _double_well_rate(beta_dE, 600)
    lam_fine, _ = _double_well_rate(beta_dE, 1800)
    assert abs(lam / lam_fine - 1.0) < 0.005                               # grid converged
    assert abs(lam / two_k - 1.0) < 0.10                                   # within the proposal's 10 %
    assert lam < two_k                                                     # leading-order Kramers overestimates


# ----------------------------------------------------------------------------------------------------------
# V4: analytic uptake
# ----------------------------------------------------------------------------------------------------------
def test_slab_uptake_matches_crank_series():
    n, h, D = 200, 0.5, 1.0
    V = np.zeros((n, 1, 1))
    fixed = np.zeros((n, 1, 1), bool)
    fixed[0] = True
    p = DDFTProblem(V, T_K, (h, 1.0, 1.0), D, periodic=(False, False, False), fixed=fixed)
    ell = n * h                                                            # effective thickness of the discrete slab
    tau = ell**2 / D
    res = p.run(np.zeros(n), 2 * tau, tau / 1000, rho_fixed=1.0)
    m = res.N / ((n - 1) * h)                                               # final equilibrium: rho = 1 in all free voxels
    ref = crank_slab(res.t, D, ell)
    sel = res.t > 0.02 * tau
    assert np.max(np.abs(m - ref)[sel]) < 0.01                             # backward Euler is first order in time
    assert time_to_fraction(res.t, m, 0.9) == pytest.approx(time_to_fraction(res.t, ref, 0.9), rel=0.01)
    assert np.all(np.diff(res.N) >= -1e-12)                                # monotone uptake

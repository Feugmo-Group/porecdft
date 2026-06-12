"""Tests for WertheimAssociation — NumPy path and Warp fast path consistency."""
import numpy as np
import pytest

from porecdft.functional.association import AssociationSite, WertheimAssociation
from porecdft.warp_backend import WARP_AVAILABLE


def _make_simple_system():
    """3×3×3 grid, one association site at the centre."""
    N = 5
    xs = np.linspace(0, 1, N, endpoint=False) * 5.0    # Å
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    grid_xyz = np.stack([X, Y, Z], axis=-1)             # (5,5,5,3)
    rho_grid = np.ones((N, N, N)) * 0.01                # uniform density
    dV = (xs[1] - xs[0]) ** 3
    site = AssociationSite(
        position=np.array([2.5, 2.5, 2.5]),   # centre
        energy_K=500.0,
        kappa_A3=30.0,
    )
    assoc = WertheimAssociation(sites=[site])
    return assoc, rho_grid, grid_xyz, dV


def test_rho_bar_nonzero():
    assoc, rho_grid, grid_xyz, dV = _make_simple_system()
    rho_bar = assoc._rho_bar_all(rho_grid, grid_xyz, dV)
    assert rho_bar.shape == (1,)
    assert rho_bar[0] > 0.0


def test_fraction_unbound_between_0_and_1():
    assoc, rho_grid, grid_xyz, dV = _make_simple_system()
    X = assoc.fraction_unbound(rho_grid, grid_xyz, dV, T_K=298.0)
    assert X.shape == (1,)
    assert 0.0 < float(X[0]) < 1.0


def test_loading_contribution_positive():
    assoc, rho_grid, grid_xyz, dV = _make_simple_system()
    N_assoc = assoc.loading_contribution(rho_grid, grid_xyz, dV, T_K=298.0)
    assert 0.0 < N_assoc <= 1.0


def test_c1_correction_shape():
    assoc, rho_grid, grid_xyz, dV = _make_simple_system()
    c1 = assoc.c1_correction(rho_grid, grid_xyz, dV, T_K=298.0)
    assert c1.shape == rho_grid.shape
    assert np.any(c1 > 0.0)   # nonzero near the site


def test_c1_increases_with_energy():
    """Higher association energy → larger c¹ correction near the site."""
    _, rho_grid, grid_xyz, dV = _make_simple_system()
    site_lo = AssociationSite(position=np.array([2.5, 2.5, 2.5]), energy_K=200.0, kappa_A3=30.0)
    site_hi = AssociationSite(position=np.array([2.5, 2.5, 2.5]), energy_K=800.0, kappa_A3=30.0)
    c1_lo = WertheimAssociation(sites=[site_lo]).c1_correction(rho_grid, grid_xyz, dV, T_K=298.0)
    c1_hi = WertheimAssociation(sites=[site_hi]).c1_correction(rho_grid, grid_xyz, dV, T_K=298.0)
    assert c1_hi.max() > c1_lo.max()


@pytest.mark.skipif(not WARP_AVAILABLE, reason="warp-lang not installed")
def test_warp_path_matches_numpy_path():
    """Warp kernel should give same rho_bar as NumPy path to float32 precision."""
    assoc, rho_grid, grid_xyz, dV = _make_simple_system()
    rho_bar_np   = assoc._rho_bar_all(rho_grid, grid_xyz, dV, use_warp=False)
    rho_bar_warp = assoc._rho_bar_all(rho_grid, grid_xyz, dV, use_warp=True)
    np.testing.assert_allclose(rho_bar_warp, rho_bar_np, rtol=1e-4,
                               err_msg="Warp and NumPy rho_bar disagree")

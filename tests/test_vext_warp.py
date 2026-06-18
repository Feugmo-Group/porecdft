"""Smoke tests for the Warp 3D Vext kernels.

These tests verify that the Warp implementation matches a plain NumPy
reference on small problems.  Skip cleanly when ``warp-lang`` is not
installed.

Run with:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python \
        -m pytest tests/test_vext_warp.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from porecdft.warp_backend import WARP_AVAILABLE


pytestmark = pytest.mark.skipif(
    not WARP_AVAILABLE, reason="warp-lang not installed"
)


# ─── LJ Vext kernel ───────────────────────────────────────────────────────

def _lj_reference(grid_xyz, site_offset, host_pos,
                  sigma_ij, epsilon_ij, active, cutoff):
    """Plain NumPy LJ Vext for cross-check."""
    Ng, S = grid_xyz.shape[0], site_offset.shape[0]
    cutoff2 = cutoff * cutoff
    out = np.zeros(Ng)
    for s in range(S):
        pos = grid_xyz + site_offset[s]
        for a in range(host_pos.shape[0]):
            if not active[s, a]:
                continue
            dr = host_pos[a] - pos
            r2 = (dr * dr).sum(axis=1)
            mask = (r2 > 0) & (r2 < cutoff2)
            if not mask.any():
                continue
            sig, eps = sigma_ij[s, a], epsilon_ij[s, a]
            sr2 = sig * sig / r2[mask]
            sr6 = sr2 * sr2 * sr2
            out[mask] += 4.0 * eps * (sr6 * sr6 - sr6)
    return out


def test_lj_vext_warp_matches_numpy():
    from porecdft.warp_backend import lj_vext_grid_warp

    rng = np.random.default_rng(0)
    Ng, S, Na = 50, 2, 10
    grid_xyz = rng.normal(size=(Ng, 3)) * 2.0
    site_offset = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    host_pos = rng.normal(size=(Na, 3)) * 3.0
    sigma_ij = np.full((S, Na), 3.0)
    epsilon_ij = np.full((S, Na), 100.0)
    active = np.ones((S, Na), dtype=np.int32)
    cutoff = 15.0

    ref = _lj_reference(grid_xyz, site_offset, host_pos,
                        sigma_ij, epsilon_ij, active, cutoff)
    out = lj_vext_grid_warp(grid_xyz, site_offset, host_pos,
                            sigma_ij, epsilon_ij, active, cutoff)
    # Warp uses float32; NumPy reference is float64.  Allow 1e-3 relative.
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


# ─── Boltzmann orientation-average kernel ──────────────────────────────────

def test_boltzmann_orient_avg_matches_numpy():
    from porecdft.warp_backend import boltzmann_orient_avg_warp

    rng = np.random.default_rng(0)
    N_orient, Ng = 20, 100
    v = rng.uniform(low=-2000.0, high=2000.0, size=(N_orient, Ng))
    T_K = 298.0

    # NumPy reference: V(r;T) = -kT log<exp(-βV)>_Ω with min-shift
    beta = 1.0 / T_K
    v_min = v.min(axis=0)
    shifted = v - v_min
    boltz_mean = np.exp(-beta * shifted).mean(axis=0)
    ref = v_min - T_K * np.log(boltz_mean)

    out = boltzmann_orient_avg_warp(v, T_K)
    # float32 vs float64 — tolerant
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1.0)


# ─── build_vext_on_grid Warp vs NumPy integration ─────────────────────────

def _make_synthetic_host_and_fluid():
    """Small orthorhombic host + single-site fluid for integration testing."""
    from porecdft.structure.host import HostAtoms
    from porecdft.fluid.base import Fluid
    from porecdft.forcefield.lj import LJPotential
    from porecdft.io.forcefield import FFEntry

    # 8 C atoms on a simple-cubic lattice inside a 12×12×12 Å box.
    L = np.eye(3) * 12.0
    pos = np.array([[x, y, z]
                    for x in [3.0, 9.0]
                    for y in [3.0, 9.0]
                    for z in [3.0, 9.0]], dtype=float)
    host = HostAtoms(
        positions=pos,
        species=["C"] * 8,
        charges=np.zeros(8),
        lattice=L,
    )

    # Single-site monatomic fluid ("M")
    fluid = Fluid(
        body_sites=np.zeros((1, 3)),
        site_labels=["M"],
    )

    # LJ potential C–M: σ=3.5 Å, ε=50 K
    host_ff  = {"C": FFEntry(element="C", sigma=3.5, epsilon=50.0)}
    fluid_ff = {"M": FFEntry(element="M", sigma=3.5, epsilon=50.0)}
    potential = LJPotential(host_ff=host_ff, fluid_ff=fluid_ff, cutoff=8.0)
    return host, fluid, potential


def test_build_vext_warp_matches_numpy():
    """Warp path of build_vext_on_grid must agree with NumPy path to float32 tol."""
    host, fluid, potential = _make_synthetic_host_and_fluid()

    # Single identity orientation (monatomic fluid — orientation is irrelevant)
    orientations = np.eye(3)[None]  # (1, 3, 3)

    from porecdft.vext.builder import build_vext_on_grid

    result_np = build_vext_on_grid(
        host, fluid, potential, orientations,
        spacing=1.5, pbc_supercell=(3, 3, 3),
        use_warp=False, dtype=np.float64,
    )
    result_wp = build_vext_on_grid(
        host, fluid, potential, orientations,
        spacing=1.5, pbc_supercell=(3, 3, 3),
        use_warp=True, warp_device="cpu", dtype=np.float64,
    )

    vmin_np = result_np["orient_min"].ravel()
    vmin_wp = result_wp["orient_min"].ravel()

    # Warp kernel runs in float32; expect ~1e-3 relative error on finite voxels.
    finite = np.isfinite(vmin_np) & np.isfinite(vmin_wp)
    assert finite.any(), "No finite voxels — check supercell / cutoff setup"
    np.testing.assert_allclose(
        vmin_wp[finite], vmin_np[finite],
        rtol=1e-3, atol=1e-3,
        err_msg="Warp LJ Vext deviates from NumPy reference by more than float32 tolerance",
    )


def test_build_vext_warp_dtype_float32():
    """dtype=np.float32 should reduce memory without affecting Warp path correctness."""
    host, fluid, potential = _make_synthetic_host_and_fluid()
    orientations = np.eye(3)[None]

    from porecdft.vext.builder import build_vext_on_grid

    r32 = build_vext_on_grid(
        host, fluid, potential, orientations,
        spacing=1.5, pbc_supercell=(3, 3, 3),
        use_warp=True, warp_device="cpu", dtype=np.float32,
    )
    r64 = build_vext_on_grid(
        host, fluid, potential, orientations,
        spacing=1.5, pbc_supercell=(3, 3, 3),
        use_warp=True, warp_device="cpu", dtype=np.float64,
    )
    assert r32["orient_min"].dtype == np.float32
    assert r64["orient_min"].dtype == np.float64
    finite = np.isfinite(r32["orient_min"]) & np.isfinite(r64["orient_min"])
    np.testing.assert_allclose(
        r32["orient_min"][finite], r64["orient_min"][finite],
        rtol=1e-3, atol=1e-3,
    )

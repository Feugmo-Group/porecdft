"""Quasi-uniform sampling of orientations on the sphere.

For linear molecules (CO₂, N₂) we only need the direction of the molecular axis
in the lab frame — one direction per orientation. A Fibonacci sphere gives a
near-uniform set of N points without numerical artifacts of latitude-longitude
grids.

`fibonacci_rotations(N)` builds the SO(3) rotation that takes body +z to each
of the N directions; an arbitrary in-plane angle around z is then irrelevant
(linear molecule symmetry). For non-linear molecules we would need a full SO(3)
sampler (Hopf fibration or quaternion grid) — placeholder for the future.
"""

from __future__ import annotations

import numpy as np


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n quasi-uniform unit vectors on the sphere as an (n, 3) array."""
    i = np.arange(n)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = 2.0 * np.pi * i / phi
    r_xy = np.sqrt(1.0 - z * z)
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    return np.stack([x, y, z], axis=-1)


def fibonacci_rotations(n: int) -> np.ndarray:
    """Return n rotation matrices (n, 3, 3) such that R @ ẑ = directions_i.

    Linear-molecule convention: only the polar direction matters; the rotation
    around the molecular axis is left at zero (any choice is fine).
    """
    dirs = fibonacci_sphere(n)
    z_body = np.array([0.0, 0.0, 1.0])
    rots = np.empty((n, 3, 3))
    for k, u in enumerate(dirs):
        rots[k] = _rotation_to_align_z_with(u)
    return rots


def _rotation_to_align_z_with(u: np.ndarray) -> np.ndarray:
    """Build a rotation matrix that maps body +z to the unit vector `u`."""
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, u)
    s = np.linalg.norm(v)
    c = float(z @ u)
    if s < 1e-12:
        # u is parallel (c=+1) or antiparallel (c=-1) to z
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

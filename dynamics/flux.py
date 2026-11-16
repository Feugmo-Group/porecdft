"""Scharfetter-Gummel finite-volume operator for the DDFT flux law.

The dynamic density functional theory flux of species k is

    J = -D(r) [ grad rho + rho grad v ],        v = beta V_ext - c1[rho]

and the face flux between neighbouring voxels i -> j (spacing h) is

    F_ij = (D_ij / h) [ B(dv) rho_i - B(-dv) rho_j ],   dv = v_j - v_i,   B(x) = x / (exp(x) - 1).

For frozen v this is linear in rho, so d rho / dt = A rho with a sparse A that

  * has zero column sums (the number of molecules is conserved exactly),
  * has non-negative off-diagonals (an M-matrix: backward Euler preserves positivity),
  * has the exact stationary solution rho_j / rho_i = exp(-dv), which is the discrete form of the
    Euler-Lagrange equation  ln rho - c1 + beta V = const  solved by the static porecdft solvers.

ln(rho) never enters, so near-empty voxels and deep wells (hundreds of kT) are harmless.
Units: lengths in Angstrom, D in Angstrom^2 / ps, v dimensionless.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def bernoulli(x):
    """B(x) = x / (e^x - 1), with B(0) = 1, finite for |x| -> infinity."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-6
    pos = x > 0
    xp = np.where(pos & ~small, x, 1.0)
    xn = np.where(~pos & ~small, x, -1.0)
    # x > 0: x e^-x / (1 - e^-x) avoids overflow; x < 0: x / expm1(x)
    big = np.where(pos, xp * np.exp(-xp) / (-np.expm1(-xp)), xn / np.expm1(xn))
    return np.where(small, 1.0 - x / 2.0 + x**2 / 12.0, big)


def _face_diffusivity(D, shape, axis, I, J):
    """Diffusion coefficient on the faces (I, J) along `axis`: scalar, per-axis, or voxel field (harmonic mean)."""
    D = np.asarray(D, dtype=float)
    if D.ndim == 0:
        return np.full(I.shape, float(D))
    if D.shape == (3,):
        return np.full(I.shape, float(D[axis]))
    if D.shape == tuple(shape):
        d = D.ravel()
        di, dj = d[I], d[J]
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where((di > 0) & (dj > 0), 2.0 * di * dj / (di + dj), 0.0)
        return h
    raise ValueError(f"D must be a scalar, shape (3,), or shape {tuple(shape)}; got {D.shape}")


def face_weights(shape, D, spacing, periodic=(True, True, True), access=None):
    """Per-axis face weights w_ax = D_face / h^2 (grid-shaped arrays, or None for an axis without faces).

    w_ax[i] is the weight of the face between voxel i and its +1 neighbour along axis `ax`; it is zero on the wrap
    face of a non-periodic axis and on faces touching an inaccessible voxel. Shared by the sparse operator and the
    matrix-free JAX divergence so that both discretise the same equation.
    """
    shape = tuple(shape)
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    acc = None if access is None else np.asarray(access, dtype=bool)
    out = []
    for ax in range(3):
        N = shape[ax]
        if N < 2 or (periodic[ax] and N < 3):
            out.append(None)
            continue
        I = idx.ravel()
        J = np.roll(idx, -1, axis=ax).ravel()
        w = (_face_diffusivity(D, shape, ax, I, J) / spacing[ax] ** 2).reshape(shape)
        if not periodic[ax]:
            sel = [slice(None)] * 3
            sel[ax] = N - 1
            w[tuple(sel)] = 0.0
        if acc is not None:
            w = np.where(acc & np.roll(acc, -1, axis=ax), w, 0.0)
        out.append(w)
    return out


def build_operator(v, D, spacing, periodic=(True, True, True), access=None):
    """Sparse matrix A (n x n, n = v.size) with d rho / dt = A rho for a frozen field v.

    Parameters
    ----------
    v        : (Nx, Ny, Nz) dimensionless field beta V_ext - c1.
    D        : mobility in A^2/ps (scalar, (3,) per axis, or a voxel field).
    spacing  : (hx, hy, hz) in A.
    periodic : per-axis; a non-periodic axis has no flux through its outer faces (walls).
    access   : optional bool mask; faces touching an inaccessible voxel carry no flux.
    """
    v = np.asarray(v, dtype=float)
    shape = v.shape
    n = v.size
    idx = np.arange(n).reshape(shape)
    vf = v.ravel()
    rows, cols, vals = [], [], []
    for ax, wax in enumerate(face_weights(shape, D, spacing, periodic, access)):
        if wax is None:
            continue
        keep = (wax > 0).ravel()
        I = idx.ravel()[keep]
        J = np.roll(idx, -1, axis=ax).ravel()[keep]
        w = wax.ravel()[keep]
        dv = vf[J] - vf[I]
        a = w * bernoulli(dv)      # coefficient of rho_I in the flux I -> J
        b = w * bernoulli(-dv)     # coefficient of rho_J
        rows += [I, I, J, J]
        cols += [I, J, I, J]
        vals += [-a, b, a, -b]
    if not rows:
        return sp.csr_matrix((n, n))
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return A.tocsr()


def bernoulli_jax(x):
    """JAX version of ``bernoulli`` (safe under jit and autodiff: no NaN in the unselected branch)."""
    import jax.numpy as jnp

    small = jnp.abs(x) < 1e-6
    pos = x > 0
    xp = jnp.where(pos & ~small, x, 1.0)
    xn = jnp.where(~pos & ~small, x, -1.0)
    big = jnp.where(pos, xp * jnp.exp(-xp) / (-jnp.expm1(-xp)), xn / jnp.expm1(xn))
    return jnp.where(small, 1.0 - x / 2.0 + x**2 / 12.0, big)


def flux_divergence_jax(rho, v, weights):
    """Matrix-free d rho / dt for a (grid-shaped) rho and v, from the weights of ``face_weights`` (jit/grad safe)."""
    import jax.numpy as jnp

    out = jnp.zeros_like(rho)
    for ax, w in enumerate(weights):
        if w is None:
            continue
        dv = jnp.roll(v, -1, axis=ax) - v
        F = w * (bernoulli_jax(dv) * rho - bernoulli_jax(-dv) * jnp.roll(rho, -1, axis=ax))
        out = out - F + jnp.roll(F, 1, axis=ax)
    return out

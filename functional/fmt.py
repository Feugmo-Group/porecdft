"""Fundamental Measure Theory functional (aWBII default).

The FMT class provides the hard-sphere excess free energy and the first
functional derivative ``c¹_HS(r) = −β δF_HS/δρ(r)``, computed via FFT
convolutions on a regular real-space grid:

    n_α(r) = ∫ ρ(r') w_α(r − r') dr'    ⇔    n_α_hat = ρ_hat · w_α_hat
    Φ_HS(r) = n0·φ1 + φ2·(n1·n2 − n1vec·n2vec) + φ3·{tensor cubic in n2/n2vec}
    F_HS = k_B T ∫ Φ_HS(r) dV

The cDFT self-consistent equation in the rigid-host case:

    ρ(r) = ρ_bulk · exp[ −β V_ext(r) + c¹_HS(r) − c¹_HS(ρ_bulk) ]

iterated to fixed point via Picard mixing.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import jit

from porecdft.functional.fmt_weights import (
    w2FT, w3FT,
    phi1func, dphi1dnfunc,
    phi2func, dphi2dnfunc,
    phi3func, dphi3dnfunc,
)


@dataclass
class FMTWeightsHat:
    """Bundle of pre-computed FMT weight functions in Fourier space.

    Returned by :func:`make_fmt_weights_hat`.  Pass as a single object to
    :func:`compute_weighted_densities` and :func:`compute_c1` instead of
    the three separate ``w2_hat``, ``w3_hat``, ``w2vec_hat`` arrays.

    Supports tuple unpacking for backward compatibility::

        weights = make_fmt_weights_hat(K, KX, KY, KZ, sigma)
        w2_hat, w3_hat, w2vec_hat = weights   # old callers still work
    """
    w2_hat: jnp.ndarray
    w3_hat: jnp.ndarray
    w2vec_hat: jnp.ndarray   # shape (3, *K)

    def __iter__(self):
        yield self.w2_hat
        yield self.w3_hat
        yield self.w2vec_hat


@dataclass
class WeightedDensities:
    """Container for the six FMT weighted densities at every grid point."""
    n0: jnp.ndarray
    n1: jnp.ndarray
    n2: jnp.ndarray
    n3: jnp.ndarray            # packing fraction η
    n1vec: jnp.ndarray         # (3, *grid)
    n2vec: jnp.ndarray         # (3, *grid)


def make_k_grid(shape: tuple[int, int, int], dx: float, dy: float, dz: float,
                real_fft: bool = False,
                skew_angles: tuple[float, float, float] | None = None):
    """Return (kx, ky, kz, k) on the FFT k-space grid matching real-space shape.

    Parameters
    ----------
    real_fft : bool
        If True, last axis uses ``rfftfreq`` (length N//2+1).  Pair with
        ``compute_weighted_densities(..., real_fft=True, shape=...)`` and
        ``rfftn`` / ``irfftn`` to halve memory + ~2× FFT speedup (Stierle &
        Gross 2024).
    skew_angles : (α, β, γ), optional
        Cell angles in radians.  Default ``None`` → orthorhombic.  When set,
        applies the Stierle 2024 ``_skewed2cart`` transformation so the same
        FFT machinery handles MIL-53 monoclinic, hexagonal COFs, etc.  Note:
        non-orthorhombic cells also need a Jacobian correction on ``dV``
        (``dV *= sin γ · √(1 − cos²β − ζ²)``).
    """
    kx = 2 * jnp.pi * jnp.fft.fftfreq(shape[0], d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(shape[1], d=dy)
    if real_fft:
        kz = 2 * jnp.pi * jnp.fft.rfftfreq(shape[2], d=dz)
    else:
        kz = 2 * jnp.pi * jnp.fft.fftfreq(shape[2], d=dz)
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    if skew_angles is not None:
        a, b, g = (float(x) for x in skew_angles)
        # Stierle & Gross 2024, _skewed2cart (Grid.py)
        sin_g = jnp.sin(g)
        cos_g = jnp.cos(g)
        cos_b = jnp.cos(b)
        cos_a = jnp.cos(a)
        zeta = (cos_a - cos_b * cos_g) / sin_g
        det = sin_g * jnp.sqrt(jnp.maximum(1.0 - cos_b**2 - zeta**2, 1e-16))
        KY_c = (KY - KX * cos_g) / sin_g
        KZ_c = (KX * (zeta * cos_g - sin_g * cos_b) - KY * zeta + KZ * sin_g) / det
        KX, KY, KZ = KX, KY_c, KZ_c
    K = jnp.sqrt(KX**2 + KY**2 + KZ**2)
    return KX, KY, KZ, K


def lanczos_filter(KX, KY, KZ, dx: float, dy: float, dz: float) -> jnp.ndarray:
    """Lanczos (sinc) anti-aliasing filter matching the legacy DFT code.

    sigma_L(k) = sinc(kx·dx/2π) · sinc(ky·dy/2π) · sinc(kz·dz/2π)

    At k=0: sigma_L=1; at Nyquist: sigma_L≈0.637 per axis.  Applied to all
    weight functions to suppress grid-discretisation artefacts at high density.
    """
    return (jnp.sinc(KX * dx / (2 * jnp.pi))
            * jnp.sinc(KY * dy / (2 * jnp.pi))
            * jnp.sinc(KZ * dz / (2 * jnp.pi)))


def make_fmt_weights_hat(K, KX, KY, KZ, sigma: float,
                         dx: float | None = None,
                         dy: float | None = None,
                         dz: float | None = None):
    """Build FMT scalar + vector weight functions in Fourier space.

    Returns an :class:`FMTWeightsHat` with fields ``w2_hat``, ``w3_hat``,
    ``w2vec_hat`` (shape ``(3, *K)``).  The object supports tuple unpacking
    so existing ``w2_hat, w3_hat, w2vec_hat = make_fmt_weights_hat(...)``
    callers continue to work unchanged.

    The vector weight is w2vec = (∇w3) — purely imaginary in Fourier space.
    If dx/dy/dz are provided, a Lanczos anti-aliasing filter is applied to all
    weight functions (matching the reference legacy implementation).
    """
    w2_hat = w2FT(K, sigma)
    w3_hat = w3FT(K, sigma)
    if dx is not None and dy is not None and dz is not None:
        sigma_L = lanczos_filter(KX, KY, KZ, dx, dy, dz)
        w2_hat  = w2_hat * sigma_L
        w3_hat  = w3_hat * sigma_L
    # w2vec(k) = i·k · w3(k) (gradient of volumetric weight; w3_hat already filtered)
    w2vec_hat = jnp.stack([1j * KX * w3_hat, 1j * KY * w3_hat, 1j * KZ * w3_hat], axis=0)
    return FMTWeightsHat(w2_hat=w2_hat, w3_hat=w3_hat, w2vec_hat=w2vec_hat)


def compute_weighted_densities(
    rho: jnp.ndarray,
    w2_hat, w3_hat, w2vec_hat,
    sigma: float,
    real_fft: bool = False,
) -> WeightedDensities:
    """Compute the six FMT weighted densities by FFT convolution.

    When ``real_fft=True``, uses ``rfftn`` / ``irfftn`` for ~2× speedup;
    requires that ``w2_hat`` / ``w3_hat`` / ``w2vec_hat`` were built on the
    matching rfft k-grid (see ``make_k_grid(..., real_fft=True)``).
    """
    if real_fft:
        rho_hat = jnp.fft.rfftn(rho)
        _inv = lambda H: jnp.fft.irfftn(H, s=rho.shape)
    else:
        rho_hat = jnp.fft.fftn(rho)
        _inv = lambda H: jnp.fft.ifftn(H).real
    n3 = _inv(rho_hat * w3_hat)
    n2 = _inv(rho_hat * w2_hat)
    n1 = n2 / (2 * jnp.pi * sigma)
    n0 = n2 / (jnp.pi * sigma**2)
    n2vec = jnp.stack([_inv(rho_hat * w2vec_hat[d]) for d in range(3)], axis=0)
    n1vec = n2vec / (2 * jnp.pi * sigma)
    # cap packing fraction to avoid log(0) in φ_i
    n3 = jnp.where(n3 >= 1.0, 1.0 - 1e-12, n3)
    return WeightedDensities(n0=n0, n1=n1, n2=n2, n3=n3, n1vec=n1vec, n2vec=n2vec)


def free_energy_density(wd: WeightedDensities, model: str = "aWBII") -> jnp.ndarray:
    """Hard-sphere free energy density Φ_HS(r) in units of k_B T."""
    n3 = wd.n3
    phi1 = phi1func(n3)
    phi2 = phi2func(n3, model=model)
    phi3 = phi3func(n3, model=model)
    n2v2 = jnp.sum(wd.n2vec * wd.n2vec, axis=0)
    n1v_n2v = jnp.sum(wd.n1vec * wd.n2vec, axis=0)
    if "a" in model:
        xi = jnp.where(wd.n2 < 1e-12, 0.0, n2v2 / (wd.n2 + 1e-16) ** 2)
        xi = jnp.where(xi >= 1.0, 1.0 - 1e-12, xi)
        phi_HS = (
            wd.n0 * phi1
            + phi2 * (wd.n1 * wd.n2 - n1v_n2v)
            + phi3 * wd.n2**3 * (1 - xi) ** 3
        )
    else:
        phi_HS = (
            wd.n0 * phi1
            + phi2 * (wd.n1 * wd.n2 - n1v_n2v)
            + phi3 * (wd.n2**3 - 3 * wd.n2 * n2v2)
        )
    return phi_HS


def compute_c1(
    rho: jnp.ndarray,
    wd: WeightedDensities,
    w2_hat, w3_hat, w2vec_hat,
    sigma: float,
    model: str = "aWBII",
    real_fft: bool = False,
) -> jnp.ndarray:
    """First functional derivative c¹_HS(r) = −β δF_HS/δρ(r).

    Computed in Fourier space:
      c¹_hat = − Σ_α  FT(∂Φ_HS/∂n_α) · w_α_hat   (with appropriate signs for vector terms).
    """
    phi1 = phi1func(wd.n3)
    dphi1dn3 = dphi1dnfunc(wd.n3)
    phi2 = phi2func(wd.n3, model=model)
    dphi2dn3 = dphi2dnfunc(wd.n3, model=model)
    phi3 = phi3func(wd.n3, model=model)
    dphi3dn3 = dphi3dnfunc(wd.n3, model=model)

    n2v2 = jnp.sum(wd.n2vec * wd.n2vec, axis=0)
    n1v_n2v = jnp.sum(wd.n1vec * wd.n2vec, axis=0)

    if real_fft:
        _fwd = jnp.fft.rfftn
        _inv = lambda H: jnp.fft.irfftn(H, s=rho.shape)
    else:
        _fwd = jnp.fft.fftn
        _inv = lambda H: jnp.fft.ifftn(H).real

    # ∂Φ/∂n0 = φ1
    c1_hat = -_fwd(phi1) / (jnp.pi * sigma**2) * w2_hat
    # ∂Φ/∂n1 = φ2 · n2
    c1_hat -= _fwd(wd.n2 * phi2) / (2 * jnp.pi * sigma) * w2_hat

    if "a" in model:
        xi = jnp.where(wd.n2 < 1e-12, 0.0, n2v2 / (wd.n2 + 1e-16) ** 2)
        xi = jnp.where(xi >= 1.0, 1.0 - 1e-12, xi)
        # ∂Φ/∂n2 (asymmetric)
        dphi_dn2 = wd.n1 * phi2 + 3 * wd.n2**2 * (1 + xi) * (1 - xi) ** 2 * phi3
        c1_hat -= _fwd(dphi_dn2) * w2_hat
        # ∂Φ/∂n3
        dphi_dn3 = (
            wd.n0 * dphi1dn3
            + (wd.n1 * wd.n2 - n1v_n2v) * dphi2dn3
            + wd.n2**3 * (1 - xi) ** 3 * dphi3dn3
        )
        c1_hat -= _fwd(dphi_dn3) * w3_hat
        # ∂Φ/∂n1vec
        for d in range(3):
            c1_hat += _fwd(-wd.n2vec[d] * phi2) / (2 * jnp.pi * sigma) * w2vec_hat[d]
        # ∂Φ/∂n2vec (asymmetric form)
        for d in range(3):
            dphi_dn2vec_d = -wd.n1vec[d] * phi2 - 6 * wd.n2 * wd.n2vec[d] * (1 - xi) ** 2 * phi3
            c1_hat += _fwd(dphi_dn2vec_d) * w2vec_hat[d]
    else:
        dphi_dn2 = wd.n1 * phi2 + 3 * (wd.n2**2 - n2v2) * phi3
        c1_hat -= _fwd(dphi_dn2) * w2_hat
        dphi_dn3 = (
            wd.n0 * dphi1dn3
            + (wd.n1 * wd.n2 - n1v_n2v) * dphi2dn3
            + (wd.n2**3 - 3 * wd.n2 * n2v2) * dphi3dn3
        )
        c1_hat -= _fwd(dphi_dn3) * w3_hat
        for d in range(3):
            c1_hat += _fwd(-wd.n2vec[d] * phi2) / (2 * jnp.pi * sigma) * w2vec_hat[d]
        for d in range(3):
            dphi_dn2vec_d = -wd.n1vec[d] * phi2 - 6 * wd.n2 * wd.n2vec[d] * phi3
            c1_hat += _fwd(dphi_dn2vec_d) * w2vec_hat[d]

    return _inv(c1_hat)


@dataclass
class FMTFunctional:
    """Convenience wrapper around the aWBII FMT functions.

    Parameters
    ----------
    sigma_HS : float
        Hard-sphere diameter in Å.
    model : str
        Functional variant — ``"aWBII"`` (default) or ``"WBII"``.

    Usage
    -----
    >>> fmt = FMTFunctional(sigma_HS=3.3)
    >>> c1_field = fmt.c1(rho_grid, w2_hat, w3_hat, w2vec_hat)
    >>> c1_ref   = fmt.c1_bulk(rho_bulk)
    """
    sigma_HS: float
    model: str = "aWBII"

    def c1(
        self,
        rho: jnp.ndarray,
        w2_hat,
        w3_hat,
        w2vec_hat,
    ) -> jnp.ndarray:
        """c¹_HS(r) on the grid. Pre-compute weights with make_fmt_weights_hat."""
        wd = compute_weighted_densities(rho, w2_hat, w3_hat, w2vec_hat, self.sigma_HS)
        return compute_c1(rho, wd, w2_hat, w3_hat, w2vec_hat, self.sigma_HS, self.model)

    def c1_bulk(self, rho_bulk: float) -> float:
        """c¹_HS for a uniform fluid at ρ_bulk — use as the reference offset."""
        return bulk_c1(rho_bulk, self.sigma_HS, self.model)


def make_fmt_weights_hat_multi(K, KX, KY, KZ, sigmas, dx=None, dy=None, dz=None):
    """Per-component weight stacks for multi-component FMT.

    Returns ``(w2_hat[c], w3_hat[c], w2vec_hat[c, d])`` with leading
    component axis ``c``.  Hard-sphere additive mixture rule (Rosenfeld):
    each species contributes its own σ_c-weighted FMT kernels.
    """
    sig = jnp.asarray(sigmas, dtype=K.dtype)
    w2s, w3s, w2vs = [], [], []
    for s in sigmas:
        w2, w3, w2v = make_fmt_weights_hat(K, KX, KY, KZ, float(s), dx, dy, dz)
        w2s.append(w2); w3s.append(w3); w2vs.append(w2v)
    return (jnp.stack(w2s, axis=0),
            jnp.stack(w3s, axis=0),
            jnp.stack(w2vs, axis=0))   # (C, 3, *K)


def compute_weighted_densities_multi(
    rho_c: jnp.ndarray,           # (C, Nx, Ny, Nz)
    w2_hat_c, w3_hat_c, w2vec_hat_c,
    sigmas,
    real_fft: bool = False,
) -> WeightedDensities:
    """Multi-component FMT weighted densities.

    Sums Rosenfeld additive-mixture contributions:
      n_α(r) = Σ_c ∫ ρ_c(r') w_α^{(c)}(r−r') dr'

    Parameters
    ----------
    rho_c : (C, Nx, Ny, Nz)
    w2_hat_c, w3_hat_c : (C, *kshape)
    w2vec_hat_c : (C, 3, *kshape)
    sigmas : sequence of σ_c (Å)
    """
    C = rho_c.shape[0]
    rshape = rho_c.shape[1:]
    if real_fft:
        fwd = lambda x: jnp.fft.rfftn(x, axes=(-3, -2, -1))
        inv = lambda H: jnp.fft.irfftn(H, s=rshape, axes=(-3, -2, -1))
    else:
        fwd = lambda x: jnp.fft.fftn(x, axes=(-3, -2, -1))
        inv = lambda H: jnp.fft.ifftn(H, axes=(-3, -2, -1)).real
    rho_hat_c = fwd(rho_c)                            # (C, *kshape)
    # broadcast multiply then sum over component axis
    n3 = inv(jnp.sum(rho_hat_c * w3_hat_c, axis=0))
    n2 = jnp.zeros_like(n3)
    n1 = jnp.zeros_like(n3)
    n0 = jnp.zeros_like(n3)
    n2vec = jnp.zeros((3,) + rshape, dtype=n3.dtype)
    n1vec = jnp.zeros_like(n2vec)
    for c in range(C):
        sig_c = float(sigmas[c])
        n2_c = inv(rho_hat_c[c] * w2_hat_c[c])
        n2 = n2 + n2_c
        n1 = n1 + n2_c / (2 * jnp.pi * sig_c)
        n0 = n0 + n2_c / (jnp.pi * sig_c ** 2)
        n2v_c = jnp.stack([inv(rho_hat_c[c] * w2vec_hat_c[c, d]) for d in range(3)], axis=0)
        n2vec = n2vec + n2v_c
        n1vec = n1vec + n2v_c / (2 * jnp.pi * sig_c)
    n3 = jnp.where(n3 >= 1.0, 1.0 - 1e-12, n3)
    return WeightedDensities(n0=n0, n1=n1, n2=n2, n3=n3, n1vec=n1vec, n2vec=n2vec)


def compute_c1_multi(
    rho_c: jnp.ndarray,
    wd: WeightedDensities,
    w2_hat_c, w3_hat_c, w2vec_hat_c,
    sigmas,
    model: str = "aWBII",
    real_fft: bool = False,
) -> jnp.ndarray:
    """Per-component c¹_HS_c(r) for the additive mixture.

    Returns ``(C, Nx, Ny, Nz)``.  Reuses :func:`compute_c1` per species —
    each component sees the *shared* weighted densities ``wd`` (which were
    summed over species) but is convolved with its own σ_c weights.
    """
    C = rho_c.shape[0]
    c1_list = []
    for c in range(C):
        c1_c = compute_c1(rho_c[c], wd,
                          w2_hat_c[c], w3_hat_c[c], w2vec_hat_c[c],
                          float(sigmas[c]), model=model, real_fft=real_fft)
        c1_list.append(c1_c)
    return jnp.stack(c1_list, axis=0)


def bulk_c1(rho_bulk: float, sigma: float, model: str = "aWBII") -> float:
    """c¹_HS for a uniform bulk fluid at density ρ_bulk — needed as a reference."""
    n0 = rho_bulk
    n1 = rho_bulk * sigma / 2
    n2 = rho_bulk * jnp.pi * sigma**2
    n3 = rho_bulk * jnp.pi * sigma**3 / 6
    if n3 >= 1.0:
        n3 = jnp.array(1.0 - 1e-12)
    # vector weighted densities vanish for uniform bulk
    phi1 = phi1func(n3)
    phi2 = phi2func(n3, model=model)
    phi3 = phi3func(n3, model=model)
    dphi1dn3 = dphi1dnfunc(n3)
    dphi2dn3 = dphi2dnfunc(n3, model=model)
    dphi3dn3 = dphi3dnfunc(n3, model=model)
    # c1_bulk = − (∂Φ/∂n0 · w2_vol + ∂Φ/∂n1 · w2_vol/(2πσ) + ∂Φ/∂n2 · πσ² + ∂Φ/∂n3 · πσ³/6)
    # For uniform bulk, IFFT of (df * w_hat) at k=0 equals (df) · w_α^volume.
    dphi_dn0 = phi1
    dphi_dn1 = n2 * phi2
    dphi_dn2 = n1 * phi2 + 3 * n2**2 * phi3       # symmetric variant (xi = 0 in bulk)
    dphi_dn3 = n0 * dphi1dn3 + n1 * n2 * dphi2dn3 + n2**3 * dphi3dn3
    w2_vol = jnp.pi * sigma**2
    w3_vol = jnp.pi * sigma**3 / 6
    c1 = -(dphi_dn0 * (w2_vol / (jnp.pi * sigma**2))   # = dphi_dn0 · 1
           + dphi_dn1 * (w2_vol / (2 * jnp.pi * sigma))
           + dphi_dn2 * w2_vol
           + dphi_dn3 * w3_vol)
    return float(c1)

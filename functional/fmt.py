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
class WeightedDensities:
    """Container for the six FMT weighted densities at every grid point."""
    n0: jnp.ndarray
    n1: jnp.ndarray
    n2: jnp.ndarray
    n3: jnp.ndarray            # packing fraction η
    n1vec: jnp.ndarray         # (3, *grid)
    n2vec: jnp.ndarray         # (3, *grid)


def make_k_grid(shape: tuple[int, int, int], dx: float, dy: float, dz: float):
    """Return (kx, ky, kz, k) on the FFT k-space grid matching real-space shape."""
    kx = 2 * jnp.pi * jnp.fft.fftfreq(shape[0], d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(shape[1], d=dy)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(shape[2], d=dz)
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
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

    Returns ``(w2_hat, w3_hat, w2vec_hat)`` where w2vec_hat has shape (3, *K).
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
    return w2_hat, w3_hat, w2vec_hat


def compute_weighted_densities(
    rho: jnp.ndarray,
    w2_hat, w3_hat, w2vec_hat,
    sigma: float,
) -> WeightedDensities:
    """Compute the six FMT weighted densities by FFT convolution."""
    rho_hat = jnp.fft.fftn(rho)
    n3 = jnp.fft.ifftn(rho_hat * w3_hat).real
    n2 = jnp.fft.ifftn(rho_hat * w2_hat).real
    n1 = n2 / (2 * jnp.pi * sigma)
    n0 = n2 / (jnp.pi * sigma**2)
    n2vec = jnp.stack([
        jnp.fft.ifftn(rho_hat * w2vec_hat[d]).real for d in range(3)
    ], axis=0)
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

    # ∂Φ/∂n0 = φ1
    c1_hat = -jnp.fft.fftn(phi1) / (jnp.pi * sigma**2) * w2_hat
    # ∂Φ/∂n1 = φ2 · n2
    c1_hat -= jnp.fft.fftn(wd.n2 * phi2) / (2 * jnp.pi * sigma) * w2_hat

    if "a" in model:
        xi = jnp.where(wd.n2 < 1e-12, 0.0, n2v2 / (wd.n2 + 1e-16) ** 2)
        xi = jnp.where(xi >= 1.0, 1.0 - 1e-12, xi)
        # ∂Φ/∂n2 (asymmetric)
        dphi_dn2 = wd.n1 * phi2 + 3 * wd.n2**2 * (1 + xi) * (1 - xi) ** 2 * phi3
        c1_hat -= jnp.fft.fftn(dphi_dn2) * w2_hat
        # ∂Φ/∂n3
        dphi_dn3 = (
            wd.n0 * dphi1dn3
            + (wd.n1 * wd.n2 - n1v_n2v) * dphi2dn3
            + wd.n2**3 * (1 - xi) ** 3 * dphi3dn3
        )
        c1_hat -= jnp.fft.fftn(dphi_dn3) * w3_hat
        # ∂Φ/∂n1vec
        for d in range(3):
            c1_hat += jnp.fft.fftn(-wd.n2vec[d] * phi2) / (2 * jnp.pi * sigma) * w2vec_hat[d]
        # ∂Φ/∂n2vec (asymmetric form)
        for d in range(3):
            dphi_dn2vec_d = -wd.n1vec[d] * phi2 - 6 * wd.n2 * wd.n2vec[d] * (1 - xi) ** 2 * phi3
            c1_hat += jnp.fft.fftn(dphi_dn2vec_d) * w2vec_hat[d]
    else:
        dphi_dn2 = wd.n1 * phi2 + 3 * (wd.n2**2 - n2v2) * phi3
        c1_hat -= jnp.fft.fftn(dphi_dn2) * w2_hat
        dphi_dn3 = (
            wd.n0 * dphi1dn3
            + (wd.n1 * wd.n2 - n1v_n2v) * dphi2dn3
            + (wd.n2**3 - 3 * wd.n2 * n2v2) * dphi3dn3
        )
        c1_hat -= jnp.fft.fftn(dphi_dn3) * w3_hat
        for d in range(3):
            c1_hat += jnp.fft.fftn(-wd.n2vec[d] * phi2) / (2 * jnp.pi * sigma) * w2vec_hat[d]
        for d in range(3):
            dphi_dn2vec_d = -wd.n1vec[d] * phi2 - 6 * wd.n2 * wd.n2vec[d] * phi3
            c1_hat += jnp.fft.fftn(dphi_dn2vec_d) * w2vec_hat[d]

    return jnp.fft.ifftn(c1_hat).real


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

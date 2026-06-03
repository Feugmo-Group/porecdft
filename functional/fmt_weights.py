"""FMT (Fundamental Measure Theory) weight functions and φ functions.

Migrated from the legacy `cdft/_aux.py` (Elvis do A. Soares 2022-2024). Five
functionals are supported via the ``model`` argument:

- ``"RF"``   — Rosenfeld 1989 (original FMT)
- ``"WBI"``  — White Bear I (Roth, Mecke, Oettel 2002)
- ``"WBII"`` — White Bear II (Hansen-Goos, Roth 2006)
- ``"aRF"``, ``"aWBI"``, ``"aWBII"`` — asymmetric variants with improved tensor
  weighting in tight confinement (the "a" prefix means asymmetric)

For ALF/CO2 we default to **aWBII** — most accurate in confinement for the
hard-sphere diameter d ≈ σ_CO2 ≈ 3.0 Å against framework pore walls.

All functions JAX-jit'd and ready for autograd / vmap.
"""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import jit


# =============================================================================
# Weight functions in Fourier space — used to convolve ρ(k) into weighted
# densities n_α(r) via IFFT(rho_hat × w_α_hat).
# =============================================================================

@jit
def w3FT(k: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    """Volumetric weight w3 in Fourier space (sphere of diameter σ)."""
    ks = k * 0.5 * sigma
    return (jnp.pi * sigma**3 / 6) * jnp.where(
        k * sigma < 1e-6,
        1.0 - ks**2 / 10,
        3 * (jnp.sin(ks) - ks * jnp.cos(ks)) / ks**3,
    )


@jit
def w2FT(k: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    """Surface weight w2 in Fourier space (sphere surface of diameter σ)."""
    return jnp.pi * sigma**2 * jnp.sinc(0.5 * sigma * k / jnp.pi)


@jit
def translationFT(kx, ky, kz, a) -> jnp.ndarray:
    """Phase factor for translating an FT by Cartesian displacement a."""
    return jnp.exp(1.0j * (kx * a[0] + ky * a[1] + kz * a[2]))


# =============================================================================
# Auxiliary φ_i(η) functions and their derivatives. η = n3 = packing fraction.
# Free energy density is:
#   φ_HS = n0·φ1 + φ2·(n1·n2 − n1vec·n2vec) + φ3·{n2³ − 3·n2·|n2vec|²}
#                                     (or × (1−ξ)³ for asymmetric variants)
# =============================================================================

@jit
def phi1func(eta):
    """φ1(η) = -ln(1-η)."""
    return -jnp.log(1 - eta)


@jit
def dphi1dnfunc(eta):
    """dφ1/dη = 1/(1-η)."""
    return 1 / (1 - eta)


@partial(jit, static_argnames=["model"])
def phi2func(eta, model: str = "WBI"):
    if model in ("RF", "WBI", "aRF", "aWBI"):
        return 1 / (1 - eta)
    if model in ("WBII", "aWBII"):
        return jnp.where(
            eta <= 1e-8,
            (1 + eta**2 / 9) / (1 - eta),
            ((5 - eta) * eta + 2 * (1 - eta) * jnp.log(1 - eta)) / (3 * eta * (1 - eta)),
        )
    return 1 / (1 - eta)


@partial(jit, static_argnames=["model"])
def dphi2dnfunc(eta, model: str = "WBI"):
    if model in ("RF", "WBI", "aRF", "aWBI"):
        return 1 / (1 - eta) ** 2
    if model in ("WBII", "aWBII"):
        return jnp.where(
            eta <= 1e-8,
            (1 + 2 * eta / 9 + eta**2 / 18) / (1 - eta) ** 2,
            -2 * (eta - 3 * eta**2 + (1 - eta) ** 2 * jnp.log(1 - eta))
            / (3 * eta**2 * (1 - eta) ** 2),
        )
    return 1 / (1 - eta) ** 2


@partial(jit, static_argnames=["model"])
def phi3func(eta, model: str = "WBI"):
    if model in ("RF", "aRF"):
        return 1 / (24 * jnp.pi * (1 - eta) ** 2)
    if model in ("WBI", "aWBI"):
        return jnp.where(
            eta <= 1e-8,
            (1.0 - 2 * eta / 9 - eta**2 / 18) / (24 * jnp.pi * (1 - eta) ** 2),
            (eta + (1 - eta) ** 2 * jnp.log(1 - eta)) / (36 * jnp.pi * eta**2 * (1 - eta) ** 2),
        )
    if model in ("WBII", "aWBII"):
        return jnp.where(
            eta <= 1e-8,
            (1 - 4 * eta / 9 + eta**2 / 18) / (24 * jnp.pi * (1 - eta) ** 2),
            -2 * (eta + (eta - 3) * eta**2 + jnp.log(1 - eta) * (1 - eta) ** 2)
            / ((3 * eta**2) * 24 * jnp.pi * (1 - eta) ** 2),
        )
    return jnp.where(
        eta <= 1e-8,
        (1.0 - 2 * eta / 9 - eta**2 / 18) / (24 * jnp.pi * (1 - eta) ** 2),
        (eta + (1 - eta) ** 2 * jnp.log(1 - eta)) / (36 * jnp.pi * eta**2 * (1 - eta) ** 2),
    )


@partial(jit, static_argnames=["model"])
def dphi3dnfunc(eta, model: str = "WBI"):
    if model in ("RF", "aRF"):
        return 1 / (12 * jnp.pi * (1 - eta) ** 3)
    if model in ("WBI", "aWBI"):
        return jnp.where(
            eta <= 1e-8,
            (8 / 3 - 0.5 * eta - 0.1 * eta**2) / (36 * jnp.pi * (1 - eta) ** 3),
            -(eta * (2 - 5 * eta + eta**2) + 2 * (1 - eta) ** 3 * jnp.log(1 - eta))
            / (36 * jnp.pi * eta**3 * (1 - eta) ** 3),
        )
    if model in ("WBII", "aWBII"):
        return jnp.where(
            eta <= 1e-8,
            (7 / 3 - eta / 2 + eta**2 / 10) / (36 * jnp.pi * (1 - eta) ** 3),
            (2 * eta - 5 * eta**2 + 6 * eta**3 - eta**4
             + 2 * (1 - eta) ** 3 * jnp.log(1 - eta))
            / (36 * jnp.pi * eta**3 * (1 - eta) ** 3),
        )
    return jnp.where(
        eta <= 1e-8,
        (8 / 3 - 0.5 * eta - 0.1 * eta**2) / (36 * jnp.pi * (1 - eta) ** 3),
        -(eta * (2 - 5 * eta + eta**2) + 2 * (1 - eta) ** 3 * jnp.log(1 - eta))
        / (36 * jnp.pi * eta**3 * (1 - eta) ** 3),
    )

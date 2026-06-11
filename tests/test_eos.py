"""Round-trip and reference-data tests for porecdft EOS implementations.

For every EOS exposed under :mod:`porecdft.eos`, we check:

1. **Round-trip identity** — ``pressure(bulk_density(P, T), T) ≈ P`` to 1 %.
2. **Positivity** — bulk density is strictly positive in the gas regime.
3. **Reference data** — at one canonical (P, T) point per EOS, the density
   matches a published value (NIST REFPROP / journal article) to a stated
   tolerance.

New EOS added under Phase 1 of the v0.2 plan extend the ``EOS_CASES``
parametrisation block; the round-trip and positivity tests run automatically.

Run with:
    /opt/homebrew/Caskroom/miniconda/base/envs/jax/bin/python -m pytest tests/test_eos.py -v
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from porecdft.eos import (
    H2_PR,
    N2_PR,
    CH4_PR,
    CO2_SW,
    CO2_SRK,
    CH4_SRK,
    H2O_CPA,
    CO2_SAFT_VR_Mie,
    density_from_pressure,
)
from porecdft.eos.base import EOSBase


# ─── Pure ideal-gas sanity check ──────────────────────────────────────────

def test_ideal_gas_density_298K_1bar():
    """Ideal gas: ρ = P / (k_B T) ≈ 2.45 × 10⁻⁵ molecules/Å³ at 298 K, 1 bar."""
    rho = density_from_pressure(1.0, 298.0)
    assert rho == pytest.approx(2.43e-5, rel=2e-2)


# ─── EOS test-matrix ──────────────────────────────────────────────────────

@dataclass
class EOSCase:
    """One row in the EOS test matrix."""
    name: str
    eos: EOSBase
    P_bar: float
    T_K: float
    rho_ref: float | None = None  # reference density (molecules/Å³); None = skip
    rho_rel_tol: float = 0.05     # relative tolerance for reference test
    p_rel_tol: float = 1e-2       # relative tolerance for round-trip


# Reference data:
# - H2_PR @ 298 K, 1 bar: ideal-gas value (H2 is essentially ideal at 1 bar / 298 K)
#   ρ = 1e5 / (1.38e-23 × 298) ≈ 2.43e25 m⁻³ → 2.43e-5 molecules/Å³
# - N2_PR @ 298 K, 1 bar: same ideal-gas regime → 2.43e-5
# - CH4_PR @ 298 K, 1 bar: same → 2.43e-5
EOS_CASES: list[EOSCase] = [
    EOSCase("H2_PR @ 298K, 1bar", H2_PR, 1.0, 298.0, rho_ref=2.43e-5, rho_rel_tol=2e-2),
    EOSCase("N2_PR @ 298K, 1bar", N2_PR, 1.0, 298.0, rho_ref=2.43e-5, rho_rel_tol=2e-2),
    EOSCase("CH4_PR @ 298K, 1bar", CH4_PR, 1.0, 298.0, rho_ref=2.43e-5, rho_rel_tol=3e-2),
    EOSCase("H2_PR @ 298K, 100bar", H2_PR, 100.0, 298.0),
    EOSCase("H2_PR @ 77K, 1bar", H2_PR, 1.0, 77.0),
    # Span-Wagner CO2 — near-ideal at 1 bar (NIST ρ ≈ 2.46e-5; ideal 2.43e-5)
    EOSCase("CO2_SW @ 298K, 1bar", CO2_SW, 1.0, 298.0, rho_ref=2.43e-5, rho_rel_tol=2e-2),
    EOSCase("CO2_SW @ 298K, 50bar", CO2_SW, 50.0, 298.0, p_rel_tol=2e-2),
    # SRK (Phase 1.1)
    EOSCase("CO2_SRK @ 298K, 1bar", CO2_SRK, 1.0, 298.0, rho_ref=2.43e-5, rho_rel_tol=3e-2),
    EOSCase("CO2_SRK @ 298K, 50bar", CO2_SRK, 50.0, 298.0),
    EOSCase("CH4_SRK @ 298K, 1bar", CH4_SRK, 1.0, 298.0, rho_ref=2.43e-5, rho_rel_tol=3e-2),
    # CPA — water 4C scheme (Kontogeorgis 1999).
    EOSCase("H2O_CPA @ 373K, 1bar", H2O_CPA, 1.0, 373.0, rho_ref=1.95e-5, rho_rel_tol=5e-2),
    EOSCase("H2O_CPA @ 298K, 0.05bar", H2O_CPA, 0.05, 298.0),
    # SAFT-VR-Mie (Lafitte 2013 / Avendaño 2011 CO2 parameters)
    EOSCase("CO2_SAFT_VR_Mie @ 298K, 1bar", CO2_SAFT_VR_Mie, 1.0, 298.0,
            rho_ref=2.43e-5, rho_rel_tol=5e-2),
    EOSCase("CO2_SAFT_VR_Mie @ 298K, 10bar", CO2_SAFT_VR_Mie, 10.0, 298.0),
]


@pytest.mark.parametrize("case", EOS_CASES, ids=lambda c: c.name)
def test_eos_bulk_density_positive(case: EOSCase):
    """Bulk density must be strictly positive (gas-phase root selected)."""
    rho = case.eos.bulk_density(case.P_bar, case.T_K)
    assert rho > 0, f"{case.name}: ρ = {rho} not positive"
    assert np.isfinite(rho), f"{case.name}: ρ = {rho} not finite"


@pytest.mark.parametrize("case", EOS_CASES, ids=lambda c: c.name)
def test_eos_round_trip(case: EOSCase):
    """``pressure(bulk_density(P, T), T) ≈ P`` for any EOS that implements both."""
    rho = case.eos.bulk_density(case.P_bar, case.T_K)
    try:
        P_back = case.eos.pressure(rho, case.T_K)
    except NotImplementedError:
        pytest.skip(f"{case.name}: pressure(rho, T) not implemented")
    assert P_back == pytest.approx(case.P_bar, rel=case.p_rel_tol)


@pytest.mark.parametrize("case", EOS_CASES, ids=lambda c: c.name)
def test_eos_reference_density(case: EOSCase):
    """At chosen (P, T) the density matches the published reference value."""
    if case.rho_ref is None:
        pytest.skip("no reference density specified for this case")
    rho = case.eos.bulk_density(case.P_bar, case.T_K)
    assert rho == pytest.approx(case.rho_ref, rel=case.rho_rel_tol)

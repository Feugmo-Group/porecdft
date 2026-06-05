"""Free-energy functionals.

Phase 2.2: aWBII FMT migrated.
Phase 2.4: Wertheim TPT-1 association (association.py) — host-fluid H-bond sites.
Future:
  lj_wda.py          — weighted-density approximation for LJ dispersion (MBWR-based)
  pcsaft_disp.py     — PC-SAFT dispersion c¹ on a grid
  mean_field.py      — simple MFA baseline for comparisons
"""

from porecdft.functional.association import (
    AssociationSite,
    WertheimAssociation,
)
from porecdft.functional.fmt import (
    FMTFunctional,
    WeightedDensities,
    make_k_grid,
    make_fmt_weights_hat,
    compute_weighted_densities,
    free_energy_density,
    compute_c1,
    bulk_c1,
)
from porecdft.functional import fmt_weights
from porecdft.functional.elastic import ElasticPenalty, scale_host

__all__ = [
    "AssociationSite",
    "WertheimAssociation",
    "FMTFunctional",
    "WeightedDensities",
    "make_k_grid",
    "make_fmt_weights_hat",
    "compute_weighted_densities",
    "free_energy_density",
    "compute_c1",
    "bulk_c1",
    "fmt_weights",
    "ElasticPenalty",
    "scale_host",
]

"""convoy: parallel motion as an exact counterfactual for feed-forward 3D reconstruction models.

A point that moves by ``s`` times the camera baseline, parallel to it, gives the same two images as
a static point at depth ``Z / (1 - s)``. The slope of a model's log-depth error on
``x = -log(1 - s)`` over such movers, the geometry share ``beta_s``, measures how much cross-view
geometry the model uses for moving objects (0: single-image prior, 1: triangulation).

This package holds the model-independent part of the study, ported from the notebook that produced
every result (``notebook/convoy_ambiguity.ipynb``) and tested against it:

* :mod:`convoy.geometry`   two- and multi-view geometry, the counterfactual and the static twin;
* :mod:`convoy.probe`      log-depth error, geometry share, slope-weight identity, per-object share;
* :mod:`convoy.masks`      dynamic-mask scores and their recall at a fixed static false-positive rate;
* :mod:`convoy.twin`       the static-twin test and the registered H14;
* :mod:`convoy.stats`      cluster bootstrap, paired contrasts, Holm, verdict rules;
* :mod:`convoy.registered` sample rules, thresholds and seeds fixed before the data;
* :mod:`convoy.paper`, :mod:`convoy.claims`  the tables and every number of the paper, from ``results/v6.6``.
"""
from .geometry import geometric_answer, static_twin, twin_contact_height
from .probe import (
    beta_to_effective_w,
    geometry_share,
    mixture_rho,
    motion_regime,
    mover_rho,
    per_object_share,
    prior_weight,
    scale_and_gate,
    slope_weight_table,
)
from .registered import GLOBAL_SEED, N_BOOT, PREREG, seed_for

__version__ = "1.0.0"
NOTEBOOK_VERSION = "6.6"

__all__ = [
    "__version__", "NOTEBOOK_VERSION", "GLOBAL_SEED", "N_BOOT", "PREREG", "seed_for", "geometric_answer", "static_twin",
    "twin_contact_height", "geometry_share", "mixture_rho", "slope_weight_table", "beta_to_effective_w", "prior_weight",
    "per_object_share", "motion_regime", "mover_rho", "scale_and_gate",
]

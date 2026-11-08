"""The probe: log-depth error, geometry share, slope-weight identity and per-object share.

Paper, Sec. 3 ("The probe"). For a purely parallel mover the two images are those of its static
twin at depth ``Z / (1 - s)`` (Proposition 1), so a model that triangulates has
``rho = log(Zhat / Z) = x`` with ``x = -log(1 - s)``, and a model that ignores the second view has a
``rho`` that does not depend on ``x``. The geometry share ``beta_s`` is the least-squares slope of
``rho`` on ``x`` over purely parallel movers.

Measurement functions mirror notebook cell 4.6 (``0680_measurement.py``), the identity mirrors
cell 2.4 (``0330_a2_identity.py``) and the shares mirror cells 6.A and 6.8b.
"""
from __future__ import annotations

import numpy as np

from .geometry import geometric_answer
from .registered import MOVER_DESIGN, PREREG
from .stats import slope_with_ci

__all__ = [
    "scale_and_gate", "mover_rho", "gate_passes", "motion_regime", "is_measurable", "is_identifiable", "e1_mask",
    "h9_mask", "geometry_share", "mixture_rho", "slope_weight_table", "beta_to_effective_w", "prior_weight",
    "per_object_share",
]


def scale_and_gate(pred, gt, smask, max_px=40_000, rng_seed=0):
    """Scale alignment and validity-gate statistics over the static pixels of one image.

    One scale factor per image and model, ``median(Z_true / Z_pred)`` over static pixels; the
    gate statistics are the OLS slope of ``log Z_pred`` on ``log Z_true`` and the AbsRel after
    alignment (at most ``max_px`` static pixels, sampled with a fixed seed).
    """
    p, g = pred[smask].astype(np.float64), gt[smask].astype(np.float64)
    ok = np.isfinite(p) & (p > 0)
    p, g = p[ok], g[ok]
    if len(p) < 100:
        return dict(scale=np.nan, slope=np.nan, absrel=np.nan, n_static=int(len(p)))
    s = float(np.median(g / p))
    if len(p) > max_px:
        sel = np.random.default_rng(rng_seed).choice(len(p), max_px, replace=False)
        p, g = p[sel], g[sel]
    slope = float(np.polyfit(np.log(g), np.log(p), 1)[0])
    absrel = float(np.mean(np.abs(s * p - g) / g))
    return dict(scale=s, slope=slope, absrel=absrel, n_static=int(ok.sum()))


def gate_passes(slope, absrel, slope_range=None, absrel_max=None):
    """Registered validity gate: static log-depth slope in [0.7, 1.3] and AbsRel <= 0.25.

    Applied per model and testbed to the medians over the images of :func:`scale_and_gate` (notebook
    cell 6.0); the probe is interpreted only where the gate passes.
    """
    lo, hi = slope_range or PREREG["gate_slope_range"]
    absrel_max = PREREG["gate_absrel_max"] if absrel_max is None else absrel_max
    return bool(np.isfinite(slope) and np.isfinite(absrel) and lo <= slope <= hi and absrel <= absrel_max)


def mover_rho(pred, gt, mask, scale, min_pixels=None):
    """``rho``: median of ``log(scale * Z_pred / Z_true)`` over a mover's (eroded, visible) pixels."""
    min_pixels = PREREG["min_object_pixels"] if min_pixels is None else min_pixels
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    ok = np.isfinite(p) & (p > 0) & np.isfinite(g)
    if ok.sum() < min_pixels or not np.isfinite(scale):
        return np.nan
    return float(np.median(np.log(scale * p[ok] / g[ok])))


def motion_regime(eps, parallel_max=None, perpendicular_min=None):
    """Registered motion regime from the exact epipolar offset (pixels).

    ``parallel`` if ``|eps| < 1``, ``perpendicular`` if ``|eps| >= 3``, ``intermediate`` otherwise.
    """
    parallel_max = PREREG["eps_parallel_max"] if parallel_max is None else parallel_max
    perpendicular_min = PREREG["eps_perp_min"] if perpendicular_min is None else perpendicular_min
    e = np.abs(np.asarray(eps, np.float64))
    out = np.where(e < parallel_max, "parallel", np.where(e >= perpendicular_min, "perpendicular", "intermediate"))
    return out if out.ndim else str(out)


def is_measurable(n_eroded_px, visible_frac, min_pixels=None, min_visible=0.85):
    """Registered sample rule: at least 40 visible pixels after erosion and 85% of the silhouette visible."""
    min_pixels = PREREG["min_object_pixels"] if min_pixels is None else min_pixels
    return (np.asarray(n_eroded_px) >= min_pixels) & (np.asarray(visible_frac) >= min_visible)


def is_identifiable(parallax_px, min_parallax=None):
    """Registered sample rule: translational parallax of at least 5 px (depth identifiable from two views)."""
    min_parallax = PREREG["parallax_min_px"] if min_parallax is None else min_parallax
    return np.asarray(parallax_px) >= min_parallax


def e1_mask(d):
    """E1 sample (Prediction 1, Table 1): purely parallel movers (eps = 0 at every pixel), s < 1, measurable, identifiable.

    ``d`` is a mover table with the columns ``motion_type``, ``s``, ``measurable`` and ``identifiable``
    (for example ``results/v6.6/manifest/convoy3d_main_movers.csv``).
    """
    return ((d["motion_type"] == "parallel") & (d["s"] < 1) & d["measurable"].astype(bool)
            & d["identifiable"].astype(bool))


def h9_mask(d):
    """H9 sample (Virtual KITTI 2, exploratory): parallel regime, measurable, identifiable, x_geo in the design range."""
    import pandas as pd

    lo, hi = PREREG["h9_x_range"]
    x = pd.to_numeric(d["gt_x_geo"], errors="coerce")
    return ((d["regime"] == "parallel") & d["measurable"].astype(bool) & d["identifiable"].astype(bool)
            & np.isfinite(x) & (x >= lo) & (x <= hi))


def geometry_share(s, rho, clusters, seed, n_boot=None):
    """Geometry share ``beta_s``: OLS slope of ``rho`` on ``x = -log(1 - s)`` with a cluster-bootstrap CI."""
    return slope_with_ci(geometric_answer(s), rho, clusters, seed, n_boot=n_boot)


def mixture_rho(x, w):
    """Log-depth error of a predictor that mixes geometry and an unbiased prior (Lemma 1).

    ``Zhat = w Z + (1 - w) Z / (1 - s)``  gives  ``rho = log(w + (1 - w) e^x)``.
    """
    return np.log(w + (1.0 - w) * np.exp(np.asarray(x, np.float64)))


def slope_weight_table(s_range=None, n_grid=2001, weights=None, x=None):
    """OLS slope ``beta(w)`` of ``mixture_rho`` on ``x`` over a uniform grid of the design range of ``s``.

    This is the table of notebook cell 2.4 (``a2_and_identity.json`` in the run logs); it maps a
    slope to an equivalent prior weight (the ``w_eff`` of ``results/E1_beta_s.csv``). The slope
    depends on the distribution of ``x``: over the x values of the 220 E1 movers, ``beta(0.5)`` is
    0.613 instead of 0.620 (``results/oracle_validation.csv``); pass ``x`` to use given values.
    """
    weights = np.round(np.linspace(0, 1, 11), 2) if weights is None else np.asarray(weights, float)
    if x is None:
        s_lo, s_hi = s_range or MOVER_DESIGN["s_range_main"]
        x = geometric_answer(np.linspace(s_lo, s_hi, n_grid))
    x = np.asarray(x, np.float64)
    rows = []
    for w in weights:
        beta = np.polyfit(x, mixture_rho(x, w), 1)[0]
        rows.append(dict(w=float(w), beta_ols_over_design=float(beta), local_slope_at_x0=float(1 - w)))
    return rows


def beta_to_effective_w(beta, table=None):
    """Invert ``beta(w)`` by interpolation (``beta`` is monotone decreasing in ``w``)."""
    table = table or slope_weight_table()
    tab = sorted(table, key=lambda r: r["beta_ols_over_design"])
    return float(np.interp(beta, [r["beta_ols_over_design"] for r in tab], [r["w"] for r in tab]))


def prior_weight(x_geo, rho, rho_mono, separability_min=None):
    """Per-mover prior weight ``w`` from ``Zhat = w Zhat_mono + (1 - w) Z_geo`` (E3).

    ``w = (e^rho - e^x_geo) / (e^rho_mono - e^x_geo)``, defined where the two hypotheses are
    separable (``|x_geo - rho_mono| > 0.10``), NaN elsewhere.
    """
    separability_min = PREREG["w_separability_min"] if separability_min is None else separability_min
    x, r, rm = (np.asarray(a, np.float64) for a in (x_geo, rho, rho_mono))
    sep = np.abs(x - rm) > separability_min
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        w = (np.exp(r) - np.exp(x)) / (np.exp(rm) - np.exp(x))
    return np.where(sep & np.isfinite(w), w, np.nan)


def per_object_share(rho, rho_mono, x_geo):
    """Per-object geometry share ``g = (e^rho - e^rho_mono) / (e^x_geo - e^rho_mono)``.

    0 at the model's own single-image depth, 1 at the geometric answer. The paper uses it on data
    whose motion is not designed (Virtual KITTI 2, Argoverse 2), for objects with
    ``|x_geo - rho_mono| >= 0.3``.
    """
    a, b, c = (np.exp(np.asarray(v, float)) for v in (rho, rho_mono, x_geo))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (c - b)

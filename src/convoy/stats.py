"""Statistics of the paper: cluster bootstrap, OLS slopes, paired contrasts, Holm, verdict rules.

Ported from the notebook (cells 6.A ``0810_stats.py``, 6.2 ``0840_e2.py`` and 6.13 ``0889h_twin.py``)
with the registered constants made explicit arguments; ``tests/test_notebook_equivalence.py``
checks that the results are identical to the notebook's functions.

All intervals are 95% percentile intervals of a cluster bootstrap: clusters (scenes, image pairs
or vehicle tracks) are resampled with replacement, so movers that share images are resampled
together. Linear statistics are bootstrapped exactly through per-cluster sums.
"""
from __future__ import annotations

import numpy as np

from .registered import N_BOOT, PREREG

__all__ = [
    "ClusterBootstrap", "ols_fit", "slope_with_ci", "slope_boot", "multi_slopes", "contrast_ci", "paired_slopes",
    "boot_stat", "percentile_ci", "holm", "verdict_positive", "verdict_p1", "verdict_p2",
    "stratified_recall_difference", "share_with_ci",
]


class ClusterBootstrap:
    """Resampling plan over clusters, shared by every statistic computed from it.

    ``counts[b, c]`` is how often cluster ``c`` is drawn in resample ``b``; a statistic that is a
    function of per-cluster column sums is bootstrapped as ``counts @ sums``.
    """

    def __init__(self, cluster_ids, n_boot: int | None = None, seed: int = 0):
        n_boot = n_boot or N_BOOT
        self.ids, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
        self.C, self.inv, self.B = len(self.ids), inv, n_boot
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, self.C, size=(n_boot, self.C))
        self.counts = np.zeros((n_boot, self.C), np.float64)
        np.add.at(self.counts, (np.repeat(np.arange(n_boot), self.C), self.draws.ravel()), 1.0)
        self.members = [np.nonzero(inv == c)[0] for c in range(self.C)]

    def sums(self, cols):
        """Per-cluster sums of each column, shape ``(C, k)``."""
        S = np.zeros((self.C, len(cols)))
        for k, c in enumerate(cols):
            np.add.at(S[:, k], self.inv, c)
        return S

    def linear(self, cols):
        """Bootstrap distribution of the column totals ``(B, k)`` and the point totals ``(k,)``."""
        S = self.sums(cols)
        return self.counts @ S, S.sum(0)

    def resample_indices(self, b):
        return np.concatenate([self.members[c] for c in self.draws[b]])


def ols_fit(x, y):
    """Least-squares slope, intercept and R^2 of ``y`` on ``x``."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (slope * x + icpt)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    return float(slope), float(icpt), float(r2)


def _slopes_from_totals(T):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (T[:, 0] * T[:, 4] - T[:, 1] * T[:, 2]) / (T[:, 0] * T[:, 3] - T[:, 1] ** 2)


def slope_with_ci(x, y, clusters, seed, h0=None, n_boot=None):
    """OLS slope with a 95% cluster-bootstrap interval (the estimator of the geometry share).

    Returns ``dict(n, n_clusters, beta, ci_lo, ci_hi, r2, intercept, p_h0)``; ``p_h0`` is the
    one-sided bootstrap p-value of ``beta <= h0`` when ``h0`` is given. Fewer than 8 points or
    5 clusters give NaN.
    """
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[ok], y[ok], clusters[ok]
    out = dict(n=int(len(x)), n_clusters=int(len(np.unique(clusters))))
    if len(x) < 8 or len(np.unique(clusters)) < 5 or np.var(x) == 0:
        return {**out, "beta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "r2": np.nan, "intercept": np.nan,
                "p_h0": np.nan}
    beta, icpt, r2 = ols_fit(x, y)
    bs = ClusterBootstrap(clusters, n_boot=n_boot, seed=seed)
    T, _ = bs.linear([np.ones_like(x), x, y, x * x, x * y])
    bb = _slopes_from_totals(T)
    bb = bb[np.isfinite(bb)]
    lo, hi = np.percentile(bb, [2.5, 97.5])
    p = float((1 + np.sum(bb <= h0)) / (len(bb) + 1)) if h0 is not None else np.nan
    return {**out, "beta": beta, "ci_lo": float(lo), "ci_hi": float(hi), "r2": r2, "intercept": icpt, "p_h0": p}


def slope_boot(x, y, clusters, seed, n_boot=None):
    """OLS slope and its cluster-bootstrap draws."""
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[ok], y[ok], clusters[ok]
    if len(x) < 8 or len(np.unique(clusters)) < 5:
        return np.nan, np.array([])
    beta = ols_fit(x, y)[0]
    T, _ = ClusterBootstrap(clusters, n_boot=n_boot, seed=seed).linear([np.ones_like(x), x, y, x * x, x * y])
    return beta, _slopes_from_totals(T)


def multi_slopes(x, ys, clusters, seed, n_boot=None):
    """Slopes of several responses on the same regressor (paired design) with joint bootstrap draws.

    Returns ``(slopes (k,), draws (B, k), n, n_clusters)``. This is how the paired contrasts of
    E5, E5-REAL and the designed sets are computed: every condition is a response of the same movers.
    """
    x = np.asarray(x, float)
    Y = np.column_stack([np.asarray(y, float) for y in ys])
    cl = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(Y).all(1)
    x, Y, cl = x[ok], Y[ok], cl[ok]
    k, n, nc = Y.shape[1], len(x), len(np.unique(cl))
    if n < 8 or nc < 5 or np.var(x) == 0:
        return np.full(k, np.nan), np.zeros((0, k)), n, nc
    cols = [np.ones_like(x), x, x * x] + [Y[:, i] for i in range(k)] + [x * Y[:, i] for i in range(k)]
    T, P = ClusterBootstrap(cl, n_boot=n_boot, seed=seed).linear(cols)

    def _sl(t):
        den = t[:, 0] * t[:, 2] - t[:, 1] ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return (t[:, :1] * t[:, 3 + k:3 + 2 * k] - t[:, 1:2] * t[:, 3:3 + k]) / den[:, None]

    return _sl(P[None])[0], _sl(T), n, nc


def contrast_ci(point, draws, weights):
    """Point estimate and 95% percentile interval of a linear contrast of jointly bootstrapped statistics."""
    w = np.asarray(weights, float)
    est = float(np.dot(point, w))
    if len(draws) < 50 or not np.isfinite(est):
        return est, np.nan, np.nan
    v = draws @ w
    v = v[np.isfinite(v)]
    if len(v) < 50:
        return est, np.nan, np.nan
    lo, hi = np.percentile(v, [2.5, 97.5])
    return est, float(lo), float(hi)


def paired_slopes(x, y_a, y_b, clusters, seed, n_boot=None):
    """Slopes of ``y_a`` and ``y_b`` on the same ``x`` and the bootstrap draws of their difference."""
    b, draws, _n, _nc = multi_slopes(x, [y_a, y_b], clusters, seed, n_boot=n_boot)
    return b[0], b[1], (draws[:, 0] - draws[:, 1]) if len(draws) else np.array([])


def boot_stat(df, cluster_col, fn, seed, n_boot=None, cols=None):
    """Generic (non-linear) statistic of a data frame: point estimate and 95% percentile interval.

    ``fn`` receives a dict of NumPy columns. Returns ``(point, lo, hi, draws)``.
    """
    cols = cols or [c for c in df.columns if c != cluster_col]
    arrs = {c: df[c].to_numpy() for c in cols}
    bs = ClusterBootstrap(df[cluster_col].values, n_boot=n_boot, seed=seed)
    point = fn(arrs)
    vals = []
    for b in range(bs.B):
        idx = bs.resample_indices(b)
        v = fn({c: a[idx] for c, a in arrs.items()})
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return point, np.nan, np.nan, np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), np.asarray(vals)


def percentile_ci(draws):
    """95% percentile interval of bootstrap draws (NaN with 50 or fewer finite draws)."""
    bb = np.asarray(draws)[np.isfinite(draws)] if len(draws) else np.array([])
    return tuple(np.percentile(bb, [2.5, 97.5])) if len(bb) > 50 else (np.nan, np.nan)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN entries are ignored and stay NaN)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def verdict_positive(lo, hi):
    """Registered rule for a contrast predicted to be positive (H5-H14)."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NOT TESTABLE"
    return "SUPPORTED" if lo > 0 else ("CONTRADICTED" if hi < 0 else "NOT SUPPORTED")


def verdict_p1(beta, lo, hi, r2, beta_ci_band=None, beta_support_range=None, r2_min=None):
    """Locked rule of Prediction 1 (the models inherit the degeneracy: beta_s in [0.8, 1.2])."""
    a, b = beta_ci_band or PREREG["beta_ci_band"]
    s_lo, s_hi = beta_support_range or PREREG["beta_support_range"]
    r2_min = PREREG["r2_min"] if r2_min is None else r2_min
    if not np.isfinite(beta):
        return "NOT TESTABLE"
    if hi < a or lo > b or r2 < r2_min:
        return "FALSIFIED"
    if s_lo <= beta <= s_hi and lo >= a and hi <= b:
        return "SUPPORTED"
    return "PARTIAL"


def verdict_p2(diff, ci_lo, ci_hi, recall_diff_min=None):
    """Locked rule of Prediction 2 (masks miss parallel movers by at least 0.3 in recall)."""
    recall_diff_min = PREREG["recall_diff_min"] if recall_diff_min is None else recall_diff_min
    if not np.isfinite(diff):
        return "NOT TESTABLE"
    if ci_lo <= 0 <= ci_hi or ci_hi < 0:
        return "FALSIFIED"
    return "SUPPORTED" if diff >= recall_diff_min else "PARTIAL"


def stratified_recall_difference(df, recall_col, seed, n_boot=None):
    """Recall(perpendicular) - recall(parallel), matched on |s| and object area (Prediction 2).

    ``df`` needs the columns ``regime``, ``s``, ``gt_area_px``, ``scene`` and ``recall_col``. The
    difference is computed within the 3 x 3 strata of |s| tertile and area tertile and averaged
    with weights ``n_par + n_perp``; the interval is a cluster bootstrap over scenes.
    """
    import pandas as pd

    d = df[df["regime"].isin(["parallel", "perpendicular"]) & np.isfinite(df[recall_col])].copy()
    if d["regime"].nunique() < 2 or len(d) < 10:
        return dict(diff=np.nan, ci_lo=np.nan, ci_hi=np.nan, rec_par=np.nan, rec_perp=np.nan, n_par=0, n_perp=0)
    d["s_bin"] = pd.qcut(d["s"].abs().rank(method="first"), 3, labels=False)
    d["a_bin"] = pd.qcut(d["gt_area_px"].rank(method="first"), 3, labels=False)
    d["stratum"] = d["s_bin"] * 3 + d["a_bin"]
    cols = []
    for k in range(9):
        ink = (d["stratum"] == k).values
        par = ink & (d["regime"] == "parallel").values
        perp = ink & (d["regime"] == "perpendicular").values
        rv = d[recall_col].values
        cols += [par.astype(float), rv * par, perp.astype(float), rv * perp]
    bs = ClusterBootstrap(d["scene"].values, n_boot=n_boot, seed=seed)
    T, P = bs.linear(cols)

    def _diff(t):
        t = t.reshape(-1, 9, 4)
        n_par, s_par, n_perp, s_perp = t[..., 0], t[..., 1], t[..., 2], t[..., 3]
        ok = (n_par > 0) & (n_perp > 0)
        w = np.where(ok, n_par + n_perp, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dk = np.where(ok, s_perp / np.maximum(n_perp, 1) - s_par / np.maximum(n_par, 1), 0.0)
            return (w * dk).sum(-1) / w.sum(-1)

    point = float(_diff(P[None])[0])
    boots = _diff(T)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(diff=point, ci_lo=float(lo), ci_hi=float(hi),
                rec_par=float(d.loc[d["regime"] == "parallel", recall_col].mean()),
                rec_perp=float(d.loc[d["regime"] == "perpendicular", recall_col].mean()),
                n_par=int((d["regime"] == "parallel").sum()), n_perp=int((d["regime"] == "perpendicular").sum()))


def share_with_ci(flags, clusters, seed, n_boot=None):
    """Share of flagged items with a cluster-bootstrap interval (recall of the twin and epipolar tests).

    ``flags`` is a boolean or 0/1 array (or a list of such arrays sharing ``clusters``); returns a
    list of ``(share, lo, hi)``. Fewer than 5 items or clusters give no interval.
    """
    flags = [np.asarray(f, float) for f in (flags if isinstance(flags, (list, tuple)) else [flags])]
    clusters = np.asarray(clusters)
    if len(clusters) < 5 or len(np.unique(clusters)) < 5:
        return [(float(f.mean()) if len(f) else np.nan, np.nan, np.nan) for f in flags]
    T, P = ClusterBootstrap(clusters, n_boot=n_boot, seed=seed).linear([np.ones(len(clusters))] + flags)
    out = []
    for k in range(len(flags)):
        lo, hi = percentile_ci(T[:, k + 1] / T[:, 0])
        out.append((float(P[k + 1] / P[0]), lo, hi))
    return out

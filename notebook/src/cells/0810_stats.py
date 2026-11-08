# PLACEHOLDER
# 6.A Statistics: cluster bootstrap (vectorised for linear statistics), OLS slope with R^2,
#     Holm-Bonferroni, three-level verdicts.
from scipy.stats import spearmanr

class ClusterBootstrap:
    def __init__(self, cluster_ids, n_boot=None, seed=0):
        n_boot = n_boot or RUN_CONFIG["n_boot"]
        self.ids, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
        self.C, self.inv, self.B = len(self.ids), inv, n_boot
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, self.C, size=(n_boot, self.C))
        self.counts = np.zeros((n_boot, self.C), np.float64)
        np.add.at(self.counts, (np.repeat(np.arange(n_boot), self.C), self.draws.ravel()), 1.0)
        self.members = [np.nonzero(inv == c)[0] for c in range(self.C)]

    def sums(self, cols):
        # per-cluster sums of each column -> (C, k)
        S = np.zeros((self.C, len(cols)))
        for k, c in enumerate(cols):
            np.add.at(S[:, k], self.inv, c)
        return S

    def linear(self, cols):
        # bootstrap distribution of the column totals -> (B, k), and the point totals (k,)
        S = self.sums(cols)
        return self.counts @ S, S.sum(0)

    def resample_indices(self, b):
        return np.concatenate([self.members[c] for c in self.draws[b]])

def ols_fit(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (slope * x + icpt)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    return float(slope), float(icpt), float(r2)

def slope_with_ci(x, y, clusters, seed, h0=None):
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[ok], y[ok], clusters[ok]
    out = dict(n=int(len(x)), n_clusters=int(len(np.unique(clusters))))
    if len(x) < 8 or len(np.unique(clusters)) < 5 or np.var(x) == 0:
        return {**out, "beta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "r2": np.nan, "intercept": np.nan, "p_h0": np.nan}
    beta, icpt, r2 = ols_fit(x, y)
    bs = ClusterBootstrap(clusters, seed=seed)
    T, _ = bs.linear([np.ones_like(x), x, y, x * x, x * y])
    with np.errstate(invalid="ignore", divide="ignore"):
        bb = (T[:, 0] * T[:, 4] - T[:, 1] * T[:, 2]) / (T[:, 0] * T[:, 3] - T[:, 1] ** 2)
    bb = bb[np.isfinite(bb)]
    lo, hi = np.percentile(bb, [2.5, 97.5])
    p = float((1 + np.sum(bb <= h0)) / (len(bb) + 1)) if h0 is not None else np.nan
    return {**out, "beta": beta, "ci_lo": float(lo), "ci_hi": float(hi), "r2": r2, "intercept": icpt, "p_h0": p}

def boot_stat(df, cluster_col, fn, seed, n_boot=None, cols=None):
    # generic (non-linear) statistic: point estimate and percentile CI. fn receives a dict of
    # NumPy columns (fast); cols lists the columns it needs.
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

def holm(pvals):
    p = np.asarray(pvals, float); out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok]); m = ok.sum()
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m); tmp[idx] = adj
    out[ok] = tmp
    return out

def verdict_p1(beta, lo, hi, r2):
    if not np.isfinite(beta):
        return "NOT TESTABLE"
    a, b = PREREG["beta_ci_band"]
    if hi < a or lo > b or r2 < PREREG["r2_min"]:
        return "FALSIFIED"
    if PREREG["beta_support_range"][0] <= beta <= PREREG["beta_support_range"][1] and lo >= a and hi <= b:
        return "SUPPORTED"
    return "PARTIAL"

def e1_mask(d):
    # E1 movers: purely parallel (eps = 0 at every pixel), s < 1, measurable, identifiable
    return ((d["motion_type"] == "parallel") & (d["s"] < 1) & d["measurable"].astype(bool)
            & d["identifiable"].astype(bool))

def h9_mask(d):
    # H9 movers (Virtual KITTI 2): parallel regime, measurable, identifiable, x_geo in the design range
    lo, hi = PREREG["h9_x_range"]
    x = pd.to_numeric(d["gt_x_geo"], errors="coerce")
    return ((d["regime"] == "parallel") & d["measurable"].astype(bool) & d["identifiable"].astype(bool)
            & np.isfinite(x) & (x >= lo) & (x <= hi))

def prior_weight(df):
    # Zhat = w Zhat_mono + (1 - w) Z_geo  =>  w = (e^rho - e^x_geo) / (e^rho_mono - e^x_geo),
    # defined only where the two hypotheses are separable (|x_geo - rho_mono| > 0.10)
    x, r, rm = df["gt_x_geo"].values, df["rho"].values, df["rho_mono"].values
    sep = np.abs(x - rm) > PREREG["w_separability_min"]
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        w = (np.exp(r) - np.exp(x)) / (np.exp(rm) - np.exp(x))
    return np.where(sep & np.isfinite(w), w, np.nan)

def slope_boot(x, y, clusters, seed):
    # OLS slope and its cluster-bootstrap draws
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[ok], y[ok], clusters[ok]
    if len(x) < 8 or len(np.unique(clusters)) < 5:
        return np.nan, np.array([])
    beta = ols_fit(x, y)[0]
    T, _ = ClusterBootstrap(clusters, seed=seed).linear([np.ones_like(x), x, y, x * x, x * y])
    with np.errstate(invalid="ignore", divide="ignore"):
        bb = (T[:, 0] * T[:, 4] - T[:, 1] * T[:, 2]) / (T[:, 0] * T[:, 3] - T[:, 1] ** 2)
    return beta, bb

def multi_slopes(x, ys, clusters, seed):
    # OLS slopes of several responses on the SAME regressor (paired design) with their joint
    # cluster-bootstrap draws: returns (slopes (k,), draws (B, k), n, n_clusters)
    x = np.asarray(x, float); Y = np.column_stack([np.asarray(y, float) for y in ys])
    cl = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(Y).all(1)
    x, Y, cl = x[ok], Y[ok], cl[ok]
    k, n, nc = Y.shape[1], len(x), len(np.unique(cl))
    if n < 8 or nc < 5 or np.var(x) == 0:
        return np.full(k, np.nan), np.zeros((0, k)), n, nc
    cols = [np.ones_like(x), x, x * x] + [Y[:, i] for i in range(k)] + [x * Y[:, i] for i in range(k)]
    T, P = ClusterBootstrap(cl, seed=seed).linear(cols)
    def _sl(t):
        den = t[:, 0] * t[:, 2] - t[:, 1] ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return (t[:, :1] * t[:, 3 + k:3 + 2 * k] - t[:, 1:2] * t[:, 3:3 + k]) / den[:, None]
    return _sl(P[None])[0], _sl(T), n, nc

def contrast_ci(point, draws, weights):
    # point estimate and 95% percentile CI of a linear contrast of jointly bootstrapped statistics
    w = np.asarray(weights, float)
    est = float(np.dot(point, w))
    if len(draws) < 50 or not np.isfinite(est):
        return est, np.nan, np.nan
    v = draws @ w; v = v[np.isfinite(v)]
    if len(v) < 50:
        return est, np.nan, np.nan
    lo, hi = np.percentile(v, [2.5, 97.5])
    return est, float(lo), float(hi)

def paired_slopes(x, y_a, y_b, clusters, seed):
    # slopes of y_a and y_b on the same x, and the bootstrap distribution of their difference
    b, draws, n, nc = multi_slopes(x, [y_a, y_b], clusters, seed)
    return b[0], b[1], (draws[:, 0] - draws[:, 1]) if len(draws) else np.array([])

def _ci(bb):
    bb = np.asarray(bb)[np.isfinite(bb)] if len(bb) else np.array([])
    return tuple(np.percentile(bb, [2.5, 97.5])) if len(bb) > 50 else (np.nan, np.nan)

def verdict_positive(lo, hi):
    # registered v6 rule for a contrast predicted to be positive
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NOT TESTABLE"
    return "SUPPORTED" if lo > 0 else ("CONTRADICTED" if hi < 0 else "NOT SUPPORTED")

BOOT_N = RUN_CONFIG["n_boot"]
print(f"Statistics ready (cluster bootstrap, {BOOT_N} resamples).")
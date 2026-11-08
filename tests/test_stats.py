"""Statistics: the vectorised cluster bootstrap equals explicit resampling; Holm; verdict rules."""
import numpy as np
import pytest

import convoy.stats as S


def _data(seed=0, n=150, clusters=40):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 1.9, n)
    cl = rng.integers(0, clusters, n)
    y = 0.3 * x + rng.normal(0, 0.2, n)
    return x, y, cl


def test_linear_bootstrap_equals_explicit_resampling():
    x, y, cl = _data()
    bs = S.ClusterBootstrap(cl, n_boot=200, seed=1)
    T, P = bs.linear([np.ones_like(x), x, y, x * x, x * y])
    assert P[0] == len(x)
    for b in range(0, 200, 23):
        idx = bs.resample_indices(b)
        np.testing.assert_allclose(T[b], [len(idx), x[idx].sum(), y[idx].sum(), (x[idx] ** 2).sum(), (x[idx] * y[idx]).sum()])
    beta, draws = S.slope_boot(x, y, cl, 1, n_boot=200)
    for b in range(0, 200, 31):
        idx = bs.resample_indices(b)
        assert draws[b] == pytest.approx(np.polyfit(x[idx], y[idx], 1)[0], rel=1e-9)
    assert beta == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-12)


def test_slope_with_ci_is_deterministic_and_covers_the_truth():
    x, y, cl = _data(seed=2, n=400, clusters=80)
    a = S.slope_with_ci(x, y, cl, seed=7, n_boot=2000)
    b = S.slope_with_ci(x, y, cl, seed=7, n_boot=2000)
    assert a == b
    assert a["ci_lo"] < 0.3 < a["ci_hi"] and a["ci_lo"] < a["beta"] < a["ci_hi"]
    assert a["n"] == 400 and a["n_clusters"] == 80
    c = S.slope_with_ci(x[:7], y[:7], cl[:7], seed=7)                           # fewer than 8 movers: not estimable
    assert np.isnan(c["beta"])


def test_paired_contrast_uses_the_same_resamples():
    x, y, cl = _data(seed=3)
    y2 = y + 0.2 * x
    b_a, b_b, diff = S.paired_slopes(x, y2, y, cl, seed=4, n_boot=500)
    assert b_a - b_b == pytest.approx(0.2, abs=1e-9)
    np.testing.assert_allclose(diff, 0.2, atol=1e-9)                             # identical movers in every resample
    est, lo, hi = S.contrast_ci(np.array([b_a, b_b]), np.column_stack([diff + b_b, np.full_like(diff, b_b)]), [1, -1])
    assert est == pytest.approx(0.2) and lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


def test_holm():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.02, 0.03, 0.06, 0.06
    np.testing.assert_allclose(S.holm(p)[:4], [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(S.holm(p)[4])
    assert S.holm([0.9, 0.8]).max() == 1.0


def test_verdict_rules():
    assert S.verdict_positive(0.01, 0.3) == "SUPPORTED"
    assert S.verdict_positive(-0.01, 0.3) == "NOT SUPPORTED"
    assert S.verdict_positive(-0.3, -0.01) == "CONTRADICTED"
    assert S.verdict_positive(np.nan, 0.3) == "NOT TESTABLE"
    assert S.verdict_p1(1.0, 0.9, 1.1, 0.8) == "SUPPORTED"
    assert S.verdict_p1(0.17, 0.09, 0.26, 0.1) == "FALSIFIED"                     # DUSt3R on CONVOY-3D
    assert S.verdict_p2(0.83, 0.80, 0.87) == "SUPPORTED"
    assert S.verdict_p2(0.25, 0.18, 0.33) == "PARTIAL"
    assert S.verdict_p2(0.02, -0.04, 0.08) == "FALSIFIED"


def test_share_with_ci():
    rng = np.random.default_rng(5)
    cl = rng.integers(0, 50, 300)
    flags = rng.random(300) < 0.3
    (share, lo, hi), = S.share_with_ci(flags, cl, seed=6, n_boot=1000)
    assert share == pytest.approx(flags.mean()) and lo < share < hi
    (share, lo, hi), = S.share_with_ci(flags[:4], cl[:4], seed=6)
    assert np.isnan(lo) and np.isnan(hi)

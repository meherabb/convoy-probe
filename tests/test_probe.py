"""The probe: slope-weight identity (Lemma 1), estimator checks and the registered sample rules on the shipped data."""
import json

import numpy as np
import pandas as pd
import pytest
from helpers import RESULTS

import convoy.probe as P
from convoy.geometry import geometric_answer

MOVER_TABLES = ["convoy3d_main_movers.csv", "convoy3d_e4_movers.csv", "convoy3d_ablation_movers.csv",
                "convoy3d_baseline_sweep_movers.csv", "convoy3d_dose_response_movers.csv", "convoy_real_movers.csv"]


def test_lemma_1_mixture():
    x = np.linspace(-0.47, 1.9, 101)
    for w in (0.0, 0.3, 0.5, 1.0):
        rho = P.mixture_rho(x, w)
        np.testing.assert_allclose(np.exp(rho), w + (1 - w) * np.exp(x))
        d = (P.mixture_rho(1e-6, w) - P.mixture_rho(-1e-6, w)) / 2e-6           # slope at x = 0 is 1 - w
        assert d == pytest.approx(1 - w, abs=1e-8)


def test_slope_weight_table_reproduces_the_run():
    """The table of notebook cell 2.4 (beta(w) over the design range of s), bit for bit up to rounding."""
    logged = json.loads((RESULTS / "logs" / "a2_and_identity.json").read_text())["slope_weight"]
    table = P.slope_weight_table()
    assert [r["w"] for r in table] == [r["w"] for r in logged]
    for a, b in zip(table, logged):
        assert a["beta_ols_over_design"] == pytest.approx(b["beta_ols_over_design"], abs=1e-12)
        assert a["local_slope_at_x0"] == pytest.approx(b["local_slope_at_x0"], abs=1e-12)
    beta = [r["beta_ols_over_design"] for r in table]
    assert beta[0] == pytest.approx(1.0) and beta[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(beta) < 0)
    assert round(table[5]["beta_ols_over_design"], 3) == 0.620                  # w = 0.5 over the design grid
    for w in (0.2, 0.5, 0.8):
        assert P.beta_to_effective_w(P.slope_weight_table(weights=[w])[0]["beta_ols_over_design"]) == pytest.approx(w, abs=1e-3)


def test_equivalent_prior_weights_of_the_paper():
    """The w_eff column of results/E1_beta_s.csv ('prior weights between 0.89 and 1', Sec. 5.1)."""
    e1 = pd.read_csv(RESULTS / "results" / "E1_beta_s.csv")
    for _, r in e1.iterrows():
        assert P.beta_to_effective_w(r["beta_s"]) == pytest.approx(r["w_eff"], abs=1e-12)


def test_identity_over_the_e1_movers_matches_the_oracle_check():
    """beta(w) over the x of the 220 E1 movers is the 'beta_theory_identity' of results/oracle_validation.csv."""
    movers = pd.read_csv(RESULTS / "manifest" / "convoy3d_main_movers.csv")
    e1 = movers[P.e1_mask(movers)]
    assert len(e1) == 220 and e1["scene"].nunique() == 159
    oracle = pd.read_csv(RESULTS / "results" / "oracle_validation.csv")
    for _, r in oracle.iterrows():
        beta = np.polyfit(e1["x_par"], P.mixture_rho(e1["x_par"], r["w_true"]), 1)[0]
        assert beta == pytest.approx(r["beta_theory_identity"], abs=1e-12)
        assert r["n"] == len(e1)
        assert abs(r["beta_s"] - beta) < 5e-4                                    # planted predictors return beta(w)
        assert abs(r["median_w_recovered"] - r["w_true"]) < 3e-4                 # and their weight
    assert round(float(oracle.loc[oracle["w_true"] == 0.5, "beta_theory_identity"].iloc[0]), 2) == 0.61


def test_prior_weight_and_per_object_share():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.4, 1.8, 500)
    rho_mono = rng.normal(0, 0.2, 500)
    w = rng.uniform(0, 1, 500)
    rho = np.log(w * np.exp(rho_mono) + (1 - w) * np.exp(x))                     # Zhat = w Zhat_mono + (1 - w) Z_geo
    est = P.prior_weight(x, rho, rho_mono)
    sep = np.abs(x - rho_mono) > 0.1
    np.testing.assert_allclose(est[sep], w[sep], atol=1e-9)
    assert np.isnan(est[~sep]).all()
    np.testing.assert_allclose(P.per_object_share(rho_mono, rho_mono, x), 0.0, atol=1e-12)
    np.testing.assert_allclose(P.per_object_share(x, rho_mono, x), 1.0, atol=1e-9)
    np.testing.assert_allclose(P.per_object_share(rho, rho_mono, x), 1 - w, atol=1e-9)


@pytest.mark.parametrize("table", MOVER_TABLES)
def test_registered_sample_rules_reproduce_the_mover_tables(table):
    """Regime, measurability and identifiability of every shipped mover follow from the registered rules."""
    d = pd.read_csv(RESULTS / "manifest" / table)
    assert (P.motion_regime(d["gt_eps_med"].to_numpy()) == d["regime"].to_numpy()).all()
    assert (P.is_measurable(d["gt_n_eroded"], d["gt_visible_frac"]) == d["measurable"].astype(bool)).all()
    assert (P.is_identifiable(d["gt_parallax_med"]) == d["identifiable"].astype(bool)).all()
    below = d["s"] < 1
    np.testing.assert_allclose(d.loc[below, "x_par"], geometric_answer(d.loc[below, "s"]), rtol=1e-12, atol=1e-12)
    if "motion_type" in d:                          # visible, purely parallel movers have eps = 0 and x_geo = x
        par = d[(d["motion_type"] == "parallel") & below & d["measurable"].astype(bool)]
        assert len(par) > 50
        assert (par["regime"] == "parallel").all()
        np.testing.assert_allclose(par["gt_x_geo"], par["x_par"], atol=1e-6)


def test_scale_alignment_and_gate():
    rng = np.random.default_rng(1)
    gt = rng.uniform(2, 80, (60, 80))
    smask = np.ones_like(gt, bool)
    g = P.scale_and_gate(gt / 7.0, gt, smask)                                    # a scaled copy of the truth
    assert g["scale"] == pytest.approx(7.0) and g["slope"] == pytest.approx(1.0) and g["absrel"] == pytest.approx(0.0, abs=1e-12)
    assert P.gate_passes(g["slope"], g["absrel"])
    flat = np.full_like(gt, 10.0) * np.exp(rng.normal(0, 1e-3, gt.shape))
    g = P.scale_and_gate(flat, gt, smask)
    assert not P.gate_passes(g["slope"], g["absrel"])                            # no parallax: the gate fails
    assert P.motion_regime(0.5) == "parallel" and P.motion_regime(2.0) == "intermediate"
    assert P.motion_regime(-3.0) == "perpendicular"
    mask = np.zeros_like(smask)
    mask[:10, :10] = True
    assert P.mover_rho(gt * 2, gt, mask, 0.5) == pytest.approx(0.0)
    assert np.isnan(P.mover_rho(gt, gt, mask & (np.arange(80) < 3)[None], 1.0))   # fewer than 40 pixels

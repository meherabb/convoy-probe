# PLACEHOLDER
# 6.10 Designed experiments (v6.5, D55): H12 baseline sweep and H13 contact dose-response (registered), with the shape of
#      each curve, the sweep by support and M2 recall by level (exploratory). Paired by mover; cluster = base scene.
_DSG_VALS = ["rho", "rho_mono", "gt_n_eroded", "gt_parallax_med", "gt_visible_frac", "measurable", "contact_px0", "contact_px1",
             "has_wall", "recall_M2_exact"]

def designed_wide(res, movers, model):
    # one row per base mover, one column per (quantity, level)
    d = res[res["model"] == model].copy()
    if not len(d):
        return pd.DataFrame()
    for c in ("has_wall", "measurable"):
        d[c] = d[c].astype(float)
    w = d.pivot_table(index=["uid", "scene"], columns="condition", values=[c for c in _DSG_VALS if c in d], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    gt = movers.drop_duplicates("uid")[["uid", "cls", "s", "x_par"]]
    return w.reset_index().merge(gt, on="uid", how="left")

def group_multi_slopes(x, ys, group, clusters, seed):
    """Slopes of several paired responses on the same x in two groups of movers that share clusters (joint cluster
    bootstrap). Returns slopes (2, k) [group True, group False], draws (B, 2, k) and the group sizes."""
    x = np.asarray(x, float); Y = np.column_stack([np.asarray(y, float) for y in ys]); gp = np.asarray(group, bool)
    cl = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(Y).all(1)
    x, Y, gp, cl = x[ok], Y[ok], gp[ok], cl[ok]
    k = Y.shape[1]
    if min(gp.sum(), (~gp).sum()) < 8 or len(np.unique(cl)) < 5:
        return np.full((2, k), np.nan), np.zeros((0, 2, k)), (int(gp.sum()), int((~gp).sum()))
    cols = []
    for g in (gp, ~gp):
        gf = g.astype(float)
        cols += [gf, gf * x, gf * x * x] + [gf * Y[:, i] for i in range(k)] + [gf * x * Y[:, i] for i in range(k)]
    T, P = ClusterBootstrap(cl, seed=seed).linear(cols)
    L = 3 + 2 * k
    def _sl(t, o):
        den = t[:, o] * t[:, o + 2] - t[:, o + 1] ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return (t[:, o:o + 1] * t[:, o + 3 + k:o + 3 + 2 * k] - t[:, o + 1:o + 2] * t[:, o + 3:o + 3 + k]) / den[:, None]
    return (np.stack([_sl(P[None], 0)[0], _sl(P[None], L)[0]]), np.stack([_sl(T, 0), _sl(T, L)], axis=1),
            (int(gp.sum()), int((~gp).sum())))

def _mean_ci(df, cols, seed):
    # mean of each column with a cluster-bootstrap interval (scenes), and the draws (linear statistic)
    bs = ClusterBootstrap(df["scene"].values, seed=seed)
    T, P = bs.linear([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    with np.errstate(invalid="ignore", divide="ignore"):
        dr = T[:, 1:] / T[:, :1]
    return P[1:] / P[0], dr

def _p_one_sided(draws):
    draws = np.asarray(draws)[np.isfinite(draws)]
    return float((1 + np.sum(draws <= 0)) / (len(draws) + 1)) if len(draws) else np.nan

_ok_dose = lambda n, nc: n >= PREREG["dose_min_movers"] and nc >= PREREG["dose_min_scenes"]
BSW_BETA_ROWS, BSW_TEST_ROWS, BSW_RECALL_ROWS, WDR_BETA_ROWS, WDR_TEST_ROWS, WDR_RECALL_ROWS = [], [], [], [], [], []
BSW_WIDE, WDR_WIDE = {}, {}
_hi12, _lo12 = PREREG["h12_contrast"]
_hi13, _lo13 = PREREG["h13_contrast"]
for model in EVALUATED_MODELS:
    # ---------------------------------------------------------------- baseline sweep (H12)
    w = designed_wide(BSW_RES, MOVERS_BSW, model)
    if len(w):
        BSW_WIDE[model] = w
        gate_ok = GATE_PASS.get((model, "BSW"), False)
        meas = np.all([w.get(f"measurable_{lv}", 0) == 1 for lv in BSW_LEVELS], axis=0)
        fin = np.all([np.isfinite(w.get(f"rho_{lv}", np.nan)) for lv in BSW_LEVELS], axis=0)
        w["h12_sample"] = meas & fin & (w.get("gt_parallax_med_B1", 0) >= PREREG["parallax_min_px"])
        s_ = w[w["h12_sample"]]
        b, dr, n, nc = multi_slopes(s_["x_par"], [s_[f"rho_{lv}"] for lv in BSW_LEVELS], s_["scene"], seed_for("bsw", model))
        for k, lv in enumerate(BSW_LEVELS):
            lo, hi = _ci(dr[:, k]) if len(dr) else (np.nan, np.nan)
            BSW_BETA_ROWS.append(dict(model=model, support="all", level=lv, factor=SWEEP_DESIGN["levels"][lv], n=n, n_scenes=nc,
                                      beta=b[k], ci_lo=lo, ci_hi=hi,
                                      median_parallax=float(s_[f"gt_parallax_med_{lv}"].median()) if len(s_) else np.nan,
                                      median_rho_mono=float(s_[f"rho_mono_{lv}"].median()) if len(s_) else np.nan))
        wts = [1.0 if lv == _hi12 else -1.0 if lv == _lo12 else 0.0 for lv in BSW_LEVELS]
        est, lo, hi = contrast_ci(b, dr, wts)
        p = _p_one_sided(dr @ np.asarray(wts)) if len(dr) else np.nan
        registered = model in PREREG["h12_models"]
        testable = _ok_dose(n, nc)
        v = ("NOT INTERPRETABLE (gate)" if not gate_ok else "NOT TESTABLE" if not testable
             else verdict_positive(lo, hi) if registered else "NO PREDICTION")
        BSW_TEST_ROWS.append(dict(model=model, test="H12", contrast=f"beta({SWEEP_DESIGN['labels'][_hi12]}) - beta({SWEEP_DESIGN['labels'][_lo12]})",
                                  registered=registered, n=n, n_scenes=nc, estimate=est, ci_lo=lo, ci_hi=hi, p_boot=p, verdict=v))
        # exploratory: the same rise for grounded (cars, crates) and floating movers, and its difference
        pt, drg, (n_f, n_g) = group_multi_slopes(s_["x_par"], [s_[f"rho_{lv}"] for lv in BSW_LEVELS], s_["cls"] == "crate_floating",
                                                 s_["scene"], seed_for("bsw-support", model))
        for gi, sup in enumerate(("floating", "grounded")):
            for k, lv in enumerate(BSW_LEVELS):
                lo_, hi_ = _ci(drg[:, gi, k]) if len(drg) else (np.nan, np.nan)
                BSW_BETA_ROWS.append(dict(model=model, support=sup, level=lv, factor=SWEEP_DESIGN["levels"][lv],
                                          n=(n_f, n_g)[gi], n_scenes=nc, beta=pt[gi, k], ci_lo=lo_, ci_hi=hi_))
        rise = pt @ np.asarray(wts)
        rdr = drg @ np.asarray(wts) if len(drg) else np.zeros((0, 2))
        for gi, sup in enumerate(("floating", "grounded")):
            lo_, hi_ = _ci(rdr[:, gi]) if len(rdr) else (np.nan, np.nan)
            BSW_TEST_ROWS.append(dict(model=model, test=f"H12-{sup}", contrast=f"rise x0.5 -> x3, {sup} movers", registered=False,
                                      n=(n_f, n_g)[gi], n_scenes=nc, estimate=rise[gi], ci_lo=lo_, ci_hi=hi_, p_boot=np.nan,
                                      verdict="EXPLORATORY" if gate_ok else "NOT INTERPRETABLE (gate)"))
        dd = rdr[:, 0] - rdr[:, 1] if len(rdr) else np.zeros(0)
        lo_, hi_ = _ci(dd) if len(dd) else (np.nan, np.nan)
        BSW_TEST_ROWS.append(dict(model=model, test="H12-interaction", contrast="rise of floating minus rise of grounded movers",
                                  registered=False, n=n_f + n_g, n_scenes=nc, estimate=float(rise[0] - rise[1]), ci_lo=lo_, ci_hi=hi_,
                                  p_boot=np.nan, verdict="EXPLORATORY" if gate_ok else "NOT INTERPRETABLE (gate)"))
        # exploratory: M2 recall at every baseline on the same movers
        rc = s_.dropna(subset=[f"recall_M2_exact_{lv}" for lv in BSW_LEVELS])
        if _ok_dose(len(rc), rc["scene"].nunique()):
            mu, mdr = _mean_ci(rc, [f"recall_M2_exact_{lv}" for lv in BSW_LEVELS], seed_for("bsw-m2", model))
            for k, lv in enumerate(BSW_LEVELS):
                lo_, hi_ = _ci(mdr[:, k])
                BSW_RECALL_ROWS.append(dict(model=model, level=lv, factor=SWEEP_DESIGN["levels"][lv], n=len(rc), recall_M2=mu[k],
                                            ci_lo=lo_, ci_hi=hi_))
    # ---------------------------------------------------------------- contact dose-response (H13)
    w = designed_wide(WDR_RES, MOVERS_WDR, model)
    if len(w):
        WDR_WIDE[model] = w
        gate_ok = GATE_PASS.get((model, "WDR"), False)
        for lv in WDR_LEVELS:
            w[f"ok_{lv}"] = ((w.get(f"gt_n_eroded_{lv}", 0) >= PREREG["min_object_pixels"])
                             & (w.get(f"gt_parallax_med_{lv}", 0) >= PREREG["parallax_min_px"]) & np.isfinite(w.get(f"rho_{lv}", np.nan)))
        cmin = PREREG["h13_contact_px_min"]
        w["contact_G"] = (w.get("contact_px0_G", 0) >= cmin) & (w.get("contact_px1_G", 0) >= cmin) & (w.get("gt_visible_frac_G", 0) >= 0.85)
        for lv in WDR_LEVELS[1:]:
            w[f"hidden_{lv}"] = ((w.get(f"has_wall_{lv}", 0) == 1) & (w.get(f"contact_px0_{lv}", 1) == 0)
                                 & (w.get(f"contact_px1_{lv}", 1) == 0))
        s13 = w[w["contact_G"] & w[f"ok_{_lo13}"] & w[f"ok_{_hi13}"] & w[f"hidden_{_hi13}"]]
        b, dr, n, nc = multi_slopes(s13["x_par"], [s13[f"rho_{_hi13}"], s13[f"rho_{_lo13}"]], s13["scene"], seed_for("wdr-h13", model))
        est, lo, hi = contrast_ci(b, dr, [1, -1])
        p = _p_one_sided(dr @ np.array([1.0, -1.0])) if len(dr) else np.nan
        v = ("NOT INTERPRETABLE (gate)" if not gate_ok else "NOT TESTABLE" if not _ok_dose(n, nc) else verdict_positive(lo, hi))
        WDR_TEST_ROWS.append(dict(model=model, test="H13", contrast=f"beta({WALL_DOSE_DESIGN['labels'][_hi13]} wall) - beta(no wall)",
                                  registered=True, n=n, n_scenes=nc, estimate=est, ci_lo=lo, ci_hi=hi, p_boot=p, verdict=v,
                                  beta_hi=b[0], beta_lo=b[1]))
        # exploratory: the whole curve on movers valid at every height whose contact every wall hides
        sa = w[w["contact_G"] & np.all([w[f"ok_{lv}"] for lv in WDR_LEVELS], axis=0)
               & np.all([w[f"hidden_{lv}"] for lv in WDR_LEVELS[1:]], axis=0)]
        b, dr, n, nc = multi_slopes(sa["x_par"], [sa[f"rho_{lv}"] for lv in WDR_LEVELS], sa["scene"], seed_for("wdr-all", model))
        for k, lv in enumerate(WDR_LEVELS):
            lo_, hi_ = _ci(dr[:, k]) if len(dr) else (np.nan, np.nan)
            WDR_BETA_ROWS.append(dict(model=model, level=lv, cover_m=WALL_DOSE_DESIGN["levels"][lv], n=n, n_scenes=nc, beta=b[k],
                                      ci_lo=lo_, ci_hi=hi_, median_rho_mono=float(sa[f"rho_mono_{lv}"].median()) if len(sa) else np.nan,
                                      median_visible_px=float(sa[f"gt_n_eroded_{lv}"].median()) if len(sa) else np.nan,
                                      median_hidden_share=float((1.0 - sa[f"gt_visible_frac_{lv}"]).median()) if len(sa) else np.nan))
        i15, i70, i0 = WDR_LEVELS.index("W15"), WDR_LEVELS.index("W70"), WDR_LEVELS.index("G")
        est, lo_, hi_ = contrast_ci(b, dr, [1.0 if k == i70 else -1.0 if k == i15 else 0.0 for k in range(len(WDR_LEVELS))])
        WDR_TEST_ROWS.append(dict(model=model, test="dose 70-15", contrast="beta(70 cm) - beta(15 cm)", registered=False, n=n, n_scenes=nc,
                                  estimate=est, ci_lo=lo_, ci_hi=hi_, p_boot=np.nan,
                                  verdict="EXPLORATORY" if gate_ok else "NOT INTERPRETABLE (gate)"))
        e70, lo70, hi70 = contrast_ci(b, dr, [1.0 if k == i70 else -1.0 if k == i0 else 0.0 for k in range(len(WDR_LEVELS))])
        WDR_TEST_ROWS.append(dict(model=model, test="dose 70-0", contrast="beta(70 cm) - beta(no wall)", registered=False, n=n,
                                  n_scenes=nc, estimate=e70, ci_lo=lo70, ci_hi=hi70, p_boot=np.nan,
                                  verdict="EXPLORATORY" if gate_ok else "NOT INTERPRETABLE (gate)"))
        # the share of the 70 cm effect reached at 15 cm is a ratio: it is reported only where the 70 cm effect itself is
        # clearly positive (its interval excludes 0); otherwise the denominator can be ~0 and the ratio means nothing
        share, lo_, hi_ = np.nan, np.nan, np.nan
        if np.isfinite(lo70) and lo70 > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                share = float((b[i15] - b[i0]) / (b[i70] - b[i0]))
                sdr = (dr[:, i15] - dr[:, i0]) / (dr[:, i70] - dr[:, i0]) if len(dr) else np.zeros(0)
            lo_, hi_ = _ci(sdr[np.isfinite(sdr)]) if np.isfinite(sdr).sum() else (np.nan, np.nan)
        WDR_TEST_ROWS.append(dict(model=model, test="dose share", contrast="(beta 15 cm - beta no wall) / (beta 70 cm - beta no wall)",
                                  registered=False, n=n, n_scenes=nc, estimate=share, ci_lo=lo_, ci_hi=hi_, p_boot=np.nan,
                                  verdict=("NOT INTERPRETABLE (gate)" if not gate_ok else "EXPLORATORY" if np.isfinite(share)
                                           else "NOT ESTIMABLE (no clear 70 cm effect)")))
        rc = sa.dropna(subset=[f"recall_M2_exact_{lv}" for lv in WDR_LEVELS])
        if _ok_dose(len(rc), rc["scene"].nunique()):
            mu, mdr = _mean_ci(rc, [f"recall_M2_exact_{lv}" for lv in WDR_LEVELS], seed_for("wdr-m2", model))
            for k, lv in enumerate(WDR_LEVELS):
                lo_, hi_ = _ci(mdr[:, k])
                WDR_RECALL_ROWS.append(dict(model=model, level=lv, cover_m=WALL_DOSE_DESIGN["levels"][lv], n=len(rc), recall_M2=mu[k],
                                            ci_lo=lo_, ci_hi=hi_))
BSW_BETA, BSW_TESTS, BSW_RECALL = pd.DataFrame(BSW_BETA_ROWS), pd.DataFrame(BSW_TEST_ROWS), pd.DataFrame(BSW_RECALL_ROWS)
WDR_BETA, WDR_TESTS, WDR_RECALL = pd.DataFrame(WDR_BETA_ROWS), pd.DataFrame(WDR_TEST_ROWS), pd.DataFrame(WDR_RECALL_ROWS)
for _df, _t, _models in ((BSW_TESTS, "H12", PREREG["h12_models"]), (WDR_TESTS, "H13", None)):
    if len(_df):
        _m = (_df["test"] == _t) & (_df["model"].isin(_models) if _models else True) & _df["verdict"].isin(
            ["SUPPORTED", "NOT SUPPORTED", "CONTRADICTED"])
        _df["p_holm"] = np.nan
        _df.loc[_m, "p_holm"] = holm(_df.loc[_m, "p_boot"].values)

_f3 = lambda v: f"{v:.3f}"
with pd.option_context("display.width", 250, "display.max_colwidth", 60):
    if len(BSW_BETA):
        print("Baseline sweep: beta_s at each baseline (common sample: measurable at every baseline, identifiable at x1):")
        print(BSW_BETA[BSW_BETA["support"] == "all"].pivot_table(index="model", columns="level", values="beta")[BSW_LEVELS]
              .rename(columns=SWEEP_DESIGN["labels"]).to_string(float_format=_f3))
        _pp = BSW_BETA[BSW_BETA["support"] == "all"].groupby("level")["median_parallax"].median().reindex(BSW_LEVELS)
        print("median parallax of the sample (px): " + ", ".join(f"{SWEEP_DESIGN['labels'][k]} {v:.1f}" for k, v in _pp.items()))
        print("\nH12 (registered for DUSt3R and MASt3R) and the exploratory split by support:")
        print(BSW_TESTS[["model", "test", "n", "n_scenes", "estimate", "ci_lo", "ci_hi", "p_holm", "verdict"]]
              .to_string(index=False, na_rep="–", float_format=_f3))
    if len(BSW_RECALL):
        print("\nM2 recall (exact correspondences) at each baseline, same movers:")
        print(BSW_RECALL.pivot_table(index="model", columns="level", values="recall_M2")[BSW_LEVELS]
              .rename(columns=SWEEP_DESIGN["labels"]).to_string(float_format=_f3))
    if len(WDR_BETA):
        print("\nContact dose-response: beta_s at each wall height (common sample of all four heights):")
        print(WDR_BETA.pivot_table(index="model", columns="level", values="beta")[WDR_LEVELS]
              .rename(columns=WALL_DOSE_DESIGN["labels"]).to_string(float_format=_f3))
        _hs = WDR_BETA.groupby("level")["median_hidden_share"].median().reindex(WDR_LEVELS)
        print("median share of the mover hidden in view 1 (same sample): "
              + ", ".join(f"{WALL_DOSE_DESIGN['labels'][k]} {v:.2f}" for k, v in _hs.items()))
    if len(WDR_TESTS):
        print("\nH13 (registered, every model) and the shape of the curve (exploratory):")
        print(WDR_TESTS[["model", "test", "n", "n_scenes", "estimate", "ci_lo", "ci_hi", "p_holm", "verdict"]]
              .to_string(index=False, na_rep="–", float_format=_f3))
    if len(WDR_RECALL):
        print("\nM2 recall at each wall height, same movers:")
        print(WDR_RECALL.pivot_table(index="model", columns="level", values="recall_M2")[WDR_LEVELS]
              .rename(columns=WALL_DOSE_DESIGN["labels"]).to_string(float_format=_f3))
for _n, _df in (("BSW_beta", BSW_BETA), ("BSW_tests", BSW_TESTS), ("BSW_recall", BSW_RECALL), ("WDR_beta", WDR_BETA),
                ("WDR_tests", WDR_TESTS), ("WDR_recall", WDR_RECALL)):
    _df.to_csv(os.path.join(RESULTS_DIR, f"{_n}.csv"), index=False)

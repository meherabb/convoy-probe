# PLACEHOLDER
# 6.11 Two channels: the cue model (v6.5, exploratory, D57 and D59). Per model: fit on CONVOY-3D and E5, cross-validation
#      against the constant-weight and support-free models, out-of-sample slope predictions for every other testbed, the
#      hump, MonST3R against DUSt3R, the mask trade-off, and CONVOY-REAL by parallax.
_SUPPORTS = ("contact in view", "contact hidden", "unsupported")
_E5_SUP = {"G": "contact in view", "NS": "contact in view", "CH": "contact hidden", "F": "unsupported", "FH": "unsupported"}


def _phi(d, kappa):
    d = np.asarray(d, float)
    return np.ones_like(d) if kappa <= 0 else d * d / (d * d + kappa * kappa)


def _frame(x, rho, rho_mono, d, support, cl, **extra):
    df = pd.DataFrame(dict(x=np.asarray(x, float), rho=np.asarray(rho, float), rho_mono=np.asarray(rho_mono, float),
                           d=np.asarray(d, float), support=np.asarray(support, object), cl=np.asarray(cl, object), **extra))
    df = df[np.isfinite(df[["x", "rho", "rho_mono", "d"]]).all(1)].copy()
    df["y"], df["z"] = df["rho"] - df["rho_mono"], df["x"] - df["rho_mono"]
    return df


def cue_fit(df, per_support=True, kappas=None):
    # least squares of y = c_support * phi(d; kappa) * z; kappa on the grid, c in closed form for each kappa
    kappas = CUE_DESIGN["kappa_grid_px"] if kappas is None else kappas
    y, z, d = df["y"].to_numpy(float), df["z"].to_numpy(float), df["d"].to_numpy(float)
    groups = df["support"].to_numpy(object) if per_support else np.full(len(df), "all", object)
    best = None
    for k in kappas:
        a = _phi(d, k) * z
        c, sse = {}, 0.0
        for g in np.unique(groups):
            m = groups == g
            den = float(np.sum(a[m] ** 2))
            c[g] = float(np.sum(a[m] * y[m]) / den) if den > 0 else 0.0
            sse += float(np.sum((y[m] - c[g] * a[m]) ** 2))
        if best is None or sse < best["sse"]:
            best = dict(kappa=float(k), c=c, sse=sse, n=len(df), per_support=per_support)
    return best


def cue_predict(fit, df):
    c = df["support"].map(fit["c"]).to_numpy(float) if fit["per_support"] else np.full(len(df), fit["c"]["all"])
    return df["rho_mono"].to_numpy(float) + c * _phi(df["d"], fit["kappa"]) * df["z"].to_numpy(float)


def cue_training(model):
    # CONVOY-3D parallel movers (measurable; parallax not required) and the five E5 conditions (the E5 sample rules)
    parts = []
    d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN")]
    d = d[(d["motion_type"] == "parallel") & (d["s"] < 1) & d["measurable"].astype(bool)]
    parts.append(_frame(d["x_par"], d["rho"], d["rho_mono"], d["gt_parallax_med"],
                        np.where(d["cls"] == "crate_floating", "unsupported", "contact in view"), "MAIN-" + d["scene"].astype(str),
                        source="CONVOY-3D", uid=d["uid"].to_numpy()))
    w = E5_WIDE.get(model)
    if w is not None and len(w):
        for c in E5_CONDS:
            m = w[f"ok_{c}"] & (w["vis_ok"] if c in ("G", "NS", "F") else w["walled"])
            q = w[m]
            parts.append(_frame(q["x_par"], q[f"rho_{c}"], q[f"rho_mono_{c}"], q[f"gt_parallax_med_{c}"], [_E5_SUP[c]] * len(q),
                                "ABL-" + q["scene"].astype(str), source=f"E5 {c}", uid=q["uid"].to_numpy()))
    return pd.concat(parts, ignore_index=True)


def cue_tests(model):
    # samples on which the fitted model predicts the slope (each with its own testbed rules); x is x = -log(1-s) where s
    # was designed and x_geo in driving
    out = []
    if model in MODELS_OK["BSW"] and model in BSW_WIDE:
        w = BSW_WIDE[model]; q = w[w["h12_sample"]]
        sup = np.where(q["cls"] == "crate_floating", "unsupported", "contact in view")
        for lv in BSW_LEVELS:
            for lab, m in (("all", np.ones(len(q), bool)), ("grounded", sup == "contact in view"), ("floating", sup == "unsupported")):
                out.append((f"baseline sweep {SWEEP_DESIGN['labels'][lv]}", lab, "BSW", lv,
                            _frame(q["x_par"][m], q[f"rho_{lv}"][m], q[f"rho_mono_{lv}"][m], q[f"gt_parallax_med_{lv}"][m], sup[m],
                                   q["scene"][m].astype(str))))
    if model in MODELS_OK["WDR"] and model in WDR_WIDE:
        w = WDR_WIDE[model]
        q = w[w["contact_G"] & np.all([w[f"ok_{lv}"] for lv in WDR_LEVELS], axis=0)
              & np.all([w[f"hidden_{lv}"] for lv in WDR_LEVELS[1:]], axis=0)]
        for lv in WDR_LEVELS:
            out.append((f"dose-response {WALL_DOSE_DESIGN['labels'][lv]}", "all", "WDR", lv,
                        _frame(q["x_par"], q[f"rho_{lv}"], q[f"rho_mono_{lv}"], q[f"gt_parallax_med_{lv}"],
                               ["contact in view" if lv == "G" else "contact hidden"] * len(q), q["scene"].astype(str))))
    if model in MODELS_OK["REAL"]:
        d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "REAL")]
        d = d[e1_mask(d)]
        out.append(("CONVOY-REAL", "all", "REAL", "", _frame(d["x_par"], d["rho"], d["rho_mono"], d["gt_parallax_med"],
                                                             ["unsupported"] * len(d), d["scene"].astype(str))))
    if model in MODELS_OK["RABL"] and model in RABL_WIDE:
        w = RABL_WIDE[model]; q = w[w["sample"]]
        for c, sup in (("G", "contact in view"), ("F", "unsupported")):
            out.append((f"E5-REAL {RABL_DESIGN['labels'][c]}", "all", "RABL", c,
                        _frame(q["x_par"], q[f"rho_{c}"], q[f"rho_mono_{c}"], q[f"gt_parallax_med_{c}"], [sup] * len(q),
                               q["scene"].astype(str))))
    for set_, name in (("VK", "Virtual KITTI 2"), ("AV2", "Argoverse 2")):
        if model not in MODELS_OK.get(set_, []):
            continue
        d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == set_)]
        if not len(d) or "gt_contact" not in d:
            continue
        big = (d["gt_n_eroded"] >= PREREG["x6_min_eroded_px"]) if set_ == "VK" else d["measurable"].astype(bool)
        lo_x, hi_x = PREREG["h9_x_range"]
        d = d[big & (d["regime"] == "parallel") & (d["gt_parallax_med"] >= PREREG["parallax_min_px"])
              & d["gt_contact"].isin(["visible", "hidden"]) & d["gt_x_geo"].between(lo_x, hi_x)]
        for cont, sup in (("visible", "contact in view"), ("hidden", "contact hidden")):
            q = d[d["gt_contact"] == cont]
            out.append((f"{name}, contact {cont}", "all", set_, cont,
                        _frame(q["gt_x_geo"], q["rho"], q["rho_mono"], q["gt_parallax_med"], [sup] * len(q),
                               q["seq"].astype(str) + "/" + q["track"].astype(str))))
    return out


def _fold_ids(cl, n_folds, seed):
    u = np.unique(cl)
    perm = np.random.default_rng(seed).permutation(len(u))
    return dict(zip(u[perm], np.arange(len(u)) % n_folds))


CUE_FIT_ROWS, CUE_CV_ROWS, CUE_PRED_ROWS, CUE_HUMP_ROWS, CUE_FITS, CUE_TRAIN = [], [], [], [], {}, {}
for model in EVALUATED_MODELS:
    if model not in MODELS_OK["MAIN"] or model not in MODELS_OK["ABL"]:
        continue
    tr = cue_training(model)
    if len(tr) < 40 or tr["cl"].nunique() < 10:
        continue
    CUE_TRAIN[model] = tr
    fits = {"two channels": cue_fit(tr), "contact only (kappa = 0)": cue_fit(tr, kappas=[0.0]),
            "parallax only (one c)": cue_fit(tr, per_support=False)}
    CUE_FITS[model] = fits["two channels"]
    for name, f_ in fits.items():
        CUE_FIT_ROWS.append(dict(model=model, fit=name, kappa_px=f_["kappa"], n=f_["n"], n_clusters=int(tr["cl"].nunique()),
                                 rmse=float(np.sqrt(f_["sse"] / max(f_["n"], 1))),
                                 **{f"c_{s.replace(' ', '_')}": f_["c"].get(s, f_["c"].get("all", np.nan)) for s in _SUPPORTS}))
    # 10-fold cross-validation over scenes: squared error of every held-out object under each model
    fold = tr["cl"].map(_fold_ids(tr["cl"].to_numpy(), CUE_DESIGN["n_folds"], seed_for("cue-folds", model))).to_numpy()
    se = {k: np.full(len(tr), np.nan) for k in fits}
    for f_id in range(CUE_DESIGN["n_folds"]):
        trn, tst = tr[fold != f_id], tr[fold == f_id]
        if not len(tst) or len(trn) < 20:
            continue
        for name, per_sup, ks in (("two channels", True, None), ("contact only (kappa = 0)", True, [0.0]),
                                  ("parallax only (one c)", False, None)):
            fit_k = cue_fit(trn, per_support=per_sup, kappas=ks)
            se[name][fold == f_id] = (tst["rho"].to_numpy(float) - cue_predict(fit_k, tst)) ** 2
    okcv = np.all([np.isfinite(v) for v in se.values()], axis=0)
    bs = ClusterBootstrap(tr["cl"].to_numpy()[okcv], seed=seed_for("cue-cv", model))
    for other in ("contact only (kappa = 0)", "parallax only (one c)"):
        diff = se[other][okcv] - se["two channels"][okcv]
        T, P = bs.linear([np.ones(int(okcv.sum())), diff])
        dr = T[:, 1] / T[:, 0]
        lo, hi = _ci(dr)
        CUE_CV_ROWS.append(dict(model=model, comparison=f"{other} minus two channels", n=int(okcv.sum()),
                                mse_two_channels=float(np.mean(se["two channels"][okcv])), mse_other=float(np.mean(se[other][okcv])),
                                delta_mse=float(P[1] / P[0]), ci_lo=lo, ci_hi=hi,
                                direction="two channels predict better" if lo > 0 else "worse" if hi < 0 else "no clear difference"))
    # in-sample slopes by support and condition (reference), then out-of-sample predictions
    fit = fits["two channels"]
    tr = tr.assign(rho_hat=cue_predict(fit, tr))
    for src, g in tr.groupby("source", sort=False):
        for sup, q in g.groupby("support"):
            r = slope_with_ci(q["x"], q["rho"], q["cl"], seed_for("cue-in", model, src, sup))
            CUE_PRED_ROWS.append(dict(model=model, testbed=src, group=sup, set="fit", level="", role="fitted", n=r["n"],
                                      n_clusters=r["n_clusters"], beta_observed=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                      beta_predicted=ols_fit(q["x"], q["rho_hat"])[0] if len(q) >= 8 else np.nan,
                                      median_parallax=float(q["d"].median())))
    for name, grp, set_, lv, q in cue_tests(model):
        if len(q) < 8 or q["cl"].nunique() < 5:
            continue
        r = slope_with_ci(q["x"], q["rho"], q["cl"], seed_for("cue-out", model, name, grp))
        CUE_PRED_ROWS.append(dict(model=model, testbed=name, group=grp, set=set_, level=lv, role="predicted", n=r["n"],
                                  n_clusters=r["n_clusters"], beta_observed=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                  beta_predicted=ols_fit(q["x"], cue_predict(fit, q))[0], median_parallax=float(q["d"].median())))
    # the hump: CONVOY-3D movers binned by x; observed and fitted medians (two channels and contact only)
    main = tr[tr["source"] == "CONVOY-3D"].assign(rho_c0=cue_predict(fits["contact only (kappa = 0)"], tr[tr["source"] == "CONVOY-3D"]))
    xb = CUE_DESIGN["x_bins"]
    for k in range(len(xb) - 1):
        q = main[(main["x"] >= xb[k]) & (main["x"] < xb[k + 1])]
        if len(q) >= 5:
            CUE_HUMP_ROWS.append(dict(model=model, x_lo=xb[k], x_hi=xb[k + 1], n=len(q), x_median=float(q["x"].median()),
                                      rho_observed=float(q["rho"].median()), rho_two_channels=float(q["rho_hat"].median()),
                                      rho_contact_only=float(q["rho_c0"].median()), parallax_median=float(q["d"].median())))
CUE_FIT_TABLE, CUE_CV, CUE_PRED, CUE_HUMP = (pd.DataFrame(CUE_FIT_ROWS), pd.DataFrame(CUE_CV_ROWS), pd.DataFrame(CUE_PRED_ROWS),
                                              pd.DataFrame(CUE_HUMP_ROWS))
# how much of the hump each fit reproduces (v6.6): the fall of the median rho from the bin where the observed median peaks
# to the bin of largest x, observed and as each fit has it
_fall = []
for _m, _g in (CUE_HUMP.groupby("model") if len(CUE_HUMP) else []):
    _g = _g.sort_values("x_median").reset_index(drop=True)
    if len(_g) >= 3:
        _k = int(_g["rho_observed"].idxmax())
        _fall.append(dict(model=_m, x_peak=float(_g["x_median"].iloc[_k]), x_last=float(_g["x_median"].iloc[-1]),
                          fall_observed=float(_g["rho_observed"].iloc[_k] - _g["rho_observed"].iloc[-1]),
                          fall_two_channels=float(_g["rho_two_channels"].iloc[_k] - _g["rho_two_channels"].iloc[-1]),
                          fall_contact_only=float(_g["rho_contact_only"].iloc[_k] - _g["rho_contact_only"].iloc[-1])))
CUE_HUMP_FALL = pd.DataFrame(_fall)
if len(CUE_PRED):
    _p = CUE_PRED[CUE_PRED["role"] == "predicted"]
    CUE_PRED["inside_ci"] = (CUE_PRED["beta_predicted"] >= CUE_PRED["ci_lo"]) & (CUE_PRED["beta_predicted"] <= CUE_PRED["ci_hi"])
    _pp = CUE_PRED[(CUE_PRED["role"] == "predicted") & CUE_PRED[["beta_observed", "beta_predicted"]].notna().all(1)]
    CUE_CALIBRATION = dict(n=len(_pp), pearson_r=float(np.corrcoef(_pp["beta_observed"], _pp["beta_predicted"])[0, 1]) if len(_pp) >= 3 else np.nan,
                           mean_abs_error=float(np.mean(np.abs(_pp["beta_observed"] - _pp["beta_predicted"]))) if len(_pp) else np.nan,
                           share_inside_ci=float(_pp["inside_ci"].mean()) if len(_pp) else np.nan)
else:
    CUE_CALIBRATION = dict(n=0, pearson_r=np.nan, mean_abs_error=np.nan, share_inside_ci=np.nan)

# ---- MonST3R against DUSt3R on the same movers --------------------------------------------------------------------
MONDUS_ROWS = []
if all(m in EVALUATED_MODELS for m in ("DUSt3R", "MonST3R")):
    _a = MOV_RES[(MOV_RES["set"] == "MAIN") & MOV_RES["model"].isin(["DUSt3R", "MonST3R"])]
    _a = _a[e1_mask(_a)].pivot_table(index=["uid", "scene"], columns="model", values=["rho"], aggfunc="first")
    _a.columns = [f"{a}_{b}" for a, b in _a.columns]
    _a = _a.reset_index().merge(MOVERS_MAIN[["uid", "x_par", "gt_parallax_med"]], on="uid", how="left")
    if all(m in MODELS_OK["MAIN"] for m in ("DUSt3R", "MonST3R")):
        for lab, sel in (("parallax > 40 px", _a["gt_parallax_med"] > 40), ("parallax 5-40 px", _a["gt_parallax_med"] <= 40)):
            q = _a[sel]
            b, dr, n, nc = multi_slopes(q["x_par"], [q["rho_MonST3R"], q["rho_DUSt3R"]], q["scene"], seed_for("mondus", lab))
            est, lo, hi = contrast_ci(b, dr, [1, -1])
            MONDUS_ROWS.append(dict(contrast=f"beta_s, CONVOY-3D movers with {lab}", n=n, n_scenes=nc, MonST3R=b[0], DUSt3R=b[1],
                                    difference=est, ci_lo=lo, ci_hi=hi, direction=_dir_label(lo, hi)))
    if all(m in MODELS_OK["ABL"] and m in E5_WIDE for m in ("DUSt3R", "MonST3R")):
        _wm, _wd = E5_WIDE["MonST3R"], E5_WIDE["DUSt3R"]
        _j = _wm.merge(_wd, on=["uid", "scene"], suffixes=("_m", "_d"))
        q = _j[_j["walled_m"] & _j["ok_G_m"] & _j["ok_CH_m"] & _j["ok_G_d"] & _j["ok_CH_d"] & _j["vis_ok_m"]]
        b, dr, n, nc = multi_slopes(q["x_par_m"], [q["rho_CH_m"], q["rho_G_m"], q["rho_CH_d"], q["rho_G_d"]], q["scene"],
                                    seed_for("mondus-h6"))
        est, lo, hi = contrast_ci(b, dr, [1, -1, -1, 1])
        MONDUS_ROWS.append(dict(contrast="H6 contrast beta(CH) - beta(G), E5 movers", n=n, n_scenes=nc, MonST3R=b[0] - b[1],
                                DUSt3R=b[2] - b[3], difference=est, ci_lo=lo, ci_hi=hi, direction=_dir_label(lo, hi)))
    if all(m in MODELS_OK["BSW"] and m in BSW_WIDE for m in ("DUSt3R", "MonST3R")):
        _j = BSW_WIDE["MonST3R"].merge(BSW_WIDE["DUSt3R"], on=["uid", "scene"], suffixes=("_m", "_d"))
        q = _j[_j["h12_sample_m"] & _j["h12_sample_d"]]
        b, dr, n, nc = multi_slopes(q["x_par_m"], [q[f"rho_{_hi12}_m"], q[f"rho_{_lo12}_m"], q[f"rho_{_hi12}_d"], q[f"rho_{_lo12}_d"]],
                                    q["scene"], seed_for("mondus-bsw"))
        est, lo, hi = contrast_ci(b, dr, [1, -1, -1, 1])
        MONDUS_ROWS.append(dict(contrast="rise of beta_s from x0.5 to x3, baseline sweep", n=n, n_scenes=nc, MonST3R=b[0] - b[1],
                                DUSt3R=b[2] - b[3], difference=est, ci_lo=lo, ci_hi=hi, direction=_dir_label(lo, hi)))
MONDUS = pd.DataFrame(MONDUS_ROWS)

# ---- masks that defeat themselves: change of M2 recall against change of beta_s ---------------------------------------
TRADEOFF_ROWS = []
for _src, _beta, _rec, _key, _ref, _lab in (
        ("E5", E5_BETA, E5_MASKS, "condition", "G", ABLATION_DESIGN["labels"]),
        ("baseline sweep", BSW_BETA[BSW_BETA["support"] == "all"] if len(BSW_BETA) else BSW_BETA, BSW_RECALL, "level", _lo12, SWEEP_DESIGN["labels"]),
        ("dose-response", WDR_BETA, WDR_RECALL, "level", "G", WALL_DOSE_DESIGN["labels"])):
    if not len(_beta) or not len(_rec):
        continue
    _rc = _rec.rename(columns={"recall_M2": "recall"})
    for model in _rec["model"].unique():
        b_ = _beta[_beta["model"] == model].set_index(_key)["beta"]
        r_ = _rc[_rc["model"] == model].set_index(_key)["recall"]
        if _ref not in b_ or _ref not in r_:
            continue
        for lv in r_.index:
            if lv == _ref or lv not in b_:
                continue
            TRADEOFF_ROWS.append(dict(source=_src, model=model, level=lv, label=_lab.get(lv, lv),
                                      delta_beta=float(b_[lv] - b_[_ref]), delta_recall=float(r_[lv] - r_[_ref])))
TRADEOFF = pd.DataFrame(TRADEOFF_ROWS)
if len(TRADEOFF) >= 5:
    from scipy.stats import spearmanr as _spr
    TRADEOFF_SUMMARY = dict(n=len(TRADEOFF), spearman=float(_spr(TRADEOFF["delta_beta"], TRADEOFF["delta_recall"])[0]),
                            share_opposite_sign=float(np.mean(np.sign(TRADEOFF["delta_beta"]) != np.sign(TRADEOFF["delta_recall"]))))
else:
    TRADEOFF_SUMMARY = dict(n=len(TRADEOFF), spearman=np.nan, share_opposite_sign=np.nan)

# ---- CONVOY-REAL against CONVOY-3D floating crates, within parallax bins (D57) ----------------------------------------
REALPAR_ROWS = []
_pb = [5, 20, 40, 80, 1e9]
for model in EVALUATED_MODELS:
    if model not in MODELS_OK["MAIN"] or model not in MODELS_OK["REAL"]:
        continue
    for tb, d in (("CONVOY-3D floating crates", MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN")]),
                  ("CONVOY-REAL", MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "REAL")])):
        d = d[e1_mask(d)]
        if tb.startswith("CONVOY-3D"):
            d = d[d["cls"] == "crate_floating"]
        REALPAR_ROWS.append(dict(model=model, testbed=tb, bin="all", n=len(d), median_parallax=float(d["gt_parallax_med"].median()) if len(d) else np.nan,
                                 **{k: v for k, v in slope_with_ci(d["x_par"], d["rho"], d["scene"], seed_for("realpar", model, tb)).items()
                                    if k in ("beta", "ci_lo", "ci_hi")}))
        for k in range(len(_pb) - 1):
            q = d[(d["gt_parallax_med"] >= _pb[k]) & (d["gt_parallax_med"] < _pb[k + 1])]
            r = slope_with_ci(q["x_par"], q["rho"], q["scene"], seed_for("realpar", model, tb, k)) if len(q) >= 12 else dict(beta=np.nan, ci_lo=np.nan, ci_hi=np.nan)
            REALPAR_ROWS.append(dict(model=model, testbed=tb, bin=f"{_pb[k]:g}-{_pb[k + 1]:g} px" if _pb[k + 1] < 1e8 else f">{_pb[k]:g} px",
                                     n=len(q), median_parallax=float(q["gt_parallax_med"].median()) if len(q) else np.nan,
                                     beta=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))
REAL_PARALLAX = pd.DataFrame(REALPAR_ROWS)

_f3 = lambda v: f"{v:.3f}"
with pd.option_context("display.width", 250, "display.max_colwidth", 70):
    if len(CUE_FIT_TABLE):
        print("Cue model fitted on CONVOY-3D and E5 (rho - rho_mono = w (x - rho_mono), w = c d^2 / (d^2 + kappa^2)):")
        print(CUE_FIT_TABLE.to_string(index=False, na_rep="–", float_format=_f3))
        print("\n10-fold cross-validation over scenes (difference of the mean squared error of held-out rho; > 0: the two-channel "
              "model predicts better):")
        print(CUE_CV.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.4f}"))
    if len(CUE_PRED):
        print("\nObserved and predicted slopes (fitted rows: CONVOY-3D and E5; predicted rows: no refitting):")
        print(CUE_PRED.drop(columns=["set", "level"]).to_string(index=False, na_rep="–", float_format=_f3))
        print(f"\npredicted vs observed over {CUE_CALIBRATION['n']} out-of-sample slopes: Pearson r = {CUE_CALIBRATION['pearson_r']:.2f}, "
              f"mean |error| = {CUE_CALIBRATION['mean_abs_error']:.3f}, prediction inside the observed 95% interval: "
              f"{100 * CUE_CALIBRATION['share_inside_ci']:.0f}%")
    if len(CUE_HUMP):
        print("\nThe hump: median rho of CONVOY-3D parallel movers by x, observed and fitted:")
        print(CUE_HUMP.to_string(index=False, float_format=_f3))
        if len(CUE_HUMP_FALL):
            print("fall of the median rho from the observed peak to the largest x (observed, two channels, contact only):")
            print(CUE_HUMP_FALL.to_string(index=False, float_format=_f3))
    if len(MONDUS):
        print("\nMonST3R minus DUSt3R on the same movers (95% cluster-bootstrap intervals):")
        print(MONDUS.to_string(index=False, na_rep="–", float_format=_f3))
    if len(TRADEOFF):
        print(f"\nMask trade-off: change of M2 recall against change of beta_s ({TRADEOFF_SUMMARY['n']} model x condition pairs): "
              f"Spearman {TRADEOFF_SUMMARY['spearman']:.2f}, opposite signs in {100 * TRADEOFF_SUMMARY['share_opposite_sign']:.0f}%")
    if len(REAL_PARALLAX):
        print("\nCONVOY-REAL and CONVOY-3D floating crates by parallax (beta_s, 95% CI; bins with >= 12 movers):")
        print(REAL_PARALLAX.to_string(index=False, na_rep="–", float_format=_f3))
for _n, _df in (("cue_fits", CUE_FIT_TABLE), ("cue_cv", CUE_CV), ("cue_predictions", CUE_PRED), ("cue_hump", CUE_HUMP),
                ("cue_hump_fall", CUE_HUMP_FALL),
                ("monst3r_vs_dust3r", MONDUS), ("mask_tradeoff", TRADEOFF), ("convoy_real_by_parallax", REAL_PARALLAX)):
    _df.to_csv(os.path.join(RESULTS_DIR, f"{_n}.csv"), index=False)

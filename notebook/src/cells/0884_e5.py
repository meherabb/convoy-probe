# PLACEHOLDER
# 6.7 E5 ground-contact ablation (H5-H8 registered; shadow and class contrasts exploratory).
E5_CONDS = ABLATION_DESIGN["conditions"]
_E5_VALS = ["rho", "rho_mono", "gt_n_eroded", "gt_parallax_med", "gt_visible_frac", "contact_px0", "contact_px1", "has_wall"]

def e5_wide(model):
    # one row per base mover, one column per (quantity, condition)
    d = ABL_RES[ABL_RES["model"] == model].copy()
    d["has_wall"] = d["has_wall"].astype(float)
    w = d.pivot_table(index=["uid", "scene"], columns="condition", values=_E5_VALS, aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    gt = MOVERS_ABL[MOVERS_ABL["condition"] == "G"][["uid", "cls", "s", "x_par"]]
    w = w.reset_index().merge(gt, on="uid", how="left")
    for c in E5_CONDS:
        w[f"ok_{c}"] = (w.get(f"gt_n_eroded_{c}", 0) >= PREREG["min_object_pixels"]) & \
                       (w.get(f"gt_parallax_med_{c}", 0) >= PREREG["parallax_min_px"]) & np.isfinite(w.get(f"rho_{c}", np.nan))
    w["vis_ok"] = np.all([w[f"gt_visible_frac_{c}"] >= 0.85 for c in ("G", "NS", "F")], axis=0)
    w["walled"] = ((w["has_wall_CH"] == 1) & (w["contact_px0_CH"] == 0) & (w["contact_px1_CH"] == 0)
                   & (w["contact_px0_G"] >= 5) & (w["contact_px1_G"] >= 5))
    return w

def _sample(w, conds, walled=False):
    m = w["vis_ok"] & np.all([w[f"ok_{c}"] for c in conds], axis=0)
    return w[m & w["walled"]] if walled else w[m]

E5_TESTS_SPEC = [  # (id, label, conditions, weights, needs wall, registered)
    ("H5", "beta(F) - beta(NS): lifting removes the contact", ["F", "NS"], [1, -1], False, True),
    ("H6", "beta(CH) - beta(G): hiding the contact", ["CH", "G"], [1, -1], True, True),
    ("H7", "(beta_CH - beta_G) - (beta_FH - beta_F): the wall acts through the contact", ["CH", "G", "FH", "F"], [1, -1, -1, 1], True, True),
    ("SH", "beta(NS) - beta(G): removing the contact shadow", ["NS", "G"], [1, -1], False, False),
]

def _group_slope_diff(x, y, is_a, clusters, seed):
    # slope in group a minus slope in group b (unpaired movers, clusters resampled jointly)
    x, y, a = np.asarray(x, float), np.asarray(y, float), np.asarray(is_a, bool)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, a, cl = x[ok], y[ok], a[ok], np.asarray(clusters)[ok]
    if a.sum() < 8 or (~a).sum() < 8 or len(np.unique(cl)) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    cols = []
    for g in (a, ~a):
        gf = g.astype(float)
        cols += [gf, gf * x, gf * x * x, gf * y, gf * x * y]
    T, P = ClusterBootstrap(cl, seed=seed).linear(cols)
    def sl(t, o):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (t[:, o] * t[:, o + 4] - t[:, o + 1] * t[:, o + 3]) / (t[:, o] * t[:, o + 2] - t[:, o + 1] ** 2)
    pa, pb = sl(P[None], 0)[0], sl(P[None], 5)[0]
    dd = sl(T, 0) - sl(T, 5); dd = dd[np.isfinite(dd)]
    lo, hi = np.percentile(dd, [2.5, 97.5]) if len(dd) > 50 else (np.nan, np.nan)
    return float(pa), float(pb), float(pa - pb), float(lo), float(hi)

E5_BETA_ROWS, E5_TEST_ROWS, E5_WIDE = [], [], {}
for model in EVALUATED_MODELS:
    w = e5_wide(model)
    E5_WIDE[model] = w
    gate_ok = GATE_PASS.get((model, "ABL"), False)
    # slopes per condition on the common sample (valid in all five conditions, wall built and contact hidden)
    s5 = _sample(w, E5_CONDS, walled=True)
    b5, d5, n5, nc5 = multi_slopes(s5["x_par"], [s5[f"rho_{c}"] for c in E5_CONDS], s5["scene"], seed_for("e5-all", model))
    for k, c in enumerate(E5_CONDS):
        lo, hi = _ci(d5[:, k]) if len(d5) else (np.nan, np.nan)
        E5_BETA_ROWS.append(dict(model=model, condition=c, label=ABLATION_DESIGN["labels"][c], n=n5, n_scenes=nc5,
                                 beta=b5[k], ci_lo=lo, ci_hi=hi, median_rho_mono=float(s5[f"rho_mono_{c}"].median()) if len(s5) else np.nan))
    for tid, lab, conds, wts, walled, registered in E5_TESTS_SPEC:
        sm_ = _sample(w, conds, walled)
        b, dr, n, nc = multi_slopes(sm_["x_par"], [sm_[f"rho_{c}"] for c in conds], sm_["scene"], seed_for("e5", model, tid))
        est, lo, hi = contrast_ci(b, dr, wts)
        p = float((1 + np.sum(dr @ np.asarray(wts, float) <= 0)) / (len(dr) + 1)) if len(dr) else np.nan
        v = verdict_positive(lo, hi) if gate_ok else "NOT INTERPRETABLE (gate)"
        E5_TEST_ROWS.append(dict(model=model, test=tid, contrast=lab, registered=registered, n=n, n_scenes=nc,
                                 **{f"beta_{c}": b[i] for i, c in enumerate(conds)}, estimate=est, ci_lo=lo, ci_hi=hi,
                                 p_boot=p, verdict=v if registered else ("EXPLORATORY" if gate_ok else v)))
    # H8: single-image depth of the same mover, floating vs grounded without shadow
    s8 = w[w["vis_ok"] & (w["gt_n_eroded_F"] >= PREREG["min_object_pixels"]) & (w["gt_n_eroded_NS"] >= PREREG["min_object_pixels"])
           & np.isfinite(w["rho_mono_F"]) & np.isfinite(w["rho_mono_NS"])]
    if len(s8) >= 8 and s8["scene"].nunique() >= 5:
        est, lo, hi, dr = boot_stat(s8, "scene", lambda c: float(np.median(c["rho_mono_F"]) - np.median(c["rho_mono_NS"])),
                                    seed_for("e5-h8", model), cols=["rho_mono_F", "rho_mono_NS"])
        p = float((1 + np.sum(dr <= 0)) / (len(dr) + 1))
    else:
        est = lo = hi = p = np.nan
    E5_TEST_ROWS.append(dict(model=model, test="H8", contrast="median rho_mono(F) - median rho_mono(NS): single-image depth of a "
                             "lifted object", registered=True, n=len(s8), n_scenes=int(s8["scene"].nunique()), estimate=est,
                             ci_lo=lo, ci_hi=hi, p_boot=p, verdict=verdict_positive(lo, hi) if gate_ok else "NOT INTERPRETABLE (gate)"))
    # exploratory: floating car vs floating crate (different movers, same scenes)
    sf = _sample(w, ["F"])
    ba, bb_, est, lo, hi = _group_slope_diff(sf["x_par"], sf["rho_F"], sf["cls"] == "car", sf["scene"], seed_for("e5-cls", model))
    E5_TEST_ROWS.append(dict(model=model, test="CL", contrast="beta(F, car) - beta(F, crate): floating car vs floating crate",
                             registered=False, n=len(sf), n_scenes=int(sf["scene"].nunique()), beta_car=ba, beta_crate=bb_,
                             estimate=est, ci_lo=lo, ci_hi=hi, p_boot=np.nan, verdict="EXPLORATORY" if gate_ok else "NOT INTERPRETABLE (gate)"))
E5_BETA = pd.DataFrame(E5_BETA_ROWS)
E5_TESTS = pd.DataFrame(E5_TEST_ROWS)
for _t in ["H5", "H6", "H7", "H8"]:
    _m = E5_TESTS["test"] == _t
    E5_TESTS.loc[_m, "p_holm"] = holm(E5_TESTS.loc[_m, "p_boot"].values)
print("E5: beta_s per condition (common sample: valid in all five conditions, wall built, contact hidden in CH):")
print(E5_BETA.pivot_table(index="model", columns="condition", values="beta")[E5_CONDS]
      .to_string(float_format=lambda v: f"{v:.3f}"), f"\n(n = {E5_BETA.groupby('model')['n'].first().to_dict()})")
print("\nE5 contrasts:")
with pd.option_context("display.max_colwidth", 50, "display.width", 250):
    print(E5_TESTS[["model", "test", "n", "n_scenes", "estimate", "ci_lo", "ci_hi", "p_holm", "verdict"]]
          .to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
E5_BETA.to_csv(os.path.join(RESULTS_DIR, "E5_beta_by_condition.csv"), index=False)
E5_TESTS.to_csv(os.path.join(RESULTS_DIR, "E5_contrasts.csv"), index=False)

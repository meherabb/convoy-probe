# PLACEHOLDER
# 6.7b E5-REAL: the ground-contact ablation on real backgrounds (H11a, H11b; registered in v6.1).
RABL_CONDS = RABL_DESIGN["conditions"]
_RABL_VALS = ["rho", "rho_mono", "gt_n_eroded", "gt_parallax_med", "gt_visible_frac", "contact_px0", "contact_px1"]

def rabl_wide(model):
    # one row per crate, one column per (quantity, condition)
    d = RABL_RES[RABL_RES["model"] == model]
    if not len(d):
        return pd.DataFrame()
    w = d.pivot_table(index=["uid", "scene"], columns="condition", values=_RABL_VALS, aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    gt = MOVERS_RABL[MOVERS_RABL["condition"] == "G"][["uid", "s", "x_par", "seq"]]
    w = w.reset_index().merge(gt, on="uid", how="left")
    for c in RABL_CONDS:
        w[f"ok_{c}"] = ((w[f"gt_n_eroded_{c}"] >= PREREG["min_object_pixels"]) & (w[f"gt_parallax_med_{c}"] >= PREREG["parallax_min_px"])
                        & (w[f"gt_visible_frac_{c}"] >= 0.85) & np.isfinite(w[f"rho_{c}"]))
    # the contact is visible in both views when grounded and absent when lifted (measured on the composites)
    w["contact_ok"] = ((w["contact_px0_G"] >= 5) & (w["contact_px1_G"] >= 5) & (w["contact_px0_F"] == 0) & (w["contact_px1_F"] == 0))
    w["sample"] = w["ok_G"] & w["ok_F"] & w["contact_ok"] & np.isfinite(w["x_par"])
    return w

RABL_BETA_ROWS, RABL_TEST_ROWS, RABL_WIDE = [], [], {}
# positive control on the same composites: classical triangulation with the true pose follows x exactly
_rb = BASELINES[BASELINES["set"] == "RABL"].merge(MOVERS_RABL[["uid", "condition", "x_par", "measurable", "identifiable"]],
                                                   on=["uid", "condition"], how="left") if len(MOVERS_RABL) else pd.DataFrame()
RABL_CONTROL = {}
for c in RABL_CONDS:
    if len(_rb):
        dc = _rb[(_rb["condition"] == c) & _rb["measurable"].astype(bool) & _rb["identifiable"].astype(bool)]
        RABL_CONTROL[c] = slope_with_ci(dc["x_par"], dc["rho_classical_exact"], dc["scene"], seed_for("rabl-control", c))
for model in EVALUATED_MODELS:
    if "RABL" not in EVAL_STAGES[model]:
        continue
    w = rabl_wide(model)
    if not len(w):
        continue
    RABL_WIDE[model] = w
    ok = interpretable(model, "RABL")
    s_ = w[w["sample"]]
    b, dr, n, nc = multi_slopes(s_["x_par"], [s_["rho_G"], s_["rho_F"]], s_["scene"], seed_for("rabl", model))
    for k, c in enumerate(RABL_CONDS):
        lo, hi = _ci(dr[:, k]) if len(dr) else (np.nan, np.nan)
        RABL_BETA_ROWS.append(dict(model=model, condition=c, label=RABL_DESIGN["labels"][c], n=n, n_pairs=nc, beta=b[k],
                                   ci_lo=lo, ci_hi=hi, median_rho_mono=float(s_[f"rho_mono_{c}"].median()) if len(s_) else np.nan))
    est, lo, hi = contrast_ci(b, dr, [-1, 1])
    p = float((1 + np.sum(dr @ np.array([-1.0, 1.0]) <= 0)) / (len(dr) + 1)) if len(dr) else np.nan
    if n < 8 or nc < 5:                                  # registered minimum sample (AMENDMENT_V6_1)
        lo = hi = p = np.nan
    RABL_TEST_ROWS.append(dict(model=model, test="H11a", contrast="beta(F) - beta(G): lifting the crate off the real support",
                               n=n, n_pairs=nc, beta_G=b[0], beta_F=b[1], estimate=est, ci_lo=lo, ci_hi=hi, p_boot=p,
                               verdict=verdict_positive(lo, hi) if ok else "NOT INTERPRETABLE (gate)"))
    s8 = w[w["ok_G"] & w["ok_F"] & w["contact_ok"] & np.isfinite(w["rho_mono_G"]) & np.isfinite(w["rho_mono_F"])]
    if len(s8) >= 8 and s8["scene"].nunique() >= 5:
        est, lo, hi, d8 = boot_stat(s8, "scene", lambda c_: float(np.median(c_["rho_mono_F"]) - np.median(c_["rho_mono_G"])),
                                    seed_for("rabl-h11b", model), cols=["rho_mono_F", "rho_mono_G"])
        p = float((1 + np.sum(d8 <= 0)) / (len(d8) + 1))
    else:
        est = lo = hi = p = np.nan
    RABL_TEST_ROWS.append(dict(model=model, test="H11b", contrast="median rho_mono(F) - median rho_mono(G): single-image depth "
                               "of the lifted crate", n=len(s8), n_pairs=int(s8["scene"].nunique()), estimate=est, ci_lo=lo,
                               ci_hi=hi, p_boot=p, verdict=verdict_positive(lo, hi) if ok else "NOT INTERPRETABLE (gate)"))
RABL_BETA = pd.DataFrame(RABL_BETA_ROWS)
RABL_TESTS = pd.DataFrame(RABL_TEST_ROWS)
for _t in ["H11a", "H11b"]:
    if len(RABL_TESTS):
        _m = RABL_TESTS["test"] == _t
        RABL_TESTS.loc[_m, "p_holm"] = holm(RABL_TESTS.loc[_m, "p_boot"].values)
if len(RABL_TESTS):
    print("Positive control (classical triangulation, true pose, same composites): "
          + "; ".join(f"{RABL_DESIGN['labels'][c]}: beta = {v['beta']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}], n = {v['n']}"
                      for c, v in RABL_CONTROL.items()))
    print("\nE5-REAL: beta_s per condition (same crates in both conditions):")
    print(RABL_BETA.pivot_table(index="model", columns="condition", values="beta")[RABL_CONDS]
          .to_string(float_format=lambda v: f"{v:.3f}"), f"\n(n = {RABL_BETA.groupby('model')['n'].first().to_dict()})")
    print("\nE5-REAL contrasts:")
    with pd.option_context("display.max_colwidth", 50, "display.width", 250):
        print(RABL_TESTS[["model", "test", "n", "n_pairs", "estimate", "ci_lo", "ci_hi", "p_holm", "verdict"]]
              .to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
    RABL_BETA.to_csv(os.path.join(RESULTS_DIR, "E5real_beta_by_condition.csv"), index=False)
    RABL_TESTS.to_csv(os.path.join(RESULTS_DIR, "E5real_contrasts.csv"), index=False)
else:
    print("E5-REAL is not part of this run (no real-background composites).")

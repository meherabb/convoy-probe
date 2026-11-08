# PLACEHOLDER
# 11. Tables: every table is written as CSV, JSON and a booktabs LaTeX fragment
#     (outputs/tables/T*.{csv,json,tex}) and printed here. Only what this run evaluated appears.
TABLE_MANIFEST = []

def _tex_escape(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("^", "\\^{}"), ("<", "$<$"), (">", "$>$"),
                 ("π³", "$\\pi^3$"), ("β", "$\\beta$"), ("ε", "$\\varepsilon$"), ("ρ", "$\\rho$"), ("≥", "$\\geq$"),
                 ("≤", "$\\leq$"), ("−", "-")):
        s = s.replace(a, b)
    return s

def _cell(v, nd):
    if v is None:
        return "--"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return "--" if not np.isfinite(v) else f"{v:.{nd}f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)

def ci_str(est, lo, hi, nd=2):
    if est is None or not np.isfinite(est):
        return "--"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{est:.{nd}f}"
    return f"{est:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"

def _cis(df, est, lo, hi, nd=2):
    return [ci_str(a, b, c, nd) for a, b, c in zip(df[est], df[lo], df[hi])]

def _finite(df, col):
    # paper tables list only estimates this run could compute (the result CSVs keep every row)
    return df[np.isfinite(pd.to_numeric(df[col], errors="coerce"))] if len(df) else df

_NI = "NOT INTERPRETABLE (gate)"

def _interp(df, col="verdict"):
    # rows of a model that fails the static-scene gate on the testbed are left out of the paper tables
    return df[df[col].astype(str) != _NI] if len(df) and col in df.columns else df

def _gate_note(tb):
    no = [MODEL_LABELS.get(m, m) for m in EVALUATED_MODELS if m not in MODELS_OK[tb] and (m, tb) in GATE_PASS]
    return f" Not interpreted because they fail the static-scene gate on {TESTBED_NAMES[tb]}: {', '.join(no)}." if no else ""

def _display_models(col):
    # paper names of the models (pi3 -> π³, VGGT-1B -> VGGT), also inside composite labels
    out = col.astype(str)
    for k, v in MODEL_LABELS.items():
        out = out.str.replace(rf"(?<![A-Za-z0-9]){re.escape(k)}(?![A-Za-z0-9])", v, regex=True)
    return out

def write_table(df, tid, caption, nd=2):
    df = df.reset_index(drop=True)
    if "model" in df.columns and len(df):
        df = df.assign(model=_display_models(df["model"]))
    if not len(df):
        print(f"\n{tid}: not written (no estimable rows in this run)")
        return
    base = os.path.join(TABLE_DIR, tid)
    df.to_csv(base + ".csv", index=False, na_rep="–")          # "–": not applicable
    df.to_json(base + ".json", orient="records", indent=1, force_ascii=False)
    cols = list(df.columns)
    lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{_tex_escape(caption)}}}",
             f"\\label{{tab:{tid}}}", "\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule",
             " & ".join(_tex_escape(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(_tex_escape(_cell(r[c], nd)) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(base + ".tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    TABLE_MANIFEST.append(dict(table=tid, caption=caption, rows=len(df), csv=base + ".csv", tex=base + ".tex"))
    print(f"\n{tid}: {caption}")
    with pd.option_context("display.max_colwidth", 70, "display.width", 250):
        print(df.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.{nd}f}"))

# T1 models and data
t1m = MODEL_TABLE[MODEL_TABLE["model"].isin(EVALUATED_MODELS)][["model", "family", "training", "checkpoint", "input",
                                                                 "precision", "params_M", "peak_gpu_GiB"]].copy()
if VK_STATUS["available"]:
    t1m["virtual_kitti_in_training"] = ["yes" if VK_TRAINING_EXPOSURE.get(m) else "no" for m in t1m["model"]]
write_table(t1m, "T1a_models", "Models evaluated: family, training data, checkpoint, input size, precision, size, peak GPU memory"
            + (", and whether Virtual KITTI is among the published training data." if VK_STATUS["available"] else "."))
_t1d = [dict(dataset="CONVOY-3D main (+ knockout)", role="synthetic, exact", items=len(META_MAIN), movers=len(MOVERS_MAIN),
             check="exact", used_in="E1-E3, A1"),
        dict(dataset="CONVOY-3D E4 twins", role="synthetic, exact", items=len(META_E4), movers=len(MOVERS_E4), check="exact", used_in="E4"),
        dict(dataset="CONVOY-3D ablation (5 conditions)", role="synthetic, exact", items=len(META_ABL),
             movers=int((MOVERS_ABL["condition"] == "G").sum()), check="exact", used_in="E5"),
        dict(dataset="CONVOY-3D baseline sweep (4 baselines)", role="synthetic, exact (v6.5)", items=len(META_BSW),
             movers=int(MOVERS_BSW["uid"].nunique()), check="exact", used_in="H12"),
        dict(dataset="CONVOY-3D dose-response (4 wall heights)", role="synthetic, exact (v6.5)", items=len(META_WDR),
             movers=int(MOVERS_WDR["uid"].nunique()), check="exact", used_in="H13"),
        dict(dataset="LEGACY-FLAT", role="negative control", items=len(META_FLAT), movers=0, check="not applicable", used_in="gate")]
_a3_excluded = []
for k, v in A3_VALIDATION.items():
    if not v["passed"]:
        _a3_excluded.append(f"TUM {k}"); continue
    _t1d.append(dict(dataset=f"TUM {k}", role=TUM_SEQUENCES[k]["role"], items=v["n_frames"],
                     movers=len(MOVERS_REAL[MOVERS_REAL["seq"] == k]) if len(MOVERS_REAL) and "seq" in MOVERS_REAL else 0,
                     check=f"A3 {'PASS' if v['passed'] else 'FAIL'} ({v['median_rel_depth_err']:.3f})",
                     used_in=("CONVOY-REAL" + (", E5-REAL" if len(MOVERS_RABL) and (MOVERS_RABL["seq"] == k).any() else ""))
                     if TUM_SEQUENCES[k]["role"] == "static" else "prevalence"))
if len(MOVERS_RABL):
    _t1d.append(dict(dataset="E5-REAL composites (2 conditions)", role="real background, inserted crate",
                     items=int(MOVERS_RABL["scene"].nunique()), movers=int((MOVERS_RABL["condition"] == "G").sum()),
                     check="A3 of the source sequence", used_in="E5-REAL"))
for k, v in VK_A3.items():
    if not v["passed"]:
        _a3_excluded.append(f"Virtual KITTI 2 {k}"); continue
    _t1d.append(dict(dataset=f"Virtual KITTI 2 {k}", role="photo-realistic, exact", items=v["n_frames"],
                     movers=int((MOVERS_VK["seq"] == k).sum()) if len(MOVERS_VK) else 0,
                     check=f"A3 {'PASS' if v['passed'] else 'FAIL'} ({v['static_warp_err']:.3f} / {v['vehicle_warp_err']:.3f})",
                     used_in="E6"))
for k, v in KT_A3.items():
    if not v["passed"]:
        _a3_excluded.append(f"KITTI tracking {k}"); continue
    _t1d.append(dict(dataset=f"KITTI tracking {k}", role="real driving, LiDAR (exploratory)", items=v["n_frames"],
                     movers=int((MOVERS_KT["seq"] == k).sum()) if len(MOVERS_KT) else 0,
                     check=f"A3 PASS ({v['warp_err_' + v['convention']]:.3f})", used_in="E6x"))
# Argoverse 2: one row for all logs here; the per-log record is T1c (v6.4: 64 logs would not fit a paper table)
_av2_a3_all = {**AV2_A3, **AV2X_A3}
_av2_used = {k: v for k, v in _av2_a3_all.items() if v["passed"] and v.get("n_pairs_used", 0)}
_n_av2_fail = sum(1 for v in _av2_a3_all.values() if not v["passed"])
if _av2_used:
    _sw = np.array([v["static_warp_err"] for v in _av2_used.values()], float)
    _t1d.append(dict(dataset=f"Argoverse 2 ({len(_av2_used)} validation logs)", role="real driving, LiDAR (exploratory)",
                     items=int(sum(v["n_pairs_used"] for v in _av2_used.values())), movers=len(MOVERS_AV2),
                     check=f"A3 PASS (median {np.median(_sw):.3f}, max {np.max(_sw):.3f})", used_in="E6x, 6.9"))
if _n_av2_fail:
    _a3_excluded.append(f"{_n_av2_fail} Argoverse 2 log(s)")
for k, v in BONN_A3.items():                               # v6.5: Bonn RGB-D Dynamic
    if not v["passed"]:
        _a3_excluded.append(f"Bonn {k}"); continue
    _t1d.append(dict(dataset=f"Bonn RGB-D Dynamic {k}", role="real objects, measured depth (exploratory)", items=v["n_frames"],
                     movers=int((MOVERS_BONN["seq"] == k).sum()) if len(MOVERS_BONN) else 0,
                     check=f"A3 PASS ({v['median_rel_depth_err']:.3f}, {v['convention']})", used_in="6.12"))
write_table(pd.DataFrame(_t1d), "T1b_data", "Data: testbeds, sizes, and the ground-truth self-consistency test (A3: median relative "
            "depth disagreement after warping" + ("; static / vehicles for Virtual KITTI 2)." if VK_SCENE_OBJS else ").")
            + (" Excluded because their ground truth failed A3: " + ", ".join(_a3_excluded) + "." if _a3_excluded else ""), nd=3)
if _av2_a3_all:
    _t1c = pd.DataFrame([dict(log=k[:8], stage="v6.3" if k in AV2_A3 else "v6.4", pairs_used=v.get("n_pairs_used", 0),
                              moving_vehicles=int((MOVERS_AV2["seq"] == k[:8]).sum()) if len(MOVERS_AV2) else 0,
                              static_warp=v["static_warp_err"], grey_static_warp=v["photo_static_warp"],
                              grey_static_nowarp=v["photo_static_nowarp"], A3="PASS" if v["passed"] else "FAIL")
                         for k, v in _av2_a3_all.items()])
    _vcs = "; ".join(f"{lab}: {vc.get('note', '–')}" for lab, vc in (("v6.3 logs", AV2_VEHICLE_CHECK), ("v6.4 logs", AV2X_VEHICLE_CHECK)) if vc)
    write_table(_t1c, "T1c_argoverse2_logs", "Argoverse 2 logs and their ground-truth checks (A3): median relative depth disagreement "
                "of warped static LiDAR, and the grey-value error (0-255) of static LiDAR pixels with the exact warp and without it. "
                "Moving vehicles, pooled over the logs of each stage: " + (_vcs or "–") + ".", nd=3)
# T2 gate and oracle
t2 = GATE.assign(slope=_cis(GATE, "slope_median", "slope_q25", "slope_q75"), absrel=_cis(GATE, "absrel_median", "absrel_q25", "absrel_q75"))
write_table(t2[["testbed", "model", "n_images", "slope", "absrel", "gate"]], "T2_validity_gate",
            "Static-scene validity gate: median [IQR] log-depth slope and AbsRel on static pixels.")
_t2b = ORACLE_TABLE.assign(beta_s_ci=_cis(ORACLE_TABLE, "beta_s", "ci_lo", "ci_hi", 3))[["w_true", "n", "beta_s_ci", "beta_theory_identity",
                                                                                     "median_w_recovered", "verdict_P1"]]
if len(ORACLE_VK):
    _t2b = _t2b.merge(ORACLE_VK.assign(beta_geo_vkitti=_cis(ORACLE_VK, "beta_geo", "ci_lo", "ci_hi", 3))[["w_true", "beta_geo_vkitti",
                                                                                                       "beta_theory_identity"]]
                      .rename(columns={"beta_theory_identity": "identity_vkitti"}), on="w_true", how="left")
write_table(_t2b, "T2b_oracle_validation", "Estimator validation with predictors of known prior weight w (CONVOY-3D E1 estimator"
            + ("; Virtual KITTI 2 H9 estimator)." if len(ORACLE_VK) else ")."), nd=3)
# T3 E1
t3 = _finite(_interp(E1_TABLE), "beta_s")
t3 = t3.assign(beta_s_ci=_cis(t3, "beta_s", "ci_lo", "ci_hi"), beta_mono_ci=_cis(t3, "beta_mono", "beta_mono_lo", "beta_mono_hi"),
               beta_geo_ci=_cis(t3, "beta_geo", "beta_geo_lo", "beta_geo_hi"))
t3 = pd.concat([t3, pd.DataFrame([dict(model=k, training="control", n=v["n"], n_clips=v["n_clusters"],
                                       beta_s_ci=ci_str(v["beta"], v["ci_lo"], v["ci_hi"]), r2=v["r2"], verdict="control")
                                  for k, v in CONTROL_E1.items()])], ignore_index=True)
write_table(t3[["model", "training", "n", "n_clips", "beta_s_ci", "r2", "beta_mono_ci", "beta_geo_ci", "w_eff", "p_holm", "verdict"]],
            "T3_E1_geometry_share", "E1: geometry share beta_s (slope of log depth error on -log(1-s)) for purely parallel movers, "
            "CONVOY-3D; verdict under the locked Prediction 1." + _gate_note("MAIN"))
# T4 E2
t4 = _finite(_interp(E2_TABLE), "diff"); t4 = t4.assign(diff_ci=_cis(t4, "diff", "ci_lo", "ci_hi"))
write_table(t4[["model", "method", "n_par", "n_perp", "rec_par", "rec_perp", "diff_ci", "auroc_par", "auroc_perp", "verdict"]],
            "T4_E2_masks", "E2 (Prediction 2): dynamic-mask recall at 5% static FPR, parallel vs perpendicular movers (stratified by "
            "|s| and size)." + _gate_note("MAIN"))
# T5 E3
t5 = _interp(E3_TABLE, "verdict_P3a")
t5 = t5.assign(spearman_ci=_cis(t5, "spearman_w_eps", "sp_lo", "sp_hi"), class_diff_ci=_cis(t5, "class_diff", "cd_lo", "cd_hi"))
write_table(t5[["model", "n_w", "median_w_parallel", "median_w_perp", "spearman_ci", "verdict_P3a", "w_car", "w_crate",
                "w_crate_floating", "class_diff_ci", "verdict_P3b", "lock_condition_c"]],
            "T5_E3_prior_weight", "E3 (Prediction 3): prior weight w versus epipolar offset and object class." + _gate_note("MAIN"))
# T6 E4
if len(_finite(_interp(E4_SUMMARY), "drop_K8")):
    t6 = _finite(_interp(E4_SUMMARY), "drop_K8"); t6 = t6.assign(drop_ci=_cis(t6, "drop_K8", "drop_lo", "drop_hi"))
    write_table(t6[[c for c in t6.columns if c.startswith(("model", "straight", "curved"))] + ["n_K8", "drop_ci", "verdict"]],
                "T6_E4_multiview", "E4 (Prediction 4): beta_s for K = 2, 4, 8 frames on straight paths and their curved twins."
                + _gate_note("MAIN"))
# T7 A1
t7 = _finite(_interp(A1_TABLE, "A1"), "drop")
t7 = t7.assign(full=_cis(t7, "beta_full", "full_lo", "full_hi"), knockout=_cis(t7, "beta_knockout", "ko_lo", "ko_hi"),
               drop_ci=_cis(t7, "drop", "drop_lo", "drop_hi"))
write_table(t7[["model", "n", "n_clips", "full", "knockout", "drop_ci", "static_sensitivity", "A1"]], "T7_A1_knockout",
            "A1 correspondence knockout: beta_s with the mover present in / deleted from the second view." + _gate_note("MAIN"))
# T8 real replication
t8 = _finite(_interp(REAL_E1), "beta_s"); t8 = t8.assign(beta_s_ci=_cis(t8, "beta_s", "ci_lo", "ci_hi"))
write_table(t8[["model", "n", "n_pairs", "beta_s_ci", "r2", "beta_mono", "verdict"]], "T8_real_E1",
            "CONVOY-REAL: E1 replication on real TUM backgrounds." + _gate_note("REAL"))
if len(_finite(_interp(REAL_E2), "diff")):
    t8b = _finite(_interp(REAL_E2), "diff"); t8b = t8b.assign(diff_ci=_cis(t8b, "diff", "ci_lo", "ci_hi"))
    write_table(t8b[["model", "method", "n_par", "n_perp", "rec_par", "rec_perp", "diff_ci", "verdict"]], "T8b_real_E2",
                "CONVOY-REAL: E2 replication on real TUM backgrounds." + _gate_note("REAL"))
# T9 E5 ablation
_e5b = _finite(E5_BETA[E5_BETA["model"].isin(MODELS_OK["ABL"])], "beta") if len(E5_BETA) else E5_BETA
if len(_e5b):
    t9 = _e5b.assign(beta_ci=_cis(_e5b, "beta", "ci_lo", "ci_hi")).pivot_table(index="model", columns="condition",
                                                                                values="beta_ci", aggfunc="first")
    t9 = t9[[c for c in E5_CONDS if c in t9.columns]].reset_index()
    t9.columns = ["model"] + [ABLATION_DESIGN["labels"][c] for c in t9.columns[1:]]
    t9["n"] = [int(_e5b[_e5b["model"] == m]["n"].iloc[0]) for m in t9["model"]]
    write_table(t9, "T9a_E5_beta_by_condition", "E5: geometry share beta_s of the same movers in each ground-contact condition."
                + _gate_note("ABL"))
    t9b = _finite(_interp(E5_TESTS), "estimate"); t9b = t9b.assign(estimate_ci=_cis(t9b, "estimate", "ci_lo", "ci_hi", 3))
    write_table(t9b[["model", "test", "contrast", "n", "n_scenes", "estimate_ci", "p_holm", "verdict"]], "T9b_E5_contrasts",
                "E5: registered contrasts H5-H8 and exploratory contrasts (shadow, class), paired by mover, 95% cluster-bootstrap "
                "intervals." + _gate_note("ABL"), nd=3)
# T10 E6 driving
_e6t = _interp(E6_TABLE[E6_TABLE["verdict"] != "NOT TESTABLE"]) if len(E6_TABLE) else E6_TABLE
if len(_finite(_e6t, "beta_geo")) and (_e6t["verdict"] != "control").any():
    t10 = _finite(_e6t, "beta_geo")
    t10 = t10.assign(beta_geo_ci=_cis(t10, "beta_geo", "ci_lo", "ci_hi"))
    write_table(t10[["model", "trained_on_vkitti", "n", "n_pairs", "beta_geo_ci", "r2", "beta_mono", "w_eff", "verdict"]],
                "T10a_E6_driving", "E6 (H9, exploratory in v6.1: the sample rule was changed after the v6.0 count, D34): "
                "geometry share beta_geo for moving vehicles in Virtual KITTI 2 (parallel regime)." + _gate_note("VK"))
if len(VK_ATTRITION):
    write_table(VK_ATTRITION, "T10d_vkitti_sample", "Virtual KITTI 2: the H9 sample step by step (vehicles and image pairs left "
                "after each criterion); the last row applies the v6.0 rule for comparison.", nd=0)
if len(PREVALENCE_ALL):
    write_table(PREVALENCE_ALL[["dataset", "seq", "n_pairs", "n_moving_pixels", "frac_parallel", "frac_intermediate",
                                "frac_perpendicular", "median_eps"]],
                "T10b_prevalence_two_view", "Share of moving pixels per motion regime between two views ("
                + "; ".join(({"TUM fr3 walking (people)": "TUM walking people", "Virtual KITTI 2 (vehicles)":
                              "Virtual KITTI 2 moving vehicles"}).get(x, x) for x in PREVALENCE_ALL["dataset"].unique()) + ").", nd=3)
if len(KVIEW_SUMMARY):
    write_table(KVIEW_SUMMARY, "T10c_prevalence_k_view", "Virtual KITTI 2: share of moving vehicles whose K-frame track a static "
                "point explains within 1 px.", nd=3)
# T11 E5-REAL: the contact ablation on real images
_rb = _finite(RABL_BETA[RABL_BETA["model"].isin(MODELS_OK["RABL"])], "beta") if len(RABL_BETA) else RABL_BETA
if len(_rb):
    t11 = _rb.assign(beta_ci=_cis(_rb, "beta", "ci_lo", "ci_hi")).pivot_table(index="model", columns="condition",
                                                                              values="beta_ci", aggfunc="first")
    t11 = t11[[c for c in RABL_CONDS if c in t11.columns]].reset_index()
    t11.columns = ["model"] + [RABL_DESIGN["labels"][c] for c in t11.columns[1:]]
    t11["n"] = [int(_rb[_rb["model"] == m]["n"].iloc[0]) for m in t11["model"]]
    t11["n_pairs"] = [int(_rb[_rb["model"] == m]["n_pairs"].iloc[0]) for m in t11["model"]]
    _ctl = "; ".join(f"{RABL_DESIGN['labels'][c]} {ci_str(v['beta'], v['ci_lo'], v['ci_hi'])}" for c, v in RABL_CONTROL.items())
    write_table(t11, "T11a_E5real_beta_by_condition", "E5-REAL: geometry share beta_s of the same crates on real TUM backgrounds, "
                "grounded with a contact shadow or lifted without one. Classical triangulation with the true pose on the same "
                f"composites: {_ctl}." + _gate_note("RABL"))
    t11b = _finite(_interp(RABL_TESTS), "estimate"); t11b = t11b.assign(estimate_ci=_cis(t11b, "estimate", "ci_lo", "ci_hi", 3))
    write_table(t11b[["model", "test", "n", "n_pairs", "estimate_ci", "p_holm", "verdict"]], "T11b_E5real_contrasts",
                "E5-REAL: H11a, beta(lifted) - beta(grounded), and H11b, median single-image rho(lifted) - rho(grounded), "
                "paired by crate, 95% cluster-bootstrap intervals (cluster = image pair), Holm across models." + _gate_note("RABL"),
                nd=3)
# T12 summary, T13 robustness
_t12 = SUMMARY[~SUMMARY["verdict"].isin(["NOT TESTABLE", _NI])]
_t12_note = "".join(_gate_note(tb) for tb in ["MAIN", "ABL", "REAL", "RABL", "VK"])
write_table(_t12.assign(estimate_ci=_cis(_t12, "estimate", "ci_lo", "ci_hi", 3))
            [["family", "prediction", "model", "statistic", "estimate_ci", "n", "verdict"]], "T12_summary",
            "All registered tests (locked P1-P4 and A1; v6 H5-H8; v6.1 H11a-b) and the exploratory H9: estimate [95% CI] and "
            "verdict. H10 was withdrawn after the v6 run (D33)." + _t12_note, nd=3)
for _v in ["NOT TESTABLE", _NI]:
    if (SUMMARY["verdict"] == _v).any():
        print(f"{_v.capitalize()} in this run (left out of T12, kept in results/prediction_summary.csv):",
              SUMMARY[SUMMARY["verdict"] == _v].groupby("prediction").size().to_dict())
_rob = ROBUST[ROBUST["model"].isin(MODELS_OK["MAIN"])] if len(ROBUST) else ROBUST
if len(_finite(_rob, "beta_s")):
    t13 = _finite(_rob, "beta_s"); t13 = t13.assign(beta_ci=_cis(t13, "beta_s", "ci_lo", "ci_hi"))
    write_table(t13.pivot_table(index=["factor", "level"], columns="model", values="beta_ci", aggfunc="first").reset_index()
                .rename(columns=lambda c: MODEL_LABELS.get(c, c)),
                "T13_robustness", "beta_s by camera heading, mover class, depth and focal length (E1 movers).")
# ---- v6.2 exploratory tables ---------------------------------------------------------------------
if len(E5_MASK_TESTS):
    _t14 = E5_MASKS.assign(recall_ci=_cis(E5_MASKS, "recall_M2", "ci_lo", "ci_hi", 3)).pivot_table(
        index="model", columns="condition", values="recall_ci", aggfunc="first")[E5_CONDS].reset_index()
    _t14.columns = ["model"] + [ABLATION_DESIGN["labels"][c] for c in E5_CONDS]
    write_table(_t14, "T14a_masks_by_condition", "Exploratory: M2 recall (exact correspondences, 5% static FPR) on the same "
                "purely parallel movers in each ground-contact condition of E5." + _gate_note("ABL"), nd=3)
    _t14b = E5_MASK_TESTS.assign(change_ci=_cis(E5_MASK_TESTS, "estimate", "ci_lo", "ci_hi", 3))
    write_table(_t14b[["model", "contrast", "meaning", "n", "n_scenes", "change_ci", "direction"]], "T14b_masks_contrasts",
                "Exploratory: paired change of M2 recall when the contact is hidden or the mover lifted (95% cluster-bootstrap "
                "intervals over base scenes).", nd=3)
if len(X6_CONTACT_TEST):
    _dsm = {"Virtual KITTI 2": "VK", "Argoverse 2": "AV2"}
    _x6c = X6_CONTACT[[m in MODELS_OK.get(_dsm.get(d, ""), []) for d, m in zip(X6_CONTACT["dataset"], X6_CONTACT["model"])]]
    _t15 = _x6c.assign(g_ci=_cis(_x6c, "median_g", "ci_lo", "ci_hi", 3)).pivot_table(
        index=["dataset", "model"], columns="contact", values="g_ci", aggfunc="first").reset_index()
    _t15 = _t15.merge(X6_CONTACT_TEST.assign(diff_ci=_cis(X6_CONTACT_TEST, "diff_hidden_minus_visible", "ci_lo", "ci_hi", 3))
                      [["dataset", "model", "n_visible", "n_hidden", "n_hidden_tracks", "diff_ci", "direction"]],
                      on=["dataset", "model"], how="left")
    if len(X6_MATCHED):
        _t15 = _t15.merge(X6_MATCHED.assign(matched_ci=_cis(X6_MATCHED, "median_paired_diff", "ci_lo", "ci_hi", 3))
                          [["dataset", "model", "n_matched", "matched_ci", "direction"]].rename(columns={"direction": "direction_matched"}),
                          on=["dataset", "model"], how="left")
    _t15 = (_t15.assign(_o=_t15["dataset"].map({"Virtual KITTI 2": 0, "Argoverse 2": 1}).fillna(9))
            .sort_values(["_o", "model"]).drop(columns="_o"))
    write_table(_t15[[c for c in ["dataset", "model", "n_visible", "visible", "n_hidden", "n_hidden_tracks", "hidden", "diff_ci",
                                  "direction", "n_matched", "matched_ci", "direction_matched"] if c in _t15]],
                "T15a_driving_contact", "Exploratory: median per-object geometry share g of parallel vehicles whose wheel contact "
                "points are visible or hidden (Virtual KITTI 2: instance masks, D37; Argoverse 2: geometry and LiDAR, D50), the "
                "difference, and the median paired difference after matching every hidden-contact vehicle to the nearest "
                "visible-contact vehicle on log depth, log image area and x_geo (within 0.5 SD on each). 95% cluster-bootstrap "
                "intervals over vehicle tracks." + _gate_note("VK") + (_gate_note("AV2") if AV2_STATUS["available"] else ""), nd=3)
    if len(X6_MATCHED):
        _bal = X6_MATCHED[["dataset", "model", "n_hidden", "n_matched"] + [c for c in X6_MATCHED if c.startswith("smd_")]]
        write_table(_bal, "T15c_matching_balance", "Covariate balance of the matched hidden/visible comparison: standardized "
                    "mean difference (hidden minus visible, in units of the pooled SD of the unmatched groups) of log depth, log "
                    "image area and x_geo before and after matching (|SMD| < 0.1 is conventionally taken as balanced).", nd=2)
if len(CONTACT_RULE_CHECK) and "agreement" in CONTACT_RULE_CHECK:
    write_table(CONTACT_RULE_CHECK, "T15d_contact_rule_check", "Check of the Argoverse 2 contact rule (D50) on Virtual KITTI 2: "
                "the same geometric rule, with the exact depth of static pixels in place of the LiDAR, against the labels from the "
                "instance masks (D37), on the vehicles where both rules decide; agreement, Cohen's kappa, the number of hidden "
                "contacts under each rule and under both, and the vehicles the instance masks decide but the rule leaves "
                "undetermined.", nd=3)
if len(X6_PACE):
    _t15b = X6_PACE.assign(g_ci=_cis(X6_PACE, "median_g", "ci_lo", "ci_hi", 3))
    write_table(_t15b[["dataset", "model", "n_s_below_1", "n_tracks", "median_x_geo", "g_ci", "share_closer_to_single_image",
                       "n_s_at_least_1", "share_placed_farther_than_single_image"]], "T15b_convoy_case",
                "Exploratory: vehicles keeping pace with the camera (0.85 <= s < 1.15). For s < 1, the median per-object geometry "
                "share g (95% cluster-bootstrap intervals over vehicle tracks) and the share of vehicles whose depth is closer to "
                "the model's own single-image depth than to the geometric answer; for s >= 1 (no static solution), the share "
                "placed farther than the single-image depth.", nd=3)
if len(REAL_PREVALENCE):
    write_table(REAL_PREVALENCE, "T16a_real_driving_regimes", "Exploratory, real driving with LiDAR: share of moving vehicles (and "
                "of their LiDAR pixels) per motion regime, and the share keeping pace with the camera (0.85 <= s < 1.15); the "
                "parallel threshold is the label-noise floor measured on parked vehicles. The ego speed (camera baseline over the "
                "time between the two images) describes the traffic sampled: shares measured in urban traffic need not hold at "
                "highway speeds.", nd=3)
if len(_finite(REAL_BETA, "beta_geo")):
    _t16 = _finite(REAL_BETA, "beta_geo"); _t16 = _t16.assign(beta_geo_ci=_cis(_t16, "beta_geo", "ci_lo", "ci_hi", 3))
    write_table(_t16[["dataset", "model", "n", "n_tracks", "beta_geo_ci", "r2"]], "T16b_real_driving_beta",
                "Exploratory, real driving with LiDAR: slope beta_geo of the log depth error on x_geo for parallel moving vehicles "
                "(correspondences from labelled boxes and tracks, Sections 3.10-3.11; 95% cluster-bootstrap intervals over "
                "vehicle tracks)." + "".join(_gate_note(k) for k in REAL_BETA["set"].unique()), nd=3)
if len(PROTOCOL_CHECK):
    _t17 = PROTOCOL_CHECK.copy()
    if "H11b_one" in _t17:
        _t17 = _t17.assign(H11b_one_image=_cis(_t17, "H11b_one", "H11b_one_lo", "H11b_one_hi", 3),
                           H11b_image_twice=_cis(_t17, "H11b_twice", "H11b_twice_lo", "H11b_twice_hi", 3))
    write_table(_t17[[c for c in ["model", "n_crates", "H11b_one_image", "H11b_image_twice", "median_abs_diff", "spearman",
                                  "static_absrel_one", "static_absrel_twice", "real_mono_slope_one", "real_mono_slope_twice",
                                  "protocol"] if c in _t17]], "T17_single_image_protocol",
                "Single-image protocol check for the multi-view models: H11b and the CONVOY-REAL single-image slope with the "
                "image given once (the protocol of the paper) or twice.", nd=3)
# ---- v6.4: the synthesis by ground support (Section 6.9, descriptive) --------------------------------------
if len(SUPPORT_LEVELS):
    _s18 = SUPPORT_LEVELS.assign(value=_cis(SUPPORT_LEVELS, "estimate", "ci_lo", "ci_hi", 2),
                                 row=SUPPORT_LEVELS["testbed"] + ": " + SUPPORT_LEVELS["condition"])
    _ord = list(dict.fromkeys(_s18["row"]))
    _t18 = (_s18.pivot_table(index=["row", "support", "statistic"], columns="model", values="value", aggfunc="first")
            .reset_index().set_index("row").reindex(_ord).reset_index().rename(columns=lambda c: MODEL_LABELS.get(c, c)))
    write_table(_t18.rename(columns={"row": "testbed: objects"}), "T18a_support_levels",
                "Descriptive (Section 6.9): geometry share by ground support in every testbed, beta_s where s was designed "
                "(CONVOY-3D, CONVOY-REAL, E5, E5-REAL) and the median per-object share g in driving; grounded = ground contact in "
                "view. Every CONVOY-REAL mover is a floating crate (D52). 95% cluster-bootstrap intervals; only interpretable "
                "model x testbed results.")
    _f18 = SUPPORT_LEVELS[SUPPORT_LEVELS["statistic"] == "beta_s"]
    _f18 = _f18.assign(value=_cis(_f18, "implied_factor", "implied_lo", "implied_hi", 2), row=_f18["testbed"] + ": " + _f18["condition"])
    write_table(_f18.pivot_table(index="row", columns="model", values="value", aggfunc="first")
                .reindex([o for o in _ord if o in set(_f18["row"])]).reset_index()
                .rename(columns=lambda c: MODEL_LABELS.get(c, c)).rename(columns={"row": "testbed: objects"}),
                "T18c_implied_depth_error", f"Depth overestimate exp(beta_s x) implied by each slope for an object keeping pace "
                f"with the camera at s = {PREREG['implied_factor_s']} (x = {X_IMPLIED:.2f}; 1 = true depth, "
                f"{np.exp(X_IMPLIED):.1f} = the geometric answer), with the slope's interval mapped through the same function.")
if len(SUPPORT_CONTRASTS):
    _c18 = SUPPORT_CONTRASTS.assign(value=[f"{ci_str(e, l, h, 3)} {d_}" if np.isfinite(e) else "--" for e, l, h, d_ in
                                           SUPPORT_CONTRASTS[["estimate", "ci_lo", "ci_hi", "direction"]].itertuples(index=False)],
                                    row=SUPPORT_CONTRASTS["testbed"] + ": " + SUPPORT_CONTRASTS["contrast"])
    write_table(_c18.pivot_table(index=["row", "statistic"], columns="model", values="value", aggfunc="first")
                .reindex(list(dict.fromkeys(zip(_c18["row"], _c18["statistic"])))).reset_index()
                .rename(columns=lambda c: MODEL_LABELS.get(c, c)).rename(columns={"row": "testbed: contrast"}),
                "T18b_support_contrasts", "Descriptive (Section 6.9): unsupported or hidden-contact objects minus grounded ones within "
                "each testbed, estimate [95% CI] and the direction the interval gives; H5, H6 and H11a as registered, the driving "
                "contrasts as in T15a, and floating minus grounded CONVOY-3D movers (new in v6.4, unpaired).", nd=3)
# ---- v6.5: designed experiments, cue model, Bonn and the twin test ---------------------------------------------------
if len(BSW_BETA):
    _b19 = BSW_BETA[BSW_BETA["model"].isin(MODELS_OK["BSW"])]
    _b19 = _b19.assign(value=_cis(_b19, "beta", "ci_lo", "ci_hi", 2), level_label=_b19["level"].map(SWEEP_DESIGN["labels"]))
    _t19 = (_b19.pivot_table(index=["support", "level_label"], columns="model", values="value", aggfunc="first").reset_index()
            .rename(columns=lambda c: MODEL_LABELS.get(c, c)).rename(columns={"level_label": "baseline"}))
    _t19["_o"] = _t19["support"].map({"all": 0, "grounded": 1, "floating": 2}) * 10 + _t19["baseline"].map(
        {v: i for i, v in enumerate(SWEEP_DESIGN["labels"].values())})
    write_table(_t19.sort_values("_o").drop(columns="_o"), "T19a_baseline_sweep",
                "Baseline sweep (Section 6.10): beta_s [95% CI] of the same movers with the camera baseline scaled by 0.5, 1, 2 "
                "and 3 (only the parallax changes), for all movers and separately for grounded and floating ones; common sample: "
                "measurable at every baseline and identifiable at x1." + _gate_note("BSW"))
if len(BSW_TESTS):
    _t = _interp(BSW_TESTS)
    write_table(_t.assign(estimate_ci=_cis(_t, "estimate", "ci_lo", "ci_hi", 3))[["test", "model", "contrast", "n", "n_scenes",
                                                                                    "estimate_ci", "p_holm", "verdict"]],
                "T19b_baseline_sweep_tests", "H12 (registered for DUSt3R and MASt3R; no prediction for the other models) and "
                "the exploratory split by support: the rise of beta_s from x0.5 to x3, paired by mover, 95% cluster-bootstrap "
                "intervals over base scenes, Holm across the two registered models." + _gate_note("BSW"), nd=3)
if len(WDR_BETA):
    _b20 = WDR_BETA[WDR_BETA["model"].isin(MODELS_OK["WDR"])]
    _b20 = _b20.assign(value=_cis(_b20, "beta", "ci_lo", "ci_hi", 2), wall=_b20["level"].map(WALL_DOSE_DESIGN["labels"]))
    _hid20 = WDR_BETA.groupby("level")["median_hidden_share"].median()
    _t20 = (_b20.pivot_table(index="wall", columns="model", values="value", aggfunc="first")
            .reindex([WALL_DOSE_DESIGN["labels"][k] for k in WDR_LEVELS]).reset_index()
            .rename(columns=lambda c: MODEL_LABELS.get(c, c)))
    _t20.insert(1, "share hidden", [f"{_hid20.get(k, np.nan):.2f}" for k in WDR_LEVELS])
    write_table(_t20, "T20a_dose_response",
                "Contact dose-response (Section 6.10): beta_s [95% CI] of the same grounded movers with no wall and behind walls "
                "built to hide at least their lowest 15, 35 and 70 cm in both views; share hidden: median share of the mover's "
                "silhouette hidden in view 1 (a wall must cover the whole visible base, so it hides more of the near end); common "
                "sample: contact visible without a wall and hidden by every wall." + _gate_note("WDR"))
if len(WDR_TESTS):
    _t = _interp(WDR_TESTS)
    write_table(_t.assign(estimate_ci=_cis(_t, "estimate", "ci_lo", "ci_hi", 3))[["test", "model", "contrast", "n", "n_scenes",
                                                                                    "estimate_ci", "p_holm", "verdict"]],
                "T20b_dose_response_tests", "H13 (registered, every model, Holm across models) and the shape of the curve "
                "(exploratory): beta(70 cm) - beta(no wall), beta(70 cm) - beta(15 cm) and the share of the 70 cm effect reached by "
                "the 15 cm wall, reported only where the 70 cm effect is clearly positive (its interval excludes 0)."
                + _gate_note("WDR"), nd=3)
if len(CUE_FIT_TABLE):
    write_table(CUE_FIT_TABLE, "T21a_cue_model_fits", "Cue model (Section 6.11, exploratory): rho - rho_mono = w (x - rho_mono), "
                "w = c_support d^2 / (d^2 + kappa^2), fitted by least squares on CONVOY-3D and E5 (two channels), with kappa = 0 "
                "(contact only) and with one c for every support (parallax only); kappa in px, rmse of rho.", nd=3)
if len(CUE_CV):
    _t = CUE_CV.assign(delta_ci=_cis(CUE_CV, "delta_mse", "ci_lo", "ci_hi", 4))
    write_table(_t[["model", "comparison", "n", "mse_two_channels", "mse_other", "delta_ci", "direction"]], "T21b_cue_model_cv",
                "Cue model: 10-fold cross-validation over scenes; mean squared error of held-out rho and the difference to the "
                "two-channel model with a 95% cluster-bootstrap interval (positive: the two-channel model predicts better).", nd=4)
if len(CUE_PRED):
    _t = CUE_PRED[CUE_PRED["role"] == "predicted"]
    write_table(_t.assign(observed=_cis(_t, "beta_observed", "ci_lo", "ci_hi", 2))[
                    ["model", "testbed", "group", "n", "median_parallax", "observed", "beta_predicted", "inside_ci"]],
                "T21c_cue_model_predictions", "Cue model, out of sample: the slope each testbed shows (95% cluster-bootstrap "
                "interval) and the slope the model fitted on CONVOY-3D and E5 predicts from the testbed's own objects (x, rho_mono, "
                f"parallax, support), without refitting. Over all rows: Pearson r = {CUE_CALIBRATION['pearson_r']:.2f}, mean "
                f"absolute error {CUE_CALIBRATION['mean_abs_error']:.3f}.", nd=3)
if len(MONDUS) or len(REAL_PARALLAX):
    if len(MONDUS):
        write_table(MONDUS.assign(difference_ci=_cis(MONDUS, "difference", "ci_lo", "ci_hi", 3))[
                        ["contrast", "n", "n_scenes", "DUSt3R", "MonST3R", "difference_ci", "direction"]],
                    "T21d_monst3r_vs_dust3r", "What dynamic fine-tuning changed: MonST3R minus DUSt3R on the same movers, for the "
                    "parallax channel (slope of movers with more than 40 px of parallax; rise over the baseline sweep) and the "
                    "contact channel (the H6 contrast); 95% cluster-bootstrap intervals over scenes.", nd=3)
    if len(REAL_PARALLAX):
        _t = REAL_PARALLAX[REAL_PARALLAX["n"] > 0]
        write_table(_t.assign(beta_ci=_cis(_t, "beta", "ci_lo", "ci_hi", 2))[["model", "testbed", "bin", "n", "median_parallax",
                                                                              "beta_ci"]],
                    "T21e_convoy_real_by_parallax", "Correction of the v6.4 reading (D57): beta_s of CONVOY-REAL movers and of "
                    "CONVOY-3D floating crates (both unsupported) overall and within parallax bins (bins with >= 12 movers).", nd=2)
if len(BONN_PREV):
    write_table(BONN_PREV, "T22a_bonn_regimes", "Bonn RGB-D Dynamic (Section 6.12, exploratory): motion regimes of real moving "
                f"objects (parallel: median |eps| below the flow-noise floor, {BONN_TAU_PAR:.2f} px), the share without a static "
                "solution and the share on and off the floor, by kind of object.", nd=3)
if len(BONN_G):
    _t = BONN_G.assign(g_ci=_cis(BONN_G, "median_g", "ci_lo", "ci_hi", 3))
    write_table(_t[["model", "grouping", "group", "n", "n_pairs", "median_x_geo", "g_ci"]], "T22b_bonn_geometry_share",
                "Bonn RGB-D Dynamic: median per-object geometry share g of parallel, identifiable objects with |x - rho_mono| >= 0.3 "
                "(95% cluster-bootstrap intervals over image pairs), by kind and by measured floor contact; x_geo from optical flow "
                "and sensor depth, so no slope is fitted (D24)." + _gate_note("BONN"), nd=3)
if len(TWIN_RECALL):
    _t = TWIN_RECALL.copy()
    for c in ("det_epipolar", "det_twin", "det_either", "twin_given_epi_miss"):
        _t[c + "_ci"] = _cis(_t, c, c + "_lo", c + "_hi", 3)
    write_table(_t[["regime", "n", "n_tracks", "det_epipolar_ci", "det_twin_ci", "det_either_ci", "twin_below_road", "n_epi_miss",
                    "twin_given_epi_miss_ci"]],
                "T23a_twin_test", "The static twin test on Argoverse 2 (Section 6.13, exploratory): share of moving vehicles "
                f"flagged by the epipolar test (|eps| > {TWIN_THRESH.get('eps_px', np.nan):.2f} px), by the twin test (|twin height| "
                f"> {TWIN_THRESH.get('height_m', np.nan):.2f} m) and by either, both at 5% false positives on "
                f"{TWIN_THRESH.get('n_parked', 0)} parked vehicles, by motion regime (95% intervals over vehicle tracks; "
                "both tests use the same exact correspondences from labelled boxes); twin_below_road: flagged with the twin below the "
                "road; n_epi_miss: vehicles the epipolar test misses; twin_given_epi_miss: the share of those the twin test flags.", nd=3)
if len(TWIN_S):
    write_table(TWIN_S[["s_lo", "s_hi", "n", "frac_parallel", "det_epipolar", "det_twin", "det_either"]], "T23b_twin_test_by_s",
                "The static twin test by parallel fraction s (moving Argoverse 2 vehicles with a static solution).", nd=3)
if len(FTWIN_RECALL):
    _t = FTWIN_RECALL.copy()
    for c in ("det_epipolar", "det_twin", "det_either", "twin_given_epi_miss"):
        _t[c + "_ci"] = _cis(_t, c, c + "_lo", c + "_hi", 3)
    _h14 = H14_RESULT
    write_table(_t[["regime", "n", "n_tracks", "det_epipolar_ci", "det_twin_ci", "det_either_ci", "twin_below_road", "n_epi_miss",
                    "twin_given_epi_miss_ci"]],
                "T23c_twin_test_flow", f"The twin test with optical flow ({FTWIN_THRESH.get('flow', '')}; Section 6.13, D63): the same "
                "vehicles, with correspondences from forward-backward checked flow at their LiDAR pixels and the road plane fitted "
                "to the LiDAR returns (no labelled box). Epipolar test: |eps| > "
                f"{FTWIN_THRESH.get('eps_px', np.nan):.2f} px; twin test: the 10th percentile of the static reconstruction's "
                f"heights above the road outside [{FTWIN_THRESH.get('h_low_lo_m', np.nan):.2f}, "
                f"{FTWIN_THRESH.get('h_low_hi_m', np.nan):.2f}] m; both at 5% false positives on {FTWIN_THRESH.get('n_parked', 0)} "
                "parked vehicles. H14 (registered): in the parallel regime the twin test flags more vehicles than the epipolar test, "
                + (f"difference {_h14['estimate']:+.3f} [{_h14['ci_lo']:+.3f}, {_h14['ci_hi']:+.3f}] over {_h14['n']} vehicles "
                   f"({_h14['n_tracks']} tracks): {_h14['verdict']}." if np.isfinite(_h14.get("estimate", np.nan))
                   else f"{_h14.get('verdict', 'NOT TESTABLE')} ({_h14.get('n', 0)} vehicles, {_h14.get('n_tracks', 0)} tracks)."), nd=3)


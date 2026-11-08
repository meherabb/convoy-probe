# PLACEHOLDER
# 7. Statistical summary: every registered test (locked P1-P4 and A1, v6 H5-H8, v6.1 H11a-b, v6.5 H12-H13, v6.6 H14), its
#    estimate, interval and verdict; H9 is listed as exploratory (its sample rule changed, D34). For H12 only
#    DUSt3R and MASt3R carry a prediction; the other models are listed with "NO PREDICTION".
SUMMARY_ROWS = []
def _add(prediction, model, statistic, est, lo, hi, n, verdict, p_holm=np.nan, family="locked (v5)"):
    SUMMARY_ROWS.append(dict(family=family, prediction=prediction, model=model, statistic=statistic, estimate=est,
                             ci_lo=lo, ci_hi=hi, p_holm=p_holm, n=n, verdict=verdict))

for r in E1_TABLE.itertuples():
    _add("P1 degeneracy (CONVOY-3D)", r.model, "beta_s", r.beta_s, r.ci_lo, r.ci_hi, r.n, r.verdict, r.p_holm)
for r in REAL_E1.itertuples():
    if r.verdict != "control":
        _add("P1 degeneracy (CONVOY-REAL)", r.model, "beta_s", r.beta_s, r.ci_lo, r.ci_hi, r.n, r.verdict)
for r in E2_TABLE[E2_TABLE["method_key"] != "M5"].itertuples():
    _add("P2 masks (CONVOY-3D)", f"{r.model} / {r.method}", "recall(perp) - recall(par)", r.diff, r.ci_lo, r.ci_hi,
         r.n_par + r.n_perp, r.verdict)
for r in E3_TABLE.itertuples():
    _add("P3a prior weight vs eps", r.model, "Spearman(w, |eps|)", r.spearman_w_eps, r.sp_lo, r.sp_hi, r.n_w, r.verdict_P3a)
    _add("P3b class dependence at eps~0", r.model, "w(car) - w(crate)", r.class_diff, r.cd_lo, r.cd_hi, r.n_parallel, r.verdict_P3b)
for r in E4_SUMMARY.itertuples():
    _add("P4 convoy persistence", r.model, "beta_s(straight,K=8) - beta_s(curved,K=8)", r.drop_K8, r.drop_lo, r.drop_hi,
         r.n_K8, r.verdict)
for r in A1_TABLE.itertuples():
    _add("A1 correspondence (knockout)", r.model, "beta_s(full) - beta_s(knockout)", r.drop, r.drop_lo, r.drop_hi, r.n, r.A1)
_e5_names = {"H5": "H5 lifting raises the geometry share", "H6": "H6 hiding the contact raises it",
             "H7": "H7 the wall acts through the contact", "H8": "H8 lifted objects look farther (single image)"}
for r in E5_TESTS[E5_TESTS["registered"].astype(bool)].itertuples():
    _add(_e5_names[r.test], r.model, r.contrast.split(":")[0], r.estimate, r.ci_lo, r.ci_hi, r.n, r.verdict,
         getattr(r, "p_holm", np.nan), family="registered (v6)")
_h11_names = {"H11a": "H11a real images: lifting raises the geometry share", "H11b": "H11b real images: lifted crates look farther (single image)"}
for r in RABL_TESTS.itertuples() if len(RABL_TESTS) else []:
    _add(_h11_names[r.test], r.model, r.contrast.split(":")[0], r.estimate, r.ci_lo, r.ci_hi, r.n, r.verdict,
         getattr(r, "p_holm", np.nan), family="registered (v6.1)")
_h1213_names = {"H12": "H12 a longer baseline raises the geometry share (baseline sweep)",
                "H13": "H13 hiding the contact with the lowest wall raises it (dose-response)"}
for _df in (BSW_TESTS, WDR_TESTS):
    for r in (_df[_df["test"].isin(["H12", "H13"])].itertuples() if len(_df) else []):
        _add(_h1213_names[r.test], r.model, r.contrast, r.estimate, r.ci_lo, r.ci_hi, r.n, r.verdict,
             getattr(r, "p_holm", np.nan), family="registered (v6.5)")
if H14_RESULT:
    _add("H14 twin test with flow flags more parallel movers than the epipolar test (Argoverse 2)",
         f"optical flow ({FTWIN_THRESH.get('flow', '').split(' (')[0] or 'flow'})", "det_twin - det_epipolar (parallel regime)",
         H14_RESULT["estimate"], H14_RESULT["ci_lo"], H14_RESULT["ci_hi"], H14_RESULT["n"], H14_RESULT["verdict"],
         family="registered (v6.6)")
for r in E6_TABLE[E6_TABLE["verdict"] != "control"].itertuples():
    _add("H9 driving: beta_geo < 0.3 (Virtual KITTI 2)", r.model + (" (trained on Virtual KITTI)" if r.trained_on_vkitti == "yes" else ""),
         "beta_geo", r.beta_geo, r.ci_lo, r.ci_hi, r.n, r.verdict, family="exploratory (v6.1 sample rule)")
SUMMARY = pd.DataFrame(SUMMARY_ROWS)
with pd.option_context("display.max_colwidth", 60, "display.width", 250):
    print(SUMMARY.drop(columns=["family"]).to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
SUMMARY.to_csv(os.path.join(RESULTS_DIR, "prediction_summary.csv"), index=False)
print("\nVerdict counts:")
print(SUMMARY.groupby(["prediction", "verdict"]).size().unstack(fill_value=0).to_string())

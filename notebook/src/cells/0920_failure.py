# PLACEHOLDER
# 9. Failure analysis: where do models depart from the static-equivalent depth?
#    Residual of each E1 mover from its model's fitted line, related (Spearman) to ground-truth
#    covariates; the largest residuals are listed for inspection.
FAIL_ROWS, WORST = [], []
for model in EVALUATED_MODELS:
    d = E1_DATA[model].copy()
    d = d[np.isfinite(d["rho"])]
    if len(d) < 10:
        continue
    b, a, _ = ols_fit(d["x_par"], d["rho"])
    d["abs_resid"] = np.abs(d["rho"] - (b * d["x_par"] + a))
    d["dev_from_geometry"] = np.abs(d["rho"] - d["x_par"])
    for cov in ["gt_parallax_med", "gt_Z_true", "s"]:
        rs = spearmanr(d[cov], d["dev_from_geometry"]).correlation
        FAIL_ROWS.append(dict(model=model, covariate=cov, spearman_with_deviation=float(rs)))
    for cls_, g in d.groupby("cls"):
        FAIL_ROWS.append(dict(model=model, covariate=f"class={cls_}", spearman_with_deviation=np.nan,
                              median_deviation=float(g["dev_from_geometry"].median()), n=len(g)))
    w = d.sort_values("dev_from_geometry", ascending=False).head(3)
    for r in w.itertuples():
        WORST.append(dict(model=model, uid=r.uid, cls=r.cls, cam_kind=r.cam_kind, s=r.s, parallax_px=r.gt_parallax_med,
                          Z=r.gt_Z_true, rho=r.rho, x_par=r.x_par, rho_mono=r.rho_mono))
FAILURE = pd.DataFrame(FAIL_ROWS, columns=["model", "covariate", "spearman_with_deviation", "median_deviation", "n"])
WORST = pd.DataFrame(WORST)
print("Spearman correlation of |rho - x| (deviation from the static-equivalent depth) with covariates:")
print(FAILURE[FAILURE["covariate"].isin(["gt_parallax_med", "gt_Z_true", "s"])]
      .pivot(index="covariate", columns="model", values="spearman_with_deviation").to_string(na_rep="–", float_format=lambda v: f"{v:.2f}"))
print("\nMedian deviation by class:")
print(FAILURE[FAILURE["covariate"].str.startswith("class=")]
      .pivot(index="covariate", columns="model", values="median_deviation").to_string(na_rep="–", float_format=lambda v: f"{v:.3f}"))
print("\nLargest deviations (3 per model):")
print(WORST.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
FAILURE.to_csv(os.path.join(RESULTS_DIR, "failure_covariates.csv"), index=False)
WORST.to_csv(os.path.join(RESULTS_DIR, "largest_deviations.csv"), index=False)
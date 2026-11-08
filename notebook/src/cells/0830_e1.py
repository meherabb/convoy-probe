# PLACEHOLDER
# 6.1 E1 / Prediction 1: is the depth of a parallel mover the static-equivalent depth?
#     beta_s = OLS slope of rho = log(Zhat/Z) on x = -log(1-s), purely parallel movers
#     (eps = 0 at every pixel), measurable and identifiable, CONVOY-3D.
E1_ROWS, E1_DATA, E1_PARALLAX = [], {}, []
_base = BASE_RES[(BASE_RES["set"] == "MAIN") & e1_mask(BASE_RES)]
CONTROL_E1 = {
    "Classical triangulation, exact correspondences": slope_with_ci(_base["x_par"], _base["rho_classical_exact"], _base["scene"], seed_for("e1", "cx")),
    f"Classical triangulation, {FLOW_NAME} flow": slope_with_ci(_base["x_par"], _base["rho_classical_flow"], _base["scene"], seed_for("e1", "cf")),
}
for model in EVALUATED_MODELS:
    d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN")]
    dp = d[e1_mask(d)]
    r = slope_with_ci(dp["x_par"], dp["rho"], dp["scene"], seed_for("e1", model), h0=0.5)
    rm = slope_with_ci(dp["x_par"], dp["rho_mono"], dp["scene"], seed_for("e1m", model))
    dg = d[(d["regime"] == "parallel") & (d["s"] < 1) & d["measurable"].astype(bool) & d["identifiable"].astype(bool)]
    rg = slope_with_ci(dg["gt_x_geo"], dg["rho"], dg["scene"], seed_for("e1g", model))
    gate_ok = GATE_PASS.get((model, "MAIN"), False)
    E1_ROWS.append(dict(model=model, training=MODEL_TRAINING[model], n=r["n"], n_clips=r["n_clusters"],
                        beta_s=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], intercept=r["intercept"], r2=r["r2"],
                        p_beta_le_half=r["p_h0"], beta_mono=rm["beta"], beta_mono_lo=rm["ci_lo"], beta_mono_hi=rm["ci_hi"],
                        beta_geo=rg["beta"], beta_geo_lo=rg["ci_lo"], beta_geo_hi=rg["ci_hi"], n_geo=rg["n"],
                        w_eff=beta_to_effective_w(r["beta"]) if np.isfinite(r["beta"]) else np.nan,
                        gate=("PASS" if gate_ok else "FAIL")))
    E1_DATA[model] = dp[["uid", "scene", "x_par", "rho", "rho_mono", "rho_ko", "gt_parallax_med", "cls", "cam_kind",
                         "gt_Z_true", "s"]].copy()
    for lo_, hi_ in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]:
        db = d[(d["motion_type"] == "parallel") & (d["s"] < 1) & d["measurable"].astype(bool)
               & (d["gt_parallax_med"] >= lo_) & (d["gt_parallax_med"] < hi_)]
        rb = slope_with_ci(db["x_par"], db["rho"], db["scene"], seed_for("e1p", model, lo_))
        E1_PARALLAX.append(dict(model=model, par_lo=lo_, par_hi=hi_, n=rb["n"], beta=rb["beta"], ci_lo=rb["ci_lo"], ci_hi=rb["ci_hi"]))
E1_TABLE = pd.DataFrame(E1_ROWS)
E1_TABLE["p_holm"] = holm(E1_TABLE["p_beta_le_half"].values)
E1_TABLE["verdict"] = [verdict_p1(b, l, h, r2) if g == "PASS" else "NOT INTERPRETABLE (gate)"
                       for b, l, h, r2, g in zip(E1_TABLE["beta_s"], E1_TABLE["ci_lo"], E1_TABLE["ci_hi"], E1_TABLE["r2"], E1_TABLE["gate"])]
E1_PARALLAX = pd.DataFrame(E1_PARALLAX)

print("Positive controls (same movers):")
for k, v in CONTROL_E1.items():
    print(f"   {k:<52} beta = {v['beta']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}], R^2 = {v['r2']:.3f}, n = {v['n']}")
print("\nE1 (Prediction 1), CONVOY-3D, purely parallel movers:")
_cols = ["model", "training", "n", "n_clips", "beta_s", "ci_lo", "ci_hi", "r2", "beta_mono", "beta_geo", "w_eff", "p_holm", "verdict"]
print(E1_TABLE[_cols].to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
E1_TABLE.to_csv(os.path.join(RESULTS_DIR, "E1_beta_s.csv"), index=False)
E1_PARALLAX.to_csv(os.path.join(RESULTS_DIR, "E1_beta_by_parallax.csv"), index=False)
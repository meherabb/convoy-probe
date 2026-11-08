# PLACEHOLDER
# 8. Robustness of beta_s (E1 movers): camera heading, mover class, depth,
#    focal length and translational parallax; and the exploratory regime s > 1, where no
#    static point explains the two views (Z* < 0).
ROBUST_ROWS = []
for model in EVALUATED_MODELS:
    d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN")]
    dp = d[e1_mask(d)].copy()
    dp["depth_tercile"] = pd.qcut(dp["gt_Z_true"].rank(method="first"), 3, labels=["near", "mid", "far"])
    dp["focal_tercile"] = pd.qcut(dp["focal"].rank(method="first"), 3, labels=["short f", "mid f", "long f"])
    for factor in ["cam_kind", "cls", "depth_tercile", "focal_tercile"]:
        for level, g in dp.groupby(factor, observed=True):
            r = slope_with_ci(g["x_par"], g["rho"], g["scene"], seed_for("rob", model, factor, str(level)))
            ROBUST_ROWS.append(dict(model=model, factor=factor, level=str(level), n=r["n"], beta_s=r["beta"],
                                    ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))
ROBUST = pd.DataFrame(ROBUST_ROWS)
_piv = ROBUST.pivot_table(index=["factor", "level"], columns="model", values="beta_s")
print("beta_s by condition (rows) and model (columns):")
print(_piv.to_string(na_rep="–", float_format=lambda v: f"{v:.2f}"))
ROBUST.to_csv(os.path.join(RESULTS_DIR, "robustness.csv"), index=False)

# exploratory: s > 1 (the static explanation lies behind the camera)
BEYOND_ROWS = []
for model in EVALUATED_MODELS:
    d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN") & (MOV_RES["s"] > 1)
                & (MOV_RES["regime"] == "parallel") & MOV_RES["measurable"].astype(bool)]
    if len(d) < 5:
        continue
    diff, lo, hi, _ = boot_stat(d, "scene", lambda c: float(np.median(c["rho"] - c["rho_mono"])), seed_for("beyond", model),
                                cols=["rho", "rho_mono"])
    BEYOND_ROWS.append(dict(model=model, n=len(d), median_rho=float(d["rho"].median()),
                            median_rho_mono=float(d["rho_mono"].median()), median_rho_minus_mono=diff, ci_lo=lo, ci_hi=hi,
                            frac_placed_farther_than_mono=float(np.mean(d["rho"] > d["rho_mono"]))))
BEYOND = pd.DataFrame(BEYOND_ROWS)
print("\nExploratory: parallel movers with s > 1 (no static explanation in front of the camera):")
print(BEYOND.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}") if len(BEYOND) else "  (too few movers)")
BEYOND.to_csv(os.path.join(RESULTS_DIR, "s_beyond_one.csv"), index=False)
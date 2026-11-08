# PLACEHOLDER
# 6.8 E6 driving scenes (Virtual KITTI 2): H9 and the prevalence of undetectable motion.
def verdict_h9(beta, lo, hi, n, n_pairs):
    if not np.isfinite(beta) or n < PREREG["h9_min_movers"] or n_pairs < PREREG["h9_min_pairs"]:
        return "NOT TESTABLE"
    if np.isfinite(lo) and lo > PREREG["h9_beta_max"]:
        return "FALSIFIED"
    if beta < PREREG["h9_beta_max"] and np.isfinite(hi) and hi < PREREG["h9_ci_hi_max"]:
        return "SUPPORTED"
    return "PARTIAL"

E6_ROWS, E6_DATA, E6_MASKS = [], {}, []
if VK_STATUS["available"] and len(MOVERS_VK):
    _bv = BASE_RES[(BASE_RES["set"] == "VK")]
    _bvp = _bv[h9_mask(_bv)]
    r = slope_with_ci(_bvp["gt_x_geo"], _bvp["rho_classical_exact"], _bvp["scene"], seed_for("e6", "classical"))
    E6_ROWS.append(dict(model="Classical triangulation, exact correspondences", trained_on_vkitti="-", gate="-", n=r["n"],
                        n_pairs=r["n_clusters"], beta_geo=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], r2=r["r2"],
                        beta_mono=np.nan, w_eff=np.nan, verdict="control"))
    for model in EVALUATED_MODELS:
        d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "VK")]
        if not len(d):
            continue
        dp = d[h9_mask(d)]
        r = slope_with_ci(dp["gt_x_geo"], dp["rho"], dp["scene"], seed_for("e6", model))
        rm = slope_with_ci(dp["gt_x_geo"], dp["rho_mono"], dp["scene"], seed_for("e6m", model))
        gate_ok = GATE_PASS.get((model, "VK"), False)
        v = verdict_h9(r["beta"], r["ci_lo"], r["ci_hi"], r["n"], r["n_clusters"]) if gate_ok else "NOT INTERPRETABLE (gate)"
        E6_ROWS.append(dict(model=model, trained_on_vkitti="yes" if VK_TRAINING_EXPOSURE.get(model) else "no",
                            gate="PASS" if gate_ok else "FAIL", n=r["n"], n_pairs=r["n_clusters"], beta_geo=r["beta"],
                            ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], r2=r["r2"], beta_mono=rm["beta"],
                            w_eff=beta_to_effective_w(r["beta"]) if np.isfinite(r["beta"]) else np.nan, verdict=v))
        E6_DATA[model] = dp[["uid", "scene", "seq", "gt_x_geo", "rho", "rho_mono", "s", "gt_Z_true", "gt_parallax_med"]].copy()
        # exploratory: model-based dynamic masks on moving vehicles, by regime. M1 is not listed: with exact
        # correspondences its static false-positive threshold is 0 px, so on these frames it flags any
        # offset above the precision of the released poses, which no real two-view test can do.
        dm = d[d["measurable"].astype(bool)]
        for reg, g in dm.groupby("regime"):
            E6_MASKS.append(dict(model=model, regime=reg, n=len(g), recall_M2_exact=g["recall_M2_exact"].mean(),
                                 recall_M5=g["recall_M5"].mean()))
E6_TABLE = pd.DataFrame(E6_ROWS, columns=["model", "trained_on_vkitti", "gate", "n", "n_pairs", "beta_geo", "ci_lo", "ci_hi",
                                          "r2", "beta_mono", "w_eff", "verdict"])
E6_MASKS = pd.DataFrame(E6_MASKS, columns=["model", "regime", "n", "recall_M2_exact", "recall_M5"])

# prevalence of motion that two views (and K views) cannot see
PREVALENCE_ALL = pd.concat([PREVALENCE.assign(dataset="TUM fr3 walking (people)") if len(PREVALENCE) else None,
                            PREVALENCE_VK.assign(dataset="Virtual KITTI 2 (vehicles)") if len(PREVALENCE_VK) else None],
                           ignore_index=True) if (len(PREVALENCE) or len(PREVALENCE_VK)) else pd.DataFrame()
if len(KVIEW_VK):
    KVIEW_SUMMARY = (KVIEW_VK.assign(static_fit=KVIEW_VK["residual_px"] < 1.0)
                     .groupby("K").agg(n_tracks=("static_fit", "size"), share_static_fit_below_1px=("static_fit", "mean"),
                                       median_residual_px=("residual_px", "median")).reset_index())
else:
    KVIEW_SUMMARY = pd.DataFrame(columns=["K", "n_tracks", "share_static_fit_below_1px", "median_residual_px"])

if len(E6_TABLE):
    print("E6 / H9 (exploratory in v6.1, D34), Virtual KITTI 2 moving vehicles (parallel regime, measurable, identifiable, "
          "x_geo in the design range):")
    print(E6_TABLE.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
    print("\nExploratory: mean recall at 5% static FPR on moving vehicles, by regime:")
    print(E6_MASKS.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
else:
    print("E6 is not run: Virtual KITTI 2 is not available in this run (" + VK_STATUS["note"] + ").")
if len(PREVALENCE_ALL):
    print("\nShare of moving pixels per motion regime (two views):")
    print(PREVALENCE_ALL.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
if len(KVIEW_SUMMARY):
    print("\nVirtual KITTI 2 moving vehicles whose K-frame track a static point fits within 1 px:")
    print(KVIEW_SUMMARY.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
E6_TABLE.to_csv(os.path.join(RESULTS_DIR, "E6_vkitti_h9.csv"), index=False)
E6_MASKS.to_csv(os.path.join(RESULTS_DIR, "E6_vkitti_masks.csv"), index=False)
PREVALENCE_ALL.to_csv(os.path.join(RESULTS_DIR, "prevalence_two_view.csv"), index=False)
KVIEW_SUMMARY.to_csv(os.path.join(RESULTS_DIR, "prevalence_kview_vkitti.csv"), index=False)

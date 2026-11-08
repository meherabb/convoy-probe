# PLACEHOLDER
# 6.0b Estimator validation with oracle predictors of known prior weight.
#      Before reading any model result, the E1 and E3 estimators are run on synthetic
#      predictors whose answer is known: on every mover pixel the oracle predicts
#      Zhat = w * Z_true + (1 - w) * Z_geo (w fixed), with 3% log-normal noise and an arbitrary
#      global scale per image; its "single-image" prediction is Z_true with independent noise.
#      Same movers, same scale alignment, same regressions as for the real models.
#      Expected: beta_s = beta(w) from the slope-weight identity (Section 2.4), median w = w.
#      The same check is run for the Virtual KITTI 2 estimator of H9. One checkpointed stage.
ORACLE_WS = [0.0, 0.25, 0.5, 0.75, 1.0]
ORACLE_STAGE = Stage("oracle", "eval", dict(ws=ORACLE_WS, measure=MEASURE_VERSION, prereg=_EVAL_PREREG, vk=len(VK_PAIR_META)),
                     upstream=[RENDER_FP["MAIN"], VK_GT_STAGE.fp if VK_STATUS["available"] else ""])
ORACLE_DIR = os.path.join(CACHE_DIR, "oracle")
with stage("oracle_validation"):
    if not ORACLE_STAGE.done():
        _orng = np.random.default_rng(seed_for("oracle"))
        _om_rows, _ov_rows = [], []
        for meta in META_MAIN:
            sc = meta["scene"]
            r0, r1 = load_frame(meta["dir"], 0), load_frame(meta["dir"], 1)
            masks, pB, vis = mover_masks(sc, r0, r1)
            sm = static_mask(r0, "synth")
            H, W = r0["depth"].shape
            vv, uu = np.mgrid[0:H, 0:W]
            _, zgeo, _ = pixel_epipolar_quantities(sc, 0, 1, np.stack([uu, vv], -1).astype(np.float64), pB)
            zt = np.where(np.isfinite(r0["depth"]), r0["depth"], 1e3).astype(np.float64)
            noise_p = np.exp(_orng.normal(0, 0.03, zt.shape)); noise_m = np.exp(_orng.normal(0, 0.03, zt.shape))
            scale_p, scale_m = np.exp(_orng.normal(0, 1)), np.exp(_orng.normal(0, 1))
            mono = zt * noise_m * scale_m
            g_m = scale_and_gate(mono, r0["depth"], sm)
            ok = (r0["mover_id"] >= 0) & np.isfinite(zgeo) & (zgeo > 0)
            for w in ORACLE_WS:
                pred = zt.copy()
                pred[ok] = w * zt[ok] + (1 - w) * zgeo[ok]
                pred = pred * noise_p * scale_p
                g = scale_and_gate(pred, r0["depth"], sm)
                for j, mk in enumerate(masks):
                    _om_rows.append(dict(w=w, uid=f"MAIN-{meta['index']:04d}-{j}", scene=meta["index"],
                                         rho=mover_rho(pred, r0["depth"], mk, g["scale"]),
                                         rho_mono=mover_rho(mono, r0["depth"], mk, g_m["scale"])))
        for meta in VK_PAIR_META:
            vsc = VK_SCENE_OBJS[meta["seq"]]; i, j = meta["frames"]
            g = vsc.pair_gt(i, j)
            zt = np.where(np.isfinite(g["depth"]), g["depth"], 1e3).astype(np.float64)
            sm = vk_static_mask(g["depth"], g["inst"])
            masks = [erode_mask(g["mover_id"] == k, PREREG["erosion_px"]) & g["visible"] for k in range(len(g["movers"]))]
            noise_p = np.exp(_orng.normal(0, 0.03, zt.shape)); scale_p = np.exp(_orng.normal(0, 1))
            ok = (g["mover_id"] >= 0) & np.isfinite(g["zgeo"]) & (g["zgeo"] > 0)
            for w in ORACLE_WS:
                pred = zt.copy()
                pred[ok] = w * zt[ok] + (1 - w) * g["zgeo"][ok]
                pred = pred * noise_p * scale_p
                gg = scale_and_gate(pred, g["depth"], sm)
                for k, mk in enumerate(masks):
                    _ov_rows.append(dict(w=w, uid=f"VK-{meta['index']:04d}-{k}", scene=meta["index"],
                                         rho=mover_rho(pred, g["depth"], mk, gg["scale"])))
        os.makedirs(ORACLE_DIR, exist_ok=True)
        pd.DataFrame(_om_rows).to_csv(os.path.join(ORACLE_DIR, "oracle_main.csv"), index=False)
        pd.DataFrame(_ov_rows, columns=["w", "uid", "scene", "rho"]).to_csv(os.path.join(ORACLE_DIR, "oracle_vkitti.csv"), index=False)
        ORACLE_STAGE.commit([ORACLE_DIR])
_om = pd.read_csv(os.path.join(ORACLE_DIR, "oracle_main.csv"))
_ov = _read(os.path.join(ORACLE_DIR, "oracle_vkitti.csv"))

_mv = MOVERS_MAIN[["uid", "x_par", "gt_x_geo", "motion_type", "s", "measurable", "identifiable", "regime", "gt_eps_med"]]
ORACLE_ROWS = []
for w in ORACLE_WS:
    d = _om[np.isclose(_om["w"], w)].merge(_mv, on="uid")
    dp = d[e1_mask(d)]
    r = slope_with_ci(dp["x_par"], dp["rho"], dp["scene"], seed_for("oracle-e1", w))
    beta_theory = float(np.polyfit(dp["x_par"], np.log(w + (1 - w) * np.exp(dp["x_par"])), 1)[0]) if len(dp) > 2 else np.nan
    de = d[d["measurable"].astype(bool) & (d["s"] < 1) & np.isfinite(d["gt_x_geo"])].copy()
    de["w"] = prior_weight(de)
    ORACLE_ROWS.append(dict(w_true=w, n=r["n"], beta_s=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                            beta_theory_identity=beta_theory, verdict_P1=verdict_p1(r["beta"], r["ci_lo"], r["ci_hi"], r["r2"]),
                            median_w_recovered=float(np.nanmedian(de["w"])) if len(de) else np.nan))
ORACLE_TABLE = pd.DataFrame(ORACLE_ROWS)
print(ORACLE_TABLE.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
_dev = np.nanmax(np.abs(ORACLE_TABLE["beta_s"] - ORACLE_TABLE["beta_theory_identity"]))
_wdev = np.nanmax(np.abs(ORACLE_TABLE["median_w_recovered"] - ORACLE_TABLE["w_true"]))
_v0 = ORACLE_TABLE.loc[ORACLE_TABLE["w_true"] == 0.0, "verdict_P1"].iloc[0]
_v1 = ORACLE_TABLE.loc[ORACLE_TABLE["w_true"] == 1.0, "verdict_P1"].iloc[0]
print(f"\nmax |beta_s(recovered) - beta(w)(identity)| = {_dev:.3f};  max |median w(recovered) - w| = {_wdev:.3f}")
print(f"P1 verdict for the pure-geometry oracle (w = 0): {_v0};  for the pure-prior oracle (w = 1): {_v1}")
ORACLE_OK = bool(_dev < 0.1 and _wdev < 0.1 and _v0 == "SUPPORTED" and _v1 == "FALSIFIED")
print("Estimator validation:", "PASS" if ORACLE_OK else "CHECK -- interpret model-level E1/E3 results with caution")
ORACLE_TABLE.to_csv(os.path.join(RESULTS_DIR, "oracle_validation.csv"), index=False)

# Virtual KITTI 2 estimator of H9 (regression of rho on x_geo over the H9 movers; real-looking images,
# no exact parallel motion)
ORACLE_VK = pd.DataFrame()
if VK_STATUS["available"] and len(MOVERS_VK) and len(_ov):
    _rows = []
    for w in ORACLE_WS:
        d = _ov[np.isclose(_ov["w"], w)].merge(MOVERS_VK[["uid", "gt_x_geo", "regime", "measurable", "identifiable"]], on="uid")
        dp = d[h9_mask(d)]
        r = slope_with_ci(dp["gt_x_geo"], dp["rho"], dp["scene"], seed_for("oracle-vk", w))
        bt = float(np.polyfit(dp["gt_x_geo"], np.log(w + (1 - w) * np.exp(dp["gt_x_geo"])), 1)[0]) if len(dp) > 2 else np.nan
        _rows.append(dict(w_true=w, n=r["n"], beta_geo=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], beta_theory_identity=bt))
    ORACLE_VK = pd.DataFrame(_rows)
    if ORACLE_VK["beta_geo"].notna().any():
        print("\nVirtual KITTI 2 (H9 estimator), planted predictors:")
        print(ORACLE_VK.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
        print(f"max |beta_geo(recovered) - beta(w)(identity)| = "
              f"{np.nanmax(np.abs(ORACLE_VK['beta_geo'] - ORACLE_VK['beta_theory_identity'])):.3f}")
        ORACLE_VK.to_csv(os.path.join(RESULTS_DIR, "oracle_validation_vkitti.csv"), index=False)
    else:
        print("\nVirtual KITTI 2: too few H9 movers for the oracle check (not reported).")
        ORACLE_VK = pd.DataFrame()

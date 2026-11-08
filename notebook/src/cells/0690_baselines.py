# PLACEHOLDER
# 4.7 Model-independent baselines on every CONVOY-3D clip, CONVOY-REAL pair (and both conditions of
#     the real-background contact ablation) and Virtual KITTI 2 pair:
#   * classical two-view triangulation with the TRUE relative pose (positive control), from
#     exact correspondences and (CONVOY-3D, CONVOY-REAL) from estimated optical flow;
#   * M1 epipolar-residual mask from exact correspondences and (CONVOY-3D, CONVOY-REAL) from flow.
#   One checkpointed stage (Section 0.5).
BASELINE_ROWS = []
BASELINE_STAGE = Stage("baselines", "eval", dict(measure=MEASURE_VERSION, prereg=_EVAL_PREREG, m1_resolution=M1_RESOLUTION_PX,
                                                 sets=dict(real=len(META_REAL), rabl=len(META_RABL), vk=len(VK_PAIR_META))),
                       upstream=[RENDER_FP["MAIN"], TUM_STAGES["REAL"].fp, RABL_STAGE.fp, FLOW_FP.get("MAIN", ""),
                                 FLOW_FP.get("REAL", ""), VK_GT_STAGE.fp if VK_STATUS["available"] else ""])
BASELINE_PATH = os.path.join(CACHE_DIR, "baselines", "baselines_per_mover.csv")

def _baseline_rows(set_, idx, masks, gt, z_exact, m1_exact, sm_exact, seed, z_flow=None, m1_flow=None, sm_flow=None):
    sc_e = mask_scores(m1_exact, sm_exact, masks, seed=seed)
    sc_f = mask_scores(m1_flow, sm_flow, masks, seed=seed) if m1_flow is not None else [dict(recall=np.nan, auroc=np.nan)] * len(masks)
    rows = []
    for j, m in enumerate(masks):
        g = gt[m].astype(np.float64); ze = z_exact[m]
        okf = np.zeros(0, bool)
        if z_flow is not None:
            zf = z_flow[m]; okf = np.isfinite(zf) & (zf > 0)
        rows.append(dict(set=set_, scene=idx, mover=j, uid=f"{set_}-{idx:04d}-{j}",
                         rho_classical_flow=float(np.median(np.log(zf[okf] / g[okf]))) if okf.sum() >= PREREG["min_object_pixels"] else np.nan,
                         frac_behind_classical_flow=float(np.mean(zf <= 0)) if z_flow is not None and len(zf) else np.nan,
                         rho_classical_exact=float(np.median(np.log(ze[ze > 0] / g[ze > 0]))) if np.sum(ze > 0) >= PREREG["min_object_pixels"] else np.nan,
                         recall_M1_exact=sc_e[j]["recall"], auroc_M1_exact=sc_e[j]["auroc"],
                         recall_M1_flow=sc_f[j]["recall"], auroc_M1_flow=sc_f[j]["auroc"]))
    return rows

with stage("baselines"):
    if not BASELINE_STAGE.done():
        for meta in META_MAIN + META_REAL:
            kind = "real" if meta["set"] == "REAL" else "synth"
            sc = meta["scene"]
            r0, r1 = load_frame(meta["dir"], 0), load_frame(meta["dir"], 1)
            masks, pB, vis = mover_masks(sc, r0, r1)
            sm = static_mask(r0, kind)
            p_flow = flow_to_pobs(load_flow(meta))
            BASELINE_ROWS += _baseline_rows(meta["set"], meta["index"], masks, r0["depth"], classical_triangulation_depth(sc, pB),
                                            epipolar_residual_map(sc, pB), sm & vis, meta["index"],
                                            classical_triangulation_depth(sc, p_flow), epipolar_residual_map(sc, p_flow), sm)
        for meta in META_RABL:
            sc = meta["scene"]
            r0, r1 = load_frame(meta["dir"], 0), load_frame(meta["dir"], 1)
            masks, pB, vis = mover_masks(sc, r0, r1)
            sm = static_mask(r0, "real")
            rows = _baseline_rows("RABL", meta["index"], masks, r0["depth"], classical_triangulation_depth(sc, pB),
                                  epipolar_residual_map(sc, pB), sm & vis, meta["index"])
            BASELINE_ROWS += [dict(r, condition=meta["condition"]) for r in rows]
        for meta in VK_PAIR_META:
            vsc = VK_SCENE_OBJS[meta["seq"]]
            i, j = meta["frames"]
            g = vsc.pair_gt(i, j)
            masks = [erode_mask(g["mover_id"] == k, PREREG["erosion_px"]) & g["visible"] for k in range(len(g["movers"]))]
            sm = vk_static_mask(g["depth"], g["inst"])
            BASELINE_ROWS += _baseline_rows(meta["set"], meta["index"], masks, g["depth"], classical_triangulation_depth(g["scene"], g["pB"]),
                                            epipolar_residual_map(g["scene"], g["pB"]), sm & g["visible"], meta["index"])
        for _sc in VK_SCENE_OBJS.values():
            _sc._cache.clear()
        os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
        pd.DataFrame(BASELINE_ROWS).to_csv(BASELINE_PATH, index=False)
        BASELINE_STAGE.commit([BASELINE_PATH])
BASELINES = pd.read_csv(BASELINE_PATH)
if "condition" not in BASELINES:
    BASELINES["condition"] = np.nan
release_memory()
print(f"baselines: {len(BASELINES)} movers")
_gt_b = pd.concat([MOVERS_MAIN, MOVERS_REAL, MOVERS_VK] if len(MOVERS_VK) else [MOVERS_MAIN, MOVERS_REAL], ignore_index=True)
_tmp = BASELINES[BASELINES["set"] != "RABL"].merge(_gt_b.reindex(columns=["uid", "regime", "motion_type", "x_par", "s", "gt_x_geo"]), on="uid")
print(_tmp.groupby(["set", "regime"])[["recall_M1_exact", "recall_M1_flow", "auroc_M1_exact"]].mean().round(3).to_string(na_rep="–"))
_p = _tmp[(_tmp["motion_type"] == "parallel") & (_tmp["s"] < 1)]
print(f"\nclassical triangulation, parallel movers: max |rho_exact - x_par| = "
      f"{np.nanmax(np.abs(_p['rho_classical_exact'] - _p['x_par'])):.2e} (exact correspondences); "
      f"median |rho_flow - x_par| = {np.nanmedian(np.abs(_p['rho_classical_flow'] - _p['x_par'])):.3f} ({FLOW_NAME} flow)")
_v = _tmp[_tmp["set"] == "VK"]
if len(_v):
    _vv = _v[np.isfinite(_v["gt_x_geo"])]
    print(f"Virtual KITTI 2: max |rho_exact - x_geo| = {np.nanmax(np.abs(_vv['rho_classical_exact'] - _vv['gt_x_geo'])):.2e} "
          f"over {len(_vv)} moving vehicles (exact correspondences; a check of the ground-truth pipeline)")

_ = mem_report("baselines")

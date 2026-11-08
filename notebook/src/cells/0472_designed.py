# PLACEHOLDER
# 3.2b Designed experiments of v6.5 (D55; H12 and H13 registered in AMENDMENT_V6_5). Two checkpointed render stages:
#      BSW  baseline sweep: 64 base scenes x 4 baselines (x0.5, x1, x2, x3); only the parallax changes
#      WDR  contact dose-response: 64 base scenes x 4 wall heights (none, 15, 35, 70 cm)
#      Both use the CONVOY-3D sampler, renderer and the E5 wall design; frames 0 and 1 are rendered.
N_BSW, N_WDR = RUN_CONFIG["n_scenes_bsw"], RUN_CONFIG["n_scenes_wdr"]
BSW_LEVELS, WDR_LEVELS = list(SWEEP_DESIGN["levels"]), list(WALL_DOSE_DESIGN["levels"])

def _center_quantities(sc, mv):
    # exact epipolar offset, static-hypothesis depth and translational parallax at the mover's centre (frames 0 -> 1)
    cam0, cam1 = PinholeCamera.from_dict(sc["cams"][0]), PinholeCamera.from_dict(sc["cams"][1])
    R0, C0 = np.asarray(sc["cams"][0]["R"]), np.asarray(sc["cams"][0]["C"])
    R1, C1 = np.asarray(sc["cams"][1]["R"]), np.asarray(sc["cams"][1]["C"])
    c, d = np.asarray(mv["centroid0"], float), np.asarray(mv["delta"], float)
    p0 = cam0.project(world_to_cam(c, R0, C0)); p1 = cam1.project(world_to_cam(c + d, R1, C1))
    R21, t21 = relative_pose(R0, C0, R1, C1)
    eps = abs(float(epipolar_offset(p0, p1, fundamental_matrix(cam0.K, cam1.K, R21, t21))))
    zgeo, _ = triangulate_depth(p0, p1, cam0.K, cam1.K, R21, t21)
    return eps, float(zgeo), float(translational_parallax(p0, p1, cam0.K, cam1.K, R21))

def sweep_variant(base, level):
    # the base scene seen with the baseline scaled by m: camera 2 keeps the rotation of frame 1 and moves along the same
    # line, C = C0 + m (C1 - C0), and every mover is displaced by m delta, so s, eps = 0 and x = -log(1-s) are unchanged
    m = float(SWEEP_DESIGN["levels"][level])
    sc = json.loads(json.dumps(scene_to_json(base)))
    c0, c1 = sc["cams"][0], dict(sc["cams"][1])
    C0, C1 = np.asarray(c0["C"], float), np.asarray(c1["C"], float)
    c1["C"] = (C0 + m * (C1 - C0)).tolist()
    sc["cams"] = [c0, c1]
    sc["camera"] = dict(sc["camera"], baseline=float(sc["camera"]["baseline"] * m), level=level, factor=m)
    for mv in sc["movers"]:
        mv["delta"] = [float(m * v) for v in mv["delta"]]
        mv["eps_center"], mv["Zgeo_center"], mv["parallax_center"] = _center_quantities(sc, mv)
    sc["condition"] = level
    return sc

def _wdr_job(job):
    # one wall height of one base scene: no wall (G) or static walls hiding the lowest cover_m of every mover
    base, level = job["scene"], job["level"]
    cover = float(WALL_DOSE_DESIGN["levels"][level])
    walls = []
    if cover > 0:
        walls = [w for w in (design_wall(base, j, cover_m=cover) for j in range(len(base["movers"]))) if w is not None]
    sc = ablation_variant(base, "CH" if cover > 0 else "G", walls, [0.0] * len(base["movers"]))
    sc["walls"], sc["condition"] = walls, level
    return _render_job(dict(set="WDR", index=job["index"], scene=sc, frames=[0, 1], dir=job["dir"]))

DESIGN_STAGES = {
    "BSW": Stage("render:BSW", "render65", dict(_base_cfg, n=N_BSW, design=_jsonable(SWEEP_DESIGN))),
    "WDR": Stage("render:WDR", "render65", dict(_base_cfg, n=N_WDR, design=_jsonable(WALL_DOSE_DESIGN), wall=ABLATION_DESIGN,
                                                band=PREREG["contact_band_m"])),
}
RENDER_FP.update({k: v.fp for k, v in DESIGN_STAGES.items()})

with stage("render_designed"):
    st = DESIGN_STAGES["BSW"]
    if not st.done():
        deck = make_design_deck(N_BSW * M_PER, seed_for("deck-bsw"), classes=SWEEP_DESIGN["classes"],
                                eps_targets={0.0: 1.0}, p_beyond=0.0)
        specs, unplaced = _specs("BSW", [seed_for("bsw", i) for i in range(N_BSW)],
                                 [deck[i * M_PER:(i + 1) * M_PER] for i in range(N_BSW)],
                                 SWEEP_DESIGN["n_frames_sampled"], "straight", False)
        _run_jobs(_render_job, [dict(set="BSW", index=i, scene=sweep_variant(specs[i], lv), frames=[0, 1],
                                     dir=os.path.join(RENDER_DIR, "bsw", f"{i:04d}_{lv}"))
                                for i in range(N_BSW) for lv in BSW_LEVELS])
        st.commit([os.path.join(RENDER_DIR, "bsw"), os.path.join(RENDER_DIR, "specs_BSW_straight.json")],
                  extra=dict(unplaced=len(unplaced)))
    release_memory(); mem_report("render:BSW")
    st = DESIGN_STAGES["WDR"]
    if not st.done():
        deck = make_design_deck(N_WDR * M_PER, seed_for("deck-wdr"), classes=WALL_DOSE_DESIGN["classes"],
                                eps_targets={0.0: 1.0}, p_beyond=0.0)
        specs, unplaced = _specs("WDR", [seed_for("wdr", i) for i in range(N_WDR)],
                                 [deck[i * M_PER:(i + 1) * M_PER] for i in range(N_WDR)], 2, "straight", False)
        _run_jobs(_wdr_job, [dict(index=i, scene=specs[i], level=lv, dir=os.path.join(RENDER_DIR, "wdr", f"{i:04d}_{lv}"))
                             for i in range(N_WDR) for lv in WDR_LEVELS])
        st.commit([os.path.join(RENDER_DIR, "wdr"), os.path.join(RENDER_DIR, "specs_WDR_straight.json")],
                  extra=dict(unplaced=len(unplaced)))
    release_memory(); mem_report("render:WDR")

META_BSW = sorted(_load_metas(os.path.join(RENDER_DIR, "bsw")), key=lambda m: (m["index"], BSW_LEVELS.index(m["condition"])))
META_WDR = sorted(_load_metas(os.path.join(RENDER_DIR, "wdr")), key=lambda m: (m["index"], WDR_LEVELS.index(m["condition"])))
for _k, _l in [("BSW", META_BSW), ("WDR", META_WDR)]:
    assert len(_l) > 0, f"no {_k} renders"
MOVERS_BSW = _regimes(_mover_table(META_BSW))
MOVERS_BSW["factor"] = MOVERS_BSW["condition"].map(SWEEP_DESIGN["levels"]).astype(float)
MOVERS_WDR = _regimes(_mover_table(META_WDR))
MOVERS_WDR["cover_m"] = MOVERS_WDR["condition"].map(WALL_DOSE_DESIGN["levels"]).astype(float)
for _n, _df in [("baseline_sweep", MOVERS_BSW), ("dose_response", MOVERS_WDR)]:
    _df.to_csv(os.path.join(MANIFEST_DIR, f"convoy3d_{_n}_movers.csv"), index=False)

# ---- self-tests of the designed sets ----------------------------------------------------------------------------
DESIGN_CHECKS = {}
_b = MOVERS_BSW[MOVERS_BSW["gt_n_eroded"] > 0]
DESIGN_CHECKS["bsw_n_base_movers"] = int(MOVERS_BSW["uid"].nunique())
DESIGN_CHECKS["bsw_s_identical_across_levels"] = bool((MOVERS_BSW.groupby("uid")["s"].nunique() == 1).all())
DESIGN_CHECKS["bsw_max_abs_xgeo_minus_xpar"] = float(np.nanmax(np.abs(_b["gt_x_geo"] - _b["x_par"]))) if len(_b) else np.nan
DESIGN_CHECKS["bsw_max_pixel_eps_px"] = float(np.nanmax(_b["gt_eps_max"])) if len(_b) else np.nan
_pm = MOVERS_BSW.pivot_table(index="uid", columns="condition", values="gt_parallax_med", aggfunc="first")
for _lv in BSW_LEVELS:
    if _lv in _pm and "B1" in _pm:
        _r = (_pm[_lv] / _pm["B1"]).replace([np.inf, -np.inf], np.nan)
        DESIGN_CHECKS[f"bsw_parallax_ratio_{_lv}_to_B1_median"] = float(np.nanmedian(_r))
_mm = MOVERS_BSW.pivot_table(index="uid", columns="condition", values="measurable", aggfunc="first").astype(float)
DESIGN_CHECKS["bsw_measurable_at_every_level"] = int((_mm[BSW_LEVELS].sum(1) == len(BSW_LEVELS)).sum()) if len(_mm) else 0
_w = MOVERS_WDR
DESIGN_CHECKS["wdr_n_base_movers"] = int(_w["uid"].nunique())
for _lv in WDR_LEVELS[1:]:
    _q = _w[_w["condition"] == _lv]
    DESIGN_CHECKS[f"wdr_{_lv}_wall_built"] = int(_q["has_wall"].astype(bool).sum())
    DESIGN_CHECKS[f"wdr_{_lv}_contact_hidden_both_views"] = int(((_q["contact_px0"] == 0) & (_q["contact_px1"] == 0)
                                                                 & _q["has_wall"].astype(bool)).sum())
_g = _w[_w["condition"] == "G"]
DESIGN_CHECKS["wdr_G_contact_visible_both_views"] = int(((_g["contact_px0"] >= PREREG["h13_contact_px_min"])
                                                         & (_g["contact_px1"] >= PREREG["h13_contact_px_min"])).sum())
# the dose as built: a wall must cover the whole visible base in both views, so it hides more than its nominal band at
# the mover's near end. Share of each mover's silhouette that is hidden (view 1) and the physical height of the walls.
for _lv in WDR_LEVELS:
    _q = _w[(_w["condition"] == _lv) & ((_w["has_wall"].astype(bool)) if _lv != "G" else True)]
    DESIGN_CHECKS[f"wdr_{_lv}_median_hidden_share"] = float(np.nanmedian(1.0 - _q["gt_visible_frac"])) if len(_q) else np.nan
    _hw = [w_["height"] for m_ in META_WDR if m_["condition"] == _lv for w_ in m_.get("scene", {}).get("walls", [])]
    DESIGN_CHECKS[f"wdr_{_lv}_median_wall_height_m"] = float(np.median(_hw)) if _hw else np.nan
TESTBED_CHECKS_V65 = DESIGN_CHECKS
with open(os.path.join(LOG_DIR, "design_checks.json"), "w") as f:
    json.dump(DESIGN_CHECKS, f, indent=2)
print(stage_table("render:").to_string(index=False))
print(f"\nBSW: {DESIGN_CHECKS['bsw_n_base_movers']} base movers in {N_BSW} scenes x {len(BSW_LEVELS)} baselines "
      f"({MOVERS_BSW[MOVERS_BSW['condition'] == 'B1'].groupby('cls').size().to_dict()}); s identical across levels: "
      f"{DESIGN_CHECKS['bsw_s_identical_across_levels']}; Proposition 1 on the renders: max |x_geo - x| = "
      f"{DESIGN_CHECKS['bsw_max_abs_xgeo_minus_xpar']:.1e}, max pixel |eps| = {DESIGN_CHECKS['bsw_max_pixel_eps_px']:.1e} px")
print("     median parallax relative to x1: " + ", ".join(f"{SWEEP_DESIGN['labels'][lv]} {DESIGN_CHECKS.get(f'bsw_parallax_ratio_{lv}_to_B1_median', np.nan):.2f}"
                                                        for lv in BSW_LEVELS)
      + f"; measurable at every baseline: {DESIGN_CHECKS['bsw_measurable_at_every_level']}")
print(f"WDR: {DESIGN_CHECKS['wdr_n_base_movers']} base movers in {N_WDR} scenes x {len(WDR_LEVELS)} wall heights; contact visible "
      f"without a wall (>= {PREREG['h13_contact_px_min']} px in both views): {DESIGN_CHECKS['wdr_G_contact_visible_both_views']}")
for _lv in WDR_LEVELS[1:]:
    print(f"     {WALL_DOSE_DESIGN['labels'][_lv]:>6} wall: built for {DESIGN_CHECKS[f'wdr_{_lv}_wall_built']}, contact hidden in both "
          f"views for {DESIGN_CHECKS[f'wdr_{_lv}_contact_hidden_both_views']}; median wall height "
          f"{DESIGN_CHECKS[f'wdr_{_lv}_median_wall_height_m']:.2f} m, median share of the mover hidden in view 1 "
          f"{DESIGN_CHECKS[f'wdr_{_lv}_median_hidden_share']:.2f} (no wall: {DESIGN_CHECKS['wdr_G_median_hidden_share']:.2f})")
assert DESIGN_CHECKS["bsw_s_identical_across_levels"]
assert not np.isfinite(DESIGN_CHECKS["bsw_max_abs_xgeo_minus_xpar"]) or DESIGN_CHECKS["bsw_max_abs_xgeo_minus_xpar"] < 1e-3

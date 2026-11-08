# PLACEHOLDER
# 5.2 Collect the per-stage results of every evaluated model into analysis tables. No model is in
#     memory from here on; memory is released and reported before the analysis starts.
release_memory(); _ = mem_report("analysis start")

def _read(path):
    return pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) > 5 else pd.DataFrame()

_imgs, _movs, _e4s = [], [], []
EXEMPLARS, RUN_TIMING = {}, {}
for _n in EVALUATED_MODELS:
    EXEMPLARS[_n], RUN_TIMING[_n] = {}, {}
    for _tb, _st in EVAL_STAGES[_n].items():
        for _lst, _f in ((_imgs, "images.csv"), (_movs, "movers.csv"), (_e4s, "e4.csv")):
            _df = _read(os.path.join(_st.out_dir, _f))
            if len(_df):
                _df["model"] = _n
                _lst.append(_df)
        _z = np.load(os.path.join(_st.out_dir, "exemplars.npz"))
        EXEMPLARS[_n].update({k: _z[k].astype(np.float32) for k in _z.files})
        with open(os.path.join(_st.out_dir, "done.json")) as _f:
            RUN_TIMING[_n][_tb] = json.load(_f)["seconds"]

def _cols(df, cols):
    return df.reindex(columns=cols)

IMG_RES = pd.concat(_imgs, ignore_index=True) if _imgs else pd.DataFrame()
_gtcols = ["uid", "cls", "style", "s", "x_par", "eps_target", "eps_center", "beyond", "motion_type", "Z_center",
           "gt_Z_true", "gt_eps_med", "gt_eps_max", "gt_x_geo", "gt_parallax_med", "gt_area_px", "gt_visible_frac",
           "gt_n_eroded", "regime", "measurable", "identifiable", "cam_kind", "focal", "baseline"]
_extra = ["testbed", "seq", "speed_ratio", "gt_occupancy", "gt_truncation", "gt_vis_in_b", "track", "label",
          "gt_contact", "gt_wheels_visible", "gt_wheels_in_view", "gt_n_lidar_px", "gt_occlusion", "state", "gt_dt_ms",
          "kind", "floor", "gt_height_above_floor", "gt_x_geo_raw"]
MOVERS_ALL = pd.concat([_cols(MOVERS_MAIN.assign(testbed="CONVOY-3D"), _gtcols + _extra),
                        _cols(MOVERS_REAL.assign(testbed="CONVOY-REAL"), _gtcols + _extra) if len(MOVERS_REAL) else None,
                        _cols(MOVERS_VK.assign(testbed="Virtual KITTI 2"), _gtcols + _extra) if len(MOVERS_VK) else None,
                        _cols(MOVERS_KT.assign(testbed="KITTI tracking"), _gtcols + _extra) if len(MOVERS_KT) else None,
                        _cols(MOVERS_AV2.assign(testbed="Argoverse 2"), _gtcols + _extra) if len(MOVERS_AV2) else None,
                        _cols(MOVERS_BONN.assign(testbed="Bonn RGB-D"), _gtcols + _extra) if len(MOVERS_BONN) else None],
                       ignore_index=True)
_mr = pd.concat(_movs, ignore_index=True) if _movs else pd.DataFrame(columns=["set", "uid", "condition"])
MOV_RES = (_mr[~_mr["set"].isin(["ABL", "RABL", "BSW", "WDR"])].drop(columns=["condition"], errors="ignore")
           .merge(MOVERS_ALL, on="uid", how="left"))
# E5 ablation: one row per (mover, condition, model), joined with the ground truth of that condition
_abl_cols = _gtcols + ["condition", "contact_px0", "contact_px1", "has_wall", "lift", "support"]
ABL_RES = _mr[_mr["set"] == "ABL"].merge(_cols(MOVERS_ABL, _abl_cols), on=["uid", "condition"], how="left")
# E5-REAL: one row per (mover, condition, model), joined with the ground truth of that condition
RABL_RES = (_mr[_mr["set"] == "RABL"].merge(_cols(MOVERS_RABL, _abl_cols + ["seq"]), on=["uid", "condition"], how="left")
            if len(MOVERS_RABL) else pd.DataFrame())
# v6.5 designed experiments: one row per (mover, level, model), joined with the ground truth of that level
_dsg_cols = _abl_cols + ["factor", "cover_m"]
BSW_RES = _mr[_mr["set"] == "BSW"].merge(_cols(MOVERS_BSW, _dsg_cols), on=["uid", "condition"], how="left")
WDR_RES = _mr[_mr["set"] == "WDR"].merge(_cols(MOVERS_WDR, _dsg_cols), on=["uid", "condition"], how="left")
E4_RES = (pd.concat(_e4s, ignore_index=True).merge(MOVERS_E4[["uid", "cls", "s", "x_par", "measurable", "identifiable",
                                                               "gt_parallax_med", "gtlast_eps_med", "omega_deg"]], on="uid", how="left")
          if _e4s else pd.DataFrame())
BASE_RES = BASELINES.merge(MOVERS_ALL, on="uid", how="left")
MODEL_TRAINING = {n: MODEL_STATUS[n]["training"] for n in EVALUATED_MODELS}
MODEL_FAMILY = {n: MODEL_STATUS[n]["family"] for n in EVALUATED_MODELS}
print(f"image-level rows: {len(IMG_RES)}  mover-level rows: {len(MOV_RES)} "
      f"({MOV_RES['set'].value_counts().to_dict() if len(MOV_RES) else {}})")
print(f"ablation rows: {len(ABL_RES)}  real-background ablation rows: {len(RABL_RES)}  E4 rows: {len(E4_RES)}  "
      f"baseline rows: {len(BASE_RES)}")
print(f"designed experiments (v6.5): baseline sweep rows {len(BSW_RES)}, dose-response rows {len(WDR_RES)}; "
      f"Bonn object rows {int((MOV_RES['set'] == 'BONN').sum()) if len(MOV_RES) else 0}")
print("inference time per model and testbed (min):")
print(pd.DataFrame(RUN_TIMING).T.div(60).round(1).to_string(na_rep="–"))

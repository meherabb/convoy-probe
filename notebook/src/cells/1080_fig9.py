# PLACEHOLDER
# Figure 9 (double column): qualitative. A parallel and a perpendicular CONVOY-3D mover and, when
# available, a moving Virtual KITTI 2 vehicle: image, exact depth, the model's two-view and single-image
# depth (same colour scale per row), and the dynamic scores M2 (rigid-flow residual) and M5 (two-view vs
# single-image depth), each divided by its 5%-FPR threshold on static pixels (1 = detection threshold).
_M8 = MODELS_OK["MAIN"] or EVALUATED_MODELS
_qmodel = next((m for m in ["DUSt3R", "MASt3R", "MonST3R"] if m in _M8), _M8[0])
_cand = MOV_RES[(MOV_RES["model"] == _qmodel) & (MOV_RES["set"] == "MAIN") & MOV_RES["scene"].isin(EXEMPLAR_MAIN)
                & MOV_RES["measurable"].astype(bool)]
_rows = []
_p = _cand[_cand["motion_type"] == "parallel"].sort_values("gt_area_px", ascending=False)
if len(_p):
    _rows.append(("parallel", "MAIN", _p.iloc[0]))
_q = _cand[_cand["regime"] == "perpendicular"].sort_values("gt_area_px", ascending=False)
if len(_q):
    _rows.append(("perpendicular", "MAIN", _q.iloc[0]))
if VK_STATUS["available"]:
    # driving pixel rule (>= 40 eroded visible px, as in Section 6.8b): almost no Virtual KITTI 2 vehicle meets the
    # CONVOY-3D measurability rule (D34), which left this row out in v6.3 (D54); a parallel vehicle is preferred
    _v = MOV_RES[(MOV_RES["model"] == _qmodel) & (MOV_RES["set"] == "VK") & MOV_RES["scene"].isin(EXEMPLAR_VK)
                 & (MOV_RES["gt_n_eroded"] >= PREREG["x6_min_eroded_px"])].copy()
    if len(_v):
        _v["_par"] = (_v["regime"] == "parallel").astype(int)
        _rows.append(("vehicle", "VK", _v.sort_values(["_par", "gt_area_px"], ascending=False).iloc[0]))

def _ratio_map(score, static_mask_half):
    s = np.asarray(score, np.float32)
    thr = np.nanpercentile(s[static_mask_half], 95) if static_mask_half.any() else np.nanmax(s)
    return np.clip(s / max(thr, 1e-6), 0, 2.5)

def _half(a, shape):
    return cv2.resize(np.asarray(a, np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

titles = ["image", "exact depth", "depth, two views", "depth, one view", "M2 / threshold", "M5 / threshold"]
_row_title = {"parallel": "Parallel", "perpendicular": "Perpendicular", "vehicle": "Driving"}
_row_cap = {"parallel": ("Parallel mover", "A CONVOY-3D mover with ε = 0: it leaves no geometric residual, so M2 cannot flag it."),
            "perpendicular": ("Perpendicular mover", "A mover off its epipolar line: M2 flags it whatever depth the model predicts."),
            "vehicle": ("Driving", "A moving Virtual KITTI 2 vehicle (exact depth and poses).")}
_nr = max(len(_rows), 1)
_th = 0.49; _tw = _th / 0.75; _gp = 0.035
_H9 = min(FIG_DOUBLE[1], 0.19 + _nr * _th + (_nr - 1) * 0.04 + 0.06)       # the canvas height follows the rows (D54)
fig = plt.figure(figsize=(FIG_DOUBLE[0], _H9))
_x0 = 0.74                                   # room for the row titles (v6.5)
_ytop = _H9 - 0.19
im_last = None
for i, (name, set_, row) in enumerate(_rows):
    ex = EXEMPLARS[_qmodel]
    if set_ == "MAIN":
        meta = META_MAIN[int(row["scene"])]
        r0 = load_frame(meta["dir"], 0)
        rgb, depth, mvmask = r0["rgb"], r0["depth"], r0["mover_id"] == int(row["mover"])
        sm = static_mask(r0, "synth"); key = f"MAIN-{int(row['scene'])}"
    else:
        meta = VK_PAIR_META[int(row["scene"])]
        vsc = VK_SCENE_OBJS[meta["seq"]]; g = vsc.pair_gt(*meta["frames"])
        rgb, depth, mvmask = vsc.rgb(meta["frames"][0]), g["depth"], g["mover_id"] == int(row["mover"])
        sm = vk_static_mask(g["depth"], g["inst"]); key = f"VK-{int(row['scene'])}"
    hs = ex[f"{key}|depth"].shape
    st_half = cv2.resize(sm.astype(np.uint8), (hs[1], hs[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    bb = [v * hs[0] / depth.shape[0] for v in mask_bbox(mvmask)]
    dfin = np.where(np.isfinite(depth), depth, np.nan)
    vmin = np.log(np.nanpercentile(dfin, 2)); vmax = np.log(np.nanpercentile(dfin, 97))
    m5 = np.abs(np.log(np.maximum(ex[f"{key}|depth"], 1e-6)) - np.log(np.maximum(ex[f"{key}|mono"], 1e-6)))
    tiles = [_half(rgb, hs) / 255.0, depth_to_rgb(_half(np.where(np.isfinite(depth), depth, 1e3), hs), vmin, vmax),
             depth_to_rgb(ex[f"{key}|depth"], vmin, vmax), depth_to_rgb(ex[f"{key}|mono"], vmin, vmax),
             _ratio_map(ex[f"{key}|m2"], st_half), _ratio_map(m5, st_half)]
    for j, t in enumerate(tiles):
        a = fig.add_axes([(_x0 + j * (_tw + _gp)) / 5.5, (_ytop - (i + 1) * _th - i * 0.04) / _H9, _tw / 5.5, _th / _H9])
        crop = crop_around(t, bb, (3, 4), pad=0.55)
        if j >= 4:
            im_last = a.imshow(crop, cmap="magma", vmin=0, vmax=2.5, aspect="auto")
        else:
            a.imshow(np.clip(crop, 0, 1), aspect="auto")
        clean_image_axes(a)
        if i == 0:
            a.set_title(titles[j], fontsize=5.6, pad=2)
        if j == 0:                                      # row titles (D60): "(a) Parallel", "(b) Perpendicular", "(c) Driving"
            lab = f"({'abc'[i]}) {_row_title[name]}\nε = {row['gt_eps_med']:.1f} px"
            a.text(-0.06, 0.5, lab, transform=a.transAxes, fontsize=5.3, ha="right", va="center")
if im_last is not None:
    _xr = (_x0 + 6 * _tw + 5 * _gp + 0.08) / 5.5
    cax = fig.add_axes([_xr, (_ytop - _nr * _th - (_nr - 1) * 0.04) / _H9, 0.012, (_nr * _th + (_nr - 1) * 0.04) / _H9])
    cb = fig.colorbar(im_last, cax=cax, ticks=[0, 1, 2])
    cb.ax.tick_params(labelsize=5.2, width=0.4, length=1.5); cb.outline.set_linewidth(0.4)
    cb.set_label("score / threshold", fontsize=5.4, labelpad=1)
save_figure(fig, "fig9_qualitative", "Qualitative: what geometry and the prior see",
            "Why do geometric masks miss parallel movers, and what does the single-image prediction add?",
            "Sections 6.1, 6.2 and 6.8", "double column", dict(model=_qmodel, movers=[r[2]["uid"] for r in _rows]),
            caption=" ".join(cap("abc"[i], *_row_cap[name]) for i, (name, _, _) in enumerate(_rows)) + " "
                    f"**Layout.** Crops around a mover ({mlab(_qmodel)}); depth panels share one colour scale per row. "
                    "**Reading the masks.** M2 and M5 are divided by their 5% false-positive threshold on the static pixels of "
                    "the same image, so values above 1 are detections. **What it shows.** A parallel mover leaves no geometric "
                    "residual; the disagreement between the two-view and the single-image depth (M5) can still mark it when the "
                    "model follows geometry.")

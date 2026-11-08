# PLACEHOLDER
# Figure 1 (teaser, double column). (a) Parallel motion as an exact counterfactual: a point moving
# by s*b and a static point at X/(1-s) give the same pixels in both views. (b) A parallel mover in
# CONVOY-3D: both frames, true depth and a model's depth on the same colour scale. (c) The probe:
# log depth error rho against the geometric answer x = -log(1-s) (binned medians): geometry alone
# gives the diagonal (beta = 1), the single-image prior gives zero (beta = 0).
def _pick_showcase(model):
    base = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN") & MOV_RES["scene"].isin(EXEMPLAR_MAIN)]
    for cond in [(base["motion_type"] == "parallel") & base["measurable"].astype(bool) & base["identifiable"].astype(bool)
                 & (base["s"] > 0.25) & (base["s"] < 0.9),
                 base["motion_type"] == "parallel", base["s"].notna()]:
        cand = base[cond]
        if len(cand):
            return cand.sort_values("gt_area_px", ascending=False).iloc[0]

_M1 = MODELS_OK["MAIN"] or EVALUATED_MODELS
_showcase_model = next((m for m in ["MASt3R", "DUSt3R", "VGGT-1B", "pi3", "MonST3R"] if m in _M1), _M1[0])
_sc_row = _pick_showcase(_showcase_model)
_meta = META_MAIN[int(_sc_row["scene"])]
_mv = _meta["scene"]["movers"][int(_sc_row["mover"])]
_r0 = load_frame(_meta["dir"], 0); _rgb1 = load_rgb(_meta["dir"], 1)
_pred = cv2.resize(EXEMPLARS[_showcase_model][f"MAIN-{int(_sc_row['scene'])}|depth"], (_r0["depth"].shape[1], _r0["depth"].shape[0]),
                   interpolation=cv2.INTER_LINEAR)
_bb = mask_bbox(_r0["mover_id"] == int(_sc_row["mover"])) or _mv["bbox0"]
_fin = np.isfinite(_r0["depth"])
_vmin = np.log(np.nanpercentile(_r0["depth"][_fin], 2)); _vmax = np.log(np.nanpercentile(_r0["depth"][_fin], 97))

fig = plt.figure(figsize=FIG_DOUBLE)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.3], left=0.012, right=0.985, bottom=0.2, top=0.9, wspace=0.28)
# (a) schematic, top view: forward (camera translation) to the right, lateral upwards
ax = fig.add_subplot(gs[0]); ax.set_anchor("W")
fig_panel_title(fig, 0.004, 0.985, "a", "The counterfactual")
s_, b_ = 0.5, 2.0
X = np.array([4.0, 2.2]); Xm = X + np.array([s_ * b_, 0.0]); Xs = X / (1 - s_)
C1, C2 = np.array([0.0, 0.0]), np.array([b_, 0.0])
for C in (C1, C2):
    ax.plot([C[0], Xs[0]], [C[1], Xs[1]], color="#7F7F7F", lw=0.55, ls=(0, (3, 2)))
ax.annotate("", xy=C2, xytext=C1, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#0072B2", mutation_scale=6,
                                                  shrinkA=0, shrinkB=2))
ax.annotate("", xy=Xm, xytext=X, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#D55E00", mutation_scale=6,
                                                 shrinkA=2, shrinkB=2))
for P in (C1, C2):
    ax.add_patch(mpatches.Polygon([[P[0] - 0.32, P[1] - 0.36], [P[0] - 0.32, P[1] + 0.36], [P[0] + 0.3, P[1]]],
                                  closed=True, fc="#0072B2", ec="none"))
ax.plot(*X, "o", color="#D55E00", ms=3.2); ax.plot(*Xm, "o", mfc="white", mec="#D55E00", ms=3.2, mew=0.7)
ax.plot(*Xs, "*", color="black", ms=5.5)
ax.text(C1[0], C1[1] - 0.95, "$C_1$", fontsize=6, ha="center")
ax.text(C2[0], C2[1] - 0.95, "$C_2$", fontsize=6, ha="center")
ax.text(1.0, -0.62, "$b$", fontsize=6, color="#0072B2", ha="center")
ax.text(X[0] - 0.35, X[1] + 0.5, "$X$", fontsize=6, color="#D55E00", ha="center")
ax.text(Xm[0] + 1.25, Xm[1] - 0.55, "$X + sb$", fontsize=6, color="#D55E00", ha="center")
ax.text(Xs[0] - 0.3, Xs[1] + 0.55, "static $X/(1-s)$", fontsize=6, ha="center")
ax.set_xlim(-0.8, 9.8); ax.set_ylim(-1.35, 5.6); ax.set_aspect("equal")
clean_image_axes(ax, frame=False)
ax.text(0.5, -0.03, "moving and static point:\nthe same pixels in both views", transform=ax.transAxes, fontsize=5.6,
        ha="center", va="top")
# (b) mover crops: frame 1, frame 2, true depth, model depth
sub = gs[1].subgridspec(2, 2, wspace=0.04, hspace=0.05)
tiles = [(crop_around(_r0["rgb"], _bb, (3, 4)), "frame 1"), (crop_around(_rgb1, _bb, (3, 4)), "frame 2"),
         (crop_around(depth_to_rgb(_r0["depth"], _vmin, _vmax), _bb, (3, 4)), "true depth"),
         (crop_around(depth_to_rgb(_pred, _vmin, _vmax), _bb, (3, 4)), f"{mlab(_showcase_model)}, two views")]
for k, (img, lab) in enumerate(tiles):
    a2 = fig.add_subplot(sub[k // 2, k % 2])
    a2.imshow(img); clean_image_axes(a2); tag(a2, lab)
_ax_b = fig.axes[1]
fig_panel_title(fig, _ax_b.get_position().x0 - 0.012, 0.985, "b", "A parallel mover")
fig.text((fig.axes[1].get_position().x0 + fig.axes[2].get_position().x1) / 2, 0.075,
         f"parallel mover, s = {_mv['s']:.2f}: true Z = {_sc_row['gt_Z_true']:.1f} m, geometric answer Z/(1 − s) = "
         f"{_sc_row['gt_Z_true'] / (1 - _mv['s']):.1f} m", fontsize=5.4, ha="center")
# (c) the probe
ax = fig.add_subplot(gs[2])
fig_panel_title(fig, ax.get_position().x0 - 0.045, 0.985, "c", "Prior or geometry")
xs = np.linspace(-0.55, 1.75, 10)
ax.plot(xs, xs, color=CONTROL_COLOR, lw=0.7, ls=(0, (4, 2)))
ax.axhline(0, color=MONO_COLOR, lw=0.7, ls=":")
ax.text(0.62, 0.72, "geometry only, β = 1", fontsize=5.2, ha="center", va="bottom", color=CONTROL_COLOR, rotation=45,
        rotation_mode="anchor", transform_rotates_text=True)
ax.text(1.72, -0.06, "prior only, β = 0", fontsize=5.2, ha="right", va="top", color="#666666")
fig1_data = {}
for model in MODELS_OK["MAIN"]:
    d = E1_DATA[model]
    if len(d) < 6:
        continue
    q = pd.qcut(d["x_par"].rank(method="first"), min(6, len(d) // 3), labels=False)
    g = d.groupby(q).agg(x=("x_par", "median"), y=("rho", "median"))
    ax.plot(g["x"], g["y"], marker=MODEL_MARKERS[model], color=mcol(model), lw=0.8, ms=2.6, label=mlab(model))
    fig1_data[model] = g.to_dict("list")
ax.set_xlabel("geometric answer x = −log(1 − s)"); ax.set_ylabel("log depth error ρ")
ax.set_xlim(-0.55, 1.75); ax.set_ylim(-0.7, 1.75)
ax.legend(loc="upper left", ncol=2, fontsize=5.0, handlelength=1.2, columnspacing=0.7)
save_figure(fig, "fig1_teaser", "Parallel motion as an exact counterfactual",
            "How much of the cross-view geometry do feed-forward models use for a moving object?",
            "E1, CONVOY-3D", "double column", dict(binned=fig1_data, showcase=dict(uid=_sc_row["uid"], model=_showcase_model)),
            caption=cap("a", "The counterfactual", "A point moving parallel to the camera baseline by s b and a static point at "
                        "X/(1-s) project to the same pixels in both views.")
                    + " " + cap("b", "A parallel mover", "A mover in CONVOY-3D and the depth a model predicts for it from both views.")
                    + " " + cap("c", "Prior or geometry", "Log depth error against the geometric answer x = -log(1-s), binned medians "
                                "over purely parallel movers: a model that used only geometry would follow the diagonal, a model that "
                                "uses only its single-image prior stays at zero.") + gate_caption_note("MAIN"))

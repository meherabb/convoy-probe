# PLACEHOLDER
# Figure 11 (double column, v6.5): the static twin. (a) Proposition 4 in a side view: a grounded object moving with the
# camera (s = 0.5) and its static twin, twice as large, sunk below the road. (b) Twin height over camera height against s:
# the law -s/(1-s) and the Argoverse 2 vehicles in the parallel regime. (c) Recall of the epipolar test, the twin test and
# either, by s, with the detection curve implied by parked vehicles. (d) v6.6: the parallel regime with labelled
# correspondences and with optical flow and the LiDAR road plane (H14); by motion regime if the flow test is not in the run.
_has_tw = len(TWIN_RECALL) > 0 and bool(TWIN_THRESH)
fig = plt.figure(figsize=FIG_DOUBLE)
# (a) schematic, side view: depth to the right, height up; camera height H = 1.5, s = 0.5. The drawing is anchored to the
# top of the plot panels and shares their title line.
# without Argoverse 2 camera heights only (a) and (b) are drawn, centred and wider
_xa, _wa = (0.005, 0.27) if _has_tw else (0.1, 0.36)
title_axes(fig, [_xa, 0.25, _wa, 0.62], "a", "The static twin")
ax = fig.add_axes([_xa, 0.14, _wa, 0.71])
ax.set_anchor("N")
_H, _s = 1.5, 0.5
_C = np.array([0.0, _H])
_obj = np.array([[2.2, 0.0], [3.4, 0.0], [3.4, 1.2], [2.2, 1.2]])
_twin = _C + (_obj - _C) / (1 - _s)
ax.add_patch(mpatches.Rectangle((-0.75, -2.15), 8.1, 2.15, fc="#EDEDED", ec="none", zorder=0))
ax.plot([-0.75, 7.35], [0, 0], color="#555555", lw=0.8, zorder=1)
ax.text(0.75, -0.12, "road", fontsize=5.2, color="#555555", ha="left", va="top")
for p_ in (_twin[0], _twin[1]):
    ax.plot([_C[0], p_[0]], [_C[1], p_[1]], color="#7F7F7F", lw=0.5, ls=(0, (3, 2)), zorder=2)
ax.add_patch(mpatches.Polygon(_obj, closed=True, fc="#D55E00", ec="none", alpha=0.85, zorder=3))
ax.add_patch(mpatches.Polygon(_twin, closed=True, fc="none", ec="black", lw=0.7, ls=(0, (2.5, 1.5)), zorder=3))
ax.add_patch(mpatches.Polygon([[-0.3, _H - 0.28], [-0.3, _H + 0.28], [0.25, _H]], closed=True, fc="#0072B2", ec="none", zorder=4))
ax.annotate("", xy=(1.25, _H), xytext=(0.3, _H), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#0072B2", mutation_scale=5))
ax.annotate("", xy=(3.95, 0.6), xytext=(3.45, 0.6), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#D55E00", mutation_scale=5))
ax.plot([-0.5, -0.5], [0, _H], color="#0072B2", lw=0.6)
ax.text(-0.43, _H / 2, "H", fontsize=5.6, color="#0072B2", ha="left", va="center")
ax.text(2.8, 1.3, "object", fontsize=5.2, color="#D55E00", ha="center", va="bottom")
ax.text(5.6, _twin[2, 1] + 0.1, "static twin", fontsize=5.2, ha="center", va="bottom")
ax.annotate("", xy=(4.05, _twin[0, 1]), xytext=(4.05, 0.0), arrowprops=dict(arrowstyle="<->", lw=0.6, color="black", mutation_scale=4))
ax.text(4.05, _twin[0, 1] - 0.1, "H s/(1−s)", fontsize=5.0, ha="center", va="top")
ax.set_xlim(-0.75, 7.35); ax.set_ylim(-2.15, 1.95); ax.set_aspect("equal")
clean_image_axes(ax, frame=False)
ax.text(0.5, -0.03, "s = 0.5: twice as large, sunk below the road", transform=ax.transAxes, fontsize=5.0, ha="center", va="top")
# (b) twin height against s
ax = fig.add_axes([0.355, 0.25, 0.18, 0.62] if _has_tw else [0.6, 0.25, 0.3, 0.62])
panel_title(ax, "b", "Twin height", x=-0.34 if _has_tw else -0.2)
_sg = np.linspace(-0.9, 0.83, 200)
ax.plot(_sg, -_sg / (1 - _sg), color=CONTROL_COLOR, lw=0.9, zorder=3, label="−s/(1−s)")
ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0)
if _has_tw and len(TWIN_VEH):
    _pv = TWIN_VEH[(TWIN_VEH["regime"] == "parallel") & np.isfinite(TWIN_VEH["h_twin"])]
    ax.scatter(_pv["s"], _pv["h_twin"] / _pv["H_cam"], s=2.5, color="#0072B2", alpha=0.6, lw=0, zorder=2, label="Argoverse 2")
    _bh = TWIN_THRESH["height_m"] / TWIN_THRESH["H_median_m"]
    ax.axhspan(-_bh, _bh, color="#F2F2F2", zorder=0, lw=0)
ax.set_xlim(-0.9, 0.85); ax.set_ylim(-4.2, 1.0); ax.set_yticks([-4, -3, -2, -1, 0])
ax.set_xlabel("parallel fraction s"); ax.set_ylabel("twin contact height / H", labelpad=1)
ax.legend(loc="lower left", fontsize=4.9, handlelength=1.2, markerscale=2)
if _has_tw:
    # (c) recall by s
    ax = fig.add_axes([0.6, 0.25, 0.2, 0.62])
    panel_title(ax, "c", "Twin test by s", x=-0.28)
    _ts = TWIN_S[TWIN_S["n"] >= 5]
    for col, colr, lab, ls in (("det_epipolar", MONO_COLOR, "epipolar", "-"), ("det_twin", "#0072B2", "twin", "-"),
                               ("det_either", CONTROL_COLOR, "either", (0, (3, 1.5)))):
        ax.errorbar(_ts["s_median"], _ts[col], yerr=[(_ts[col] - _ts[col + "_lo"]).clip(lower=0), (_ts[col + "_hi"] - _ts[col]).clip(lower=0)],
                    color=colr, marker="o", ms=2.0, lw=0.8, ls=ls, capsize=0.8, label=lab)
    ax.plot(TWIN_THEORY["s"], TWIN_THEORY["p_detect_twin"], color="#0072B2", lw=0.6, alpha=0.6, ls=(0, (1, 1)))
    ax.set_xlim(-0.9, 0.95); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("parallel fraction s"); ax.set_ylabel("share flagged", labelpad=1)
    # (d) parallel regime: labelled correspondences against optical flow (v6.6), else by regime
    _ftp11 = FTWIN_RECALL[(FTWIN_RECALL["regime"] == "parallel") & (FTWIN_RECALL["n"] > 0)] if len(FTWIN_RECALL) else FTWIN_RECALL
    _flow11 = len(_ftp11) > 0
    ax = fig.add_axes([0.86, 0.25, 0.13, 0.62])
    _tests11 = (("det_epipolar", MONO_COLOR), ("det_twin", "#0072B2"), ("det_either", CONTROL_COLOR))
    if _flow11:
        panel_title(ax, "d", "Labels vs flow", x=-0.45)
        _grp = [TWIN_RECALL[TWIN_RECALL["regime"] == "parallel"].iloc[0], _ftp11.iloc[0]]
    else:
        panel_title(ax, "d", "By regime", x=-0.45)
        _grp = [r for _, r in TWIN_RECALL[TWIN_RECALL["regime"] != "all"].set_index("regime")
                .reindex(["parallel", "intermediate", "perpendicular"]).reset_index().iterrows()]
    for k, (col, colr) in enumerate(_tests11):
        _v = np.array([float(g[col]) if np.isfinite(g[col]) else 0.0 for g in _grp])
        _lo = np.array([float(g[col] - g[col + "_lo"]) if np.isfinite(g[col + "_lo"]) else 0.0 for g in _grp]).clip(0)
        _hi = np.array([float(g[col + "_hi"] - g[col]) if np.isfinite(g[col + "_hi"]) else 0.0 for g in _grp]).clip(0)
        ax.bar(np.arange(len(_grp)) + (k - 1) * 0.26, _v, width=0.24, color=colr, lw=0, yerr=[_lo, _hi],
               error_kw=dict(lw=0.5, capsize=0.8, ecolor="#333333"))
    ax.set_xticks(range(len(_grp)))
    ax.set_xticklabels(["labels", "flow"] if _flow11 else ["par.", "int.", "perp."], fontsize=5.2)
    ax.set_ylim(0, 1.12); ax.set_ylabel("share flagged", labelpad=1)
    if _flow11 and np.isfinite(H14_RESULT.get("estimate", np.nan)):
        ax.text(0.5, 0.985, f"H14 {H14_RESULT['estimate']:+.2f}", transform=ax.transAxes, fontsize=4.8, ha="center", va="top",
                color="#555555")
    # one key for (c) and (d)
    _hl11 = [Line2D([], [], color=MONO_COLOR, marker="o", ms=2.0, lw=0.8, label="epipolar"),
             Line2D([], [], color="#0072B2", marker="o", ms=2.0, lw=0.8, label="twin"),
             Line2D([], [], color=CONTROL_COLOR, marker="o", ms=2.0, lw=0.8, ls=(0, (3, 1.5)), label="either"),
             Line2D([], [], color="#0072B2", lw=0.6, alpha=0.6, ls=(0, (1, 1)), label="twin, parked-noise model")]
    fig.legend(handles=_hl11, loc="lower center", bbox_to_anchor=(0.78, 0.0), ncol=4, fontsize=4.8, handletextpad=0.3,
               columnspacing=0.9, handlelength=1.6)
_cap11 = (cap("a", "The static twin", "Side view of Proposition 4: a grounded object moving with the camera (s = 0.5) is "
              "explained by a static twin that is twice as large and whose contact lies H s/(1-s) below the road (H: camera height "
              "above the object's contact plane); the road would hide it. An oncoming object (s < 0) has a twin floating above the road.")
          + " " + cap("b", "Twin height", "Height of the twin's contact over H, against s: the law -s/(1-s)"
                      + (" and Argoverse 2 vehicles in the parallel regime (camera height from each vehicle's labelled box, x_geo from "
                         "its LiDAR pixels and labelled motion); shaded: the 95% range of parked vehicles." if _has_tw else ".")))
if _has_tw:
    _trp = TWIN_RECALL[(TWIN_RECALL["regime"] == "parallel") & (TWIN_RECALL["n"] > 0)]
    _trp = _trp.iloc[0] if len(_trp) else None
    _cap11 += (" " + cap("c", "Twin test by s", f"Share of moving Argoverse 2 vehicles flagged by the epipolar test (|eps| above its "
                         f"95th percentile over parked vehicles, {TWIN_THRESH['eps_px']:.2f} px), by the twin test (|twin height| above "
                         f"its 95th percentile, {TWIN_THRESH['height_m']:.2f} m) and by either, binned by s; dotted: the twin test's "
                         "detection curve for exactly parallel motion implied by the parked vehicles.")
               + " " + (cap("d", "Labels vs flow", "Parallel-regime vehicles flagged by each test (bars: epipolar, twin, either; "
                                "95% intervals over vehicle tracks) with the exact correspondences of panels (b-c) and with optical flow "
                                f"({FTWIN_THRESH.get('flow', '').split(' (')[0]}, forward-backward checked) and the road plane fitted to "
                                "the LiDAR returns instead of the labelled box; both at 5% false positives on parked vehicles. H14: "
                                + (f"twin minus epipolar with flow {H14_RESULT['estimate']:+.2f} [{H14_RESULT['ci_lo']:+.2f}, "
                                   f"{H14_RESULT['ci_hi']:+.2f}] over {H14_RESULT['n']} vehicles, {H14_RESULT['verdict']}."
                                   if np.isfinite(H14_RESULT.get("estimate", np.nan))
                                   else f"{H14_RESULT.get('verdict', 'NOT TESTABLE')} ({H14_RESULT.get('n', 0)} vehicles)."))
                        if _flow11 else
                        cap("d", "By regime", "The same shares by motion regime (bars: epipolar, twin, either)"
                            + (f"; of the {int(_trp['n_epi_miss'])} parallel-regime vehicles the epipolar test misses, the twin test "
                               f"flags {100 * _trp['twin_given_epi_miss']:.0f}%" if _trp is not None and _trp["n_epi_miss"] else "")
                            + ". Both tests use the same exact correspondences from labelled boxes, so the panels compare the tests, "
                              "not a deployed detector; 95% intervals over vehicle tracks.")))
save_figure(fig, "fig11_static_twin", "The static twin and the ground plane",
            "Where is the static twin of a grounded mover, and how much invisible motion does it expose?", "Sections 1 and 6.13",
            "double column", dict(thresholds=TWIN_THRESH, by_regime=TWIN_RECALL.to_dict("records"), by_s=TWIN_S.to_dict("records"),
                                  curve=TWIN_THEORY.to_dict("records"), flow_thresholds=FTWIN_THRESH,
                                  flow_by_regime=FTWIN_RECALL.to_dict("records"), H14=H14_RESULT), caption=_cap11)

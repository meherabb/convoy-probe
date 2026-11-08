# PLACEHOLDER
# Figure 7 (double column): driving. (a) A Virtual KITTI 2 frame with its moving vehicles outlined by
# exact regime. (b) Motion that geometry cannot see: share of moving-vehicle tracks that a static
# point explains within 1 px over K = 2, 4, 8 frames (one metric for all bars). (c) Exploratory:
# g of vehicles keeping pace with the camera (0.85 <= s < 1): Virtual KITTI 2 and the real-driving data
# with LiDAR (Argoverse 2; KITTI tracking when attached). The H9 slope is reported in the tables only;
# the ground-contact contrast in traffic moved to Figure 10b in v6.4, next to the other testbeds.
_X6P = X6_PACE.copy() if len(X6_PACE) else X6_PACE
_has_p = len(_X6P) > 0 and np.isfinite(_X6P["median_g"]).any()
_P6 = {"img": [0.006, 0.2, 0.288, 0.66], "prev": [0.47, 0.24, 0.16, 0.62]}
if _has_p:
    _P6["pace"] = [0.775, 0.24, 0.215, 0.62]
_L6 = {k: "(" + "abc"[i] + ")" for i, k in enumerate(_P6)}

def _fig6_image(fig):
    cnt = MOVERS_VK[MOVERS_VK["measurable"].astype(bool)].groupby("scene").agg(n=("regime", "size"), k=("regime", "nunique"))
    best = int(cnt.sort_values(["k", "n"]).index[-1]) if len(cnt) else int(MOVERS_VK["scene"].iloc[0])
    vm = VK_PAIR_META[best]; vs = VK_SCENE_OBJS[vm["seq"]]; vg = vs.pair_gt(*vm["frames"])
    a = fig.add_axes(_P6["img"])
    a.imshow(vs.rgb(vm["frames"][0]), aspect="auto"); clean_image_axes(a)
    for k in range(len(vg["movers"])):
        reg = MOVERS_VK[(MOVERS_VK["scene"] == best) & (MOVERS_VK["mover"] == k)]["regime"]
        if len(reg) and (vg["mover_id"] == k).any():
            a.contour((vg["mover_id"] == k).astype(float), levels=[0.5], colors=[REGIME_COLORS[reg.iloc[0]]], linewidths=0.8)
    tag(a, f"{vm['seq']}, frames {vm['frames'][0]}–{vm['frames'][1]}", fs=4.8)
    panel_title(a, _L6["img"][1], "Exact regimes")
    return vm

def _fig7_prev(fig):
    ax = fig.add_axes(_P6["prev"])
    panel_title(ax, _L6["prev"][1], "Invisible motion", x=-0.9)
    k = KVIEW_SUMMARY.sort_values("K")
    yb = np.arange(len(k))[::-1]
    ax.barh(yb, k["share_static_fit_below_1px"], color="#0072B2", height=0.62)
    for y_, v in zip(yb, k["share_static_fit_below_1px"]):
        inside = v > 0.72
        ax.text(v - 0.03 if inside else v + 0.03, y_, f"{100 * v:.0f}%", va="center", ha="right" if inside else "left",
                fontsize=5.0, color="white" if inside else "black")
    ax.set_yticks(yb); ax.set_yticklabels([f"{int(K)} frames" for K in k["K"]], fontsize=5.2)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_ylim(-0.6, len(k) - 0.4); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("vehicle tracks fit\nby a static point", labelpad=1, fontsize=5.6)

def _g_axis(ax):
    ax.axhline(0, color=MONO_COLOR, lw=0.6, ls=":")
    ax.axhline(1, color=CONTROL_COLOR, lw=0.6, ls=(0, (4, 2)))
    ax.set_ylim(-0.35, 1.15); ax.set_ylabel("geometry share g", labelpad=1)
    ax.text(1.0, 1.0, "geometry", transform=ax.get_yaxis_transform(), fontsize=4.8, ha="right", va="bottom", color="#555555")
    # at the left end of the line, clear of the markers, which sit near 0 when a model follows its prior
    ax.text(0.015, 0.0, "prior", transform=ax.get_yaxis_transform(), fontsize=4.8, ha="left", va="bottom", color="#555555")

def _fig7_pace(fig):
    ax = fig.add_axes(_P6["pace"])
    panel_title(ax, _L6["pace"][1], "The convoy case", x=-0.3)
    sets = [d for d in ("Virtual KITTI 2", "Argoverse 2", "KITTI tracking") if (_X6P["dataset"] == d).any()]
    ms = [m for m in EVALUATED_MODELS if (_X6P["model"] == m).any()]
    for k_, dset in enumerate(sets):
        for i, m in enumerate(ms):
            r = _X6P[(_X6P["dataset"] == dset) & (_X6P["model"] == m)]
            if not len(r) or not np.isfinite(r["median_g"].iloc[0]):
                continue
            r = r.iloc[0]
            ax.errorbar(k_ + (i - (len(ms) - 1) / 2) * 0.09, r["median_g"], yerr=ci_err(r["median_g"], r["ci_lo"], r["ci_hi"]),
                        fmt=MODEL_MARKERS[m], color=mcol(m), ms=2.2, capsize=0.8, lw=0.6,
                        mfc="white" if dset == "Virtual KITTI 2" else mcol(m), mew=0.6)
    _g_axis(ax)
    _xl = {"Virtual KITTI 2": "VKITTI 2", "Argoverse 2": "Argoverse 2\n(real)", "KITTI tracking": "KITTI\n(real)"}
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([_xl[d] for d in sets], fontsize=5.2)
    ax.set_xlim(-0.5, len(sets) - 0.5)

_K6 = sorted(int(k) for k in KVIEW_SUMMARY["K"]) if len(KVIEW_SUMMARY) else []
if VK_STATUS["available"] and len(MOVERS_VK):
    fig = plt.figure(figsize=FIG_DOUBLE)
    _vm = _fig6_image(fig)
    _fig7_prev(fig)
    if _has_p:
        _fig7_pace(fig)
    _h = [Line2D([], [], color=cc, lw=1.0, label=rg) for rg, cc in REGIME_COLORS.items()]
    _mm = [m for m in EVALUATED_MODELS if len(_X6P) and (_X6P["model"] == m).any()]
    _h += [Line2D([], [], ls="", marker=MODEL_MARKERS[m], color=mcol(m), ms=2.6, label=mlab(m)) for m in _mm]
    fig.legend(handles=_h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(_h), fontsize=4.9, handletextpad=0.25,
               columnspacing=0.75, handlelength=1.1)
    _pv = (np.average(PREVALENCE_VK["frac_parallel"], weights=PREVALENCE_VK["n_moving_pixels"]) if len(PREVALENCE_VK) else np.nan)
    _cap = (cap(_L6["img"][1], "Exact regimes", "A Virtual KITTI 2 frame with its moving vehicles outlined by exact motion regime.")
            + " " + cap(_L6["prev"][1], "Invisible motion", "Share of moving-vehicle tracks whose image track over "
                        + ", ".join(str(k) for k in _K6) + " frames a static point explains within 1 px"
                        + (f" (between two views, {100 * _pv:.0f}% of moving-vehicle pixels lie within 1 px of their epipolar line)."
                           if np.isfinite(_pv) else ".")))
    if _has_p:
        _real = [d for d in ("Argoverse 2", "KITTI tracking") if (_X6P["dataset"] == d).any()]
        _cap += " " + cap(_L6["pace"][1], "The convoy case (exploratory)", "Median per-object geometry share g (0: the model's own "
                          "single-image depth, 1: the geometric answer) of vehicles keeping pace with the camera (0.85 ≤ s < 1, "
                          "whose geometric answer lies at least 6.7 times their distance); open markers: Virtual KITTI 2"
                          + (", filled: real images with LiDAR depth (" + " and ".join(_real) + ")" if _real else "") + ".")
    _cap += (" Intervals are 95% cluster-bootstrap intervals over vehicle tracks."
             + gate_caption_note("VK", *[k for k, _, _ in REAL_DRIVE if _has_p and (_X6P["set"] == k).any()]))
    save_figure(fig, "fig7_driving", "Driving: invisible motion and the convoy case",
                "How common is motion that geometry cannot see, and where do the models place vehicles that keep pace with the camera?",
                "Sections 6.8 and 6.8b", "double column",
                dict(kview=KVIEW_SUMMARY.to_dict("records"), pace=_X6P.to_dict("records") if _has_p else [],
                     prevalence=PREVALENCE_ALL.to_dict("records"), exemplar=_vm),
                caption=_cap)
else:
    print("Figure 7 is not drawn: Virtual KITTI 2 is not part of this run (" + VK_STATUS["note"] + ").")

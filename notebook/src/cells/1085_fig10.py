# PLACEHOLDER
# Figure 10 (double column, v6.4, descriptive): one cue across the testbeds. (a) The slope beta_s of grounded objects
# and of unsupported or hidden-contact objects in every testbed with designed motion (CONVOY-3D by class, CONVOY-REAL,
# whose movers are all floating crates, E5, E5-REAL). (b) The same split in traffic: the median per-object share g of
# parallel vehicles whose wheel contact is visible or hidden (Virtual KITTI 2; real Argoverse 2 vehicles). (c) What
# the slope costs: the depth overestimate the E5 slopes imply for an object keeping pace with the camera (s = 0.85).
_SL = SUPPORT_LEVELS.copy() if len(SUPPORT_LEVELS) else pd.DataFrame(columns=["testbed", "condition", "model", "estimate"])

def _has_cat(tb, cond):
    return bool(((_SL["testbed"] == tb) & (_SL["condition"] == cond) & np.isfinite(pd.to_numeric(_SL["estimate"]))).any())

# tick codes, as in Figure 4: G = ground contact in view (shaded), F = floating or lifted, CH = contact hidden
_A_GROUPS = [("CONVOY-3D", [("cars and crates", "G"), ("floating crates", "F")]),
             ("CONVOY-REAL", [("floating crates (every mover)", "F")]),
             ("E5", [(ABLATION_DESIGN["labels"][c], c) for c in ("G", "CH")]),
             ("E5-REAL", [(RABL_DESIGN["labels"][c], c) for c in ("G", "F")])]
_TB_KEY = {"E5": "E5 ablation"}
_A_GROUPS = [(tb, [c for c in cats if _has_cat(_TB_KEY.get(tb, tb), c[0])]) for tb, cats in _A_GROUPS]
_A_GROUPS = [(tb, cats) for tb, cats in _A_GROUPS if cats]
_B_GROUPS = [(ds, [(f"contact {c}", code) for c, code in (("visible", "G"), ("hidden", "CH")) if _has_cat(ds, f"contact {c}")])
             for ds in ("Virtual KITTI 2", "Argoverse 2")]
_B_GROUPS = [(ds, cats) for ds, cats in _B_GROUPS if cats]
_C_CATS = [(ABLATION_DESIGN["labels"][c], c) for c in ("G", "CH", "F") if _has_cat("E5 ablation", ABLATION_DESIGN["labels"][c])]
_M10 = [m for m in EVALUATED_MODELS if (_SL["model"] == m).any()]

def _grouped_positions(groups, gap=0.5):
    pos, x = [], 0.0
    for tb, cats in groups:
        xs = [x + k for k in range(len(cats))]
        pos.append((tb, cats, xs))
        x = xs[-1] + 1.0 + gap
    return pos

def _support_of(tb, cond):
    s_ = _SL[(_SL["testbed"] == tb) & (_SL["condition"] == cond)]["support"]
    return s_.iloc[0] if len(s_) else ""

def _dots(ax, tb, cond, x, col_est="estimate", col_lo="ci_lo", col_hi="ci_hi"):
    for i, m in enumerate(_M10):
        r = _SL[(_SL["testbed"] == tb) & (_SL["condition"] == cond) & (_SL["model"] == m)]
        if not len(r) or not np.isfinite(r[col_est].iloc[0]):
            continue
        r = r.iloc[0]
        ax.errorbar(x + (i - (len(_M10) - 1) / 2) * 0.12, r[col_est], yerr=ci_err(r[col_est], r[col_lo], r[col_hi]),
                    fmt=MODEL_MARKERS[m], color=mcol(m), ms=2.2, capsize=0.8, lw=0.6)

def _group_axis(ax, pos, tb_key=lambda tb: tb, names=None):
    ticks, labels = [], []
    for tb, cats, xs in pos:
        for (cond, lab), x in zip(cats, xs):
            if _support_of(tb_key(tb), cond) == "grounded":
                ax.axvspan(x - 0.5, x + 0.5, color=SUPPORT_SHADE, lw=0, zorder=0)
            ticks.append(x); labels.append(lab)
        name = (names or {}).get(tb, tb)
        if len(cats) == 1 and len(name) > 8 and "-" in name:      # a one-column group gets its long name on two lines
            name = name.replace("-", "-\n", 1)
        ax.text((xs[0] + xs[-1]) / 2, 1.0, name, transform=ax.get_xaxis_transform(), fontsize=5.0,
                ha="center", va="bottom", color="#555555", linespacing=1.0)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=5.4)
    ax.set_xlim(pos[0][2][0] - 0.55, pos[-1][2][-1] + 0.55)
    ax.tick_params(axis="x", length=0)

if _A_GROUPS and _M10:
    fig = plt.figure(figsize=FIG_DOUBLE)
    _panels = ["a"] + (["b"] if _B_GROUPS else []) + (["c"] if _C_CATS else [])
    _lab = {k: "abc"[i] for i, k in enumerate(_panels)}
    _wA = 0.405 if len(_panels) == 3 else 0.55 if len(_panels) == 2 else 0.86
    # (a) slopes by support in the testbeds with designed motion
    ax = fig.add_axes([0.072, 0.2, _wA, 0.645])
    fig_panel_title(fig, 0.004, 0.985, _lab["a"], "Support decides the slope")
    _pa = _grouped_positions(_A_GROUPS)
    for tb, cats, xs in _pa:
        for (cond, _), x in zip(cats, xs):
            _dots(ax, _TB_KEY.get(tb, tb), cond, x)
    _group_axis(ax, _pa, tb_key=lambda tb: _TB_KEY.get(tb, tb))
    ax.axhline(0, color=MONO_COLOR, lw=0.6, ls=":", zorder=1)
    _sa = _SL[_SL["statistic"] == "beta_s"]
    ax.set_ylim(min(-0.2, float(np.nanmin(_sa["ci_lo"])) - 0.05 if _sa["ci_lo"].notna().any() else -0.2),
                max(1.0, float(np.nanmax(_sa["ci_hi"])) + 0.05 if _sa["ci_hi"].notna().any() else 1.0))
    ax.set_ylabel("β$_s$ (95% CI)", labelpad=1)
    _x_right = 0.072 + _wA
    # (b) driving: per-object share by wheel contact
    if _B_GROUPS:
        _xb = _x_right + 0.085
        ax = fig.add_axes([_xb, 0.2, 0.17, 0.645])
        fig_panel_title(fig, _xb - 0.07, 0.985, _lab["b"], "Contact in traffic")
        _pb = _grouped_positions(_B_GROUPS, gap=0.4)
        for ds, cats, xs in _pb:
            for (cond, _), x in zip(cats, xs):
                _dots(ax, ds, cond, x)
        _group_axis(ax, _pb, names={"Virtual KITTI 2": "VKITTI 2", "Argoverse 2": "AV2 (real)"})
        ax.axhline(0, color=MONO_COLOR, lw=0.6, ls=":", zorder=1)
        ax.set_ylim(-0.35, 1.0); ax.set_ylabel("geometry share g", labelpad=1)
        _x_right = _xb + 0.17
    # (c) the depth overestimate implied by the E5 slopes at s = 0.85
    if _C_CATS:
        _xc = _x_right + 0.09
        ax = fig.add_axes([_xc, 0.2, 0.99 - _xc, 0.645])
        fig_panel_title(fig, _xc - 0.075, 0.985, _lab["c"], "What the slope costs")
        for k, (cond, lab) in enumerate(_C_CATS):
            if _support_of("E5 ablation", cond) == "grounded":
                ax.axvspan(k - 0.5, k + 0.5, color=SUPPORT_SHADE, lw=0, zorder=0)
            _dots(ax, "E5 ablation", cond, k, "implied_factor", "implied_lo", "implied_hi")
        _geo = 1.0 / (1.0 - PREREG["implied_factor_s"])
        ax.set_yscale("log"); ax.set_ylim(0.8, _geo * 1.35)
        ax.axhline(1.0, color=MONO_COLOR, lw=0.6, ls=":", zorder=1)
        ax.axhline(_geo, color=CONTROL_COLOR, lw=0.6, ls=(0, (4, 2)), zorder=1)
        ax.set_yticks([1, 2, 4, _geo]); ax.set_yticklabels(["1", "2", "4", f"{_geo:.1f}"]); ax.minorticks_off()
        ax.text(1.0, _geo, "geometry", transform=ax.get_yaxis_transform(), fontsize=4.8, ha="right", va="bottom", color="#555555")
        ax.set_xticks(range(len(_C_CATS))); ax.set_xticklabels([lab for _, lab in _C_CATS], fontsize=5.4)
        ax.set_xlim(-0.6, len(_C_CATS) - 0.4); ax.tick_params(axis="x", length=0)
        ax.set_ylabel("depth overestimate at s = 0.85", labelpad=1, fontsize=5.6)
        ax.text(0.5, 1.0, "E5", transform=ax.transAxes, fontsize=5.0, ha="center", va="bottom", color="#555555")
    fig.legend(handles=[Line2D([], [], ls="", marker=MODEL_MARKERS[m], color=mcol(m), ms=2.6, label=mlab(m)) for m in _M10]
                       + [mpatches.Patch(color=SUPPORT_SHADE, label="contact in view")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(_M10) + 1, fontsize=5.0, handletextpad=0.3,
               columnspacing=0.9, handlelength=1.2)
    _n_real = int(len(SUPPORT_LEVELS[SUPPORT_LEVELS["testbed"] == "CONVOY-REAL"]))
    _cap10 = cap(_lab["a"], "Support decides the slope", "Geometry share β_s of objects whose ground contact is in view (G, "
                 "shaded), that float (F) or whose contact is hidden by a wall (CH), in every testbed with designed motion: "
                 "CONVOY-3D cars and crates against floating crates"
                 + (", CONVOY-REAL (every inserted mover is a floating crate)" if _n_real else "")
                 + ", the same E5 movers grounded and behind the wall, and the same E5-REAL crates on and above a real support.")
    if _B_GROUPS:
        _cap10 += " " + cap(_lab["b"], "Contact in traffic (exploratory)", "Median per-object geometry share g of parallel "
                            "vehicles whose wheel contact is visible (G, shaded) or hidden (CH), in " + " and ".join(
                                {"Virtual KITTI 2": "Virtual KITTI 2", "Argoverse 2": "real Argoverse 2 images with LiDAR depth"}[ds]
                                for ds, _ in _B_GROUPS) + "; 0 is the model's own single-image depth, 1 the geometric answer.")
    if _C_CATS:
        _cap10 += " " + cap(_lab["c"], "What the slope costs", f"Depth overestimate exp(β_s x) that the E5 slopes (G, CH, F) imply for an "
                            f"object keeping pace with the camera (s = {PREREG['implied_factor_s']}, whose geometric answer lies "
                            f"{1 / (1 - PREREG['implied_factor_s']):.1f} times too far); computed from the slope, not measured separately.")
    _cap10 += (" Intervals are 95% cluster-bootstrap intervals (scenes, image pairs or vehicle tracks). Descriptive: the grouping "
               "by support was chosen after the v6.3 results were read."
               + gate_caption_note("MAIN", "REAL", "ABL", "RABL", *[{"Virtual KITTI 2": "VK", "Argoverse 2": "AV2"}[ds] for ds, _ in _B_GROUPS]))
    save_figure(fig, "fig10_support", "One cue across the testbeds: ground support",
                "Does the same support cue explain the geometry share in every testbed, and what does it cost?",
                "Section 6.9", "double column",
                dict(levels=SUPPORT_LEVELS.to_dict("records"), contrasts=SUPPORT_CONTRASTS.to_dict("records")), caption=_cap10)
else:
    print("Figure 10 is not drawn: no support levels could be estimated in this run.")

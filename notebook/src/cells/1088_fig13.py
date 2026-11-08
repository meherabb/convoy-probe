# PLACEHOLDER
# Figure 13 (double column, v6.5): the designed experiments. (a) One scene of each set: the second view at x0.5 and x3 the
# baseline, and one mover behind the 15 cm and the 70 cm wall. (b) H12: beta_s against the baseline factor. (c) The rise from
# x0.5 to x3 for grounded and floating movers. (d) H13: beta_s against the height of the wall.
_M13b = [m for m in MODELS_OK["BSW"] if len(BSW_BETA) and (BSW_BETA["model"] == m).any()]
_M13w = [m for m in MODELS_OK["WDR"] if len(WDR_BETA) and (WDR_BETA["model"] == m).any()]

def _crop_mover(meta, k, j, pad=0.6):
    fr = load_frame(meta["dir"], k)
    bb = mask_bbox(fr["mover_id"] == j)
    return crop_around(fr["rgb"], bb, (3, 4), pad=pad) if bb is not None else fr["rgb"]

def _bsw_example():
    w = BSW_WIDE.get(_M13b[0]) if _M13b else None
    if w is None or not len(w):
        return None
    q = w[w["h12_sample"]].sort_values(f"gt_n_eroded_{BSW_LEVELS[0]}", ascending=False)
    if not len(q):
        return None
    r = q.iloc[0]
    j = int(str(r["uid"]).rsplit("-", 1)[1])
    metas = {m["condition"]: m for m in META_BSW if m["index"] == int(r["scene"])}
    lo, hi = BSW_LEVELS[0], BSW_LEVELS[-1]
    fr_lo, fr_hi = load_frame(metas[lo]["dir"], 1), load_frame(metas[hi]["dir"], 1)
    union = (fr_lo["mover_id"] == j) | (fr_hi["mover_id"] == j)
    bb = mask_bbox(union)
    return [(crop_around(fr_lo["rgb"], bb, (3, 4), pad=0.5), f"view 2, {SWEEP_DESIGN['labels'][lo]}"),
            (crop_around(fr_hi["rgb"], bb, (3, 4), pad=0.5), f"view 2, {SWEEP_DESIGN['labels'][hi]}")], r

def _wdr_example():
    w = WDR_WIDE.get(_M13w[0]) if _M13w else None
    if w is None or not len(w):
        return None
    q = w[w["contact_G"] & w["hidden_W15"] & w["hidden_W70"] & w["ok_W70"]].sort_values("gt_n_eroded_G", ascending=False)
    if not len(q):
        return None
    r = q.iloc[0]
    j = int(str(r["uid"]).rsplit("-", 1)[1])
    metas = {m["condition"]: m for m in META_WDR if m["index"] == int(r["scene"])}
    frs = {lv: load_frame(metas[lv]["dir"], 0) for lv in ("G", "W15", "W70")}
    bb = mask_bbox(frs["G"]["mover_id"] == j)
    return [(crop_around(frs[lv]["rgb"], bb, (3, 4), pad=0.55), f"{WALL_DOSE_DESIGN['labels'][lv]} wall") for lv in ("W15", "W70")], r

_ex_b, _ex_w = _bsw_example(), _wdr_example()
if _M13b or _M13w:
    fig = plt.figure(figsize=FIG_DOUBLE)
    # (a) a 2 x 2 block of 4:3 tiles, centred on the plot panels and sharing their title line
    _th, _gap = 0.52, 0.03
    _tw = _th * 4 / 3
    _y0 = 0.25 * 1.8 + (0.62 * 1.8 - (2 * _th + _gap)) / 2
    title_axes(fig, [0.03 / 5.5, 0.25, (2 * _tw + _gap) / 5.5, 0.62], "a", "One factor at a time")
    for row, ex in enumerate((_ex_b, _ex_w)):
        if ex is None:
            continue
        for c_, (img, lab) in enumerate(ex[0]):
            a = fig.add_axes([(0.03 + c_ * (_tw + _gap)) / 5.5, (_y0 + (1 - row) * (_th + _gap)) / 1.8, _tw / 5.5, _th / 1.8])
            a.imshow(img, aspect="auto"); clean_image_axes(a); tag(a, lab, fs=4.6)
    fig.text((0.03 + _tw + _gap / 2) / 5.5, (_y0 - 0.04) / 1.8, "top: baseline sweep; bottom: walls", fontsize=5.0,
             ha="center", va="top")
    # (b) beta against the baseline factor
    ax = fig.add_axes([0.35, 0.25, 0.18, 0.62])
    panel_title(ax, "b", "Parallax (H12)", x=-0.25)
    _fx = [SWEEP_DESIGN["levels"][lv] for lv in BSW_LEVELS]
    for i, m in enumerate(_M13b):
        g = BSW_BETA[(BSW_BETA["model"] == m) & (BSW_BETA["support"] == "all")].set_index("level").reindex(BSW_LEVELS)
        xo = np.array(_fx) * (1 + (i - (len(_M13b) - 1) / 2) * 0.025)
        ax.errorbar(xo, g["beta"], yerr=[(g["beta"] - g["ci_lo"]).clip(lower=0), (g["ci_hi"] - g["beta"]).clip(lower=0)],
                    fmt=MODEL_MARKERS[m], color=mcol(m), ms=2.2, capsize=0.8, lw=0.6, ls="-")
    ax.set_xscale("log"); ax.set_xticks(_fx); ax.set_xticklabels([SWEEP_DESIGN["labels"][lv] for lv in BSW_LEVELS]); ax.minorticks_off()
    ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0)
    ax.set_xlabel("camera baseline"); ax.set_ylabel("β$_s$ (95% CI)", labelpad=1)
    # (c) rise by support
    ax = fig.add_axes([0.615, 0.25, 0.11, 0.62])
    panel_title(ax, "c", "By support", x=-0.55)
    _n13c = 0
    for i, m in enumerate(_M13b):
        for k, sup in enumerate(("grounded", "floating")):
            r = BSW_TESTS[(BSW_TESTS["model"] == m) & (BSW_TESTS["test"] == f"H12-{sup}")]
            if len(r) and np.isfinite(r["estimate"].iloc[0]):
                r = r.iloc[0]
                ax.errorbar(k + (i - (len(_M13b) - 1) / 2) * 0.1, r["estimate"], yerr=ci_err(r["estimate"], r["ci_lo"], r["ci_hi"]),
                            fmt=MODEL_MARKERS[m], color=mcol(m), ms=2.2, capsize=0.8, lw=0.6)
                _n13c += 1
    if _n13c:
        ax.axhline(0, color="#7F7F7F", lw=0.6, zorder=0)
        ax.margins(y=0.12)
    else:
        empty_panel(ax, "too few movers\nper group\nin this run")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["grounded", "floating"], fontsize=5.2); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("rise of β$_s$, x0.5 → x3", labelpad=1, fontsize=5.6)
    # (d) beta against the wall height
    ax = fig.add_axes([0.81, 0.25, 0.18, 0.62])
    panel_title(ax, "d", "Contact (H13)", x=-0.28)
    _cv = [100 * WALL_DOSE_DESIGN["levels"][lv] for lv in WDR_LEVELS]
    for i, m in enumerate(_M13w):
        g = WDR_BETA[WDR_BETA["model"] == m].set_index("level").reindex(WDR_LEVELS)
        xo = np.array(_cv) + (i - (len(_M13w) - 1) / 2) * 1.6
        ax.errorbar(xo, g["beta"], yerr=[(g["beta"] - g["ci_lo"]).clip(lower=0), (g["ci_hi"] - g["beta"]).clip(lower=0)],
                    fmt=MODEL_MARKERS[m], color=mcol(m), ms=2.2, capsize=0.8, lw=0.6, ls="-")
    ax.axvspan(-6, 6, color=SUPPORT_SHADE, lw=0, zorder=0)
    ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0)
    ax.set_xticks(_cv); ax.set_xticklabels(["none", "15", "35", "70"]); ax.set_xlim(-8, 78)
    ax.margins(y=0.1)
    ax.set_xlabel("hidden band (cm)"); ax.set_ylabel("β$_s$ (95% CI)", labelpad=1)
    _mm = [m for m in EVALUATED_MODELS if m in set(_M13b) | set(_M13w)]
    fig.legend(handles=[Line2D([], [], ls="-", lw=0.6, marker=MODEL_MARKERS[m], color=mcol(m), ms=2.4, label=mlab(m)) for m in _mm],
               loc="lower center", bbox_to_anchor=(0.66, 0.0), ncol=len(_mm), fontsize=5.0, handlelength=1.4, columnspacing=0.9)
    _h12 = BSW_TESTS[(BSW_TESTS["test"] == "H12") & BSW_TESTS["model"].isin(PREREG["h12_models"])] if len(BSW_TESTS) else BSW_TESTS
    _h13 = WDR_TESTS[WDR_TESTS["test"] == "H13"] if len(WDR_TESTS) else WDR_TESTS
    _h13i = _h13[~_h13["verdict"].str.startswith("NOT INTERPRETABLE")] if len(_h13) else _h13
    _hs13 = WDR_BETA.groupby("level")["median_hidden_share"].median() if len(WDR_BETA) else pd.Series(dtype=float)
    _cap13 = (cap("a", "One factor at a time", "Top: the second view of one baseline-sweep scene at 0.5 and 3 times the baseline "
                  "(first view, s, eps = 0 and the geometric answer are identical); bottom: one mover behind the 15 cm and the 70 cm wall.")
              + " " + cap("b", "Parallax (H12)", "beta_s of the same movers at each baseline (measurable at every baseline, identifiable "
                          "at x1); H12 predicts a rise for DUSt3R and MASt3R"
                          + (" (" + "; ".join(f"{mlab(r.model)}: {r.estimate:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}], {r.verdict}"
                                               for r in _h12.itertuples()) + ")" if len(_h12) else "") + ".")
              + " " + cap("c", "By support", "The rise from x0.5 to x3 for grounded movers (cars, crates) and floating crates, "
                          "exploratory.")
              + " " + cap("d", "Contact (H13)", "beta_s of the same grounded movers with no wall (shaded) and behind walls built "
                          "to hide at least their lowest 15, 35 and 70 cm in both views"
                          + (" (median share of the mover hidden: " + ", ".join(f"{_hs13.get(lv, np.nan):.2f}" for lv in WDR_LEVELS)
                             + ")" if len(_hs13) else "")
                          + "; H13 predicts a rise at 15 cm for every model"
                          + (f" (SUPPORTED for {int((_h13i['verdict'] == 'SUPPORTED').sum())} of the {len(_h13i)} models that pass the "
                             "static-scene gate)" if len(_h13i) else "") + ".")
              + " Intervals are 95% cluster-bootstrap intervals over base scenes." + gate_caption_note("BSW", "WDR"))
    save_figure(fig, "fig13_designed", "Designed experiments: parallax and contact, one at a time",
                "Does a longer baseline alone raise the geometry share, and does the lowest wall that hides the contact suffice?",
                "Section 6.10", "double column", dict(sweep=BSW_BETA.to_dict("records"), sweep_tests=BSW_TESTS.to_dict("records"),
                                                      dose=WDR_BETA.to_dict("records"), dose_tests=WDR_TESTS.to_dict("records")),
                caption=_cap13)
else:
    print("Figure 13 is not drawn: no interpretable model on the designed sets in this run.")

# PLACEHOLDER
# Figure 4 (double column): E5, what controls the geometry share. (a) One mover in four of the five
# conditions (same s, same camera, same scene). (b) beta_s per condition (common sample; 95% CI).
# (c) The registered contrasts H5-H7. (d) H8: the single-image depth of a lifted object.
_cond_lab = {"G": "grounded", "NS": "no shadow", "CH": "contact hidden", "F": "floating", "FH": "floating + wall"}
_M4 = MODELS_OK["ABL"]
_E5B = E5_BETA[E5_BETA["model"].isin(_M4)] if len(E5_BETA) else E5_BETA
_has_e5 = len(_E5B) > 0 and np.isfinite(_E5B["beta"]).any()
if _has_e5:
    _w0 = E5_WIDE[_M4[0]]
    _cands = _sample(_w0, E5_CONDS, walled=True)
    _cands = _cands[_cands["scene"].isin(sorted({m["index"] for m in META_ABL}))]
    _ex = _cands.sort_values("gt_n_eroded_G", ascending=False).iloc[0] if len(_cands) else _w0.iloc[0]
    _ex_scene, _ex_mover = int(_ex["scene"]), int(str(_ex["uid"]).rsplit("-", 1)[1])
    _ex_meta = {m["condition"]: m for m in META_ABL if m["index"] == _ex_scene}
    _frames = {c: load_frame(_ex_meta[c]["dir"], 0) for c in ["G", "CH", "F", "FH"]}
    _union = np.zeros_like(_frames["G"]["mover_id"], bool)
    for c, fr in _frames.items():
        _union |= fr["mover_id"] == _ex_mover
    _bbx = mask_bbox(_union)

    fig = plt.figure(figsize=FIG_DOUBLE)
    _tw, _gap = 0.615, 0.03                                   # tile width and gap (inches); tiles are 4:3
    _th = 0.75 * _tw
    _y0 = (0.26 * 1.8)
    for k, c in enumerate(["G", "CH", "F", "FH"]):
        r_, c_ = k // 2, k % 2
        a = fig.add_axes([(0.03 + c_ * (_tw + _gap)) / 5.5, (_y0 + (1 - r_) * (_th + _gap)) / 1.8, _tw / 5.5, _th / 1.8])
        a.imshow(crop_around(_frames[c]["rgb"], _bbx, (3, 4), pad=0.45), aspect="auto"); clean_image_axes(a)
        tag(a, _cond_lab[c], fs=4.8)
    fig_panel_title(fig, 0.004, 0.985, "a", "One mover")
    fig.text((0.03 + _tw + _gap / 2) / 5.5, 0.2, f"one mover, s = {_ex['s']:.2f}, ε = 0", fontsize=5.0, ha="center", va="top")
    # (b) beta by condition
    ax = fig.add_axes([0.305, 0.25, 0.29, 0.635])
    fig_panel_title(fig, 0.258, 0.985, "b", "Geometry share per condition")
    _nm = len(_M4)
    for i, model in enumerate(_M4):
        g = _E5B[_E5B["model"] == model].set_index("condition").reindex(E5_CONDS)
        xo = np.arange(len(E5_CONDS)) + (i - (_nm - 1) / 2) * 0.1
        ax.plot(xo, g["beta"], color=mcol(model), lw=0.6, alpha=0.55)
        ax.errorbar(xo, g["beta"], yerr=[(g["beta"] - g["ci_lo"]).clip(lower=0), (g["ci_hi"] - g["beta"]).clip(lower=0)],
                    fmt=MODEL_MARKERS[model], color=mcol(model), ms=2.4, capsize=0.8, lw=0.6, label=mlab(model))
    ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0)
    ax.axvspan(2.5, 4.5, color="#F2F2F2", zorder=0, lw=0)
    ax.set_xticks(range(len(E5_CONDS))); ax.set_xticklabels(["grounded", "no\nshadow", "contact\nhidden", "floating", "floating\n+ wall"],
                                                            fontsize=5.2)
    ax.set_xlim(-0.5, len(E5_CONDS) - 0.5); ax.set_ylabel("β$_s$ (95% CI)")
    _ylo = np.nanmin(_E5B["ci_lo"]) if _E5B["ci_lo"].notna().any() else 0.0
    _yhi = np.nanmax(_E5B["ci_hi"]) if _E5B["ci_hi"].notna().any() else 1.0
    ax.set_ylim(min(-0.1, _ylo - 0.05), max(1.0, _yhi + 0.05))
    _n5 = int(_E5B["n"].max()) if len(_E5B) else 0
    ax.text(0.02, 0.97, f"same {_n5} movers in every condition", transform=ax.transAxes, fontsize=5.0, va="top", color="#555555")
    # (c) contrasts H5-H7
    ax = fig.add_axes([0.665, 0.25, 0.2, 0.635])
    fig_panel_title(fig, 0.618, 0.985, "c", "Registered contrasts")
    _tests = ["H5", "H6", "H7"]
    for i, model in enumerate(_M4):
        g = E5_TESTS[(E5_TESTS["model"] == model)].set_index("test").reindex(_tests)
        xo = np.arange(len(_tests)) + (i - (_nm - 1) / 2) * 0.11
        ax.errorbar(xo, g["estimate"], yerr=[(g["estimate"] - g["ci_lo"]).clip(lower=0), (g["ci_hi"] - g["estimate"]).clip(lower=0)],
                    fmt=MODEL_MARKERS[model], color=mcol(model), ms=2.4, capsize=0.8, lw=0.6)
    ax.axhline(0, color="#7F7F7F", lw=0.6, zorder=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(["F − NS", "CH − G", "interaction"], fontsize=5.2)
    ax.set_xlim(-0.5, 2.5); ax.set_ylabel("difference in β$_s$")
    for k, t in enumerate(_tests):
        ax.text(k, 1.0, t, transform=ax.get_xaxis_transform(), fontsize=5.0, ha="center", va="bottom", color="#555555")
    # (d) H8
    ax = fig.add_axes([0.925, 0.25, 0.07, 0.635])
    fig_panel_title(fig, 0.868, 0.985, "d", "One view")
    for i, model in enumerate(_M4):
        g = E5_TESTS[(E5_TESTS["model"] == model) & (E5_TESTS["test"] == "H8")]
        if len(g):
            r = g.iloc[0]
            ax.errorbar((i - (_nm - 1) / 2) * 0.12, r["estimate"], yerr=ci_err(r["estimate"], r["ci_lo"], r["ci_hi"]),
                        fmt=MODEL_MARKERS[model], color=mcol(model), ms=2.4, capsize=0.8, lw=0.6)
    ax.axhline(0, color="#7F7F7F", lw=0.6, zorder=0)
    ax.set_xlim(-0.5, 0.5); ax.set_xticks([0]); ax.set_xticklabels(["H8"], fontsize=5.2)
    ax.set_ylabel("Δ single-image ρ", labelpad=1)
    fig.legend(handles=[Line2D([], [], ls="-", lw=0.6, marker=MODEL_MARKERS[m], color=mcol(m), ms=2.4, label=mlab(m))
                        for m in _M4], loc="lower center", bbox_to_anchor=(0.64, 0.0), ncol=len(_M4),
               fontsize=5.0, handlelength=1.4, columnspacing=0.9)
    save_figure(fig, "fig4_ablation", "What controls the geometry share: ground contact",
                "Does removing or hiding an object's ground contact make the models use more geometry?", "Section 6.7",
                "double column", dict(beta=_E5B.to_dict("records"), tests=E5_TESTS[E5_TESTS["model"].isin(_M4)].to_dict("records"),
                                      exemplar=dict(scene=_ex_scene, mover=_ex_mover)),
                caption=cap("a", "One mover", "The same parallel mover rendered grounded, behind a low wall "
                            "that hides its ground contact in both views, lifted off the ground, and lifted behind the wall; "
                            "everything else is identical.")
                        + " " + cap("b", "Geometry share per condition", "beta_s on the same movers; shaded: the two lifted conditions.")
                        + " " + cap("c", "Registered contrasts", "Lifting the object (H5), hiding its contact (H6) and their "
                                    "interaction (H7).")
                        + " " + cap("d", "One view", "Shift of the single-image depth of the same object when it is "
                                    "lifted (H8).")
                        + " Intervals are 95% cluster-bootstrap intervals over base scenes." + gate_caption_note("ABL"))

else:
    print("Figure 4 is not drawn: the ablation produced no estimable geometry share in this run.")

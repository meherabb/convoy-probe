# PLACEHOLDER
# Figure 5 (double column): E5-REAL, the support cue on real images. (a) Two crates, each inserted
# into a real TUM frame grounded with a contact shadow and lifted without one (same pair, crate, s
# and texture). (b) beta_s of the same crates in both conditions, per model, with classical
# triangulation on the same composites. (c) H11a, beta(lifted) - beta(grounded). (d) H11b, the shift
# of the single-image depth when the crate is lifted.
_MR = [m for m in MODELS_OK["RABL"] if m in RABL_WIDE]
_RB = RABL_BETA[RABL_BETA["model"].isin(_MR)] if len(RABL_BETA) else RABL_BETA
_has_rabl = len(_RB) > 0 and bool(np.isfinite(_RB["beta"]).any())
if _has_rabl:
    _wr = RABL_WIDE[_MR[0]]
    _cand = _wr[_wr["sample"]].sort_values("gt_n_eroded_G", ascending=False)
    _picks = []
    for _seq in _cand["seq"].dropna().unique():                 # one crate per sequence first, then the largest others
        _picks.append(_cand[_cand["seq"] == _seq].iloc[0])
    for _, r in _cand.iterrows():
        if len(_picks) >= 2:
            break
        if all(r["uid"] != p["uid"] for p in _picks):
            _picks.append(r)
    _picks = _picks[:2]
    _rmeta = {(m["index"], m["condition"]): m for m in META_RABL}
    _tiles = []
    for r in _picks:
        _sc, _mv = int(r["scene"]), int(str(r["uid"]).rsplit("-", 1)[1])
        fr = {c: load_frame(_rmeta[(_sc, c)]["dir"], 0) for c in RABL_CONDS}
        _u = np.zeros_like(fr["G"]["mover_id"], bool)
        for c in RABL_CONDS:
            _u |= fr[c]["mover_id"] == _mv
        _bb = mask_bbox(_u)
        _tiles.append({c: crop_around(fr[c]["rgb"], _bb, (3, 4), pad=0.65) for c in RABL_CONDS} | dict(s=float(r["s"]), seq=r["seq"]))
        del fr
    _cl = {"G": "grounded", "F": "lifted"}

    fig = plt.figure(figsize=FIG_DOUBLE)
    _tw, _gap = 0.615, 0.03
    _th = 0.75 * _tw
    _y0 = 0.26 * 1.8
    for r_, t in enumerate(_tiles):
        for c_, c in enumerate(RABL_CONDS):
            a = fig.add_axes([(0.03 + c_ * (_tw + _gap)) / 5.5, (_y0 + (1 - r_) * (_th + _gap)) / 1.8, _tw / 5.5, _th / 1.8])
            a.imshow(t[c], aspect="auto"); clean_image_axes(a)
            tag(a, _cl[c], fs=4.8)
    fig_panel_title(fig, 0.004, 0.985, "a", "Real support")
    fig.text((0.03 + _tw + _gap / 2) / 5.5, 0.2, "real TUM frames, same crate, ε = 0", fontsize=5.0, ha="center", va="top")
    # (b) beta_s grounded vs lifted, paired by model, with the classical control
    ax = fig.add_axes([0.335, 0.25, 0.22, 0.635])
    fig_panel_title(fig, 0.272, 0.985, "b", "Geometry share")
    _nm = len(_MR)
    for i, model in enumerate(_MR):
        g = _RB[_RB["model"] == model].set_index("condition").reindex(RABL_CONDS)
        xo = np.arange(len(RABL_CONDS)) + (i - (_nm - 1) / 2) * 0.09
        ax.plot(xo, g["beta"], color=mcol(model), lw=0.6, alpha=0.55)
        ax.errorbar(xo, g["beta"], yerr=[(g["beta"] - g["ci_lo"]).clip(lower=0), (g["ci_hi"] - g["beta"]).clip(lower=0)],
                    fmt=MODEL_MARKERS[model], color=mcol(model), ms=2.4, capsize=0.8, lw=0.6)
    _ctl = [RABL_CONTROL.get(c, {}).get("beta", np.nan) for c in RABL_CONDS]
    ax.plot(np.arange(len(RABL_CONDS)) + 0.5 * 0.09 * (_nm + 1), _ctl, ls="none", marker="D", ms=2.2, color=CONTROL_COLOR)
    ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0)
    ax.set_xticks(range(len(RABL_CONDS))); ax.set_xticklabels(["grounded,\ncontact shadow", "lifted,\nno shadow"], fontsize=5.2)
    ax.set_xlim(-0.55, len(RABL_CONDS) - 0.45); ax.set_ylabel("β$_s$ (95% CI)")
    _lo = np.nanmin(_RB["ci_lo"]) if _RB["ci_lo"].notna().any() else 0.0
    _hi = np.nanmax(_RB["ci_hi"]) if _RB["ci_hi"].notna().any() else 1.0
    ax.set_ylim(min(-0.1, _lo - 0.05), max(1.32, _hi + 0.05))        # room above the triangulation control (v6.5)
    _nr = int(_RB["n"].max())
    ax.text(0.03, 0.97, f"same {_nr} crates, {int(_RB['n_pairs'].max())} pairs", transform=ax.transAxes, fontsize=5.0, va="top",
            color="#555555")
    # (c) H11a and (d) H11b
    for k_, (t_, lab_, x0_, lx_, tt_) in enumerate([("H11a", "difference in β$_s$", 0.645, 0.6, "Lifting (H11a)"),
                                                    ("H11b", "Δ single-image ρ", 0.865, 0.82, "One view (H11b)")]):
        ax = fig.add_axes([x0_, 0.25, 0.1, 0.635])
        fig_panel_title(fig, lx_, 0.985, "cd"[k_], tt_)
        for i, model in enumerate(_MR):
            g = RABL_TESTS[(RABL_TESTS["model"] == model) & (RABL_TESTS["test"] == t_)]
            if len(g) and np.isfinite(g.iloc[0]["estimate"]):
                r = g.iloc[0]
                ax.errorbar((i - (_nm - 1) / 2) * 0.12, r["estimate"], yerr=ci_err(r["estimate"], r["ci_lo"], r["ci_hi"]),
                            fmt=MODEL_MARKERS[model], color=mcol(model), ms=2.4, capsize=0.8, lw=0.6)
        ax.axhline(0, color="#7F7F7F", lw=0.6, zorder=0)
        ax.set_xlim(-0.5, 0.5); ax.set_xticks([0]); ax.set_xticklabels([t_ + "\nlifted − grounded"], fontsize=5.2)
        ax.set_ylabel(lab_, labelpad=1)
    _h = [Line2D([], [], ls="-", lw=0.6, marker=MODEL_MARKERS[m], color=mcol(m), ms=2.4, label=mlab(m)) for m in _MR]
    _h += [Line2D([], [], ls="none", marker="D", ms=2.2, color=CONTROL_COLOR, label="triangulation, true pose")]
    fig.legend(handles=_h, loc="lower center", bbox_to_anchor=(0.665, 0.0), ncol=len(_h), fontsize=5.0, handlelength=1.4,
               columnspacing=0.8)
    save_figure(fig, "fig5_real_contact", "The support cue on real images",
                "Does lifting an object off its real support, with everything else fixed, make the models use more geometry?",
                "Section 6.7b", "double column",
                dict(beta=_RB.to_dict("records"), tests=RABL_TESTS[RABL_TESTS["model"].isin(_MR)].to_dict("records"),
                     control={c: {k: v for k, v in RABL_CONTROL[c].items() if np.isscalar(v)} for c in RABL_CONTROL},
                     exemplars=[dict(uid=str(p["uid"]), seq=str(p["seq"])) for p in _picks]),
                caption=cap("a", "Real support", "The same textured crate inserted into real TUM RGB-D frames, "
                            "resting on the real support plane with a soft contact shadow (grounded) and lifted 0.25-0.50 m "
                            "without a shadow (lifted); the frame pair, the crate, its texture and its parallel motion (ε = 0) "
                            "are identical.")
                        + " " + cap("b", "Geometry share", "beta_s of the same crates in both conditions; diamonds: classical "
                                    "triangulation with the true pose on the same composites.")
                        + " " + cap("c", "Lifting (H11a)", "The change of beta_s when the crate is lifted.")
                        + " " + cap("d", "One view (H11b)", "The shift of the single-image depth of the same crate "
                                    "when it is lifted.")
                        + " Intervals are 95% cluster-bootstrap intervals over image pairs." + gate_caption_note("RABL"))
    del _tiles
    release_memory()
else:
    print("Figure 5 is not drawn: the real-image contact ablation produced no estimable geometry share in this run.")

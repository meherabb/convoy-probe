# PLACEHOLDER
# Figure 14 (double column, v6.5, exploratory): real objects that float (Bonn RGB-D Dynamic). (a) Four pairs, two with a
# balloon and two with a box, the moving object outlined (people in white). (b) Motion regimes of real objects by kind.
# (c) Median per-object geometry share g of parallel objects by kind and by floor contact.
_OBJ_COL, _PERSON_COL = "#F0E442", "#FFFFFF"
_bg = BONN_G[BONN_G["model"].isin(MODELS_OK.get("BONN", []))] if len(BONN_G) else BONN_G


def _bonn_examples(kind, k):
    # the k pairs with the largest object of this kind, from different sequences where possible
    cands = sorted(((max(o["n_px"] for o in m_["objects"] if o["kind"] == kind), m_["index"], m_) for m_ in BONN_PAIR_META
                    if any(o["kind"] == kind for o in m_["objects"])), key=lambda t: (-t[0], t[1]))
    out, seqs = [], set()
    for _, _, m_ in cands:
        if m_["seq"] not in seqs and len(out) < k:
            out.append(m_); seqs.add(m_["seq"])
    for _, _, m_ in cands:
        if len(out) < k and all(m_["index"] != o_["index"] for o_ in out):
            out.append(m_)
    return out


if len(BONN_PREV) and BONN_PAIR_META and not BONN_STATUS.get("suspect"):
    _ex14 = _bonn_examples("balloon", 2) + _bonn_examples("box", 2)
    fig = plt.figure(figsize=FIG_DOUBLE)
    # (a) a 2 x 2 block of 4:3 tiles, centred on the plot panels and sharing their title line
    _th, _gap = 0.52, 0.03
    _tw = _th * 4 / 3
    _y0 = 0.25 * 1.8 + (0.62 * 1.8 - (2 * _th + _gap)) / 2
    title_axes(fig, [0.03 / 5.5, 0.25, (2 * _tw + _gap) / 5.5, 0.62], "a", "Real objects")
    for k, m_ in enumerate(_ex14[:4]):
        img = np.asarray(Image.open(os.path.join(m_["dir"], "rgb0.png")).convert("RGB"))
        a = fig.add_axes([(0.03 + (k % 2) * (_tw + _gap)) / 5.5, (_y0 + (1 - k // 2) * (_th + _gap)) / 1.8, _tw / 5.5, _th / 1.8])
        a.imshow(img, aspect="auto"); clean_image_axes(a)
        for o in m_["objects"]:
            mk = np.zeros(img.shape[:2], bool)
            px = np.asarray(o["pixels"], int).reshape(-1, 2)
            mk[px[:, 1], px[:, 0]] = True
            a.contour(mk.astype(float), levels=[0.5], colors=[_PERSON_COL if o["kind"] == "person" else _OBJ_COL], linewidths=0.7)
        tag(a, m_["seq"].replace("moving_nonobstructing_", "").replace("_", " "), fs=4.6)
    # (b) regimes by kind
    ax = fig.add_axes([0.43, 0.25, 0.15, 0.62])
    panel_title(ax, "b", "Regimes", x=-0.6)
    _bp = BONN_PREV[BONN_PREV["kind"] != "all objects"].reset_index(drop=True)
    left = np.zeros(len(_bp))
    for col, reg in (("frac_parallel", "parallel"), ("frac_intermediate", "intermediate"), ("frac_perpendicular", "perpendicular")):
        ax.barh(np.arange(len(_bp))[::-1], _bp[col], left=left, color=REGIME_COLORS[reg], height=0.6, lw=0)
        left = left + _bp[col].to_numpy()
    ax.set_yticks(np.arange(len(_bp))[::-1]); ax.set_yticklabels([f"{k} ({n})" for k, n in zip(_bp["kind"], _bp["n_objects"])],
                                                                    fontsize=5.2)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("share of objects", labelpad=1)
    # (c) geometry share by kind and floor contact
    ax = fig.add_axes([0.685, 0.25, 0.305, 0.62])
    panel_title(ax, "c", "Geometry share", x=-0.17)
    _groups = [("kind", k) for k in ("balloon", "box", "person")] + [("floor contact", f_) for f_ in ("off the floor", "on the floor")]
    _groups = [g for g in _groups if len(_bg) and ((_bg["grouping"] == g[0]) & (_bg["group"] == g[1])).any()]
    _m14 = [m for m in EVALUATED_MODELS if len(_bg) and (_bg["model"] == m).any()]
    for k, (gt_, gn) in enumerate(_groups):
        for i, m in enumerate(_m14):
            r = _bg[(_bg["grouping"] == gt_) & (_bg["group"] == gn) & (_bg["model"] == m)]
            if len(r) and np.isfinite(r["median_g"].iloc[0]):
                r = r.iloc[0]
                _small = r["n"] < 8                              # no interval below 8 objects: open marker
                ax.errorbar(k + (i - (len(_m14) - 1) / 2) * 0.1, r["median_g"], yerr=ci_err(r["median_g"], r["ci_lo"], r["ci_hi"]),
                            fmt=MODEL_MARKERS[m], color=mcol(m), ms=2.2, capsize=0.8, lw=0.6, mfc="white" if _small else mcol(m),
                            mew=0.6)
    if _groups:
        ax.axvspan(len([g for g in _groups if g[0] == "kind"]) - 0.5, len(_groups) - 0.5, color="#F7F7F7", lw=0, zorder=0)
        ax.axhline(0, color=MONO_COLOR, lw=0.6, ls=":"); ax.axhline(1, color=CONTROL_COLOR, lw=0.6, ls=(0, (4, 2)))
        ax.set_xticks(range(len(_groups))); ax.set_xticklabels([gn.replace(" the floor", "\nthe floor") for _, gn in _groups], fontsize=5.0)
        ax.set_xlim(-0.5, len(_groups) - 0.5); ax.set_ylim(-0.4, 1.15)
    else:
        ax.set_xticks([]); empty_panel(ax, "no model has enough\nparallel objects\nin this run")
    ax.set_ylabel("geometry share g", labelpad=1)
    _h14 = [mpatches.Patch(color=REGIME_COLORS[r_], lw=0, label=r_) for r_ in ("parallel", "intermediate", "perpendicular")]
    _h14 += [Line2D([], [], ls="", marker=MODEL_MARKERS[m], color=mcol(m), ms=2.4, label=mlab(m)) for m in _m14]
    fig.legend(handles=_h14, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(_h14), fontsize=4.8, handletextpad=0.3,
               columnspacing=0.8, handlelength=1.0)
    _cap14 = (cap("a", "Real objects", "Bonn RGB-D Dynamic frames with a balloon (top) and a box (bottom); the moving object is "
                  "outlined in yellow and people in white. People are found by an instance segmenter; the rest of a moving region "
                  "is the object.")
              + " " + cap("b", "Regimes", f"Share of measurable moving objects in each motion regime, by kind (number of objects); "
                          f"parallel: median |eps| below the flow-noise floor ({BONN_TAU_PAR:.2f} px).")
              + " " + cap("c", "Geometry share", "Median per-object geometry share g (0, dotted: the model's own single-image depth; "
                          "1, dashed: the geometric answer) of parallel, identifiable objects with |x - rho_mono| >= 0.3, by kind and by "
                          "measured floor contact (shaded); 95% intervals over image pairs; open markers: fewer than 8 objects, "
                          "no interval." if _groups else
                          "No model has enough parallel objects in this run.")
              + " Exploratory: x_geo comes from optical flow and sensor depth, so no slope is fitted (D24)." + gate_caption_note("BONN"))
    save_figure(fig, "fig14_bonn", "Real objects that float", "How often is real object motion invisible, and how much geometry do "
                "the models use for real floating objects?", "Sections 3.12 and 6.12", "double column",
                dict(prevalence=BONN_PREV.to_dict("records"), geometry_share=BONN_G.to_dict("records"),
                     examples=[m_["dir"].rsplit(os.sep, 1)[-1] for m_ in _ex14]), caption=_cap14)
else:
    print("Figure 14 is not drawn: " + ("the Bonn objects fail the quality gate of Section 3.12" if BONN_STATUS.get("suspect") else
                                        "Bonn RGB-D Dynamic is not part of this run (" + (BONN_STATUS["note"] or "no data") + ")") + ".")

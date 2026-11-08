# PLACEHOLDER
# Figure 3 (double column): how much geometry the models use. (a) beta_s with 95% CI per model,
# the single-image control and classical triangulation; top axis: the equivalent prior weight w
# (slope-weight identity). (b) A1 knockout: beta_s with the mover deleted from view 2 (rows aligned
# with (a)). (c) beta_s by translational parallax: geometry can only be used where it determines depth.
# A panel whose estimates are not available in this run is left out.
_SW_W = SLOPE_WEIGHT_TABLE["w"].to_numpy(float); _SW_B = SLOPE_WEIGHT_TABLE["beta_ols_over_design"].to_numpy(float)

def _interp_lin(x, xp, fp):
    # np.interp with linear extrapolation beyond both ends (keeps the axis mapping strictly monotone)
    x = np.asarray(x, float); y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    y = np.where(lo, fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
    return np.where(hi, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)

_w_of_beta = lambda b: _interp_lin(b, _SW_B[::-1], _SW_W[::-1])
_beta_of_w = lambda w: _interp_lin(w, _SW_W, _SW_B)
_A1_OK = A1_TABLE[A1_TABLE["model"].isin(MODELS_OK["MAIN"])] if len(A1_TABLE) else A1_TABLE
_PAR_OK = E1_PARALLAX[E1_PARALLAX["model"].isin(MODELS_OK["MAIN"])] if len(E1_PARALLAX) else E1_PARALLAX
_has_a1 = len(_A1_OK) > 0 and np.isfinite(pd.to_numeric(_A1_OK["beta_knockout"], errors="coerce")).any()
_has_par = len(_PAR_OK) > 0 and bool(((_PAR_OK["n"] >= 8) & np.isfinite(_PAR_OK["beta"])).any())
_rows3 = [(r.model, mlab(r.model), r.beta_s, r.ci_lo, r.ci_hi, r.beta_mono, mcol(r.model)) for r in E1_TABLE.itertuples()
          if r.model in MODELS_OK["MAIN"]]
_rows3 += [(None, "triang., exact" if "exact" in k else "triang., flow", v["beta"], v["ci_lo"], v["ci_hi"], np.nan,
            CONTROL_COLOR) for k, v in CONTROL_E1.items()]
_ypos3 = {r[1]: len(_rows3) - 1 - i for i, r in enumerate(_rows3)}

def _forest_frame(a):
    a.axvspan(*PREREG["beta_support_range"], color=SUPPORT_SHADE, zorder=0, lw=0)
    a.axvline(0, color=GRID_COLOR, lw=0.5, zorder=0); a.axvline(1, color=GRID_COLOR, lw=0.5, zorder=0)
    a.set_xlim(-0.22, 1.3); a.set_ylim(-0.7, len(_rows3) - 0.3)

def _fig3_forest(ax):
    _forest_frame(ax)
    for m, lab, b, lo, hi, bm, c in _rows3:
        y = _ypos3[lab]
        if np.isfinite(b):
            ax.errorbar(b, y, xerr=ci_err(b, lo, hi), fmt="o", color=c, ms=3, capsize=1.2, lw=0.8)
        if np.isfinite(bm):
            ax.plot(bm, y, "o", mfc="white", mec=MONO_COLOR, ms=3, mew=0.6)
    ax.set_yticks([_ypos3[r[1]] for r in _rows3]); ax.set_yticklabels([r[1] for r in _rows3])
    ax.set_xlabel("β$_s$ (95% CI)")
    top = ax.secondary_xaxis("top", functions=(_w_of_beta, _beta_of_w))
    top.set_xticks([1.0, 0.8, 0.6, 0.4, 0.0]); top.tick_params(labelsize=5.4, pad=1)
    top.set_xlabel("equivalent prior weight w", fontsize=5.8, labelpad=2)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color="#444444", ms=3, label="two views"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=MONO_COLOR, ms=3, label="one view")],
              loc="lower left", fontsize=5.0)

def _fig3_knockout(ax):
    _forest_frame(ax)
    for r in _A1_OK.itertuples():
        y = _ypos3[mlab(r.model)]; c = mcol(r.model)
        if np.isfinite(r.beta_full):
            ax.plot(r.beta_full, y, "o", color=c, ms=2.2, alpha=0.5)
        if np.isfinite(r.beta_knockout):
            ax.errorbar(r.beta_knockout, y, xerr=ci_err(r.beta_knockout, r.ko_lo, r.ko_hi), fmt="s", mfc="white", mec=c, color=c,
                        ms=3, capsize=1.2, lw=0.8, mew=0.7)
    plt.setp(ax.get_yticklabels(), visible=False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("β$_s$, mover deleted from view 2")
    ax.legend(handles=[Line2D([], [], ls="", marker="s", mfc="white", mec="#444444", ms=3, mew=0.7, label="mover deleted"),
                       Line2D([], [], ls="", marker="o", color="#444444", alpha=0.5, ms=2.2, label="full")],
              loc="lower left", fontsize=5.0)

def _fig3_parallax(ax):
    bins = _PAR_OK[["par_lo", "par_hi"]].drop_duplicates().sort_values("par_lo").values.tolist()
    labs = [(f"<{int(b)}" if a == 0 else f"{int(a)}–{int(b)}") if b < 1e8 else f">{int(a)}" for a, b in bins]
    ax.axhspan(*PREREG["beta_support_range"], color=SUPPORT_SHADE, zorder=0, lw=0)
    n_low = sum(1 for _, b in bins if b <= PREREG["parallax_min_px"])
    if n_low:
        ax.axvspan(-0.5, n_low - 0.5, color="#F2F2F2", zorder=0, lw=0)
    ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0)
    for model in MODELS_OK["MAIN"]:
        g = _PAR_OK[_PAR_OK["model"] == model].sort_values("par_lo")
        ok = (g["n"] >= 8).values & np.isfinite(g["beta"].values)
        if ok.any():
            ax.plot(np.arange(len(g))[ok], g["beta"].values[ok], marker=MODEL_MARKERS[model], color=mcol(model), ms=2.6, lw=0.8,
                    label=mlab(model))
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=5.4)
    ax.set_xlabel("translational parallax (px)"); ax.set_ylabel("β$_s$", labelpad=1.0)
    ax.set_ylim(-0.25, 1.3); ax.set_xlim(-0.5, len(labs) - 0.5)
    if n_low:
        ax.text((n_low - 1) / 2, 1.22, "not identifiable", fontsize=5.0, color="#666666", ha="center", va="top")
    ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
    ax.spines["right"].set_visible(True); ax.spines["left"].set_visible(False)
    # no legend: the curves carry the model colours of (a) (a six-entry legend did not fit beside six curves)

_wr = [1.3] + ([0.78] if _has_a1 else []) + ([1.15] if _has_par else [])
fig = plt.figure(figsize=FIG_DOUBLE)
gs = fig.add_gridspec(1, len(_wr), width_ratios=_wr, left=0.112, right=0.93 if _has_par else 0.97, bottom=0.215, top=0.8,
                      wspace=0.12)
_ax3 = [fig.add_subplot(gs[0])]
_fig3_forest(_ax3[0])
if _has_a1:
    _ax3.append(fig.add_subplot(gs[1], sharey=_ax3[0]))
    _fig3_knockout(_ax3[-1])
if _has_par:
    _ax3.append(fig.add_subplot(gs[len(_ax3)]))
    _fig3_parallax(_ax3[-1])
_t3 = ["Geometry share"] + (["Knockout (A1)"] if _has_a1 else []) + (["Parallax"] if _has_par else [])
for i, a in enumerate(_ax3):
    fig_panel_title(fig, 0.004 if i == 0 else a.get_position().x0 - 0.005, 0.985, "abc"[i], _t3[i])
_cap3 = cap("a", "Geometry share", "beta_s of every model for purely parallel movers in CONVOY-3D (95% cluster-bootstrap intervals), "
            "the same model's single-image prediction (open circles) and classical triangulation with the true pose from exact "
            f"correspondences and from {FLOW_NAME} optical flow (triang., flow); the top axis "
            "gives the equivalent prior weight, the shaded band is the range the locked Prediction 1 expected.")
_k3 = iter("bc")
if _has_a1:
    _cap3 += " " + cap(next(_k3), "Knockout (A1)", "With the mover deleted from the second view, the remaining slope is "
                       "the part not carried by the mover's own correspondence.")
if _has_par:
    _cap3 += " " + cap(next(_k3), "Parallax", f"beta_s by translational parallax (model colours as in (a)); movers below "
                       f"{PREREG['parallax_min_px']:g} px (shaded) are not identifiable from two views and are excluded from (a).")
_cap3 += gate_caption_note("MAIN")
save_figure(fig, "fig3_geometry_share", "How much geometry the models use",
            "What is beta_s, does it come from the correspondence, and does it need parallax?", "Sections 6.1 and 6.5",
            "double column", dict(e1=E1_TABLE[E1_TABLE["model"].isin(MODELS_OK["MAIN"])].to_dict("records"),
                                  parallax=_PAR_OK.to_dict("records"), a1=_A1_OK.to_dict("records"), controls=CONTROL_E1),
            caption=_cap3)

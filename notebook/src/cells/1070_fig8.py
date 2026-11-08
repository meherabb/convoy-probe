# PLACEHOLDER
# Figure 8 (double column): the prior weight and the number of views. (a) Prior weight w against the
# exact epipolar offset (median per bin). (b) w at |eps| < 1 px by mover class. (c) E4: beta_s against
# the number of frames given jointly to the multi-view models, straight camera path (filled, solid)
# and its curved twin (open, dashed).
_M7 = MODELS_OK["MAIN"]
_E4T = E4_TABLE[E4_TABLE["model"].isin(_M7)] if len(E4_TABLE) else E4_TABLE
_has_e4 = len(_E4T) > 0 and _E4T["beta_s"].notna().any()
fig = plt.figure(figsize=FIG_DOUBLE)
_ax_specs = ([[0.065, 0.29, 0.25, 0.6], [0.385, 0.29, 0.19, 0.6], [0.655, 0.29, 0.165, 0.6]] if _has_e4 else
             [[0.07, 0.29, 0.38, 0.6], [0.55, 0.29, 0.28, 0.6]])
_axs = [fig.add_axes(sp) for sp in _ax_specs]
for i, (a, t) in enumerate(zip(_axs, ["Prior weight by offset", "Prior weight by class", "More views"])):
    panel_title(a, "abc"[i], t, x=-0.14 if i == 0 else -0.25)
ax = _axs[0]
ax.axvspan(-0.5, 1.5, color=PAR_SHADE, zorder=0, lw=0); ax.axvspan(3.5, 6.5, color=PERP_SHADE, zorder=0, lw=0)
for model in _M7:
    g = E3_BY_EPS[(E3_BY_EPS["model"] == model) & (E3_BY_EPS["n"] >= 3)].sort_values("eps_bin")
    ax.plot(g["eps_bin"], g["w_median"], marker=MODEL_MARKERS[model], color=mcol(model), ms=2.4, lw=0.8)
ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0); ax.axhline(1, color=GRID_COLOR, lw=0.5, zorder=0)
ax.set_xticks(range(7)); ax.set_xticklabels(["<0.25", "0.25–1", "1–2", "2–3", "3–6", "6–10", ">10"], rotation=40, ha="right",
                                            rotation_mode="anchor", fontsize=5.2)
ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.3, 1.32)
ax.set_xlabel("epipolar offset ε (px)", labelpad=0.5); ax.set_ylabel("prior weight w (median)")
ax.text(6.4, 1.24, "w = 1: single-image depth", fontsize=5.0, ha="right", va="top", color="#555555")
ax.text(6.4, -0.22, "w = 0: geometric depth", fontsize=5.0, ha="right", va="bottom", color="#555555")
# (b) class at eps ~ 0
ax = _axs[1]
classes = [("car", "w_car", "car"), ("crate", "w_crate", "crate"), ("crate_floating", "w_crate_floating", "floating\ncrate")]
_nm = len(_M7)
for i, model in enumerate(_M7):
    r = E3_TABLE[E3_TABLE["model"] == model].iloc[0]
    xs_ = [k + (i - (_nm - 1) / 2) * 0.11 for k in range(3)]
    ax.plot(xs_, [r[c[1]] for c in classes], MODEL_MARKERS[model], color=mcol(model), ms=2.6)
ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0); ax.axhline(1, color=GRID_COLOR, lw=0.5, zorder=0)
ax.set_xticks(range(3)); ax.set_xticklabels([c[2] for c in classes], fontsize=5.4)
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.3, 1.32); ax.set_ylabel("w at |ε| < 1 px (median)")
if _has_e4:
    ax = _axs[2]
    ax.axhspan(*PREREG["beta_support_range"], color=SUPPORT_SHADE, zorder=0, lw=0)
    ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0)
    Ks = list(RUN_CONFIG["e4_window_sizes"])
    for model in sorted(set(_E4T["model"]), key=EVALUATED_MODELS.index):
        for path, ls, off in (("straight", "-", -0.07), ("curved", (0, (3, 1.5)), 0.07)):
            g = _E4T[(_E4T["model"] == model) & (_E4T["path"] == path)].sort_values("K")
            ax.errorbar(np.arange(len(g)) + off, g["beta_s"],
                        yerr=[(g["beta_s"] - g["ci_lo"]).clip(lower=0), (g["ci_hi"] - g["beta_s"]).clip(lower=0)],
                        ls=ls, marker=MODEL_MARKERS[model], color=mcol(model), ms=2.6, lw=0.8, capsize=0.9,
                        mfc=mcol(model) if path == "straight" else "white", mew=0.7)
    ax.set_xticks(range(len(Ks))); ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlim(-0.4, len(Ks) - 0.6); ax.set_ylim(-0.25, 1.32)
    ax.set_xlabel("frames given jointly (K)"); ax.set_ylabel("β$_s$")
    ax.legend(handles=[Line2D([], [], color="#444444", ls="-", lw=0.8, marker="s", ms=2.6, label="straight path"),
                       Line2D([], [], color="#444444", ls=(0, (3, 1.5)), lw=0.8, marker="s", mfc="white", ms=2.6, label="curved twin")],
              loc="upper right", fontsize=4.9, handlelength=1.8)
fig.legend(handles=[Line2D([], [], color=mcol(m), marker=MODEL_MARKERS[m], ms=2.4, lw=0.8, label=mlab(m)) for m in _M7],
           loc="center left", bbox_to_anchor=(0.835 if _has_e4 else 0.86, 0.58), fontsize=5.1, handlelength=1.3, labelspacing=0.35)
save_figure(fig, "fig8_prior_multiview", "Prior weight, object class and number of views",
            "Does the prior weight grow with the epipolar offset and depend on the class, and do more views change beta_s?",
            "Sections 6.3 and 6.4", "double column",
            dict(by_eps=E3_BY_EPS[E3_BY_EPS["model"].isin(_M7)].to_dict("records"),
                 e3=E3_TABLE[E3_TABLE["model"].isin(_M7)].to_dict("records"), e4=_E4T.to_dict("records") if _has_e4 else []),
            caption=cap("a", "Prior weight by offset", "Prior weight w (0: the geometric static-hypothesis depth, 1: the "
                        "model's single-image depth) against the exact epipolar offset, median per bin, where the two hypotheses "
                        "are separable.")
                    + " " + cap("b", "Prior weight by class", "w for movers in the parallel regime by object class.")
                    + (" " + cap("c", "More views", "beta_s of the multi-view models given K frames jointly, on a straight camera "
                                 "path and on its curved twin (identical frames 1-2).") if _has_e4 else "") + gate_caption_note("MAIN"))

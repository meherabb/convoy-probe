# PLACEHOLDER
# Figure 6 (double column): what the geometry share implies for dynamic masks. Recall at the image's
# 5% static false-positive rate: (a) against the exact epipolar offset, M1 epipolar residual (no
# model) and M2 rigid-flow residual with each model's depth (CONVOY-3D); (b) exploratory, M2 recall
# on the same parallel movers in each ablation condition (Section 6.7c). The M5 remedy is in Fig. S3.
EPS_TICKLABELS = ["<0.25", "0.25–1", "1–2", "2–3", "3–6", "6–10", ">10"]

def _eps_axes(ax, ylabel=True):
    ax.axvspan(-0.5, 1.5, color=PAR_SHADE, zorder=0, lw=0)
    ax.axvspan(3.5, 6.5, color=PERP_SHADE, zorder=0, lw=0)
    ax.set_xticks(range(7)); ax.set_xticklabels(EPS_TICKLABELS, rotation=40, ha="right", rotation_mode="anchor", fontsize=5.2)
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel("epipolar offset ε (px)", labelpad=0.5)
    if ylabel:
        ax.set_ylabel(f"recall at {_fpr_pct} static FPR")

_E5M = E5_MASKS[E5_MASKS["model"].isin(MODELS_OK["ABL"])] if len(E5_MASKS) else E5_MASKS
_has_e5m = len(_E5M) > 0
_panels = ["m12"] + (["abl"] if _has_e5m else [])
_fpr_pct = f"{100 * PREREG['mask_fpr']:g}%"
_titles = {"m12": "Recall by epipolar offset", "abl": "Recall by support"}
_lt5 = {k: "ab"[i] for i, k in enumerate(_panels)}
fig = plt.figure(figsize=FIG_DOUBLE)
_x0, _gap = 0.07, 0.08
_w = (0.79 - _x0 - _gap * (len(_panels) - 1)) / len(_panels)
_axes = {k: fig.add_axes([_x0 + i * (_w + _gap), 0.29, _w, 0.6]) for i, k in enumerate(_panels)}
for i, k in enumerate(_panels):
    panel_title(_axes[k], "ab"[i], _titles[k], x=-0.08 if i == 0 else -0.06)
# (a) M1 and M2
ax = _axes["m12"]; _eps_axes(ax)
g = E2_BY_EPS[(E2_BY_EPS["model"] == "(model-independent)") & (E2_BY_EPS["method_key"] == "M1_exact") & (E2_BY_EPS["n"] >= 3)]
ax.plot(g.sort_values("eps_bin")["eps_bin"], g.sort_values("eps_bin")["recall"], color=CONTROL_COLOR, marker="o", ms=2.0, lw=0.9, zorder=5)
for model in MODELS_OK["MAIN"]:
    gg = E2_BY_EPS[(E2_BY_EPS["model"] == model) & (E2_BY_EPS["method_key"] == "M2_exact") & (E2_BY_EPS["n"] >= 3)].sort_values("eps_bin")
    if len(gg):
        ax.plot(gg["eps_bin"], gg["recall"], marker=MODEL_MARKERS[model], color=mcol(model), ms=2.0, lw=0.7)
# (b) M2 recall by ablation condition (same parallel movers, exact correspondences)
if _has_e5m:
    ax = _axes["abl"]
    _nm = len(MODELS_OK["ABL"])
    for i, model in enumerate(MODELS_OK["ABL"]):
        gg = _E5M[_E5M["model"] == model].set_index("condition").reindex(E5_CONDS)
        xo = np.arange(len(E5_CONDS)) + (i - (_nm - 1) / 2) * 0.09
        ax.plot(xo, gg["recall_M2"], color=mcol(model), lw=0.6, alpha=0.55)
        ax.errorbar(xo, gg["recall_M2"], yerr=[(gg["recall_M2"] - gg["ci_lo"]).clip(lower=0), (gg["ci_hi"] - gg["recall_M2"]).clip(lower=0)],
                    fmt=MODEL_MARKERS[model], color=mcol(model), ms=2.2, capsize=0.8, lw=0.6)
    ax.axvspan(2.5, 4.5, color="#F2F2F2", zorder=0, lw=0)
    ax.set_xticks(range(len(E5_CONDS)))
    ax.set_xticklabels(["grounded", "no\nshadow", "contact\nhidden", "floating", "floating\n+ wall"], fontsize=5.2)
    ax.set_xlim(-0.5, len(E5_CONDS) - 0.5); ax.set_ylim(-0.03, 1.06)
    ax.text(0.02, 0.97, f"same {int(_E5M['n'].max())} parallel movers (ε = 0)", transform=ax.transAxes, fontsize=5.0, va="top",
            color="#555555")
_leg5 = [m for m in EVALUATED_MODELS if m in MODELS_OK["MAIN"] or m in MODELS_OK["ABL"]]
handles = [Line2D([], [], color=mcol(m), marker=MODEL_MARKERS[m], ms=2.4, lw=0.8, label=mlab(m)) for m in _leg5]
handles += [Line2D([], [], color=CONTROL_COLOR, marker="o", ms=2.0, lw=0.9, label="M1 (no model)")]
handles += [mpatches.Patch(color=PAR_SHADE, label=f"|ε| < {PREREG['eps_parallel_max']:g} px"),
            mpatches.Patch(color=PERP_SHADE, label=f"ε ≥ {PREREG['eps_perp_min']:g} px")]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.805, 0.56), fontsize=5.2, handlelength=1.6, labelspacing=0.32)
_cap5 = (f"Recall at the image's {_fpr_pct} static false-positive rate. "
         + cap(_lt5["m12"], "Recall by epipolar offset", "The epipolar residual M1 (black) needs no model and is the most any "
               "two-view geometric test can see; the rigid-flow residual M2 (colours) uses each model's depth; shading marks the "
               "parallel and perpendicular regimes."))
if _has_e5m:
    _cap5 += " " + cap(_lt5["abl"], "Recall by support", "Exploratory: M2 recall on the same purely parallel movers in "
                       "each ground-contact condition of E5 (95% cluster-bootstrap intervals over base scenes); shaded: the lifted "
                       "conditions. The exploratory remedy M5 is shown in Fig. S3.")
_cap5 += gate_caption_note("MAIN", "ABL")
save_figure(fig, "fig6_masks", "Dynamic masks miss what geometry cannot see",
            "How does the recall of geometric masks depend on the epipolar offset and on the support cue?",
            "Sections 6.2 and 6.7c", "double column",
            dict(by_eps=E2_BY_EPS[E2_BY_EPS["model"].isin(["(model-independent)"] + MODELS_OK["MAIN"])].to_dict("records"),
                 by_condition=_E5M.to_dict("records") if _has_e5m else []),
            caption=_cap5)

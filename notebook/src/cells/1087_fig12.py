# PLACEHOLDER
# Figure 12 (double column, v6.5, exploratory): two channels. (a) The cue model fitted on CONVOY-3D and E5 predicts the slope
# of every other testbed without refitting: observed against predicted. (b) The hump: binned rho against x on CONVOY-3D,
# observed and as the two-channel and contact-only models fit it. (c) The fitted channels per model: the share of geometry at
# unlimited parallax for each support (contact) and the half-saturation parallax kappa (parallax). (d) Masks that defeat
# themselves: the change of M2 recall against the change of beta_s between conditions.
_TB_MARK12 = {"BSW": "o", "WDR": "s", "REAL": "^", "RABL": "v", "VK": "D", "AV2": "P"}
_TB_NAME12 = {"BSW": "baseline sweep", "WDR": "dose-response", "REAL": "CONVOY-REAL", "RABL": "E5-REAL", "VK": "VKITTI 2",
              "AV2": "Argoverse 2"}
_pred12 = CUE_PRED[(CUE_PRED["role"] == "predicted") & (CUE_PRED["group"] == "all")] if len(CUE_PRED) else CUE_PRED
_has12 = len(_pred12) > 0 and len(CUE_FIT_TABLE) > 0
if _has12:
    fig = plt.figure(figsize=FIG_DOUBLE)
    # (a) observed against predicted
    ax = fig.add_axes([0.08, 0.25, 0.18, 0.62])
    panel_title(ax, "a", "Out of sample", x=-0.33)
    _lo12 = float(np.nanmin(np.r_[_pred12["beta_predicted"], _pred12["ci_lo"], -0.1]))
    _hi12 = float(np.nanmax(np.r_[_pred12["beta_predicted"], _pred12["ci_hi"], 0.8]))
    _lim = [_lo12 - 0.05, _hi12 + 0.05]
    ax.plot(_lim, _lim, color=CONTROL_COLOR, lw=0.6, ls=(0, (4, 2)), zorder=1)
    for r in _pred12.itertuples():
        ax.errorbar(r.beta_predicted, r.beta_observed, yerr=ci_err(r.beta_observed, r.ci_lo, r.ci_hi), fmt=_TB_MARK12.get(r.set, "o"),
                    color=mcol(r.model), ms=2.2, capsize=0.6, lw=0.5, mfc="white" if r.set in ("BSW", "WDR") else mcol(r.model), mew=0.5)
    ax.set_xlim(_lim); ax.set_ylim(_lim)
    ax.set_xlabel("predicted β (cue model)"); ax.set_ylabel("observed β (95% CI)", labelpad=1)
    ax.text(0.04, 0.96, f"r = {CUE_CALIBRATION['pearson_r']:.2f}", transform=ax.transAxes, fontsize=5.0, va="top", color="#555555")
    # (b) the hump
    ax = fig.add_axes([0.335, 0.25, 0.18, 0.62])
    panel_title(ax, "b", "The hump", x=-0.25)
    _hm = [m for m in ("DUSt3R", "MASt3R") if len(CUE_HUMP) and (CUE_HUMP["model"] == m).any()] or \
          (list(CUE_HUMP["model"].unique()[:2]) if len(CUE_HUMP) else [])
    for m in _hm:
        g = CUE_HUMP[CUE_HUMP["model"] == m]
        ax.plot(g["x_median"], g["rho_observed"], MODEL_MARKERS[m], color=mcol(m), ms=2.6, ls="none", zorder=3)
        ax.plot(g["x_median"], g["rho_two_channels"], color=mcol(m), lw=0.9, zorder=2)
        ax.plot(g["x_median"], g["rho_contact_only"], color=mcol(m), lw=0.7, ls=(0, (2, 1.5)), alpha=0.8, zorder=2)
    ax.set_xlabel("geometric answer x"); ax.set_ylabel("median ρ", labelpad=1)
    if _hm:
        ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0)
        ax.margins(y=0.1)
        ax.legend(handles=[Line2D([], [], ls="none", marker="o", color="#555555", ms=2.6, label="observed"),
                           Line2D([], [], color="#555555", lw=0.9, label="two channels"),
                           Line2D([], [], color="#555555", lw=0.7, ls=(0, (2, 1.5)), label="contact only")],
                  loc="best", fontsize=4.8, handlelength=1.4)
    else:
        ax.set_xticks([]); empty_panel(ax, "fewer than 5 movers\nper bin of x\nin this run")
    # (c) the fitted channels
    ax = fig.add_axes([0.61, 0.25, 0.17, 0.62])
    panel_title(ax, "c", "Two channels", x=-0.42)
    _ft = CUE_FIT_TABLE[CUE_FIT_TABLE["fit"] == "two channels"].reset_index(drop=True)
    _sup_mark = {"c_contact_in_view": ("o", "in view"), "c_contact_hidden": ("^", "hidden"), "c_unsupported": ("s", "floating")}
    for i, r in _ft.iterrows():
        y = len(_ft) - 1 - i
        for col, (mk, _) in _sup_mark.items():
            ax.plot(r[col], y, mk, color=mcol(r["model"]), ms=2.8, mfc="white" if col == "c_contact_in_view" else mcol(r["model"]), mew=0.6)
        ax.text(1.02, y, f"κ {r['kappa_px']:.0f}", transform=ax.get_yaxis_transform(), fontsize=4.8, va="center", ha="left")
    ax.set_yticks(range(len(_ft))); ax.set_yticklabels([mlab(m) for m in _ft["model"][::-1]], fontsize=5.2)
    ax.axvline(0, color=GRID_COLOR, lw=0.5, zorder=0)
    # the bottom band of the panel is kept free of data for the key of the three support classes
    ax.set_ylim(-1.45, len(_ft) - 0.4); ax.set_xlabel("c (geometry share at large parallax)", fontsize=5.6)
    ax.legend(handles=[Line2D([], [], ls="none", marker=mk, color="#555555", mfc="white" if k == "c_contact_in_view" else "#555555",
                              ms=2.8, label=lab) for k, (mk, lab) in _sup_mark.items()],
              loc="lower center", ncol=3, fontsize=4.6, handletextpad=0.3, columnspacing=0.8, handlelength=0.8, borderaxespad=0.1)
    # (d) masks that defeat themselves
    ax = fig.add_axes([0.875, 0.25, 0.115, 0.62])
    panel_title(ax, "d", "Masks", x=-0.5)
    _src_mark = {"E5": "o", "baseline sweep": "s", "dose-response": "^"}
    if len(TRADEOFF):
        for r in TRADEOFF.itertuples():
            ax.plot(r.delta_beta, r.delta_recall, _src_mark.get(r.source, "o"), color=mcol(r.model), ms=2.2, alpha=0.85)
        ax.axhline(0, color=GRID_COLOR, lw=0.5, zorder=0); ax.axvline(0, color=GRID_COLOR, lw=0.5, zorder=0)
        ax.text(0.95, 0.95, f"ρ$_S$ = {TRADEOFF_SUMMARY['spearman']:.2f}", transform=ax.transAxes, fontsize=4.8, ha="right", va="top",
                color="#555555")
    ax.set_xlabel("Δβ$_s$", labelpad=0.5); ax.set_ylabel("Δ M2 recall", labelpad=1)
    _ms12 = [m for m in EVALUATED_MODELS if (_ft["model"] == m).any()]
    _hl = [Line2D([], [], ls="", marker="o", color=mcol(m), ms=2.6, label=mlab(m)) for m in _ms12]
    _hl += [Line2D([], [], ls="", marker=_TB_MARK12[k], color="#555555", mfc="white" if k in ("BSW", "WDR") else "#555555", ms=2.6,
                   label=_TB_NAME12[k]) for k in _TB_MARK12 if (_pred12["set"] == k).any()]
    fig.legend(handles=_hl, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(_hl), fontsize=4.7, handletextpad=0.15,
               columnspacing=0.6, handlelength=0.9)
    _cap12 = (cap("a", "Out of sample", "The cue model rho - rho_mono = w (x - rho_mono), w = c_support d^2/(d^2 + kappa^2), fitted on "
                  "CONVOY-3D and the E5 conditions, predicts the slope of every other testbed from that testbed's own objects "
                  "(geometric answer, single-image error, parallax, support) without refitting; observed slope (95% interval) "
                  f"against predicted slope (Pearson r = {CUE_CALIBRATION['pearson_r']:.2f}, mean absolute error "
                  f"{CUE_CALIBRATION['mean_abs_error']:.2f}; open markers: the new designed sets).")
              + " " + cap("b", "The hump", "Median rho of CONVOY-3D parallel movers by x (markers) and the fits of the two-channel model "
                          "(solid) and of the contact-only model with constant weight (dashed)" + (f", for {' and '.join(mlab(m) for m in _hm)}" if _hm else "")
                          + "; in the two-channel model, any fall at large x comes from the parallax, which vanishes as s approaches 1"
                          + ("; fall from the observed peak to the largest x: " + ", ".join(
                              f"{mlab(r.model)} {r.fall_observed:.2f} observed, {r.fall_two_channels:.2f} two channels, "
                              f"{r.fall_contact_only:.2f} contact only" for r in CUE_HUMP_FALL[CUE_HUMP_FALL["model"].isin(_hm)].itertuples())
                             if len(CUE_HUMP_FALL) and CUE_HUMP_FALL["model"].isin(_hm).any() else "") + ".")
              + " " + cap("c", "Two channels", "Fitted share of geometry at unlimited parallax for objects with the contact in view "
                          "(open circles), with the contact hidden (triangles) and floating, i.e. without support (squares), and the "
                          "half-saturation parallax kappa in px (right).")
              + " " + cap("d", "Masks", "Change of M2 recall against the change of beta_s between conditions of the same movers "
                          "(E5 conditions against grounded, baseline levels against x0.5, wall heights against no wall); "
                          f"Spearman {TRADEOFF_SUMMARY['spearman']:.2f}." if len(TRADEOFF) else "No paired recall in this run.")
              + " Exploratory (AMENDMENT_V6_5).")
    save_figure(fig, "fig12_two_channels", "Two channels: contact and parallax",
                "Does one cue-combination model with a contact channel and a parallax channel explain every testbed?",
                "Section 6.11", "double column", dict(fits=CUE_FIT_TABLE.to_dict("records"), predictions=CUE_PRED.to_dict("records"),
                                                      hump=CUE_HUMP.to_dict("records"), tradeoff=TRADEOFF.to_dict("records")),
                caption=_cap12)
else:
    print("Figure 12 is not drawn: the cue model could not be fitted in this run.")

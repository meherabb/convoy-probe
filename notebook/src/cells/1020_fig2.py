# PLACEHOLDER
# Figure 2 (double column): the testbeds and the two checks that make the probe interpretable.
# (a-c) Three testbeds (CONVOY-3D, then CONVOY-REAL and Virtual KITTI 2 when they are part of the run,
# filled up with LEGACY-FLAT and a second CONVOY-3D camera): image with the movers outlined by regime,
# and exact depth. Then the static-scene validity gate and the oracle check (beta_s recovered from
# planted predictors against the slope-weight identity). Panel letters follow the panels drawn.
def _exemplar_scene(movers, metas, prefer_kind=None):
    best, best_n = metas[0], -1
    for meta in metas[: min(len(metas), 30)]:
        mv = movers[movers["scene"] == meta["index"]]
        n = mv["regime"].nunique() + 0.1 * len(mv) + (0.5 if prefer_kind and meta["scene"]["camera"]["kind"] == prefer_kind else 0)
        if n > best_n:
            best, best_n = meta, n
    return best

def _outline(ax, mask, color, lw=0.7):
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)

def _main_panel(meta, name):
    fr = load_frame(meta["dir"], 0)
    rg = {int(r.idx): r.regime for r in MOVERS_MAIN[MOVERS_MAIN["scene"] == meta["index"]].itertuples()}
    return (name, fr["rgb"], depth_to_rgb(fr["depth"]), [(fr["mover_id"] == j, REGIME_COLORS[r]) for j, r in rg.items()])

_tb_panels = [_main_panel(_exemplar_scene(MOVERS_MAIN, META_MAIN, prefer_kind="forward"), "CONVOY-3D")]
if META_REAL:
    _mr = _exemplar_scene(MOVERS_REAL, META_REAL)
    _fr = load_frame(_mr["dir"], 0)
    _rgr = {int(r.idx): r.regime for r in MOVERS_REAL[MOVERS_REAL["scene"] == _mr["index"]].itertuples()}
    _tb_panels.append(("CONVOY-REAL", _fr["rgb"], depth_to_rgb(_fr["depth"]),
                       [(_fr["mover_id"] == j, REGIME_COLORS[r]) for j, r in _rgr.items()]))
if VK_STATUS["available"] and len(MOVERS_VK):
    _vk_best = MOVERS_VK.groupby("scene").agg(n=("regime", "size"), k=("regime", "nunique")).sort_values(["k", "n"]).index[-1]
    _vm = VK_PAIR_META[int(_vk_best)]
    _vs = VK_SCENE_OBJS[_vm["seq"]]; _vg = _vs.pair_gt(*_vm["frames"])
    _rgv = MOVERS_VK[MOVERS_VK["scene"] == _vk_best].set_index("mover")["regime"].to_dict()
    _tb_panels.append(("Virtual KITTI 2", _vs.rgb(_vm["frames"][0]), depth_to_rgb(np.where(np.isfinite(_vg["depth"]), _vg["depth"], 655.0)),
                       [(_vg["mover_id"] == k, REGIME_COLORS[r]) for k, r in _rgv.items()]))
if len(_tb_panels) < 3:
    _fl = META_FLAT[0]
    _tb_panels.append(("LEGACY-FLAT", load_rgb(_fl["dir"], 0), load_rgb(_fl["dir"], 1) / 255.0, []))
if len(_tb_panels) < 3:
    _ms = [mm for mm in META_MAIN if mm["scene"]["camera"]["kind"] == "side"]
    if _ms:
        _tb_panels.append(_main_panel(_exemplar_scene(MOVERS_MAIN, _ms), "CONVOY-3D, side"))
_L2 = "abcdefg"
_n_tb = len(_tb_panels)

fig = plt.figure(figsize=FIG_DOUBLE)
gs = fig.add_gridspec(2, 3, left=0.008, right=0.505, bottom=0.17, top=0.885, wspace=0.035, hspace=0.04)
for c, (name, top, bottom, outlines) in enumerate(_tb_panels):
    a_top = fig.add_subplot(gs[0, c]); a_bot = fig.add_subplot(gs[1, c])
    a_top.imshow(top); a_bot.imshow(bottom)
    for m, col in outlines:
        _outline(a_top, m, col)
    clean_image_axes(a_top); clean_image_axes(a_bot)
    panel_title(a_top, _L2[c], name)
    tag(a_bot, {"LEGACY-FLAT": "second view", "CONVOY-REAL": "depth"}.get(name, "exact depth"), fs=4.8)
fig.legend(handles=[Line2D([], [], color=cc, lw=1.0, label=rg) for rg, cc in REGIME_COLORS.items()],
           loc="lower center", bbox_to_anchor=(0.257, 0.0), ncol=3, fontsize=5.3, handlelength=1.2, columnspacing=1.2)
# (d) gate
ax = fig.add_axes([0.585, 0.24, 0.17, 0.645])
panel_title(ax, _L2[_n_tb], "Validity gate", x=-0.3)
ax.add_patch(mpatches.Rectangle((0, PREREG["gate_slope_range"][0]), PREREG["gate_absrel_max"],
                                PREREG["gate_slope_range"][1] - PREREG["gate_slope_range"][0], fc=SUPPORT_SHADE, ec="none", zorder=0))
ax.text(0.012, PREREG["gate_slope_range"][1] + 0.04, "pass", fontsize=5.2, color="#2E7D32", va="bottom")
TB_MARK = {"MAIN": "o", "ABL": "o", "REAL": "^", "VK": "D", "KT": "v", "AV2": "P", "BONN": "h", "FLAT": "X"}
for r in GATE.itertuples():
    if str(r.set) in ("ABL", "RABL", "BSW", "WDR"):   # same backgrounds as CONVOY-3D and CONVOY-REAL; listed in T2
        continue
    ax.plot(min(r.absrel_median, 0.95), r.slope_median, TB_MARK[str(r.set)], color=mcol(r.model), ms=3.0,
            mec="black" if str(r.set) == "FLAT" else mcol(r.model), mew=0.3, alpha=0.9)
ax.set_xlabel("AbsRel, static pixels"); ax.set_ylabel("log-depth slope, static pixels")
ax.set_xlim(0, 1.0); ax.set_ylim(-0.3, 1.45); ax.set_xticks([0, 0.5, 1.0])
_present = [k for k in ["MAIN", "REAL", "VK", "AV2", "KT", "BONN", "FLAT"] if k in set(GATE["set"].astype(str))]
_tbl = {"MAIN": "CONVOY-3D", "REAL": "CONVOY-REAL", "VK": "VKITTI 2", "AV2": "Argoverse 2", "KT": "KITTI", "BONN": "Bonn",
        "FLAT": "LEGACY-FLAT"}
ax.legend(handles=[Line2D([], [], ls="", marker=TB_MARK[k], color="#555555", mec="black" if k == "FLAT" else "#555555",
                          mew=0.3, ms=3, label=_tbl[k]) for k in _present], loc="upper right", fontsize=4.9, handletextpad=0.2)
# (e) oracle
ax = fig.add_axes([0.835, 0.24, 0.145, 0.645])
panel_title(ax, _L2[_n_tb + 1], "Estimator check", x=-0.3)
ax.plot(SLOPE_WEIGHT_TABLE["w"], SLOPE_WEIGHT_TABLE["beta_ols_over_design"], color=CONTROL_COLOR, lw=0.8, label="identity β(w)")
ax.errorbar(ORACLE_TABLE["w_true"], ORACLE_TABLE["beta_s"],
            yerr=[ORACLE_TABLE["beta_s"] - ORACLE_TABLE["ci_lo"], ORACLE_TABLE["ci_hi"] - ORACLE_TABLE["beta_s"]],
            fmt="o", color="#0072B2", ms=2.8, capsize=1.0, lw=0.7, label="CONVOY-3D")
if len(ORACLE_VK) and ORACLE_VK["beta_geo"].notna().any():
    ax.plot(ORACLE_VK["w_true"], ORACLE_VK["beta_geo"], "D", mfc="white", mec="#D55E00", ms=2.6, mew=0.7, label="VKITTI 2")
ax.set_xlabel("planted weight w"); ax.set_ylabel("recovered β$_s$")
ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.12, 1.12); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.legend(loc="lower left", fontsize=4.9, handlelength=1.0)
fig.legend(handles=[Line2D([], [], ls="", marker="o", color=mcol(m), ms=2.8, label=mlab(m)) for m in EVALUATED_MODELS],
           loc="lower center", bbox_to_anchor=(0.66, 0.0), ncol=len(EVALUATED_MODELS), fontsize=5.0, handletextpad=0.1,
           columnspacing=0.7)
_tb_desc = {"CONVOY-3D": "CONVOY-3D (ray-cast streets)",
            "CONVOY-REAL": "CONVOY-REAL (real TUM images and their measured depth, with inserted movers of exact depth)",
            "Virtual KITTI 2": "Virtual KITTI 2 (photo-realistic driving)",
            "LEGACY-FLAT": "LEGACY-FLAT, the 2-D testbed of the previous version (bottom: its second view)",
            "CONVOY-3D, side": "CONVOY-3D with a side-looking camera"}
_tb_list = [_tb_desc[p[0]] for p in _tb_panels]
_cap2 = (f"({_L2[0]})-({_L2[_n_tb - 1]}) **Testbeds.** " + (", ".join(_tb_list[:-1]) + " and " + _tb_list[-1] if _n_tb > 1 else _tb_list[0])
         + "; top: first view with the movers outlined by their exact motion regime, bottom: ground-truth depth. "
         + cap(_L2[_n_tb], "Validity gate", "Medians over images of AbsRel and of the log-depth slope on static pixels; "
               "the shaded box is the pre-registered pass region"))
_gf = GATE[GATE["set"] == "FLAT"]
if len(_gf):
    _nf = int((_gf["gate"] == "FAIL").sum())
    _cap2 += (", and LEGACY-FLAT falls outside it for every model" if _nf == len(_gf)
              else f", and LEGACY-FLAT falls outside it for {_nf} of {len(_gf)} models")
_odev = float(np.nanmax(np.abs(ORACLE_TABLE["beta_s"] - ORACLE_TABLE["beta_theory_identity"])))
_cap2 += (". " + cap(_L2[_n_tb + 1], "Estimator check", "Predictors with a planted prior weight w") +
          (f" recover the slope-weight identity (largest deviation {_odev:.3f}), so beta_s reads as a geometry share" if ORACLE_OK else
           f": recovered beta_s against the slope-weight identity (largest deviation {_odev:.3f}; the estimator check did not pass)")
          + ("; open diamonds: the Virtual KITTI 2 estimator of H9." if len(ORACLE_VK) and ORACLE_VK["beta_geo"].notna().any() else "."))
save_figure(fig, "fig2_testbeds_checks", "Testbeds and the checks that make the probe interpretable",
            "Do the models reconstruct the static scene of each testbed, and does the estimator recover a known answer?",
            "Sections 3, 6.0 and 6.0b", "double column",
            dict(gate=GATE.astype({"set": str}).to_dict("records"), oracle=ORACLE_TABLE.to_dict("records"),
                 oracle_vk=ORACLE_VK.to_dict("records") if len(ORACLE_VK) else []),
            caption=_cap2)

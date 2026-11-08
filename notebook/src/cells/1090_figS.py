# PLACEHOLDER
# Figure S1 (single column): assumption A2. The locked first-order epipolar offset f||delta_perp||/Z
#     and the corrected first-order offset (out-of-plane component only) against the exact offset.
fig = plt.figure(figsize=FIG_SINGLE)
ax = fig.add_axes([0.17, 0.19, 0.79, 0.74])
sub = A2_DIAG.sample(min(2500, len(A2_DIAG)), random_state=0)
ax.scatter(sub["eps_exact"], sub["eps_locked_first_order"], s=1.2, color="#D55E00", alpha=0.45, lw=0,
           label="locked: f‖δ$_⊥$‖/Z", rasterized=True)
ax.scatter(sub["eps_exact"], sub["eps_corrected_first_order"], s=1.2, color="#0072B2", alpha=0.45, lw=0,
           label="corrected: out-of-plane part", rasterized=True)
lim = [0.05, 400]
ax.plot(lim, lim, color="black", lw=0.6, ls=(0, (4, 2)))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("exact epipolar offset ε (px)"); ax.set_ylabel("first-order prediction (px)")
ax.legend(loc="lower right", fontsize=5.0, markerscale=3)
panel_title(ax, None, "First-order versus exact offset")
save_figure(fig, "figS1_a2_first_order", "Assumption A2: first-order epipolar offset",
            "Is eps ~ f||delta_perp||/Z adequate?", "Section 2.3", "single column", dict(summary=A2_SUMMARY),
            caption="**First-order offset.** The locked first-order prediction of the epipolar offset (orange) and the corrected "
                    "prediction that keeps only the motion component leaving the epipolar plane (blue), against the exact "
                    "offset, over street-scene configurations. **Consequence.** Every regime in the paper is assigned from the "
                    "exact offset.")

# Figure S2 (double column): gallery of the testbeds (CONVOY-3D camera kinds, an ablation render,
#     CONVOY-REAL, a real-image contact-ablation composite, Virtual KITTI 2, TUM walking).
gal = []                                       # (image, tag on the tile, description in the caption)
for kind in ["forward", "oblique", "side"]:
    m = next((mm for mm in META_MAIN if mm["scene"]["camera"]["kind"] == kind), None)
    if m is not None:
        gal.append((load_rgb(m["dir"], 0), f"CONVOY-3D, {kind}",
                     f"CONVOY-3D with a {kind}{'' if kind == 'oblique' else '-looking'} camera"))
_abl_fh = next((mm for mm in META_ABL if mm["condition"] == "FH" and mm["index"] in EXEMPLAR_ABL), None)
if _abl_fh is not None:
    gal.append((load_rgb(_abl_fh["dir"], 0), "E5, floating behind wall",
                "the ground-contact ablation with the mover lifted and behind the wall"))
_short = {"fr3_structure_texture_far": "structure-texture", "fr3_long_office_household": "long office"}
for seq_name in list(dict.fromkeys(mm["seq"] for mm in META_REAL))[:1]:
    mm = next(mm for mm in META_REAL if mm["seq"] == seq_name)
    _sn = _short.get(seq_name, seq_name.replace('fr3_', ''))
    gal.append((load_rgb(mm["dir"], 0), f"CONVOY-REAL, {_sn}", f"a CONVOY-REAL composite (TUM {_sn})"))
_rg = next((mm for mm in META_RABL if mm["condition"] == "G" and mm["seq"] not in {m["seq"] for m in META_REAL[:1]}), None) \
    or next((mm for mm in META_RABL if mm["condition"] == "G"), None)
if _rg is not None:
    _sn = _short.get(_rg["seq"], _rg["seq"].replace('fr3_', ''))
    gal.append((load_rgb(_rg["dir"], 0), f"E5-REAL, grounded, {_sn}",
                f"a real-image contact-ablation composite with grounded crates (TUM {_sn})"))
if VK_STATUS["available"]:
    for name in list(VK_SCENE_OBJS)[:1]:
        f0 = VK_SELECTION[name]["pairs"][0][0] if VK_SELECTION[name]["pairs"] else None
        if f0 is not None:
            gal.append((VK_SCENE_OBJS[name].rgb(int(f0)), f"Virtual KITTI 2, {name}", f"a Virtual KITTI 2 frame ({name})"))
if AV2_STATUS["available"] and AV2_PAIR_META:
    _am = AV2_PAIR_META[0]
    gal.append((np.asarray(Image.open(os.path.join(_am["dir"], "rgb0.png")).convert("RGB")), f"Argoverse 2, {_am['seq']}",
                f"an Argoverse 2 front-camera frame (log {_am['seq']}, 4:3 window)"))
for name, p in list(TUM_THUMBS.items())[:1]:
    _sn = name.replace('fr3_', '')
    gal.append((np.asarray(Image.open(p).convert("RGB")), f"TUM {_sn}", f"a TUM frame with walking people ({_sn})"))
if len(gal) > 8:                               # eight tiles: the oblique CONVOY-3D camera gives way to the real-driving frame
    gal = [g for g in gal if g[1] != "CONVOY-3D, oblique"][:8]
fig = plt.figure(figsize=FIG_DOUBLE)
gs = fig.add_gridspec(2, 4, left=0.071, right=0.929, bottom=0.01, top=0.99, wspace=0.03, hspace=0.04)
for k, (img, lab, _) in enumerate(gal):
    a = fig.add_subplot(gs[k // 4, k % 4])
    a.imshow(img); clean_image_axes(a, frame=False)
    tag(a, f"({'abcdefgh'[k]}) {lab}", fs=4.8)            # tile titles (D60)
save_figure(fig, "figS2_gallery", "Testbed gallery", "What do the inputs look like?", "Section 3", "double column",
            dict(labels=[g[1] for g in gal]),
            caption=" ".join(cap("abcdefgh"[k], g[1], g[2][0].upper() + g[2][1:] + ".") for k, g in enumerate(gal)))

# Figure S3 (single column): the exploratory remedy M5 (disagreement between a model's two-view and
#     single-image depth) on parallel CONVOY-3D movers, by the depth discrepancy |x| that geometry
#     imposes, against M2 averaged over models (moved here from Fig. 6 in v6.2).
_M5X = E2_M5_BY_X[E2_M5_BY_X["model"].isin(MODELS_OK["MAIN"])] if len(E2_M5_BY_X) else E2_M5_BY_X
_m5 = _M5X[_M5X["n"] >= 3] if len(_M5X) else _M5X
if len(_m5):
    fig = plt.figure(figsize=FIG_SINGLE)
    ax = fig.add_axes([0.15, 0.25, 0.5, 0.68])
    _xv = [0, 0.1, 0.3, 0.7]
    for model in MODELS_OK["MAIN"]:
        gg = _m5[_m5["model"] == model]
        if len(gg):
            ax.plot([_xv.index(v) for v in gg["abs_x_lo"]], gg["recall_M5"], marker=MODEL_MARKERS[model], color=mcol(model), ms=2.0, lw=0.8)
    _m2m = _m5.groupby("abs_x_lo")["recall_M2_exact"].mean()
    ax.plot([_xv.index(v) for v in _m2m.index], _m2m.values, color=MONO_COLOR, ls=(0, (3, 1.5)), lw=0.9)
    ax.set_xticks(range(4)); ax.set_xticklabels(["<0.1", "0.1–0.3", "0.3–0.7", ">0.7"], rotation=40, ha="right",
                                                rotation_mode="anchor", fontsize=5.2)
    ax.set_xlim(-0.4, 3.4); ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel("|x| = |log(1 − s)|", labelpad=0.5); ax.set_ylabel(f"recall at {100 * PREREG['mask_fpr']:g}% static FPR")
    _h = [Line2D([], [], color=mcol(m), marker=MODEL_MARKERS[m], ms=2.2, lw=0.8, label=mlab(m)) for m in MODELS_OK["MAIN"]
          if (_m5["model"] == m).any()]
    _h += [Line2D([], [], color=MONO_COLOR, ls=(0, (3, 1.5)), lw=0.9, label="M2, mean")]
    fig.legend(handles=_h, loc="center left", bbox_to_anchor=(0.67, 0.58), fontsize=5.0, handlelength=1.5, labelspacing=0.3)
    panel_title(ax, None, "Remedy M5 by depth discrepancy")
    save_figure(fig, "figS3_m5_remedy", "Exploratory remedy M5", "Does a model's own single-image depth expose parallel movers?",
                "Section 6.2", "single column", dict(m5_by_x=_m5.to_dict("records")),
                caption="**Remedy M5 (exploratory).** The disagreement between a model's two-view and single-image depth, on "
                        "purely parallel CONVOY-3D movers, by the depth discrepancy |x| that geometry imposes. **Reference.** "
                        "Dashed: the rigid-flow residual M2 averaged over models, on the same movers. Recall at the image's 5% "
                        "static false-positive rate." + gate_caption_note("MAIN"))
else:
    print("Figure S3 is not drawn: no M5 bins with at least 3 movers in this run.")
release_memory()

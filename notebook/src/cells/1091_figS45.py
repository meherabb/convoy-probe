# PLACEHOLDER
# Figure S4 (double column, v6.5): real vehicles, their wheel contact points and their static twin (Argoverse 2). Each tile:
#     one moving vehicle, its wheel contact points coloured by the visibility rule of Section 3.11b, with its parallel
#     fraction s and the height of its static twin above the road.
# Figure S5 (double column, v6.5): the designed sets. Top: the second view of one baseline-sweep scene at the four baselines.
#     Bottom: one dose-response scene with no wall and the three walls.
_PT_STYLE = {"visible": ("o", "#00C853"), "body": ("x", "#BDBDBD"), "object": ("x", "#FF9800"), "static": ("x", "#E53935")}


def _crop_offset(img, bbox, out_hw, pad):
    # the crop of crop_around and the image coordinates of its top-left corner
    H_, W_ = img.shape[:2]
    u0, v0, u1, v1 = bbox
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    h, w = (v1 - v0) * (1 + 2 * pad), (u1 - u0) * (1 + 2 * pad)
    ar = out_hw[1] / out_hw[0]
    if w / max(h, 1) > ar:
        h = w / ar
    else:
        w = h * ar
    h, w = min(h, H_), min(w, W_)
    a0 = int(np.clip(cu - w / 2, 0, W_ - w)); b0 = int(np.clip(cv - h / 2, 0, H_ - h))
    return img[b0:b0 + int(h), a0:a0 + int(w)], a0, b0


if len(AV2_TWIN) and len(MOVERS_AV2_ALL):
    _g = MOVERS_AV2_ALL.merge(AV2_TWIN.drop(columns=["track"]), on=["scene", "mover"], how="left")
    _g = _g[(_g["state"] == "moving") & _g["measurable"].astype(bool) & _g["gt_contact"].isin(["visible", "hidden"])
            & np.isfinite(_g["H_cam"])].copy()
    _g["h_twin"] = _g["H_cam"] * (1.0 - np.exp(_g["gt_x_geo"]))
    _g = _g[np.isfinite(_g["h_twin"])]                     # a vehicle without a static solution has no twin to show
    _g["_par"] = (_g["regime"] == "parallel").astype(int)
    _rows4 = [(c, _g[_g["gt_contact"] == c].sort_values(["_par", "gt_n_lidar_px"], ascending=False).drop_duplicates("track").head(4))
              for c in ("visible", "hidden")]
    _rows4 = [(c, q) for c, q in _rows4 if len(q)]
    _metas = {m_["index"]: m_ for m_ in AV2_PAIRS_ALL}
    if _rows4:
        fig = plt.figure(figsize=FIG_DOUBLE)
        # two rows (contact visible, contact hidden) of up to four 4:3 tiles, centred, with the row name on the left
        _th, _gap = 0.76, 0.04
        _tw = _th * 4 / 3
        _x0 = (5.5 - (4 * _tw + 3 * _gap + 0.14)) / 2 + 0.14
        _tags, _caps, _recs = "abcdefgh", [], []
        k = 0
        for row, (c, q) in enumerate(_rows4):
            y0 = 0.2 + (len(_rows4) - 1 - row) * (_th + _gap)
            fig.text((_x0 - 0.07) / 5.5, (y0 + _th / 2) / 1.8, f"contact {c}", rotation=90, ha="center", va="center", fontsize=5.2)
            for col, r in enumerate(q.itertuples()):
                m_ = _metas[int(r.scene)]
                img = np.asarray(Image.open(os.path.join(m_["dir"], "rgb0.png")).convert("RGB"))
                px = np.asarray(m_["vehicles"][int(r.mover)]["pixels"], int).reshape(-1, 2)
                pts = json.loads(r.wheel_points) if isinstance(r.wheel_points, str) else []
                allp = np.vstack([px] + ([np.array([[p_[0], p_[1]] for p_ in pts])] if pts else []))
                bb = (allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max())
                crop, a0, b0 = _crop_offset(img, bb, (3, 4), 0.35)
                a = fig.add_axes([(_x0 + col * (_tw + _gap)) / 5.5, y0 / 1.8, _tw / 5.5, _th / 1.8])
                a.imshow(crop, aspect="auto"); clean_image_axes(a)
                for (u, v, st) in pts:
                    mk, colr = _PT_STYLE.get(st, ("o", "white"))
                    a.plot(u - a0, v - b0, mk, color=colr, ms=3.2, mew=0.9, mfc="none" if mk == "o" else colr)
                a.set_xlim(-0.5, crop.shape[1] - 0.5); a.set_ylim(crop.shape[0] - 0.5, -0.5)
                tag(a, f"({_tags[k]}) s = {r.s:.2f}, twin {r.h_twin:+.1f} m".replace("-", "−"), fs=4.5)
                _caps.append(cap(_tags[k], f"Contact {c}", f"A moving Argoverse 2 vehicle ({r.regime} regime, s = {r.s:.2f}); its "
                                 f"static twin's contact would lie {abs(r.h_twin):.1f} m {'below' if r.h_twin < 0 else 'above'} its "
                                 "contact plane."))
                _recs.append(dict(uid=r.uid, gt_contact=c, s=float(r.s), regime=r.regime, h_twin=float(r.h_twin), H_cam=float(r.H_cam)))
                k += 1
        fig.legend(handles=[Line2D([], [], ls="", marker=mk, color=colr, mfc="none" if mk == "o" else colr, ms=3.2, mew=0.9,
                                   label={"visible": "visible", "body": "hidden by its body", "object": "hidden by an object",
                                          "static": "hidden by static structure"}[st]) for st, (mk, colr) in _PT_STYLE.items()],
                   loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=5.0, handletextpad=0.2, columnspacing=1.0)
        save_figure(fig, "figS4_av2_contact_twin", "Real vehicles: contact points and static twins",
                    "What does the contact rule see on real vehicles, and where are their static twins?", "Sections 3.11b, 3.11c and 6.13",
                    "double column", dict(vehicles=_recs),
                    caption=" ".join(_caps) + " **Markers.** Wheel contact points by the rule of Section 3.11b: visible (circles), "
                                              "hidden by the vehicle's own body, by another labelled object or by static structure "
                                              "(crosses).")
else:
    print("Figure S4 is not drawn: no Argoverse 2 camera heights in this run.")

# ---- Figure S5: the designed sets ------------------------------------------------------------------------------------
_s5b = sorted({m["index"] for m in META_BSW})
_s5w = sorted({m["index"] for m in META_WDR})
if _s5b and _s5w:
    def _richest(metas, idx_list, cond):
        best = None
        for i in idx_list[: min(12, len(idx_list))]:
            m_ = next((m for m in metas if m["index"] == i and m["condition"] == cond), None)
            if m_ is None:
                continue
            n = sum(1 for g_ in m_["gt01"] if g_.get("n_eroded", 0) >= PREREG["min_object_pixels"])
            if best is None or n > best[0]:
                best = (n, i)
        return best[1] if best else idx_list[0]
    _ib = _richest(META_BSW, _s5b, BSW_LEVELS[0])
    _iw = _richest(META_WDR, _s5w, "W70")
    fig = plt.figure(figsize=FIG_DOUBLE)
    gs = fig.add_gridspec(2, 4, left=0.071, right=0.929, bottom=0.01, top=0.99, wspace=0.03, hspace=0.04)
    _tiles = [(next(m for m in META_BSW if m["index"] == _ib and m["condition"] == lv), 1, f"baseline {SWEEP_DESIGN['labels'][lv]}, view 2")
              for lv in BSW_LEVELS]
    _tiles += [(next(m for m in META_WDR if m["index"] == _iw and m["condition"] == lv), 0, f"{WALL_DOSE_DESIGN['labels'][lv]} wall"
                if lv != "G" else "no wall") for lv in WDR_LEVELS]
    for k, (m_, fk, lab) in enumerate(_tiles):
        a = fig.add_subplot(gs[k // 4, k % 4])
        a.imshow(load_rgb(m_["dir"], fk)); clean_image_axes(a, frame=False)
        tag(a, f"({'abcdefgh'[k]}) {lab}", fs=4.8)
    save_figure(fig, "figS5_designed_gallery", "The designed sets", "What do the baseline sweep and the dose-response look like?",
                "Section 3.2b", "double column", dict(sweep_scene=int(_ib), dose_scene=int(_iw)),
                caption=" ".join(cap("abcd"[k], f"Baseline {SWEEP_DESIGN['labels'][lv]}", "The second view of the same scene; the first "
                                     "view, s and eps = 0 are identical at every baseline.") for k, lv in enumerate(BSW_LEVELS))
                        + " " + " ".join(cap("efgh"[k], "No wall" if lv == "G" else WALL_DOSE_DESIGN["labels"][lv] + " wall",
                                             "The first view of the same dose-response scene." if lv == "G" else
                                             f"Static walls built to hide at least the lowest {WALL_DOSE_DESIGN['labels'][lv]} of every "
                                             "visible point of each mover in both views.")
                                         for k, lv in enumerate(WDR_LEVELS)))
release_memory()

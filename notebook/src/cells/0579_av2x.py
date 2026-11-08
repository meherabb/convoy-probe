# PLACEHOLDER
# 3.11b Argoverse 2, v6.4 (exploratory, AMENDMENT_V6_4, D49-D50).
#       (1) A second stage ("av2_gt_ext") takes further validation logs in the fixed order of Section 3.11,
#           after the logs the first stage tried, with every rule unchanged; the first stage is untouched,
#           so a session resumed from a v6.3 run restores it and downloads only the new logs.
#       (2) A third stage ("av2_contact") decides for every labelled vehicle of every pair whether its
#           ground contact is visible in the first image. Only the compact tables (poses, calibration,
#           labelled boxes) are read again; the LiDAR depth comes from the stored ground truth.
#       (3) The vehicle table of all logs, with motion regimes and contact labels, used by every
#           Argoverse 2 analysis.
AV2X_DIR = os.path.join(SCRATCH_DIR, "av2x")
AV2X_GT_DIR = os.path.join(AV2X_DIR, "pairs")
AV2C_DIR = os.path.join(SCRATCH_DIR, "av2c")
AV2X_PAIR_META, AV2X_A3, AV2X_VEHICLE_CHECK = [], {}, {}
AV2X_STATUS = dict(available=False, note="", tried=0, download={})
AV2C_STATUS = dict(available=False, note="", download={}, notes={})
AV2_CONTACT = pd.DataFrame()

# ---- (1) more logs -------------------------------------------------------------------------------------
AV2X_STAGE = Stage("av2_gt_ext", "av2x", dict(design={k: v for k, v in AV2_DESIGN.items() if k != "base_url"},
                                              n_logs=RUN_CONFIG["n_av2_logs_ext"], n_pairs=RUN_CONFIG["n_av2_pairs_per_log"],
                                              max_tried_total=AV2X_DESIGN["max_logs_tried_total"],
                                              prereg={k: v for k, v in PREREG.items() if k.startswith("kt_")}),
                   upstream=[AV2_STAGE.fp])
with stage("av2_gt_ext"):
    if not AV2_STATUS["available"]:
        AV2X_STATUS["note"] = "the first Argoverse 2 stage is not part of this run"
    elif RUN_CONFIG.get("n_av2_logs_ext", 0) <= 0:
        AV2X_STATUS["note"] = "switched off in RUN_CONFIG"
    elif not AV2X_STAGE.done():
        try:
            shutil.rmtree(AV2X_DIR, ignore_errors=True); os.makedirs(AV2X_GT_DIR)
            _first = json.load(open(os.path.join(AV2_DIR, "a3.json")))
            s3 = PublicS3(AV2_DESIGN["base_url"])
            log_ids, order = av2_list_logs(s3)
            _index0 = max(m_["index"] for m_ in AV2_PAIR_META) + 1
            accepted, rejected, AV2X_STATUS["tried"], _vc = av2_run_logs(
                s3, log_ids, order, int(_first["tried"]), set(_first["accepted"]) | set(_first["rejected"]),
                RUN_CONFIG["n_av2_logs_ext"], AV2X_DESIGN["max_logs_tried_total"], AV2X_GT_DIR, _index0, AV2X_A3)
            AV2X_STATUS["download"] = dict(requests=s3.n_requests, MiB=round(s3.n_bytes / 2**20, 1))
            with open(os.path.join(AV2X_DIR, "a3.json"), "w") as f:
                json.dump(dict(a3=AV2X_A3, vehicle_check=_vc, accepted=accepted, rejected=rejected, tried=AV2X_STATUS["tried"],
                               download=AV2X_STATUS["download"], n_logs_listed=len(log_ids), first_index=_index0),
                          f, indent=1, default=str)
            shutil.rmtree(AV2_RAW, ignore_errors=True)
            if not accepted:
                raise IOError(f"no further log passed (tried {AV2X_STATUS['tried']}: {list(rejected.values())[:3]})")
            if not _vc["passed"]:
                raise IOError("the moving-vehicle ground truth of the new logs is not confirmed by the images: " + _vc["note"])
            AV2X_STAGE.commit([AV2X_DIR])
        except Exception as e:
            AV2X_STATUS["note"] = f"{type(e).__name__}: {str(e)[:300]}"
            shutil.rmtree(AV2X_DIR, ignore_errors=True)

if os.path.exists(os.path.join(AV2X_DIR, "a3.json")):
    _rec = json.load(open(os.path.join(AV2X_DIR, "a3.json")))
    AV2X_A3, AV2X_VEHICLE_CHECK, AV2X_STATUS["download"] = _rec["a3"], _rec.get("vehicle_check", {}), _rec.get("download", {})
    for d in sorted(glob.glob(os.path.join(AV2X_GT_DIR, "*", "meta.json"))):
        m_ = json.load(open(d)); m_["dir"] = os.path.dirname(d)
        AV2X_PAIR_META.append(m_)
    AV2X_PAIR_META.sort(key=lambda m_: m_["index"])
    AV2X_STATUS.update(available=bool(AV2X_PAIR_META), logs=sorted({m_["log"] for m_ in AV2X_PAIR_META}))
if AV2X_PAIR_META:
    print(f"Argoverse 2, second stage (v6.4): {len(AV2X_PAIR_META)} pairs from {len(AV2X_STATUS['logs'])} further validation logs "
          f"(downloaded {AV2X_STATUS['download'].get('MiB', 0):,.0f} MiB in {AV2X_STATUS['download'].get('requests', 0)} requests)")
    print(av2_a3_table(AV2X_A3).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("moving-vehicle ground truth of the new logs against the images (pooled): "
          + AV2X_VEHICLE_CHECK.get("note", "–") + (", PASS" if AV2X_VEHICLE_CHECK.get("passed") else ", FAIL"))
elif AV2_STATUS["available"]:
    print("Argoverse 2, second stage: not part of this run (" + (AV2X_STATUS["note"] or "no data") + "); the analyses use the first stage")


# ---- (2) ground contact of every labelled vehicle (D50) ---------------------------------------------------
def _segment_blocked(C, X, T, half, gap):
    """True if the line of sight from the camera centre C to the point X crosses one of the boxes (poses T (n, 4, 4),
    half sizes half (n, 3)) at least `gap` metres in front of X (slab test in each box's frame)."""
    if not len(T):
        return False
    Rb, tb = T[:, :3, :3], T[:, :3, 3]
    o = np.einsum("nji,nj->ni", Rb, C[None] - tb)
    d = np.einsum("nji,j->ni", Rb, X - C)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o) / d, (half - o) / d
    inside = np.abs(o) <= half
    lo = np.where(d == 0, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    hi = np.where(d == 0, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    t_near, t_far = lo.max(1), hi.min(1)
    t_lim = 1.0 - gap / max(float(np.linalg.norm(X - C)), 1e-9)
    return bool(np.any((t_near <= t_far) & (t_far >= 0) & (t_near <= t_lim)))


def _static_blocked(D_static, u, v, z):
    # a static LiDAR return in the columns u +- c and the rows v - r ... v that is nearer than the point
    c, r = AV2X_DESIGN["lidar_window_cols"], AV2X_DESIGN["lidar_window_rows_above"]
    win = D_static[max(v - r, 0):v + 1, max(u - c, 0):u + c + 1]
    tol = max(AV2X_DESIGN["lidar_occluder_tol"][0] * z, AV2X_DESIGN["lidar_occluder_tol"][1])
    with np.errstate(invalid="ignore"):
        return bool(np.any(win < z - tol))


def wheel_contact_rule(C, R, cam, T, lwh, T_others, half_others, D_static, points=None):
    """D50 for one vehicle: box pose T (4 x 4; x along the length, y along the width, z up, origin at the box centre)
    and size lwh (length, width, height), camera centre C and camera-to-world rotation R, the boxes of every other
    labelled object (T_others, half sizes half_others, already shrunk) and the depth of static points (NaN elsewhere).
    Each in-view wheel contact point is hidden by the vehicle's own body, by another object, by static structure, or
    visible; the label follows the Virtual KITTI 2 rule (D37). If a list is passed as points, (u, v, status) of every
    in-view point is appended to it (v6.5, for the gallery of Fig. S4)."""
    H, W = D_static.shape
    l, w, h = (float(a) for a in lwh)
    hgt, gap, cl = PREREG["x6_wheel_height_m"], AV2X_DESIGN["occluder_gap_m"], AV2X_DESIGN["body_clearance_m"]
    c_obj = T[:3, :3].T @ (C - T[:3, 3])                # camera centre in the vehicle's frame
    T_body = T.copy(); T_body[:3, 3] = T[:3, 3] + T[:3, 2] * cl / 2       # the body: the box above the ground clearance
    half_body = np.array([[l / 2, w / 2, max(h / 2 - cl / 2, 0.01)]])
    out = dict(wheels_in_view=0, wheels_visible=0, hidden_by_body=0, hidden_by_object=0, hidden_by_static=0)
    for a in (-1, 1):
        for b in (-1, 1):
            X = T[:3, :3] @ np.array([a * 0.35 * l, b * 0.40 * w, -h / 2 + hgt]) + T[:3, 3]
            q = (X - C) @ R
            if q[2] <= 0.5:
                continue
            uv = cam.project(q)
            u_, v_ = int(round(float(uv[0]))), int(round(float(uv[1])))
            if not (0 <= u_ < W and 0 <= v_ < H):
                continue
            out["wheels_in_view"] += 1
            if not (b * c_obj[1] > w / 2 or a * c_obj[0] > l / 2) or _segment_blocked(C, X, T_body[None], half_body, gap):
                out["hidden_by_body"] += 1; status = "body"
            elif _segment_blocked(C, X, T_others, half_others, gap):
                out["hidden_by_object"] += 1; status = "object"
            elif _static_blocked(D_static, u_, v_, float(q[2])):
                out["hidden_by_static"] += 1; status = "static"
            else:
                out["wheels_visible"] += 1; status = "visible"
            if points is not None:
                points.append((float(uv[0]), float(uv[1]), status))
    n_in, n_vis = out["wheels_in_view"], out["wheels_visible"]
    out["contact"] = "visible" if n_vis >= 1 else "hidden" if n_in >= 2 else "undetermined"
    return out


def av2_contact_rows(lg, meta):
    """Wheel contact visibility (D50) of every labelled vehicle of one pair, in the pair's first image."""
    ti = float(lg.sec(int(meta["frames"][0])))
    z = np.load(os.path.join(meta["dir"], "gt.npz"))
    D_static = np.where(z["static"], z["depth"], np.nan).astype(np.float64)
    H, W = D_static.shape
    R, C = lg.cam_pose(ti)
    uids, Tb, half = [], [], []                          # every labelled object at the image's time
    for uid, tr in lg.tracks.items():
        if tr["t"][0] - 0.06 <= ti <= tr["t"][-1] + 0.06:
            uids.append(uid); Tb.append(lg.track_T(uid, [ti])[0])
            half.append(np.maximum(tr["lwh"] / 2 - AV2X_DESIGN["occluder_shrink_m"], 0.01))
    uids = np.asarray(uids, dtype=object)
    Tb = np.stack(Tb) if Tb else np.zeros((0, 4, 4)); half = np.stack(half) if half else np.zeros((0, 3))
    rows = []
    for k, v in enumerate(meta["vehicles"]):
        rec = dict(scene=int(meta["index"]), mover=k, track=v["track"], contact="undetermined", wheels_visible=np.nan,
                   wheels_in_view=np.nan, hidden_by_body=0, hidden_by_object=0, hidden_by_static=0, box_area_px=np.nan,
                   visible_area_px=np.nan)
        if v["track"] not in lg.tracks:
            rows.append(rec); continue
        T = lg.track_T(v["track"], [ti])[0]
        l, w, h = lg.tracks[v["track"]]["lwh"]
        others = uids != v["track"]
        rec.update(wheel_contact_rule(C, R, lg.cam, T, (l, w, h), Tb[others], half[others], D_static))
        corners = np.array([[a * l / 2, b * w / 2, c * h / 2] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        qc = (corners @ T[:3, :3].T + T[:3, 3] - C) @ R
        if np.all(qc[:, 2] > 0.1):
            uvc = lg.cam.project(qc)
            iw = max(0.0, min(uvc[:, 0].max(), W - 1) - max(uvc[:, 0].min(), 0.0))
            ih = max(0.0, min(uvc[:, 1].max(), H - 1) - max(uvc[:, 1].min(), 0.0))
            rec.update(box_area_px=float(iw * ih), visible_area_px=float(iw * ih * (1.0 - float(v.get("occlusion", 0.0)))))
        rows.append(rec)
    return rows


AV2_PAIRS_ALL = AV2_PAIR_META + AV2X_PAIR_META
AV2C_STAGE = Stage("av2_contact", "av2x", dict(rules={k: v for k, v in AV2X_DESIGN.items() if k != "max_logs_tried_total"},
                                               wheel_height=PREREG["x6_wheel_height_m"], wheel_position=[0.35, 0.40]),
                   upstream=[AV2_STAGE.fp] + ([AV2X_STAGE.fp] if AV2X_PAIR_META else []))
with stage("av2_contact"):
    if not AV2_PAIRS_ALL:
        AV2C_STATUS["note"] = "no Argoverse 2 pairs in this run"
    elif not AV2C_STAGE.done():
        try:
            shutil.rmtree(AV2C_DIR, ignore_errors=True); os.makedirs(AV2C_DIR)
            s3 = PublicS3(AV2_DESIGN["base_url"])
            _by_log, _rows = {}, []
            for m_ in AV2_PAIRS_ALL:
                _by_log.setdefault(m_["log"], []).append(m_)
            for log_id, metas in sorted(_by_log.items()):
                try:
                    lg = AV2Log(s3, log_id, raw_root=os.path.join(AV2C_DIR, "raw"))
                    for m_ in metas:
                        _rows += av2_contact_rows(lg, m_)
                except Exception as e:                  # the vehicles of an unreadable log keep no contact label
                    AV2C_STATUS["notes"][log_id] = f"{type(e).__name__}: {str(e)[:120]}"
                finally:
                    shutil.rmtree(os.path.join(AV2C_DIR, "raw"), ignore_errors=True)
            if not _rows:
                raise IOError("no contact label could be computed: " + "; ".join(list(AV2C_STATUS["notes"].values())[:3]))
            pd.DataFrame(_rows).to_csv(os.path.join(AV2C_DIR, "contact.csv"), index=False)
            AV2C_STATUS["download"] = dict(requests=s3.n_requests, MiB=round(s3.n_bytes / 2**20, 1))
            with open(os.path.join(AV2C_DIR, "contact.json"), "w") as f:
                json.dump(dict(download=AV2C_STATUS["download"], notes=AV2C_STATUS["notes"], n_logs=len(_by_log),
                               n_rows=len(_rows)), f, indent=1)
            AV2C_STAGE.commit([AV2C_DIR])
        except Exception as e:
            AV2C_STATUS["note"] = f"{type(e).__name__}: {str(e)[:300]}"
            shutil.rmtree(AV2C_DIR, ignore_errors=True)
if os.path.exists(os.path.join(AV2C_DIR, "contact.csv")):
    AV2_CONTACT = pd.read_csv(os.path.join(AV2C_DIR, "contact.csv"))
    _cj = json.load(open(os.path.join(AV2C_DIR, "contact.json")))
    AV2C_STATUS.update(available=len(AV2_CONTACT) > 0, download=_cj.get("download", {}), notes=_cj.get("notes", {}))

# ---- (3) the vehicle table of all logs ------------------------------------------------------------------
MOVERS_AV2_ALL = lidar_mover_table(AV2_PAIRS_ALL, "AV2") if AV2_PAIRS_ALL else pd.DataFrame()
if len(MOVERS_AV2_ALL):
    MOVERS_AV2_ALL, AV2_EPS_NOISE, AV2_TAU_PAR, _n_pk = lidar_regimes(MOVERS_AV2_ALL)
    if len(AV2_CONTACT):
        MOVERS_AV2_ALL = MOVERS_AV2_ALL.merge(
            AV2_CONTACT.rename(columns={"contact": "gt_contact", "wheels_visible": "gt_wheels_visible",
                                        "wheels_in_view": "gt_wheels_in_view", "visible_area_px": "gt_area_px",
                                        "box_area_px": "gt_box_area_px"}).drop(columns=["track"]),
            on=["scene", "mover"], how="left")
    MOVERS_AV2 = MOVERS_AV2_ALL[MOVERS_AV2_ALL["state"] == "moving"].reset_index(drop=True)
    AV2_STATUS["logs"] = sorted({m_["log"] for m_ in AV2_PAIRS_ALL})
    _speed = np.array([m_["baseline"] / max(m_["dt_s"], 1e-6) for m_ in AV2_PAIRS_ALL])
    AV2_EGO_SPEED = dict(median=float(np.median(_speed)), p90=float(np.quantile(_speed, 0.9)), max=float(np.max(_speed)))
    print(f"\nArgoverse 2, all stages: {len(AV2_PAIRS_ALL)} pairs from {len(AV2_STATUS['logs'])} validation logs; ego speed "
          f"(camera baseline / time between the images) median {AV2_EGO_SPEED['median']:.1f} m/s, 90th percentile "
          f"{AV2_EGO_SPEED['p90']:.1f} m/s, maximum {AV2_EGO_SPEED['max']:.1f} m/s")
    print(f"label-noise floor: 90th percentile of |eps| over {_n_pk} parked vehicles = {AV2_EPS_NOISE:.2f} px; "
          f"parallel regime on Argoverse 2: |eps| < {AV2_TAU_PAR:.2f} px")
    print(f"moving vehicles: {len(MOVERS_AV2)}; regimes {MOVERS_AV2['regime'].value_counts().to_dict()}; "
          f"measurable (>= {PREREG['kt_min_lidar_px']} LiDAR px) {int(MOVERS_AV2['measurable'].sum())}, identifiable "
          f"{int((MOVERS_AV2['measurable'] & MOVERS_AV2['identifiable']).sum())}; median time between a vehicle's "
          f"LiDAR returns and its image {MOVERS_AV2['gt_dt_ms'].median():.0f} ms (compensated along the track)")
    if "gt_contact" in MOVERS_AV2:
        _pv = MOVERS_AV2[MOVERS_AV2["measurable"].astype(bool) & (MOVERS_AV2["regime"] == "parallel")]
        print(f"ground contact (D50) of the parallel, measurable moving vehicles ({len(_pv)}): "
              f"{_pv['gt_contact'].fillna('no label').value_counts().to_dict()}; in-view wheel points hidden by the vehicle's own "
              f"body {int(_pv['hidden_by_body'].sum())}, by another labelled object {int(_pv['hidden_by_object'].sum())}, by "
              f"static structure (LiDAR) {int(_pv['hidden_by_static'].sum())}"
              + (f"; logs without contact labels: {len(AV2C_STATUS['notes'])}" if AV2C_STATUS["notes"] else ""))
    elif AV2C_STATUS["note"]:
        print("ground contact of Argoverse 2 vehicles is not part of this run:", AV2C_STATUS["note"])
else:
    MOVERS_AV2 = pd.DataFrame(); AV2_EPS_NOISE = AV2_TAU_PAR = np.nan; AV2_EGO_SPEED = {}


# ---- (4) check of the rule on Virtual KITTI 2 (D50): the same rule, with the exact depth of static pixels in place of
#      the LiDAR, applied to the Virtual KITTI 2 vehicles, whose instance masks give the reference labels (D37) --------
def vk_contact_rule_rows():
    rows, by_index = [], {m_["index"]: m_ for m_ in VK_PAIR_META}
    for seq, d in MOVERS_VK.groupby("seq", sort=False):
        sc = VK_SCENE_OBJS.get(seq)
        conv = sc.box_convention() if sc is not None else None
        if conv is None:
            continue
        frame_tids = {}
        for (f, tid) in sc.pose:
            frame_tids.setdefault(int(f), []).append(int(tid))
        for scene, g in d.groupby("scene", sort=True):
            f = int(by_index[int(scene)]["frames"][0])
            A = sc.frame_gt(f)
            D_static = np.where(A["inst"] < 0, A["depth"], np.nan).astype(np.float64)
            boxes = {}
            for tid in frame_tids.get(f, []):
                gb = sc._box_geometry(f, tid, conv)
                if gb is not None:
                    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = gb["e_len"], gb["e_wid"], -gb["down"], gb["c"]
                    boxes[tid] = (T, np.array([gb["l"], gb["w"], gb["h"]], float))
            for r in g.itertuples():
                if int(r.track) not in boxes:
                    continue
                T, lwh = boxes[int(r.track)]
                oth = [t for t in boxes if t != int(r.track)]
                T_o = np.stack([boxes[t][0] for t in oth]) if oth else np.zeros((0, 4, 4))
                h_o = (np.stack([np.maximum(boxes[t][1] / 2 - AV2X_DESIGN["occluder_shrink_m"], 0.01) for t in oth])
                       if oth else np.zeros((0, 3)))
                res = wheel_contact_rule(sc.C[f], sc.R[f], sc.cam, T, lwh, T_o, h_o, D_static)
                rows.append(dict(seq=seq, scene=int(scene), mover=int(r.mover), track=int(r.track), regime=r.regime,
                                 n_eroded=int(r.gt_n_eroded), mask_rule=str(r.gt_contact), geometry_rule=res["contact"],
                                 **{k: v for k, v in res.items() if k != "contact"}))
        sc._cache.clear()
    return pd.DataFrame(rows)


def contact_rule_agreement(df, sample):
    # agreement of the geometry rule with the instance-mask rule where both decide
    d = df[df["mask_rule"].isin(["visible", "hidden"]) & df["geometry_rule"].isin(["visible", "hidden"])]
    n = len(d)
    if not n:
        return dict(sample=sample, n=0)
    hm, hg = d["mask_rule"] == "hidden", d["geometry_rule"] == "hidden"
    po = float(np.mean(hm == hg))
    pe = float(hm.mean() * hg.mean() + (1 - hm.mean()) * (1 - hg.mean()))
    return dict(sample=sample, n=n, agreement=po, kappa=(po - pe) / (1 - pe) if pe < 1 else np.nan,
                hidden_mask_rule=int(hm.sum()), hidden_geometry_rule=int(hg.sum()), hidden_both=int((hm & hg).sum()),
                share_of_mask_hidden_found=float((hm & hg).sum() / hm.sum()) if hm.sum() else np.nan,
                share_of_rule_hidden_confirmed=float((hm & hg).sum() / hg.sum()) if hg.sum() else np.nan,
                undetermined_by_rule=int((df["mask_rule"].isin(["visible", "hidden"]) & (df["geometry_rule"] == "undetermined")).sum()))


VK_CONTACT_CHECK, CONTACT_RULE_CHECK = pd.DataFrame(), pd.DataFrame()
if VK_STATUS["available"] and len(MOVERS_VK) and "gt_contact" in MOVERS_VK:
    try:
        VK_CONTACT_CHECK = vk_contact_rule_rows()
    except Exception as e:                                  # a check, never a reason to stop the run
        print(f"check of the contact rule on Virtual KITTI 2 not computed: {type(e).__name__}: {str(e)[:200]}")
if len(VK_CONTACT_CHECK):
    _x6s = (VK_CONTACT_CHECK["regime"] == "parallel") & (VK_CONTACT_CHECK["n_eroded"] >= PREREG["x6_min_eroded_px"])
    CONTACT_RULE_CHECK = pd.DataFrame([contact_rule_agreement(VK_CONTACT_CHECK, "moving vehicles"),
                                       contact_rule_agreement(VK_CONTACT_CHECK[_x6s], "parallel, >= 40 eroded px (6.8b sample)")])
    print("\nCheck of the contact rule (D50) on Virtual KITTI 2: the same rule, with the exact depth of static pixels in place "
          "of the LiDAR,\nagainst the labels from the instance masks (D37), on the vehicles where both decide:")
    print(CONTACT_RULE_CHECK.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("rows: instance-mask rule; columns: geometry rule (moving vehicles):")
    print(pd.crosstab(VK_CONTACT_CHECK["mask_rule"], VK_CONTACT_CHECK["geometry_rule"]).to_string())
    VK_CONTACT_CHECK.to_csv(os.path.join(MANIFEST_DIR, "contact_rule_check_vkitti.csv"), index=False)
    CONTACT_RULE_CHECK.to_csv(os.path.join(RESULTS_DIR, "contact_rule_check.csv"), index=False)
elif VK_STATUS["available"]:
    print("check of the contact rule on Virtual KITTI 2: no vehicle with a 3-D box in this run")
release_memory(); mem_report("Argoverse 2 extension and ground contact");

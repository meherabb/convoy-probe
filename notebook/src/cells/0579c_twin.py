# PLACEHOLDER
# 3.11c The static twin of Argoverse 2 vehicles (v6.5, exploratory, D58). One stage ("av2_twin") reads the compact tables
#       of every log again and stores, for every labelled vehicle of every pair (moving and parked), the camera height
#       above its contact plane (the bottom face of its labelled box, along the box's vertical axis), the box size and
#       the image positions and visibility status of its wheel contact points (Section 3.11b rule).
AV2T_DIR = os.path.join(SCRATCH_DIR, "av2t")
AV2T_STATUS = dict(available=False, note="", download={}, notes={})
AV2_TWIN = pd.DataFrame()


def av2_twin_rows(lg, meta):
    ti = float(lg.sec(int(meta["frames"][0])))
    R, C = lg.cam_pose(ti)
    z = np.load(os.path.join(meta["dir"], "gt.npz"))
    D_static = np.where(z["static"], z["depth"], np.nan).astype(np.float64)
    uids, Tb, half = [], [], []
    for uid, tr in lg.tracks.items():
        if tr["t"][0] - 0.06 <= ti <= tr["t"][-1] + 0.06:
            uids.append(uid); Tb.append(lg.track_T(uid, [ti])[0])
            half.append(np.maximum(tr["lwh"] / 2 - AV2X_DESIGN["occluder_shrink_m"], 0.01))
    uids = np.asarray(uids, dtype=object)
    Tb = np.stack(Tb) if Tb else np.zeros((0, 4, 4)); half = np.stack(half) if half else np.zeros((0, 3))
    rows = []
    for k, v in enumerate(meta["vehicles"]):
        rec = dict(scene=int(meta["index"]), mover=k, track=v["track"], H_cam=np.nan, box_l=np.nan, box_w=np.nan, box_h=np.nan,
                   bottom_u=np.nan, bottom_v=np.nan, up_tilt_deg=np.nan, wheel_points="[]")
        if v["track"] not in lg.tracks:
            rows.append(rec); continue
        T = lg.track_T(v["track"], [ti])[0]
        l, w, h = (float(a) for a in lg.tracks[v["track"]]["lwh"])
        n_up = T[:3, 2] / np.linalg.norm(T[:3, 2])           # the box's vertical axis (up, normal to the road under it)
        X_b = T[:3, 3] - n_up * h / 2                         # centre of the bottom face: the vehicle's contact plane
        q = (X_b - C) @ R
        uv = lg.cam.project(q) if q[2] > 0.5 else np.array([np.nan, np.nan])
        pts = []
        others = uids != v["track"]
        wheel_contact_rule(C, R, lg.cam, T, (l, w, h), Tb[others], half[others], D_static, points=pts)
        rec.update(H_cam=float((C - X_b) @ n_up), box_l=l, box_w=w, box_h=h, bottom_u=float(uv[0]), bottom_v=float(uv[1]),
                   up_tilt_deg=float(np.degrees(np.arccos(np.clip(abs(n_up[2]), -1.0, 1.0)))), wheel_points=json.dumps(pts))
        rows.append(rec)
    return rows


AV2T_STAGE = Stage("av2_twin", "av2t", dict(what="camera height above each labelled vehicle's box bottom; wheel points",
                                            wheel_height=PREREG["x6_wheel_height_m"], wheel_position=[0.35, 0.40],
                                            rules={k: v for k, v in AV2X_DESIGN.items() if k != "max_logs_tried_total"}),
                   upstream=[AV2_STAGE.fp] + ([AV2X_STAGE.fp] if AV2X_PAIR_META else []))
with stage("av2_twin"):
    if not AV2_PAIRS_ALL:
        AV2T_STATUS["note"] = "no Argoverse 2 pairs in this run"
    elif not AV2T_STAGE.done():
        try:
            shutil.rmtree(AV2T_DIR, ignore_errors=True); os.makedirs(AV2T_DIR)
            s3 = PublicS3(AV2_DESIGN["base_url"])
            _by_log, _rows = {}, []
            for m_ in AV2_PAIRS_ALL:
                _by_log.setdefault(m_["log"], []).append(m_)
            for log_id, metas in sorted(_by_log.items()):
                try:
                    lg = AV2Log(s3, log_id, raw_root=os.path.join(AV2T_DIR, "raw"))
                    for m_ in metas:
                        _rows += av2_twin_rows(lg, m_)
                except Exception as e:                      # the vehicles of an unreadable log keep no camera height
                    AV2T_STATUS["notes"][log_id] = f"{type(e).__name__}: {str(e)[:120]}"
                finally:
                    shutil.rmtree(os.path.join(AV2T_DIR, "raw"), ignore_errors=True)
            if not _rows:
                raise IOError("no camera height could be computed: " + "; ".join(list(AV2T_STATUS["notes"].values())[:3]))
            pd.DataFrame(_rows).to_csv(os.path.join(AV2T_DIR, "twin.csv"), index=False)
            AV2T_STATUS["download"] = dict(requests=s3.n_requests, MiB=round(s3.n_bytes / 2**20, 1))
            with open(os.path.join(AV2T_DIR, "twin.json"), "w") as f:
                json.dump(dict(download=AV2T_STATUS["download"], notes=AV2T_STATUS["notes"], n_logs=len(_by_log), n_rows=len(_rows)),
                          f, indent=1)
            AV2T_STAGE.commit([AV2T_DIR])
        except Exception as e:
            AV2T_STATUS["note"] = f"{type(e).__name__}: {str(e)[:300]}"
            shutil.rmtree(AV2T_DIR, ignore_errors=True)
if os.path.exists(os.path.join(AV2T_DIR, "twin.csv")):
    AV2_TWIN = pd.read_csv(os.path.join(AV2T_DIR, "twin.csv"))
    _tj = json.load(open(os.path.join(AV2T_DIR, "twin.json")))
    AV2T_STATUS.update(available=len(AV2_TWIN) > 0, download=_tj.get("download", {}), notes=_tj.get("notes", {}))
if len(AV2_TWIN) and len(MOVERS_AV2_ALL):
    _tw = MOVERS_AV2_ALL.merge(AV2_TWIN.drop(columns=["track"]), on=["scene", "mover"], how="left")
    _tw = _tw[_tw["measurable"].astype(bool)]
    print(f"Argoverse 2 camera heights above the vehicles' contact planes: {int(_tw['H_cam'].notna().sum())} measurable vehicles "
          f"(moving {int((_tw['state'] == 'moving').sum())}, parked {int((_tw['state'] == 'parked').sum())}); median H = "
          f"{_tw['H_cam'].median():.2f} m (5-95%: {_tw['H_cam'].quantile(0.05):.2f}-{_tw['H_cam'].quantile(0.95):.2f} m); box axis "
          f"tilt from the city vertical: median {_tw['up_tilt_deg'].median():.1f} deg")
    if AV2T_STATUS["notes"]:
        print("logs without camera heights:", len(AV2T_STATUS["notes"]))
else:
    print("The static twin of Argoverse 2 vehicles is not part of this run:", AV2T_STATUS["note"] or "no Argoverse 2 data")
release_memory(); mem_report("Argoverse 2 static twin")

# PLACEHOLDER
# 3.11d The twin test with optical flow (v6.6, D63, H14). One stage ("av2_flowtwin") computes optical flow in both
#       directions for every Argoverse 2 pair, fits the road plane to all LiDAR returns of the first image (RANSAC over
#       horizontal planes, city z up; no labels) and stores, for every labelled vehicle (moving and parked), its epipolar
#       offset, static-hypothesis depth and the heights of its static reconstruction above the road, all from the flow at
#       its LiDAR pixels (forward-backward checked). The labels only say which pixels belong to a vehicle and whether it
#       moves; the correspondences, the plane and the camera height are what a deployed system would have.
AV2F_DIR = os.path.join(SCRATCH_DIR, "av2ft")
AV2F_STATUS = dict(available=False, note="", flow="", n_pairs=0, roads={})
AV2_FLOWTWIN = pd.DataFrame()


def av2_road_plane(depth, cam, R, C, rng):
    """Road plane of one image from all its LiDAR returns: the horizontal plane (normal within road_max_tilt_deg of the
    city z axis) with the most returns within road_inlier_m, sampled from returns below the camera, refined on its
    inliers; None if no such plane has a plausible camera height."""
    D = FLOWTWIN_DESIGN
    vv, uu = np.nonzero(np.isfinite(depth) & (depth >= D["road_depth_m"][0]) & (depth <= D["road_depth_m"][1]))
    if len(vv) < 200:
        return None
    X = C + (cam.rays(uu.astype(np.float64), vv.astype(np.float64)) * depth[vv, uu].astype(np.float64)[:, None]) @ R.T
    low = np.nonzero(X[:, 2] < C[2] - 0.5)[0]
    if len(low) < 100:
        return None
    cos_max = np.cos(np.radians(D["road_max_tilt_deg"]))
    best, n_best = None, 0
    for _ in range(D["ransac_iters"]):
        a = X[rng.choice(low, 3, replace=False)]
        n = np.cross(a[1] - a[0], a[2] - a[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn * (1.0 if n[2] >= 0 else -1.0)
        if n[2] < cos_max or not (D["road_cam_height_m"][0] <= (C - a[0]) @ n <= D["road_cam_height_m"][1]):
            continue
        inl = np.abs((X - a[0]) @ n) < D["road_inlier_m"]
        if inl.sum() > n_best:
            best, n_best = inl, int(inl.sum())
    if best is None or n_best < 100:
        return None
    Q = X[best]; o = Q.mean(0)
    n = np.linalg.svd(Q - o, full_matrices=False)[2][-1]
    n = n if n[2] >= 0 else -n
    h_cam, tilt = float((C - o) @ n), float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
    if not (D["road_cam_height_m"][0] <= h_cam <= D["road_cam_height_m"][1]) or tilt > D["road_max_tilt_deg"]:
        return None
    return dict(n=n, o=o, cam_height=h_cam, tilt_deg=tilt, inlier_frac=float(n_best / len(X)), n_points=int(len(X)))


def av2_flow_vehicles(meta, f_ij, f_ji, depth, plane):
    """Flow-based quantities of every labelled vehicle of one pair (one row per vehicle, in the order of meta)."""
    D = FLOWTWIN_DESIGN
    cam, R0, C0 = frame_camera(meta["scene"], 0)
    _, R1, C1 = frame_camera(meta["scene"], 1)
    R21, t21 = relative_pose(R0, C0, R1, C1)
    F = fundamental_matrix(cam.K, cam.K, R21, t21)
    rows = []
    for k, v in enumerate(meta["vehicles"]):
        rec = dict(scene=int(meta["index"]), mover=k, n_lidar_px=0, n_flow_px=0, frac_fb_ok=np.nan, eps_flow=np.nan,
                   x_flow=np.nan, frac_behind=np.nan, h_low_flow=np.nan, h_low_true=np.nan, road=plane is not None)
        px = np.asarray(v.get("pixels", []), np.int64).reshape(-1, 2)
        rec["n_lidar_px"] = int(len(px))
        if len(px) < D["min_px"]:
            rows.append(rec); continue
        Z = depth[px[:, 1], px[:, 0]].astype(np.float64)
        p = px.astype(np.float64)
        f = f_ij[px[:, 1], px[:, 0]].astype(np.float64)
        q = p + f
        with np.errstate(invalid="ignore"):
            fb = np.linalg.norm(f + sample_map(f_ji, q), axis=1)
            ok = np.isfinite(Z) & np.isfinite(fb) & (fb < D["fb_px"] + D["fb_rel"] * np.linalg.norm(f, axis=1))
        rec.update(frac_fb_ok=float(ok.mean()), n_flow_px=int(ok.sum()))
        if ok.sum() < D["min_px"]:
            rows.append(rec); continue
        p, q, Z = p[ok], q[ok], Z[ok]
        zt, _ = triangulate_depth(p, q, cam.K, cam.K, R21, t21)
        pos = np.isfinite(zt) & (zt > 0)
        rec.update(eps_flow=float(np.median(np.abs(epipolar_offset(p, q, F)))), frac_behind=float(1.0 - pos.mean()))
        if rec["frac_behind"] <= D["behind_frac"]:
            rec["x_flow"] = float(np.median(np.log(zt[pos] / Z[pos])))
            if plane is not None:
                Xs = C0 + (cam.rays(p[pos, 0], p[pos, 1]) * zt[pos][:, None]) @ R0.T     # the static reconstruction
                Xt = C0 + (cam.rays(p[:, 0], p[:, 1]) * Z[:, None]) @ R0.T             # the LiDAR returns
                rec["h_low_flow"] = float(np.percentile((Xs - plane["o"]) @ plane["n"], D["low_pct"]))
                rec["h_low_true"] = float(np.percentile((Xt - plane["o"]) @ plane["n"], D["low_pct"]))
        rows.append(rec)
    return rows


AV2F_STAGE = Stage("av2_flowtwin", "av2ft", dict(design=_jsonable(FLOWTWIN_DESIGN), flow=FLOW_PREFERRED),
                   upstream=[AV2_STAGE.fp] + ([AV2X_STAGE.fp] if AV2X_PAIR_META else []))
with stage("av2_flowtwin"):
    if not AV2_PAIRS_ALL:
        AV2F_STATUS["note"] = "no Argoverse 2 pairs in this run"
    elif not AV2F_STAGE.done():
        try:
            shutil.rmtree(AV2F_DIR, ignore_errors=True); os.makedirs(AV2F_DIR)
            if FLOW.get("model") is None and str(FLOW.get("method") or "").startswith("RAFT"):
                FLOW["method"] = None                        # RAFT was released after an earlier stage: load it again
            _tmp = os.path.join(FLOW_DIR, "av2ft_tmp")
            _rows, _roads = [], []
            for _s in range(0, len(AV2_PAIRS_ALL), 16):     # flows are cached on disk one chunk at a time
                _chunk = AV2_PAIRS_ALL[_s:_s + 16]
                _ims = [tuple(np.asarray(Image.open(os.path.join(m_["dir"], f"rgb{k_}.png")).convert("RGB")) for k_ in (0, 1))
                        for m_ in _chunk]
                _tags = [os.path.basename(m_["dir"]) for m_ in _chunk]
                _fw = compute_flows(_ims, [os.path.join(_tmp, f"{t_}_f.npz") for t_ in _tags])
                _bw = compute_flows([(b_, a_) for a_, b_ in _ims], [os.path.join(_tmp, f"{t_}_b.npz") for t_ in _tags])
                shutil.rmtree(_tmp, ignore_errors=True)
                for m_, f1, f2 in zip(_chunk, _fw, _bw):
                    _dep = np.load(os.path.join(m_["dir"], "gt.npz"))["depth"]
                    _cam, _R0, _C0 = frame_camera(m_["scene"], 0)
                    _pl = av2_road_plane(_dep, _cam, _R0, _C0, np.random.default_rng(seed_for("av2-road", m_["index"])))
                    _roads.append(dict(scene=int(m_["index"]), log=m_["log"], found=_pl is not None,
                                       cam_height=_pl["cam_height"] if _pl else np.nan, tilt_deg=_pl["tilt_deg"] if _pl else np.nan,
                                       inlier_frac=_pl["inlier_frac"] if _pl else np.nan))
                    _rows += av2_flow_vehicles(m_, f1, f2, _dep, _pl)
                del _ims, _fw, _bw
            FLOW["model"] = None
            release_memory()
            if not _rows:
                raise IOError("no vehicle with flow")
            pd.DataFrame(_rows).to_csv(os.path.join(AV2F_DIR, "flowtwin.csv"), index=False)
            pd.DataFrame(_roads).to_csv(os.path.join(AV2F_DIR, "roads.csv"), index=False)
            with open(os.path.join(AV2F_DIR, "flowtwin.json"), "w") as f:
                json.dump(dict(flow=FLOW.get("method"), fallback_reason=FLOW.get("fallback_reason"), n_pairs=len(AV2_PAIRS_ALL),
                               n_rows=len(_rows)), f, indent=1)
            AV2F_STAGE.commit([AV2F_DIR])
        except Exception as e:
            AV2F_STATUS["note"] = f"{type(e).__name__}: {str(e)[:300]}"
            shutil.rmtree(AV2F_DIR, ignore_errors=True)
            FLOW["model"] = None
if os.path.exists(os.path.join(AV2F_DIR, "flowtwin.csv")):
    AV2_FLOWTWIN = pd.read_csv(os.path.join(AV2F_DIR, "flowtwin.csv"))
    _fj = json.load(open(os.path.join(AV2F_DIR, "flowtwin.json")))
    _rd = pd.read_csv(os.path.join(AV2F_DIR, "roads.csv"))
    AV2F_STATUS.update(available=len(AV2_FLOWTWIN) > 0, flow=_fj.get("flow") or "", n_pairs=_fj.get("n_pairs", 0),
                       roads=dict(found_share=float(_rd["found"].mean()), cam_height_median=float(_rd["cam_height"].median()),
                                  tilt_median_deg=float(_rd["tilt_deg"].median())))
if len(AV2_FLOWTWIN) and len(MOVERS_AV2_ALL):
    _fw_ = MOVERS_AV2_ALL[["scene", "mover", "state", "measurable"]].merge(AV2_FLOWTWIN, on=["scene", "mover"], how="inner")
    _fw_ = _fw_[_fw_["measurable"].astype(bool)]
    _pkf = _fw_[_fw_["state"] == "parked"]
    print(f"Twin test with optical flow ({AV2F_STATUS['flow']}): {AV2F_STATUS['n_pairs']} Argoverse 2 pairs; road plane found in "
          f"{100 * AV2F_STATUS['roads']['found_share']:.0f}% of them (camera {AV2F_STATUS['roads']['cam_height_median']:.2f} m above "
          f"it, tilt {AV2F_STATUS['roads']['tilt_median_deg']:.1f} deg, medians)")
    print(f"measurable vehicles: {len(_fw_)}; with >= {FLOWTWIN_DESIGN['min_px']} forward-backward consistent LiDAR pixels: "
          f"{int((_fw_['n_flow_px'] >= FLOWTWIN_DESIGN['min_px']).sum())} (median share kept {_fw_['frac_fb_ok'].median():.2f}); "
          f"parked vehicles: median |x_flow| {np.nanmedian(np.abs(_pkf['x_flow'])):.3f}, median |eps_flow| "
          f"{np.nanmedian(_pkf['eps_flow']):.2f} px, lowest returns {np.nanmedian(_pkf['h_low_true']):.2f} m above the road "
          f"(static reconstruction from flow: {np.nanmedian(_pkf['h_low_flow']):.2f} m)")
else:
    print("The twin test with optical flow is not part of this run:", AV2F_STATUS["note"] or "no Argoverse 2 data")
release_memory(); mem_report("Argoverse 2 twin test with flow")

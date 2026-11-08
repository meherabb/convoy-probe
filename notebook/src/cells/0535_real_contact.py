# PLACEHOLDER
# 3.5.1 E5-REAL (v6.1): the ground-contact ablation on real images. In real TUM frame pairs (static
#     sequences, motion-capture poses, measured depth) the same textured crate is inserted twice:
#       G  resting on the support plane (floor or table top) with a soft contact shadow,
#       F  lifted 0.25-0.50 m off the plane, without a shadow;
#     everything else (pair, crate, texture, s, motion) is identical. The support plane is fitted
#     to the measured depth of view 1 (RANSAC). It must be horizontal: its normal within 12 deg of the
#     vertical of the motion-capture frame when the trajectory identifies one (cameras held upright
#     along one world axis), otherwise within 35 deg of the camera's up direction; the camera must be
#     at least 0.3 m above it. On-plane tests allow for the depth noise of the sensor at range z;
#     the crate's footprint must lie on that plane in the real depth, the crate may neither intersect
#     nor be hidden by real surfaces, and the motion is purely parallel (delta = s b) on pairs whose
#     baseline is within 6 deg of the plane, so the grounded crate stays on the plane in both views.
#     Each composite is rendered in a frame aligned with the plane (y down, plane at y = 0), so the
#     crate stands on one of its faces; depth, correspondences and every model input are unaffected
#     by this choice of world frame. One checkpointed stage (Section 0.5).
RABL_STAGE = TUM_STAGES["RABL"]

def _ray_box_t(Cc, Dd, center, half, yaw):
    # depth of the first intersection of camera rays (origin Cc, directions Dd with unit camera z)
    # with an oriented box (np.inf where the ray misses)
    Rb = rot_y(yaw)
    o = (Cc - np.asarray(center)) @ Rb
    d = Dd @ Rb
    h = np.asarray(half)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-h - o) / d; t2 = (h - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=-1); tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit = (tmax >= np.maximum(tmin, 0)) & np.isfinite(tmin)
    return np.where(hit, np.maximum(tmin, 0), np.inf)

def sequence_vertical(seq):
    # the world axis along which the cameras are held upright (motion-capture frames are levelled
    # to the floor); None if no axis is consistent (mean cosine < 0.6), then the camera's up is used
    ups = np.stack([-seq.pose(k)[0][:, 1] for k in range(0, len(seq), 5)])
    m = ups.mean(0)
    a = int(np.argmax(np.abs(m)))
    if abs(m[a]) < 0.6:
        return None
    g = np.zeros(3); g[a] = np.sign(m[a])
    return g

def kinect_sigma(z):
    # axial depth noise of a structured-light RGB-D sensor (Nguyen, Izadi and Lovell, 3DIMPVT 2012)
    return 0.0012 + 0.0019 * (np.asarray(z, np.float64) - 0.4) ** 2

def _plane_tol(z):
    return np.maximum(0.02, RABL_DESIGN["on_plane_sigma_k"] * kinect_sigma(z))

def fit_support_plane(seq, i, rng, vertical=None):
    D = RABL_DESIGN
    d = seq.depth(i); cam = seq.cam; R, C = seq.pose(i)
    H, W = d.shape
    vv, uu = np.mgrid[0:H:4, 0:W:4]
    z = d[vv, uu]
    ok = np.isfinite(z) & (z > 0.3) & (z < D["plane_fit_max_depth_m"])
    if ok.sum() < 400:
        return None
    Xw = C + (cam.rays(uu[ok].astype(np.float64), vv[ok].astype(np.float64)) * z[ok][:, None]) @ R.T
    up = vertical if vertical is not None else -R[:, 1]   # motion-capture vertical, else the image's up
    max_tilt = D["vertical_max_tilt_deg"] if vertical is not None else D["plane_max_tilt_deg"]
    best = None
    for _ in range(D["plane_ransac_iters"]):
        p = Xw[rng.choice(len(Xw), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        n = n if n @ up >= 0 else -n
        if np.degrees(np.arccos(np.clip(n @ up, -1, 1))) > max_tilt or (C - p[0]) @ n < 0.3:
            continue                                     # not horizontal, or not below the camera (a ceiling)
        inl = np.abs((Xw - p[0]) @ n) < D["plane_inlier_m"]
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < D["plane_min_inlier_frac"] * len(Xw):
        return None
    P = Xw[best]; c0 = P.mean(0)
    n = np.linalg.svd(P - c0, full_matrices=False)[2][-1]
    n = n if n @ up >= 0 else -n
    h = float((C - c0) @ n)
    if h < 0.3:
        return None
    tilt_v = float(np.degrees(np.arccos(np.clip(n @ vertical, -1, 1)))) if vertical is not None else np.nan
    return dict(n=n, origin=c0, height=h, inlier_frac=float(best.mean()), tilt_to_vertical_deg=tilt_v,
                reference="motion-capture vertical" if vertical is not None else "camera up")

def _plane_frame(n, R_cam):
    # rotation A (rows) from the world frame to a frame with y = -n (down) and x along the camera's right
    e_y = -n
    x0 = R_cam[:, 0] - (R_cam[:, 0] @ e_y) * e_y
    e_x = x0 / np.linalg.norm(x0)
    return np.stack([e_x, e_y, np.cross(e_x, e_y)])

def _on_plane(depth, cam, R, C):
    # mask of real pixels whose measured 3-D point lies on the plane y = 0 (plane frame), within the
    # sensor's noise at that range
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    fin = np.isfinite(depth)
    Y = np.full((H, W), np.nan)
    D_ = cam.rays(uu[fin].astype(np.float64), vv[fin].astype(np.float64)) @ R.T
    Y[fin] = (C + D_ * depth[fin][:, None])[:, 1]
    tol = np.full((H, W), np.nan); tol[fin] = _plane_tol(depth[fin])
    with np.errstate(invalid="ignore"):
        return np.abs(Y) < tol, Y

def _crate_ok(center, half, yaw, k_off, cams_L, depths, cam, margin):
    # bounding box per view; rejects crates outside the image, colliding with or hidden by real surfaces
    bbs = []
    for k, (R_, C_) in enumerate(cams_L):
        c = np.asarray(center) + k_off[k]
        corners = box_corners(dict(center=c, half=half, yaw=yaw))
        Xc = (corners - C_) @ R_
        if np.any(Xc[:, 2] < 0.3):
            return None
        uv = cam.project(Xc)
        u0, v0 = uv.min(0); u1, v1 = uv.max(0)
        if u0 < margin or v0 < margin or u1 > cam.W - 1 - margin or v1 > cam.H - 1 - margin or \
           (k == 0 and (u1 - u0) * (v1 - v0) < 600):
            return None
        vv, uu = np.mgrid[int(v0):int(v1) + 1, int(u0):int(u1) + 1]
        Dd = cam.rays(uu.ravel().astype(np.float64), vv.ravel().astype(np.float64)) @ R_.T
        t = _ray_box_t(C_, Dd, c, half, yaw)
        hit = np.isfinite(t)
        real = depths[k][vv.ravel(), uu.ravel()]
        if hit.sum() < 300 or np.mean(np.isfinite(real[hit])) < 0.7:
            return None
        rh, th = real[hit], t[hit]
        with np.errstate(invalid="ignore"):
            Py = C_[1] + Dd[hit][:, 1] * rh                     # plane-frame height of the real point
            front = (rh < th - np.maximum(0.03, _plane_tol(rh))) & ~(np.abs(Py) < _plane_tol(rh))
        if np.mean(front) > 0.02:                                # a real surface in front of (or inside) the crate;
            return None                                          # the support plane itself cannot hide it
        bbs.append((u0, v0, u1, v1))
    return bbs

def _footprint_on_plane(center, half, yaw, R_, C_, cam, depth):
    # the real depth under the crate's footprint (its corners and edge midpoints) lies on the plane
    sg = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)], np.float64)
    pts = np.column_stack([sg[:, 0] * half[0], np.zeros(len(sg)), sg[:, 1] * half[2]]) @ rot_y(yaw).T
    pts = pts + np.array([center[0], 0.0, center[2]])
    Xc = (pts - C_) @ R_
    if np.any(Xc[:, 2] < 0.3):
        return False
    uv = np.round(cam.project(Xc)).astype(int)
    if np.any((uv[:, 0] < 0) | (uv[:, 0] >= cam.W) | (uv[:, 1] < 0) | (uv[:, 1] >= cam.H)):
        return False
    z = depth[uv[:, 1], uv[:, 0]]
    if not np.all(np.isfinite(z)):
        return False
    P = C_ + (cam.rays(uv[:, 0].astype(np.float64), uv[:, 1].astype(np.float64)) * z[:, None]) @ R_.T
    return bool(np.all(np.abs(P[:, 1]) < _plane_tol(z)))

def _contact_shadow(rgb, depth, cam, R_, C_, movers_k, onplane):
    # soft darkening of the real support-plane pixels around each crate's footprint (view k)
    D = RABL_DESIGN
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    fin = onplane & np.isfinite(depth)
    w = np.zeros((H, W), np.float32)
    if fin.any():
        P = C_ + (cam.rays(uu[fin].astype(np.float64), vv[fin].astype(np.float64)) * depth[fin][:, None]) @ R_.T
        for c, half, yaw in movers_k:
            q = (P - np.asarray(c)) @ rot_y(yaw)
            r = np.maximum(np.abs(q[:, 0]) / half[0], np.abs(q[:, 2]) / half[2])
            wk = np.clip((1 + D["shadow_softness"] - r) / D["shadow_softness"], 0, 1) ** 1.5
            w[fin] = np.maximum(w[fin], wk)
    out = rgb.astype(np.float32) * (1 - D["shadow_strength"] * w[..., None])
    return out, w

def build_rabl_pair(seq, i, j, n_items, seed, vertical=None):
    # -> ((scenes {cond: scene}, renders {cond: [r0, r1]}, plane), None) or (None, reason)
    D = RABL_DESIGN
    rng = np.random.default_rng(seed)
    cam = seq.cam
    Ri, Ci = seq.pose(i); Rj, Cj = seq.pose(j)
    plane = fit_support_plane(seq, i, rng, vertical)
    if plane is None:
        return None, "no support plane"
    b = Cj - Ci
    tilt = np.degrees(np.arcsin(min(1.0, abs(plane["n"] @ b) / max(np.linalg.norm(b), 1e-9))))
    if tilt > D["max_baseline_tilt_deg"]:
        return None, "baseline not parallel to the plane"
    A = _plane_frame(plane["n"], Ri); O = plane["origin"]
    cams_L = [(A @ Ri, A @ (Ci - O)), (A @ Rj, A @ (Cj - O))]
    bL = cams_L[1][1] - cams_L[0][1]
    di, dj = seq.depth(i), seq.depth(j)
    onp0, _ = _on_plane(di, cam, *cams_L[0])
    cand = np.argwhere(onp0 & (di >= D["depth_m"][0]) & (di <= D["depth_m"][1]))
    m = D["margin_px"]
    cand = cand[(cand[:, 0] > m) & (cand[:, 0] < cam.H - m) & (cand[:, 1] > m) & (cand[:, 1] < cam.W - m)]
    if len(cand) == 0:
        return None, "no free plane in view"
    placed, bbs_all = [], []
    for item in range(n_items):
        for attempt in range(D["attempts"]):
            v_, u_ = cand[int(rng.integers(len(cand)))]
            P = cams_L[0][1] + cams_L[0][0] @ (cam.rays(float(u_), float(v_)) * float(di[v_, u_]))
            a_ = rng.uniform(*D["crate_size_m"])
            half = np.array([a_ / 2, a_ / 2 * rng.uniform(0.8, 1.0), a_ / 2 * rng.uniform(0.8, 1.2)])
            yaw = rng.uniform(0, np.pi / 2)
            s = rng.uniform(*MOVER_DESIGN["s_range_main"])
            delta = s * bL
            lift = rng.uniform(*D["lift_m"])
            cG = np.array([P[0], -half[1], P[2]]); cF = cG + np.array([0.0, -lift, 0.0])
            if not _footprint_on_plane(cG, half, yaw, *cams_L[0], cam, di):
                continue
            offs = [np.zeros(3), delta]
            bG = _crate_ok(cG, half, yaw, offs, cams_L, (di, dj), cam, 8)
            bF = _crate_ok(cF, half, yaw, offs, cams_L, (di, dj), cam, 8) if bG else None
            if not bG or not bF:
                continue
            boxes_all = bG + bF
            clash = any(not (u1 + 8 < ou0 or ou1 + 8 < u0 or v1 + 8 < ov0 or ov1 + 8 < v0)
                        for (u0, v0, u1, v1) in boxes_all for ob in bbs_all for (ou0, ov0, ou1, ov1) in ob)
            if clash:
                continue
            style = ["wood", "container", "pattern"][int(rng.integers(3))]
            tint = CONTAINER_TINTS[int(rng.integers(len(CONTAINER_TINTS)))] if style == "container" else (1, 1, 1)
            placed.append(dict(cG=cG, cF=cF, half=half, yaw=yaw, s=s, delta=delta, lift=lift, style=style, tint=tint,
                               bbG=bG, bbF=bF))
            bbs_all.append(boxes_all)
            break
    if not placed:
        return None, "no crate fits"
    scenes, renders = {}, {}
    R21, t21 = relative_pose(*cams_L[0], *cams_L[1])
    F_ = fundamental_matrix(cam.K, cam.K, R21, t21)
    for cond in RABL_DESIGN["conditions"]:
        sc = dict(seed=int(seed), street=_DUMMY_STREET, boxes=[], shadows=[], movers=[], condition=cond,
                  cams=[dict(**cam.to_dict(), R=Rk.tolist(), C=Ck.tolist()) for Rk, Ck in cams_L],
                  camera=dict(kind="tum", path="straight", focal=0.5 * (cam.fx + cam.fy), baseline=float(np.linalg.norm(b)),
                              psi_deg=0.0, omega_deg=0.0),
                  light=dict(exposure=1.0, white_balance=[1.0, 1.0, 1.0], noise_sigma=1.2 / 255),
                  plane=dict(normal_world=plane["n"].tolist(), origin_world=O.tolist(), camera_height_m=plane["height"],
                             inlier_frac=plane["inlier_frac"], baseline_tilt_deg=float(tilt), reference=plane["reference"],
                             tilt_to_vertical_deg=plane["tilt_to_vertical_deg"]))
        for p in placed:
            c = p["cG"] if cond == "G" else p["cF"]
            idx = len(sc["movers"])
            boxes = crate_boxes(c, p["half"], p["yaw"], p["style"], p["tint"], kind="mover", mover=idx)
            p0 = cam.project(world_to_cam(c, *cams_L[0])); p1 = cam.project(world_to_cam(c + p["delta"], *cams_L[1]))
            Zg, _ = triangulate_depth(p0, p1, cam.K, cam.K, R21, t21)
            mv = dict(idx=idx, cls="crate", style=p["style"], tint=[float(t) for t in p["tint"]], s=float(p["s"]),
                      eps_target=0.0, eps_center=abs(float(epipolar_offset(p0, p1, F_))), beyond=False, motion_type="parallel",
                      delta=[float(x) for x in p["delta"]], centroid0=c.tolist(), Z_center=float(world_to_cam(c, *cams_L[0])[2]),
                      Zgeo_center=float(Zg), parallax_center=float(translational_parallax(p0, p1, cam.K, cam.K, R21)),
                      dims=[float(x) for x in p["half"]], yaw=float(p["yaw"]), lift=float(p["lift"]) if cond == "F" else 0.0,
                      support="plane" if cond == "G" else "none", bbox0=[float(x) for x in (p["bbG"] if cond == "G" else p["bbF"])[0]])
            mv["box_ids"] = list(range(len(sc["boxes"]), len(sc["boxes"]) + len(boxes)))
            sc["boxes"] += boxes; sc["movers"].append(mv)
        sc.pop("_arr", None)
        rr = []
        for k, (fi, dk) in enumerate(((i, di), (j, dj))):
            R_, C_ = cams_L[k]
            col, alpha, zt, mid = render_movers_only(sc, k)
            real = seq.rgb(fi).astype(np.float32) / 255.0
            onp, _ = _on_plane(dk, cam, R_, C_)
            if cond == "G":
                real, w_sh = _contact_shadow(real, dk, cam, R_, C_, [(np.asarray(p["cG"]) + k * p["delta"], p["half"], p["yaw"])
                                                                     for p in placed], onp)
            else:
                w_sh = np.zeros(dk.shape, np.float32)
            nz = np.random.default_rng(seed * 7 + k).normal(0, 1.2 / 255, col.shape).astype(np.float32)
            # real surfaces in front hide the crate; the support plane cannot (the camera is above it), so
            # sensor noise on the plane never eats into the contact line
            with np.errstate(invalid="ignore"):
                occl = np.isfinite(dk) & np.isfinite(zt) & (dk < zt - np.maximum(0.02, _plane_tol(dk))) & ~onp
            vis = (alpha > 0) & ~occl
            a_ = np.where(vis, alpha, 0.0)
            comp = a_[..., None] * (col + nz) + (1 - a_[..., None]) * real
            mover_px = a_ >= 0.5
            depth = np.where(mover_px, zt, dk).astype(np.float32)
            sem = np.where(mover_px, SEM["mover"], np.where(np.isfinite(dk) & (w_sh < 0.1), SEM["static_box"], 0)).astype(np.int8)
            rr.append(dict(rgb=(np.clip(comp, 0, 1) * 255 + 0.5).astype(np.uint8), depth=depth, sem=sem,
                           mover_id=np.where(mover_px, mid, -1).astype(np.int16), mover_shadow=w_sh >= 0.1))
        scenes[cond], renders[cond] = sc, rr
    return (scenes, renders, plane), None

def rabl_contact_pixels(scene, render, k, j):
    # visible pixels of crate j within rabl_contact_band_m of the support plane (plane frame: y = 0)
    cam, R, C = frame_camera(scene, k)
    m_ = render["mover_id"] == j
    if not m_.any():
        return 0
    vv, uu = np.nonzero(m_)
    P = C + (cam.rays(uu.astype(np.float64), vv.astype(np.float64)) * render["depth"][vv, uu][:, None].astype(np.float64)) @ R.T
    return int(np.sum(-P[:, 1] <= PREREG["rabl_contact_band_m"]))

RABL_BUILD_LOG_PATH = os.path.join(LOG_DIR, "convoy_real_contact_build.json")
with stage("convoy_real_contact"):
    if not RABL_STAGE.done():
        shutil.rmtree(os.path.join(RENDER_DIR, "rabl"), ignore_errors=True)
        _log_out, _n_base = {}, 0
        for name, seq in TUM.items():
            if DATASET_STATUS[name]["role"] != "static":
                continue
            vert = sequence_vertical(seq)
            log = dict(pairs=0, candidates=0, reasons=Counter(), crates=0,
                       plane_reference="motion-capture vertical" if vert is not None else "camera up",
                       vertical_axis=None if vert is None else vert.tolist(), camera_height_m=[], tilt_to_vertical_deg=[])
            for pk, (i, j, b, ang, werr) in enumerate(iter_static_pairs(seq, np.random.default_rng(seed_for("rabl-pairs", name)))):
                if log["pairs"] >= RUN_CONFIG["n_rabl_pairs_per_seq"] or log["candidates"] >= RABL_DESIGN["max_candidates_per_seq"]:
                    break
                log["candidates"] += 1
                res, why = build_rabl_pair(seq, i, j, RUN_CONFIG["rabl_movers_per_pair"], seed_for("rabl", name, pk), vert)
                if res is None:
                    log["reasons"][why] += 1
                    continue
                scenes, renders, plane = res
                log["camera_height_m"].append(round(plane["height"], 3))
                log["tilt_to_vertical_deg"].append(round(plane["tilt_to_vertical_deg"], 2) if np.isfinite(plane["tilt_to_vertical_deg"]) else None)
                for cond in RABL_DESIGN["conditions"]:
                    d = os.path.join(RENDER_DIR, "rabl", f"{name}_{log['pairs']:03d}_{cond}")
                    os.makedirs(d, exist_ok=True)
                    for k, r in enumerate(renders[cond]):
                        _save_frame(d, k, r)
                    sc = scenes[cond]
                    gt01 = mover_gt_stats(sc, renders[cond], 0, 1)
                    meta = dict(scene=scene_to_json(sc), gt01=gt01, gt0last=gt01, set="RABL", index=_n_base, condition=cond,
                                seq=name, frames=[i, j], baseline=b, rot_deg=ang, warp_err=werr, dir=d, complete=True,
                                contact_px=[[rabl_contact_pixels(sc, renders[cond][k], k, jj) for k in (0, 1)]
                                            for jj in range(len(sc["movers"]))])
                    with open(os.path.join(d, "scene.json"), "w") as f:
                        json.dump(meta, f)
                log["crates"] += len(scenes["G"]["movers"])
                log["pairs"] += 1; _n_base += 1
            log["reasons"] = dict(log["reasons"])
            _log_out[name] = log
        with open(RABL_BUILD_LOG_PATH, "w") as f:
            json.dump(_log_out, f, indent=1)
        if all(DATASET_STATUS[n]["downloaded"] for n, v in TUM_SEQUENCES.items() if v["role"] == "static"):
            RABL_STAGE.commit([os.path.join(RENDER_DIR, "rabl"), RABL_BUILD_LOG_PATH])
        else:
            print("E5-REAL is incomplete (a static TUM sequence is missing); it is not checkpointed.")
META_RABL = sorted(_load_metas(os.path.join(RENDER_DIR, "rabl")), key=lambda m: (m["index"], m["condition"])) \
    if os.path.isdir(os.path.join(RENDER_DIR, "rabl")) else []
MOVERS_RABL = _regimes(_mover_table(META_RABL)) if META_RABL else pd.DataFrame()
if len(MOVERS_RABL):
    MOVERS_RABL["seq"] = MOVERS_RABL["scene"].map({m["index"]: m["seq"] for m in META_RABL})
_rlog = json.load(open(RABL_BUILD_LOG_PATH)) if os.path.exists(RABL_BUILD_LOG_PATH) else {}
for k, v in _rlog.items():
    _h = [x for x in v.get("camera_height_m", []) if x is not None]
    print(f"{k}: {v['pairs']} pairs with {v['crates']} crates from {v['candidates']} candidate pairs; skipped: {v['reasons']}")
    if _h:
        print(f"    support plane referenced to the {v['plane_reference']}; camera {np.min(_h):.2f}-{np.max(_h):.2f} m above it "
              f"(median {np.median(_h):.2f} m)")
if len(MOVERS_RABL):
    _g = MOVERS_RABL[MOVERS_RABL["condition"] == "G"]
    print(f"E5-REAL: {len(_g)} crates x {len(RABL_DESIGN['conditions'])} conditions in {MOVERS_RABL['scene'].nunique()} pairs; "
          f"contact visible when grounded: {int((_g['contact_px0'] >= 5).sum())}/{len(_g)}; "
          f"measurable and identifiable in both conditions: "
          f"{int(MOVERS_RABL.groupby('uid').apply(lambda d: bool((d['measurable'] & d['identifiable']).all())).sum())}")
else:
    print("E5-REAL: no composite could be built (no TUM static sequence or no support plane); it is left out of the run.")
release_memory(); _ = mem_report("E5-REAL composites")

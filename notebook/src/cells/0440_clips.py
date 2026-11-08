# PLACEHOLDER
# 3.1.4 Clip sampler: cameras on a straight or curved path, and movers with an exactly
#       controlled parallel fraction s and exact epipolar offset eps (frames 0 -> 1).


def scene_to_json(scene):
    return {k: v for k, v in scene.items() if not k.startswith("_")}

def make_design_deck(n_items, seed, classes=None, eps_targets=None, p_beyond=None):
    # exact proportions (rounded), shuffled with a fixed seed
    rng = np.random.default_rng(seed)
    classes = classes or MOVER_DESIGN["classes"]
    eps_targets = eps_targets or MOVER_DESIGN["eps_targets_px"]
    p_beyond = MOVER_DESIGN["p_beyond"] if p_beyond is None else p_beyond
    def _alloc(d):
        keys = list(d); w = np.array([d[k] for k in keys], np.float64); w /= w.sum()
        cnt = np.floor(w * n_items).astype(int)
        for j in np.argsort(-(w * n_items - cnt))[: n_items - cnt.sum()]:
            cnt[j] += 1
        out = [k for k, c in zip(keys, cnt) for _ in range(c)]
        rng.shuffle(out)
        return out
    cls = _alloc(classes); eps = _alloc(eps_targets)
    n_b = int(round(p_beyond * n_items))
    beyond = np.array([True] * n_b + [False] * (n_items - n_b)); rng.shuffle(beyond)
    return [dict(cls=c, eps=float(e), beyond=bool(b)) for c, e, b in zip(cls, eps, beyond)]

def _camera_params(rng, kind, street, n_frames, e4=False):
    cd = CAMERA_DESIGN[kind]
    psi = np.deg2rad(rng.uniform(*cd["yaw_deg"])) * (1 if kind == "forward" else rng.choice([-1, 1]))
    lanes = lane_centres(street["road_half"])
    jit = np.deg2rad(CAMERA_DESIGN["rot_jitter_deg"])
    return dict(kind=kind, psi=psi, pitch0=np.deg2rad(rng.uniform(-5.0, 1.0)),
                f=float(rng.uniform(*CAMERA_DESIGN["focal_px"])),
                mag=float(rng.uniform(*cd["baseline_m"])) * (0.75 if e4 else 1.0),
                eta=np.deg2rad(rng.uniform(-2, 2)),
                omega=np.deg2rad(rng.uniform(5.0, 8.0)) * rng.choice([-1, 1]),
                x_cam=float(rng.choice([-1.75, 1.75])) if kind == "forward"
                      else float(-np.sign(psi) * (lanes.max() if kind == "side" else 1.75)),
                jitter=rng.uniform(-jit, jit, size=(n_frames, 3)))

def _cams_from_params(p, n_frames, path):
    # Straight: constant translation b per frame. Curved: the translation direction turns by
    # omega per frame (frames 0 and 1 are identical in both cases). The per-frame rotations
    # (small independent jitter around a fixed heading) are identical for both paths, so a
    # straight/curved twin differs ONLY in the translation path (Proposition 2).
    omega = p["omega"] if path == "curved" else 0.0
    cams, C = [], np.array([p["x_cam"], 0.0, 0.0])
    for k in range(n_frames):
        R = rot_ypr(p["psi"], p["pitch0"], 0.0) if k == 0 else \
            rot_ypr(p["psi"] + p["jitter"][k, 0], p["pitch0"] + p["jitter"][k, 1], p["jitter"][k, 2])
        cams.append(dict(fx=p["f"], fy=p["f"], cx=319.5, cy=239.5, W=RUN_CONFIG["W"], H=RUN_CONFIG["H"],
                         R=R.tolist(), C=C.tolist()))
        heading = p["eta"] + k * omega
        C = C + p["mag"] * np.array([np.sin(heading), 0.0, np.cos(heading)])
    meta = dict(kind=p["kind"], psi_deg=float(np.rad2deg(p["psi"])), focal=p["f"], baseline=p["mag"],
                omega_deg=float(np.rad2deg(omega)), path=path)
    return cams, meta

def _object_boxes(cls, ref, dims, yaw, style, tint, mover_idx):
    if cls == "car":
        return car_boxes(ref, yaw, dims[0], dims[1], tint, kind="mover", mover=mover_idx)
    return crate_boxes(ref, dims, yaw, style, tint, kind="mover", mover=mover_idx)

def _project_boxes(boxes, offset, cam, R, C, znear=2.0):
    # image coordinates of all box corners; None if any corner is closer than znear (metres)
    pts = np.concatenate([box_corners(b, offset) for b in boxes])
    Xc = (pts - C) @ R
    if np.any(Xc[:, 2] < znear):
        return None
    return cam.project(Xc)

def _place_mover(rng, item, scene, cam_sets, mover_idx, placed_sets, n_frames):
    # cam_sets: one or more camera paths sharing frames 0 and 1; every constraint (inside
    # the image with a margin, no overlap with other movers, inside the street, clear of
    # furniture) must hold in every frame of every path.
    cams = cam_sets[0]
    st = scene["street"]; g = st["ground_y"]; wR = st["road_half"]; xf = st["facade_x_right"]
    cam0 = PinholeCamera.from_dict(cams[0]); R0 = np.asarray(cams[0]["R"]); C0 = np.asarray(cams[0]["C"])
    cam1 = PinholeCamera.from_dict(cams[1]); R1 = np.asarray(cams[1]["R"]); C1 = np.asarray(cams[1]["C"])
    b = C1 - C0
    furn = [bx for bx in scene["boxes"] if bx["kind"] == "static"]
    furn_c = np.array([bx["center"] for bx in furn]); furn_r = np.array([np.linalg.norm(bx["half"]) for bx in furn])
    W, H = cam0.W, cam0.H
    for attempt in range(300):
        cls = item["cls"]
        Zc = rng.uniform(*MOVER_DESIGN["depth_range_m"])
        u = rng.uniform(0.08 * W, 0.92 * W)
        a = (u - cam0.cx) / cam0.fx
        if cls == "crate_floating":
            hover = rng.uniform(1.5, 5.0)
            y_target = g - hover
        else:
            y_target = g
        bv = ((y_target - C0[1]) / Zc - R0[1, 0] * a - R0[1, 2]) / R0[1, 1]
        P = C0 + Zc * (R0 @ np.array([a, bv, 1.0]))
        if cls == "car":
            L, Wd = rng.uniform(4.1, 4.8), rng.uniform(1.70, 1.85)
            lanes = lane_centres(wR)
            lane = lanes[np.argmin(np.abs(lanes - P[0]))]
            if abs(lane - P[0]) > 2.5:
                continue
            P[0] = lane + rng.uniform(-0.3, 0.3)
            dims = (L, Wd); style = "car"; tint = CAR_COLORS[int(rng.integers(len(CAR_COLORS)))]
            ref = np.array([P[0], g, P[2]])
            centroid = ref + np.array([0, -0.8, 0])
            ext = np.array([Wd / 2, 0.85, L / 2])
        else:
            a_ = rng.uniform(0.9, 2.0)
            half = np.array([a_ / 2, a_ / 2 * rng.uniform(0.8, 1.0), a_ / 2 * rng.uniform(0.8, 1.2)])
            style = ["wood", "container", "pattern"][int(rng.integers(3))]
            tint = CONTAINER_TINTS[int(rng.integers(len(CONTAINER_TINTS)))] if style == "container" else (1, 1, 1)
            ref = np.array([P[0], (g - half[1]) if cls == "crate" else (y_target - half[1]), P[2]])
            centroid = ref.copy(); dims = half; ext = half
        s = rng.uniform(*(MOVER_DESIGN["s_range_beyond"] if item["beyond"] else MOVER_DESIGN["s_range_main"]))
        yaw = (0.0 if s >= 0 else np.pi) + np.deg2rad(rng.uniform(-3, 3)) if cls == "car" else rng.uniform(0, np.pi / 2)
        delta = s * b
        motion_type = "parallel"
        if item["eps"] > 0:
            if cls == "crate_floating":
                n_dir = np.cross(centroid - C0, b); n_dir /= np.linalg.norm(n_dir)
            else:
                n_dir = perp_unit(b, "horizontal")
            m, ach = solve_perp_for_eps(centroid, s * b, n_dir, item["eps"], cam0, R0, C0, cam1, R1, C1, max_m=4.0)
            if m is None or abs(m) > 3.5 or not np.isfinite(ach) or abs(ach - item["eps"]) > 1e-3:
                continue
            delta = s * b + m * n_dir
            motion_type = "perpendicular_component"
        boxes = _object_boxes(cls, ref, dims, yaw, style, tint, mover_idx)
        ok = True; bbs_sets = []
        for cset, placed_bboxes in zip(cam_sets, placed_sets):
            bbs = []
            for k in range(n_frames):
                off = k * delta
                cR = np.asarray(cset[k]["R"]); cC = np.asarray(cset[k]["C"]); ck = PinholeCamera.from_dict(cset[k])
                uv = _project_boxes(boxes, off, ck, cR, cC)
                if uv is None:
                    ok = False; break
                u0, v0 = uv.min(0); u1, v1 = uv.max(0)
                if u0 < 6 or v0 < 6 or u1 > W - 7 or v1 > H - 7:
                    ok = False; break
                if k == 0 and (u1 - u0) * (v1 - v0) < 900:
                    ok = False; break
                for other in placed_bboxes:
                    ou0, ov0, ou1, ov1 = other[k]
                    if not (u1 + 8 < ou0 or ou1 + 8 < u0 or v1 + 8 < ov0 or ov1 + 8 < v0):
                        ok = False; break
                if not ok:
                    break
                ck_c = centroid + off
                if abs(ck_c[0]) + ext.max() > xf - 0.3 or (cls == "car" and abs(ck_c[0]) > wR - 0.9):
                    ok = False; break
                if len(furn_c) and np.any(np.linalg.norm(furn_c - ck_c, axis=1) < furn_r + np.linalg.norm(ext) + 0.2):
                    ok = False; break
                bbs.append((u0, v0, u1, v1))
            if not ok:
                break
            bbs_sets.append(bbs)
        if not ok:
            continue
        # exact centre quantities between frames 0 and 1
        p0 = cam0.project(world_to_cam(centroid, R0, C0)); p1 = cam1.project(world_to_cam(centroid + delta, R1, C1))
        R21, t21 = relative_pose(R0, C0, R1, C1)
        F = fundamental_matrix(cam0.K, cam1.K, R21, t21)
        eps_c = abs(float(epipolar_offset(p0, p1, F)))
        Zgeo, gap = triangulate_depth(p0, p1, cam0.K, cam1.K, R21, t21)
        par = float(translational_parallax(p0, p1, cam0.K, cam1.K, R21))
        Zc0 = float(world_to_cam(centroid, R0, C0)[2])
        mover = dict(idx=mover_idx, cls=cls, style=style, tint=[float(v) for v in tint], s=float(s),
                     eps_target=float(item["eps"]), eps_center=eps_c, beyond=bool(item["beyond"]),
                     motion_type=motion_type, delta=[float(v) for v in delta], centroid0=centroid.tolist(),
                     Z_center=Zc0, Zgeo_center=float(Zgeo), parallax_center=par,
                     dims=[float(v) for v in np.ravel(dims)], yaw=float(yaw),
                     bbox0=[float(v) for v in bbs_sets[0][0]])
        return mover, boxes, bbs_sets
    return None, None, None

def sample_clip(seed, n_frames=3, path="straight", kind=None, items=None, e4=False, return_failed=False):
    # path: "straight", "curved" or "twin" (returns a straight and a curved scene that share
    # the street, the movers and frames 0-1, and differ only in the later translation path)
    rng = np.random.default_rng(seed)
    if kind is None:
        kinds = ["forward", "oblique", "side"]
        kind = kinds[int(rng.choice(3, p=[CAMERA_DESIGN[k]["p"] for k in kinds]))]
    street = sample_street(rng, kind)
    furn_boxes, furn_shadows = sample_static_furniture(rng, street)
    scene = dict(seed=int(seed), street=street, boxes=furn_boxes, shadows=furn_shadows, movers=[],
                 light=dict(exposure=float(rng.uniform(0.92, 1.08)),
                            white_balance=[float(v) for v in rng.uniform(0.97, 1.03, 3)],
                            noise_sigma=1.2 / 255))
    params = _camera_params(rng, kind, street, n_frames, e4=e4)
    paths = ["straight", "curved"] if path == "twin" else [path]
    cam_sets, metas = zip(*[_cams_from_params(params, n_frames, p) for p in paths])
    scene["cams"] = cam_sets[0]; scene["camera"] = metas[0]
    placed_sets, failed = [[] for _ in paths], []
    items = items if items is not None else [dict(cls="car", eps=0.0, beyond=False)]
    for item in items:
        mv, boxes, bbs_sets = _place_mover(rng, item, scene, list(cam_sets), len(scene["movers"]), placed_sets, n_frames)
        if mv is None:
            failed.append(item)
            continue
        mv["box_ids"] = list(range(len(scene["boxes"]), len(scene["boxes"]) + len(boxes)))
        scene["boxes"] += boxes
        if mv["cls"] == "car":
            c0 = mv["centroid0"]
            scene["shadows"].append(shadow_fp(c0[0], c0[2], mv["dims"][1] / 2, mv["dims"][0] / 2, mv["yaw"], 0.60, mv["idx"]))
        elif mv["cls"] == "crate":
            c0 = mv["centroid0"]
            scene["shadows"].append(shadow_fp(c0[0], c0[2], mv["dims"][0], mv["dims"][2], mv["yaw"], 0.50, mv["idx"]))
        scene["movers"].append(mv)
        for ps, bb in zip(placed_sets, bbs_sets):
            ps.append(bb)
    scene.pop("_arr", None)
    if path == "twin":
        out = []
        for cams, meta in zip(cam_sets, metas):
            sc = json.loads(json.dumps(scene_to_json(scene)))
            sc["cams"], sc["camera"] = cams, meta
            out.append(sc)
        return (out, failed) if return_failed else out
    return (scene, failed) if return_failed else scene

def sample_clip_sequence(seeds, decks, n_frames, path, e4, max_movers=4):
    # Sequential sampling with carry-over: an item that cannot be placed in clip i is offered
    # again in clip i+1, so the realised design matches the deck as closely as possible.
    scenes, carry, unplaced = [], [], []
    for seed, deck_items in zip(seeds, decks):
        items = (carry + list(deck_items))[:max_movers]
        overflow = (carry + list(deck_items))[max_movers:]
        sc, failed = sample_clip(seed, n_frames=n_frames, path=path, items=items, e4=e4, return_failed=True)
        carry = failed + overflow
        scenes.append(sc)
    unplaced = carry
    return scenes, unplaced

print("Clip sampler ready: make_design_deck, sample_clip.")
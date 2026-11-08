# PLACEHOLDER
# 3.5 CONVOY-REAL: real TUM image pairs (real background, real camera motion, motion-capture
#     poses) with inserted rigid movers whose motion (s, eps) is controlled exactly. Movers
#     are ray-cast with the real intrinsics and poses, depth-tested against the measured
#     depth so that they are never inside or behind real surfaces, and alpha-blended at
#     their silhouettes. Positions are proposed only where the measured depth leaves free
#     space in front of the camera. A pair with no room for any mover (e.g. the camera close
#     to a desk or wall) is replaced by the next candidate pair, and a mover that does not fit
#     one pair is carried over to the next, so the designed number of movers and the designed
#     mix of epipolar offsets are kept.
from collections import Counter

REAL_EPS_TARGETS = {0.0: 0.45, 0.5: 0.10, 1.5: 0.05, 2.0: 0.05, 3.0: 0.10, 5.0: 0.10, 8.0: 0.10, 12.0: 0.05}
REAL_PLACEMENT = dict(attempts=300, z_min=0.7, z_max=3.0, clearance_window_px=41, min_clearance_m=1.1,
                      min_valid_frac=0.7, margin_px=70, max_pairs_per_mover=6)
_DUMMY_STREET = dict(ground_y=1e6, road_half=1.0, facade_x_left=-1e6, facade_x_right=1e6, backdrop_z=1e7,
                     skyline_shift=0.0, blocks_left=[[0, 1, 1, 0, 0]], blocks_right=[[0, 1, 1, 0, 0]])

def iter_static_pairs(seq, rng, b_range=(0.10, 0.30), max_angle=20.0, max_warp=0.06, max_starts=2000):
    # candidate frame pairs in random order: baseline in b_range, rotation <= max_angle, enough
    # valid depth in both frames, and depth/pose self-consistency (warp error) <= max_warp
    n = len(seq)
    starts = np.unique(np.linspace(0, n - 2, num=max(2, min(max_starts, n - 1))).astype(int))
    rng.shuffle(starts)
    seen = set()
    for i in starts:
        target = rng.uniform(*b_range)
        for j in range(i + 1, min(n, i + 150)):
            b, ang = baseline_and_angle(seq, i, j)
            if b >= target:
                if ang <= max_angle and b <= b_range[1] + 0.05 and (i, j) not in seen:
                    di, dj = seq.depth(i), seq.depth(j)
                    if np.isfinite(di).mean() >= 0.5 and np.isfinite(dj).mean() >= 0.5:
                        e = depth_warp_error(seq, i, j)
                        if np.isfinite(e) and e <= max_warp:
                            seen.add((i, j))
                            yield int(i), int(j), float(b), float(ang), float(e)
                break

def _free_space_candidates(depth):
    # pixels with at least min_clearance_m of free space in front of the nearest real surface
    # within the window, enough valid depth around them, and away from the image border
    P = REAL_PLACEMENT
    valid = np.isfinite(depth)
    k = np.ones((P["clearance_window_px"], P["clearance_window_px"]), np.uint8)
    clearance = cv2.erode(np.where(valid, depth, 1e3).astype(np.float32), k)
    vfrac = cv2.blur(valid.astype(np.float32), (P["clearance_window_px"], P["clearance_window_px"]))
    ok = valid & (clearance >= P["min_clearance_m"]) & (vfrac >= P["min_valid_frac"])
    m = P["margin_px"]
    ok[:m] = False; ok[-m:] = False; ok[:, :m] = False; ok[:, -m:] = False
    return clearance, np.argwhere(ok)

def render_movers_only(scene, frame_k):
    # colour, coverage (alpha), depth and mover id of the movers alone (anti-aliased edges)
    cam, R, C = frame_camera(scene, frame_k)
    W, H = cam.W, cam.H
    if not scene["movers"]:
        return (np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32),
                np.full((H, W), np.inf), np.full((H, W), -1, np.int32))
    vv, uu = np.mgrid[0:H, 0:W]
    D = cam.rays(uu.ravel().astype(np.float64), vv.ravel().astype(np.float64)) @ R.T
    ids = list(range(len(scene["movers"])))
    hit = cast_rays(scene, C, D, frame_k, True, grid=(W, H), cam=cam, R=R, only_movers=ids)
    f_mean = 0.5 * (cam.fx + cam.fy)
    hitm = np.isfinite(hit["t"])
    col = np.zeros((H * W, 3), np.float32)
    if hitm.any():
        hs = np.nonzero(hitm)[0]
        sub = {k: v[hs] for k, v in hit.items()}
        col[hs] = shade_rays(scene, C, D[hs], sub, frame_k, f_mean)[0]
    alpha = hitm.astype(np.float32)
    key = np.where(hitm, hit["bid"], -1).reshape(H, W)
    e = np.zeros((H, W), bool)
    dh = key[:, 1:] != key[:, :-1]; dv = key[1:, :] != key[:-1, :]
    e[:, 1:] |= dh; e[:, :-1] |= dh; e[1:, :] |= dv; e[:-1, :] |= dv
    ei = np.nonzero(e.ravel())[0]
    if ei.size:
        acc_c = col[ei] * alpha[ei, None]; acc_a = alpha[ei].copy()
        for du, dv_ in ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)):
            pu = uu.ravel()[ei] + du; pv = vv.ravel()[ei] + dv_
            Ds = cam.rays(pu, pv) @ R.T
            h2 = cast_rays(scene, C, Ds, frame_k, True, pix=(pu, pv), cam=cam, R=R, only_movers=ids)
            m2 = np.isfinite(h2["t"])
            if m2.any():
                q = np.nonzero(m2)[0]
                c2 = shade_rays(scene, C, Ds[q], {k: v[q] for k, v in h2.items()}, frame_k, f_mean)[0]
                acc_c[q] += c2
            acc_a += m2
        alpha[ei] = acc_a / 5.0
        col[ei] = np.where(acc_a[:, None] > 0, acc_c / np.maximum(acc_a[:, None], 1e-6), 0)
    arr = _scene_box_arrays(scene)
    mid = np.full(H * W, -1, np.int32)
    mid[hitm] = arr["mover"][hit["bid"][hitm]]                  # only pixels that hit a mover box
    return col.reshape(H, W, 3), alpha.reshape(H, W), hit["t"].reshape(H, W), mid.reshape(H, W)

def build_composite(seq, i, j, items, seed):
    # returns (scene, renders or None if no mover fits, placed flag per item, failure counts)
    P = REAL_PLACEMENT
    rng = np.random.default_rng(seed)
    cam = seq.cam
    Ri, Ci = seq.pose(i); Rj, Cj = seq.pose(j)
    di, dj = seq.depth(i), seq.depth(j)
    cams = [dict(**cam.to_dict(), R=Ri.tolist(), C=Ci.tolist()), dict(**cam.to_dict(), R=Rj.tolist(), C=Cj.tolist())]
    b = Cj - Ci
    scene = dict(seed=int(seed), street=_DUMMY_STREET, boxes=[], shadows=[], movers=[], cams=cams,
                 camera=dict(kind="tum", path="straight", focal=0.5 * (cam.fx + cam.fy), baseline=float(np.linalg.norm(b)),
                             psi_deg=0.0, omega_deg=0.0),
                 light=dict(exposure=1.0, white_balance=[1.0, 1.0, 1.0], noise_sigma=1.2 / 255))
    real_d = [di, dj]
    placed, flags, fails = [], [], Counter()
    clearance, cand = _free_space_candidates(di)
    if len(cand) == 0:
        fails["no free space in view 1"] += 1
        return scene, None, [False] * len(items), fails
    for item in items:
        ok_item = False
        for attempt in range(P["attempts"]):
            v_, u_ = cand[int(rng.integers(len(cand)))]
            u, v = u_ + rng.uniform(-0.5, 0.5), v_ + rng.uniform(-0.5, 0.5)
            z_hi = min(P["z_max"], float(clearance[v_, u_]) - 0.35)
            Zc = rng.uniform(P["z_min"], z_hi)
            a_ = rng.uniform(0.12, 0.30)
            half = np.array([a_ / 2, a_ / 2 * rng.uniform(0.8, 1.0), a_ / 2 * rng.uniform(0.8, 1.2)])
            yaw = rng.uniform(0, np.pi / 2)
            centroid = Ci + Ri @ (cam.rays(u, v) * Zc)
            s = rng.uniform(*MOVER_DESIGN["s_range_main"])
            delta = s * b; motion_type = "parallel"
            if item["eps"] > 0:
                n_dir = np.cross(centroid - Ci, b); n_dir /= np.linalg.norm(n_dir)
                m, ach = solve_perp_for_eps(centroid, s * b, n_dir, item["eps"], cam, Ri, Ci, cam, Rj, Cj, max_m=1.0)
                if m is None or abs(ach - item["eps"]) > 1e-3:
                    fails["offset not reachable"] += 1
                    continue
                delta = s * b + m * n_dir; motion_type = "perpendicular_component"
            style = ["wood", "container", "pattern"][int(rng.integers(3))]
            tint = CONTAINER_TINTS[int(rng.integers(len(CONTAINER_TINTS)))] if style == "container" else (1, 1, 1)
            boxes = crate_boxes(centroid, half, yaw, style, tint, kind="mover", mover=len(scene["movers"]))
            ok, bbs, why = True, [], None
            for k, (R_, C_) in enumerate(((Ri, Ci), (Rj, Cj))):
                uv = _project_boxes(boxes, k * delta, cam, R_, C_, znear=0.3)
                if uv is None:
                    ok, why = False, "outside view"; break
                u0, v0 = uv.min(0); u1, v1 = uv.max(0)
                if u0 < 8 or v0 < 8 or u1 > cam.W - 9 or v1 > cam.H - 9 or (k == 0 and (u1 - u0) * (v1 - v0) < 600):
                    ok, why = False, "outside view"; break
                zmax = ((np.concatenate([box_corners(bx, k * delta) for bx in boxes]) - C_) @ R_)[:, 2].max()
                reg = real_d[k][int(v0):int(v1) + 1, int(u0):int(u1) + 1]
                fin = np.isfinite(reg)
                if fin.mean() < 0.6 or np.nanmin(reg) < zmax + 0.15:
                    ok, why = False, "behind a real surface"; break
                for other in placed:
                    ou0, ov0, ou1, ov1 = other[k]
                    if not (u1 + 8 < ou0 or ou1 + 8 < u0 or v1 + 8 < ov0 or ov1 + 8 < v0):
                        ok, why = False, "overlaps another mover"; break
                if not ok:
                    break
                bbs.append((u0, v0, u1, v1))
            if not ok:
                fails[why] += 1
                continue
            p0 = cam.project(world_to_cam(centroid, Ri, Ci)); p1 = cam.project(world_to_cam(centroid + delta, Rj, Cj))
            R21, t21 = relative_pose(Ri, Ci, Rj, Cj)
            F = fundamental_matrix(cam.K, cam.K, R21, t21)
            Zg, _ = triangulate_depth(p0, p1, cam.K, cam.K, R21, t21)
            mv = dict(idx=len(scene["movers"]), cls="crate_floating", style=style, tint=[float(t) for t in tint],
                      s=float(s), eps_target=float(item["eps"]), eps_center=abs(float(epipolar_offset(p0, p1, F))),
                      beyond=False, motion_type=motion_type, delta=[float(x) for x in delta],
                      centroid0=centroid.tolist(), Z_center=float(world_to_cam(centroid, Ri, Ci)[2]),
                      Zgeo_center=float(Zg), parallax_center=float(translational_parallax(p0, p1, cam.K, cam.K, R21)),
                      dims=[float(x) for x in half], yaw=float(yaw), bbox0=[float(x) for x in bbs[0]])
            mv["box_ids"] = list(range(len(scene["boxes"]), len(scene["boxes"]) + len(boxes)))
            scene["boxes"] += boxes; scene["movers"].append(mv); placed.append(bbs)
            scene.pop("_arr", None)
            ok_item = True
            break
        flags.append(ok_item)
    if not scene["movers"]:
        return scene, None, flags, fails
    renders = []
    for k, (fi, dk) in enumerate(((i, di), (j, dj))):
        col, alpha, zt, mid = render_movers_only(scene, k)
        real = seq.rgb(fi).astype(np.float32) / 255.0
        nz = np.random.default_rng(seed * 7 + k).normal(0, 1.2 / 255, col.shape).astype(np.float32)
        comp = alpha[..., None] * (col + nz) + (1 - alpha[..., None]) * real
        mover_px = alpha >= 0.5
        depth = np.where(mover_px, zt, dk).astype(np.float32)
        sem = np.where(mover_px, SEM["mover"], np.where(np.isfinite(dk), SEM["static_box"], 0)).astype(np.int8)
        renders.append(dict(rgb=(np.clip(comp, 0, 1) * 255 + 0.5).astype(np.uint8), depth=depth, sem=sem,
                            mover_id=np.where(mover_px, mid, -1).astype(np.int16),
                            mover_shadow=np.zeros(depth.shape, bool)))
    return scene, renders, flags, fails

REAL_BUILD_LOG_PATH = os.path.join(LOG_DIR, "convoy_real_build.json")
with stage("convoy_real"):
    _st = TUM_STAGES["REAL"]
    if not _st.done():
        # built from scratch: earlier composites (and their optical flows) are removed first
        shutil.rmtree(os.path.join(RENDER_DIR, "real"), ignore_errors=True)
        shutil.rmtree(os.path.join(FLOW_DIR, "real"), ignore_errors=True)
        _log_out = {}
        _n_meta = 0
        for name, seq in TUM.items():
            if DATASET_STATUS[name]["role"] != "static":
                continue
            n_target, n_m = RUN_CONFIG["n_real_pairs_per_seq"], RUN_CONFIG["real_movers_per_pair"]
            deck = make_design_deck(n_target * n_m, seed_for("deck-real", name), classes={"crate_floating": 1.0},
                                    eps_targets=REAL_EPS_TARGETS, p_beyond=0.0)
            queue = [dict(item=it, tries=0) for it in deck]
            used, log = [], dict(designed=len(deck), designed_eps=Counter(it["eps"] for it in deck), candidates=0,
                                 no_room=0, carried=0, dropped=0, fails=Counter())
            for pk, (i, j, b, ang, werr) in enumerate(iter_static_pairs(seq, np.random.default_rng(seed_for("real-pairs", name)))):
                if not queue:
                    break
                log["candidates"] += 1
                take, queue = queue[:n_m], queue[n_m:]
                scene, renders, flags, fails = build_composite(seq, i, j, [t["item"] for t in take], seed_for("real", name, pk))
                log["fails"].update(fails)
                back = []
                for t, ok in zip(take, flags):
                    if not ok:
                        t["tries"] += 1
                        if t["tries"] < REAL_PLACEMENT["max_pairs_per_mover"]:
                            back.append(t); log["carried"] += 1
                        else:
                            log["dropped"] += 1
                queue = back + queue                                 # unplaced movers go first on the next pair
                if renders is None:
                    log["no_room"] += 1
                    continue
                d = os.path.join(RENDER_DIR, "real", f"{name}_{len(used):03d}")
                os.makedirs(d, exist_ok=True)
                for k, r in enumerate(renders):
                    _save_frame(d, k, r)
                gt01 = mover_gt_stats(scene, renders, 0, 1)
                meta = dict(scene=scene_to_json(scene), gt01=gt01, gt0last=gt01, set="REAL", index=_n_meta,
                            seq=name, frames=[i, j], baseline=b, rot_deg=ang, warp_err=werr, dir=d, complete=True)
                with open(os.path.join(d, "scene.json"), "w") as f:
                    json.dump(meta, f)
                _n_meta += 1
                used.append((i, j, b, ang, werr))
            log["dropped"] += len(queue)                             # candidate pairs exhausted
            log = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in log.items()}
            log["designed_eps"] = {str(k): v for k, v in log["designed_eps"].items()}
            _log_out[name] = dict(pairs=used, log=log)
        with open(REAL_BUILD_LOG_PATH, "w") as f:
            json.dump(_log_out, f, indent=1)
        if all(DATASET_STATUS[n]["downloaded"] for n, v in TUM_SEQUENCES.items() if v["role"] == "static"):
            _st.commit([os.path.join(RENDER_DIR, "real"), REAL_BUILD_LOG_PATH, A3_PATH])
        else:
            print("CONVOY-REAL is incomplete (a static TUM sequence is missing); it is not checkpointed.")
META_REAL = sorted(_load_metas(os.path.join(RENDER_DIR, "real")), key=lambda m: m["index"]) \
    if os.path.isdir(os.path.join(RENDER_DIR, "real")) else []
_blog = {}
if os.path.exists(REAL_BUILD_LOG_PATH):
    with open(REAL_BUILD_LOG_PATH) as f:
        _blog = json.load(f)
REAL_PAIRS = {k: v["pairs"] for k, v in _blog.items()}
REAL_BUILD_LOG = {k: v["log"] for k, v in _blog.items()}

MOVERS_REAL = _mover_table(META_REAL) if META_REAL else pd.DataFrame()
if len(MOVERS_REAL):
    MOVERS_REAL["seq"] = [META_REAL[i]["seq"] for i in MOVERS_REAL["scene"]]
    MOVERS_REAL["x_par"] = -np.log(1 - MOVERS_REAL["s"])
    MOVERS_REAL["regime"] = np.where(MOVERS_REAL["gt_eps_med"] < PREREG["eps_parallel_max"], "parallel",
                                     np.where(MOVERS_REAL["gt_eps_med"] >= PREREG["eps_perp_min"], "perpendicular", "intermediate"))
    MOVERS_REAL["measurable"] = (MOVERS_REAL["gt_n_eroded"] >= PREREG["min_object_pixels"]) & (MOVERS_REAL["gt_visible_frac"] >= 0.85)
    MOVERS_REAL["identifiable"] = MOVERS_REAL["gt_parallax_med"] >= PREREG["parallax_min_px"]
    MOVERS_REAL.to_csv(os.path.join(MANIFEST_DIR, "convoy_real_movers.csv"), index=False)
for name, pairs in REAL_PAIRS.items():
    lg = REAL_BUILD_LOG[name]
    n_placed = int((MOVERS_REAL["seq"] == name).sum()) if len(MOVERS_REAL) else 0
    print(f"{name}: {len(pairs)} pairs from {lg['candidates']} candidates ({lg['no_room']} without room for a mover); "
          f"{n_placed}/{lg['designed']} designed movers placed ({lg['carried']} moved to another pair, {lg['dropped']} not placed)")
    if pairs:
        bs = [p[2] for p in pairs]
        print(f"    baseline {np.min(bs):.2f}-{np.max(bs):.2f} m, median warp error {np.median([p[4] for p in pairs]):.3f}; "
              f"rejected placement attempts: {dict(lg['fails'])}")
if len(MOVERS_REAL):
    print(f"CONVOY-REAL: {len(MOVERS_REAL)} movers in {len(META_REAL)} pairs;",
          MOVERS_REAL.groupby("regime").size().to_dict(), "| measurable:", int(MOVERS_REAL["measurable"].sum()),
          "| identifiable:", int(MOVERS_REAL["identifiable"].sum()))
    _designed = sum((Counter({float(k): v for k, v in l["designed_eps"].items()}) for l in REAL_BUILD_LOG.values()), Counter())
    print("placed / designed movers per target offset (px):",
          {e: f"{int((MOVERS_REAL['eps_target'] == e).sum())}/{_designed[e]}" for e in sorted(_designed)})

release_memory(); _ = mem_report("CONVOY-REAL composites")

# PLACEHOLDER
# 3.10 KITTI tracking (exploratory, v6.2): find an attached copy, read calibration, OXTS poses and
#      labels, check the poses against the LiDAR (A3), choose frame pairs and build the LiDAR ground
#      truth of every vehicle. One checkpointed stage ("kitti_gt"); its archive holds the cropped
#      images, so the evaluation of a later session does not need the dataset attached again.
KT_REQUIRED = ("image_02", "velodyne", "label_02", "calib", "oxts")
KT_STATUS = dict(available=False, root=None, note="", sequences=[])
KT_DIR = os.path.join(SCRATCH_DIR, "kitti")
KT_PAIR_META, KT_A3 = [], {}

def kt_find_root():
    for base in [os.environ.get(KT_DESIGN["root_env"], ""), "/kaggle/input"]:
        if not base or not os.path.isdir(base):
            continue
        for dp, dns, _ in os.walk(base):
            if dp[len(base):].count(os.sep) >= 6:
                dns[:] = []
                continue
            if all(k in dns for k in KT_REQUIRED):
                return dp
    return None

def _kt_rx(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def _kt_ry(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def _kt_rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def _h4(M):
    T = np.eye(4); T[:M.shape[0], :M.shape[1]] = M; return T

def kt_calib(path):
    d = {}
    with open(path) as f:
        for line in f:
            p = line.replace(":", " ").split()
            if len(p) > 1:
                d[p[0]] = np.array([float(v) for v in p[1:]])
    alias = lambda *ks: next(d[k] for k in ks if k in d)
    P2 = alias("P2").reshape(3, 4)
    R_rect = alias("R_rect", "R0_rect").reshape(3, 3)
    Tvc = alias("Tr_velo_cam", "Tr_velo_to_cam").reshape(3, 4)
    Tiv = alias("Tr_imu_velo", "Tr_imu_to_velo").reshape(3, 4)
    K2 = P2[:, :3].copy()
    t2 = np.linalg.solve(K2, P2[:, 3])                   # camera 2 = rectified camera 0 shifted by t2
    T_rect_velo = _h4(R_rect) @ _h4(Tvc)
    return dict(K2=K2, t2=t2, T_rect_velo=T_rect_velo, T_rect_imu=T_rect_velo @ _h4(Tiv))

def kt_oxts_poses(path):
    a = np.loadtxt(path, ndmin=2)
    er = 6378137.0
    scale = np.cos(a[0, 0] * np.pi / 180.0)
    t = np.stack([scale * a[:, 1] * np.pi * er / 180.0, scale * er * np.log(np.tan((90.0 + a[:, 0]) * np.pi / 360.0)), a[:, 2]], 1)
    t = t - t[0]
    out = []
    for k in range(len(a)):
        T = np.eye(4); T[:3, :3] = _kt_rz(a[k, 5]) @ _kt_ry(a[k, 4]) @ _kt_rx(a[k, 3]); T[:3, 3] = t[k]
        out.append(T)
    return out

def kt_labels(path):
    lab = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 17:
                continue
            fr = int(p[0])
            lab.setdefault(fr, []).append(dict(track=int(p[1]), type=p[2], trunc=float(p[3]), occ=int(float(p[4])),
                                               bbox=[float(v) for v in p[6:10]], h=float(p[10]), w=float(p[11]), l=float(p[12]),
                                               loc=np.array([float(v) for v in p[13:16]]), ry=float(p[16])))
    return lab

class KTSequence:
    def __init__(self, root, name):
        self.root, self.name = root, name
        self.calib = kt_calib(os.path.join(root, "calib", name + ".txt"))
        self.imu = kt_oxts_poses(os.path.join(root, "oxts", name + ".txt"))
        self.labels = kt_labels(os.path.join(root, "label_02", name + ".txt"))
        self.frames = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(os.path.join(root, "image_02", name)) if f.endswith(".png"))
        self.frames = [f for f in self.frames if f < len(self.imu)
                       and os.path.exists(os.path.join(root, "velodyne", name, f"{f:06d}.bin"))]
        self.convention = "documented"
        im = self.rgb(self.frames[0]) if self.frames else np.zeros((375, 1242, 3), np.uint8)
        self.H, self.W = im.shape[:2]

    def rgb(self, f):
        return np.asarray(Image.open(os.path.join(self.root, "image_02", self.name, f"{f:06d}.png")).convert("RGB"))

    def pose(self, f):
        # camera 2: rotation camera -> world (columns = camera axes) and centre, in the OXTS world
        Tri = self.calib["T_rect_imu"]
        T_w_rect = self.imu[f] @ (np.linalg.inv(Tri) if self.convention == "documented" else Tri)
        R = T_w_rect[:3, :3]
        return R, T_w_rect[:3, :3] @ (-self.calib["t2"]) + T_w_rect[:3, 3]

    def lidar_cam(self, f):
        # LiDAR points in camera-2 coordinates (N, 3), in front of the camera within the depth range
        X = np.fromfile(os.path.join(self.root, "velodyne", self.name, f"{f:06d}.bin"), np.float32).reshape(-1, 4)
        Xr = (self.calib["T_rect_velo"] @ np.c_[X[:, :3], np.ones(len(X))].T).T[:, :3] + self.calib["t2"]
        z = Xr[:, 2]
        return Xr[(z >= KT_DESIGN["lidar_min_depth_m"]) & (z <= KT_DESIGN["lidar_max_depth_m"])]

    def project(self, Xc):
        uvw = Xc @ self.calib["K2"].T
        return uvw[:, :2] / uvw[:, 2:3]

    def box_world(self, f, ob):
        # labelled box (bottom centre, KITTI rotation about the rectified camera's y axis) in the world
        R, C = self.pose(f)
        loc_c = ob["loc"] + self.calib["t2"]                 # rectified camera 0 -> camera 2 coordinates
        return R @ loc_c + C, R @ _kt_ry(ob["ry"])

def kt_in_box(Xc, ob, t2, margin, ground_cut=None):
    # points (camera-2 coordinates) inside a labelled box, optionally without its lowest ground_cut metres
    q = (Xc - t2 - ob["loc"]) @ _kt_ry(ob["ry"])           # object frame: x length, y down (0 = bottom), z width
    ins = (np.abs(q[:, 0]) <= ob["l"] / 2 + margin) & (np.abs(q[:, 2]) <= ob["w"] / 2 + margin) & \
          (q[:, 1] >= -ob["h"] - margin) & (q[:, 1] <= margin)
    if ground_cut is not None:
        ins &= q[:, 1] <= -ground_cut
    return ins

def _kt_depth_image(uv, z, H, W):
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], z[ok]
    D = np.full((H, W), np.inf, np.float32)
    np.minimum.at(D, (v, u), z.astype(np.float32))
    D[~np.isfinite(D)] = np.nan
    return D

def kt_a3(seq, pairs):
    # median relative disagreement between frame j's LiDAR and frame i's static LiDAR warped with the poses
    errs = []
    for i, j in pairs:
        Ri, Ci = seq.pose(i); Rj, Cj = seq.pose(j)
        X = seq.lidar_cam(i)
        stat = np.ones(len(X), bool)
        for ob in seq.labels.get(i, []):
            if ob["type"] != "DontCare":
                stat &= ~kt_in_box(X, ob, seq.calib["t2"], KT_DESIGN["static_box_margin_m"])
        Xw = X[stat] @ Ri.T + Ci
        Xj = (Xw - Cj) @ Rj
        Xj = Xj[Xj[:, 2] > 1.0]
        Dj = _kt_depth_image(seq.project(seq.lidar_cam(j)), seq.lidar_cam(j)[:, 2], seq.H, seq.W)
        uv = np.round(seq.project(Xj)).astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < seq.W) & (uv[:, 1] >= 0) & (uv[:, 1] < seq.H)
        zo = Dj[uv[ok, 1], uv[ok, 0]]; zp = Xj[ok, 2]
        m = np.isfinite(zo)
        if m.sum() >= 200:
            errs.append(float(np.median(np.abs(zo[m] - zp[m]) / zo[m])))
    return float(np.median(errs)) if errs else np.nan

def _kt_cam_dict(seq, f, x0, y0, w, h):
    R, C = seq.pose(f); K = seq.calib["K2"]
    return dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2] - x0), cy=float(K[1, 2] - y0), W=int(w), H=int(h),
                R=R.tolist(), C=C.tolist())

def kt_pair_gt(seq, i, j, x0, y0, w, h):
    # LiDAR ground truth of one pair in the crop: sparse depth (nearest return per pixel after the
    # occlusion test of Section 3.10.0), static mask, per-vehicle pixels and statistics
    t2 = seq.calib["t2"]
    Ri, Ci = seq.pose(i); Rj, Cj = seq.pose(j)
    X = seq.lidar_cam(i)
    uv = seq.project(X) - np.array([x0, y0])
    inside = (uv[:, 0] >= -0.5) & (uv[:, 0] < w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < h - 0.5)
    X, uv = X[inside], uv[inside]
    D, idx = lidar_zbuffer(uv, X[:, 2], h, w, KT_DESIGN["zbuffer_window_px"], *KT_DESIGN["zbuffer_tol"])
    has = idx >= 0
    win = np.zeros(len(X), bool)
    win[idx[has]] = True                                   # points that give a pixel its depth
    labs_i = {o["track"]: o for o in seq.labels.get(i, []) if o["type"] != "DontCare"}
    labs_j = {o["track"]: o for o in seq.labels.get(j, []) if o["type"] != "DontCare"}
    stat = np.ones(len(X), bool)
    for ob in labs_i.values():
        stat &= ~kt_in_box(X, ob, t2, KT_DESIGN["static_box_margin_m"])
    sm = np.zeros((h, w), bool)
    sm[has] = stat[idx[has]]
    for o in seq.labels.get(i, []):
        if o["type"] == "DontCare":
            l_, t_, r_, b_ = o["bbox"]
            sm[max(int(t_ - y0), 0):max(int(b_ - y0) + 1, 0), max(int(l_ - x0), 0):max(int(r_ - x0) + 1, 0)] = False
    b = Cj - Ci
    sc = dict(cams=[_kt_cam_dict(seq, i, x0, y0, w, h), _kt_cam_dict(seq, j, x0, y0, w, h)])
    vehicles = []
    for tid, oi in labs_i.items():
        oj = labs_j.get(tid)
        if oj is None or oi["type"] not in KT_DESIGN["classes"]:
            continue
        Pi, Rwi = seq.box_world(i, oi); Pj, Rwj = seq.box_world(j, oj)
        disp = float(np.linalg.norm(Pj - Pi))
        if KT_DESIGN["max_truncation"] < max(oi["trunc"], oj["trunc"]) or max(oi["occ"], oj["occ"]) > KT_DESIGN["max_occlusion"]:
            continue
        state = "moving" if disp >= PREREG["kt_moving_disp_m"] else "parked" if disp <= PREREG["kt_parked_disp_m"] else None
        if state is None:
            continue
        own = kt_in_box(X, oi, t2, 0.1, ground_cut=0.15) & win
        if own.sum() < 3:
            continue
        Xw = X[own] @ Ri.T + Ci
        Xw2 = (Xw - Pi) @ Rwi @ Rwj.T + Pj                       # rigid motion of the labelled box
        Xj = (Xw2 - Cj) @ Rj
        uvj = seq.project(Xj) - np.array([x0, y0])
        okj = (Xj[:, 2] > 0.5) & (uvj[:, 0] >= 0) & (uvj[:, 0] <= w - 1) & (uvj[:, 1] >= 0) & (uvj[:, 1] <= h - 1)
        uva = uv[own]
        st_, ok = lidar_vehicle_stats(sc, uva, uvj, X[own, 2], okj)
        px = np.unique(np.round(uva[ok]).astype(int), axis=0) if ok.any() else np.zeros((0, 2), int)
        rec = dict(track=int(tid), label=oi["type"], state=state, disp_m=disp, s=float((Pj - Pi) @ b / (b @ b)),
                   speed_ratio=float(disp / max(np.linalg.norm(b), 1e-9)), occlusion=int(max(oi["occ"], oj["occ"])),
                   truncation=float(max(oi["trunc"], oj["trunc"])), pixels=px.tolist(), **st_)
        rec["n_lidar_px"] = int(len(px))
        vehicles.append(rec)
    return dict(depth=D, static=sm, vehicles=vehicles, scene=sc, baseline=float(np.linalg.norm(b)))

def kt_select_pairs(seq, rng, n_max):
    lo, hi = KT_DESIGN["pair_baseline_m"]
    out, last = [], -10**9
    frames = seq.frames
    starts = list(range(0, len(frames), 3)); rng.shuffle(starts)
    for si in starts:
        i = frames[si]
        if len(out) >= n_max:
            break
        if any(abs(i - p[0]) < 5 for p in out):
            continue
        Ri, Ci = seq.pose(i)
        for f in frames[si + 1: si + 1 + KT_DESIGN["max_gap_frames"]]:
            Rj, Cj = seq.pose(f)
            bl = float(np.linalg.norm(Cj - Ci))
            if bl >= lo:
                ang = float(np.degrees(np.arccos(np.clip((np.trace(Ri.T @ Rj) - 1) / 2, -1, 1))))
                if bl <= hi and ang <= KT_DESIGN["max_rot_deg"]:
                    out.append((i, f, bl, ang))
                break
    return out

def kt_crop(seq, i, j):
    h = min(seq.H, KT_DESIGN["crop_h_max"]); h -= h % 3
    w = h * 4 // 3
    y0 = (seq.H - h) // 2
    cand = [o for o in seq.labels.get(i, []) if o["type"] in KT_DESIGN["classes"] and o["occ"] <= KT_DESIGN["max_occlusion"]]
    best, best_x = -1, (seq.W - w) // 2
    for x0 in range(0, seq.W - w + 1, 10):
        n = sum(1 for o in cand if o["bbox"][0] >= x0 and o["bbox"][2] <= x0 + w - 1)
        if n > best or (n == best and abs(x0 - (seq.W - w) / 2) < abs(best_x - (seq.W - w) / 2)):
            best, best_x = n, x0
    return best_x, y0, w, h

_kt_root = kt_find_root()
KT_STAGE = Stage("kitti_gt", "kitti", dict(design=KT_DESIGN, n_pairs=RUN_CONFIG["n_kt_pairs_per_seq"],
                                           n_seq=RUN_CONFIG["n_kt_sequences"],
                                           prereg={k: v for k, v in PREREG.items() if k.startswith("kt_")}))
KT_GT_DIR = os.path.join(KT_DIR, "pairs")
with stage("kitti_gt"):
    if _kt_root is None and KT_STAGE.available() is None:
        KT_STATUS["note"] = ("no KITTI tracking folder attached (needs image_02, velodyne, label_02, calib and oxts; "
                             "attach them as a private Kaggle dataset)")
    elif not KT_STAGE.done():
        shutil.rmtree(KT_DIR, ignore_errors=True); os.makedirs(KT_GT_DIR)
        names = sorted(d for d in os.listdir(os.path.join(_kt_root, "label_02")) if d.endswith(".txt"))
        names = [os.path.splitext(d)[0] for d in names
                 if os.path.isdir(os.path.join(_kt_root, "image_02", os.path.splitext(d)[0]))][:RUN_CONFIG["n_kt_sequences"]]
        n_pair = 0
        for name in names:
            seq = KTSequence(_kt_root, name)
            if len(seq.frames) < 10:
                continue
            pairs = kt_select_pairs(seq, np.random.default_rng(seed_for("kt-pairs", name)), RUN_CONFIG["n_kt_pairs_per_seq"])
            test = [(p[0], p[1]) for p in pairs[:8]]
            errs = {}
            for conv in ("documented", "inverted"):
                seq.convention = conv
                errs[conv] = kt_a3(seq, test)
            conv = min(errs, key=lambda k: errs[k] if np.isfinite(errs[k]) else np.inf)
            seq.convention = conv
            passed = bool(np.isfinite(errs[conv]) and errs[conv] <= 0.05)
            KT_A3[name] = dict(n_frames=len(seq.frames), n_pairs=len(pairs), convention=conv,
                               warp_err_documented=errs["documented"], warp_err_inverted=errs["inverted"], passed=passed)
            if not passed:
                continue
            for (i, j, bl, ang) in pairs:
                x0, y0, w, h = kt_crop(seq, i, j)
                g = kt_pair_gt(seq, i, j, x0, y0, w, h)
                if not any(v["state"] == "moving" for v in g["vehicles"]):
                    continue
                d = os.path.join(KT_GT_DIR, f"{name}_{i:06d}_{j:06d}")
                os.makedirs(d)
                for k_, f_ in enumerate((i, j)):
                    Image.fromarray(seq.rgb(f_)[y0:y0 + h, x0:x0 + w]).save(os.path.join(d, f"rgb{k_}.png"))
                np.savez_compressed(os.path.join(d, "gt.npz"), depth=g["depth"].astype(np.float32), static=g["static"])
                with open(os.path.join(d, "meta.json"), "w") as f:
                    json.dump(dict(set="KT", index=n_pair, seq=name, frames=[int(i), int(j)], crop=[int(x0), int(y0), int(w), int(h)],
                                   baseline=bl, rot_deg=ang, scene=g["scene"], vehicles=g["vehicles"], dir=d), f)
                n_pair += 1
        with open(os.path.join(KT_DIR, "a3.json"), "w") as f:
            json.dump(KT_A3, f, indent=1)
        KT_STAGE.commit([KT_DIR])
if os.path.exists(os.path.join(KT_DIR, "a3.json")):
    KT_A3 = json.load(open(os.path.join(KT_DIR, "a3.json")))
    for d in sorted(glob.glob(os.path.join(KT_GT_DIR, "*", "meta.json"))):
        m_ = json.load(open(d)); m_["dir"] = os.path.dirname(d)
        KT_PAIR_META.append(m_)
    KT_PAIR_META.sort(key=lambda m_: m_["index"])
    KT_STATUS.update(available=bool(KT_PAIR_META), root=_kt_root, sequences=sorted({m_["seq"] for m_ in KT_PAIR_META}),
                     note="" if KT_PAIR_META else "no sequence passed A3 or no pair had a moving vehicle")

MOVERS_KT_ALL = lidar_mover_table(KT_PAIR_META, "KT") if KT_PAIR_META else pd.DataFrame()
if len(MOVERS_KT_ALL):
    MOVERS_KT_ALL, KT_EPS_NOISE, KT_TAU_PAR, _n_pk = lidar_regimes(MOVERS_KT_ALL)
    MOVERS_KT = MOVERS_KT_ALL[MOVERS_KT_ALL["state"] == "moving"].reset_index(drop=True)
    print(f"KITTI tracking: {len(KT_PAIR_META)} pairs from {len(KT_STATUS['sequences'])} sequences "
          f"({', '.join(KT_STATUS['sequences'])}); A3: " + "; ".join(f"{k} {'PASS' if v['passed'] else 'FAIL'} "
                                                                  f"({v['warp_err_' + v['convention']]:.3f}, {v['convention']})"
                                                                  for k, v in KT_A3.items()))
    print(f"label-noise floor: 90th percentile of |eps| over {_n_pk} parked vehicles = {KT_EPS_NOISE:.2f} px; "
          f"parallel regime on KITTI: |eps| < {KT_TAU_PAR:.2f} px")
    print(f"moving vehicles: {len(MOVERS_KT)}; regimes {MOVERS_KT['regime'].value_counts().to_dict()}; "
          f"measurable (>= {PREREG['kt_min_lidar_px']} LiDAR px) {int(MOVERS_KT['measurable'].sum())}, identifiable "
          f"{int((MOVERS_KT['measurable'] & MOVERS_KT['identifiable']).sum())}")
else:
    MOVERS_KT = pd.DataFrame(); KT_EPS_NOISE = KT_TAU_PAR = np.nan
    print("KITTI tracking is not used in this run:", KT_STATUS["note"] or "no data")
release_memory(); mem_report("KITTI ground truth");

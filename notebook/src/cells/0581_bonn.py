# PLACEHOLDER
# 3.12 Bonn RGB-D Dynamic (v6.5, corrected in v6.6, exploratory, D56 and D62). One checkpointed stage ("bonn_gt") downloads
#      four sequences, checks the pose convention (A3), selects frame pairs, re-estimates the relative pose of every pair
#      from its static correspondences (v6.6), finds moving regions and people, and stores for every pair the undistorted
#      images, the measured depth, the static mask and the objects with their ground truth. The raw sequences are deleted.
#      v6.6 object rule: pixels hidden in frame j are not tested; a moving region counts only if the static prediction
#      fails photometrically on it (flow errors on repeated texture and occlusion bands are rejected), it is thicker than
#      a 5 px disc, smaller than 20% of the image and not flat on the floor (a moving shadow); the floor is the lowest
#      well-supported horizontal level.
import zipfile as _zipfile

BONN_DIR = os.path.join(SCRATCH_DIR, "bonn")
BONN_GT_DIR = os.path.join(BONN_DIR, "pairs")
BONN_RAW = os.path.join(DATA_DIR, "bonn_raw")
BONN_PAIR_META, BONN_A3 = [], {}
BONN_STATUS = dict(available=False, note="", sequences=[], download={}, person_model="", noise_floor_px=np.nan)
MOVERS_BONN = pd.DataFrame()
BONN_TAU_PAR = np.nan


def _se3(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


class BonnSequence(TUMSequence):
    """TUM-format sequence with the Bonn camera: undistorted images and depth, and the marker-to-camera calibration."""
    CONVENTIONS = ("documented", "inverse calibration", "raw file")

    def __init__(self, name, root):
        super().__init__(name, root)
        k = BONN_DESIGN["intrinsics"]
        self.cam = PinholeCamera(**k)
        K = np.array([[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]])
        self._maps = cv2.initUndistortRectifyMap(K, np.asarray(BONN_DESIGN["distortion"], np.float64), None, K,
                                                 (k["W"], k["H"]), cv2.CV_32FC1)
        Tm = np.asarray(BONN_DESIGN["T_m"], np.float64)
        U, S, Vt = np.linalg.svd(Tm[:3, :3])                 # T_m = [scale * rotation | t]: split the two
        self._calib = dict(Tros=np.asarray(BONN_DESIGN["T_ros"], np.float64), Rm=U @ Vt, tm=Tm[:3, 3], scale=float(S.mean()))
        self._calib["Tros_inv"] = np.linalg.inv(self._calib["Tros"])
        self.convention = "documented"

    def rgb(self, i):
        im = np.asarray(Image.open(self.frames[i]["rgb"]).convert("RGB"))
        return cv2.remap(im, self._maps[0], self._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def raw_rgb_path(self, i):
        return self.frames[i]["rgb"]

    def depth(self, i):
        d = np.asarray(Image.open(self.frames[i]["depth"])).astype(np.float32) / BONN_DESIGN["depth_scale"]
        d[d <= 0] = np.nan
        return cv2.remap(d, self._maps[0], self._maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    def undistort_mask(self, m):
        return cv2.remap(m.astype(np.uint8), self._maps[0], self._maps[1], cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

    def pose(self, i):
        # camera -> world rotation and camera centre; the world frame is the motion-capture frame divided by T_m's scale
        R, C = self.frames[i]["R"], self.frames[i]["C"]
        if self.convention == "raw file":
            return R, C
        c = self._calib
        A = c["Tros_inv"] @ _se3(R, C) @ c["Tros"]
        RA, tA = A[:3, :3], A[:3, 3]
        if self.convention == "documented":                  # world <- camera = A T_m
            return RA @ c["Rm"], (RA @ c["tm"] + tA) / c["scale"]
        return RA @ c["Rm"].T, c["scale"] * tA - RA @ c["Rm"].T @ c["tm"]     # world <- camera = A T_m^-1


def bonn_download(name, spec):
    """Download and extract one sequence into its own folder (paths checked; the top-level folder is stripped)."""
    seq_dir = os.path.join(BONN_RAW, name)
    flag = os.path.join(seq_dir, ".extracted")
    if os.path.exists(flag):
        return seq_dir, "already extracted", 0.0
    os.makedirs(BONN_RAW, exist_ok=True)
    zpath = os.path.join(BONN_RAW, spec["file"])
    ok, msg = (True, "cached") if os.path.exists(zpath) else download_file(BONN_DESIGN["base_url"] + spec["file"], zpath)
    if not ok:
        raise IOError(f"download failed: {msg}")
    mib = os.path.getsize(zpath) / 2**20
    shutil.rmtree(seq_dir, ignore_errors=True); os.makedirs(seq_dir)
    root = os.path.abspath(seq_dir)
    with _zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            parts = [p_ for p_ in info.filename.replace("\\", "/").split("/") if p_]
            if info.is_dir() or not parts or any(p_ in ("..", ".") for p_ in parts) or info.filename.startswith("/"):
                continue
            top_level = parts[0] not in ("rgb", "depth") and not (len(parts) == 1 and parts[0].endswith(".txt"))
            rel = parts[1:] if top_level else parts            # strip the top-level folder, if there is one
            if not rel:
                continue
            if rel[0] not in ("rgb", "depth") and not (len(rel) == 1 and rel[0].endswith(".txt")):
                continue                                       # only images, depth and the three lists are needed
            dst = os.path.normpath(os.path.join(root, *rel))
            if not dst.startswith(root + os.sep):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(info) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
    os.remove(zpath)
    for f_ in ("rgb.txt", "depth.txt", "groundtruth.txt"):
        if not os.path.exists(os.path.join(seq_dir, f_)):
            raise IOError(f"{f_} missing in the archive")
    open(flag, "w").write("ok")
    return seq_dir, msg, mib


def bonn_a3(seq, rng):
    """Pose convention and the A3 test: median relative disagreement of measured depth and depth warped with the pose."""
    pairs = []
    for _ in range(400):
        i = int(rng.integers(0, len(seq) - 1))
        for j in range(i + 1, min(len(seq), i + 90)):
            b, ang = baseline_and_angle(seq, i, j)
            if b >= 0.08:
                if ang < 20:
                    pairs.append((i, j))
                break
        if len(pairs) >= 16:
            break
    errs = {}
    for conv in BonnSequence.CONVENTIONS:
        seq.convention = conv
        errs[conv] = float(np.nanmedian([depth_warp_error(seq, i, j) for i, j in pairs])) if pairs else np.nan
    best = min(errs, key=lambda k_: errs[k_] if np.isfinite(errs[k_]) else np.inf)
    seq.convention = best
    return dict(n_frames=len(seq), n_test_pairs=len(pairs), **{f"warp_err_{k_.replace(' ', '_')}": v for k_, v in errs.items()},
                convention=best, median_rel_depth_err=errs[best], passed=bool(np.isfinite(errs[best]) and errs[best] <= 0.05))


_PERSON = {"model": None, "note": ""}


def _person_model():
    if _PERSON["model"] is None and not _PERSON["note"].startswith("unavailable"):
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
            _w = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            _PERSON["model"] = maskrcnn_resnet50_fpn_v2(weights=_w, progress=False).eval().to(device)
            _PERSON["note"] = f"Mask R-CNN ResNet-50 FPN v2 (torchvision {_w.name})"
        except Exception as e:
            _PERSON["note"] = f"unavailable: {type(e).__name__}: {str(e)[:160]}"
    return _PERSON["model"]


@torch.no_grad()
def bonn_person_masks(seq, frames):
    """People in the undistorted images of the given frames (COCO person, score >= person_score_min)."""
    model = _person_model()
    if model is None:
        raise RuntimeError("no person segmentation: " + _PERSON["note"])
    out = []
    for f_ in frames:
        t = torch.from_numpy(seq.rgb(f_).copy()).permute(2, 0, 1).float().div(255.0).to(device)
        r = model([t])[0]
        keep = (r["labels"] == 1) & (r["scores"] >= BONN_DESIGN["person_score_min"])
        m = (r["masks"][keep, 0] >= 0.5).any(0) if bool(keep.any()) else torch.zeros(t.shape[1:], dtype=torch.bool)
        out.append(m.cpu().numpy())
    return out


def _release_person_model():
    _PERSON["model"] = None
    release_memory()


def _angle_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def bonn_refine_pose(cam, di, Ri, Ci, Rj, Cj, f_ij, f_ji, excl, rng):
    """Relative pose i -> j from static correspondences (v6.6, D62): RANSAC PnP of frame i's measured 3-D points against
    their optical-flow positions in frame j, started from the motion-capture pose and polished on the inliers. Returns
    (Rj, Cj, record); record["refined"] is False when the fit is not trustworthy, and the pair is then left out (with the
    motion-capture pose alone, static pixels miss their rigid flow by about 2 px and are taken for motion)."""
    D = BONN_DESIGN; H, W = di.shape
    vv, uu = np.mgrid[0:H:2, 0:W:2]
    p = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float64)
    z = di[vv, uu].ravel().astype(np.float64)
    f = f_ij[vv, uu].reshape(-1, 2).astype(np.float64)
    q = p + f
    with np.errstate(invalid="ignore"):
        fb = np.linalg.norm(f + sample_map(f_ji, q), axis=1)
        ok = (np.isfinite(z) & (z > 0.3) & (z < 6.0) & np.isfinite(fb) & (fb < 0.5) & ~excl[vv, uu].ravel()
              & (q[:, 0] >= 0) & (q[:, 0] <= W - 1) & (q[:, 1] >= 0) & (q[:, 1] <= H - 1))
    rec = dict(refined=False, reason="", n_points=int(ok.sum()), inlier_frac=np.nan, med_err_px=np.nan,
               mocap_med_err_px=np.nan, rot_change_deg=np.nan, trans_change_m=np.nan)
    if ok.sum() < 300:
        rec["reason"] = "too few static correspondences"
        return Rj, Cj, rec
    idx = rng.choice(np.nonzero(ok)[0], min(D["pnp_points"], int(ok.sum())), replace=False)
    X = cam.rays(p[idx, 0], p[idx, 1]) * z[idx][:, None]
    img = q[idx]
    R21, t21 = relative_pose(Ri, Ci, Rj, Cj)

    def _err(Rr, tt):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.linalg.norm(cam.project(X @ Rr.T + tt) - img, axis=1)
    rec["mocap_med_err_px"] = float(np.nanmedian(_err(R21, t21)))
    try:
        okp, rvec, tvec, inl = cv2.solvePnPRansac(X, img, cam.K, None, rvec=cv2.Rodrigues(R21)[0].copy(),
                                                  tvec=t21.reshape(3, 1).copy(), useExtrinsicGuess=True,
                                                  iterationsCount=300, reprojectionError=D["pnp_reproj_px"],
                                                  confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if okp and inl is not None and len(inl) >= 100:
            rvec, tvec = cv2.solvePnPRefineLM(X[inl.ravel()], img[inl.ravel()], cam.K, None, rvec, tvec)
    except cv2.error as e:
        rec["reason"] = f"PnP failed: {str(e)[:80]}"
        return Rj, Cj, rec
    if not okp or inl is None or len(inl) < 100:
        rec["reason"] = "no consistent pose"
        return Rj, Cj, rec
    R21n, t21n = cv2.Rodrigues(rvec)[0], tvec.ravel()
    e = _err(R21n, t21n)
    good_px = e < D["pnp_reproj_px"]
    rec.update(inlier_frac=float(good_px.mean()), med_err_px=float(np.median(e[good_px])) if good_px.any() else np.nan,
               rot_change_deg=_angle_deg(R21, R21n), trans_change_m=float(np.linalg.norm(t21n - t21)))
    if not (rec["inlier_frac"] >= D["pnp_min_inlier_frac"] and rec["med_err_px"] <= D["pnp_max_med_err_px"]
            and rec["rot_change_deg"] <= D["pnp_max_rot_change_deg"] and rec["trans_change_m"] <= D["pnp_max_trans_change_m"]):
        rec["reason"] = "fit outside the limits"
        return Rj, Cj, rec
    rec["refined"] = True
    Rjn = Ri @ R21n.T
    return Rjn, Ci - Rjn @ t21n, rec


def bonn_floor_plane(depth, cam, R, C, up, rng):
    """The floor (v6.6): the lowest horizontal level under the camera with enough support, refined by RANSAC; a table top is
    not the floor. None if no level has the support or the camera height above it is implausible for a hand-held camera."""
    D = BONN_DESIGN
    if up is None:
        return None
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:4, 0:W:4]
    z = depth[vv, uu]
    ok = np.isfinite(z) & (z > 0.3) & (z < 6.0)
    if ok.sum() < 400:
        return None
    Xw = C + (cam.rays(uu[ok].astype(np.float64), vv[ok].astype(np.float64)) * z[ok][:, None]) @ R.T
    h = (Xw - C) @ up
    lo, hi = -D["floor_cam_height_m"][1], -D["floor_cam_height_m"][0]
    need = max(80, D["floor_min_frac"] * len(Xw))
    edges = np.arange(lo, hi + 0.02, 0.02)
    cnt = np.convolve(np.histogram(h, edges)[0], np.ones(3), mode="same")
    levels = np.nonzero(cnt >= need)[0]
    if not len(levels):
        return None
    h0 = (edges[levels[0]] + edges[levels[0] + 1]) / 2          # the lowest well-supported level
    P = Xw[np.abs(h - h0) < 0.06]
    if len(P) < 3:
        return None
    best = None
    for _ in range(200):
        a = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(a[1] - a[0], a[2] - a[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        n = n if n @ up >= 0 else -n
        if np.degrees(np.arccos(np.clip(n @ up, -1, 1))) > D["floor_max_tilt_deg"]:
            continue
        inl = np.abs((Xw - a[0]) @ n) < 0.03
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < need:
        return None
    Q = Xw[best]; o = Q.mean(0)
    n = np.linalg.svd(Q - o, full_matrices=False)[2][-1]
    n = n if n @ up >= 0 else -n
    ch = float((C - o) @ n)
    if not (D["floor_cam_height_m"][0] <= ch <= D["floor_cam_height_m"][1]):
        return None
    return dict(n=n, origin=o, height=ch, inlier_frac=float(best.mean()))


def _window_min_max(a, valid, r):
    """Minimum and maximum of a over the (2r+1) x (2r+1) window of every pixel, ignoring invalid pixels (+-inf if none)."""
    k_ = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return (cv2.erode(np.where(valid, a, np.inf).astype(np.float32), k_),
            cv2.dilate(np.where(valid, a, -np.inf).astype(np.float32), k_))


def _warp_gray(g, q, H, W):
    """Grey image g sampled at the positions q (H*W, 2); a flag for the positions inside the image."""
    q = np.asarray(q, np.float32).reshape(H, W, 2)
    ok = np.isfinite(q).all(-1) & (q[..., 0] >= 0) & (q[..., 0] <= W - 1) & (q[..., 1] >= 0) & (q[..., 1] <= H - 1)
    q = np.where(ok[..., None], q, -1.0).astype(np.float32)
    return cv2.remap(g, q[..., 0], q[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), ok


def _zncc(a, b, valid, r):
    """Zero-mean normalised cross-correlation of a and b over the (2r+1)^2 window of every pixel, the standard deviation of
    a there and a flag for windows that are at least 80% valid."""
    k_ = (2 * r + 1, 2 * r + 1)
    n = cv2.boxFilter(valid.astype(np.float32), -1, k_, normalize=False)
    a0 = np.where(valid, a, 0).astype(np.float32); b0 = np.where(valid, b, 0).astype(np.float32)
    nn = np.maximum(n, 1.0)
    ma = cv2.boxFilter(a0, -1, k_, normalize=False) / nn; mb = cv2.boxFilter(b0, -1, k_, normalize=False) / nn
    va = cv2.boxFilter(a0 * a0, -1, k_, normalize=False) / nn - ma ** 2
    vb = cv2.boxFilter(b0 * b0, -1, k_, normalize=False) / nn - mb ** 2
    cv_ = cv2.boxFilter(a0 * b0, -1, k_, normalize=False) / nn - ma * mb
    return (cv_ / np.sqrt(np.maximum(va, 1e-3) * np.maximum(vb, 1e-3)), np.sqrt(np.maximum(va, 0.0)),
            n >= 0.8 * k_[0] * k_[1])


def bonn_pair_gt(seq, i, j, Rj, Cj, f_ij, f_ji, person, plane, kind, rng, img_i, img_j):
    """Moving regions, people, objects and their ground truth for one pair (frames i -> j), the static mask, the static
    scale check and the epipolar offsets of static patches (the flow-noise floor)."""
    D = BONN_DESIGN
    cam = seq.cam; H, W = cam.H, cam.W
    di, dj = seq.depth(i), seq.depth(j)
    Ri, Ci = seq.pose(i)                                   # (Rj, Cj): the pose of frame j re-estimated from the pair
    vv, uu = np.mgrid[0:H, 0:W]
    p = np.stack([uu, vv], -1).reshape(-1, 2).astype(np.float64)
    z = di.reshape(-1).astype(np.float64)
    valid = np.isfinite(z)
    X = cam.rays(p[:, 0], p[:, 1]) * np.where(valid, z, 1.0)[:, None]
    Xj = (X @ Ri.T + Ci - Cj) @ Rj
    with np.errstate(divide="ignore", invalid="ignore"):
        p_rig = cam.project(Xj)
    p_obs = p + f_ij.reshape(-1, 2)
    zj_obs = sample_map(np.nan_to_num(dj, nan=0.0), p_rig)
    zj_obs[zj_obs <= 0] = np.nan
    back = sample_map(f_ji, p_obs)
    fb_ok = np.linalg.norm(f_ij.reshape(-1, 2) + back, axis=1) < 1.0 + 0.05 * np.linalg.norm(f_ij.reshape(-1, 2), axis=1)
    flow_res = np.linalg.norm(p_obs - p_rig, axis=1)
    with np.errstate(invalid="ignore"):
        depth_bad = np.abs(zj_obs - Xj[:, 2]) > np.maximum(0.10, 0.08 * Xj[:, 2])
    vimg = np.isfinite(di).astype(np.float32)
    nvalid = cv2.boxFilter(vimg, -1, (3, 3), normalize=False)
    gy, gx = np.gradient(np.nan_to_num(di, nan=0.0))
    edge = (((np.hypot(gx, gy) > 0.08) & (nvalid >= 9)) | (nvalid < 7)).reshape(-1)
    inside = np.isfinite(p_obs).all(1) & (p_obs[:, 0] >= 0) & (p_obs[:, 0] <= W - 1) & (p_obs[:, 1] >= 0) & (p_obs[:, 1] <= H - 1)
    base = valid & fb_ok & ~edge & inside
    # hidden in frame j: everything observed near the predicted position is nearer than the point (occluded, not moving)
    _, dmax = _window_min_max(np.nan_to_num(dj, nan=0.0), np.isfinite(dj), D["occl_window_px"])
    zmax_j = sample_map(np.where(np.isfinite(dmax), dmax, 0.0).astype(np.float32), p_rig)
    with np.errstate(invalid="ignore"):
        occluded = (zmax_j > 0) & (zmax_j < Xj[:, 2] - np.maximum(0.10, 0.08 * Xj[:, 2]))
    dyn = base & ~occluded & np.isfinite(zj_obs) & (depth_bad | (flow_res > 3.0))
    dyn_img = cv2.morphologyEx(dyn.reshape(H, W).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(dyn_img, connectivity=8)
    keep = np.zeros(n_cc, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= BONN_DESIGN["min_component_px"]
    dyn_m = keep[lab]
    # epipolar offset, static-hypothesis depth and translational parallax of every pixel with a correspondence
    R21, t21 = relative_pose(Ri, Ci, Rj, Cj)
    F = fundamental_matrix(cam.K, cam.K, R21, t21)
    eps = np.full(H * W, np.nan); ztri = np.full(H * W, np.nan); par = np.full(H * W, np.nan)
    idx = np.nonzero(base)[0]
    eps[idx] = np.abs(epipolar_offset(p[idx], p_obs[idx], F))
    ztri[idx], _ = triangulate_depth(p[idx], p_obs[idx], cam.K, cam.K, R21, t21)
    par[idx] = translational_parallax(p[idx], p_obs[idx], cam.K, cam.K, R21)
    eps, ztri, par = eps.reshape(H, W), ztri.reshape(H, W), par.reshape(H, W)
    # static pixels: valid correspondence, depth in the real-data range, away from moving regions and people
    lo, hi = PREREG["static_depth_range_real"]
    away = ~cv2.dilate((dyn_m | person).astype(np.uint8), np.ones((19, 19), np.uint8)).astype(bool)
    static = base.reshape(H, W) & away & (di >= lo) & (di <= hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(ztri / di)
    sv = lr[static & (ztri > 0) & np.isfinite(lr)]
    scale_log = float(np.median(sv)) if len(sv) >= 500 else np.nan
    # flow-noise floor: median |eps| of static 40 x 40 patches (>= 80% static pixels)
    P_ = BONN_DESIGN["static_patch_px"]
    patches = []
    for _ in range(4 * BONN_DESIGN["n_static_patches"]):
        a, b_ = int(rng.integers(0, W - P_)), int(rng.integers(0, H - P_))
        sm = static[b_:b_ + P_, a:a + P_]
        if sm.mean() >= 0.8:
            patches.append(float(np.median(eps[b_:b_ + P_, a:a + P_][sm])))
        if len(patches) >= BONN_DESIGN["n_static_patches"]:
            break
    # photometric check of the static prediction: frame j warped with the rigid flow and with the measured flow, compared
    # with frame i by 7 x 7 ZNCC on textured windows
    gi = img_i.astype(np.float32).mean(-1); gj = img_j.astype(np.float32).mean(-1)
    w_rig, ok_rig = _warp_gray(gj, p_rig, H, W); w_obs, ok_obs = _warp_gray(gj, p_obs, H, W)
    ok_rig &= valid.reshape(H, W)                          # the rigid prediction needs a measured depth
    z_rig, sd_i, full_r = _zncc(gi, w_rig, ok_rig, D["photo_window_px"])
    z_obs, _, full_o = _zncc(gi, w_obs, ok_obs, D["photo_window_px"])
    textured = (sd_i >= D["photo_min_texture"]) & full_r & full_o
    # objects: every moving region split into its part on people and its part off them; an object larger than
    # max_object_frac of the image or wider/taller than max_object_extent of it is not an object, nor is a region that the
    # static prediction explains photometrically, one thinner than a disc of min_object_radius_px, or one lying flat on
    # the floor (v6.6)
    person_d = cv2.dilate(person.astype(np.uint8), np.ones((2 * D["person_dilate_px"] + 1,) * 2, np.uint8)).astype(bool)
    flag_frac = float((dyn_m & ~person_d).mean())          # share of the image flagged as moving outside people (QC)
    objs, n_large, n_photo, n_thin, n_flat = [], 0, 0, 0, 0
    ring = np.ones((2 * D["photo_window_px"] + 1,) * 2, np.uint8)
    for comp in range(1, n_cc):
        if not keep[comp]:
            continue
        cm = lab == comp
        for part, knd in ((cm & ~person_d, kind), (cm & person_d, "person")):
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8), connectivity=8)
            for c2 in range(1, n2):
                reg = lab2 == c2
                if knd != "person" and (st2[c2, cv2.CC_STAT_AREA] > D["max_object_frac"] * H * W
                                        or st2[c2, cv2.CC_STAT_WIDTH] > D["max_object_extent"] * W
                                        or st2[c2, cv2.CC_STAT_HEIGHT] > D["max_object_extent"] * H):
                    n_large += 1
                    continue
                px_m = erode_mask(reg, PREREG["erosion_px"]) & np.isfinite(eps) & np.isfinite(di)
                if px_m.sum() < PREREG["bonn_min_object_px"]:
                    continue
                radius = float(cv2.distanceTransform(np.pad(reg, 1).astype(np.uint8), cv2.DIST_L2, 5).max())
                if radius < D["min_object_radius_px"]:
                    n_thin += 1
                    continue
                tx = cv2.dilate(reg.astype(np.uint8), ring).astype(bool) & textured
                zs = float(np.median(z_rig[tx])) if tx.sum() >= D["photo_min_px"] else np.nan
                sup = float(np.median(z_obs[tx] - z_rig[tx])) if tx.sum() >= D["photo_min_px"] else np.nan
                if not (np.isfinite(zs) and zs <= D["photo_max_zncc_static"] and sup >= D["photo_min_support"]):
                    n_photo += 1
                    continue
                vv2, uu2 = np.nonzero(reg & np.isfinite(di))
                h_reg = None
                if plane is not None and len(vv2):
                    Xw = Ci + (cam.rays(uu2.astype(np.float64), vv2.astype(np.float64)) * di[vv2, uu2][:, None]) @ Ri.T
                    h_reg = (Xw - plane["origin"]) @ plane["n"]
                    if knd != "person" and np.percentile(h_reg, 90) < D["min_object_height_m"]:
                        n_flat += 1                        # flat on the floor: a moving shadow or reflection
                        continue
                zz, ee, tt, pp = di[px_m].astype(np.float64), eps[px_m], ztri[px_m], par[px_m]
                pos = tt > 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    xr = float(np.median(np.log(tt[pos] / zz[pos]))) if pos.mean() >= 0.5 else np.nan
                v_, u_ = np.nonzero(reg)
                rec = dict(kind=knd, n_px=int(px_m.sum()), area_px=int(reg.sum()), Z_true=float(np.median(zz)),
                           eps_med=float(np.median(ee)), parallax_med=float(np.median(pp)), x_geo_raw=xr,
                           x_geo=xr - scale_log if np.isfinite(xr) and np.isfinite(scale_log) else np.nan,
                           no_static_solution=bool(pos.mean() < 0.5),
                           bbox=[int(u_.min()), int(v_.min()), int(u_.max()), int(v_.max())],
                           touches_bottom=bool(v_.max() >= H - 3), height_above_floor=np.nan, floor="undetermined",
                           zncc_static=zs, photo_support=sup, radius_px=radius)
                if h_reg is not None and not rec["touches_bottom"]:
                    h_low = float(np.percentile(h_reg, 5))
                    rec["height_above_floor"] = h_low
                    rec["floor"] = ("on the floor" if h_low <= PREREG["bonn_floor_contact_m"]
                                    else "off the floor" if h_low >= PREREG["bonn_floor_clear_m"] else "undetermined")
                yy, xx = np.nonzero(px_m)
                rec["pixels"] = np.stack([xx, yy], 1).tolist()
                objs.append(rec)
    return dict(depth=di, static=static, objects=objs, scale_log=scale_log, static_patch_eps=patches,
                n_moving_px=int(dyn_m.sum()), n_person_px=int(person.sum()), flag_frac=flag_frac, n_large=n_large,
                n_photo=n_photo, n_thin=n_thin, n_flat=n_flat, occluded_frac=float(occluded.mean()))


def bonn_object_table(metas):
    rows = []
    for m_ in metas:
        for k, o in enumerate(m_["objects"]):
            r = {("gt_" + a if a not in ("kind", "floor") else a): b for a, b in o.items() if a not in ("pixels", "bbox")}
            r.update(set="BONN", scene=m_["index"], seq=m_["seq"], mover=k, uid=f"BONN-{m_['index']:04d}-{k}", cls=o["kind"],
                     cam_kind="hand-held", motion_type="object", s=np.nan, track=f"{m_['seq']}/{m_['frames'][0]}")
            rows.append(r)
    return pd.DataFrame(rows)


def bonn_regimes(df, metas):
    # the parallel threshold is the flow-noise floor: 90th percentile of the median |eps| of static patches, >= 1 px
    pe = np.concatenate([np.asarray(m_.get("static_patch_eps", []), float) for m_ in metas]) if metas else np.zeros(0)
    noise = float(np.nanquantile(pe, 0.9)) if np.isfinite(pe).sum() >= 20 else np.nan
    tau = max(PREREG["eps_parallel_max"], noise if np.isfinite(noise) else PREREG["eps_parallel_max"])
    df["regime"] = np.where(df["gt_eps_med"] < tau, "parallel",
                            np.where(df["gt_eps_med"] >= max(PREREG["eps_perp_min"], 2 * tau), "perpendicular", "intermediate"))
    df["x_par"] = np.nan
    df["measurable"] = df["gt_n_px"] >= PREREG["bonn_min_object_px"]
    df["identifiable"] = df["gt_parallax_med"] >= PREREG["parallax_min_px"]
    df["gt_n_eroded"] = df["gt_n_px"]
    return df, noise, tau


BONN_STAGE = Stage("bonn_gt", "bonn", dict(design={k: v for k, v in BONN_DESIGN.items() if k != "base_url"},
                                           n_pairs=RUN_CONFIG["n_bonn_pairs_per_seq"], flow=FLOW_PREFERRED,
                                           regimes=[PREREG["eps_parallel_max"], PREREG["eps_perp_min"]],
                                           erosion=PREREG["erosion_px"], static_range=PREREG["static_depth_range_real"],
                                           prereg={k: v for k, v in PREREG.items() if k.startswith("bonn_")},
                                           plane=_jsonable(RABL_DESIGN)))
with stage("bonn_gt"):
    if not RUN_CONFIG.get("use_bonn", True):
        BONN_STATUS["note"] = "switched off in RUN_CONFIG"
    elif not BONN_STAGE.done():
        try:
            shutil.rmtree(BONN_DIR, ignore_errors=True); os.makedirs(BONN_GT_DIR)
            if FLOW.get("model") is None and str(FLOW.get("method") or "").startswith("RAFT"):
                FLOW["method"] = None                    # RAFT was released after Section 3.7: load it again
            _a3, _dl, _notes, _n, _refine = {}, {}, {}, 0, []
            for name, spec in BONN_DESIGN["sequences"].items():
                try:
                    seq_dir, msg, mib = bonn_download(name, spec)
                    _dl[name] = dict(note=msg, MiB=round(mib, 1))
                    seq = BonnSequence(name, seq_dir)
                    rec = bonn_a3(seq, np.random.default_rng(seed_for("bonn-a3", name)))
                    _a3[name] = rec
                    if not rec["passed"]:
                        _notes[name] = f"A3 failed ({rec['median_rel_depth_err']:.3f} with the {rec['convention']} convention)"
                        continue
                    vert = sequence_vertical(seq)
                    pairs = select_dynamic_pairs(seq, RUN_CONFIG["n_bonn_pairs_per_seq"], np.random.default_rng(seed_for("bonn-pairs", name)),
                                                 b_range=BONN_DESIGN["pair_baseline_m"], max_angle=BONN_DESIGN["max_rot_deg"])
                    imgs = {k_: seq.rgb(k_) for k_ in sorted({x for p_ in pairs for x in p_[:2]})}
                    fw = compute_flows([(imgs[i], imgs[j]) for i, j, _, _ in pairs],
                                       [os.path.join(FLOW_DIR, "bonn", f"{name}_{i}_{j}.npz") for i, j, _, _ in pairs])
                    bw = compute_flows([(imgs[j], imgs[i]) for i, j, _, _ in pairs],
                                       [os.path.join(FLOW_DIR, "bonn", f"{name}_{j}_{i}.npz") for i, j, _, _ in pairs])
                    persons = bonn_person_masks(seq, [i for i, _, _, _ in pairs])
                    rng_b = np.random.default_rng(seed_for("bonn-gt", name))
                    for (i, j, bl, ang), f1, f2, pm in zip(pairs, fw, bw, persons):
                        Ri, Ci = seq.pose(i); Rj0, Cj0 = seq.pose(j)
                        di = seq.depth(i)
                        excl = cv2.dilate(pm.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
                        Rj, Cj, prec = bonn_refine_pose(seq.cam, di, Ri, Ci, Rj0, Cj0, f1, f2, excl,
                                                        np.random.default_rng(seed_for("bonn-pnp", name, i)))
                        _refine.append(dict(seq=name, i=int(i), j=int(j), **prec))
                        if not prec["refined"]:
                            continue                         # no trustworthy relative pose: the pair is left out
                        plane = bonn_floor_plane(di, seq.cam, Ri, Ci, vert, np.random.default_rng(seed_for("bonn-plane", name, i)))
                        g = bonn_pair_gt(seq, i, j, Rj, Cj, f1, f2, pm, plane, spec["object"], rng_b, imgs[i], imgs[j])
                        _refine[-1].update(flag_frac=g["flag_frac"], n_large=g["n_large"], n_photo=g["n_photo"],
                                           n_thin=g["n_thin"], n_flat=g["n_flat"], occluded_frac=g["occluded_frac"],
                                           n_objects=len(g["objects"]),
                                           floor_found=plane is not None)
                        if not g["objects"]:
                            continue
                        d = os.path.join(BONN_GT_DIR, f"{name}_{i:05d}_{j:05d}")
                        os.makedirs(d)
                        Image.fromarray(imgs[i]).save(os.path.join(d, "rgb0.png"))
                        Image.fromarray(imgs[j]).save(os.path.join(d, "rgb1.png"))
                        np.savez_compressed(os.path.join(d, "gt.npz"), depth=g["depth"].astype(np.float32), static=g["static"])
                        scene = dict(cams=[dict(**seq.cam.to_dict(), R=Ri.tolist(), C=Ci.tolist()),
                                           dict(**seq.cam.to_dict(), R=Rj.tolist(), C=Cj.tolist())])
                        with open(os.path.join(d, "meta.json"), "w") as f:
                            json.dump(dict(set="BONN", index=_n, seq=name, frames=[int(i), int(j)], baseline=bl, rot_deg=ang,
                                           scene=scene, objects=g["objects"], scale_log=g["scale_log"],
                                           static_patch_eps=g["static_patch_eps"], n_moving_px=g["n_moving_px"],
                                           n_person_px=g["n_person_px"], flag_frac=g["flag_frac"], pose_refine=prec,
                                           plane=None if plane is None else dict(height=plane["height"], inlier_frac=plane["inlier_frac"]),
                                           dir=d), f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
                        _n += 1
                    Image.fromarray(seq.rgb(len(seq) // 2)).save(os.path.join(BONN_DIR, f"thumb_{name}.png"))
                    del imgs, fw, bw, persons
                except Exception as e:                      # one unreadable sequence never stops the stage
                    _notes[name] = f"{type(e).__name__}: {str(e)[:200]}"
                finally:
                    shutil.rmtree(os.path.join(BONN_RAW, name), ignore_errors=True)
                    shutil.rmtree(os.path.join(FLOW_DIR, "bonn"), ignore_errors=True)
            _release_person_model()
            FLOW["model"] = None
            pd.DataFrame(_refine).to_csv(os.path.join(BONN_DIR, "pose_refine.csv"), index=False)
            with open(os.path.join(BONN_DIR, "a3.json"), "w") as f:
                json.dump(dict(a3=_a3, download=_dl, notes=_notes, person_model=_PERSON["note"], flow=FLOW.get("method"),
                               n_pairs=_n), f, indent=1, default=str)
            if _n == 0:
                raise IOError("no Bonn pair with an object: " + "; ".join(f"{k}: {v}" for k, v in list(_notes.items())[:4]))
            BONN_STAGE.commit([BONN_DIR])
        except Exception as e:
            BONN_STATUS["note"] = f"{type(e).__name__}: {str(e)[:300]}"
            shutil.rmtree(BONN_DIR, ignore_errors=True)
            _release_person_model()

if os.path.exists(os.path.join(BONN_DIR, "a3.json")):
    _brec = json.load(open(os.path.join(BONN_DIR, "a3.json")))
    BONN_A3 = _brec["a3"]
    BONN_STATUS.update(download=_brec.get("download", {}), person_model=_brec.get("person_model", ""), notes=_brec.get("notes", {}))
    for _p in sorted(glob.glob(os.path.join(BONN_GT_DIR, "*", "meta.json"))):
        _m = json.load(open(_p)); _m["dir"] = os.path.dirname(_p)
        BONN_PAIR_META.append(_m)
    BONN_PAIR_META.sort(key=lambda m_: m_["index"])
    BONN_STATUS.update(available=bool(BONN_PAIR_META), sequences=sorted({m_["seq"] for m_ in BONN_PAIR_META}))
BONN_QC = {}
if os.path.exists(os.path.join(BONN_DIR, "pose_refine.csv")):
    _pr = pd.read_csv(os.path.join(BONN_DIR, "pose_refine.csv"))
    _ok = _pr[_pr["refined"].astype(bool)]
    BONN_QC = dict(n_pairs_tried=len(_pr), n_refined=len(_ok), refined_share=float(len(_ok) / max(len(_pr), 1)),
                   median_inlier_frac=float(_ok["inlier_frac"].median()) if len(_ok) else np.nan,
                   median_err_px=float(_ok["med_err_px"].median()) if len(_ok) else np.nan,
                   median_mocap_err_px=float(_pr["mocap_med_err_px"].median()),
                   median_rot_change_deg=float(_ok["rot_change_deg"].median()) if len(_ok) else np.nan,
                   median_trans_change_m=float(_ok["trans_change_m"].median()) if len(_ok) else np.nan,
                   median_flag_frac=float(_ok["flag_frac"].median()) if len(_ok) and "flag_frac" in _ok else np.nan,
                   p90_flag_frac=float(_ok["flag_frac"].quantile(0.9)) if len(_ok) and "flag_frac" in _ok else np.nan,
                   floor_found_share=float(_ok["floor_found"].astype(bool).mean()) if len(_ok) and "floor_found" in _ok else np.nan,
                   n_rejected_large=int(_ok["n_large"].sum()) if len(_ok) and "n_large" in _ok else 0,
                   n_rejected_photo=int(_ok["n_photo"].sum()) if len(_ok) and "n_photo" in _ok else 0,
                   n_rejected_thin=int(_ok["n_thin"].sum()) if len(_ok) and "n_thin" in _ok else 0,
                   n_rejected_flat=int(_ok["n_flat"].sum()) if len(_ok) and "n_flat" in _ok else 0)
    # quality gate of the ground truth: if the median refined pair still flags much of its static scene as moving,
    # the objects are not trustworthy and Bonn is not interpreted (v6.6)
    BONN_STATUS["suspect"] = bool(np.isfinite(BONN_QC["median_flag_frac"])
                                  and BONN_QC["median_flag_frac"] > BONN_DESIGN["qc_max_object_frac"])
if BONN_PAIR_META:
    MOVERS_BONN, _bn_noise, BONN_TAU_PAR = bonn_regimes(bonn_object_table(BONN_PAIR_META), BONN_PAIR_META)
    BONN_STATUS["noise_floor_px"] = _bn_noise
    MOVERS_BONN.to_csv(os.path.join(MANIFEST_DIR, "bonn_objects.csv"), index=False)
    print(f"Bonn RGB-D Dynamic: {len(BONN_PAIR_META)} pairs with at least one object from {len(BONN_STATUS['sequences'])} sequences "
          f"({sum(v.get('MiB', 0) for v in BONN_STATUS['download'].values()):,.0f} MiB downloaded); people: {BONN_STATUS['person_model']}")
    print(pd.DataFrame([dict(sequence=k, frames=v["n_frames"], convention=v["convention"],
                             warp_err=f"{v['median_rel_depth_err']:.3f}", A3="PASS" if v["passed"] else "FAIL")
                        for k, v in BONN_A3.items()]).to_string(index=False))
    if BONN_QC:
        print(f"relative pose re-estimated from static correspondences for {BONN_QC['n_refined']} of {BONN_QC['n_pairs_tried']} "
              f"pairs (the others are left out): median reprojection error {BONN_QC['median_mocap_err_px']:.2f} px with the "
              f"motion-capture pose, {BONN_QC['median_err_px']:.2f} px after (inliers {BONN_QC['median_inlier_frac']:.2f}); "
              f"median change {BONN_QC['median_rot_change_deg']:.2f} deg, {100 * BONN_QC['median_trans_change_m']:.1f} cm")
        print(f"share of the image flagged as moving outside people: median {BONN_QC['median_flag_frac']:.3f} (90th percentile "
              f"{BONN_QC['p90_flag_frac']:.3f}); regions not taken as objects: {BONN_QC['n_rejected_large']} too large, "
              f"{BONN_QC['n_rejected_photo']} explained by the static prediction, {BONN_QC['n_rejected_thin']} too thin, "
              f"{BONN_QC['n_rejected_flat']} flat on the floor (shadows); floor "
              f"found in {100 * BONN_QC['floor_found_share']:.0f}% of the pairs"
              + ("  -> QUALITY GATE FAILED: Bonn is not interpreted" if BONN_STATUS.get("suspect") else ""))
    _sc = np.array([m_["scale_log"] for m_ in BONN_PAIR_META], float)
    print(f"static scale check: median |log(Z_tri / Z)| over static pixels per pair = {np.nanmedian(np.abs(_sc)):.3f} "
          f"(90th percentile {np.nanpercentile(np.abs(_sc), 90):.3f}); subtracted from every object's x_geo")
    print(f"flow-noise floor: 90th percentile of the median |eps| of static patches = {_bn_noise:.2f} px; "
          f"parallel regime on Bonn: |eps| < {BONN_TAU_PAR:.2f} px")
    print("objects by kind and regime:")
    print(MOVERS_BONN.groupby(["cls", "regime"]).size().unstack(fill_value=0).to_string())
    print("objects by kind and floor contact:")
    print(MOVERS_BONN.groupby(["cls", "floor"]).size().unstack(fill_value=0).to_string())
    if BONN_STATUS.get("notes"):
        print("sequences left out:", BONN_STATUS["notes"])
else:
    print("Bonn RGB-D Dynamic is not used in this run:", BONN_STATUS["note"] or "no data")
release_memory(); mem_report("Bonn RGB-D Dynamic")

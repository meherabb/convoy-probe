# PLACEHOLDER
# 3.11 Argoverse 2 (exploratory, v6.3): real driving with LiDAR, read directly from the public AWS
#      Open Data bucket (anonymous access, no account or registration). A deterministic sample of
#      validation logs is drawn; for each log the compact tables (ego poses, calibration, labelled 3-D
#      boxes with track IDs) are read first, frame pairs of the front-centre camera are chosen from
#      them, and only the images and LiDAR sweeps of those pairs are then fetched. The ground truth is
#      built as for KITTI (Section 3.10) with the occlusion test of Section 3.10.0. One checkpointed
#      stage ("av2_gt") holds the cropped images and the ground truth, so a later session needs no
#      download. If the bucket cannot be reached or a log fails the checks (A3), that part is left out
#      of every experiment, table and figure with a note. Section 3.11b (v6.4) adds further logs in a
#      second stage, the ground contact of every vehicle and the vehicle table of all logs.
import urllib.parse, urllib.request, urllib.error
import xml.etree.ElementTree as _ET
from concurrent.futures import ThreadPoolExecutor as _AV2Pool
from scipy.spatial.transform import Rotation as _Rot

AV2_STATUS = dict(available=False, note="", logs=[], tried=0, download={})
AV2_DIR = os.path.join(SCRATCH_DIR, "av2")
AV2_RAW = os.path.join(AV2_DIR, "raw")
AV2_GT_DIR = os.path.join(AV2_DIR, "pairs")
AV2_PAIR_META, AV2_A3, AV2_VEHICLE_CHECK = [], {}, {}


class PublicS3:
    """Anonymous read access to a public S3 bucket over HTTPS: ListObjectsV2 (with continuation) and GET."""

    def __init__(self, base, timeout=60, retries=4):
        self.base = base.rstrip("/") + "/"
        self.timeout, self.retries = timeout, retries
        self.n_requests, self.n_bytes = 0, 0

    def _read(self, url):
        err = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                self.n_requests += 1
                self.n_bytes += len(data)
                return data
            except urllib.error.HTTPError as e:
                if e.code in (400, 403, 404):
                    raise IOError(f"HTTP {e.code} for {url[len(self.base):][:120]}")
                err = e
            except Exception as e:
                err = e
            time.sleep(2 + 3 * attempt)
        raise IOError(f"the public bucket could not be read after {self.retries} attempts ({type(err).__name__}: {str(err)[:160]})")

    def list(self, prefix, delimiter="/"):
        keys, prefixes, token = [], [], None
        tag = lambda e: e.tag.rsplit("}", 1)[-1]
        while True:
            q = {"list-type": "2", "prefix": prefix}
            if delimiter:
                q["delimiter"] = delimiter
            if token:
                q["continuation-token"] = token
            root = _ET.fromstring(self._read(self.base + "?" + urllib.parse.urlencode(q)))
            token, truncated = None, False
            for e in root:
                t = tag(e)
                if t == "Contents":
                    keys += [c.text for c in e if tag(c) == "Key"]
                elif t == "CommonPrefixes":
                    prefixes += [c.text for c in e if tag(c) == "Prefix"]
                elif t == "IsTruncated":
                    truncated = (e.text or "").strip().lower() == "true"
                elif t == "NextContinuationToken":
                    token = e.text
            if not truncated or not token:
                return keys, prefixes

    def get(self, key, dest):
        if os.path.exists(dest):
            return dest
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        data = self._read(self.base + urllib.parse.quote(key))
        with open(dest + ".part", "wb") as f:
            f.write(data)
        os.replace(dest + ".part", dest)
        return dest


def _av2_se3(df):
    # (N, 4, 4) rigid transforms from the qw, qx, qy, qz, tx_m, ty_m, tz_m columns (scalar-first quaternions)
    q = df[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)
    T = np.repeat(np.eye(4)[None], len(df), 0)
    if len(df):
        T[:, :3, :3] = _Rot.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()
        T[:, :3, 3] = df[["tx_m", "ty_m", "tz_m"]].to_numpy(np.float64)
    return T


def _interp_se3(ts, T, t):
    # geodesic interpolation of rigid transforms sampled at sorted times ts (s), at times t (s); beyond
    # the ends, constant-velocity extrapolation from the two nearest samples
    t = np.atleast_1d(np.asarray(t, np.float64))
    if len(ts) == 1:
        return np.repeat(T[:1], len(t), 0)
    b = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
    a = b - 1
    w = (t - ts[a]) / np.maximum(ts[b] - ts[a], 1e-6)
    Ra, Rb = T[a, :3, :3], T[b, :3, :3]
    rv = _Rot.from_matrix(np.einsum("nji,njk->nik", Ra, Rb)).as_rotvec()
    out = np.repeat(np.eye(4)[None], len(t), 0)
    out[:, :3, :3] = Ra @ _Rot.from_rotvec(rv * w[:, None]).as_matrix()
    out[:, :3, 3] = T[a, :3, 3] + w[:, None] * (T[b, :3, 3] - T[a, :3, 3])
    return out


def _rot_angle_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


class AV2Log:
    """One Argoverse 2 sensor log: poses, calibration, labelled tracks, front-centre camera and LiDAR."""

    TABLES = ("annotations.feather", "city_SE3_egovehicle.feather", "calibration/egovehicle_SE3_sensor.feather",
              "calibration/intrinsics.feather")

    def __init__(self, s3, log_id, raw_root=None):
        self.s3, self.id = s3, log_id
        self.key = AV2_DESIGN["prefix"] + AV2_DESIGN["split"] + "/" + log_id + "/"
        self.dir = os.path.join(raw_root or AV2_RAW, log_id)
        for t in self.TABLES:
            s3.get(self.key + t, os.path.join(self.dir, *t.split("/")))
        cam = AV2_DESIGN["camera"]
        ck, _ = s3.list(self.key + f"sensors/cameras/{cam}/")
        lk, _ = s3.list(self.key + "sensors/lidar/")
        self.cam_ts = np.array(sorted(int(os.path.basename(k).split(".")[0]) for k in ck if k.endswith(".jpg")), np.int64)
        self.lidar_ts = np.array(sorted(int(os.path.basename(k).split(".")[0]) for k in lk if k.endswith(".feather")), np.int64)
        if len(self.cam_ts) < 10 or len(self.lidar_ts) < 5:
            raise IOError("too few camera images or LiDAR sweeps")
        P = pd.read_feather(os.path.join(self.dir, "city_SE3_egovehicle.feather")).sort_values("timestamp_ns")
        P = P.drop_duplicates("timestamp_ns")
        self.t0 = int(P["timestamp_ns"].iloc[0])
        self.pose_t = (P["timestamp_ns"].to_numpy(np.int64) - self.t0) * 1e-9
        self.pose_T = _av2_se3(P)
        self.origin = self.pose_T[0, :3, 3].copy()                 # local city origin (keeps coordinates small)
        self.pose_T[:, :3, 3] -= self.origin
        E = pd.read_feather(os.path.join(self.dir, "calibration", "egovehicle_SE3_sensor.feather"))
        I = pd.read_feather(os.path.join(self.dir, "calibration", "intrinsics.feather"))
        e, i_ = E[E["sensor_name"] == cam], I[I["sensor_name"] == cam]
        if not len(e) or not len(i_):
            raise IOError(f"no calibration for {cam}")
        self.ego_T_cam = _av2_se3(e)[0]
        r = i_.iloc[0]
        self.W_full, self.H_full = int(r["width_px"]), int(r["height_px"])
        # 4:3 window of the full image width (centred on the principal point's row), resized to out_wh
        Wo, Ho = AV2_DESIGN["out_wh"]
        wc = self.W_full - self.W_full % 4
        hc = wc * 3 // 4
        if hc > self.H_full:
            hc = self.H_full - self.H_full % 3; wc = hc * 4 // 3
        x0 = (self.W_full - wc) // 2
        y0 = int(np.clip(round(float(r["cy_px"]) - hc / 2), 0, self.H_full - hc))
        self.crop = (int(x0), int(y0), int(wc), int(hc))
        s = Wo / wc
        self.cam = PinholeCamera(float(r["fx_px"]) * s, float(r["fy_px"]) * s, (float(r["cx_px"]) - x0 + 0.5) * s - 0.5,
                                 (float(r["cy_px"]) - y0 + 0.5) * s - 0.5, Wo, Ho)
        A = pd.read_feather(os.path.join(self.dir, "annotations.feather"))
        A = A[A["timestamp_ns"].isin(set(self.lidar_ts.tolist()))].copy()
        A["Tego"] = list(_av2_se3(A))
        self.boxes = {int(ts): g for ts, g in A.groupby("timestamp_ns")}
        self.tracks = {}
        for uid, g in A.sort_values("timestamp_ns").groupby("track_uuid"):
            ts = (g["timestamp_ns"].to_numpy(np.int64) - self.t0) * 1e-9
            Te = self.ego_T(ts)
            Tw = np.einsum("nij,njk->nik", Te, np.stack(g["Tego"].to_list()))
            self.tracks[uid] = dict(t=ts, T=Tw, category=str(g["category"].iloc[0]),
                                    lwh=g[["length_m", "width_m", "height_m"]].median().to_numpy(np.float64))

    # ---- time and poses ---------------------------------------------------------------------------
    def sec(self, ts_ns):
        return (np.asarray(ts_ns, np.int64) - self.t0) * 1e-9

    def ego_T(self, t):
        # city_SE3_egovehicle maps ego coordinates into the city frame (documented; checked by A3 below)
        return _interp_se3(self.pose_t, self.pose_T, t)

    def cam_pose(self, t):
        # camera -> city rotation (columns: camera axes) and camera centre at time t (s)
        T = self.ego_T(t)[0] @ self.ego_T_cam
        return T[:3, :3], T[:3, 3]

    def cam_dict(self, t):
        R, C = self.cam_pose(t)
        return dict(**self.cam.to_dict(), R=R.tolist(), C=C.tolist())

    def track_T(self, uid, t):
        tr = self.tracks[uid]
        return _interp_se3(tr["t"], tr["T"], t)

    def nearest_sweep(self, ts_ns):
        return int(self.lidar_ts[np.argmin(np.abs(self.lidar_ts - int(ts_ns)))])

    # ---- pair selection from the tables alone ---------------------------------------------------------
    def _moving_vehicle_in_view(self, ts_ns):
        t = float(self.sec(ts_ns))
        R, C = self.cam_pose(t)
        for uid in self.boxes.get(self.nearest_sweep(ts_ns), pd.DataFrame(columns=["track_uuid"]))["track_uuid"]:
            tr = self.tracks.get(uid)
            if tr is None or tr["category"] not in AV2_DESIGN["classes"] or len(tr["t"]) < 2:
                continue
            if not (tr["t"][0] - 0.05 <= t <= tr["t"][-1] + 0.05):
                continue
            T2 = self.track_T(uid, [t - 0.1, t + 0.1])
            if np.linalg.norm(T2[1, :3, 3] - T2[0, :3, 3]) < 0.2 * AV2_DESIGN["min_moving_speed_mps"]:
                continue
            q = (T2[:, :3, 3].mean(0) - C) @ R
            if not (3.0 <= q[2] <= 60.0):
                continue
            u, v = self.cam.project(q)
            if 0 <= u < self.cam.W and 0 <= v < self.cam.H:
                return True
        return False

    def select_pairs(self, rng, n_max):
        lo, hi = AV2_DESIGN["pair_baseline_m"]
        t = self.sec(self.cam_ts)
        poses = [self.cam_pose(x) for x in t]
        good = [f for f in range(len(t)) if self._moving_vehicle_in_view(self.cam_ts[f])]
        out = []
        for f in rng.permutation(good):
            f = int(f)
            if len(out) >= n_max:
                break
            if any(abs(f - p[0]) < AV2_DESIGN["min_start_gap_frames"] for p in out):
                continue
            target = rng.uniform(lo, hi)
            for g in range(f + 1, min(f + 1 + AV2_DESIGN["max_gap_frames"], len(t))):
                bl = float(np.linalg.norm(poses[g][1] - poses[f][1]))
                if bl >= target:
                    ang = _rot_angle_deg(poses[f][0], poses[g][0])
                    if bl <= hi + 0.5 and ang <= AV2_DESIGN["max_rot_deg"]:
                        out.append((f, g, bl, ang))
                    break
        return [(int(self.cam_ts[f]), int(self.cam_ts[g]), bl, ang) for f, g, bl, ang in sorted(out)]

    # ---- sensor files ------------------------------------------------------------------------------
    def image_path(self, ts_ns):
        return os.path.join(self.dir, "cam", f"{int(ts_ns)}.jpg")

    def sweep_path(self, ts_ns):
        return os.path.join(self.dir, "lidar", f"{int(ts_ns)}.feather")

    def fetch(self, cam_ts, sweep_ts, workers=8):
        jobs = [(self.key + f"sensors/cameras/{AV2_DESIGN['camera']}/{int(x)}.jpg", self.image_path(x)) for x in cam_ts]
        jobs += [(self.key + f"sensors/lidar/{int(x)}.feather", self.sweep_path(x)) for x in sweep_ts]
        with _AV2Pool(workers) as ex:
            list(ex.map(lambda kd: self.s3.get(*kd), jobs))

    def image(self, ts_ns):
        x0, y0, wc, hc = self.crop
        im = Image.open(self.image_path(ts_ns)).convert("RGB").crop((x0, y0, x0 + wc, y0 + hc))
        return np.asarray(im.resize((self.cam.W, self.cam.H), Image.LANCZOS))

    def frame_points(self, ts_cam):
        """LiDAR returns of the sweep nearest to a camera image, in city coordinates at the image's time.
        Static returns are exact (ego-motion compensated by the dataset). A return on a labelled vehicle
        was captured at its own time (sweep time + offset_ns); it is carried to the image's time along
        the vehicle's labelled track, which uses only the track's relative motion."""
        ts_l = self.nearest_sweep(ts_cam)
        L = pd.read_feather(self.sweep_path(ts_l))
        X = L[["x", "y", "z"]].to_numpy(np.float64)
        tp = float(self.sec(ts_l)) + L["offset_ns"].to_numpy(np.float64) * 1e-9
        Te = self.ego_T(float(self.sec(ts_l)))[0]
        Xw = X @ Te[:3, :3].T + Te[:3, 3]
        t = float(self.sec(ts_cam))
        R, C = self.cam_pose(t)
        q = (Xw - C) @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = self.cam.project(q)
        pre = (q[:, 2] > 0.5) & (uv[:, 0] > -60) & (uv[:, 0] < self.cam.W + 60) & (uv[:, 1] > -60) & (uv[:, 1] < self.cam.H + 60)
        X, Xw, tp = X[pre], Xw[pre], tp[pre]
        static = np.ones(len(X), bool)
        owner = np.full(len(X), -1, np.int64)
        vehicles = []
        m_s, m_o, cut = AV2_DESIGN["static_box_margin_m"], AV2_DESIGN["own_box_margin_m"], AV2_DESIGN["ground_cut_m"]
        for r in self.boxes.get(ts_l, pd.DataFrame()).itertuples():
            Tb = r.Tego
            qb = (X - Tb[:3, 3]) @ Tb[:3, :3]
            half = np.array([r.length_m, r.width_m, r.height_m]) / 2
            static &= ~np.all(np.abs(qb) <= half + m_s, axis=1)
            if r.category in AV2_DESIGN["classes"] and r.track_uuid in self.tracks:
                ins = np.all(np.abs(qb[:, :2]) <= half[:2] + m_o, axis=1) & (qb[:, 2] <= half[2] + m_o) & (qb[:, 2] >= -half[2] + cut)
                ins &= owner < 0
                owner[ins] = len(vehicles)
                vehicles.append(dict(track=str(r.track_uuid), category=str(r.category), lwh=half * 2))
        Xt = Xw.copy()
        dt_ms = np.full(len(X), np.nan)
        for n, v in enumerate(vehicles):
            m = owner == n
            if not m.any():
                continue
            Tp = self.track_T(v["track"], tp[m])
            Ti = self.track_T(v["track"], [t])[0]
            Xo = np.einsum("nji,nj->ni", Tp[:, :3, :3], Xw[m] - Tp[:, :3, 3])      # object coordinates
            Xt[m] = Xo @ Ti[:3, :3].T + Ti[:3, 3]
            dt_ms[m] = (tp[m] - t) * 1e3
        return dict(Xw=Xt, static=static, owner=owner, vehicles=vehicles, t=t, sweep=ts_l, dt_ms=dt_ms)

    def project(self, P, t):
        R, C = self.cam_pose(t)
        q = (P["Xw"] - C) @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = self.cam.project(q)
        z = np.where((q[:, 2] >= AV2_DESIGN["lidar_min_depth_m"]) & (q[:, 2] <= AV2_DESIGN["lidar_max_depth_m"]), q[:, 2], np.nan)
        D, idx = lidar_zbuffer(uv, z, self.cam.H, self.cam.W, AV2_DESIGN["zbuffer_window_px"], *AV2_DESIGN["zbuffer_tol"])
        return uv, z, D, idx

    def _box_truncation(self, uid, t):
        # share of the projected box outside the image (1 if any corner is behind the camera)
        T = self.track_T(uid, [t])[0]
        l, w, h = self.tracks[uid]["lwh"] / 2
        corners = np.array([[a * l, b * w, c * h] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        R, C = self.cam_pose(t)
        q = (corners @ T[:3, :3].T + T[:3, 3] - C) @ R
        if np.any(q[:, 2] <= 0.1):
            return 1.0
        uv = self.cam.project(q)
        u0, v0 = uv.min(0); u1, v1 = uv.max(0)
        area = max((u1 - u0) * (v1 - v0), 1e-9)
        iw = max(0.0, min(u1, self.cam.W - 1) - max(u0, 0.0)); ih = max(0.0, min(v1, self.cam.H - 1) - max(v0, 0.0))
        return float(1.0 - iw * ih / area)

    def pair_gt(self, ci, cj, img_i=None, img_j=None):
        ti, tj = float(self.sec(ci)), float(self.sec(cj))
        P = self.frame_points(ci)
        uv, z, D, idx = self.project(P, ti)
        has = idx >= 0
        win = np.zeros(len(z), bool)
        win[idx[has]] = True
        with np.errstate(invalid="ignore"):
            inimg = (uv[:, 0] >= -0.5) & (uv[:, 0] < self.cam.W - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < self.cam.H - 0.5)
        sm = np.zeros(D.shape, bool)
        sm[has] = P["static"][idx[has]]
        Ri, Ci = self.cam_pose(ti)
        Rj, Cj = self.cam_pose(tj)
        b = Cj - Ci
        sc = dict(cams=[self.cam_dict(ti), self.cam_dict(tj)])
        vehicles, photo = [], []
        for n, v in enumerate(P["vehicles"]):
            uid = v["track"]
            tr = self.tracks[uid]
            if not (tr["t"][0] - 0.06 <= min(ti, tj) and max(ti, tj) <= tr["t"][-1] + 0.06):
                continue                                         # the track must cover both images
            own_all = (P["owner"] == n) & np.isfinite(z) & inimg
            own = own_all & win
            if own.sum() < 3:
                continue
            Ti, Tj = self.track_T(uid, [ti, tj])
            disp = float(np.linalg.norm(Tj[:3, 3] - Ti[:3, 3]))
            state = "moving" if disp >= PREREG["kt_moving_disp_m"] else "parked" if disp <= PREREG["kt_parked_disp_m"] else None
            if state is None:
                continue
            n_proj = len(np.unique(np.round(uv[own_all]).astype(int), axis=0))
            occlusion = 1.0 - len(np.unique(np.round(uv[own]).astype(int), axis=0)) / max(n_proj, 1)
            trunc = self._box_truncation(uid, ti)
            if trunc > AV2_DESIGN["max_truncation"] or occlusion > AV2_DESIGN["max_occlusion"]:
                continue
            Xi = P["Xw"][own]
            Xj = (Xi - Ti[:3, 3]) @ Ti[:3, :3] @ Tj[:3, :3].T + Tj[:3, 3]      # rigid motion along the track
            qj = (Xj - Cj) @ Rj
            uvj = self.cam.project(qj)
            okj = (qj[:, 2] > 0.5) & (uvj[:, 0] >= 0) & (uvj[:, 0] <= self.cam.W - 1) & (uvj[:, 1] >= 0) & (uvj[:, 1] <= self.cam.H - 1)
            st_, ok = lidar_vehicle_stats(sc, uv[own], uvj, z[own], okj)
            px = np.unique(np.round(uv[own][ok]).astype(int), axis=0) if ok.any() else np.zeros((0, 2), int)
            rec = dict(track=uid, label=v["category"], state=state, disp_m=disp, s=float((Tj[:3, 3] - Ti[:3, 3]) @ b / (b @ b)),
                       speed_ratio=float(disp / max(np.linalg.norm(b), 1e-9)), occlusion=float(occlusion), truncation=trunc,
                       dt_ms=float(np.nanmedian(np.abs(P["dt_ms"][own]))), pixels=px.tolist(), **st_)
            rec["n_lidar_px"] = int(len(px))
            vehicles.append(rec)
            if img_i is not None and state == "moving" and ok.sum() >= 10:
                # photometric check of the vehicle ground truth: grey value at the rigidly moved position
                # versus at the position a static point at the same depth would have
                qs = (Xi[ok] - Cj) @ Rj
                uvs = self.cam.project(qs)
                informative = np.linalg.norm(uvs - uvj[ok], axis=1) >= 3.0
                if informative.sum() >= 10:
                    g0 = sample_gray(img_i, uv[own][ok][informative])
                    photo.append((g0, sample_gray(img_j, uvj[ok][informative]), sample_gray(img_j, uvs[informative])))
        return dict(depth=D, static=sm, vehicles=vehicles, scene=sc, baseline=float(np.linalg.norm(b)), points=P,
                    uv=uv, z=z, idx=idx, photo=photo)

    def a3(self, pairs):
        # (i) static LiDAR of image i, warped with the poses, against the LiDAR of image j (relative depth
        # disagreement); (ii) grey-value error of static LiDAR pixels with the exact warp versus no warp;
        # (iii) the same for moving vehicles: exact rigid motion versus a static point at the same depth
        werr, pw, pid, veh = [], [], [], []
        for ci, cj in pairs:
            ti, tj = float(self.sec(ci)), float(self.sec(cj))
            Ii, Ij = self.image(ci), self.image(cj)
            g = self.pair_gt(ci, cj, Ii, Ij)
            Pj = self.frame_points(cj)
            if not len(Pj["static"]):
                continue
            _, zj, Dj, idxj = self.project(Pj, tj)
            Dj = np.where((idxj >= 0) & Pj["static"][np.maximum(idxj, 0)], Dj, np.nan)
            m = (g["idx"] >= 0) & g["static"]
            sel = g["idx"][m]
            Rj, Cj = self.cam_pose(tj)
            q = (g["points"]["Xw"][sel] - Cj) @ Rj
            ok = q[:, 2] > 1.0
            uvj = self.cam.project(q[ok])
            ins = (uvj[:, 0] >= 0) & (uvj[:, 0] <= self.cam.W - 1) & (uvj[:, 1] >= 0) & (uvj[:, 1] <= self.cam.H - 1)
            zo = Dj[np.round(uvj[ins, 1]).astype(int), np.round(uvj[ins, 0]).astype(int)]
            zp = q[ok][ins, 2]
            f = np.isfinite(zo)
            if f.sum() >= 200:
                werr.append(float(np.median(np.abs(zo[f] - zp[f]) / zo[f])))
            src = g["uv"][sel][ok][ins]                          # sub-pixel position of the return in image i
            if len(src) >= 200:
                g0 = sample_gray(Ii, src)
                pw.append(float(np.nanmedian(np.abs(g0 - sample_gray(Ij, uvj[ins])))))
                pid.append(float(np.nanmedian(np.abs(g0 - sample_gray(Ij, src)))))
            for g0, gr, gs in g["photo"]:
                veh.append((np.abs(g0 - gr), np.abs(g0 - gs)))
        out = dict(n_test_pairs=len(pairs), static_warp_err=float(np.median(werr)) if werr else np.nan,
                   photo_static_warp=float(np.median(pw)) if pw else np.nan, photo_static_nowarp=float(np.median(pid)) if pid else np.nan)
        er = np.concatenate([a for a, _ in veh]) if veh else np.zeros(0)
        es = np.concatenate([b_ for _, b_ in veh]) if veh else np.zeros(0)
        out.update(photo_vehicle_rigid=float(np.nanmedian(er)) if len(er) else np.nan,
                   photo_vehicle_static=float(np.nanmedian(es)) if len(es) else np.nan, n_vehicle_px=int(len(er)))
        return out, (er, es)


def _av2_a3_pass(r):
    # per log: static geometry (depth warp) and the camera model against the images (grey-value warp)
    ok = np.isfinite(r["static_warp_err"]) and r["static_warp_err"] <= 0.05
    ok &= np.isfinite(r["photo_static_warp"]) and r["photo_static_warp"] <= 0.8 * r["photo_static_nowarp"]
    return bool(ok)


def _av2_vehicle_check(err_rigid, err_static, min_px=300):
    # pooled over the accepted logs: grey values of moving vehicles agree better with their rigid motion along
    # the labelled track than with a static point at the same depth (a single log has too few pixels to decide)
    er, es = np.concatenate(err_rigid) if err_rigid else np.zeros(0), np.concatenate(err_static) if err_static else np.zeros(0)
    n = int(np.isfinite(er).sum())
    if n < min_px:
        return dict(n_px=n, rigid=float(np.nanmedian(er)) if n else np.nan, static=float(np.nanmedian(es)) if n else np.nan,
                    passed=True, note=f"not decided (only {n} informative vehicle pixels)")
    rg, st = float(np.nanmedian(er)), float(np.nanmedian(es))
    return dict(n_px=n, rigid=rg, static=st, passed=bool(rg <= 0.9 * st),
                note=f"median grey-value error {rg:.1f} (rigid motion) vs {st:.1f} (static point) over {n} pixels")


def av2_list_logs(s3):
    # validation log IDs in the bucket, sorted; the order in which logs are tried is a fixed permutation of this list
    _, prefixes = s3.list(AV2_DESIGN["prefix"] + AV2_DESIGN["split"] + "/")
    log_ids = sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes)
    if not log_ids:
        raise IOError("no logs listed in the bucket")
    return log_ids, np.random.default_rng(seed_for("av2-logs")).permutation(len(log_ids))


def av2_take_log(s3, log_id, gt_dir, index0):
    """Select, download and check one log, and write its pairs (pair indices from index0) to gt_dir.
    Returns (A3 record or None, reason for leaving the log out or None, (grey errors rigid, static) or None).
    The raw files are deleted in every case; a log that fails half-way leaves no pair behind."""
    lg = AV2Log(s3, log_id)
    written = []
    try:
        pairs = lg.select_pairs(np.random.default_rng(seed_for("av2-pairs", log_id)), RUN_CONFIG["n_av2_pairs_per_log"])
        if len(pairs) < AV2_DESIGN["min_pairs_per_log"]:
            return None, f"{len(pairs)} usable pairs", None
        cams = sorted({c for p in pairs for c in p[:2]})
        lg.fetch(cams, sorted({lg.nearest_sweep(c) for c in cams}))
        r, err = lg.a3([(p[0], p[1]) for p in pairs[:6]])
        rec = dict(n_pairs=len(pairs), convention="documented (city_SE3_egovehicle: ego -> city)", **r, passed=_av2_a3_pass(r))
        if not rec["passed"]:
            return rec, "A3 failed", None
        for (ci, cj, bl, ang) in pairs:
            g = lg.pair_gt(ci, cj)
            if not any(v["state"] == "moving" for v in g["vehicles"]):
                continue
            d = os.path.join(gt_dir, f"{log_id[:8]}_{ci}_{cj}")
            os.makedirs(d)
            written.append(d)
            for k_, c_ in enumerate((ci, cj)):
                Image.fromarray(lg.image(c_)).save(os.path.join(d, f"rgb{k_}.png"))
            np.savez_compressed(os.path.join(d, "gt.npz"), depth=g["depth"].astype(np.float32), static=g["static"])
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(dict(set="AV2", index=index0 + len(written) - 1, seq=log_id[:8], log=log_id, frames=[int(ci), int(cj)],
                               crop=list(lg.crop), out_wh=list(AV2_DESIGN["out_wh"]), baseline=bl, rot_deg=ang,
                               dt_s=float(lg.sec(cj) - lg.sec(ci)), scene=g["scene"], vehicles=g["vehicles"], dir=d),
                          f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        rec["n_pairs_used"] = len(written)
        return rec, None, err
    except Exception:
        for d in written:
            shutil.rmtree(d, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(lg.dir, ignore_errors=True)                # raw images and sweeps are no longer needed


def av2_run_logs(s3, log_ids, order, start, skip, n_target, max_tried, gt_dir, index0, a3):
    """Try logs in the fixed order from position start (skipping the IDs in skip) until n_target are accepted or
    max_tried have been tried in all; writes their pairs to gt_dir and the A3 records into a3."""
    n_pair, accepted, rejected, veh_err, tried = index0, [], {}, ([], []), 0
    for k in order[start:]:
        if len(accepted) >= n_target or start + tried >= max_tried:
            break
        log_id = log_ids[int(k)]
        if log_id in skip:
            continue
        tried += 1
        try:
            rec, why, err = av2_take_log(s3, log_id, gt_dir, n_pair)
        except Exception as e:                      # one unreadable log never stops the stage
            rejected[log_id] = f"{type(e).__name__}: {str(e)[:120]}"
            continue
        if rec is not None:
            a3[log_id] = rec
        if why:
            rejected[log_id] = why
            continue
        n_pair += rec["n_pairs_used"]
        accepted.append(log_id)
        veh_err[0].append(err[0]); veh_err[1].append(err[1])
    return accepted, rejected, tried, _av2_vehicle_check(*veh_err)


AV2_STAGE = Stage("av2_gt", "av2", dict(design={k: v for k, v in AV2_DESIGN.items() if k != "base_url"},
                                        n_logs=RUN_CONFIG["n_av2_logs"], n_pairs=RUN_CONFIG["n_av2_pairs_per_log"],
                                        prereg={k: v for k, v in PREREG.items() if k.startswith("kt_")}))
with stage("av2_gt"):
    if not RUN_CONFIG.get("use_av2", True):
        AV2_STATUS["note"] = "switched off in RUN_CONFIG"
    elif not AV2_STAGE.done():
        try:
            shutil.rmtree(AV2_DIR, ignore_errors=True); os.makedirs(AV2_GT_DIR)
            s3 = PublicS3(AV2_DESIGN["base_url"])
            log_ids, order = av2_list_logs(s3)
            accepted, rejected, AV2_STATUS["tried"], _vc = av2_run_logs(
                s3, log_ids, order, 0, set(), RUN_CONFIG["n_av2_logs"], AV2_DESIGN["max_logs_tried"], AV2_GT_DIR, 0, AV2_A3)
            AV2_STATUS["download"] = dict(requests=s3.n_requests, MiB=round(s3.n_bytes / 2**20, 1))
            with open(os.path.join(AV2_DIR, "a3.json"), "w") as f:
                json.dump(dict(a3=AV2_A3, vehicle_check=_vc, accepted=accepted, rejected=rejected, tried=AV2_STATUS["tried"],
                               download=AV2_STATUS["download"], n_logs_listed=len(log_ids)), f, indent=1, default=str)
            shutil.rmtree(AV2_RAW, ignore_errors=True)
            if not accepted:
                raise IOError(f"no log passed (tried {AV2_STATUS['tried']}: {list(rejected.values())[:3]})")
            if not _vc["passed"]:
                raise IOError("the moving-vehicle ground truth is not confirmed by the images: " + _vc["note"])
            AV2_STAGE.commit([AV2_DIR])
        except Exception as e:
            AV2_STATUS["note"] = f"{type(e).__name__}: {str(e)[:300]}"
            shutil.rmtree(AV2_DIR, ignore_errors=True)

if os.path.exists(os.path.join(AV2_DIR, "a3.json")):
    _av2_rec = json.load(open(os.path.join(AV2_DIR, "a3.json")))
    AV2_A3 = _av2_rec["a3"]
    AV2_VEHICLE_CHECK = _av2_rec.get("vehicle_check", {})
    AV2_STATUS["download"] = _av2_rec.get("download", {})
    for d in sorted(glob.glob(os.path.join(AV2_GT_DIR, "*", "meta.json"))):
        m_ = json.load(open(d)); m_["dir"] = os.path.dirname(d)
        AV2_PAIR_META.append(m_)
    AV2_PAIR_META.sort(key=lambda m_: m_["index"])
    AV2_STATUS.update(available=bool(AV2_PAIR_META), logs=sorted({m_["log"] for m_ in AV2_PAIR_META}),
                      note=AV2_STATUS["note"] or ("" if AV2_PAIR_META else "no log passed A3 or no pair had a moving vehicle"))


def av2_a3_table(a3):
    return pd.DataFrame([dict(log=k[:8], pairs_used=v.get("n_pairs_used", 0), static_warp=v["static_warp_err"],
                              grey_static=f"{v['photo_static_warp']:.1f} vs {v['photo_static_nowarp']:.1f}",
                              grey_vehicles=(f"{v['photo_vehicle_rigid']:.1f} vs {v['photo_vehicle_static']:.1f} ({v['n_vehicle_px']} px)"
                                             if v.get("n_vehicle_px", 0) else "–"),
                              A3="PASS" if v["passed"] else "FAIL") for k, v in a3.items()])


if AV2_PAIR_META:
    print(f"Argoverse 2, first stage (v6.3): {len(AV2_PAIR_META)} pairs from {len(AV2_STATUS['logs'])} validation logs "
          f"(downloaded {AV2_STATUS['download'].get('MiB', 0):,.0f} MiB in {AV2_STATUS['download'].get('requests', 0)} requests)")
    print("A3 per log: static warp = median relative depth disagreement; grey-value error (0-255) of static LiDAR pixels with "
          "the exact warp vs without, and of moving vehicles with their rigid motion vs a static point at the same depth:")
    print(av2_a3_table(AV2_A3).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("moving-vehicle ground truth against the images (pooled over the logs): "
          + (AV2_VEHICLE_CHECK.get("note", "–") + (", PASS" if AV2_VEHICLE_CHECK.get("passed") else ", FAIL")))
else:
    print("Argoverse 2 is not used in this run:", AV2_STATUS["note"] or "no data")
release_memory(); mem_report("Argoverse 2 ground truth (first stage)");

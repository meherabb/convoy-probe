# PLACEHOLDER
# 3.9 Virtual KITTI 2 ground truth. Frames are cropped to the centred 500 x 375 window (4:3, native
#     pixels, principal point at the crop centre). Conventions that the files do not state are
#     decided from the data, as for TUM (assumption A3): (i) whether extrinsic.txt maps world to
#     camera or the reverse (static depth warped between frames must agree with the measured
#     depth), (ii) the axis and sign of the vehicle yaw (the rigid motion must carry a vehicle's
#     pixels onto its pixels in the other frame), (iii) how instance-mask values map to track IDs
#     (overlap with bbox.txt). Every pixel then gets its exact correspondence: static points stay,
#     moving vehicles move rigidly with their pose. Movers are the vehicles flagged as moving.
#     Two prevalence statistics need no model: the share of moving-vehicle pixels within 1 px of
#     their epipolar line (two views), and the share of moving vehicles whose track over K = 2, 4, 8
#     frames is fitted by a static point to within 1 px (no K-view geometric test can see them).

VK_CROP_W, VK_CROP_H = VK_DESIGN["crop_wh"]

def vk_measurable(df):
    # the CONVOY-3D pixel rule (D34): >= 40 eroded pixels visible in both views, inside the crop, and
    # >= 85% of the vehicle's silhouette not occluded (occupancy ratio of the released ground truth)
    return (df["gt_n_eroded"] >= PREREG["min_object_pixels"]) & (df["gt_occupancy"] >= PREREG["vk_min_occupancy"])

def _rot_axis(axis, a):
    return {"y": rot_y, "z": rot_z, "x": rot_x}[axis](a)

class VKScene:
    def __init__(self, name):
        self.name, self.dir = name, os.path.join(VK_RAW, name)
        self.tg = vk_parse_textgt(self.dir)
        self.sel = VK_SELECTION[name]
        self.x0 = int(self.sel["x0"])
        I = self.tg["intrinsic"].iloc[0]
        self.cam = PinholeCamera(I["fx"], I["fy"], I["cx"] - self.x0, I["cy"], VK_CROP_W, VK_CROP_H)
        self.inverted = False
        self.yaw_axis, self.yaw_sign = "y", 1.0
        self.inst_mode = "track+1"
        self._set_poses()
        P = self.tg["pose"]
        self.pose = {(int(r.frame), int(r.trackID)): r for r in P.itertuples()}
        B = self.tg["bbox"]
        self.bbox = {(int(r.frame), int(r.trackID)): r for r in B.itertuples()}
        self.labels = {int(r.trackID): str(r.label) for r in self.tg["info"].itertuples()}
        self._cache = {}

    def _set_poses(self):
        fr, Rs, Cs = vk_camera_centres(self.tg, inverted=self.inverted)
        self.R = {int(f): R for f, R in zip(fr, Rs)}
        self.C = {int(f): C for f, C in zip(fr, Cs)}

    def cam_dict(self, f):
        return dict(**self.cam.to_dict(), R=self.R[f].tolist(), C=self.C[f].tolist())

    def rgb(self, f):
        return np.asarray(Image.open(os.path.join(self.dir, f"rgb_{f:05d}.jpg")).convert("RGB"))[:, self.x0:self.x0 + VK_CROP_W].copy()

    def depth(self, f):
        d = cv2.imread(os.path.join(self.dir, f"depth_{f:05d}.png"), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        d = d.astype(np.float32)[:, self.x0:self.x0 + VK_CROP_W] / 100.0
        d[(d <= 0) | (d >= VK_DESIGN["sky_depth_m"])] = np.inf
        return d

    def _inst_raw(self, f):
        a = np.asarray(Image.open(os.path.join(self.dir, f"inst_{f:05d}.png")))
        if a.ndim == 3:
            a = a[..., 0].astype(np.int64) * 65536 + a[..., 1].astype(np.int64) * 256 + a[..., 2]
        return a.astype(np.int64)

    def _match_values(self, f, raw):
        # map instance values to track IDs by overlap with the bbox.txt boxes of frame f
        out = {}
        for v in np.unique(raw):
            if v == 0:
                continue
            vv, uu = np.nonzero(raw == v)
            box = (uu.min(), uu.max(), vv.min(), vv.max())
            best, best_iou = None, 0.2
            for (ff, tid), r in self.bbox.items():
                if ff != f:
                    continue
                ix = min(box[1], r.right) - max(box[0], r.left); iy = min(box[3], r.bottom) - max(box[2], r.top)
                if ix <= 0 or iy <= 0:
                    continue
                inter = ix * iy
                iou = inter / ((box[1] - box[0]) * (box[3] - box[2]) + (r.right - r.left) * (r.bottom - r.top) - inter + 1e-9)
                if iou > best_iou:
                    best, best_iou = tid, iou
            if best is not None:
                out[int(v)] = int(best)
        return out

    def inst(self, f):
        # per-pixel track ID (-1: no vehicle), cropped
        raw = self._inst_raw(f)
        if self.inst_mode == "track+1":
            tid = np.where(raw > 0, raw - 1, -1)
        else:
            m = self._match_values(f, raw)
            tid = np.full(raw.shape, -1, np.int64)
            for v, t in m.items():
                tid[raw == v] = t
        return tid[:, self.x0:self.x0 + VK_CROP_W]

    def obj_transform(self, f, tid):
        r = self.pose.get((f, tid))
        if r is None:
            return None
        Rw = _rot_axis(self.yaw_axis, self.yaw_sign * r.rotation_world_space_y)
        return Rw, np.array([r.world_space_X, r.world_space_Y, r.world_space_Z], np.float64)

    def moving(self, f, tid):
        r = self.bbox.get((f, tid))
        return bool(r.isMoving) if r is not None else False

    def frame_gt(self, f):
        if ("f", f) not in self._cache:
            self._cache[("f", f)] = dict(rgb=self.rgb(f), depth=self.depth(f), inst=self.inst(f))
            if sum(1 for k in self._cache if k[0] == "f") > 24:          # bounded memory
                del self._cache[next(k for k in self._cache if k[0] == "f")]
        return self._cache[("f", f)]

    def pair_gt(self, i, j, tol_rel=0.02, tol_abs=0.05):
        # exact correspondences i -> j (static pixels fixed, moving vehicles rigid), movers, masks
        key = ("p", i, j)
        if key in self._cache:
            return self._cache[key]
        A, B = self.frame_gt(i), self.frame_gt(j)
        H, W = A["depth"].shape
        vv, uu = np.mgrid[0:H, 0:W]
        Ri, Ci, Rj, Cj = self.R[i], self.C[i], self.R[j], self.C[j]
        D = self.cam.rays(uu.astype(np.float64), vv.astype(np.float64)) @ Ri.T
        P = Ci + np.where(np.isfinite(A["depth"]), A["depth"], np.nan)[..., None] * D
        tids = [int(t) for t in np.unique(A["inst"]) if t >= 0 and self.moving(i, int(t))]
        movers, mover_id = [], np.full((H, W), -1, np.int16)
        for t in tids:
            Ti, Tj = self.obj_transform(i, t), self.obj_transform(j, t)
            if Ti is None or Tj is None:
                continue
            m = A["inst"] == t
            Pm = P[m]
            P[m] = (Pm - Ti[1]) @ Ti[0] @ Tj[0].T + Tj[1]          # rigid motion of the vehicle
            mover_id[m] = len(movers)
            movers.append(dict(track=t, label=self.labels.get(t, "?"), pos_i=Ti[1], pos_j=Tj[1]))
        Xc = (P - Cj) @ Rj
        with np.errstate(invalid="ignore", divide="ignore"):
            pB = self.cam.project(Xc)
        zB = Xc[..., 2]
        valid = np.isfinite(pB).all(-1) & (zB > 0.1) & (pB[..., 0] >= 0) & (pB[..., 0] <= W - 1) & (pB[..., 1] >= 0) & (pB[..., 1] <= H - 1)
        ui = np.clip(np.round(np.nan_to_num(pB[..., 0])).astype(int), 0, W - 1)
        vi = np.clip(np.round(np.nan_to_num(pB[..., 1])).astype(int), 0, H - 1)
        visible = valid & (np.abs(B["depth"][vi, ui] - zB) <= tol_rel * zB + tol_abs) & (B["inst"][vi, ui] == A["inst"])
        sc = dict(cams=[self.cam_dict(i), self.cam_dict(j)])
        eps, zgeo, par = pixel_epipolar_quantities(sc, 0, 1, np.stack([uu, vv], -1).astype(np.float64), pB)
        out = dict(pB=pB, zB=zB, valid=valid, visible=visible, eps=eps, zgeo=zgeo, parallax=par,
                   mover_id=mover_id, movers=movers, scene=sc, depth=A["depth"], inst=A["inst"])
        self._cache[key] = out
        if sum(1 for k in self._cache if k[0] == "p") > 12:            # bounded memory
            del self._cache[next(k for k in self._cache if k[0] == "p")]
        return out

    # ---- ground contact of a vehicle (v6.2, exploratory driving analysis) ----------------------
    # The 3-D box of pose.txt is read under four candidate conventions (location at the bottom face
    # or at the box centre; length along the first or the second horizontal axis of the object);
    # the one whose projection best matches the released 2-D boxes is used (decided from the data,
    # like the other conventions here). A wheel contact point is 0.10 m above the bottom face at
    # (+-0.35 length, +-0.40 width); it counts as visible if the vehicle's own pixel is there at the
    # point's depth (within 0.6 m), i.e. neither another object nor the vehicle's own body hides it.
    def _box_geometry(self, f, tid, conv):
        r = self.pose.get((f, tid)); T = self.obj_transform(f, tid)
        if r is None or T is None:
            return None
        Rw, loc = T
        v = {"x": 0, "y": 1, "z": 2}[self.yaw_axis]
        hz = [a for a in range(3) if a != v]
        e_len, e_wid = (Rw[:, hz[0]], Rw[:, hz[1]]) if conv[1] == 0 else (Rw[:, hz[1]], Rw[:, hz[0]])
        down = Rw[:, v] * np.sign(Rw[:, v] @ self.R[f][:, 1] or 1.0)
        w, h, l = float(r.width), float(r.height), float(r.length)
        c = loc - 0.5 * h * down if conv[0] == "bottom" else loc
        return dict(c=c, e_len=e_len, e_wid=e_wid, down=down, w=w, h=h, l=l)

    def _project_world(self, f, X):
        Xc = (np.atleast_2d(X) - self.C[f]) @ self.R[f]
        return self.cam.project(Xc), Xc[:, 2]

    def box_convention(self):
        if hasattr(self, "_box_conv"):
            return self._box_conv
        cands = [(a, b) for a in ("bottom", "centre") for b in (0, 1)]
        keys = [k for k, r in self.bbox.items() if k in self.pose and (r.right - r.left) * (r.bottom - r.top) >= 400
                and r.truncation_ratio < 0.1][:120]
        Wf = 2.0 * float(self.tg["intrinsic"].iloc[0]["cx"])
        best = (None, 0.0)
        for conv in cands:
            ious = []
            for (f, t) in keys:
                if f not in self.R:
                    continue
                g = self._box_geometry(f, t, conv)
                if g is None:
                    continue
                sg = np.array([(a, b, d) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
                X = g["c"] + sg[:, :1] * 0.5 * g["l"] * g["e_len"] + sg[:, 1:2] * 0.5 * g["w"] * g["e_wid"] + sg[:, 2:] * 0.5 * g["h"] * g["down"]
                uv, z = self._project_world(f, X)
                if np.any(z < 0.5):
                    continue
                u0, u1 = np.clip([uv[:, 0].min() + self.x0, uv[:, 0].max() + self.x0], 0, Wf - 1)
                v0, v1 = np.clip([uv[:, 1].min(), uv[:, 1].max()], 0, VK_CROP_H - 1)
                r = self.bbox[(f, t)]
                iw = max(0.0, min(u1, r.right) - max(u0, r.left)); ih = max(0.0, min(v1, r.bottom) - max(v0, r.top))
                inter = iw * ih; uni = (u1 - u0) * (v1 - v0) + (r.right - r.left) * (r.bottom - r.top) - inter
                ious.append(inter / uni if uni > 0 else 0.0)
            m = float(np.median(ious)) if len(ious) >= 5 else 0.0
            if m > best[1]:
                best = (conv, m)
        self._box_conv = best[0] if best[1] >= 0.5 else None
        self.box_iou = best[1]
        return self._box_conv

    def wheel_contact(self, f, tid, A):
        conv = self.box_convention()
        g = self._box_geometry(f, tid, conv) if conv is not None else None
        if g is None:
            return np.nan, np.nan
        hgt, tol = PREREG["x6_wheel_height_m"], PREREG["x6_wheel_depth_tol_m"]
        n_vis = n_in = 0
        H, W = A["depth"].shape
        for a in (-1, 1):
            for b in (-1, 1):
                X = g["c"] + a * 0.35 * g["l"] * g["e_len"] + b * 0.40 * g["w"] * g["e_wid"] + (0.5 * g["h"] - hgt) * g["down"]
                uv, z = self._project_world(f, X)
                u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
                if z[0] <= 0.5 or not (0 <= u < W and 0 <= v < H):
                    continue
                n_in += 1
                win = (slice(max(v - 1, 0), v + 2), slice(max(u - 1, 0), u + 2))
                if np.any((A["inst"][win] == tid) & (np.abs(A["depth"][win] - z[0]) <= tol)):
                    n_vis += 1
        return n_vis, n_in

    def static_fit_residual(self, frames, tid):
        # rms reprojection error (px) of the best static point for the vehicle's reference point
        pix, Rs, Cs = [], [], []
        for f in frames:
            T = self.obj_transform(f, tid)
            if T is None or f not in self.R:
                return np.nan
            q = world_to_cam(T[1], self.R[f], self.C[f])
            if q[2] <= 1.0:
                return np.nan
            p = self.cam.project(q)
            if not (0 <= p[0] < VK_CROP_W and 0 <= p[1] < VK_CROP_H):
                return np.nan
            pix.append(p); Rs.append(self.R[f]); Cs.append(self.C[f])
        return triangulate_multiview(pix, [self.cam] * len(frames), Rs, Cs)[1]


def _vk_warp_error(sc, i, j, stride=4):
    A, B = sc.frame_gt(i), sc.frame_gt(j)
    H, W = A["depth"].shape
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    z = A["depth"][vv, uu]; st_ = (A["inst"][vv, uu] < 0) & np.isfinite(z) & (z < 80)
    X = sc.C[i] + (sc.cam.rays(uu[st_].astype(float), vv[st_].astype(float)) * z[st_][:, None]) @ sc.R[i].T
    Xj = (X - sc.C[j]) @ sc.R[j]
    ok = Xj[:, 2] > 0.5
    p = sc.cam.project(Xj[ok])
    ins = (p[:, 0] >= 0) & (p[:, 0] <= W - 1) & (p[:, 1] >= 0) & (p[:, 1] <= H - 1)
    zo = B["depth"][np.round(p[ins, 1]).astype(int), np.round(p[ins, 0]).astype(int)]
    zp = Xj[ok][ins, 2]
    m = np.isfinite(zo)
    return float(np.median(np.abs(zo[m] - zp[m]) / zo[m])) if m.sum() > 200 else np.nan

def _vk_object_error(sc, pairs):
    # median relative depth disagreement of moving-vehicle pixels warped rigidly from i to j
    errs = []
    for i, j in pairs:
        g = sc.pair_gt(i, j)
        B = sc.frame_gt(j)
        m = (g["mover_id"] >= 0) & g["valid"]
        if m.sum() < 50:
            continue
        ui = np.round(g["pB"][m][:, 0]).astype(int); vi = np.round(g["pB"][m][:, 1]).astype(int)
        zo = B["depth"][vi, ui]; ok = np.isfinite(zo) & (B["inst"][vi, ui] >= 0)
        if ok.sum() > 20:
            errs.append(np.median(np.abs(zo[ok] - g["zB"][m][ok]) / zo[ok]))
    return float(np.median(errs)) if errs else np.nan

def _vk_inst_mode(sc, frames):
    # does instance value v mean track v-1? (bbox overlap must agree for >= 80% of the instances)
    agree, total = 0, 0
    for f in frames:
        raw = sc._inst_raw(f)
        m = sc._match_values(f, raw)
        for v, t in m.items():
            total += 1; agree += int(t == v - 1)
    return "track+1" if total and agree / total >= 0.8 else "overlap"

VK_SCENE_OBJS, VK_A3, VK_PAIR_META, _vk_rows, _vk_prev_rows, _vk_kview = {}, {}, [], [], [], []
VK_GT_STAGE = Stage("vk:gt", "vk", dict(regimes=[PREREG["eps_parallel_max"], PREREG["eps_perp_min"]], erosion=PREREG["erosion_px"],
                                        min_px=PREREG["min_object_pixels"]), upstream=[VK_STAGE.fp])
VK_GT_DIR = os.path.join(VK_DIR, "gt")
_VK_RESTORED_A3 = None
_VK_GT_READY = bool(VK_STATUS["available"] and VK_GT_STAGE.available() is not None and VK_GT_STAGE.done())
if _VK_GT_READY:
    with open(os.path.join(VK_GT_DIR, "a3.json")) as f:
        _VK_RESTORED_A3 = json.load(f)["a3"]
if VK_STATUS["available"] and _VK_RESTORED_A3 is not None:
    # conventions were decided when the stage was computed; reuse them exactly
    for name, v in _VK_RESTORED_A3.items():
        if not v["passed"]:
            continue
        sc = VKScene(name)
        sc.inverted, sc.yaw_axis, sc.yaw_sign, sc.inst_mode = v["convention"] == "inverted", v["yaw_axis"], v["yaw_sign"], v["instance_mode"]
        sc._set_poses()
        VK_SCENE_OBJS[name] = sc
    VK_A3 = _VK_RESTORED_A3
elif VK_STATUS["available"]:
    with stage("vkitti_gt"):
        for name in VK_SELECTION:
            sc = VKScene(name)
            pairs = [tuple(p[:2]) for p in VK_SELECTION[name]["pairs"]]
            test = pairs[: min(12, len(pairs))]
            sc.inst_mode = _vk_inst_mode(sc, sorted({f for p in test for f in p})[:6])
            eA = np.nanmedian([_vk_warp_error(sc, i, j) for i, j in test])
            sc.inverted = True; sc._set_poses(); sc._cache.clear()
            eB = np.nanmedian([_vk_warp_error(sc, i, j) for i, j in test])
            sc.inverted = bool(np.isfinite(eB) and (not np.isfinite(eA) or eB < 0.5 * eA)); sc._set_poses(); sc._cache.clear()
            werr = eB if sc.inverted else eA
            obj = {}
            for axis in ("y", "z"):
                for sign in (1.0, -1.0):
                    sc.yaw_axis, sc.yaw_sign = axis, sign; sc._cache.clear()
                    obj[(axis, sign)] = _vk_object_error(sc, test)
            (sc.yaw_axis, sc.yaw_sign), oerr = min(obj.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else 1e9)
            sc._cache.clear()
            VK_A3[name] = dict(n_frames=len(sc.R), n_test_pairs=len(test), warp_err_documented=float(eA), warp_err_inverted=float(eB),
                               convention="inverted" if sc.inverted else "documented", static_warp_err=float(werr),
                               yaw_axis=sc.yaw_axis, yaw_sign=sc.yaw_sign, vehicle_warp_err=float(oerr), instance_mode=sc.inst_mode,
                               passed=bool(np.isfinite(werr) and werr <= 0.05 and np.isfinite(oerr) and oerr <= 0.05))
            if not VK_A3[name]["passed"]:
                continue
            VK_SCENE_OBJS[name] = sc
        VK_STATUS["available"] = len(VK_SCENE_OBJS) > 0
        if not VK_STATUS["available"]:
            VK_STATUS["note"] = "no scene passed the ground-truth self-consistency test (A3)"
VK_STATUS["available"] = bool(VK_STATUS["available"] and VK_SCENE_OBJS)

if VK_STATUS["available"]:
    def vk_meta_pair(name, n, i, j):
        sc = VK_SCENE_OBJS[name]
        return dict(set="VK", index=n, seq=name, frames=[int(i), int(j)], dir=sc.dir,
                    baseline=float(np.linalg.norm(sc.C[j] - sc.C[i])))

    _n = 0
    for name, sc in VK_SCENE_OBJS.items():
        for (i, j, b, rot) in VK_SELECTION[name]["pairs"]:
            VK_PAIR_META.append(vk_meta_pair(name, _n, i, j)); _n += 1

    def vk_mover_stats(sc, i, j):
        g = sc.pair_gt(i, j)
        b = sc.C[j] - sc.C[i]
        out = []
        for k, mv in enumerate(g["movers"]):
            mask = g["mover_id"] == k
            em = erode_mask(mask, PREREG["erosion_px"]) & g["visible"]
            n = int(em.sum())
            bb = sc.bbox.get((i, mv["track"]))
            inside = bb is not None and bb.left >= sc.x0 and bb.right <= sc.x0 + VK_CROP_W - 1
            d = mv["pos_j"] - mv["pos_i"]
            rec = dict(mover=k, track=mv["track"], label=mv["label"], s=float(d @ b / (b @ b)),
                       speed_ratio=float(np.linalg.norm(d) / max(np.linalg.norm(b), 1e-9)),
                       area_px=int(mask.sum()), n_eroded=n, occupancy=float(bb.occupancy_ratio) if bb is not None else np.nan,
                       truncation=float(bb.truncation_ratio) if bb is not None else np.nan, inside_crop=bool(inside),
                       vis_in_b=float(g["visible"][mask].mean()) if mask.any() else 0.0)
            n_vis, n_in = sc.wheel_contact(i, mv["track"], sc.frame_gt(i))
            rec.update(wheels_visible=n_vis, wheels_in_view=n_in,
                       contact=("visible" if n_vis >= 1 else "hidden" if n_in >= 2 else "undetermined") if np.isfinite(n_vis) else "undetermined")
            if n > 0:
                Zt = g["depth"][em].astype(np.float64); zg = g["zgeo"][em]
                rec.update(Z_true=float(np.median(Zt)), eps_med=float(np.median(g["eps"][em])), eps_max=float(np.max(g["eps"][em])),
                           x_geo=float(np.median(np.log(np.where(zg > 0, zg, np.nan) / Zt))) if np.all(zg > 0) else float("nan"),
                           parallax_med=float(np.median(g["parallax"][em])))
            out.append(rec)
        return out

    def _vk_table(metas):
        rows = []
        for meta in metas:
            sc = VK_SCENE_OBJS[meta["seq"]]
            i, j = meta["frames"]
            for r in vk_mover_stats(sc, i, j):
                r = {("gt_" + k if k not in ("mover", "track", "label", "s", "speed_ratio") else k): v for k, v in r.items()}
                r.update(set=meta["set"], scene=meta["index"], seq=meta["seq"], uid=f"{meta['set']}-{meta['index']:04d}-{r['mover']}",
                         cls="vehicle", cam_kind="driving", motion_type="vehicle")
                rows.append(r)
        df = pd.DataFrame(rows)
        if not len(df):
            return df
        df["x_par"] = np.where(df["s"] < 1, -np.log(np.clip(1 - df["s"], 1e-9, None)), np.nan)
        df["regime"] = np.where(df["gt_eps_med"] < PREREG["eps_parallel_max"], "parallel",
                                np.where(df["gt_eps_med"] >= PREREG["eps_perp_min"], "perpendicular", "intermediate"))
        df["measurable"] = vk_measurable(df)
        df["identifiable"] = df["gt_parallax_med"] >= PREREG["parallax_min_px"]
        return df

    with stage("vkitti_gt"):
        if not _VK_GT_READY and not VK_GT_STAGE.done():
            os.makedirs(VK_GT_DIR, exist_ok=True)
            _vk_table(VK_PAIR_META).to_csv(os.path.join(VK_GT_DIR, "movers_pairs.csv"), index=False)
            # prevalence 1: moving-vehicle pixels in the two-view pairs
            for name, sc in VK_SCENE_OBJS.items():
                e_all = []
                for meta in [m for m in VK_PAIR_META if m["seq"] == name]:
                    g = sc.pair_gt(*meta["frames"])
                    m = (g["mover_id"] >= 0) & g["visible"]
                    e_all.append(g["eps"][m])
                e = np.concatenate(e_all) if e_all else np.zeros(0)
                if len(e):
                    _vk_prev_rows.append(dict(dataset="Virtual KITTI 2", seq=name, n_pairs=len(e_all), n_moving_pixels=int(len(e)),
                                              frac_parallel=float(np.mean(e < PREREG["eps_parallel_max"])),
                                              frac_intermediate=float(np.mean((e >= PREREG["eps_parallel_max"]) & (e < PREREG["eps_perp_min"]))),
                                              frac_perpendicular=float(np.mean(e >= PREREG["eps_perp_min"])),
                                              frac_parallel_no_static_solution=float("nan"), median_eps=float(np.median(e))))
                # prevalence 2: every moving vehicle in every frame, static fit over K frames (text GT only)
                frames = sorted(sc.R)
                fidx = {f: n for n, f in enumerate(frames)}
                for (f, tid), r in sc.bbox.items():
                    if not r.isMoving or (r.right - r.left) * (r.bottom - r.top) < VK_DESIGN["min_vehicle_px"]:
                        continue
                    if not (r.left >= sc.x0 and r.right <= sc.x0 + VK_CROP_W - 1):
                        continue
                    gap = next((g_ for g_ in range(1, VK_DESIGN["max_gap_frames"] + 1)
                                if f + g_ in sc.C and np.linalg.norm(sc.C[f + g_] - sc.C[f]) >= VK_DESIGN["pair_baseline_m"][0]), None)
                    if gap is None:
                        continue
                    for K in (2, 4, 8):
                        res = sc.static_fit_residual([f + k * gap for k in range(K)], tid)
                        if np.isfinite(res):
                            _vk_kview.append(dict(seq=name, frame=f, track=tid, K=K, gap=gap, residual_px=float(res)))
            pd.DataFrame(_vk_prev_rows).to_csv(os.path.join(VK_GT_DIR, "prevalence_pixels.csv"), index=False)
            pd.DataFrame(_vk_kview).to_csv(os.path.join(VK_GT_DIR, "prevalence_kview.csv"), index=False)
            with open(os.path.join(VK_GT_DIR, "a3.json"), "w") as f:
                json.dump(dict(a3=VK_A3, pairs=VK_PAIR_META), f, indent=1, default=str)
            VK_GT_STAGE.commit([VK_GT_DIR])
    MOVERS_VK = pd.read_csv(os.path.join(VK_GT_DIR, "movers_pairs.csv"))
    # v6.1 measurability (D34): the CONVOY-3D pixel rule, recomputed from the stored columns so that a
    # restored v6.0 checkpoint gives the same table; the registered v6.0 rule is kept for the record
    MOVERS_VK["measurable_v6_0"] = ((MOVERS_VK["gt_n_eroded"] >= PREREG["min_object_pixels"]) & (MOVERS_VK["gt_occupancy"] >= 0.85)
                                    & MOVERS_VK["gt_inside_crop"].astype(bool) & (MOVERS_VK["gt_truncation"] <= 0.05)
                                    & (MOVERS_VK["gt_vis_in_b"] >= 0.85))
    MOVERS_VK["measurable"] = vk_measurable(MOVERS_VK)
    for _sc in VK_SCENE_OBJS.values():
        _sc._cache.clear()
    PREVALENCE_VK = pd.read_csv(os.path.join(VK_GT_DIR, "prevalence_pixels.csv"))
    KVIEW_VK = pd.read_csv(os.path.join(VK_GT_DIR, "prevalence_kview.csv"))
    with open(os.path.join(VK_GT_DIR, "a3.json")) as f:
        _a3 = json.load(f)
    VK_A3 = _a3["a3"]
    MOVERS_VK.to_csv(os.path.join(MANIFEST_DIR, "vkitti_movers.csv"), index=False)
    print(f"{'scene':<9}{'frames':>7}{'pairs':>6}  extrinsics   static warp  yaw   vehicle warp  instances  A3")
    for name, v in VK_A3.items():
        print(f"{name:<9}{v['n_frames']:>7}{sum(m['seq'] == name for m in VK_PAIR_META):>6}"
              f"  {v['convention']:<11}{v['static_warp_err']:>12.3f}  {('+' if v['yaw_sign'] > 0 else '-') + v['yaw_axis']:<5}"
              f"{v['vehicle_warp_err']:>12.3f}  {v['instance_mode']:<10} {'PASS' if v['passed'] else 'FAIL'}")
    print(f"\nVirtual KITTI 2: {len(MOVERS_VK)} moving vehicles in {len(VK_PAIR_META)} pairs; regimes "
          f"{MOVERS_VK.groupby('regime').size().to_dict()}; measurable {int(MOVERS_VK['measurable'].sum())}, "
          f"identifiable {int((MOVERS_VK['measurable'] & MOVERS_VK['identifiable']).sum())}")
    _lo, _hi = PREREG["h9_x_range"]
    _par = MOVERS_VK["regime"] == "parallel"
    _inx = pd.to_numeric(MOVERS_VK["gt_x_geo"], errors="coerce").between(_lo, _hi)
    _att = [("moving vehicles in the pairs", np.ones(len(MOVERS_VK), bool)), ("parallel regime (|eps| < 1 px)", _par),
            (f"... and >= {PREREG['min_object_pixels']} eroded visible px", _par & (MOVERS_VK["gt_n_eroded"] >= PREREG["min_object_pixels"])),
            (f"... and >= {100 * PREREG['vk_min_occupancy']:.0f}% of the silhouette visible (measurable)", _par & MOVERS_VK["measurable"]),
            (f"... and parallax >= {PREREG['parallax_min_px']:g} px (identifiable)", _par & MOVERS_VK["measurable"] & MOVERS_VK["identifiable"]),
            (f"... and x_geo in [{_lo}, {_hi}] (the H9 sample)", _par & MOVERS_VK["measurable"] & MOVERS_VK["identifiable"] & _inx),
            ("H9 sample under the v6.0 rule (whole box in the crop, truncation <= 5%, visible in view 2 >= 85%)",
             _par & MOVERS_VK["measurable_v6_0"] & MOVERS_VK["identifiable"] & _inx)]
    VK_ATTRITION = pd.DataFrame([dict(step=k, vehicles=int(np.sum(v)), pairs=int(MOVERS_VK.loc[np.asarray(v, bool), "scene"].nunique()))
                                 for k, v in _att])
    print("\nVirtual KITTI 2 sample, step by step:\n" + VK_ATTRITION.to_string(index=False))
    if "gt_contact" in MOVERS_VK:
        _pm = MOVERS_VK[_par & (MOVERS_VK["gt_n_eroded"] >= PREREG["x6_min_eroded_px"])]
        print("ground contact of the parallel vehicles with >= 40 eroded px (wheel contact points visible / hidden / "
              "undetermined):", _pm["gt_contact"].value_counts().to_dict())
    print(PREVALENCE_VK.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    _kv = KVIEW_VK.assign(invisible=KVIEW_VK["residual_px"] < 1.0).groupby("K").agg(
        n=("invisible", "size"), share_static_fit_below_1px=("invisible", "mean"), median_residual_px=("residual_px", "median"))
    print("\nmoving vehicles whose K-frame track a static point fits within 1 px:\n" + _kv.to_string(float_format=lambda v: f"{v:.3f}"))
else:
    MOVERS_VK = PREVALENCE_VK = KVIEW_VK = VK_ATTRITION = pd.DataFrame()
    print("Virtual KITTI 2 is not used in this run:", VK_STATUS["note"])

release_memory(); _ = mem_report("Virtual KITTI 2 ground truth")

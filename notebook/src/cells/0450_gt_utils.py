# PLACEHOLDER
# 3.1.5 Exact ground truth from a render: back-projection, correspondences between frames,
#       per-pixel epipolar offset and static-hypothesis depth, and per-mover statistics.

def frame_camera(scene, k):
    cd = scene["cams"][k]
    return PinholeCamera.from_dict(cd), np.asarray(cd["R"], np.float64), np.asarray(cd["C"], np.float64)

def backproject(depth, cam, R, C):
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    D = cam.rays(uu.astype(np.float64), vv.astype(np.float64)) @ R.T
    with np.errstate(invalid="ignore"):
        return C + np.where(np.isfinite(depth), depth, np.nan)[..., None].astype(np.float64) * D

def gt_correspondence(scene, rA, a, b, rB=None, tol_rel=0.01, tol_abs=0.02):
    # For every pixel of frame a: its exact position in frame b (static points stay fixed,
    # mover points translate by (b-a) * delta). 'visible' also checks occlusion in frame b.
    camA, RA, CA = frame_camera(scene, a)
    camB, RB, CB = frame_camera(scene, b)
    P = backproject(rA["depth"], camA, RA, CA)
    mid = rA["mover_id"].astype(np.int64)
    if scene["movers"]:
        deltas = np.array([m["delta"] for m in scene["movers"]], np.float64)
        P = P + np.where(mid[..., None] >= 0, deltas[np.maximum(mid, 0)] * (b - a), 0.0)
    Xc = (P - CB) @ RB
    with np.errstate(invalid="ignore", divide="ignore"):
        pB = camB.project(Xc)
    zB = Xc[..., 2]
    valid = np.isfinite(pB).all(-1) & (zB > 0.1)
    valid &= (pB[..., 0] >= 0) & (pB[..., 0] <= camB.W - 1) & (pB[..., 1] >= 0) & (pB[..., 1] <= camB.H - 1)
    visible = valid.copy()
    if rB is not None:
        ui = np.clip(np.round(np.nan_to_num(pB[..., 0])).astype(int), 0, camB.W - 1)
        vi = np.clip(np.round(np.nan_to_num(pB[..., 1])).astype(int), 0, camB.H - 1)
        zb_obs = rB["depth"][vi, ui]
        visible &= np.abs(zb_obs - zB) <= tol_rel * zB + tol_abs
        visible &= rB["mover_id"][vi, ui] == rA["mover_id"]
    return pB, zB, valid, visible

def pixel_epipolar_quantities(scene, a, b, pA_grid, pB):
    # per-pixel exact epipolar offset, static-hypothesis depth, translational parallax
    camA, RA, CA = frame_camera(scene, a)
    camB, RB, CB = frame_camera(scene, b)
    R21, t21 = relative_pose(RA, CA, RB, CB)
    F = fundamental_matrix(camA.K, camB.K, R21, t21)
    eps = np.abs(epipolar_offset(pA_grid, pB, F))
    zgeo, _ = triangulate_depth(pA_grid, pB, camA.K, camB.K, R21, t21)
    par = translational_parallax(pA_grid, pB, camA.K, camB.K, R21)
    return eps, zgeo, par

def erode_mask(mask, px):
    if px <= 0:
        return mask
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k, borderValue=0).astype(bool)

def mover_silhouette_area(scene, frame_k, j):
    # pixel count of mover j rendered alone (no occluders) -> visible fraction
    cam, R, C = frame_camera(scene, frame_k)
    m = scene["movers"][j]
    off = np.asarray(m["delta"]) * frame_k
    boxes = [scene["boxes"][i] for i in m["box_ids"]]
    uv = _project_boxes(boxes, off, cam, R, C, znear=0.05)
    if uv is None:
        return 0
    u0, v0 = np.maximum(np.floor(uv.min(0)).astype(int) - 1, 0)
    u1, v1 = np.minimum(np.ceil(uv.max(0)).astype(int) + 1, [cam.W - 1, cam.H - 1])
    vv, uu = np.mgrid[v0:v1 + 1, u0:u1 + 1]
    D = cam.rays(uu.ravel().astype(np.float64), vv.ravel().astype(np.float64)) @ R.T
    h = cast_rays(scene, C, D, frame_k, True, pix=(uu.ravel(), vv.ravel()), cam=cam, R=R, only_movers=[j])
    return int(np.isfinite(h["t"]).sum())

def mover_gt_stats(scene, renders, a=0, b=1):
    # Per-mover ground truth between frames a and b, measured on the mover's visible,
    # eroded pixels in frame a (the pixels every model-based statistic is read from).
    rA, rB = renders[a], renders[b]
    camA, _, _ = frame_camera(scene, a)
    H, W = rA["depth"].shape
    vv, uu = np.mgrid[0:H, 0:W]
    pA = np.stack([uu, vv], -1).astype(np.float64)
    pB, zB, valid, visible = gt_correspondence(scene, rA, a, b, rB)
    eps, zgeo, par = pixel_epipolar_quantities(scene, a, b, pA, pB)
    out = []
    for j, m in enumerate(scene["movers"]):
        mask = rA["mover_id"] == j
        area = int(mask.sum())
        sil = mover_silhouette_area(scene, a, j)
        em = erode_mask(mask, PREREG["erosion_px"]) & visible
        n = int(em.sum())
        rec = dict(mover=j, area_px=area, visible_frac=float(area / sil) if sil else 0.0, n_eroded=n)
        if n > 0:
            Zt = rA["depth"][em].astype(np.float64)
            rec.update(Z_true=float(np.median(Zt)), eps_med=float(np.median(eps[em])),
                       eps_max=float(np.max(eps[em])), Zgeo_med=float(np.median(zgeo[em])),
                       x_geo=float(np.median(np.log(np.where(zgeo[em] > 0, zgeo[em], np.nan) / Zt)))
                       if np.all(zgeo[em] > 0) else float("nan"),
                       parallax_med=float(np.median(par[em])),
                       vis_in_b=float(visible[mask].mean()) if area else 0.0)
        out.append(rec)
    return out

print("Ground-truth utilities ready.")
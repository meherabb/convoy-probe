# PLACEHOLDER
# 4.6 Measurement protocol shared by every model and baseline.
#   * static pixels: finite ground-truth depth inside the pre-registered range, not on a mover,
#     not within 9 px of a mover, not under a mover's moving shadow (Virtual KITTI 2: not on or
#     within 9 px of any vehicle, moving or parked);
#   * one scale factor per image and model: median(Z_true / Z_pred) over static pixels;
#   * validity-gate statistics per image: OLS slope of log Z_pred on log Z_true and AbsRel
#     after scale alignment, over static pixels;
#   * per mover: rho = median log(scale * Z_pred / Z_true) over its visible pixels eroded
#     by 2 px;
#   * dynamic-mask scores: threshold = 95th percentile over the image's static pixels
#     (5% false-positive rate); recall = share of mover pixels above it; AUROC of mover
#     pixels against a fixed random sample of static pixels.
from scipy.stats import rankdata

def static_mask(r, kind):
    d = r["depth"]
    mv = r["mover_id"] >= 0
    near = cv2.dilate(mv.astype(np.uint8), np.ones((19, 19), np.uint8)).astype(bool)
    lo, hi = PREREG["static_depth_range_real"] if kind == "real" else PREREG["static_depth_range_synth"]
    m = np.isfinite(d) & (d >= lo) & (d <= hi) & ~near
    if kind == "real":
        m &= r["sem"] == SEM["static_box"]
    else:
        m &= ~r["mover_shadow"]
    return m

def vk_static_mask(depth, inst):
    # Virtual KITTI 2: finite depth in range, not on any vehicle (moving or parked) or within 9 px of one
    veh = cv2.dilate((inst >= 0).astype(np.uint8), np.ones((19, 19), np.uint8)).astype(bool)
    lo, hi = PREREG["static_depth_range_vk"]
    return np.isfinite(depth) & (depth >= lo) & (depth <= hi) & ~veh

def mover_masks(scene, r0, r1, a=0, b=1):
    # eroded pixels of every mover in frame a that are visible in frame b (the measurement pixels)
    pB, zB, valid, vis = gt_correspondence(scene, r0, a, b, r1)
    out = []
    for j in range(len(scene["movers"])):
        out.append(erode_mask(r0["mover_id"] == j, PREREG["erosion_px"]) & vis)
    return out, pB, vis

def scale_and_gate(pred, gt, smask, max_px=40000, rng_seed=0):
    p, g = pred[smask].astype(np.float64), gt[smask].astype(np.float64)
    ok = np.isfinite(p) & (p > 0)
    p, g = p[ok], g[ok]
    if len(p) < 100:
        return dict(scale=np.nan, slope=np.nan, absrel=np.nan, n_static=int(len(p)))
    s = float(np.median(g / p))
    if len(p) > max_px:
        sel = np.random.default_rng(rng_seed).choice(len(p), max_px, replace=False)
        p, g = p[sel], g[sel]
    slope = float(np.polyfit(np.log(g), np.log(p), 1)[0])
    absrel = float(np.mean(np.abs(s * p - g) / g))
    return dict(scale=s, slope=slope, absrel=absrel, n_static=int(ok.sum()))

def mover_rho(pred, gt, mask, scale):
    p = pred[mask].astype(np.float64); g = gt[mask].astype(np.float64)
    ok = np.isfinite(p) & (p > 0) & np.isfinite(g)
    if ok.sum() < PREREG["min_object_pixels"] or not np.isfinite(scale):
        return np.nan
    return float(np.median(np.log(scale * p[ok] / g[ok])))

def _auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def mask_scores(score, smask, masks, seed=0, n_neg=20000):
    # recall at the image's 5%-FPR threshold and AUROC, for every mover mask
    s_static = score[smask]
    s_static = s_static[np.isfinite(s_static)]
    if len(s_static) < 100:
        return [dict(recall=np.nan, auroc=np.nan) for _ in masks]
    thr = np.quantile(s_static, 1 - PREREG["mask_fpr"])
    neg = s_static if len(s_static) <= n_neg else np.random.default_rng(seed).choice(s_static, n_neg, replace=False)
    out = []
    for m in masks:
        pos = score[m]; pos = pos[np.isfinite(pos)]
        if len(pos) < PREREG["min_object_pixels"]:
            out.append(dict(recall=np.nan, auroc=np.nan)); continue
        out.append(dict(recall=float(np.mean(pos > thr)), auroc=_auroc(pos, neg)))
    return out

M1_RESOLUTION_PX = 1e-6        # D27: exact offsets are compared at 1e-6 px (float64 noise is ~1e-11 px)

def epipolar_residual_map(scene, p_obs, a=0, b=1):
    # M1: |eps| of every pixel's correspondence under the true relative pose (frames a -> b). Rounded to
    # 1e-6 px: with exact correspondences, the static scene and a parallel mover both have eps = 0 up to
    # floating-point noise, and that noise must not decide which of the two a threshold flags.
    H, W = p_obs.shape[:2]
    vv, uu = np.mgrid[0:H, 0:W]
    cam0, R0, C0 = frame_camera(scene, a); cam1, R1, C1 = frame_camera(scene, b)
    R21, t21 = relative_pose(R0, C0, R1, C1)
    F = fundamental_matrix(cam0.K, cam1.K, R21, t21)
    e = np.abs(epipolar_offset(np.stack([uu, vv], -1).astype(np.float64), p_obs, F))
    return np.round(e / M1_RESOLUTION_PX) * M1_RESOLUTION_PX

def rigid_flow_residual_map(scene, depth_scaled, p_obs, a=0, b=1):
    # M2: |observed correspondence - correspondence implied by the depth and the true motion|
    cam0, R0, C0 = frame_camera(scene, a); cam1, R1, C1 = frame_camera(scene, b)
    H, W = depth_scaled.shape
    vv, uu = np.mgrid[0:H, 0:W]
    X0 = cam0.rays(uu.astype(np.float64), vv.astype(np.float64)) * depth_scaled[..., None].astype(np.float64)
    R21, t21 = relative_pose(R0, C0, R1, C1)
    X1 = X0 @ R21.T + t21
    with np.errstate(divide="ignore", invalid="ignore"):
        p_rig = cam1.project(X1)
    res = np.linalg.norm(p_obs - p_rig, axis=-1)
    res[X1[..., 2] <= 0.05] = np.nan
    return res

def classical_triangulation_depth(scene, p_obs, a=0, b=1):
    cam0, R0, C0 = frame_camera(scene, a); cam1, R1, C1 = frame_camera(scene, b)
    H, W = p_obs.shape[:2]
    vv, uu = np.mgrid[0:H, 0:W]
    R21, t21 = relative_pose(R0, C0, R1, C1)
    z, _ = triangulate_depth(np.stack([uu, vv], -1).astype(np.float64), p_obs, cam0.K, cam1.K, R21, t21)
    return z

def flow_to_pobs(flow):
    H, W = flow.shape[:2]
    vv, uu = np.mgrid[0:H, 0:W]
    return np.stack([uu, vv], -1).astype(np.float64) + flow

print("Measurement protocol ready.")
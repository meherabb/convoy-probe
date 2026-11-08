# PLACEHOLDER
# 3.10.0 LiDAR ground truth shared by the real-driving data (KITTI tracking, Argoverse 2; v6.3).
#        LiDAR returns are sparse, so a far surface can show through the gaps between the returns of
#        a near one. A pixel is therefore kept only if its nearest return is also the nearest surface
#        in a small window around it (within max(10%, 1 m) of the window minimum); pixels that fail
#        this test carry no ground truth. This is the usual occlusion test for projected LiDAR.
from scipy.ndimage import minimum_filter as _minimum_filter


def lidar_zbuffer(uv, z, H, W, window=5, tol_rel=0.10, tol_abs=1.0):
    """Nearest return per pixel after the occlusion test.
    uv (N, 2) pixel coordinates, z (N,) depth. Returns D (H, W) float32 (NaN: no ground truth) and
    idx (H, W) int64, the index of the point that gives the pixel its depth (-1: none)."""
    uv = np.asarray(uv, np.float64); z = np.asarray(z, np.float64)
    u = np.round(uv[:, 0]).astype(np.int64) if len(uv) else np.zeros(0, np.int64)
    v = np.round(uv[:, 1]).astype(np.int64) if len(uv) else np.zeros(0, np.int64)
    ok = np.isfinite(z) & (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ids = np.nonzero(ok)[0]
    D = np.full(H * W, np.inf, np.float64)
    idx = np.full(H * W, -1, np.int64)
    if len(ids):
        pix = v[ids] * W + u[ids]
        order = np.lexsort((z[ids], pix))
        ps = pix[order]
        first = np.ones(len(order), bool)
        first[1:] = ps[1:] != ps[:-1]
        D[ps[first]] = z[ids[order[first]]]
        idx[ps[first]] = ids[order[first]]
    D = D.reshape(H, W); idx = idx.reshape(H, W)
    if window and window > 1:
        Dmin = _minimum_filter(D, size=int(window), mode="constant", cval=np.inf)
        hidden = np.isfinite(D) & (D > Dmin + np.maximum(tol_rel * Dmin, tol_abs))
        D[hidden] = np.inf
        idx[hidden] = -1
    D[~np.isfinite(D)] = np.nan
    return D.astype(np.float32), idx


def sample_gray(img, uv):
    """Bilinear grey value of an RGB uint8 image at sub-pixel positions uv (N, 2); NaN outside."""
    g = img.astype(np.float32).mean(-1) if img.ndim == 3 else img.astype(np.float32)
    uv = np.asarray(uv, np.float32).reshape(-1, 1, 2)
    out = cv2.remap(g, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return out.reshape(-1)


def lidar_vehicle_stats(sc, uv_i, uv_j, Z, in_j):
    """Epipolar offset, static-hypothesis depth and parallax of a vehicle's ground-truth pixels."""
    eps, zgeo, par = pixel_epipolar_quantities(sc, 0, 1, uv_i, uv_j)
    ok = in_j & np.isfinite(eps)
    rec = dict(n_lidar_px=int(ok.sum()))
    if ok.sum() >= 3:
        zg = zgeo[ok]
        rec.update(Z_true=float(np.median(Z[ok])), eps_med=float(np.median(eps[ok])), parallax_med=float(np.median(par[ok])),
                   x_geo=float(np.median(np.log(zg / Z[ok]))) if np.all(zg > 0) else float("nan"))
    return rec, ok


def lidar_mover_table(metas, set_):
    # one row per labelled vehicle of every pair (moving and parked), in the column layout of Virtual KITTI 2
    rows = []
    for m_ in metas:
        for k, v in enumerate(m_["vehicles"]):
            r = {("gt_" + a if a not in ("track", "label", "s", "speed_ratio", "state") else a): b
                 for a, b in v.items() if a != "pixels"}
            r.update(set=set_, scene=m_["index"], seq=m_["seq"], mover=k, uid=f"{set_}-{m_['index']:04d}-{k}", cls="vehicle",
                     cam_kind="driving", motion_type="vehicle")
            rows.append(r)
    return pd.DataFrame(rows)


def lidar_regimes(df):
    # Motion regimes on real data. Labelled boxes carry a few centimetres of annotation noise, which
    # enters the correspondence; the epipolar offset of parked vehicles (zero without noise) measures it.
    # Parallel regime: |eps| below the 90th percentile over parked vehicles, and at least 1 px.
    pk = df[df["state"] == "parked"]
    noise = (float(np.nanquantile(pk["gt_eps_med"], PREREG["kt_eps_noise_quantile"]))
             if "gt_eps_med" in pk and pk["gt_eps_med"].notna().sum() >= 10 else np.nan)
    tau = max(PREREG["eps_parallel_max"], noise if np.isfinite(noise) else PREREG["eps_parallel_max"])
    df["regime"] = np.where(df["gt_eps_med"] < tau, "parallel",
                            np.where(df["gt_eps_med"] >= max(PREREG["eps_perp_min"], 2 * tau), "perpendicular", "intermediate"))
    df["x_par"] = np.where(df["s"] < 1, -np.log(np.clip(1 - df["s"], 1e-9, None)), np.nan)
    df["measurable"] = df["gt_n_lidar_px"] >= PREREG["kt_min_lidar_px"]
    df["identifiable"] = df["gt_parallax_med"] >= PREREG["parallax_min_px"]
    return df, noise, tau, len(pk)


print("LiDAR ground-truth helpers ready (nearest return per pixel with a 5 x 5 occlusion test).")

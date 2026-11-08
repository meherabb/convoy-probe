"""Dynamic-mask scores and their evaluation at a fixed static false-positive rate (paper, Sec. 6.1).

M1 is the epipolar residual under the true relative pose (model-free, the most any two-view
geometric test can see); M2 is the rigid-flow residual, the distance between the observed
correspondence and the correspondence implied by the model's depth and the true camera motion.
Each image is thresholded at the 95th percentile of the score over its static pixels (5% false
positives) and a mover's recall is the share of its pixels above that threshold.

Mirrors notebook cell 4.6 (``0680_measurement.py``) with the camera passed explicitly.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .geometry import epipolar_offset, fundamental_matrix, relative_pose, triangulate_depth
from .registered import PREREG

__all__ = ["M1_RESOLUTION_PX", "auroc", "mask_scores", "epipolar_residual_map", "rigid_flow_residual_map",
           "classical_triangulation_depth", "flow_to_pobs"]

#: Exact offsets are compared at 1e-6 px: with exact correspondences the static scene and a
#: parallel mover both have eps = 0 up to float64 noise (~1e-11 px), which must not decide a threshold.
M1_RESOLUTION_PX = 1e-6


def auroc(pos, neg):
    """Area under the ROC curve of positive (mover) against negative (static) scores."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def mask_scores(score, smask, masks, seed=0, n_neg=20_000, fpr=None, min_pixels=None):
    """Recall at the image's static-FPR threshold and AUROC for every mover mask of one image."""
    fpr = PREREG["mask_fpr"] if fpr is None else fpr
    min_pixels = PREREG["min_object_pixels"] if min_pixels is None else min_pixels
    s_static = score[smask]
    s_static = s_static[np.isfinite(s_static)]
    if len(s_static) < 100:
        return [dict(recall=np.nan, auroc=np.nan) for _ in masks]
    thr = np.quantile(s_static, 1 - fpr)
    neg = s_static if len(s_static) <= n_neg else np.random.default_rng(seed).choice(s_static, n_neg, replace=False)
    out = []
    for m in masks:
        pos = score[m]
        pos = pos[np.isfinite(pos)]
        if len(pos) < min_pixels:
            out.append(dict(recall=np.nan, auroc=np.nan))
            continue
        out.append(dict(recall=float(np.mean(pos > thr)), auroc=auroc(pos, neg)))
    return out


def _pixel_grid(H, W):
    vv, uu = np.mgrid[0:H, 0:W]
    return np.stack([uu, vv], -1).astype(np.float64)


def epipolar_residual_map(p_obs, cam0, R0, C0, cam1, R1, C1):
    """M1: ``|eps|`` of every pixel's correspondence ``p_obs`` (H x W x 2) under the true pose."""
    H, W = p_obs.shape[:2]
    R21, t21 = relative_pose(R0, C0, R1, C1)
    F = fundamental_matrix(cam0.K, cam1.K, R21, t21)
    e = np.abs(epipolar_offset(_pixel_grid(H, W), p_obs, F))
    return np.round(e / M1_RESOLUTION_PX) * M1_RESOLUTION_PX


def rigid_flow_residual_map(depth_scaled, p_obs, cam0, R0, C0, cam1, R1, C1):
    """M2: ``|p_obs - p_rigid|`` with ``p_rigid`` implied by the (scale-aligned) depth and the true motion."""
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


def classical_triangulation_depth(p_obs, cam0, R0, C0, cam1, R1, C1):
    """Depth of every pixel triangulated from its correspondence under the true pose (the classical control).

    For a parallel mover it returns the static twin's depth ``Z / (1 - s)`` exactly (``beta_s = 1``).
    """
    H, W = p_obs.shape[:2]
    R21, t21 = relative_pose(R0, C0, R1, C1)
    z, _ = triangulate_depth(_pixel_grid(H, W), p_obs, cam0.K, cam1.K, R21, t21)
    return z


def flow_to_pobs(flow):
    """Correspondences ``p_obs`` (H x W x 2) from a dense optical flow field."""
    H, W = flow.shape[:2]
    return _pixel_grid(H, W) + flow

"""Dynamic masks see parallel movers only through a wrong depth; the twin test sees them through the contact."""
import numpy as np
import pandas as pd
import pytest

import convoy.geometry as G
import convoy.masks as M
import convoy.twin as T

H, W = 60, 80


def _two_views(s, perp=0.0):
    """A static random depth map seen from two cameras; a block of pixels moves by s*b (+ perp along the image y axis)."""
    rng = np.random.default_rng(0)
    cam = G.PinholeCamera(80.0, 80.0, 39.5, 29.5, W, H)
    R0, C0 = np.eye(3), np.zeros(3)
    R1, C1 = G.rot_ypr(0.02), np.array([1.5, 0.0, 0.2])
    b = C1 - C0
    depth = rng.uniform(5, 30, (H, W))
    vv, uu = np.mgrid[0:H, 0:W]
    X = G.cam_to_world(cam.rays(uu.astype(float), vv.astype(float)) * depth[..., None], R0, C0)
    mover = np.zeros((H, W), bool)
    mover[20:40, 30:55] = True
    X[mover] += s * b + perp * np.array([0.0, 1.0, 0.0])
    p_obs = cam.project(G.world_to_cam(X, R1, C1))
    return cam, (R0, C0), (R1, C1), depth, p_obs, mover


def test_m1_cannot_see_parallel_motion():
    cam, (R0, C0), (R1, C1), depth, p_obs, mover = _two_views(s=0.6)
    eps = M.epipolar_residual_map(p_obs, cam, R0, C0, cam, R1, C1)
    assert np.nanmax(eps) == 0.0                                  # static scene and parallel mover alike
    cam, (R0, C0), (R1, C1), depth, p_obs, mover = _two_views(s=0.6, perp=0.5)
    eps = M.epipolar_residual_map(p_obs, cam, R0, C0, cam, R1, C1)
    assert eps[~mover].max() == 0.0 and np.median(eps[mover]) > 0.1


def test_classical_triangulation_returns_the_static_twin():
    s = 0.6
    cam, (R0, C0), (R1, C1), depth, p_obs, mover = _two_views(s=s)
    z = M.classical_triangulation_depth(p_obs, cam, R0, C0, cam, R1, C1)
    np.testing.assert_allclose(z[~mover], depth[~mover], rtol=1e-6)
    np.testing.assert_allclose(z[mover], depth[mover] / (1 - s), rtol=1e-6)        # rho = x exactly
    rf = M.rigid_flow_residual_map(z, p_obs, cam, R0, C0, cam, R1, C1)
    assert np.nanmax(rf) < 1e-6                                                     # triangulated depth leaves no residual
    rf = M.rigid_flow_residual_map(depth, p_obs, cam, R0, C0, cam, R1, C1)          # the true depth does
    assert rf[~mover].max() < 1e-6 and np.median(rf[mover]) > 1.0


def test_mask_scores_threshold_at_five_percent_false_positives():
    rng = np.random.default_rng(1)
    score = rng.normal(0, 1, (H, W))
    smask = np.ones((H, W), bool)
    smask[:20] = False
    mover = ~smask
    score[mover] += 10.0
    same = np.zeros((H, W), bool)
    same[30:, :] = True                                            # static pixels: about 5% above the threshold
    out = M.mask_scores(score, smask, [mover, same], seed=0)
    assert out[0]["recall"] == 1.0 and out[0]["auroc"] == pytest.approx(1.0)
    assert out[1]["recall"] == pytest.approx(0.05, abs=0.02)
    assert np.isnan(M.mask_scores(score, smask, [np.zeros((H, W), bool)])[0]["recall"])


def test_twin_height_law_and_thresholds():
    s = np.array([-0.5, 0.0, 0.5, 0.8])
    np.testing.assert_allclose(T.twin_height(1.7, G.geometric_answer(s)), -1.7 * s / (1 - s))
    rng = np.random.default_rng(2)
    eps, h = np.abs(rng.normal(0, 0.3, 2000)), rng.normal(0, 0.15, 2000)
    thr = T.label_thresholds(eps, h)
    flags = T.label_flags(eps, h, np.zeros(2000, bool), thr)
    assert flags["det_epipolar"].mean() == pytest.approx(0.05, abs=0.002)
    assert flags["det_twin"].mean() == pytest.approx(0.05, abs=0.002)
    ft = T.flow_thresholds(eps, h)
    ff = T.flow_flags(eps, h, np.ones(2000, bool), ft)
    assert ff["det_twin"].mean() == pytest.approx(0.05, abs=0.002)


def test_h14_contrast_on_a_known_difference():
    rng = np.random.default_rng(3)
    n = 400
    cl = rng.integers(0, 150, n)
    twin = rng.random(n) < 0.3
    epi = rng.random(n) < 0.08
    r = T.h14_contrast(twin, epi, cl, seed=1, n_boot=2000)
    assert r["estimate"] == pytest.approx(twin.mean() - epi.mean())
    assert r["ci_lo"] < r["estimate"] < r["ci_hi"] and r["verdict"] == "SUPPORTED"
    assert T.h14_contrast(twin[:10], epi[:10], cl[:10], seed=1)["verdict"] == "NOT TESTABLE"


def test_detection_curve_is_symmetric_in_the_road():
    curve = T.detection_curve(1.7, np.zeros(10), 0.3, s_values=[-0.9, -0.1, 0.0, 0.1, 0.5])
    assert isinstance(curve, pd.DataFrame)
    np.testing.assert_allclose(curve["twin_height_m"], [-1.7 * s / (1 - s) for s in (-0.9, -0.1, 0.0, 0.1, 0.5)])
    assert list(curve["p_detect_twin"]) == [1.0, 0.0, 0.0, 0.0, 1.0]

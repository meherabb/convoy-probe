"""The package computes exactly what the notebook computed.

Each test loads the original definitions from the notebook's cell sources (``notebook/src/cells``)
and runs them and the package on the same inputs; results must be identical (bit for bit, or equal
as data frames). The notebook itself is the code that produced ``results/v6.6``.
"""
import hashlib

import numpy as np
import pandas as pd
import pytest
from helpers import cell_definitions, run_cell

import convoy.geometry as G
import convoy.masks as M
import convoy.probe as P
import convoy.stats as S
import convoy.twin as T
from convoy.registered import FLOWTWIN_DESIGN, GLOBAL_SEED, PREREG, TWIN_DESIGN, seed_for

B = 400          # bootstrap resamples in these tests; the code path does not depend on the number
eq = np.testing.assert_array_equal


@pytest.fixture(scope="module")
def nb_geo():
    return cell_definitions("0310_geometry_core.py")


@pytest.fixture(scope="module")
def nb_stats():
    return cell_definitions("0810_stats.py", RUN_CONFIG={"n_boot": B}, PREREG=PREREG)


def _cams(nb_geo):
    args = (700.0, 690.0, 319.5, 239.5, 640, 480)
    return nb_geo["PinholeCamera"](*args), G.PinholeCamera(*args)


# ---- seeds ---------------------------------------------------------------------------------------------------------

def test_seed_for_matches_notebook():
    nb = cell_definitions("0080_seeds.py", names={"seed_for"}, imports=False, hashlib=hashlib, GLOBAL_SEED=GLOBAL_SEED)
    for keys in [("prop-checks",), ("e1", "DUSt3R"), ("h14",), ("twin", "parallel"), ("oracle-e1", 0.5), ("main", 17)]:
        assert nb["seed_for"](*keys) == seed_for(*keys)


# ---- geometry (cell 2.1) -------------------------------------------------------------------------------------------

def test_camera_and_rotations(nb_geo):
    cam_nb, cam = _cams(nb_geo)
    rng = np.random.default_rng(0)
    eq(cam_nb.K, cam.K)
    u, v = rng.uniform(0, 640, 200), rng.uniform(0, 480, 200)
    eq(cam_nb.rays(u, v), cam.rays(u, v))
    X = rng.uniform(-5, 5, (200, 3)) + [0.0, 0.0, 12.0]
    eq(cam_nb.project(X), cam.project(X))
    assert cam_nb.scaled(0.8, 0.8).to_dict() == cam.scaled(0.8, 0.8).to_dict()
    for name in ("rot_x", "rot_y", "rot_z"):
        eq(nb_geo[name](0.37), getattr(G, name)(0.37))
    eq(nb_geo["rot_ypr"](0.1, -0.2, 0.3), G.rot_ypr(0.1, -0.2, 0.3))
    eq(nb_geo["random_rotation"](np.random.default_rng(5)), G.random_rotation(np.random.default_rng(5)))


def test_two_view_and_multiview(nb_geo):
    cam_nb, cam = _cams(nb_geo)
    rng = np.random.default_rng(1)
    R1, R2 = G.random_rotation(rng), G.random_rotation(rng)
    C1, C2 = rng.normal(size=3), rng.normal(size=3)
    X = rng.uniform(-5, 5, (100, 3))
    for name, args in [("world_to_cam", (X, R1, C1)), ("cam_to_world", (X, R1, C1)), ("skew", (C1,))]:
        eq(nb_geo[name](*args), getattr(G, name)(*args))
    R21, t21 = G.relative_pose(R1, C1, R2, C2)
    for a, b in zip(nb_geo["relative_pose"](R1, C1, R2, C2), (R21, t21)):
        eq(a, b)
    F = G.fundamental_matrix(cam.K, cam.K, R21, t21)
    eq(nb_geo["fundamental_matrix"](cam.K, cam.K, R21, t21), F)
    p1, p2 = rng.uniform(0, 640, (300, 2)), rng.uniform(0, 640, (300, 2))
    eq(nb_geo["epipolar_offset"](p1, p2, F), G.epipolar_offset(p1, p2, F))
    tri_nb = nb_geo["triangulate_depth"](p1, p2, cam.K, cam.K, R21, t21)
    for a, b in zip(tri_nb, G.triangulate_depth(p1, p2, cam.K, cam.K, R21, t21)):
        eq(a, b)
    eq(nb_geo["infinite_homography_map"](p1, cam.K, cam.K, R21), G.infinite_homography_map(p1, cam.K, cam.K, R21))
    eq(nb_geo["translational_parallax"](p1, p2, cam.K, cam.K, R21), G.translational_parallax(p1, p2, cam.K, cam.K, R21))
    Rs = [G.random_rotation(rng) for _ in range(4)]
    Cs = [rng.normal(size=3) for _ in range(4)]
    pix = [rng.uniform(100, 500, 2) for _ in range(4)]
    a, b = nb_geo["triangulate_multiview"](pix, [cam_nb] * 4, Rs, Cs), G.triangulate_multiview(pix, [cam] * 4, Rs, Cs)
    eq(a[0], b[0])
    assert a[1] == b[1]


# ---- measurement (cell 4.6) ----------------------------------------------------------------------------------------

def _scene(cam, rng):
    """Two views of a random static depth map, with exact correspondences and a moving patch."""
    H, W = 48, 64
    cam = cam.scaled(W / 640, H / 480)
    R0, C0 = np.eye(3), np.zeros(3)
    R1, C1 = G.rot_ypr(0.03, 0.01, 0.0), np.array([1.2, 0.0, 0.3])
    depth = rng.uniform(4, 30, (H, W))
    vv, uu = np.mgrid[0:H, 0:W]
    Xw = G.cam_to_world(cam.rays(uu.astype(float), vv.astype(float)) * depth[..., None], R0, C0)
    mover = np.zeros((H, W), bool)
    mover[10:30, 20:45] = True
    Xw[mover] += np.array([0.4, 0.05, 0.6])                                # a moving object (generic motion)
    p_obs = cam.project(G.world_to_cam(Xw, R1, C1))
    return cam, (R0, C0), (R1, C1), depth, p_obs, mover


def test_measurement_functions(nb_geo):
    cam_nb, cam = _cams(nb_geo)
    rng = np.random.default_rng(2)
    nb = cell_definitions("0680_measurement.py", names={"scale_and_gate", "mover_rho", "_auroc", "mask_scores",
                                                         "M1_RESOLUTION_PX", "epipolar_residual_map",
                                                         "rigid_flow_residual_map", "classical_triangulation_depth",
                                                         "flow_to_pobs"},
                          PREREG=PREREG, frame_camera=lambda scene, k: scene[k], **{k: nb_geo[k] for k in (
                              "relative_pose", "fundamental_matrix", "epipolar_offset", "triangulate_depth")})
    c, (R0, C0), (R1, C1), depth, p_obs, mover = _scene(cam, rng)
    scene = {0: (c, R0, C0), 1: (c, R1, C1)}
    eq(nb["epipolar_residual_map"](scene, p_obs), M.epipolar_residual_map(p_obs, c, R0, C0, c, R1, C1))
    pred = depth * np.exp(rng.normal(0, 0.05, depth.shape)) * 3.0
    eq(nb["rigid_flow_residual_map"](scene, pred, p_obs), M.rigid_flow_residual_map(pred, p_obs, c, R0, C0, c, R1, C1))
    eq(nb["classical_triangulation_depth"](scene, p_obs), M.classical_triangulation_depth(p_obs, c, R0, C0, c, R1, C1))
    flow = rng.normal(0, 2, depth.shape + (2,))
    eq(nb["flow_to_pobs"](flow), M.flow_to_pobs(flow))
    smask = ~mover
    assert nb["scale_and_gate"](pred, depth, smask) == P.scale_and_gate(pred, depth, smask)
    big = rng.uniform(1, 9, (300, 300))                                      # more than 40,000 static pixels
    assert nb["scale_and_gate"](big * 2, big, np.ones_like(big, bool)) == P.scale_and_gate(big * 2, big, np.ones_like(big, bool))
    g = P.scale_and_gate(pred, depth, smask)
    assert nb["mover_rho"](pred, depth, mover, g["scale"]) == P.mover_rho(pred, depth, mover, g["scale"])
    score = M.rigid_flow_residual_map(pred, p_obs, c, R0, C0, c, R1, C1)
    masks = [mover, mover & (np.arange(depth.shape[1]) < 30)[None], np.zeros_like(mover)]
    assert nb["mask_scores"](score, smask, masks, seed=3) == M.mask_scores(score, smask, masks, seed=3)
    pos, neg = rng.normal(1, 1, 500), rng.normal(0, 1, 800)
    assert nb["_auroc"](pos, neg) == M.auroc(pos, neg)


# ---- statistics (cells 6.A and 6.2) --------------------------------------------------------------------------------

def _data(rng, n=240, clusters=60):
    x = rng.uniform(-0.5, 1.9, n)
    cl = rng.integers(0, clusters, n)
    y = 0.2 * x + rng.normal(0, 0.3, n) + 0.1 * rng.normal(size=clusters)[cl]
    return x, y, cl


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float) and np.isnan(a[k]):
            assert np.isnan(b[k]), k
        else:
            assert a[k] == b[k], k


def test_bootstrap_and_slopes(nb_stats):
    rng = np.random.default_rng(3)
    x, y, cl = _data(rng)
    a, b = nb_stats["ClusterBootstrap"](cl, seed=9), S.ClusterBootstrap(cl, n_boot=B, seed=9)
    eq(a.draws, b.draws)
    eq(a.counts, b.counts)
    assert nb_stats["ols_fit"](x, y) == S.ols_fit(x, y)
    _same(nb_stats["slope_with_ci"](x, y, cl, 11, h0=0.5), S.slope_with_ci(x, y, cl, 11, h0=0.5, n_boot=B))
    _same(nb_stats["slope_with_ci"](x[:6], y[:6], cl[:6], 11), S.slope_with_ci(x[:6], y[:6], cl[:6], 11, n_boot=B))
    ba, da = nb_stats["slope_boot"](x, y, cl, 12)
    bb, db = S.slope_boot(x, y, cl, 12, n_boot=B)
    assert ba == bb
    eq(da, db)
    y2 = y + 0.3 * x + rng.normal(0, 0.1, len(x))
    ra, rb = nb_stats["multi_slopes"](x, [y, y2], cl, 13), S.multi_slopes(x, [y, y2], cl, 13, n_boot=B)
    for u, v in zip(ra, rb):
        eq(u, v)
    assert nb_stats["contrast_ci"](ra[0], ra[1], [-1, 1]) == S.contrast_ci(rb[0], rb[1], [-1, 1])
    pa, pb = nb_stats["paired_slopes"](x, y2, y, cl, 14), S.paired_slopes(x, y2, y, cl, 14, n_boot=B)
    assert pa[:2] == pb[:2]
    eq(pa[2], pb[2])
    assert nb_stats["_ci"](pa[2]) == S.percentile_ci(pb[2])
    df = pd.DataFrame(dict(cl=cl, x=x, y=y))
    fn = lambda d: float(np.median(d["y"]))
    pa, pb = nb_stats["boot_stat"](df, "cl", fn, 15, n_boot=200), S.boot_stat(df, "cl", fn, 15, n_boot=200)
    assert pa[:3] == pb[:3]
    eq(pa[3], pb[3])


def test_rules_and_corrections(nb_stats):
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 0.1, 9)
    p[3] = np.nan
    eq(nb_stats["holm"](p), S.holm(p))
    for args in [(0.17, 0.09, 0.26, 0.1), (1.0, 0.95, 1.05, 0.9), (0.9, 0.4, 1.4, 0.8), (np.nan, np.nan, np.nan, np.nan),
                 (1.0, 0.9, 1.1, 0.3)]:
        assert nb_stats["verdict_p1"](*args) == S.verdict_p1(*args)
    for lo, hi in [(0.1, 0.3), (-0.1, 0.3), (-0.3, -0.1), (np.nan, 0.2)]:
        assert nb_stats["verdict_positive"](lo, hi) == S.verdict_positive(lo, hi)
    nb2 = cell_definitions("0840_e2.py", names={"verdict_p2"}, PREREG=PREREG)
    for r in [dict(diff=0.5, ci_lo=0.4, ci_hi=0.6), dict(diff=0.2, ci_lo=0.1, ci_hi=0.3), dict(diff=0.1, ci_lo=-0.1, ci_hi=0.3),
              dict(diff=np.nan, ci_lo=np.nan, ci_hi=np.nan)]:
        assert nb2["verdict_p2"](r) == S.verdict_p2(r["diff"], r["ci_lo"], r["ci_hi"])


def test_stratified_recall_difference(nb_stats):
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame(dict(regime=rng.choice(["parallel", "perpendicular", "intermediate"], n), s=rng.uniform(-0.6, 0.85, n),
                           gt_area_px=rng.integers(40, 5000, n), scene=rng.integers(0, 80, n)))
    df["recall"] = np.where(df["regime"] == "parallel", rng.uniform(0, 0.4, n), rng.uniform(0.5, 1, n))
    nb = cell_definitions("0840_e2.py", names={"stratified_recall_difference"}, ClusterBootstrap=nb_stats["ClusterBootstrap"])
    assert nb["stratified_recall_difference"](df, "recall", 21) == S.stratified_recall_difference(df, "recall", 21, n_boot=B)


def test_sample_masks_and_prior_weight(nb_stats):
    rng = np.random.default_rng(6)
    n = 300
    d = pd.DataFrame(dict(motion_type=rng.choice(["parallel", "perpendicular_component"], n), s=rng.uniform(-0.6, 1.4, n),
                          measurable=rng.random(n) < 0.9, identifiable=rng.random(n) < 0.8,
                          regime=rng.choice(["parallel", "intermediate", "perpendicular"], n),
                          gt_x_geo=rng.uniform(-1, 2.5, n), rho=rng.normal(0.1, 0.3, n), rho_mono=rng.normal(0, 0.3, n)))
    eq(nb_stats["e1_mask"](d).to_numpy(), P.e1_mask(d).to_numpy())
    eq(nb_stats["h9_mask"](d).to_numpy(), P.h9_mask(d).to_numpy())
    eq(nb_stats["prior_weight"](d), P.prior_weight(d["gt_x_geo"], d["rho"], d["rho_mono"]))


# ---- the static-twin test (cell 6.13) ------------------------------------------------------------------------------

def _vehicles(rng, n_tracks=140):
    rows, twin, flow = [], [], []
    for t in range(n_tracks):
        state = "parked" if t % 3 == 0 else "moving"
        seq, track = f"log{t % 17}", f"trk{t}"
        for k in range(int(rng.integers(1, 4))):
            scene, mover = int(rng.integers(0, 10_000)), k
            if state == "parked":
                s, eps = rng.normal(0, 0.02), abs(rng.normal(0, 0.3))
            else:
                s = rng.uniform(-1.0, 1.4)
                eps = abs(rng.normal(0, 0.3)) if rng.random() < 0.6 else rng.uniform(1, 10)
            regime = "parallel" if eps < 1 else ("perpendicular" if eps >= 3 else "intermediate")
            x_geo = -np.log(1 - s) if s < 1 else np.nan
            rows.append(dict(scene=scene, mover=mover, seq=seq, track=track, state=state, regime=regime, s=s,
                             measurable=bool(rng.random() < 0.95), gt_eps_med=eps, gt_x_geo=x_geo))
            H = rng.uniform(1.5, 2.0) if rng.random() < 0.97 else np.nan
            twin.append(dict(scene=scene, mover=mover, track=track, H_cam=H))
            if rng.random() < 0.9:
                h_twin = (H if np.isfinite(H) else 1.7) * (1 - np.exp(x_geo if np.isfinite(x_geo) else 0.0))
                flow.append(dict(scene=scene, mover=mover, n_flow_px=int(rng.integers(0, 200)),
                                 eps_flow=eps + abs(rng.normal(0, 0.5)), h_low_flow=0.3 * h_twin + rng.normal(0.4, 0.5),
                                 frac_behind=float(rng.uniform(0, 1) if s > 0.9 else rng.uniform(0, 0.3)),
                                 h_low_true=rng.normal(0.4, 0.1), x_flow=rng.normal(0, 0.05)))
    return pd.DataFrame(rows).drop_duplicates(["scene", "mover"]), pd.DataFrame(twin).drop_duplicates(["scene", "mover"]), \
        pd.DataFrame(flow).drop_duplicates(["scene", "mover"])


def test_twin_test_matches_notebook(nb_stats, tmp_path):
    movers, twin, flow = _vehicles(np.random.default_rng(7))
    nb = run_cell("0889h_twin.py", os=__import__("os"), json=__import__("json"), ClusterBootstrap=nb_stats["ClusterBootstrap"],
                  _ci=nb_stats["_ci"], verdict_positive=nb_stats["verdict_positive"], seed_for=seed_for, PREREG=PREREG,
                  TWIN_DESIGN=TWIN_DESIGN, FLOWTWIN_DESIGN=FLOWTWIN_DESIGN, AV2_TWIN=twin, MOVERS_AV2_ALL=movers,
                  AV2_FLOWTWIN=flow, AV2T_STATUS=dict(note=""), AV2F_STATUS=dict(note="", flow="test flow"),
                  RESULTS_DIR=str(tmp_path))
    lab = T.run_label_test(movers.merge(twin.drop(columns=["track"]), on=["scene", "mover"], how="left"), n_boot=B)
    pd.testing.assert_frame_equal(nb["TWIN_RECALL"], lab["by_regime"])
    pd.testing.assert_frame_equal(nb["TWIN_S"], lab["by_s"])
    pd.testing.assert_frame_equal(nb["TWIN_THEORY"], lab["curve"])
    assert nb["TWIN_THRESH"] == lab["thresholds"]
    fl = T.run_flow_test(movers.merge(flow, on=["scene", "mover"], how="inner"), n_boot=B)
    pd.testing.assert_frame_equal(nb["FTWIN_RECALL"], fl["by_regime"])
    assert {k: v for k, v in nb["FTWIN_THRESH"].items() if k != "flow"} == fl["thresholds"]
    _same(nb["H14_RESULT"], fl["H14"])
    assert fl["H14"]["verdict"] in ("SUPPORTED", "NOT SUPPORTED", "CONTRADICTED")

"""Propositions 1-3 and the epipolar-plane statement of the paper (Sec. 3), checked numerically."""
import numpy as np
import pytest

import convoy.geometry as G

CAM = G.PinholeCamera(700.0, 690.0, 319.5, 239.5, 640, 480)


def _random_pair(rng):
    R1, R2 = G.random_rotation(rng), G.random_rotation(rng)
    C1 = rng.uniform(-5, 5, 3)
    b = rng.standard_normal(3) * rng.uniform(0.1, 3.0)
    return R1, C1, R2, C1 + b, b


def test_proposition_1_parallel_motion_equals_static_twin():
    """A point moved by s*b and the static point C1 + (X - C1)/(1 - s) give identical images; triangulation returns Z/(1 - s)."""
    rng = np.random.default_rng(0)
    n = 0
    for _ in range(3000):
        R1, C1, R2, C2, b = _random_pair(rng)
        Xc1 = np.array([rng.uniform(-8, 8), rng.uniform(-6, 6), rng.uniform(2, 80)])
        X = G.cam_to_world(Xc1, R1, C1)
        s = rng.uniform(-2.0, 0.95)
        Xs = G.static_twin(X, s, C1)
        q_moved, q_static = G.world_to_cam(X + s * b, R2, C2), G.world_to_cam(Xs, R2, C2)
        if q_moved[2] <= 0.1 or q_static[2] <= 0.1:
            continue
        n += 1
        np.testing.assert_allclose(CAM.project(G.world_to_cam(Xs, R1, C1)), CAM.project(Xc1), atol=1e-7)
        np.testing.assert_allclose(CAM.project(q_static), CAM.project(q_moved), atol=1e-7)
        R21, t21 = G.relative_pose(R1, C1, R2, C2)
        Z, gap = G.triangulate_depth(CAM.project(Xc1), CAM.project(q_moved), CAM.K, CAM.K, R21, t21)
        assert Z == pytest.approx(Xc1[2] / (1 - s), rel=1e-6)
        assert np.log(Z / Xc1[2]) == pytest.approx(float(G.geometric_answer(s)), abs=1e-6)
    assert n > 1000


def test_proposition_1_no_static_explanation_beyond_one():
    """For s > 1 the two rays meet behind the first camera."""
    rng = np.random.default_rng(1)
    behind = []
    for _ in range(500):
        R1, C1, R2, C2, b = _random_pair(rng)
        Xc1 = np.array([rng.uniform(-4, 4), rng.uniform(-3, 3), rng.uniform(4, 40)])
        X = G.cam_to_world(Xc1, R1, C1)
        q = G.world_to_cam(X + rng.uniform(1.05, 2.0) * b, R2, C2)
        if q[2] <= 0.1:
            continue
        R21, t21 = G.relative_pose(R1, C1, R2, C2)
        Z, _ = G.triangulate_depth(CAM.project(Xc1), CAM.project(q), CAM.K, CAM.K, R21, t21)
        behind.append(Z < 0)
    assert len(behind) > 100 and all(behind)


def _multiview_rms(rng, K, path):
    s = rng.uniform(-0.5, 0.8)
    b0 = rng.standard_normal(3)
    b0 *= rng.uniform(0.5, 2.0) / np.linalg.norm(b0)
    X = np.array([rng.uniform(-6, 6), rng.uniform(-3, 3), rng.uniform(8, 40)])
    C, Cs, Rs, pix = np.zeros(3), [], [], []
    for k in range(K):
        if path == "straight":
            Ck, Xk = k * b0, X + k * s * b0
        else:                                   # the camera turns by 6 degrees per frame
            Ck = C.copy()
            C = C + G.rot_y(np.deg2rad(6.0) * k) @ b0
            Xk = X + k * s * b0 if path == "curved" else X + s * Ck
        Rk = G.rot_ypr(rng.uniform(-0.3, 0.3), rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)) if k else np.eye(3)
        q = G.world_to_cam(Xk, Rk, Ck)
        if q[2] <= 0.5:
            return np.nan
        Cs.append(Ck)
        Rs.append(Rk)
        pix.append(CAM.project(q))
    return G.triangulate_multiview(pix, [CAM] * K, Rs, Cs)[1]


@pytest.mark.parametrize("K", [3, 4, 8])
def test_proposition_2_several_views(K):
    """A static point fits a convoy on a straight path, and a mover copying a curved path, but not a curved path."""
    rng = np.random.default_rng(K)
    med = {p: np.nanmedian([_multiview_rms(rng, K, p) for _ in range(150)]) for p in ("straight", "curved", "curved_convoy")}
    assert med["straight"] < 1e-6
    assert med["curved_convoy"] < 1e-6
    assert med["curved"] > 0.1


def test_motion_inside_the_epipolar_plane_is_invisible():
    """Only the component normal to the epipolar plane moves the observation off its epipolar line (Sec. 3)."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        b = rng.standard_normal(3)
        b *= rng.uniform(0.5, 3.0) / np.linalg.norm(b)
        R1, C1, R2, C2 = np.eye(3), np.zeros(3), G.rot_ypr(rng.uniform(-0.2, 0.2), rng.uniform(-0.05, 0.05)), b
        X = np.array([rng.uniform(-6, 6), rng.uniform(-2, 2), rng.uniform(8, 40)])
        n = np.cross(X - C1, b)
        n /= np.linalg.norm(n)
        u = np.cross(b, n)
        u /= np.linalg.norm(u)
        R21, t21 = G.relative_pose(R1, C1, R2, C2)
        F = G.fundamental_matrix(CAM.K, CAM.K, R21, t21)
        p1 = CAM.project(G.world_to_cam(X, R1, C1))
        s, m = rng.uniform(-0.5, 0.8), rng.uniform(0.05, 1.0)
        q_in, q_out = G.world_to_cam(X + s * b + m * u, R2, C2), G.world_to_cam(X + s * b + m * n, R2, C2)
        if min(q_in[2], q_out[2]) <= 0.5:
            continue
        assert abs(G.epipolar_offset(p1, CAM.project(q_in), F)) < 1e-6
        assert abs(G.epipolar_offset(p1, CAM.project(q_out), F)) > 0.0


def test_proposition_3_static_twin_height():
    """The twin's contact lies H (1 - e^x) above the contact plane (below it for 0 < s < 1) and is e^x times larger."""
    rng = np.random.default_rng(4)
    for _ in range(3000):
        n = rng.standard_normal(3)
        n /= np.linalg.norm(n)
        H = rng.uniform(0.8, 3.0)
        C1 = rng.uniform(-5, 5, 3)
        t1 = np.cross(n, rng.standard_normal(3))
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        Xc = C1 - H * n + rng.uniform(5, 40) * t1 + rng.uniform(-6, 6) * t2
        X2 = Xc + rng.uniform(0.2, 2.0) * n
        s = rng.uniform(-2.0, 0.95)
        tw_c, tw_2 = G.static_twin(Xc, s, C1), G.static_twin(X2, s, C1)
        h = (tw_c - Xc) @ n
        assert h == pytest.approx(float(G.twin_contact_height(H, s=s)), abs=1e-9)
        assert h == pytest.approx(float(G.twin_contact_height(H, x=G.geometric_answer(s))), abs=1e-9)
        assert np.linalg.norm(tw_2 - tw_c) / np.linalg.norm(X2 - Xc) == pytest.approx(1 / (1 - s), rel=1e-9)
        if 0 < s < 1:
            assert h < 0
        elif s < 0:
            assert h > 0


def test_counterfactual_helpers():
    s = np.array([-0.6, 0.0, 0.5, 0.85])
    np.testing.assert_allclose(G.geometric_answer(s), -np.log(1 - s))
    assert float(G.geometric_answer(0.5)) == pytest.approx(np.log(2))
    X = np.array([[1.0, 2.0, 10.0], [0.0, 0.0, 5.0]])
    np.testing.assert_allclose(G.static_twin(X, np.array([0.5, 0.75])), X * np.array([[2.0], [4.0]]))
    np.testing.assert_allclose(G.static_twin(X, 0.5, C1=np.ones(3)), 1 + (X - 1) * 2)
    assert float(G.twin_contact_height(1.6, s=0.5)) == pytest.approx(-1.6)
    with pytest.raises(ValueError):
        G.twin_contact_height(1.6)
    with pytest.raises(ValueError):
        G.twin_contact_height(1.6, s=0.5, x=0.7)

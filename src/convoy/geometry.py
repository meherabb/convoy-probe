"""Two-view and multi-view geometry used by the probe (float64, vectorised).

The functions mirror cell 2.1 of the notebook (``notebook/src/cells/0310_geometry_core.py``);
``tests/test_notebook_equivalence.py`` checks that both give identical results.

Conventions: OpenCV camera axes (x right, y down, z forward); a camera has a camera-to-world
rotation ``R`` and a centre ``C``; points are row vectors, so ``world_to_cam(X, R, C) = R^T (X - C)``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "PinholeCamera", "rot_x", "rot_y", "rot_z", "rot_ypr", "random_rotation", "world_to_cam", "cam_to_world",
    "relative_pose", "skew", "fundamental_matrix", "epipolar_offset", "triangulate_depth", "infinite_homography_map",
    "translational_parallax", "triangulate_multiview", "geometric_answer", "static_twin", "twin_contact_height",
]


class PinholeCamera:
    """Pinhole intrinsics with pixel centres at integer coordinates."""

    def __init__(self, fx, fy, cx, cy, W, H):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.W, self.H = int(W), int(H)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def rays(self, u, v) -> np.ndarray:
        """Camera-frame ray directions scaled so that their z-component is 1."""
        u = np.asarray(u, np.float64)
        v = np.asarray(v, np.float64)
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], -1)

    def project(self, Xc) -> np.ndarray:
        Xc = np.asarray(Xc, np.float64)
        z = Xc[..., 2]
        return np.stack([self.cx + self.fx * Xc[..., 0] / z, self.cy + self.fy * Xc[..., 1] / z], -1)

    def scaled(self, sx, sy) -> PinholeCamera:
        """Intrinsics of the same camera after resizing the image by ``(sx, sy)``."""
        return PinholeCamera(self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5,
                             (self.cy + 0.5) * sy - 0.5, round(self.W * sx), round(self.H * sy))

    def to_dict(self) -> dict:
        return dict(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, W=self.W, H=self.H)

    @staticmethod
    def from_dict(d) -> PinholeCamera:
        return PinholeCamera(d["fx"], d["fy"], d["cx"], d["cy"], d["W"], d["H"])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def rot_ypr(yaw, pitch=0.0, roll=0.0):
    """Camera-to-world rotation: yaw about the world y (down) axis, then pitch, then roll."""
    return rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation from a random unit quaternion."""
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world_to_cam(Xw, R, C):
    return (np.asarray(Xw, np.float64) - C) @ R          # = R^T (Xw - C), row-vector form


def cam_to_world(Xc, R, C):
    return np.asarray(Xc, np.float64) @ R.T + C


def relative_pose(R1, C1, R2, C2):
    """Relative pose with ``X2 = R21 X1 + t21`` (camera-1 to camera-2 coordinates)."""
    R21 = R2.T @ R1
    t21 = R2.T @ (np.asarray(C1, np.float64) - np.asarray(C2, np.float64))
    return R21, t21


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], np.float64)


def fundamental_matrix(K1, K2, R21, t21):
    return np.linalg.inv(K2).T @ skew(t21) @ R21 @ np.linalg.inv(K1)


def _hom(p):
    p = np.asarray(p, np.float64)
    return np.concatenate([p, np.ones(p.shape[:-1] + (1,))], -1)


def epipolar_offset(p1, p2, F):
    """Signed distance (pixels) of ``p2`` from the epipolar line ``F p1``; vectorised over leading dims."""
    line = _hom(p1) @ F.T
    den = np.sqrt(line[..., 0] ** 2 + line[..., 1] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sum(line * _hom(p2), -1) / den


def triangulate_depth(p1, p2, K1, K2, R21, t21):
    """Closest-point triangulation of a correspondence.

    Returns ``(Z, gap)``: the depth in camera 1 and the gap between the two rays in metres.
    ``Z < 0`` means that the rays meet behind camera 1 (no valid static explanation).
    """
    d1 = _hom(p1) @ np.linalg.inv(K1).T                      # z = 1 in camera 1
    d2 = (_hom(p2) @ np.linalg.inv(K2).T) @ R21              # camera-2 rays in the camera-1 frame
    o2 = -R21.T @ t21                                        # camera-2 centre in the camera-1 frame
    w0 = -o2
    a = np.sum(d1 * d1, -1)
    b = np.sum(d1 * d2, -1)
    c = np.sum(d2 * d2, -1)
    d = np.sum(d1 * w0, -1)
    e = np.sum(d2 * w0, -1)
    den = a * c - b * b
    with np.errstate(invalid="ignore", divide="ignore"):
        t1 = (b * e - c * d) / den
        t2 = (a * e - b * d) / den
    gap = np.linalg.norm(t1[..., None] * d1 - (o2 + t2[..., None] * d2), axis=-1)
    return t1, gap


def infinite_homography_map(p1, K1, K2, R21):
    q = _hom(p1) @ (K2 @ R21 @ np.linalg.inv(K1)).T
    return q[..., :2] / q[..., 2:3]


def translational_parallax(p1, p2, K1, K2, R21):
    """Image displacement after removing the rotation-induced part (pixels)."""
    return np.linalg.norm(np.asarray(p2, np.float64) - infinite_homography_map(p1, K1, K2, R21), axis=-1)


def triangulate_multiview(pixels, cams, Rs, Cs):
    """Least-squares intersection of K rays (world frame); returns ``(point, rms reprojection error px)``."""
    A = np.zeros((3, 3))
    bb = np.zeros(3)
    for p, cam, R, C in zip(pixels, cams, Rs, Cs):
        d = R @ cam.rays(p[0], p[1])
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        bb += P @ C
    X = np.linalg.solve(A, bb)
    errs = [np.linalg.norm(cam.project(world_to_cam(X, R, C)) - np.asarray(p))
            for p, cam, R, C in zip(pixels, cams, Rs, Cs)]
    return X, float(np.sqrt(np.mean(np.square(errs))))


# ---- the counterfactual (paper, Sec. 3) ------------------------------------------------------------------------

def geometric_answer(s):
    """``x = -log(1 - s)``: log ratio of the static twin's depth to the true depth (Proposition 1)."""
    s = np.asarray(s, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -np.log(1.0 - s)


def static_twin(X, s, C1=None):
    """Static point that explains both views of a point moving by ``s * b`` (Proposition 1).

    ``X`` is the point at the first exposure (world frame) and ``C1`` the first camera centre
    (origin if omitted). The twin is ``C1 + (X - C1) / (1 - s)``; it is valid for ``s < 1``.
    """
    X = np.asarray(X, np.float64)
    C1 = np.zeros(3) if C1 is None else np.asarray(C1, np.float64)
    factor = 1.0 / (1.0 - np.asarray(s, np.float64))
    return C1 + (X - C1) * (factor[..., None] if factor.ndim else factor)


def twin_contact_height(H, s=None, x=None):
    """Height of the static twin's contact point above the object's contact plane (Proposition 3).

    ``h = -H s / (1 - s) = H (1 - e^x)``, with ``H`` the camera height above the contact plane.
    Give either the parallel fraction ``s`` or the geometric answer ``x``.
    """
    if (s is None) == (x is None):
        raise ValueError("give exactly one of s or x")
    H = np.asarray(H, np.float64)
    if x is None:
        x = geometric_answer(s)
    return H * (1.0 - np.exp(np.asarray(x, np.float64)))

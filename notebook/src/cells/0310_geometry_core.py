# PLACEHOLDER
# 2.1 Geometry core (vectorized, float64).
from scipy.optimize import brentq


class PinholeCamera:
    def __init__(self, fx, fy, cx, cy, W, H):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.W, self.H = int(W), int(H)

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def rays(self, u, v):
        # camera-frame ray directions scaled so that their z-component is 1
        u = np.asarray(u, np.float64); v = np.asarray(v, np.float64)
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], -1)

    def project(self, Xc):
        Xc = np.asarray(Xc, np.float64)
        z = Xc[..., 2]
        return np.stack([self.cx + self.fx * Xc[..., 0] / z, self.cy + self.fy * Xc[..., 1] / z], -1)

    def scaled(self, sx, sy):
        # intrinsics of the same camera after resizing the image by (sx, sy)
        return PinholeCamera(self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5,
                             (self.cy + 0.5) * sy - 0.5, round(self.W * sx), round(self.H * sy))

    def to_dict(self):
        return dict(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, W=self.W, H=self.H)

    @staticmethod
    def from_dict(d):
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
    # camera-to-world rotation: yaw about world y (down axis), then pitch about the camera
    # x axis, then roll about the optical axis
    return rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)

def random_rotation(rng):
    q = rng.standard_normal(4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def world_to_cam(Xw, R, C):
    return (np.asarray(Xw, np.float64) - C) @ R          # = R^T (Xw - C), row-vector form

def cam_to_world(Xc, R, C):
    return np.asarray(Xc, np.float64) @ R.T + C

def relative_pose(R1, C1, R2, C2):
    # X2 = R21 X1 + t21  (camera-1 coordinates -> camera-2 coordinates)
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
    # signed distance (px) of p2 from the epipolar line F p1; vectorized over leading dims
    l = _hom(p1) @ F.T
    den = np.sqrt(l[..., 0] ** 2 + l[..., 1] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sum(l * _hom(p2), -1) / den

def triangulate_depth(p1, p2, K1, K2, R21, t21):
    # Closest-point triangulation; returns (Z in camera 1, ray gap in metres).
    # Z < 0 means the rays meet behind camera 1 (no valid static explanation).
    d1 = _hom(p1) @ np.linalg.inv(K1).T                      # z = 1 in camera 1
    d2 = (_hom(p2) @ np.linalg.inv(K2).T) @ R21              # camera-2 rays in camera-1 frame
    o2 = -R21.T @ t21                                        # camera-2 centre in camera-1 frame
    w0 = -o2
    a = np.sum(d1 * d1, -1); b = np.sum(d1 * d2, -1); c = np.sum(d2 * d2, -1)
    d = np.sum(d1 * w0, -1); e = np.sum(d2 * w0, -1)
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
    # image displacement after removing the rotation-induced part (pixels)
    return np.linalg.norm(np.asarray(p2, np.float64) - infinite_homography_map(p1, K1, K2, R21), axis=-1)

def triangulate_multiview(pixels, cams, Rs, Cs):
    # least-squares intersection of K rays (world frame); returns (point, rms reprojection px)
    A = np.zeros((3, 3)); bb = np.zeros(3)
    for p, cam, R, C in zip(pixels, cams, Rs, Cs):
        d = R @ cam.rays(p[0], p[1]); d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P; bb += P @ C
    X = np.linalg.solve(A, bb)
    errs = [np.linalg.norm(cam.project(world_to_cam(X, R, C)) - np.asarray(p))
            for p, cam, R, C in zip(pixels, cams, Rs, Cs)]
    return X, float(np.sqrt(np.mean(np.square(errs))))

def perp_unit(b, prefer=None, rng=None):
    # unit vector perpendicular to b: 'horizontal' (in the ground plane) or random
    b = np.asarray(b, np.float64); bh = b / np.linalg.norm(b)
    if prefer == "horizontal":
        n = np.cross(bh, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(n) < 1e-8:
            n = np.cross(bh, np.array([1.0, 0.0, 0.0]))
    else:
        r = (rng or np.random.default_rng(0)).standard_normal(3)
        n = r - (r @ bh) * bh
    return n / np.linalg.norm(n)

def solve_perp_for_eps(X, delta_par, n_hat, target_eps, cam0, R0, C0, cam1, R1, C1, max_m=50.0):
    # Find the magnitude m of the perpendicular motion m*n_hat such that the moved point's
    # camera-1 image lies EXACTLY |target_eps| px from the epipolar line of its camera-0 image.
    # Both signs are tried; the smaller |m| is returned. Returns (None, nan) if impossible.
    R21, t21 = relative_pose(R0, C0, R1, C1)
    F = fundamental_matrix(cam0.K, cam1.K, R21, t21)
    p0 = cam0.project(world_to_cam(X, R0, C0))
    def eps_of(m):
        Xm = X + delta_par + m * n_hat
        Xc = world_to_cam(Xm, R1, C1)
        if Xc[2] <= 0.05:
            return np.nan
        return float(epipolar_offset(p0, cam1.project(Xc), F))
    if target_eps == 0.0:
        return 0.0, abs(eps_of(0.0))
    best = None
    for sgn in (+1.0, -1.0):
        g = lambda m: eps_of(m) - sgn * target_eps
        lo, hi = 0.0, 0.05
        g_lo = g(lo)
        if not np.isfinite(g_lo):
            continue
        found = False
        while abs(hi) <= max_m:
            for h in (hi, -hi):
                gh = g(h)
                if np.isfinite(gh) and g_lo * gh <= 0:
                    try:
                        m = brentq(g, min(0.0, h), max(0.0, h), xtol=1e-12, rtol=1e-12, maxiter=200)
                    except ValueError:
                        continue
                    if best is None or abs(m) < abs(best):
                        best = m
                    found = True
                    break
            if found:
                break
            hi *= 1.6
    if best is None:
        return None, np.nan
    return float(best), abs(eps_of(best))

print("Geometry core ready: PinholeCamera, rotations, relative_pose, fundamental_matrix, "
      "epipolar_offset, triangulate_depth, translational_parallax, triangulate_multiview, "
      "perp_unit, solve_perp_for_eps.")
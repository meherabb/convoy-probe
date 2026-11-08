# PLACEHOLDER
# 2.2 Exact verification of Propositions 1-4 with arbitrary rotations and baselines.
_rng = np.random.default_rng(seed_for("prop-checks"))
_cam = PinholeCamera(700.0, 690.0, 319.5, 239.5, 640, 480)
PROP_CHECKS = {}

# ---- Proposition 1: moved point (delta = s b) == static point X/(1-s) in both views ----
_err_px, _err_rel_depth, _behind = [], [], []
for _ in range(20000):
    R1, R2 = random_rotation(_rng), random_rotation(_rng)
    C1 = _rng.uniform(-5, 5, 3)
    b = _rng.standard_normal(3) * _rng.uniform(0.05, 5.0)
    C2 = C1 + b
    Xc1 = np.array([_rng.uniform(-8, 8), _rng.uniform(-6, 6), _rng.uniform(2, 80)])
    X = cam_to_world(Xc1, R1, C1)
    s = _rng.uniform(-2.0, 0.95)
    Xs = C1 + (X - C1) / (1.0 - s)                         # static equivalent
    q_moved = world_to_cam(X + s * b, R2, C2)
    q_static = world_to_cam(Xs, R2, C2)
    if q_moved[2] <= 0.1 or q_static[2] <= 0.1:
        continue
    p1 = _cam.project(Xc1)
    p2 = _cam.project(q_moved)
    _err_px.append(max(np.abs(_cam.project(world_to_cam(Xs, R1, C1)) - p1).max(),
                       np.abs(_cam.project(q_static) - p2).max()))
    R21, t21 = relative_pose(R1, C1, R2, C2)
    Z_tri, _gap = triangulate_depth(p1, p2, _cam.K, _cam.K, R21, t21)
    _err_rel_depth.append(abs(Z_tri / (Xc1[2] / (1 - s)) - 1.0))
# s > 1: the rays meet behind camera 1
for _ in range(2000):
    R1, R2 = random_rotation(_rng), random_rotation(_rng)
    b = _rng.standard_normal(3); C1 = np.zeros(3); C2 = b
    Xc1 = np.array([_rng.uniform(-4, 4), _rng.uniform(-3, 3), _rng.uniform(4, 40)])
    X = cam_to_world(Xc1, R1, C1); s = _rng.uniform(1.05, 2.0)
    q = world_to_cam(X + s * b, R2, C2)
    if q[2] <= 0.1:
        continue
    R21, t21 = relative_pose(R1, C1, R2, C2)
    Z_tri, _ = triangulate_depth(_cam.project(Xc1), _cam.project(q), _cam.K, _cam.K, R21, t21)
    _behind.append(Z_tri < 0)
PROP_CHECKS["P1_max_reprojection_diff_px"] = float(np.max(_err_px))
PROP_CHECKS["P1_max_rel_error_triangulated_vs_Z_over_1_minus_s"] = float(np.max(_err_rel_depth))
PROP_CHECKS["P1_n_trials"] = len(_err_px)
PROP_CHECKS["P1_fraction_s_gt_1_triangulating_behind_camera"] = float(np.mean(_behind))

# ---- Proposition 2: K views, straight path (any rotations) vs curved path vs varying ratio ----
def _multi_view_residual(K, path, rng):
    s = rng.uniform(-0.5, 0.8)
    b0 = rng.standard_normal(3); b0 *= rng.uniform(0.5, 2.0) / np.linalg.norm(b0)
    X = np.array([rng.uniform(-6, 6), rng.uniform(-3, 3), rng.uniform(8, 40)])
    Cs, Rs, pix = [], [], []
    C = np.zeros(3)
    for k in range(K):
        if path == "straight":
            Ck = k * b0
            Xk = X + k * s * b0
        elif path in ("curved", "curved_convoy"):              # translation direction turns
            Ck = C.copy()
            ang = np.deg2rad(6.0)
            b_k = rot_y(ang * k) @ b0
            C = C + b_k
            # curved: the object keeps moving on a straight line; curved_convoy: the object
            # copies the camera's displacement, X_k = X_0 + s (C_k - C_0)
            Xk = X + k * s * b0 if path == "curved" else X + s * Ck
        else:                                                   # straight path, ratio changes
            Ck = k * b0
            Xk = X + k * (s + 0.25 * k) * b0
        # arbitrary per-frame heading/pitch/roll changes (up to ~17 deg yaw)
        Rk = rot_ypr(rng.uniform(-0.3, 0.3), rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)) if k > 0 else np.eye(3)
        q = world_to_cam(Xk, Rk, Ck)
        if q[2] <= 0.5:
            return np.nan
        Cs.append(Ck); Rs.append(Rk); pix.append(_cam.project(q))
    _, rms = triangulate_multiview(pix, [_cam] * K, Rs, Cs)
    return rms

for _path in ["straight", "curved", "varying_ratio", "curved_convoy"]:
    for _K in [2, 3, 4, 8]:
        _vals = [_multi_view_residual(_K, _path, _rng) for _ in range(400)]
        _vals = np.array([v for v in _vals if np.isfinite(v)])
        PROP_CHECKS[f"P2_{_path}_K{_K}_median_static_fit_rms_px"] = float(np.median(_vals))

# ---- Proposition 3: which perpendicular motions leave the epipolar line ----
# The epipolar plane of X contains C1, C2 and X. Motion with a component along the plane
# normal n = (X - C1) x b moves the observation off the epipolar line; perpendicular motion
# INSIDE the plane (u = in-plane direction orthogonal to b) does not. The second case is a
# correction to the locked wording of Proposition 3 (see Section 0.2, erratum E-1).
_eps_vals = []
for _ in range(3000):
    R1 = np.eye(3); C1 = np.zeros(3)
    b = _rng.standard_normal(3); b *= _rng.uniform(0.5, 3.0) / np.linalg.norm(b)
    R2 = rot_ypr(_rng.uniform(-0.2, 0.2), _rng.uniform(-0.05, 0.05)); C2 = C1 + b
    X = np.array([_rng.uniform(-6, 6), _rng.uniform(-2, 2), _rng.uniform(8, 40)])
    n_plane = np.cross(X - C1, b); n_plane /= np.linalg.norm(n_plane)
    u_plane = np.cross(b, n_plane); u_plane /= np.linalg.norm(u_plane)       # in-plane, orthogonal to b
    m = _rng.uniform(0.05, 1.0); s = _rng.uniform(-0.5, 0.8)
    R21, t21 = relative_pose(R1, C1, R2, C2)
    F = fundamental_matrix(_cam.K, _cam.K, R21, t21)
    p1 = _cam.project(world_to_cam(X, R1, C1))
    q_par = world_to_cam(X + s * b, R2, C2)
    q_in = world_to_cam(X + s * b + m * u_plane, R2, C2)
    q_out = world_to_cam(X + s * b + m * n_plane, R2, C2)
    if min(q_par[2], q_in[2], q_out[2]) <= 0.5:
        continue
    e_par = abs(float(epipolar_offset(p1, _cam.project(q_par), F)))
    e_in = abs(float(epipolar_offset(p1, _cam.project(q_in), F)))
    e_out = abs(float(epipolar_offset(p1, _cam.project(q_out), F)))
    # the in-plane mover is explained exactly by the static point at the ray intersection
    Z_tri, gap = triangulate_depth(p1, _cam.project(q_in), _cam.K, _cam.K, R21, t21)
    _eps_vals.append((e_par, e_in, e_out, float(gap)))
_eps_vals = np.array(_eps_vals)
PROP_CHECKS["P3_max_eps_parallel_px"] = float(_eps_vals[:, 0].max())
PROP_CHECKS["P3_max_eps_inplane_perpendicular_px"] = float(_eps_vals[:, 1].max())
PROP_CHECKS["P3_max_ray_gap_inplane_m"] = float(_eps_vals[:, 3].max())
PROP_CHECKS["P3_min_eps_out_of_plane_px"] = float(_eps_vals[:, 2].min())
PROP_CHECKS["P3_median_eps_out_of_plane_px"] = float(np.median(_eps_vals[:, 2]))

# ---- Proposition 4 (v6.5): the static twin of a grounded parallel mover ----
# A road plane with a random upward normal, camera 1 at height H above the mover's contact point, a baseline in the road
# plane, arbitrary rotations: the twin's contact (i) lies H (1 - e^x) above the road, (ii) scales every length by e^x and
# (iii) for 0 < s < 1 is reached from camera 1 only through the contact point itself, i.e. it lies below the road.
_rng4 = np.random.default_rng(seed_for("prop-checks-4"))
_h_err, _len_err, _behind_road, _same_px = [], [], [], []
for _ in range(20000):
    n = _rng4.standard_normal(3); n /= np.linalg.norm(n)
    Hc = _rng4.uniform(0.8, 3.0)
    C1 = _rng4.uniform(-5, 5, 3)
    t1 = np.cross(n, _rng4.standard_normal(3)); t1 /= np.linalg.norm(t1)       # direction inside the road plane
    t2 = np.cross(n, t1)
    Xc = C1 - Hc * n + _rng4.uniform(5, 40) * t1 + _rng4.uniform(-6, 6) * t2      # contact point on the road
    X2 = Xc + _rng4.uniform(0.2, 2.0) * n + _rng4.uniform(-1, 1) * t1             # a second point of the object
    a_ = _rng4.uniform(0, 2 * np.pi)
    b = _rng4.uniform(0.3, 3.0) * (np.cos(a_) * t1 + np.sin(a_) * t2)
    s = _rng4.uniform(-2.0, 0.95)
    xs = -np.log(1.0 - s)
    tw_c, tw_2 = C1 + (Xc - C1) / (1 - s), C1 + (X2 - C1) / (1 - s)
    _h_err.append(abs((tw_c - Xc) @ n - Hc * (1.0 - np.exp(xs))))
    _len_err.append(abs(np.linalg.norm(tw_2 - tw_c) / np.linalg.norm(X2 - Xc) - np.exp(xs)))
    R1, R2 = random_rotation(_rng4), random_rotation(_rng4)
    q1, q2 = world_to_cam(Xc + s * b, R2, C1 + b), world_to_cam(tw_c, R2, C1 + b)
    if q1[2] > 0.1 and q2[2] > 0.1 and world_to_cam(Xc, R1, C1)[2] > 0.1:
        _same_px.append(float(np.abs(_cam.project(q1) - _cam.project(q2)).max()))
    if 0 < s < 1:
        _behind_road.append(bool(((tw_c - C1) @ n) < ((Xc - C1) @ n) - 1e-12))     # below the road plane
PROP_CHECKS["P4_max_twin_height_error_m"] = float(np.max(_h_err))
PROP_CHECKS["P4_max_twin_scale_error"] = float(np.max(_len_err))
PROP_CHECKS["P4_max_reprojection_diff_px"] = float(np.max(_same_px)) if _same_px else float("nan")
PROP_CHECKS["P4_fraction_twin_below_road_for_0_lt_s_lt_1"] = float(np.mean(_behind_road)) if _behind_road else float("nan")
PROP_CHECKS["P4_n_trials"] = len(_h_err)

for _k, _v in PROP_CHECKS.items():
    print(f"{_k:<60}{_v:.3e}" if isinstance(_v, float) else f"{_k:<60}{_v}")

assert PROP_CHECKS["P1_max_reprojection_diff_px"] < 1e-6
assert PROP_CHECKS["P1_max_rel_error_triangulated_vs_Z_over_1_minus_s"] < 1e-6
assert PROP_CHECKS["P1_fraction_s_gt_1_triangulating_behind_camera"] == 1.0
for _K in [2, 3, 4, 8]:
    assert PROP_CHECKS[f"P2_straight_K{_K}_median_static_fit_rms_px"] < 1e-6
    assert PROP_CHECKS[f"P2_curved_convoy_K{_K}_median_static_fit_rms_px"] < 1e-6
for _K in [3, 4, 8]:
    assert PROP_CHECKS[f"P2_curved_K{_K}_median_static_fit_rms_px"] > 0.1
    assert PROP_CHECKS[f"P2_varying_ratio_K{_K}_median_static_fit_rms_px"] > 0.1
assert PROP_CHECKS["P3_max_eps_parallel_px"] < 1e-6
assert PROP_CHECKS["P3_max_eps_inplane_perpendicular_px"] < 1e-6
assert PROP_CHECKS["P3_max_ray_gap_inplane_m"] < 1e-5          # numerical zero at 8-40 m depth
assert PROP_CHECKS["P3_min_eps_out_of_plane_px"] > 0.0
assert PROP_CHECKS["P4_max_twin_height_error_m"] < 1e-9
assert PROP_CHECKS["P4_max_twin_scale_error"] < 1e-9
assert PROP_CHECKS["P4_max_reprojection_diff_px"] < 1e-6
assert PROP_CHECKS["P4_fraction_twin_below_road_for_0_lt_s_lt_1"] == 1.0
print("\nAll proposition checks passed: P1 exact for arbitrary rotations; P2 exact on straight "
      "paths for any K and any rotations, broken by curved paths and by a changing ratio "
      "(K>=3), and exact again on curved paths when the object copies the camera's "
      "displacement (convoy); P3 motion inside the epipolar plane (parallel OR perpendicular "
      "to b) has eps = 0 and is explained by a static point; only the out-of-plane component "
      "is visible; P4 the static twin of a grounded parallel mover lies H (1 - e^x) above the road (below it "
      "for 0 < s < 1) and is e^x times larger.")
with open(os.path.join(LOG_DIR, "proposition_checks.json"), "w") as _f:
    json.dump(PROP_CHECKS, _f, indent=2)
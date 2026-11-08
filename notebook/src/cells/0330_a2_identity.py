# PLACEHOLDER
# 2.3 Assumption A2: locked first-order formula eps ~= f||delta_perp||/Z versus the exact
#     epipolar offset, plus the corrected first-order formula that uses only the component
#     of the motion normal to the epipolar plane (Proposition 3).
# 2.4 Slope-weight identity: OLS slope beta(w) of rho = log(w + (1-w) e^x) over the design range.
_rng = np.random.default_rng(seed_for("a2"))
_rows = []
for _ in range(6000):
    _kind = _rng.choice(["forward", "oblique", "side"], p=[0.4, 0.3, 0.3])
    _cd = CAMERA_DESIGN[_kind]
    _yaw = np.deg2rad(_rng.uniform(*_cd["yaw_deg"])) * _rng.choice([-1, 1])
    _f = _rng.uniform(*CAMERA_DESIGN["focal_px"])
    _cam = PinholeCamera(_f, _f, 319.5, 239.5, 640, 480)
    R0 = rot_ypr(_yaw); C0 = np.zeros(3)
    b = np.array([0.0, 0.0, 1.0]) * _rng.uniform(*_cd["baseline_m"])
    R1 = R0 @ rot_ypr(np.deg2rad(_rng.uniform(-1, 1))); C1 = C0 + b
    # mover centre: in front of camera 0 at the design depth, below the camera
    Zc = _rng.uniform(*MOVER_DESIGN["depth_range_m"])
    Xc = np.array([_rng.uniform(-0.45, 0.45) * Zc * 640 / _f, _rng.uniform(0.0, 0.3) * Zc * 480 / _f, Zc])
    X = cam_to_world(Xc, R0, C0)
    s = _rng.uniform(*MOVER_DESIGN["s_range_main"])
    n_h = perp_unit(b, "horizontal")
    m = _rng.uniform(0.05, 1.0) * _rng.choice([-1, 1])
    q0 = world_to_cam(X, R0, C0); q1 = world_to_cam(X + s * b + m * n_h, R1, C1)
    if q1[2] <= 1.0:
        continue
    p0, p1 = _cam.project(q0), _cam.project(q1)
    if not (0 <= p0[0] < 640 and 0 <= p0[1] < 480 and 0 <= p1[0] < 640 and 0 <= p1[1] < 480):
        continue
    R21, t21 = relative_pose(R0, C0, R1, C1)
    F = fundamental_matrix(_cam.K, _cam.K, R21, t21)
    eps_exact = abs(float(epipolar_offset(p0, p1, F)))
    n_plane = np.cross(X - C0, b); n_plane /= np.linalg.norm(n_plane)
    eps_locked = _f * abs(m) / Zc
    eps_corrected = _f * abs(m * (n_h @ n_plane)) / q1[2]
    _rows.append((_kind, Zc, eps_exact, eps_locked, eps_corrected, abs(n_h @ n_plane)))
A2_DIAG = pd.DataFrame(_rows, columns=["camera", "Z", "eps_exact", "eps_locked_first_order",
                                       "eps_corrected_first_order", "normal_fraction"])
A2_DIAG = A2_DIAG[A2_DIAG["eps_exact"] > 0.05].reset_index(drop=True)

def _ratio_stats(num, den):
    r = num / den
    return np.percentile(r, [5, 50, 95])

_rl = _ratio_stats(A2_DIAG["eps_locked_first_order"].values, A2_DIAG["eps_exact"].values)
_rc = _ratio_stats(A2_DIAG["eps_corrected_first_order"].values, A2_DIAG["eps_exact"].values)
A2_SUMMARY = {
    "n_configurations": int(len(A2_DIAG)),
    "locked_over_exact_p05_p50_p95": [float(v) for v in _rl],
    "corrected_over_exact_p05_p50_p95": [float(v) for v in _rc],
    "fraction_regime_flip_locked": float(np.mean(
        (A2_DIAG["eps_locked_first_order"] >= PREREG["eps_perp_min"]) & (A2_DIAG["eps_exact"] < PREREG["eps_parallel_max"]))),
    "median_normal_fraction_horizontal_motion": float(A2_DIAG["normal_fraction"].median()),
}
print("2.3  A2 diagnostic over", A2_SUMMARY["n_configurations"], "street-scene configurations "
      "(horizontal perpendicular motion, forward/oblique/side cameras)")
print(f"     locked first-order / exact      : p5={_rl[0]:.2f}  median={_rl[1]:.2f}  p95={_rl[2]:.2f}")
print(f"     corrected first-order / exact   : p5={_rc[0]:.2f}  median={_rc[1]:.2f}  p95={_rc[2]:.2f}")
print(f"     movers the locked formula calls 'perpendicular' (>= 3 px) that are exactly "
      f"'parallel' (< 1 px): {100 * A2_SUMMARY['fraction_regime_flip_locked']:.1f}%")
print(f"     median share of horizontal perpendicular motion that leaves the epipolar plane: "
      f"{A2_SUMMARY['median_normal_fraction_horizontal_motion']:.2f}")
print("     -> A2 in its locked form does not hold; every regime in this notebook is assigned "
      "from the exact offset, as the lock prescribes for this case.")
np.savez_compressed(os.path.join(LOG_DIR, "a2_diagnostic.npz"), **{c: A2_DIAG[c].values for c in A2_DIAG.columns})

# ---- 2.4 slope-weight identity over the design range of x = -log(1-s) ----
_s_design = np.linspace(MOVER_DESIGN["s_range_main"][0], MOVER_DESIGN["s_range_main"][1], 2001)
_x_design = -np.log(1.0 - _s_design)
SLOPE_WEIGHT_TABLE = []
for _w in np.round(np.linspace(0, 1, 11), 2):
    _rho = np.log(_w + (1 - _w) * np.exp(_x_design))
    _beta = np.polyfit(_x_design, _rho, 1)[0]
    SLOPE_WEIGHT_TABLE.append({"w": float(_w), "beta_ols_over_design": float(_beta),
                               "local_slope_at_x0": float(1 - _w)})
SLOPE_WEIGHT_TABLE = pd.DataFrame(SLOPE_WEIGHT_TABLE)
print("\n2.4  Slope-weight identity: beta(w) = OLS slope of log(w + (1-w) e^x) on x over the "
      "design range s in", MOVER_DESIGN["s_range_main"])
print(SLOPE_WEIGHT_TABLE.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))

def beta_to_effective_w(beta):
    # invert beta(w) by interpolation (monotone decreasing in w)
    tab = SLOPE_WEIGHT_TABLE.sort_values("beta_ols_over_design")
    return float(np.interp(beta, tab["beta_ols_over_design"].values, tab["w"].values))

assert np.all(np.diff(SLOPE_WEIGHT_TABLE["beta_ols_over_design"].values) < 0), "beta(w) must be monotone"
with open(os.path.join(LOG_DIR, "a2_and_identity.json"), "w") as _f:
    json.dump({"A2": A2_SUMMARY, "slope_weight": SLOPE_WEIGHT_TABLE.to_dict("records")}, _f, indent=2)
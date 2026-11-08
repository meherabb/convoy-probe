# PLACEHOLDER
# 6.13 The static twin test (v6.5, exploratory, D58): Argoverse 2 vehicles of all logs; twin height H (1 - e^x_geo) and
#      epipolar offset, both thresholded at 5% false positives on parked vehicles; recall on moving vehicles by regime and
#      by s; the detection curve implied by the parked distribution for exactly parallel motion. Cluster = vehicle track.
#      v6.6 (D63): the same tests with optical flow instead of labelled correspondences and a LiDAR road plane instead of
#      the box bottom, and the registered H14 (twin minus epipolar in the parallel regime, paired over the same vehicles).
TWIN_ROWS, TWIN_BY_S, TWIN_CURVE = [], [], []
TWIN_VEH, TWIN_THRESH = pd.DataFrame(), {}


def _rec_ci(q, cols, seed):
    # share of vehicles flagged by each test, with cluster-bootstrap intervals over vehicle tracks
    if len(q) < 5 or q["cl"].nunique() < 5:
        return {c: (float(q[c].mean()) if len(q) else np.nan, np.nan, np.nan) for c in cols}
    T, P = ClusterBootstrap(q["cl"].to_numpy(), seed=seed).linear([np.ones(len(q))] + [q[c].to_numpy(float) for c in cols])
    out = {}
    for k, c in enumerate(cols):
        lo, hi = _ci(T[:, k + 1] / T[:, 0])
        out[c] = (float(P[k + 1] / P[0]), lo, hi)
    return out


if len(AV2_TWIN) and len(MOVERS_AV2_ALL):
    _t = MOVERS_AV2_ALL.merge(AV2_TWIN.drop(columns=["track"]), on=["scene", "mover"], how="left")
    _t = _t[_t["measurable"].astype(bool) & np.isfinite(_t["H_cam"]) & np.isfinite(_t["gt_eps_med"])].copy()
    _t["h_twin"] = _t["H_cam"] * (1.0 - np.exp(_t["gt_x_geo"]))
    _t["no_static_solution"] = ~np.isfinite(_t["gt_x_geo"])
    _t["cl"] = _t["seq"].astype(str) + "/" + _t["track"].astype(str)
    _pk = _t[(_t["state"] == "parked") & np.isfinite(_t["h_twin"])]
    q_ = 1.0 - PREREG["twin_fpr"]
    if len(_pk) >= 20:
        TWIN_THRESH = dict(n_parked=len(_pk), eps_px=float(np.quantile(_pk["gt_eps_med"], q_)),
                           height_m=float(np.quantile(np.abs(_pk["h_twin"]), q_)), H_median_m=float(_t["H_cam"].median()))
        _mv = _t[_t["state"] == "moving"].copy()
        _mv["det_epipolar"] = _mv["gt_eps_med"] > TWIN_THRESH["eps_px"]
        _mv["det_twin"] = np.where(_mv["no_static_solution"], False, np.abs(_mv["h_twin"]) > TWIN_THRESH["height_m"])
        _mv["det_either"] = _mv["det_epipolar"] | _mv["det_twin"]
        _mv["twin_below_road"] = _mv["det_twin"] & (_mv["h_twin"] < 0)
        TWIN_VEH = _mv


        _cols = ["det_epipolar", "det_twin", "det_either", "twin_below_road"]
        _valid = _mv[~_mv["no_static_solution"]]
        for reg in ("parallel", "intermediate", "perpendicular", "all"):
            q = _valid if reg == "all" else _valid[_valid["regime"] == reg]
            r = _rec_ci(q, _cols, seed_for("twin", reg))
            # what the twin test adds: its share among the vehicles the epipolar test misses (same exact correspondences)
            qm = q[~q["det_epipolar"]]
            rm = _rec_ci(qm, ["det_twin"], seed_for("twin-miss", reg))["det_twin"]
            TWIN_ROWS.append(dict(regime=reg, n=len(q), n_tracks=int(q["cl"].nunique()),
                                  **{f"{c}": r[c][0] for c in _cols}, **{f"{c}_lo": r[c][1] for c in _cols},
                                  **{f"{c}_hi": r[c][2] for c in _cols},
                                  n_epi_miss=len(qm), twin_given_epi_miss=rm[0], twin_given_epi_miss_lo=rm[1],
                                  twin_given_epi_miss_hi=rm[2]))
        _sb = TWIN_DESIGN["s_bins"]
        for k in range(len(_sb) - 1):
            q = _valid[(_valid["s"] >= _sb[k]) & (_valid["s"] < _sb[k + 1])]
            if not len(q):
                continue
            r = _rec_ci(q, _cols[:3], seed_for("twin-s", k))
            TWIN_BY_S.append(dict(s_lo=_sb[k], s_hi=_sb[k + 1], s_median=float(q["s"].median()), n=len(q),
                                  frac_parallel=float((q["regime"] == "parallel").mean()),
                                  **{c: r[c][0] for c in _cols[:3]}, **{f"{c}_lo": r[c][1] for c in _cols[:3]},
                                  **{f"{c}_hi": r[c][2] for c in _cols[:3]}))
        # detection curve for exactly parallel motion at the median camera height: the twin height -H s / (1 - s) plus
        # the noise of parked vehicles (their twin heights), against the same threshold
        _noise = _pk["h_twin"].to_numpy(float)
        for s_ in np.linspace(*TWIN_DESIGN["theory_s"], 75):
            h0 = -TWIN_THRESH["H_median_m"] * s_ / (1.0 - s_)
            TWIN_CURVE.append(dict(s=float(s_), twin_height_m=float(h0),
                                   p_detect_twin=float(np.mean(np.abs(h0 + _noise) > TWIN_THRESH["height_m"])),
                                   p_detect_epipolar=PREREG["twin_fpr"]))
        TWIN_THRESH.update(n_moving=len(_mv), n_no_static_solution=int(_mv["no_static_solution"].sum()),
                           parked_fpr_epipolar=float(np.mean(_pk["gt_eps_med"] > TWIN_THRESH["eps_px"])),
                           parked_fpr_twin=float(np.mean(np.abs(_pk["h_twin"]) > TWIN_THRESH["height_m"])))
TWIN_RECALL, TWIN_S, TWIN_THEORY = pd.DataFrame(TWIN_ROWS), pd.DataFrame(TWIN_BY_S), pd.DataFrame(TWIN_CURVE)
if len(TWIN_RECALL):
    print(f"Twin test on Argoverse 2: thresholds at {100 * PREREG['twin_fpr']:.0f}% false positives on {TWIN_THRESH['n_parked']} parked "
          f"vehicles: |eps| > {TWIN_THRESH['eps_px']:.2f} px (epipolar), |twin height| > {TWIN_THRESH['height_m']:.2f} m (twin); "
          f"median camera height above the contact plane {TWIN_THRESH['H_median_m']:.2f} m")
    print(f"moving vehicles: {TWIN_THRESH['n_moving']}, of which {TWIN_THRESH['n_no_static_solution']} have no static solution "
          f"(s >= 1: the twin lies behind the camera, which any cheirality check rejects)")
    with pd.option_context("display.width", 250):
        print("\nShare of moving vehicles flagged, by regime (95% intervals over tracks):")
        print(TWIN_RECALL[["regime", "n", "n_tracks", "det_epipolar", "det_twin", "det_either", "twin_below_road", "n_epi_miss",
                           "twin_given_epi_miss"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nBy parallel fraction s:")
        print(TWIN_S[["s_lo", "s_hi", "n", "frac_parallel", "det_epipolar", "det_twin", "det_either"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    _par = TWIN_RECALL[TWIN_RECALL["regime"] == "parallel"]
    if len(_par) and _par["n"].iloc[0]:
        _r = _par.iloc[0]
        if _r["n_epi_miss"]:
            _ci_t = (f" [{_r['twin_given_epi_miss_lo']:.2f}, {_r['twin_given_epi_miss_hi']:.2f}]"
                     if np.isfinite(_r["twin_given_epi_miss_lo"]) else " (too few tracks for an interval)")
            print(f"\nParallel regime: the epipolar test misses {int(_r['n_epi_miss'])} of {int(_r['n'])} moving vehicles even with "
                  f"exact correspondences; the twin test flags {_r['twin_given_epi_miss']:.2f}{_ci_t} of those.")
        else:
            print("\nParallel regime: with exact correspondences the epipolar test misses none of the moving vehicles in this run.")
else:
    print("The twin test is not part of this run:", AV2T_STATUS["note"] or "no Argoverse 2 camera heights")
for _n, _df in (("twin_recall_by_regime", TWIN_RECALL), ("twin_recall_by_s", TWIN_S), ("twin_detection_curve", TWIN_THEORY)):
    _df.to_csv(os.path.join(RESULTS_DIR, f"{_n}.csv"), index=False)
with open(os.path.join(RESULTS_DIR, "twin_thresholds.json"), "w") as f:
    json.dump(TWIN_THRESH, f, indent=1)

# ---- the twin test with optical flow (v6.6, D63; H14 registered in AMENDMENT_V6_6) -----------------------------------
# Same vehicles, but every quantity from the flow at their LiDAR pixels: the epipolar offset, and the 10th percentile of
# the heights of the static reconstruction above the road plane fitted to the LiDAR (no box). Thresholds at 5% false
# positives on parked vehicles: eps above its 95th percentile; the low height outside its 2.5-97.5% range.
FTWIN_ROWS, FTWIN_THRESH, H14_RESULT, FTWIN_VEH = [], {}, {}, pd.DataFrame()
if len(AV2_FLOWTWIN) and len(MOVERS_AV2_ALL):
    _f = MOVERS_AV2_ALL.merge(AV2_FLOWTWIN, on=["scene", "mover"], how="inner")
    _f = _f[_f["measurable"].astype(bool) & (_f["n_flow_px"] >= FLOWTWIN_DESIGN["min_px"]) & np.isfinite(_f["eps_flow"])].copy()
    _f["cl"] = _f["seq"].astype(str) + "/" + _f["track"].astype(str)
    _f["static_solution"] = (_f["frac_behind"] <= FLOWTWIN_DESIGN["behind_frac"]) & np.isfinite(_f["h_low_flow"])
    _fpk = _f[(_f["state"] == "parked") & _f["static_solution"]]
    if len(_fpk) >= 20:
        _a = PREREG["twin_fpr"]
        FTWIN_THRESH = dict(n_parked=len(_fpk), eps_px=float(np.quantile(_fpk["eps_flow"], 1 - _a)),
                            h_low_lo_m=float(np.quantile(_fpk["h_low_flow"], _a / 2)),
                            h_low_hi_m=float(np.quantile(_fpk["h_low_flow"], 1 - _a / 2)),
                            parked_h_low_true_median_m=float(_fpk["h_low_true"].median()),
                            parked_x_flow_abs_median=float(np.median(np.abs(_fpk["x_flow"]))), flow=AV2F_STATUS.get("flow", ""))
        _fm = _f[_f["state"] == "moving"].copy()
        _fm["det_epipolar"] = _fm["eps_flow"] > FTWIN_THRESH["eps_px"]
        _fm["det_twin"] = _fm["static_solution"] & ((_fm["h_low_flow"] < FTWIN_THRESH["h_low_lo_m"])
                                                    | (_fm["h_low_flow"] > FTWIN_THRESH["h_low_hi_m"]))
        _fm["det_either"] = _fm["det_epipolar"] | _fm["det_twin"]
        _fm["twin_below_road"] = _fm["det_twin"] & (_fm["h_low_flow"] < FTWIN_THRESH["h_low_lo_m"])
        FTWIN_VEH = _fm
        _fv = _fm[_fm["static_solution"]]
        _cols = ["det_epipolar", "det_twin", "det_either", "twin_below_road"]
        for reg in ("parallel", "intermediate", "perpendicular", "all"):
            q = _fv if reg == "all" else _fv[_fv["regime"] == reg]
            r = _rec_ci(q, _cols, seed_for("ftwin", reg))
            qm = q[~q["det_epipolar"]]
            rm = _rec_ci(qm, ["det_twin"], seed_for("ftwin-miss", reg))["det_twin"]
            FTWIN_ROWS.append(dict(regime=reg, n=len(q), n_tracks=int(q["cl"].nunique()), **{c: r[c][0] for c in _cols},
                                   **{f"{c}_lo": r[c][1] for c in _cols}, **{f"{c}_hi": r[c][2] for c in _cols},
                                   n_epi_miss=len(qm), twin_given_epi_miss=rm[0], twin_given_epi_miss_lo=rm[1],
                                   twin_given_epi_miss_hi=rm[2]))
        FTWIN_THRESH.update(n_moving=len(_fm), n_no_static_solution=int((~_fm["static_solution"]).sum()),
                            parked_fpr_epipolar=float(np.mean(_fpk["eps_flow"] > FTWIN_THRESH["eps_px"])),
                            parked_fpr_twin=float(np.mean((_fpk["h_low_flow"] < FTWIN_THRESH["h_low_lo_m"])
                                                          | (_fpk["h_low_flow"] > FTWIN_THRESH["h_low_hi_m"]))))
        # H14 (registered): parallel-regime moving vehicles with a static solution, det_twin - det_epipolar (paired)
        q = _fv[_fv["regime"] == "parallel"]
        H14_RESULT = dict(n=len(q), n_tracks=int(q["cl"].nunique()), det_twin=float(q["det_twin"].mean()) if len(q) else np.nan,
                          det_epipolar=float(q["det_epipolar"].mean()) if len(q) else np.nan, estimate=np.nan, ci_lo=np.nan,
                          ci_hi=np.nan, verdict="NOT TESTABLE")
        if len(q) >= PREREG["h14_min_vehicles"] and q["cl"].nunique() >= PREREG["h14_min_tracks"]:
            _T, _P = ClusterBootstrap(q["cl"].to_numpy(), seed=seed_for("h14")).linear(
                [np.ones(len(q)), q["det_twin"].to_numpy(float) - q["det_epipolar"].to_numpy(float)])
            H14_RESULT["estimate"] = float(_P[1] / _P[0])
            H14_RESULT["ci_lo"], H14_RESULT["ci_hi"] = _ci(_T[:, 1] / _T[:, 0])
            H14_RESULT["verdict"] = verdict_positive(H14_RESULT["ci_lo"], H14_RESULT["ci_hi"])
FTWIN_RECALL = pd.DataFrame(FTWIN_ROWS)
if len(FTWIN_RECALL):
    print(f"\nTwin test with optical flow ({FTWIN_THRESH['flow']}): thresholds at {100 * PREREG['twin_fpr']:.0f}% false positives on "
          f"{FTWIN_THRESH['n_parked']} parked vehicles: |eps| > {FTWIN_THRESH['eps_px']:.2f} px; lowest static-reconstruction height "
          f"outside [{FTWIN_THRESH['h_low_lo_m']:.2f}, {FTWIN_THRESH['h_low_hi_m']:.2f}] m (parked: lowest LiDAR returns "
          f"{FTWIN_THRESH['parked_h_low_true_median_m']:.2f} m above the road, median |x_flow| {FTWIN_THRESH['parked_x_flow_abs_median']:.3f})")
    print(f"moving vehicles with flow: {FTWIN_THRESH['n_moving']}, of which {FTWIN_THRESH['n_no_static_solution']} have no static "
          f"solution (most pixels triangulate behind the camera)")
    with pd.option_context("display.width", 250):
        print(FTWIN_RECALL[["regime", "n", "n_tracks", "det_epipolar", "det_twin", "det_either", "twin_below_road", "n_epi_miss",
                            "twin_given_epi_miss"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    _h = H14_RESULT
    print(f"H14 (registered): parallel regime, twin minus epipolar with the same flow = "
          + (f"{_h['estimate']:+.3f} [{_h['ci_lo']:+.3f}, {_h['ci_hi']:+.3f}]" if np.isfinite(_h["estimate"]) else "–")
          + f" over {_h['n']} vehicles / {_h['n_tracks']} tracks (twin {_h['det_twin']:.3f}, epipolar {_h['det_epipolar']:.3f}): "
          f"{_h['verdict']}")
else:
    print("\nThe twin test with optical flow is not part of this run:", AV2F_STATUS["note"] or "fewer than 20 parked vehicles with flow")
FTWIN_RECALL.to_csv(os.path.join(RESULTS_DIR, "twin_flow_recall_by_regime.csv"), index=False)
with open(os.path.join(RESULTS_DIR, "twin_flow_thresholds.json"), "w") as f:
    json.dump(dict(thresholds=FTWIN_THRESH, H14=H14_RESULT), f, indent=1, default=float)

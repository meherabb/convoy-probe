# PLACEHOLDER
# 3.3 Testbed self-tests on the rendered images (exactness and photometric consistency).
TESTBED_CHECKS = {}

# (a) Proposition 1 on rendered pixels: for parallel movers the triangulated depth of every
#     visible pixel equals Z/(1-s); x_geo is the median log(Z_geo/Z) over the mover's pixels.
_par = MOVERS_MAIN[(MOVERS_MAIN["motion_type"] == "parallel") & (MOVERS_MAIN["s"] < 1)
                   & (MOVERS_MAIN["gt_n_eroded"] > 0)]
_dev = np.abs(_par["gt_x_geo"] - _par["x_par"])
TESTBED_CHECKS["prop1_rendered_max_abs_dev_log_depth"] = float(_dev.max())
TESTBED_CHECKS["prop1_rendered_n_movers"] = int(len(_par))
# parallel motion leaves EVERY pixel of an extended mover on its epipolar line
TESTBED_CHECKS["parallel_movers_max_pixel_eps_px"] = float(_par["gt_eps_max"].max())
# (b) achieved epipolar offset vs design target (exact solver at the mover centre)
TESTBED_CHECKS["eps_center_max_abs_error_px"] = float(np.abs(MOVERS_MAIN["eps_center"] - MOVERS_MAIN["eps_target"]).max())

# (c) static pixels: eps ~ 0 and photometric consistency of the exact warp, versus a
#     control warp in which movers are (wrongly) treated as static
_static_eps, _photo_static, _photo_mover, _photo_mover_wrong = [], [], [], []
for meta in META_MAIN[: min(12, len(META_MAIN))]:
    sc = meta["scene"]; d = os.path.join(RENDER_DIR, "main", f"{meta['index']:04d}")
    r0, r1 = load_frame(d, 0), load_frame(d, 1)
    H, W = r0["depth"].shape
    vv, uu = np.mgrid[0:H, 0:W]
    pA = np.stack([uu, vv], -1).astype(np.float64)
    pB, zB, valid, vis = gt_correspondence(sc, r0, 0, 1, r1)
    eps, _, _ = pixel_epipolar_quantities(sc, 0, 1, pA, pB)
    stat = vis & (r0["mover_id"] < 0) & np.isfinite(r0["depth"]) & ~r0["mover_shadow"] & ~r1["mover_shadow"]
    stat &= erode_mask(stat, 1)
    _static_eps.append(eps[stat])
    mapx, mapy = pB[..., 0].astype(np.float32), pB[..., 1].astype(np.float32)
    warped = cv2.remap(r1["rgb"].astype(np.float32), np.nan_to_num(mapx), np.nan_to_num(mapy), cv2.INTER_LINEAR)
    err = np.abs(warped - r0["rgb"].astype(np.float32)).mean(-1)
    _photo_static.append(err[stat])
    mv = erode_mask(r0["mover_id"] >= 0, 2) & vis
    _photo_mover.append(err[mv])
    # wrong hypothesis: movers static -> use the static warp for mover pixels
    P = backproject(r0["depth"], *frame_camera(sc, 0))
    cam1, R1, C1 = frame_camera(sc, 1)
    pW = cam1.project((P - C1) @ R1)
    w2 = cv2.remap(r1["rgb"].astype(np.float32), np.nan_to_num(pW[..., 0]).astype(np.float32),
                   np.nan_to_num(pW[..., 1]).astype(np.float32), cv2.INTER_LINEAR)
    _photo_mover_wrong.append(np.abs(w2 - r0["rgb"].astype(np.float32)).mean(-1)[mv])
_static_eps = np.concatenate(_static_eps)
TESTBED_CHECKS["static_pixels_eps_median_px"] = float(np.median(_static_eps))
TESTBED_CHECKS["static_pixels_eps_p999_px"] = float(np.percentile(_static_eps, 99.9))
TESTBED_CHECKS["photometric_err_static_exact_warp"] = float(np.mean(np.concatenate(_photo_static)))
TESTBED_CHECKS["photometric_err_movers_exact_warp"] = float(np.mean(np.concatenate(_photo_mover)))
TESTBED_CHECKS["photometric_err_movers_if_assumed_static"] = float(np.mean(np.concatenate(_photo_mover_wrong)))

# (d) E5 ablation: identical static content across conditions, ground contact visible when grounded
#     and hidden behind the wall in CH (measured on the renders, not assumed from the design)
_ab = MOVERS_ABL.pivot_table(index="uid", columns="condition", values=["contact_px0", "contact_px1"], aggfunc="first")
_vis_g = (_ab[("contact_px0", "G")] >= 5) & (_ab[("contact_px1", "G")] >= 5)
_walled = MOVERS_ABL[(MOVERS_ABL["condition"] == "CH") & MOVERS_ABL["has_wall"]]["uid"]
_hid_ch = (_ab[("contact_px0", "CH")] == 0) & (_ab[("contact_px1", "CH")] == 0)
TESTBED_CHECKS["abl_contact_visible_when_grounded"] = float(_vis_g.mean())
TESTBED_CHECKS["abl_walls_designed_frac"] = float(len(_walled) / max(1, len(_ab)))
TESTBED_CHECKS["abl_contact_hidden_given_wall"] = float(_hid_ch.loc[_walled].mean()) if len(_walled) else float("nan")
TESTBED_CHECKS["abl_lifted_contact_px_max"] = int(MOVERS_ABL[MOVERS_ABL["condition"].isin(["F", "FH"])][["contact_px0", "contact_px1"]].max().max())
_diff_n, _diff_bad = 0, 0
for meta_ns in [m for m in META_ABL if m["condition"] == "NS"][:8]:
    meta_f = next(m for m in META_ABL if m["index"] == meta_ns["index"] and m["condition"] == "F")
    for k in (0, 1):
        a, b = load_frame(meta_ns["dir"], k), load_frame(meta_f["dir"], k)
        # pixels more than 3 px from either mover (anti-aliasing samples may touch a mover's edge)
        away = ~cv2.dilate(((a["mover_id"] >= 0) | (b["mover_id"] >= 0)).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        _diff_n += int(away.sum())
        _diff_bad += int((np.abs(a["rgb"].astype(int) - b["rgb"].astype(int)).max(-1)[away] > 1).sum())
TESTBED_CHECKS["abl_share_of_pixels_away_from_movers_differing_NS_vs_F"] = float(_diff_bad / max(_diff_n, 1))

for k, v in TESTBED_CHECKS.items():
    print(f"{k:<52}{v:.4g}" if isinstance(v, float) else f"{k:<52}{v}")
assert TESTBED_CHECKS["prop1_rendered_max_abs_dev_log_depth"] < 1e-3
assert TESTBED_CHECKS["eps_center_max_abs_error_px"] < 1e-2
assert TESTBED_CHECKS["parallel_movers_max_pixel_eps_px"] < 1e-3
assert TESTBED_CHECKS["static_pixels_eps_p999_px"] < 0.05
assert TESTBED_CHECKS["photometric_err_static_exact_warp"] < 6.0
assert TESTBED_CHECKS["photometric_err_movers_if_assumed_static"] > 2 * TESTBED_CHECKS["photometric_err_movers_exact_warp"]
assert TESTBED_CHECKS["abl_lifted_contact_px_max"] == 0
assert TESTBED_CHECKS["abl_share_of_pixels_away_from_movers_differing_NS_vs_F"] < 1e-4
print("\nTestbed self-tests passed: rendered parallel movers sit exactly at Z/(1-s) under "
      "static triangulation, static pixels have zero epipolar offset, the exact warp "
      "reproduces the second view up to sensor noise and resampling, and the ablation conditions "
      "differ only at the movers (lifted movers never touch the ground).")
with open(os.path.join(LOG_DIR, "testbed_checks.json"), "w") as f:
    json.dump(TESTBED_CHECKS, f, indent=2)

release_memory(); _ = mem_report("CONVOY-3D renders checked")

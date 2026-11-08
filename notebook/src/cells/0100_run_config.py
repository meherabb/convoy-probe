# PLACEHOLDER
# 0.4 Run configuration and pre-registered constants.
#     Every threshold used by a verdict is fixed here, before any model is run. The v5 entries
#     are unchanged (so the v5 testbeds are regenerated identically); the v6 entries below them
#     were added for the experiments registered in the v6 amendment (Section 0.1), and each later
#     version only adds entries (v6.5: the designed experiments, Bonn, the twin test and the cue
#     model, AMENDMENT_V6_5), so every stage of an earlier version keeps its fingerprint. v6.6 adds
#     H14 and FLOWTWIN_DESIGN and extends BONN_DESIGN (D62), so only the Bonn stage is recomputed.
RUN_CONFIG = {
    # image format shared by CONVOY-3D and TUM (640x480, 4:3)
    "W": 640, "H": 480,
    # CONVOY-3D main set (scene specifications have 3 frames as in v5; frames 0 and 1 are rendered)
    "n_scenes_main": 240, "movers_per_scene": 3,
    # A1 correspondence knockout: frame 1 re-rendered without movers, first n scenes
    "n_scenes_knockout": 96,
    # LEGACY-FLAT negative control for the validity gate
    "n_scenes_flat": 24,
    # E4 multi-frame clips per camera path type (straight / curved)
    "n_clips_e4_per_path": 48, "e4_n_frames": 8, "e4_window_sizes": (2, 4, 8),
    # CONVOY-REAL: TUM static sequences, frame pairs per sequence, inserted movers per pair
    "n_real_pairs_per_seq": 60, "real_movers_per_pair": 2,
    # TUM walking: candidate pairs scanned for the prevalence of motion regimes
    "n_dyn_candidate_pairs_per_seq": 150,
    # statistics
    "n_boot": 10000, "alpha": 0.05,
    # parallel use of CPU cores for rendering
    "render_workers": max(1, min(4, os.cpu_count() or 1)),
    # ---- v6 ----
    # E5 causal ablation: base scenes (3 parallel movers each), rendered once per condition
    "n_scenes_ablation": 72,
    # E6 Virtual KITTI 2: scenes and two-view pairs per scene (v6.3: at most 150 per scene, was 60; the
    # three short scenes are limited by their length, so only Scene18 and Scene20 gain pairs)
    "n_vk_scenes": 5, "vk_pairs_per_scene": 150,
    # ---- v6.1 ----
    # E5-REAL contact ablation on real TUM backgrounds: frame pairs per static sequence, base movers per pair
    "n_rabl_pairs_per_seq": 40, "rabl_movers_per_pair": 2,
    # KITTI tracking (only if attached): pairs per sequence and sequences used
    "n_kt_pairs_per_seq": 20, "n_kt_sequences": 12,
    # archives: copy them to a private Kaggle dataset when Kaggle API secrets are attached (Section 0.5)
    "dataset_backup": True, "dataset_backup_slug": "convoy-checkpoints",
    # ---- v6.3 ----
    # Argoverse 2 (real driving with LiDAR, downloaded automatically): validation logs and pairs per log
    "use_av2": True, "n_av2_logs": 16, "n_av2_pairs_per_log": 12,
    # ---- v6.4 ----
    # Argoverse 2 extension: further validation logs, taken in the same fixed order after the 16 logs of the v6.3
    # stage, which is kept unchanged (a session resumed from a v6.3 run restores it and adds only the new logs)
    "n_av2_logs_ext": 48,
    # ---- v6.5 (new stages only; every stage of v6.4 keeps its fingerprint) ----
    # designed experiments: base scenes of the baseline sweep and of the contact dose-response (3 parallel movers each)
    "n_scenes_bsw": 64, "n_scenes_wdr": 64,
    # Bonn RGB-D Dynamic (real balloons and boxes moved by people, measured depth, motion-capture poses)
    "use_bonn": True, "n_bonn_pairs_per_seq": 50,
}

PREREG = {
    # motion regimes (exact epipolar offset at the mover's centre, pixels)
    "eps_parallel_max": 1.0, "eps_perp_min": 3.0,
    # identifiability: translational parallax of the static-equivalent point (pixels)
    "parallax_min_px": 5.0,
    # static-scene validity gate (per model x testbed, medians over images)
    "gate_slope_range": (0.7, 1.3), "gate_absrel_max": 0.25,
    # Prediction 1
    "beta_support_range": (0.8, 1.2), "beta_ci_band": (0.5, 1.5), "r2_min": 0.5,
    # Prediction 2
    "recall_diff_min": 0.3, "mask_fpr": 0.05,
    # Prediction 3
    "spearman_min": 0.5, "w_separability_min": 0.10,
    # Prediction 4: straight-minus-curved beta_s difference at the longest window
    "e4_drop_min": 0.3,
    # object measurement
    "min_object_pixels": 40, "erosion_px": 2,
    # depth range of static pixels used for scale alignment / gate (metres)
    "static_depth_range_synth": (2.0, 80.0), "static_depth_range_real": (0.3, 4.5),
    # ---- v6 (registered in the amendment, before any v6 data was generated) ----
    "static_depth_range_vk": (2.0, 80.0),
    "h9_beta_max": 0.3, "h9_ci_hi_max": 0.5,           # H9: driving beta_geo below 0.3, CI below 0.5
    "h9_x_range": (-0.47, 1.90),                       # H9 sample: x_geo inside the CONVOY-3D design range of x
    "h9_min_movers": 20, "h9_min_pairs": 5,
    "min_movers_per_regime": 20,                       # a regime comparison is tested only with >= 20 movers each
    "contact_band_m": 0.12,                            # GT 'ground contact' pixels: <= 12 cm above the ground
    # ---- v6.1 (fixed after the first full v6 run, before any data of the analyses they govern) ----
    "vk_min_occupancy": 0.85,                          # Virtual KITTI 2 measurability: the CONVOY-3D pixel rule
                                                       # (>= 40 eroded visible px, >= 85% of the silhouette visible)
    "rabl_contact_band_m": 0.03,                       # E5-REAL: crate pixels within 3 cm of the support plane
    # ---- v6.2, exploratory analyses declared before their data existed (AMENDMENT_V6_2) ----
    "x6_min_eroded_px": 40,                            # driving sample: the pixel rule without the occupancy rule
    "x6_min_lever": 0.3,                               # per-object geometry share needs |x - rho_mono| >= 0.3
    "x6_pace_s": (0.85, 1.15),                         # "keeping pace with the camera": s in this range
    "x6_wheel_height_m": 0.10, "x6_wheel_depth_tol_m": 0.6,   # a wheel contact point is visible if the vehicle's own
                                                       # pixel is there at the point's depth (within 0.6 m)
    "kt_min_lidar_px": 40,                             # KITTI: LiDAR pixels on a vehicle (instead of eroded pixels)
    "kt_moving_disp_m": 0.5, "kt_parked_disp_m": 0.15, # world displacement of the labelled box between the two frames
    "kt_eps_noise_quantile": 0.9,                      # KITTI parallel regime: |eps| below this quantile of parked cars
    # ---- v6.3, exploratory analyses declared before their data existed (AMENDMENT_V6_3) ----
    # Argoverse 2 uses the KITTI rules above (kt_*) unchanged.
    "x6_match_caliper_sd": 0.5,                        # hidden/visible contact matched on log Z, log area and x_geo:
                                                       # nearest visible vehicle within 0.5 SD on every covariate
    # ---- v6.4, exploratory analyses declared before their data existed (AMENDMENT_V6_4) ----
    # Argoverse 2 ground contact uses the Virtual KITTI 2 wheel points and labels (x6_wheel_height_m, x6_match_caliper_sd)
    # with the visibility tests of AV2X_DESIGN; the synthesis (Section 6.9) adds no test and no threshold except this one:
    "implied_factor_s": 0.85,                          # depth overestimate implied by a slope for an object keeping pace
                                                       # with the camera at this s: exp(beta * x), x = -log(1 - s)
    # ---- v6.5, registered in AMENDMENT_V6_5 before any v6.5 data existed ----
    "h12_contrast": ("B3", "B05"),                     # H12: beta(baseline x3) - beta(baseline x0.5) > 0
    "h12_models": ("DUSt3R", "MASt3R"),                # ... predicted for these two models (Holm across them)
    "h13_contrast": ("W15", "G"),                      # H13: beta(15 cm wall) - beta(no wall) > 0, every model (Holm)
    "h13_contact_px_min": 5,                           # H13 sample: >= 5 contact pixels in both views without the wall,
                                                       # none with the 15 cm wall (the E5 rule for CH)
    "dose_min_movers": 8, "dose_min_scenes": 5,        # H12 / H13: NOT TESTABLE below 8 movers or 5 base scenes
    # exploratory (v6.5): the twin test, the cue model and Bonn
    "twin_fpr": 0.05,                                  # twin and epipolar tests: thresholds at 5% false positives on
                                                       # parked Argoverse 2 vehicles
    "bonn_min_object_px": 150,                         # Bonn: >= 150 eroded pixels with valid depth per object
    "bonn_floor_contact_m": 0.08, "bonn_floor_clear_m": 0.20,   # lowest point <= 8 cm above the floor: on the floor;
                                                       # >= 20 cm: off the floor (held or floating); otherwise undetermined
    # ---- v6.6, registered in AMENDMENT_V6_6 before any flow-based twin quantity existed ----
    "h14_min_vehicles": 20, "h14_min_tracks": 10,      # H14: on moving Argoverse 2 vehicles in the parallel regime, the twin
                                                       # test with optical flow flags more than the epipolar test with the
                                                       # same flow (paired difference, CI > 0); NOT TESTABLE below these
}

# Memory guard: a model is loaded only if the machine still has this much free RAM after the previous
# model was released (otherwise the run stops cleanly; every finished stage is already archived).
MEMORY_GUARD_GIB = 5.0

# Mover design (CONVOY-3D). Probabilities are exact proportions, not random draws:
# the sampler cycles a shuffled deck so every condition is represented as designed.
MOVER_DESIGN = {
    "classes": {"car": 0.40, "crate": 0.30, "crate_floating": 0.30},
    "eps_targets_px": {0.0: 0.40, 0.5: 0.08, 1.5: 0.07, 2.0: 0.08, 3.0: 0.09,
                       5.0: 0.10, 8.0: 0.10, 12.0: 0.08},
    "s_range_main": (-0.6, 0.85),
    "s_range_beyond": (1.10, 1.40), "p_beyond": 0.08,   # no static solution exists (Z* < 0)
    "depth_range_m": (5.0, 45.0),
}
CAMERA_DESIGN = {
    # heading of the optical axis relative to the direction of travel
    "forward": {"p": 0.40, "yaw_deg": (-8, 8), "baseline_m": (1.2, 3.0)},
    "oblique": {"p": 0.30, "yaw_deg": (20, 45), "baseline_m": (0.8, 2.0)},
    "side":    {"p": 0.30, "yaw_deg": (60, 80), "baseline_m": (0.5, 1.5)},
    "focal_px": (520.0, 820.0), "height_m": (1.3, 1.8), "rot_jitter_deg": 1.0,
}
# ---- v6 designs ----
# E5: the SAME mover rendered in five conditions; everything else in the scene is identical.
ABLATION_DESIGN = {
    "classes": {"car": 0.5, "crate": 0.5},             # base objects; every one is also lifted (floating car / crate)
    "conditions": ["G", "NS", "CH", "F", "FH"],
    "labels": {"G": "grounded", "NS": "no shadow", "CH": "contact hidden", "F": "floating", "FH": "floating, wall"},
    "lift_m": (0.8, 1.6),                              # F / FH: gap between the object and the ground
    "contact_cover_car_m": 0.70,                       # CH: wall hides the wheels (and up to 0.70 m)
    "contact_cover_crate_frac": 0.35,                  # CH: wall hides the lowest 35% of a crate
    "wall_gap_m": 0.6, "wall_max_cover_frac": 0.65, "wall_thickness_m": 0.25,
}
# E5-REAL (v6.1): the same crate inserted into a real TUM image pair twice, resting on the support plane
# (floor or table top, fitted to the measured depth) with a contact shadow (G), and lifted off it without
# a shadow (F). Purely parallel motion (delta = s b) on pairs whose baseline is almost parallel to the
# plane, so the grounded crate stays on the plane in both views.
RABL_DESIGN = {
    "conditions": ["G", "F"], "labels": {"G": "grounded, contact shadow", "F": "lifted, no shadow"},
    "crate_size_m": (0.15, 0.32), "lift_m": (0.25, 0.50), "depth_m": (0.7, 3.5),
    "max_baseline_tilt_deg": 6.0,                     # angle between the baseline and the support plane
    "vertical_max_tilt_deg": 12.0,                    # plane normal vs the motion-capture vertical (when it is identified)
    "plane_max_tilt_deg": 35.0,                       # fallback: plane normal vs the camera's up direction
    "plane_inlier_m": 0.02, "plane_fit_max_depth_m": 3.0, "plane_min_inlier_frac": 0.08, "plane_ransac_iters": 400,
    "on_plane_sigma_k": 2.5,                          # on-plane tolerance: max(2 cm, k sigma_Kinect(z)), Nguyen et al. 2012
    "shadow_strength": 0.55, "shadow_softness": 0.30,  # darkening at the footprint centre; soft edge (x size)
    "attempts": 300, "margin_px": 16, "max_candidates_per_seq": 600,
}
# E6: Virtual KITTI 2 (photo-realistic driving with exact depth, camera and object poses; no login)
# KITTI tracking (real driving, LiDAR, labelled 3-D boxes, OXTS poses). Not downloadable without a
# (free) registration: attach it as a Kaggle dataset (Section 3.10). Used for exploratory analyses only.
KT_DESIGN = {
    "root_env": "CONVOY_KITTI_ROOT", "classes": ("Car", "Van", "Truck"),
    "pair_baseline_m": (1.0, 3.0), "max_rot_deg": 10.0, "max_gap_frames": 10,
    "max_truncation": 0.3, "max_occlusion": 1, "lidar_min_depth_m": 2.0, "lidar_max_depth_m": 80.0,
    "static_box_margin_m": 0.3, "crop_h_max": 375,
    "zbuffer_window_px": 5, "zbuffer_tol": (0.10, 1.0),   # v6.3: occlusion test of projected LiDAR (Section 3.10.0)
}
# Argoverse 2 sensor dataset (v6.3, exploratory): real driving with two 32-beam LiDARs, labelled 3-D boxes with
# track IDs and ego poses, read anonymously from the public AWS Open Data bucket (CC BY-NC-SA 4.0). Front-centre
# camera (portrait 1550 x 2048, undistorted); the 4:3 window of the full width is resized to 640 x 480.
AV2_DESIGN = {
    "base_url": os.environ.get("CONVOY_AV2_BASE", "https://argoverse.s3.amazonaws.com/"),
    "prefix": "datasets/av2/sensor/", "split": "val", "camera": "ring_front_center",
    "classes": ("REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "TRUCK_CAB", "SCHOOL_BUS",
                "ARTICULATED_BUS", "VEHICULAR_TRAILER"),
    "out_wh": (640, 480),
    "pair_baseline_m": (1.0, 3.0), "max_rot_deg": 10.0, "max_gap_frames": 20, "min_start_gap_frames": 10,
    "lidar_min_depth_m": 2.0, "lidar_max_depth_m": 80.0, "static_box_margin_m": 0.3, "own_box_margin_m": 0.1,
    "ground_cut_m": 0.15, "zbuffer_window_px": 5, "zbuffer_tol": (0.10, 1.0),
    "max_truncation": 0.3, "max_occlusion": 0.5, "min_moving_speed_mps": 1.0,
    "min_pairs_per_log": 4, "max_logs_tried": 60,
}
# Argoverse 2, v6.4 (exploratory, AMENDMENT_V6_4): the extension stage and the ground contact of real vehicles. Kept apart
# from AV2_DESIGN so that the v6.3 stage keeps its fingerprint.
AV2X_DESIGN = {
    "max_logs_tried_total": 150,                       # both stages together; the validation split has 150 logs
    # A wheel contact point (Virtual KITTI 2 geometry: x6_wheel_height_m above the box bottom at +-0.35 length,
    # +-0.40 width) is visible in the first image of a pair if (i) it projects into the image in front of the camera,
    # (ii) the vehicle's own body does not hide it: the camera lies beyond the side face or the end face next to that
    # wheel, and the line of sight does not cross the vehicle's own box above body_clearance_m (a bumper overhanging the
    # wheel) at least occluder_gap_m in front of the point, (iii) no other labelled object's box, shrunk by
    # occluder_shrink_m on every side, crosses the line of sight at least occluder_gap_m in front of the point, and
    # (iv) no static LiDAR return in the window of columns u +- 2 and
    # rows v - 8 ... v (the point's row and the rows above it, which the near road cannot reach) is nearer than the
    # point by more than max(10%, 1 m). Contact labels as for Virtual KITTI 2: visible with >= 1 visible wheel point,
    # hidden with none when >= 2 lie in the image, otherwise undetermined.
    "body_clearance_m": 0.30, "occluder_shrink_m": 0.10, "occluder_gap_m": 0.30,
    "lidar_window_cols": 2, "lidar_window_rows_above": 8, "lidar_occluder_tol": (0.10, 1.0),
}
VK_DESIGN = {
    "base_url": os.environ.get("CONVOY_VKITTI_BASE", "https://download.europe.naverlabs.com/virtual_kitti_2.0.3/"),
    "scenes": ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"], "variation": "clone", "camera": 0,
    "crop_wh": (500, 375),                             # centred 4:3 crop of the 1242 x 375 frames, native pixels
    "pair_baseline_m": (1.0, 3.0), "max_rot_deg": 10.0, "max_gap_frames": 8,
    "min_vehicle_px": 300, "sky_depth_m": 600.0,
}

# ---- v6.5 designs (AMENDMENT_V6_5); new dictionaries, so no v6.4 stage changes its fingerprint ----
# Baseline sweep (BSW): the same scene and movers seen with the camera baseline scaled by each factor. The second camera
# keeps its rotation and moves along the same line, C = C0 + factor (C1 - C0), and every mover moves factor x its
# displacement, so s, eps = 0 and the geometric answer x = -log(1 - s) are unchanged: only the parallax changes.
# Scenes are sampled with 4 frames, so the sampler's constraints (movers in view, apart and in the street) hold up to
# three baselines.
SWEEP_DESIGN = {
    "classes": {"car": 0.35, "crate": 0.30, "crate_floating": 0.35},
    "levels": {"B05": 0.5, "B1": 1.0, "B2": 2.0, "B3": 3.0},
    "labels": {"B05": "x0.5", "B1": "x1", "B2": "x2", "B3": "x3"},
    "n_frames_sampled": 4,
}
# Contact dose-response (WDR): the same grounded mover with no wall (G) and behind static walls that hide its lowest
# 15, 35 and 70 cm in both views (the E5 wall, designed by ray casting, with the cover height as the dose; 70 cm is the
# E5 cover of a car). Base scenes as in E5 (cars and crates, purely parallel), new seeds.
WALL_DOSE_DESIGN = {
    "classes": {"car": 0.5, "crate": 0.5},
    "levels": {"G": 0.0, "W15": 0.15, "W35": 0.35, "W70": 0.70},
    "labels": {"G": "no wall", "W15": "15 cm", "W35": "35 cm", "W70": "70 cm"},
}
# Bonn RGB-D Dynamic Dataset (Palazzolo et al., IROS 2019): TUM format, Asus Xtion depth registered to the RGB images,
# motion-capture poses of a marker body. The pose of the RGB camera is T_ROS^-1 T(t) T_ROS T_m (dataset page); T_m is a
# calibrated similarity (scale about 1.06), so camera centres are expressed in the sensor's depth units. The
# convention is checked on the data (A3), as for TUM. Images and depth are undistorted with the published coefficients.
BONN_DESIGN = {
    "base_url": os.environ.get("CONVOY_BONN_BASE", "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/"),
    "sequences": {"balloon": dict(file="rgbd_bonn_balloon.zip", object="balloon"),
                  "balloon2": dict(file="rgbd_bonn_balloon2.zip", object="balloon"),
                  "moving_nonobstructing_box": dict(file="rgbd_bonn_moving_nonobstructing_box.zip", object="box"),
                  "moving_nonobstructing_box2": dict(file="rgbd_bonn_moving_nonobstructing_box2.zip", object="box")},
    "intrinsics": dict(fx=542.822841, fy=542.576870, cx=315.593520, cy=237.756098, W=640, H=480),
    "distortion": [0.039903, -0.099343, -0.000730, -0.000144, 0.0],
    "depth_scale": 5000.0,
    "T_ros": [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    "T_m": [[1.0157, 0.1828, -0.2389, 0.0113], [0.0009, -0.8431, -0.6413, -0.0098], [-0.3009, 0.6147, -0.8085, 0.0111],
            [0.0, 0.0, 0.0, 1.0]],
    "pair_baseline_m": (0.05, 0.25), "max_rot_deg": 20.0,
    "person_score_min": 0.6, "person_dilate_px": 5,       # people: torchvision Mask R-CNN (COCO person class)
    "min_component_px": 400,                             # moving regions: the TUM walking rule of Section 3.7
    "static_patch_px": 40, "n_static_patches": 30,       # flow-noise floor: |eps| of static patches (as parked cars)
    # v6.6 (D62): the relative pose of every pair from its static correspondences (RANSAC PnP of frame i's measured points
    # against their flow positions in frame j, started from the motion-capture pose); pairs whose fit fails are left out
    "pnp_points": 6000, "pnp_reproj_px": 1.5, "pnp_min_inlier_frac": 0.4, "pnp_max_med_err_px": 0.8,
    "pnp_max_rot_change_deg": 5.0, "pnp_max_trans_change_m": 0.10,
    "max_object_frac": 0.20, "max_object_extent": 0.70,  # an object covers <= 20% of the image and <= 70% of each side
    # a pixel hidden in frame j behind something nearer (all depth within occl_window_px of its predicted position is
    # nearer) is not tested; a moving region is an object only if the static prediction fails photometrically (median
    # ZNCC of 7 x 7 patches between frame i and frame j warped with the rigid flow <= photo_max_zncc_static on >= 30
    # textured pixels), the measured flow explains frame j at least as well, it contains a disc of radius 5 px and, if the
    # floor is found, 10% of its points are at least 5 cm above the floor (a moving shadow on the floor is not an object)
    "occl_window_px": 2, "photo_window_px": 3, "photo_min_texture": 4.0, "photo_min_px": 30,
    "photo_max_zncc_static": 0.85, "photo_min_support": 0.0, "min_object_radius_px": 5.0, "min_object_height_m": 0.05,
    "floor_cam_height_m": (0.9, 2.2), "floor_min_frac": 0.02, "floor_max_tilt_deg": 10.0,   # the floor: the lowest level
    "qc_max_object_frac": 0.15,                          # Bonn is not interpreted if the objects of the median pair cover
                                                         # more than 15% of the image (the static scene would be flagged)
}
# The static twin (Proposition 4) on Argoverse 2: the camera height above each vehicle's contact plane from its labelled
# box, and the twin test against the epipolar test (exploratory, Section 6.13).
TWIN_DESIGN = {
    "s_bins": [-1.0, -0.4, -0.15, -0.05, 0.05, 0.15, 0.4, 0.7, 1.0],
    "theory_s": [-0.9, 0.95],
}
# The twin test with optical flow (v6.6, D63, H14): correspondences from RAFT flow (forward-backward checked), the road
# plane fitted by RANSAC to all LiDAR returns of the image (no labels), and the 10th percentile of the heights of a
# vehicle's static reconstruction above that plane; a static solution behind the camera for most pixels counts as none.
FLOWTWIN_DESIGN = {"road_depth_m": (2.0, 40.0), "road_inlier_m": 0.08, "road_max_tilt_deg": 10.0, "road_cam_height_m": (0.8, 3.0),
                   "ransac_iters": 300, "fb_px": 1.0, "fb_rel": 0.05, "min_px": 10, "low_pct": 10, "behind_frac": 0.5}
# The cue model (exploratory, Section 6.11): w = c_support * d^2 / (d^2 + kappa^2), d = translational parallax (px)
CUE_DESIGN = {
    "kappa_grid_px": [0.0] + [float(v) for v in np.round(np.logspace(0, 3, 61), 3)],
    "n_folds": 10, "parallax_bins_px": [0, 5, 10, 20, 40, 80, 1e9],
    "x_bins": [-0.6, -0.25, 0.0, 0.25, 0.5, 0.8, 1.2, 2.0],
}

# Optical flow used for the flow-based variants of M1/M2 and for the real walking ground truth
FLOW_PREFERRED = "RAFT-large (torchvision C_T_SKHT_V2); DIS (OpenCV) if the weights cannot be downloaded"

# Version of each stage's code. A stage's checkpoint is reused only when its version, its slice of
# the configuration and the fingerprints of the stages it reads are all unchanged (Section 0.5).
STAGE_VERSIONS = {"render": "v6.0", "tum": "v6.0", "flow": "v6.0", "vk": "v6.2", "eval": "v6.2", "kitti": "v6.3", "av2": "v6.3",
                  "av2x": "v6.4", "render65": "v6.5", "bonn": "v6.6", "av2t": "v6.5", "av2ft": "v6.6"}


def _jsonable(d):
    return {k: ({str(a): b for a, b in v.items()} if isinstance(v, dict) else v) for k, v in d.items()}


def write_run_config():
    with open(os.path.join(MANIFEST_DIR, "run_config.json"), "w") as _f:
        json.dump({"RUN_CONFIG": RUN_CONFIG, "PREREG": PREREG, "MOVER_DESIGN": _jsonable(MOVER_DESIGN),
                   "CAMERA_DESIGN": CAMERA_DESIGN, "ABLATION_DESIGN": _jsonable(ABLATION_DESIGN),
                   "RABL_DESIGN": _jsonable(RABL_DESIGN), "VK_DESIGN": VK_DESIGN, "KT_DESIGN": KT_DESIGN, "AV2_DESIGN": AV2_DESIGN,
                   "AV2X_DESIGN": AV2X_DESIGN, "SWEEP_DESIGN": SWEEP_DESIGN, "WALL_DOSE_DESIGN": WALL_DOSE_DESIGN,
                   "BONN_DESIGN": BONN_DESIGN, "TWIN_DESIGN": TWIN_DESIGN, "CUE_DESIGN": CUE_DESIGN,
                   "FLOWTWIN_DESIGN": FLOWTWIN_DESIGN, "STAGE_VERSIONS": STAGE_VERSIONS},
                  _f, indent=2, default=str)


for _name, _d in [("RUN_CONFIG", RUN_CONFIG), ("PREREG", PREREG), ("MOVER_DESIGN", MOVER_DESIGN),
                  ("CAMERA_DESIGN", CAMERA_DESIGN), ("ABLATION_DESIGN", ABLATION_DESIGN),
                  ("RABL_DESIGN", RABL_DESIGN), ("VK_DESIGN", VK_DESIGN), ("KT_DESIGN", KT_DESIGN), ("AV2_DESIGN", AV2_DESIGN),
                  ("AV2X_DESIGN", AV2X_DESIGN), ("SWEEP_DESIGN", SWEEP_DESIGN), ("WALL_DOSE_DESIGN", WALL_DOSE_DESIGN),
                  ("BONN_DESIGN", BONN_DESIGN), ("TWIN_DESIGN", TWIN_DESIGN), ("CUE_DESIGN", CUE_DESIGN),
                  ("FLOWTWIN_DESIGN", FLOWTWIN_DESIGN)]:
    print(_name)
    for _k, _v in _d.items():
        print(f"    {_k:<30}{_v}")
write_run_config()
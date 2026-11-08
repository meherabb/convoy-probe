"""Registered constants of the study (sample rules, thresholds, verdict rules).

These are the values of ``PREREG`` and of the design dictionaries in notebook cell 0.4
(``notebook/src/cells/0100_run_config.py``) used for every number in the paper; the run record is
``results/v6.6/manifest/run_config.json`` and ``tests/test_registered.py`` checks that both agree.
The registration history (what was fixed when) is in ``registration/README.md``.
"""
from __future__ import annotations

import hashlib

GLOBAL_SEED = 20270918   # every random stream of the notebook derives from this seed (seed_for)
N_BOOT = 10_000          # cluster-bootstrap resamples of every registered interval
N_BOOT_EXPLORATORY = 2_000   # resamples of the exploratory per-object medians (driving data)

#: ``PREREG`` of notebook cell 0.4, verbatim.
PREREG = {'eps_parallel_max': 1.0,
 'eps_perp_min': 3.0,
 'parallax_min_px': 5.0,
 'gate_slope_range': [0.7, 1.3],
 'gate_absrel_max': 0.25,
 'beta_support_range': [0.8, 1.2],
 'beta_ci_band': [0.5, 1.5],
 'r2_min': 0.5,
 'recall_diff_min': 0.3,
 'mask_fpr': 0.05,
 'spearman_min': 0.5,
 'w_separability_min': 0.1,
 'e4_drop_min': 0.3,
 'min_object_pixels': 40,
 'erosion_px': 2,
 'static_depth_range_synth': [2.0, 80.0],
 'static_depth_range_real': [0.3, 4.5],
 'static_depth_range_vk': [2.0, 80.0],
 'h9_beta_max': 0.3,
 'h9_ci_hi_max': 0.5,
 'h9_x_range': [-0.47, 1.9],
 'h9_min_movers': 20,
 'h9_min_pairs': 5,
 'min_movers_per_regime': 20,
 'contact_band_m': 0.12,
 'vk_min_occupancy': 0.85,
 'rabl_contact_band_m': 0.03,
 'x6_min_eroded_px': 40,
 'x6_min_lever': 0.3,
 'x6_pace_s': [0.85, 1.15],
 'x6_wheel_height_m': 0.1,
 'x6_wheel_depth_tol_m': 0.6,
 'kt_min_lidar_px': 40,
 'kt_moving_disp_m': 0.5,
 'kt_parked_disp_m': 0.15,
 'kt_eps_noise_quantile': 0.9,
 'x6_match_caliper_sd': 0.5,
 'implied_factor_s': 0.85,
 'h12_contrast': ['B3', 'B05'],
 'h12_models': ['DUSt3R', 'MASt3R'],
 'h13_contrast': ['W15', 'G'],
 'h13_contact_px_min': 5,
 'dose_min_movers': 8,
 'dose_min_scenes': 5,
 'twin_fpr': 0.05,
 'bonn_min_object_px': 150,
 'bonn_floor_contact_m': 0.08,
 'bonn_floor_clear_m': 0.2,
 'h14_min_vehicles': 20,
 'h14_min_tracks': 10}

#: Parallel fractions and design ranges of the CONVOY-3D movers (``MOVER_DESIGN``).
MOVER_DESIGN = {'classes': {'car': 0.4, 'crate': 0.3, 'crate_floating': 0.3},
 'eps_targets_px': {'0.0': 0.4,
                    '0.5': 0.08,
                    '1.5': 0.07,
                    '2.0': 0.08,
                    '3.0': 0.09,
                    '5.0': 0.1,
                    '8.0': 0.1,
                    '12.0': 0.08},
 's_range_main': [-0.6, 0.85],
 's_range_beyond': [1.1, 1.4],
 'p_beyond': 0.08,
 'depth_range_m': [5.0, 45.0]}

#: Camera designs of CONVOY-3D (``CAMERA_DESIGN``).
CAMERA_DESIGN = {'forward': {'p': 0.4, 'yaw_deg': [-8, 8], 'baseline_m': [1.2, 3.0]},
 'oblique': {'p': 0.3, 'yaw_deg': [20, 45], 'baseline_m': [0.8, 2.0]},
 'side': {'p': 0.3, 'yaw_deg': [60, 80], 'baseline_m': [0.5, 1.5]},
 'focal_px': [520.0, 820.0],
 'height_m': [1.3, 1.8],
 'rot_jitter_deg': 1.0}

#: Bins of the parallel fraction and range of the theoretical detection curve of the twin test (``TWIN_DESIGN``).
TWIN_DESIGN = {'s_bins': [-1.0, -0.4, -0.15, -0.05, 0.05, 0.15, 0.4, 0.7, 1.0], 'theory_s': [-0.9, 0.95]}

#: Road plane and flow rules of the twin test with optical flow, H14 (``FLOWTWIN_DESIGN``).
FLOWTWIN_DESIGN = {'road_depth_m': [2.0, 40.0],
 'road_inlier_m': 0.08,
 'road_max_tilt_deg': 10.0,
 'road_cam_height_m': [0.8, 3.0],
 'ransac_iters': 300,
 'fb_px': 1.0,
 'fb_rel': 0.05,
 'min_px': 10,
 'low_pct': 10,
 'behind_frac': 0.5}


def seed_for(*keys) -> int:
    """Stable 32-bit seed derived from ``GLOBAL_SEED`` and any keys (identical to the notebook)."""
    h = hashlib.sha256(("|".join(map(str, (GLOBAL_SEED,) + keys))).encode()).hexdigest()
    return int(h[:8], 16)

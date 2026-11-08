"""The static-twin test (paper, Proposition 3 and Sec. 6.3).

A parallel mover is explained by a static twin scaled by ``e^x`` about the first camera, whose
contact point lies at height ``h = H (1 - e^x)`` above the object's contact plane (``H``: camera
height above that plane). A grounded object therefore has a twin sunk into the road
(``0 < s < 1``) or floating above it (``s < 0``); the twin test flags ``|h|`` beyond what parked
vehicles show. Both tests run at the same false-positive rate on parked vehicles: the epipolar
test thresholds ``|eps|``, the twin test thresholds the twin height.

Two versions, as in the paper:

* **labels** (exploratory): exact correspondences and camera heights from the labelled boxes of
  Argoverse 2 (``h = H (1 - e^x_geo)``; vehicles with ``s >= 1`` have no static solution);
* **flow** (registered as H14): forward-backward checked optical flow at each vehicle's LiDAR
  pixels and a road plane fitted to the LiDAR; the statistic is the 10th percentile of the heights
  of the vehicle's static reconstruction above the road, flagged outside its 2.5-97.5% range on
  parked vehicles. H14 is the paired difference ``det_twin - det_epipolar`` in the parallel regime.

Mirrors notebook cell 6.13 (``0889h_twin.py``); ``tests/test_notebook_equivalence.py`` runs that
cell and these functions on the same vehicles and checks that every output agrees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .registered import FLOWTWIN_DESIGN, PREREG, TWIN_DESIGN, seed_for
from .stats import ClusterBootstrap, percentile_ci, share_with_ci, verdict_positive

__all__ = [
    "REGIMES", "twin_height", "label_thresholds", "label_flags", "flow_static_solution", "flow_thresholds",
    "flow_flags", "recall_table", "recall_by_s", "detection_curve", "h14_contrast", "run_label_test",
    "run_flow_test",
]

REGIMES = ("parallel", "intermediate", "perpendicular", "all")
FLAG_COLUMNS = ["det_epipolar", "det_twin", "det_either", "twin_below_road"]


def twin_height(H_cam, x_geo):
    """Contact height of the static twin above the contact plane, ``H (1 - e^x)`` (Proposition 3)."""
    return np.asarray(H_cam, np.float64) * (1.0 - np.exp(np.asarray(x_geo, np.float64)))


# ---- labelled correspondences (exploratory) -------------------------------------------------------------------

def label_thresholds(parked_eps, parked_h_twin, fpr=None):
    """Thresholds at ``fpr`` false positives on parked vehicles: ``|eps|`` and ``|h|`` above their (1 - fpr) quantiles."""
    q = 1.0 - (PREREG["twin_fpr"] if fpr is None else fpr)
    eps = np.asarray(parked_eps, np.float64)
    h = np.asarray(parked_h_twin, np.float64)
    return dict(n_parked=int(len(h)), eps_px=float(np.quantile(eps, q)), height_m=float(np.quantile(np.abs(h), q)))


def label_flags(eps, h_twin, no_static_solution, thresholds):
    """Flags of the epipolar test, the twin test, either test, and a twin below the road."""
    eps = np.asarray(eps, np.float64)
    h = np.asarray(h_twin, np.float64)
    none = np.asarray(no_static_solution, bool)
    det_epipolar = eps > thresholds["eps_px"]
    det_twin = np.where(none, False, np.abs(h) > thresholds["height_m"])
    return dict(det_epipolar=det_epipolar, det_twin=det_twin, det_either=det_epipolar | det_twin,
                twin_below_road=det_twin & (h < 0))


# ---- optical flow and a LiDAR road plane (H14) ----------------------------------------------------------------

def flow_static_solution(frac_behind, h_low_flow, behind_frac=None):
    """A static reconstruction exists if at most half of the vehicle's pixels triangulate behind the camera."""
    behind_frac = FLOWTWIN_DESIGN["behind_frac"] if behind_frac is None else behind_frac
    return (np.asarray(frac_behind, np.float64) <= behind_frac) & np.isfinite(np.asarray(h_low_flow, np.float64))


def flow_thresholds(parked_eps_flow, parked_h_low, fpr=None):
    """``|eps|`` above its (1 - fpr) quantile; the low height outside its [fpr/2, 1 - fpr/2] range (parked vehicles)."""
    a = PREREG["twin_fpr"] if fpr is None else fpr
    eps = np.asarray(parked_eps_flow, np.float64)
    h = np.asarray(parked_h_low, np.float64)
    return dict(n_parked=int(len(h)), eps_px=float(np.quantile(eps, 1 - a)), h_low_lo_m=float(np.quantile(h, a / 2)),
                h_low_hi_m=float(np.quantile(h, 1 - a / 2)))


def flow_flags(eps_flow, h_low_flow, static_solution, thresholds):
    eps = np.asarray(eps_flow, np.float64)
    h = np.asarray(h_low_flow, np.float64)
    ok = np.asarray(static_solution, bool)
    det_epipolar = eps > thresholds["eps_px"]
    low, high = h < thresholds["h_low_lo_m"], h > thresholds["h_low_hi_m"]
    det_twin = ok & (low | high)
    return dict(det_epipolar=det_epipolar, det_twin=det_twin, det_either=det_epipolar | det_twin,
                twin_below_road=det_twin & low)


# ---- recall tables, detection curve and H14 -------------------------------------------------------------------

def _shares(q, cols, seed, n_boot=None):
    res = share_with_ci([q[c].to_numpy(float) for c in cols], q["cl"].to_numpy(), seed, n_boot=n_boot)
    return dict(zip(cols, res))


def recall_table(vehicles, seed_key="twin", n_boot=None):
    """Share of moving vehicles flagged by each test, by regime, with intervals over vehicle tracks.

    ``vehicles`` needs ``regime``, ``cl`` (track id) and the flag columns. ``twin_given_epi_miss`` is
    the twin test's share among the vehicles that the epipolar test misses.
    """
    rows = []
    for reg in REGIMES:
        q = vehicles if reg == "all" else vehicles[vehicles["regime"] == reg]
        r = _shares(q, FLAG_COLUMNS, seed_for(seed_key, reg), n_boot)
        qm = q[~q["det_epipolar"].astype(bool)]
        rm = _shares(qm, ["det_twin"], seed_for(f"{seed_key}-miss", reg), n_boot)["det_twin"]
        rows.append(dict(regime=reg, n=len(q), n_tracks=int(q["cl"].nunique()), **{c: r[c][0] for c in FLAG_COLUMNS},
                         **{f"{c}_lo": r[c][1] for c in FLAG_COLUMNS}, **{f"{c}_hi": r[c][2] for c in FLAG_COLUMNS},
                         n_epi_miss=len(qm), twin_given_epi_miss=rm[0], twin_given_epi_miss_lo=rm[1],
                         twin_given_epi_miss_hi=rm[2]))
    return pd.DataFrame(rows)


def recall_by_s(vehicles, s_bins=None, n_boot=None):
    """Recall of the three tests in bins of the parallel fraction ``s`` (labelled version, Fig. 4c of the paper)."""
    s_bins = TWIN_DESIGN["s_bins"] if s_bins is None else s_bins
    cols = FLAG_COLUMNS[:3]
    rows = []
    for k in range(len(s_bins) - 1):
        q = vehicles[(vehicles["s"] >= s_bins[k]) & (vehicles["s"] < s_bins[k + 1])]
        if not len(q):
            continue
        r = _shares(q, cols, seed_for("twin-s", k), n_boot)
        rows.append(dict(s_lo=s_bins[k], s_hi=s_bins[k + 1], s_median=float(q["s"].median()), n=len(q),
                         frac_parallel=float((q["regime"] == "parallel").mean()), **{c: r[c][0] for c in cols},
                         **{f"{c}_lo": r[c][1] for c in cols}, **{f"{c}_hi": r[c][2] for c in cols}))
    return pd.DataFrame(rows)


def detection_curve(H_median_m, parked_h_twin, height_thr_m, s_values=None, fpr=None):
    """Detection probability of the twin test for exactly parallel motion (dotted curve of Fig. 4c of the paper).

    The twin height ``-H s / (1 - s)`` at the median camera height plus the twin heights of parked
    vehicles as noise, against the same threshold; the epipolar test stays at the false-positive rate.
    """
    s_values = np.linspace(*TWIN_DESIGN["theory_s"], 75) if s_values is None else np.asarray(s_values, float)
    noise = np.asarray(parked_h_twin, np.float64)
    rows = []
    for s in s_values:
        h0 = -H_median_m * s / (1.0 - s)
        rows.append(dict(s=float(s), twin_height_m=float(h0), p_detect_twin=float(np.mean(np.abs(h0 + noise) > height_thr_m)),
                         p_detect_epipolar=PREREG["twin_fpr"] if fpr is None else fpr))
    return pd.DataFrame(rows)


def h14_contrast(det_twin, det_epipolar, clusters, seed=None, n_boot=None, min_vehicles=None, min_tracks=None):
    """Registered H14: mean of ``det_twin - det_epipolar`` over parallel-regime vehicles (paired), CI over tracks."""
    min_vehicles = PREREG["h14_min_vehicles"] if min_vehicles is None else min_vehicles
    min_tracks = PREREG["h14_min_tracks"] if min_tracks is None else min_tracks
    seed = seed_for("h14") if seed is None else seed
    t = np.asarray(det_twin, float)
    e = np.asarray(det_epipolar, float)
    cl = np.asarray(clusters)
    out = dict(n=int(len(t)), n_tracks=int(len(np.unique(cl))), det_twin=float(t.mean()) if len(t) else np.nan,
               det_epipolar=float(e.mean()) if len(e) else np.nan, estimate=np.nan, ci_lo=np.nan, ci_hi=np.nan,
               verdict="NOT TESTABLE")
    if len(t) >= min_vehicles and out["n_tracks"] >= min_tracks:
        T, P = ClusterBootstrap(cl, n_boot=n_boot, seed=seed).linear([np.ones(len(t)), t - e])
        out["estimate"] = float(P[1] / P[0])
        out["ci_lo"], out["ci_hi"] = percentile_ci(T[:, 1] / T[:, 0])
        out["verdict"] = verdict_positive(out["ci_lo"], out["ci_hi"])
    return out


# ---- the complete tests ----------------------------------------------------------------------------------------

def _track_ids(df):
    return df["seq"].astype(str) + "/" + df["track"].astype(str)


def run_label_test(vehicles, n_boot=None):
    """Twin test with labelled correspondences, as in the notebook.

    ``vehicles``: one row per vehicle and image pair with ``seq``, ``track``, ``state`` (moving or
    parked), ``regime``, ``s``, ``measurable``, ``H_cam``, ``gt_eps_med`` and ``gt_x_geo``.
    Returns ``dict(thresholds, by_regime, by_s, curve, vehicles)`` or ``None`` with fewer than 20 parked vehicles.
    """
    t = vehicles[vehicles["measurable"].astype(bool) & np.isfinite(vehicles["H_cam"])
                 & np.isfinite(vehicles["gt_eps_med"])].copy()
    t["h_twin"] = twin_height(t["H_cam"], t["gt_x_geo"])
    t["no_static_solution"] = ~np.isfinite(t["gt_x_geo"])
    t["cl"] = _track_ids(t)
    pk = t[(t["state"] == "parked") & np.isfinite(t["h_twin"])]
    if len(pk) < 20:
        return None
    thr = label_thresholds(pk["gt_eps_med"], pk["h_twin"])
    thr["H_median_m"] = float(t["H_cam"].median())
    mv = t[t["state"] == "moving"].copy()
    for k, v in label_flags(mv["gt_eps_med"], mv["h_twin"], mv["no_static_solution"], thr).items():
        mv[k] = v
    valid = mv[~mv["no_static_solution"]]
    thr.update(n_moving=len(mv), n_no_static_solution=int(mv["no_static_solution"].sum()),
               parked_fpr_epipolar=float(np.mean(pk["gt_eps_med"] > thr["eps_px"])),
               parked_fpr_twin=float(np.mean(np.abs(pk["h_twin"]) > thr["height_m"])))
    return dict(thresholds=thr, by_regime=recall_table(valid, "twin", n_boot), by_s=recall_by_s(valid, n_boot=n_boot),
                curve=detection_curve(thr["H_median_m"], pk["h_twin"].to_numpy(float), thr["height_m"]), vehicles=mv)


def run_flow_test(vehicles, n_boot=None):
    """Twin test with optical flow and a LiDAR road plane, and the registered H14.

    ``vehicles``: as for :func:`run_label_test`, plus ``n_flow_px``, ``eps_flow``, ``h_low_flow``
    and ``frac_behind`` from the flow stage. Returns ``dict(thresholds, by_regime, H14, vehicles)``
    or ``None`` with fewer than 20 parked vehicles.
    """
    f = vehicles[vehicles["measurable"].astype(bool) & (vehicles["n_flow_px"] >= FLOWTWIN_DESIGN["min_px"])
                 & np.isfinite(vehicles["eps_flow"])].copy()
    f["cl"] = _track_ids(f)
    f["static_solution"] = flow_static_solution(f["frac_behind"], f["h_low_flow"])
    pk = f[(f["state"] == "parked") & f["static_solution"]]
    if len(pk) < 20:
        return None
    thr = flow_thresholds(pk["eps_flow"], pk["h_low_flow"])
    if "h_low_true" in pk:          # descriptive: lowest LiDAR returns of parked vehicles above the road plane
        thr["parked_h_low_true_median_m"] = float(pk["h_low_true"].median())
    if "x_flow" in pk:
        thr["parked_x_flow_abs_median"] = float(np.median(np.abs(pk["x_flow"])))
    fm = f[f["state"] == "moving"].copy()
    for k, v in flow_flags(fm["eps_flow"], fm["h_low_flow"], fm["static_solution"], thr).items():
        fm[k] = v
    fv = fm[fm["static_solution"]]
    pk_flags = flow_flags(pk["eps_flow"], pk["h_low_flow"], pk["static_solution"], thr)
    thr.update(n_moving=len(fm), n_no_static_solution=int((~fm["static_solution"]).sum()),
               parked_fpr_epipolar=float(np.mean(pk_flags["det_epipolar"])),
               parked_fpr_twin=float(np.mean(pk_flags["det_twin"])))
    q = fv[fv["regime"] == "parallel"]
    h14 = h14_contrast(q["det_twin"], q["det_epipolar"], q["cl"], n_boot=n_boot)
    return dict(thresholds=thr, by_regime=recall_table(fv, "ftwin", n_boot), H14=h14, vehicles=fm)

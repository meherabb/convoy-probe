# PLACEHOLDER
# 6.8b Exploratory (v6.2, extended in v6.3 and v6.4): ground contact and the convoy case in driving.
#      Virtual KITTI 2 (exact ground truth), Argoverse 2 (real, LiDAR; automatic download) and, when
#      attached, KITTI tracking (real, LiDAR). Per-object geometry share g (Section 6.8b text), medians
#      with cluster-bootstrap intervals. v6.3: the cluster is the vehicle track (a vehicle seen in several
#      pairs counts once per resample), and the hidden-contact contrast is repeated on vehicles matched
#      for distance, image size and geometric lever. v6.4: the same contrast on real Argoverse 2 vehicles
#      (contact labels of Section 3.11b, D50), on all 64 logs. No verdicts: directions only.
def geo_share_obj(rho, rho_mono, x):
    a, b, c = np.exp(np.asarray(rho, float)), np.exp(np.asarray(rho_mono, float)), np.exp(np.asarray(x, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (c - b)

def _cboot(df, fn, seed, B=None, cluster="cl"):
    # cluster bootstrap of a statistic fn(df) -> float; percentile 95% CI (2,000 resamples: these
    # medians are exploratory and the resampling is done in pandas)
    B = B or min(BOOT_N, 2000)
    est = fn(df)
    groups = [g for _, g in df.groupby(cluster)]
    if len(groups) < 5 or not np.isfinite(est):
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        draws[b] = fn(pd.concat([groups[k] for k in rng.integers(0, len(groups), len(groups))], ignore_index=True))
    draws = draws[np.isfinite(draws)]
    if len(draws) < 0.9 * B:
        return est, np.nan, np.nan
    return est, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

def _dsample(set_, model):
    d = MOV_RES[(MOV_RES["set"] == set_) & (MOV_RES["model"] == model)].copy() if len(MOV_RES) else pd.DataFrame()
    if not len(d):
        return d
    if set_ == "VK":
        ok = (d["gt_n_eroded"] >= PREREG["x6_min_eroded_px"])
    else:
        ok = d["measurable"].astype(bool)
    ok &= (d["regime"] == "parallel") & np.isfinite(d["rho"]) & np.isfinite(d["rho_mono"])
    d = d[ok].copy()
    # identifiable (parallax >= 5 px) is required for the contact test and the slope, not for the convoy
    # case: a vehicle that keeps pace with the camera has almost no parallax by construction, and its
    # geometric answer then lies far away, which is exactly what the test needs
    d["ident"] = d["gt_parallax_med"] >= PREREG["parallax_min_px"]
    d["g"] = geo_share_obj(d["rho"], d["rho_mono"], d["gt_x_geo"])
    d["lever"] = np.abs(d["gt_x_geo"] - d["rho_mono"])
    d["cl"] = d["seq"].astype(str) + "/" + d["track"].astype(str)        # vehicle track (v6.3 cluster)
    return d

def _pooled_sd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)) if len(a) > 1 and len(b) > 1 else np.nan

def _smd(a, b, sd):
    # standardized mean difference; before and after matching both use the pooled SD of the unmatched groups
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((np.nanmean(a) - np.nanmean(b)) / sd) if len(a) and len(b) and np.isfinite(sd) and sd > 0 else np.nan

_MATCH_COV = ("logZ", "logA", "gt_x_geo")

def match_hidden_visible(dl, caliper):
    # every vehicle with hidden contact is paired with the nearest vehicle with visible contact (with
    # replacement) on log Z, log image area and x_geo in pooled-SD units, within the caliper on each
    dl = dl.assign(logZ=np.log(dl["gt_Z_true"].astype(float)), logA=np.log(dl["gt_area_px"].astype(float)))
    hid, vis = dl[dl["gt_contact"] == "hidden"], dl[dl["gt_contact"] == "visible"]
    sd = dl[list(_MATCH_COV)].std(ddof=1).replace(0, np.nan).to_numpy(float)
    zv = vis[list(_MATCH_COV)].to_numpy(float) / sd
    rows = []
    for (_, h), zh in zip(hid.iterrows(), hid[list(_MATCH_COV)].to_numpy(float) / sd):
        dist = np.abs(zv - zh)
        ok = np.all(dist <= caliper, axis=1) & np.all(np.isfinite(dist), axis=1)
        if not ok.any():
            continue
        k = int(np.argmin(np.where(ok, np.sqrt((dist ** 2).sum(1)), np.inf)))
        v = vis.iloc[k]
        rows.append(dict(cl=h["cl"], g_h=h["g"], g_v=v["g"], **{f"h_{c}": h[c] for c in _MATCH_COV}, **{f"v_{c}": v[c] for c in _MATCH_COV}))
    return pd.DataFrame(rows), hid, vis

_med_g = lambda q: float(np.median(q["g"])) if len(q) else np.nan
X6_CONTACT_ROWS, X6_CONTACT_TESTS, X6_MATCH_ROWS, X6_PACE_ROWS, REAL_BETA_ROWS = [], [], [], [], []
REAL_DRIVE = [(k, n, df) for k, n, df in (("AV2", "Argoverse 2", MOVERS_AV2), ("KT", "KITTI tracking", MOVERS_KT)) if len(df)]
_drive = [("VK", "Virtual KITTI 2", MODELS_OK["VK"])] + [(k, n, MODELS_OK[k]) for k, n, _ in REAL_DRIVE]
for set_, dname, models in _drive:
    for model in models:
        d = _dsample(set_, model)
        if not len(d):
            continue
        # (1) ground contact: Virtual KITTI 2 (wheels tested with instance masks, D37) and, v6.4, Argoverse 2 (wheels
        #     tested with geometry and LiDAR, D50); KITTI tracking has no contact labels
        if set_ in ("VK", "AV2") and "gt_contact" in d and d["gt_contact"].isin(["visible", "hidden"]).any():
            _sk = () if set_ == "VK" else (set_,)          # Virtual KITTI 2 keeps its v6.3 resampling seeds
            dl = d[d["ident"] & np.isfinite(d["g"]) & (d["lever"] >= PREREG["x6_min_lever"]) & d["gt_contact"].isin(["visible", "hidden"])]
            for grp in ("visible", "hidden"):
                q = dl[dl["gt_contact"] == grp]
                est, lo, hi = _cboot(q, _med_g, seed_for("x6c", *_sk, model, grp)) if len(q) >= 8 else (_med_g(q), np.nan, np.nan)
                X6_CONTACT_ROWS.append(dict(dataset=dname, model=model, contact=grp, n=len(q), n_tracks=int(q["cl"].nunique()),
                                            median_g=est, ci_lo=lo, ci_hi=hi))
            nh, nv = int((dl["gt_contact"] == "hidden").sum()), int((dl["gt_contact"] == "visible").sum())
            if nh >= 8 and nv >= 8:
                fn = lambda q: (float(np.median(q.loc[q["gt_contact"] == "hidden", "g"])) - float(np.median(q.loc[q["gt_contact"] == "visible", "g"]))
                                if (q["gt_contact"] == "hidden").any() and (q["gt_contact"] == "visible").any() else np.nan)
                est, lo, hi = _cboot(dl, fn, seed_for("x6c-diff", *_sk, model))
            else:
                est = lo = hi = np.nan
            X6_CONTACT_TESTS.append(dict(dataset=dname, model=model, n_hidden=nh, n_visible=nv,
                                         n_hidden_tracks=int(dl.loc[dl["gt_contact"] == "hidden", "cl"].nunique()),
                                         diff_hidden_minus_visible=est, ci_lo=lo, ci_hi=hi,
                                         direction=_dir_label(lo, hi) if np.isfinite(est) else "NOT TESTABLE"))
            # (1b, v6.3) the same contrast on matched vehicles: distance, image size and lever held fixed
            mt, hid, vis = match_hidden_visible(dl, PREREG["x6_match_caliper_sd"])
            rec = dict(dataset=dname, model=model, n_hidden=nh, n_matched=len(mt), n_matched_tracks=int(mt["cl"].nunique()) if len(mt) else 0)
            if len(mt) >= 8:
                fnm = lambda q: float(np.median(q["g_h"] - q["g_v"])) if len(q) else np.nan
                est, lo, hi = _cboot(mt, fnm, seed_for("x6c-match", *_sk, model))
            else:
                est, lo, hi = (float(np.median(mt["g_h"] - mt["g_v"])) if len(mt) else np.nan), np.nan, np.nan
            rec.update(median_paired_diff=est, ci_lo=lo, ci_hi=hi,
                       direction=_dir_label(lo, hi) if np.isfinite(lo) else "NOT TESTABLE")
            for c in _MATCH_COV:                               # covariate balance (standardized mean difference)
                _sd = _pooled_sd(hid[c], vis[c])
                rec[f"smd_before_{c}"] = _smd(hid[c], vis[c], _sd)
                rec[f"smd_after_{c}"] = _smd(mt[f"h_{c}"], mt[f"v_{c}"], _sd) if len(mt) else np.nan
            X6_MATCH_ROWS.append(rec)
        # (2) the convoy case: vehicles keeping pace with the camera
        lo_s, hi_s = PREREG["x6_pace_s"]
        p1 = d[(d["s"] >= lo_s) & (d["s"] < 1.0) & np.isfinite(d["g"]) & (d["lever"] >= PREREG["x6_min_lever"])]
        p2 = d[(d["s"] >= 1.0) & (d["s"] < hi_s)]
        est, lo, hi = _cboot(p1, _med_g, seed_for("x6p", set_, model)) if len(p1) >= 8 else (_med_g(p1), np.nan, np.nan)
        closer = float(np.mean(np.abs(p1["rho"] - p1["rho_mono"]) < np.abs(p1["rho"] - p1["gt_x_geo"]))) if len(p1) else np.nan
        X6_PACE_ROWS.append(dict(dataset=dname, set=set_, model=model, n_s_below_1=len(p1), n_tracks=int(p1["cl"].nunique()) if len(p1) else 0,
                                 median_g=est, ci_lo=lo, ci_hi=hi, share_closer_to_single_image=closer,
                                 median_x_geo=float(p1["gt_x_geo"].median()) if len(p1) else np.nan, n_s_at_least_1=len(p2),
                                 share_placed_farther_than_single_image=float(np.mean(p2["rho"] > p2["rho_mono"])) if len(p2) else np.nan))
        # (3) real driving: the slope beta_geo on parallel vehicles (as H9, exploratory)
        if set_ in ("KT", "AV2"):
            lo_x, hi_x = PREREG["h9_x_range"]
            q = d[d["ident"] & d["gt_x_geo"].between(lo_x, hi_x)]
            if len(q) >= 20 and q["cl"].nunique() >= 5:
                sl = slope_with_ci(q["gt_x_geo"], q["rho"], q["cl"], seed_for("real-beta", set_, model))
                REAL_BETA_ROWS.append(dict(dataset=dname, set=set_, model=model, n=len(q), n_tracks=int(q["cl"].nunique()),
                                           beta_geo=sl["beta"], ci_lo=sl["ci_lo"], ci_hi=sl["ci_hi"], r2=sl["r2"]))
            else:
                REAL_BETA_ROWS.append(dict(dataset=dname, set=set_, model=model, n=len(q), n_tracks=int(q["cl"].nunique()) if len(q) else 0,
                                           beta_geo=np.nan, ci_lo=np.nan, ci_hi=np.nan, r2=np.nan))
X6_CONTACT = pd.DataFrame(X6_CONTACT_ROWS)
X6_CONTACT_TEST = pd.DataFrame(X6_CONTACT_TESTS)
X6_MATCHED = pd.DataFrame(X6_MATCH_ROWS)
X6_PACE = pd.DataFrame(X6_PACE_ROWS)
REAL_BETA = pd.DataFrame(REAL_BETA_ROWS)
_prev = []
for set_, dname, df in REAL_DRIVE:
    _mk = df[df["measurable"].astype(bool)]
    if not len(_mk):
        continue
    _tau = {"AV2": AV2_TAU_PAR, "KT": KT_TAU_PAR}[set_]
    _metas = AV2_PAIRS_ALL if set_ == "AV2" else KT_PAIR_META
    _dts = [(m_.get("baseline"), m_.get("dt_s") or (0.1 * (m_["frames"][1] - m_["frames"][0]) if set_ == "KT" else None))
            for m_ in _metas]                                   # KITTI tracking: 10 Hz frames
    _spd = np.array([b_ / t_ for b_, t_ in _dts if b_ and t_], float)
    _prev.append(dict(dataset=f"{dname} (moving vehicles)", n_logs=int(_mk["seq"].nunique()), n_vehicles=len(_mk),
                      n_pairs=int(_mk["scene"].nunique()),
                      frac_parallel=float((_mk["regime"] == "parallel").mean()),
                      frac_intermediate=float((_mk["regime"] == "intermediate").mean()),
                      frac_perpendicular=float((_mk["regime"] == "perpendicular").mean()),
                      frac_parallel_lidar_px=float(_mk.loc[_mk["regime"] == "parallel", "gt_n_lidar_px"].sum()
                                                   / max(_mk["gt_n_lidar_px"].sum(), 1)),
                      frac_keeping_pace=float(_mk["s"].between(*PREREG["x6_pace_s"]).mean()),
                      parallel_threshold_px=_tau,
                      ego_speed_median_mps=float(np.median(_spd)) if len(_spd) else np.nan,
                      ego_speed_p90_mps=float(np.quantile(_spd, 0.9)) if len(_spd) else np.nan))
REAL_PREVALENCE = pd.DataFrame(_prev)
with pd.option_context("display.width", 250):
    for title, df in (("Ground contact (Virtual KITTI 2; Argoverse 2 from v6.4), median per-object geometry share g:", X6_CONTACT),
                      ("Hidden minus visible contact (clusters: vehicle tracks):", X6_CONTACT_TEST),
                      ("Hidden minus visible contact on matched vehicles (log Z, log area, x_geo within 0.5 SD):", X6_MATCHED),
                      ("The convoy case (vehicles keeping pace with the camera):", X6_PACE),
                      ("Real driving: share of moving vehicles per regime:", REAL_PREVALENCE),
                      ("Real driving: beta_geo on parallel vehicles (exploratory):", REAL_BETA)):
        if len(df):
            print("\n" + title)
            print(df.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
for _n, _df in (("X6_contact", X6_CONTACT), ("X6_contact_test", X6_CONTACT_TEST), ("X6_contact_matched", X6_MATCHED),
                ("X6_pace", X6_PACE), ("real_driving_prevalence", REAL_PREVALENCE), ("real_driving_beta", REAL_BETA)):
    if len(_df):
        _df.to_csv(os.path.join(RESULTS_DIR, f"{_n}.csv"), index=False)
for _k, _n, _st in (("AV2", "Argoverse 2", AV2_STATUS), ("KT", "KITTI tracking", KT_STATUS)):
    if not _st["available"]:
        print(f"\n{_n} analyses are not part of this run: {_st['note']}")

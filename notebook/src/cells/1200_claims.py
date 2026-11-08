# PLACEHOLDER
# 12. Claim audit. Each claim the paper may make is listed with the evidence that decides it and a
#     status computed from this run's results (never written by hand). A claim whose evidence is not
#     part of this run is not assessable: it is left out of the table and named in one line below it,
#     so that nothing is claimed silently.
def _count(verdicts):
    v = list(verdicts)
    keys = ["SUPPORTED", "PARTIAL", "FALSIFIED", "NOT SUPPORTED", "CONTRADICTED", "NOT TESTABLE", "NOT INTERPRETABLE (gate)"]
    return ", ".join(f"{k}: {v.count(k)}" for k in keys if v.count(k)) or "none"

def _status(verdicts, strong="SUPPORTED"):
    v = [x for x in verdicts if x not in ("NOT TESTABLE", "NOT INTERPRETABLE (gate)")]
    if not v:
        return "NOT ASSESSABLE"
    if all(x == strong for x in v):
        return "ESTABLISHED (all interpretable models)"
    if any(x == strong for x in v):
        return "ESTABLISHED FOR A SUBSET"
    if any(x == "PARTIAL" for x in v):
        return "PARTIAL ONLY"
    return "NOT ESTABLISHED"

_e1i = E1_TABLE[E1_TABLE["gate"] == "PASS"]
_low = _e1i["ci_hi"] < PREREG["beta_support_range"][0]
_m2 = E2_TABLE[E2_TABLE["method_key"] == "M2_exact"]["verdict"]
_m1 = E2_TABLE[E2_TABLE["method_key"] == "M1_exact"]["verdict"]
_a1i = A1_TABLE[A1_TABLE["model"].isin(MODELS_OK["MAIN"]) & (A1_TABLE["A1"] != "NOT TESTABLE")] if len(A1_TABLE) else A1_TABLE
_ko = _a1i["A1"].str.startswith("HOLDS") if len(_a1i) else pd.Series(dtype=bool)
_e5 = lambda t: E5_TESTS[E5_TESTS["test"] == t]["verdict"] if len(E5_TESTS) else pd.Series(dtype=str)
_h9 = E6_TABLE[E6_TABLE["verdict"] != "control"]["verdict"] if len(E6_TABLE) else pd.Series(dtype=str)
_h11 = lambda t: RABL_TESTS[RABL_TESTS["test"] == t]["verdict"] if len(RABL_TESTS) else pd.Series(dtype=str)
_m2par = dict(zip(E2_TABLE.loc[E2_TABLE["method_key"] == "M2_exact", "model"], E2_TABLE.loc[E2_TABLE["method_key"] == "M2_exact", "rec_par"]))
_prev_vk = (np.average(PREVALENCE_VK["frac_parallel"], weights=PREVALENCE_VK["n_moving_pixels"]) if len(PREVALENCE_VK) else np.nan)
_prev_tum = (np.average(PREVALENCE["frac_parallel"], weights=PREVALENCE["n_moving_pixels"]) if len(PREVALENCE) else np.nan)
CLAIMS = [
    dict(id="C1", claim="The probe measures what it is meant to: planted prior weights are recovered through the slope-weight identity.",
         evidence="T2b, Fig. 2e", per_model=f"max |beta - beta(w)| = {np.nanmax(np.abs(ORACLE_TABLE['beta_s'] - ORACLE_TABLE['beta_theory_identity'])):.3f}",
         status="ESTABLISHED" if ORACLE_OK else "NOT ESTABLISHED"),
    dict(id="C2", claim="For purely parallel movers every interpretable model uses far less geometry than triangulation would "
                        "(beta_s interval below 0.8; locked P1 not supported).",
         evidence="T3, Fig. 3a", per_model=", ".join(f"{mlab(r.model)} {r.beta_s:.2f}" for r in _e1i.itertuples()) or "none",
         status=("NOT ASSESSABLE" if not len(_e1i) else "ESTABLISHED (all interpretable models)" if _low.all() else
                 "ESTABLISHED FOR A SUBSET" if _low.any() else "NOT ESTABLISHED")),
    dict(id="C3", claim="Whatever geometry is used comes from the cross-view correspondence (deleting the mover from view 2 lowers beta_s).",
         evidence="T7, Fig. 3b", per_model=f"holds for {int(_ko.sum())}/{len(_ko)}",
         status=("NOT ASSESSABLE" if not len(_ko) else "ESTABLISHED (all interpretable models)" if _ko.all() else
                 "ESTABLISHED FOR A SUBSET" if _ko.any() else "NOT ESTABLISHED")),
    dict(id="C4", claim="Removing an object's ground contact raises its geometry share (H5).", evidence="T9b, Fig. 4c",
         per_model=_count(_e5("H5")), status=_status(_e5("H5"))),
    dict(id="C5", claim="Hiding the ground contact raises the geometry share, and the effect acts through the contact (H6, H7).",
         evidence="T9b, Fig. 4c", per_model=f"H6 {_count(_e5('H6'))}; H7 {_count(_e5('H7'))}",
         status=_status(list(_e5("H6")) + list(_e5("H7")))),
    dict(id="C6", claim="Single-image depth places a lifted object farther away (H8): the prior reads the contact point.",
         evidence="T9b, Fig. 4d", per_model=_count(_e5("H8")), status=_status(_e5("H8"))),
    dict(id="C7", claim="In photo-realistic driving the geometry share for moving vehicles is below 0.3 (H9, exploratory: its "
                        "sample rule was changed after the v6.0 count, D34).",
         evidence="T10a, T10d", per_model=_count(_h9),
         status=("EXPLORATORY: " + _status(_h9).lower()) if len(_h9) and _status(_h9) != "NOT ASSESSABLE" else
                "NOT ASSESSABLE (Virtual KITTI 2 not in this run, or too few vehicles)"),
    dict(id="C8", claim="Motion that two views cannot see is common in driving (share of moving vehicle pixels, or of real moving "
                        "vehicles, within the parallel threshold of their epipolar line).",
         evidence="T10b, T10c, T16a, Fig. 7b",
         per_model="; ".join(x for x in [
             f"vehicle pixels {100 * _prev_vk:.0f}% (Virtual KITTI 2)" if np.isfinite(_prev_vk) else "",
             "; ".join(f"real vehicles {100 * r.frac_parallel:.0f}% ({r.dataset.split(' (')[0]})" for r in REAL_PREVALENCE.itertuples())
             if len(REAL_PREVALENCE) else "",
             f"people {100 * _prev_tum:.0f}% of pixels (TUM walking)" if np.isfinite(_prev_tum) else ""] if x) or "none",
         status="DESCRIPTIVE (no prediction)" if np.isfinite(_prev_vk) or np.isfinite(_prev_tum) or len(REAL_PREVALENCE)
         else "NOT ASSESSABLE"),
    dict(id="C9", claim="Geometric masks miss parallel movers far more often than perpendicular ones (P2: M1, M2).",
         evidence="T4, Fig. 6a", per_model=f"M1 {_count(_m1)}; M2 {_count(_m2)}", status=_status(list(_m1) + list(_m2))),
    dict(id="C10a", claim="On real images, lifting the same object off its support raises its geometry share (H11a).",
         evidence="T11a, T11b, Fig. 5b-c", per_model=_count(_h11("H11a")),
         status=_status(_h11("H11a")) if len(_h11("H11a")) else "NOT ASSESSABLE (no real-image contact ablation)"),
    dict(id="C10b", claim="On real images, single-image depth places the lifted object farther away (H11b).",
         evidence="T11b, Fig. 5d", per_model=_count(_h11("H11b")),
         status=_status(_h11("H11b")) if len(_h11("H11b")) else "NOT ASSESSABLE (no real-image contact ablation)"),
    dict(id="C11", claim="Multi-view models keep the degeneracy over 2-8 frames on straight paths and lose it on curved twins (P4).",
         evidence="T6, Fig. 8c", per_model=_count(E4_SUMMARY["verdict"]) if len(E4_SUMMARY) else "none",
         status=_status(E4_SUMMARY["verdict"]) if len(E4_SUMMARY) else "NOT ASSESSABLE"),
    dict(id="C12", claim="Remedy (exploratory): a model's own single-image depth exposes parallel movers that geometric tests miss (M5).",
         evidence="T4 (M5), Fig. S3",
         per_model=", ".join(f"{mlab(r.model)} {r.rec_par:.2f} vs M2 {_m2par.get(r.model, np.nan):.2f}"
                             for r in E2_TABLE[(E2_TABLE["method_key"] == "M5") & E2_TABLE["model"].isin(MODELS_OK["MAIN"])]
                             .itertuples()) or "none",
         status=("EXPLORATORY (no prediction); recall on parallel movers, M5 vs M2"
                 if ((E2_TABLE["method_key"] == "M5") & E2_TABLE["model"].isin(MODELS_OK["MAIN"])).any() else "NOT ASSESSABLE")),
    dict(id="C13", claim="Motion inside the epipolar plane (not only along b) is invisible to two views; the locked A2 fails.",
         evidence="Section 2.2-2.3, Fig. S1",
         per_model=f"in-plane max eps {PROP_CHECKS['P3_max_eps_inplane_perpendicular_px']:.1e} px; "
                   f"locked/exact median {A2_SUMMARY['locked_over_exact_p05_p50_p95'][1]:.2f}",
         status="ESTABLISHED (mathematical, verified numerically)"),
    dict(id="C14", claim="The previous 2-D testbed (LEGACY-FLAT) is rejected by the validity gate; CONVOY-3D is accepted.",
         evidence="T2, Fig. 2d",
         per_model=f"FLAT rejected for {int((GATE[GATE['set'] == 'FLAT']['gate'] == 'FAIL').sum())}/{int((GATE['set'] == 'FLAT').sum())}, "
                   f"CONVOY-3D accepted for {int((GATE[GATE['set'] == 'MAIN']['gate'] == 'PASS').sum())}/{int((GATE['set'] == 'MAIN').sum())}",
         status="ESTABLISHED" if (GATE[GATE["set"] == "FLAT"]["gate"] == "FAIL").all() and (GATE[GATE["set"] == "MAIN"]["gate"] == "PASS").all()
         else "PARTIAL"),
]
_m2drop = E5_MASK_TESTS[E5_MASK_TESTS["contrast"] == "CH - G"]["direction"] if len(E5_MASK_TESTS) else pd.Series(dtype=str)
_cnt = lambda v: ", ".join(f"{k}: {int(n)}" for k, n in pd.Series(list(v)).value_counts().items()) or "none"
_dsm = {"Virtual KITTI 2": "VK", "Argoverse 2": "AV2"}
def _by_ds(df):
    # directions of a driving contrast per dataset, for the models interpretable on that dataset
    out = {}
    for ds, g in (df.groupby("dataset", sort=False) if len(df) else []):
        out[ds] = g.loc[g["model"].isin(MODELS_OK.get(_dsm.get(ds, ""), [])), "direction"]
    return out
_xc_ds, _xm_ds = _by_ds(X6_CONTACT_TEST), _by_ds(X6_MATCHED)
def _c16_status():
    parts = []
    for ds, v in _xc_ds.items():
        t = v[v != "NOT TESTABLE"]
        if not len(t):
            continue
        m = _xm_ds.get(ds, pd.Series(dtype=str)); tm = m[m != "NOT TESTABLE"]
        parts.append(f"{ds}: higher for {int((t == 'higher').sum())}/{len(t)}"
                     + (f", after matching {int((tm == 'higher').sum())}/{len(tm)}" if len(tm) else ", matched comparison not testable"))
    return ("EXPLORATORY: " + "; ".join(parts)) if parts else "NOT ASSESSABLE (too few vehicles with a hidden or a visible contact)"
CLAIMS += [
    dict(id="C15", claim="Exploratory: M2 catches fewer of the same parallel movers once their contact is hidden (masks follow the "
                         "support cue).", evidence="T14, Fig. 6b", per_model=_cnt(_m2drop),
         status=("EXPLORATORY: lower recall for " + f"{int((_m2drop == 'lower').sum())}/{len(_m2drop)} models") if len(_m2drop)
         else "NOT ASSESSABLE"),
    dict(id="C16", claim="Exploratory: in traffic, vehicles whose ground contact is hidden get a larger geometry share (Virtual "
                         "KITTI 2; real Argoverse 2 vehicles from v6.4).",
         evidence="T15a, T15c, T15d, Fig. 10b",
         per_model=" | ".join(f"{ds}: {_cnt(v)}; matched on depth, size and lever: {_cnt(_xm_ds.get(ds, []))}" for ds, v in _xc_ds.items())
         or "none", status=_c16_status()),
    dict(id="C17", claim="Exploratory: vehicles keeping pace with the camera are placed by the prior, not at the geometric answer "
                         "(the convoy case in driving data).", evidence="T15b, Fig. 7c",
         per_model="; ".join(f"{r.dataset}: {mlab(r.model)} g = {r.median_g:.2f}" for r in X6_PACE.itertuples()
                             if np.isfinite(r.median_g)) if len(X6_PACE) else "none",
         status="EXPLORATORY (no prediction)" if len(X6_PACE) and np.isfinite(X6_PACE["median_g"]).any() else "NOT ASSESSABLE"),
    dict(id="C18", claim="Exploratory: on real vehicles with LiDAR depth (Argoverse 2; KITTI tracking if attached) the geometry "
                         "share is small.", evidence="T16b",
         per_model="; ".join(f"{r.dataset}: {mlab(r.model)} {r.beta_geo:.2f}" for r in REAL_BETA.itertuples() if np.isfinite(r.beta_geo))
         if len(REAL_BETA) else "none",
         status="EXPLORATORY (no prediction)" if len(_finite(REAL_BETA, "beta_geo")) else
         "NOT ASSESSABLE (no real-driving data in this run, or too few parallel vehicles)"),
    dict(id="C19", claim="Descriptive: the share of real moving vehicles (and of their LiDAR pixels) in the parallel regime, where "
                         "two views carry no motion signal.", evidence="T16a",
         per_model="; ".join(f"{r.dataset}: {100 * r.frac_parallel:.0f}% of vehicles, {100 * r.frac_parallel_lidar_px:.0f}% of their "
                             f"LiDAR pixels" for r in REAL_PREVALENCE.itertuples()) if len(REAL_PREVALENCE) else "none",
         status="DESCRIPTIVE (no prediction)" if len(REAL_PREVALENCE) else "NOT ASSESSABLE (no real-driving data in this run)"),
]
# ---- v6.4: the synthesis by ground support (descriptive; grouping chosen after the v6.3 results, AMENDMENT_V6_4) ----
_sc = SUPPORT_CONTRASTS[SUPPORT_CONTRASTS["direction"] != "NOT TESTABLE"] if len(SUPPORT_CONTRASTS) else pd.DataFrame(columns=["direction"])
_e5i = SUPPORT_LEVELS[(SUPPORT_LEVELS["testbed"] == "E5 ablation") & SUPPORT_LEVELS["condition"].isin(
    [ABLATION_DESIGN["labels"]["G"], ABLATION_DESIGN["labels"]["CH"]])] if len(SUPPORT_LEVELS) else pd.DataFrame()
_e5f = {}                                       # model -> (factor grounded, factor contact hidden)
for m_ in [m for m in EVALUATED_MODELS if len(_e5i) and (_e5i["model"] == m).any()]:
    g_ = _e5i[_e5i["model"] == m_].set_index("condition")["implied_factor"]
    _e5f[m_] = (float(g_.get(ABLATION_DESIGN["labels"]["G"], np.nan)), float(g_.get(ABLATION_DESIGN["labels"]["CH"], np.nan)))
def _factor_text():
    return ", ".join(f"{mlab(m_)} {a:.2f} -> {b:.2f}" for m_, (a, b) in _e5f.items()) or "none"
_e5f_ok = [(a, b) for a, b in _e5f.values() if np.isfinite(a) and np.isfinite(b)]
def _c21_rows():
    out = []
    for m_ in MODELS_OK["MAIN"]:
        if m_ not in MODELS_OK["REAL"] or not len(SUPPORT_LEVELS):
            continue
        r_ = REAL_E1[REAL_E1["model"] == m_]["beta_s"]
        f_ = SUPPORT_LEVELS[(SUPPORT_LEVELS["model"] == m_) & (SUPPORT_LEVELS["testbed"] == "CONVOY-3D")
                            & (SUPPORT_LEVELS["condition"] == "floating crates")]["estimate"]
        p_ = E1_TABLE[E1_TABLE["model"] == m_]["beta_s"]
        if len(r_) and len(f_) and len(p_) and np.isfinite([r_.iloc[0], f_.iloc[0], p_.iloc[0]]).all():
            out.append((m_, float(r_.iloc[0]), float(f_.iloc[0]), float(p_.iloc[0])))
    return out
_c21 = _c21_rows()
CLAIMS += [
    dict(id="C20", claim="Descriptive: within each testbed, objects whose ground contact is in view get a smaller geometry share "
                         "than floating or hidden-contact objects.",
         evidence="T18a, T18b, Fig. 10a-b",
         per_model="; ".join(f"{r.testbed}, {r.contrast}: higher {r.higher}, no clear change {r.no_clear_change}, lower {r.lower}"
                             for r in SUPPORT_COUNT.itertuples()) if len(SUPPORT_COUNT) else "none",
         status=(f"DESCRIPTIVE (grouping chosen after the v6.3 results): higher in {int((_sc['direction'] == 'higher').sum())} of "
                 f"{len(_sc)} testable model x contrast cells, lower in {int((_sc['direction'] == 'lower').sum())}")
         if len(_sc) else "NOT ASSESSABLE"),
    dict(id="C21", claim="Descriptive: CONVOY-REAL, whose movers are all floating crates (D52), is closer to the floating crates of "
                         "CONVOY-3D than to the pooled CONVOY-3D slope; the two also differ in parallax (D57, C27).",
         evidence="T3, T8, T18a, T21e",
         per_model=", ".join(f"{mlab(m_)} {r_:.2f} vs floating {f_:.2f} / pooled {p_:.2f}" for m_, r_, f_, p_ in _c21) or "none",
         status=(f"DESCRIPTIVE: closer to the floating crates for {sum(abs(r_ - f_) < abs(r_ - p_) for _, r_, f_, p_ in _c21)}/"
                 f"{len(_c21)} models") if _c21 else "NOT ASSESSABLE"),
    dict(id="C22", claim="Derived: for an object keeping pace with the camera (s = 0.85), the E5 slopes imply a larger depth "
                         "overestimate when its ground contact is hidden than when it is in view (exp(beta x); 1 = true depth, "
                         "6.7 = the geometric answer).",
         evidence="T18c, Fig. 10c", per_model="E5 grounded -> contact hidden: " + _factor_text(),
         status=(f"DERIVED from the slopes (not measured separately): larger with the contact hidden for "
                 f"{sum(b > a for a, b in _e5f_ok)}/{len(_e5f_ok)} models; the registered test of the slopes is H6")
         if _e5f_ok else "NOT ASSESSABLE"),
]
# ---- v6.5: designed experiments (registered), the static twin, the cue model and Bonn (exploratory) ----
_h12v = BSW_TESTS[(BSW_TESTS["test"] == "H12") & BSW_TESTS["model"].isin(PREREG["h12_models"])]["verdict"] if len(BSW_TESTS) else pd.Series(dtype=str)
_h12o = BSW_TESTS[(BSW_TESTS["test"] == "H12") & ~BSW_TESTS["model"].isin(PREREG["h12_models"])] if len(BSW_TESTS) else pd.DataFrame()
_h13v = WDR_TESTS[WDR_TESTS["test"] == "H13"]["verdict"] if len(WDR_TESTS) else pd.Series(dtype=str)
_share = WDR_TESTS[(WDR_TESTS["test"] == "dose share") & WDR_TESTS["model"].isin(MODELS_OK["WDR"])] if len(WDR_TESTS) else pd.DataFrame()
_cvd = CUE_CV[CUE_CV["comparison"].str.startswith("contact only")]["direction"] if len(CUE_CV) else pd.Series(dtype=str)
_cvc = CUE_CV[CUE_CV["comparison"].str.startswith("parallax only")]["direction"] if len(CUE_CV) else pd.Series(dtype=str)
_rp = CUE_PRED[(CUE_PRED["set"] == "REAL") & (CUE_PRED["role"] == "predicted")] if len(CUE_PRED) else pd.DataFrame()
_tw_par = TWIN_RECALL[TWIN_RECALL["regime"] == "parallel"] if len(TWIN_RECALL) else pd.DataFrame()
_bonn_d = BONN_TESTS[BONN_TESTS["direction"] != "NOT TESTABLE"]["direction"] if len(BONN_TESTS) else pd.Series(dtype=str)
# C28: both parts must be estimable; the rise must be lower for MonST3R and the H6 contrast not lower
_md = {r.contrast: r.direction for r in MONDUS.itertuples()} if len(MONDUS) else {}
_md_rise, _md_h6 = _md.get("rise of beta_s from x0.5 to x3, baseline sweep"), _md.get("H6 contrast beta(CH) - beta(G), E5 movers")
if _md_rise in (None, "NOT TESTABLE") or _md_h6 in (None, "NOT TESTABLE"):
    _md_status = "NOT ASSESSABLE (the sweep or the E5 contrast is not estimable for both models)"
else:
    _md_status = ("EXPLORATORY: consistent (paired, same movers)" if _md_rise == "lower" and _md_h6 != "lower" else
                  f"EXPLORATORY: not consistent (rise {_md_rise}, contact effect {_md_h6})")
CLAIMS += [
    dict(id="C23", claim="More parallax alone (a longer baseline, everything else fixed) raises the geometry share of DUSt3R and "
                         "MASt3R (H12).", evidence="T19a, T19b, Fig. 13b",
         per_model=_count(_h12v) + ("; no prediction for " + ", ".join(f"{mlab(r.model)} {r.estimate:+.2f}" for r in _h12o.itertuples()
                                                                        if np.isfinite(r.estimate)) if len(_h12o) else ""),
         status=_status(_h12v) if len(_h12v) else "NOT ASSESSABLE"),
    dict(id="C24", claim="Hiding the ground contact of a grounded object with the lowest wall (built to hide at least 15 cm) "
                         "raises its geometry share (H13).",
         evidence="T20a, T20b, Fig. 13d", per_model=_count(_h13v), status=_status(_h13v) if len(_h13v) else "NOT ASSESSABLE"),
    dict(id="C25", claim="Exploratory: most of the effect of a 70 cm wall is already reached by the 15 cm wall (the contact junction, "
                         "not the hidden extent, carries the cue).", evidence="T20b, Fig. 13d",
         per_model=", ".join(f"{mlab(r.model)} {r.estimate:.2f}" for r in _share.itertuples() if np.isfinite(r.estimate)) or "none",
         status=("EXPLORATORY: share of the 70 cm effect reached at 15 cm above 0.5 for "
                 f"{int((_share['estimate'] > 0.5).sum())}/{int(np.isfinite(_share['estimate']).sum())} models")
         if len(_share) and np.isfinite(_share["estimate"]).any() else "NOT ASSESSABLE"),
    dict(id="C26", claim="Exploratory: one cue-combination model with a contact channel and a parallax channel, fitted on CONVOY-3D "
                         "and E5, predicts the slopes of the other testbeds without refitting.",
         evidence="T21a-c, Fig. 12a-c",
         per_model=(f"r = {CUE_CALIBRATION['pearson_r']:.2f}, mean |error| {CUE_CALIBRATION['mean_abs_error']:.3f}, "
                    f"{100 * CUE_CALIBRATION['share_inside_ci']:.0f}% of predictions inside the observed interval; parallax channel "
                    f"better in CV for {int((_cvd == 'two channels predict better').sum())}/{len(_cvd)}, contact channel for "
                    f"{int((_cvc == 'two channels predict better').sum())}/{len(_cvc)}") if CUE_CALIBRATION["n"] else "none",
         status=f"EXPLORATORY ({CUE_CALIBRATION['n']} out-of-sample slopes)" if CUE_CALIBRATION["n"] else "NOT ASSESSABLE"),
    dict(id="C27", claim="Correction (D57): the higher CONVOY-REAL slope is not evidence of a real-image effect; CONVOY-REAL movers "
                         "have far more parallax than the CONVOY-3D floating crates, and the cue model's prediction for that "
                         "parallax is reported next to the observed slope.", evidence="T21c, T21e",
         per_model=", ".join(f"{mlab(r.model)} observed {r.beta_observed:.2f} vs predicted {r.beta_predicted:.2f}" for r in _rp.itertuples())
         or "none",
         status=("DESCRIPTIVE: prediction inside the observed interval for "
                 f"{int(_rp['inside_ci'].sum())}/{len(_rp)} models") if len(_rp) else "NOT ASSESSABLE"),
    dict(id="C28", claim="Exploratory: on the same movers, MonST3R (DUSt3R fine-tuned on dynamic scenes) rises less with parallax "
                         "than DUSt3R in the baseline sweep, and its contact effect (H6 contrast) is not smaller.", evidence="T21d",
         per_model="; ".join(f"{r.contrast}: {r.difference:+.2f} ({r.direction})" for r in MONDUS.itertuples()) if len(MONDUS) else "none",
         status=_md_status),
    dict(id="C29", claim="The static twin of a grounded mover with 0 < s < 1 lies H s/(1-s) below the road, e^x times larger "
                         "(Proposition 4).", evidence="Section 1, Section 2.2, Fig. 11a-b",
         per_model=f"max height error {PROP_CHECKS.get('P4_max_twin_height_error_m', np.nan):.1e} m, below the road in "
                   f"{100 * PROP_CHECKS.get('P4_fraction_twin_below_road_for_0_lt_s_lt_1', np.nan):.0f}% of trials",
         status="ESTABLISHED (mathematical, verified numerically)"),
    dict(id="C30", claim="Exploratory: on real vehicles, a twin test (the static explanation would sink into the road or float above "
                         "it) flags moving vehicles that the epipolar test misses.", evidence="T23a, T23b, Fig. 11c-d",
         per_model=(f"parallel regime: epipolar {100 * _tw_par['det_epipolar'].iloc[0]:.0f}%, twin {100 * _tw_par['det_twin'].iloc[0]:.0f}%, "
                    f"either {100 * _tw_par['det_either'].iloc[0]:.0f}% of {int(_tw_par['n'].iloc[0])} vehicles; the twin test flags "
                    + (f"{100 * _tw_par['twin_given_epi_miss'].iloc[0]:.0f}% of the {int(_tw_par['n_epi_miss'].iloc[0])} the epipolar "
                       "test misses" if _tw_par["n_epi_miss"].iloc[0] else "nothing extra (the epipolar test misses none)"))
         if len(_tw_par) and _tw_par["n"].iloc[0] else "none",
         status="EXPLORATORY (both tests on the same exact correspondences from labelled boxes, at 5% false positives on parked "
                "vehicles)" if len(_tw_par) and _tw_par["n"].iloc[0] else "NOT ASSESSABLE"),
    dict(id="C31", claim="Exploratory: real floating objects (Bonn balloons) get a larger geometry share than objects on the floor.",
         evidence="T22a, T22b, Fig. 14", per_model=_cnt(_bonn_d) if len(_bonn_d) else "none",
         status=(f"EXPLORATORY: higher for {int((_bonn_d == 'higher').sum())}/{len(_bonn_d)} models (v6.6 ground truth: relative "
                 "poses re-estimated, D62)") if len(_bonn_d)
         else ("NOT ASSESSABLE (the Bonn ground truth failed its quality gate: the median pair flags "
               f"{100 * BONN_QC.get('median_flag_frac', np.nan):.0f}% of its static scene as moving)") if BONN_STATUS.get("suspect")
         else "NOT ASSESSABLE (Bonn not in this run, or too few parallel objects on and off the floor)"),
    dict(id="C32", claim="Exploratory: wherever a condition raises the geometry share, M2 recall on the same movers falls (geometric "
                         "masks defeat themselves).", evidence="Fig. 12d",
         per_model=(f"Spearman {TRADEOFF_SUMMARY['spearman']:.2f} over {TRADEOFF_SUMMARY['n']} model x condition changes; opposite "
                    f"signs in {100 * TRADEOFF_SUMMARY['share_opposite_sign']:.0f}%") if TRADEOFF_SUMMARY["n"] else "none",
         status="EXPLORATORY (no prediction)" if TRADEOFF_SUMMARY["n"] >= 5 else "NOT ASSESSABLE"),
]
# ---- v6.6: the contact gates the parallax channel (exploratory) and the twin test with optical flow (registered H14) ----
_h12i = (BSW_TESTS[(BSW_TESTS["test"] == "H12-interaction") & np.isfinite(BSW_TESTS["ci_lo"])] if len(BSW_TESTS)
         else pd.DataFrame())
_ftp = FTWIN_RECALL[FTWIN_RECALL["regime"] == "parallel"] if len(FTWIN_RECALL) else pd.DataFrame()
CLAIMS += [
    dict(id="C33", claim="Exploratory: the contact gates the parallax channel; a longer baseline raises the geometry share of "
                         "floating movers more than that of grounded movers (same scenes, baseline sweep).",
         evidence="T19a, T19b, Fig. 13b",
         per_model="; ".join(f"{mlab(r.model)} {r.estimate:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]" for r in _h12i.itertuples()) or "none",
         status=(f"EXPLORATORY (chosen after the v6.5 results): interval above 0 for {int((_h12i['ci_lo'] > 0).sum())}/{len(_h12i)} "
                 f"models, below 0 for {int((_h12i['ci_hi'] < 0).sum())}") if len(_h12i) else "NOT ASSESSABLE"),
    dict(id="C34", claim="With optical flow instead of labelled correspondences and a LiDAR road plane instead of the box, the twin "
                         "test still flags more parallel-regime moving vehicles than the epipolar test (H14).",
         evidence="T23c, Fig. 11d",
         per_model=(f"parallel regime: epipolar {100 * _ftp['det_epipolar'].iloc[0]:.0f}%, twin {100 * _ftp['det_twin'].iloc[0]:.0f}%, "
                    f"either {100 * _ftp['det_either'].iloc[0]:.0f}% of {int(_ftp['n'].iloc[0])} vehicles; difference "
                    + (f"{H14_RESULT['estimate']:+.3f} [{H14_RESULT['ci_lo']:+.3f}, {H14_RESULT['ci_hi']:+.3f}]"
                       if np.isfinite(H14_RESULT.get("estimate", np.nan)) else "not estimable"))
         if len(_ftp) and _ftp["n"].iloc[0] else "none",
         status=(f"{H14_RESULT['verdict']} (registered in AMENDMENT_V6_6; {FTWIN_THRESH.get('flow', '')})" if H14_RESULT
                 else "NOT ASSESSABLE")),
]
if PROTOCOL_DEPENDENT:
    for c in CLAIMS:
        if c["id"] in ("C6", "C10b"):
            c["per_model"] += " | single-image results protocol-dependent for " + ", ".join(mlab(m) for m in sorted(PROTOCOL_DEPENDENT))
_all_claims = pd.DataFrame(CLAIMS)
_na = _all_claims["status"].str.startswith("NOT ASSESSABLE")
CLAIM_AUDIT = _all_claims[~_na].reset_index(drop=True)
with pd.option_context("display.max_colwidth", 95, "display.width", 250):
    print(CLAIM_AUDIT.to_string(index=False, na_rep="–"))
if _na.any():
    print("\nNot assessable in this run (their evidence is not part of it): "
          + "; ".join(f"{r.id} {r.status.replace('NOT ASSESSABLE', '').strip(' ()') or 'no estimable result'}"
                      for r in _all_claims[_na].itertuples()))
CLAIM_AUDIT.to_csv(os.path.join(RESULTS_DIR, "claim_audit.csv"), index=False)
print("\nScope: findings hold for the models in T1a as evaluated here; CONVOY-3D (mover depth "
      f"{MOVER_DESIGN['depth_range_m'][0]:g}-{MOVER_DESIGN['depth_range_m'][1]:g} m, s in "
      f"{MOVER_DESIGN['s_range_main']} plus an exploratory s in {MOVER_DESIGN['s_range_beyond']}); "
      + (f"CONVOY-REAL (TUM fr3 static backgrounds, mover depth {REAL_PLACEMENT['z_min']:g}-{REAL_PLACEMENT['z_max']:g} m); "
         if len(META_REAL) else "")
      + (f"E5-REAL (the same TUM backgrounds, crate depth {RABL_DESIGN['depth_m'][0]:g}-{RABL_DESIGN['depth_m'][1]:g} m); "
         if len(META_RABL) else "") + ("Virtual KITTI 2 (" + ", ".join(VK_SCENE_OBJS) + ", clone, Camera_0); " if VK_SCENE_OBJS else "")
      + (f"Argoverse 2 ({len(AV2_STATUS['logs'])} validation logs, front-centre camera; exploratory only); " if AV2_STATUS["available"] else "")
      + (f"KITTI tracking ({', '.join(KT_STATUS['sequences'])}; exploratory only); " if KT_STATUS["available"] else "")
      + ("TUM fr3 walking for prevalence only; " if len(PREVALENCE) else "")
      + (f"Bonn RGB-D Dynamic ({', '.join(BONN_STATUS['sequences'])}; exploratory only); " if BONN_STATUS["available"] else "")
      + f"designed sets: baseline sweep x0.5-x3, walls 15-70 cm. No state-of-the-art claim is made.")

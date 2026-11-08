# PLACEHOLDER
# 6.9 Synthesis (descriptive, v6.4, D51): the geometry share by ground support in every testbed, each testbed's
#     contrast between unsupported or hidden-contact objects and grounded ones, and the depth overestimate a slope
#     implies for an object keeping pace with the camera (s = 0.85). Only interpretable model x testbed results enter.
X_IMPLIED = float(-np.log(1.0 - PREREG["implied_factor_s"]))
_PAPER_NAME = {"pi3": "π³", "VGGT-1B": "VGGT"}                 # model names as printed in the paper (also Section 10)
SUPPORT_LEVEL_ROWS, SUPPORT_CONTRAST_ROWS = [], []

def _level(model, testbed, condition, support, stat, est, lo, hi, n, n_clusters, source):
    r = dict(model=model, testbed=testbed, condition=condition, support=support, statistic=stat, estimate=est, ci_lo=lo,
             ci_hi=hi, n=n, n_clusters=n_clusters, source=source, implied_factor=np.nan, implied_lo=np.nan, implied_hi=np.nan)
    if stat == "beta_s":                    # exp(beta x) is increasing in beta, so the interval maps through it
        r.update(implied_factor=float(np.exp(est * X_IMPLIED)) if np.isfinite(est) else np.nan,
                 implied_lo=float(np.exp(lo * X_IMPLIED)) if np.isfinite(lo) else np.nan,
                 implied_hi=float(np.exp(hi * X_IMPLIED)) if np.isfinite(hi) else np.nan)
    SUPPORT_LEVEL_ROWS.append(r)

def _contrast(model, testbed, contrast, stat, est, lo, hi, n, source):
    SUPPORT_CONTRAST_ROWS.append(dict(model=model, testbed=testbed, contrast=contrast, statistic=stat, estimate=est, ci_lo=lo,
                                      ci_hi=hi, n=n, direction=_dir_label(lo, hi) if np.isfinite(est) else "NOT TESTABLE",
                                      source=source))

# CONVOY-3D: E1 movers by support (cars and crates stand on the street, floating crates do not)
for model in MODELS_OK["MAIN"]:
    d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN")]
    dp = d[e1_mask(d)]
    for lab, sel, sup in (("cars and crates", dp["cls"].isin(["car", "crate"]), "grounded"),
                          ("floating crates", dp["cls"] == "crate_floating", "unsupported")):
        r = slope_with_ci(dp.loc[sel, "x_par"], dp.loc[sel, "rho"], dp.loc[sel, "scene"], seed_for("synth-c3d", model, lab))
        _level(model, "CONVOY-3D", lab, sup, "beta_s", r["beta"], r["ci_lo"], r["ci_hi"], r["n"], r["n_clusters"], "E1 movers")
    _, _, est, lo, hi = _group_slope_diff(dp["x_par"], dp["rho"], dp["cls"] == "crate_floating", dp["scene"],
                                          seed_for("synth-c3d-diff", model))
    _contrast(model, "CONVOY-3D", "floating crates - cars and crates", "beta_s", est, lo, hi, len(dp), "new in v6.4 (unpaired)")
# CONVOY-REAL: every inserted mover is a floating crate (D52)
for r in REAL_E1.itertuples():
    if r.verdict != "control" and r.model in MODELS_OK["REAL"]:
        _level(r.model, "CONVOY-REAL", "floating crates (every mover)", "unsupported", "beta_s", r.beta_s, r.ci_lo, r.ci_hi,
               r.n, r.n_pairs, "T8")
# E5: the same movers in five conditions
_E5_SUPPORT = {"G": "grounded", "NS": "grounded", "CH": "contact hidden", "F": "unsupported", "FH": "unsupported"}
for r in (E5_BETA[E5_BETA["model"].isin(MODELS_OK["ABL"])].itertuples() if len(E5_BETA) else []):
    _level(r.model, "E5 ablation", ABLATION_DESIGN["labels"][r.condition], _E5_SUPPORT[r.condition], "beta_s", r.beta,
           r.ci_lo, r.ci_hi, r.n, r.n_scenes, "T9a")
for r in (E5_TESTS[E5_TESTS["test"].isin(["H5", "H6"]) & E5_TESTS["model"].isin(MODELS_OK["ABL"])].itertuples()
          if len(E5_TESTS) else []):
    _contrast(r.model, "E5 ablation", {"H5": "floating - no shadow (H5)", "H6": "contact hidden - grounded (H6)"}[r.test],
              "beta_s", r.estimate, r.ci_lo, r.ci_hi, r.n, "T9b")
# E5-REAL: the same crates on real images
for r in (RABL_BETA[RABL_BETA["model"].isin(MODELS_OK["RABL"])].itertuples() if len(RABL_BETA) else []):
    _level(r.model, "E5-REAL", RABL_DESIGN["labels"][r.condition], {"G": "grounded", "F": "unsupported"}[r.condition], "beta_s",
           r.beta, r.ci_lo, r.ci_hi, r.n, r.n_pairs, "T11a")
for r in (RABL_TESTS[(RABL_TESTS["test"] == "H11a") & RABL_TESTS["model"].isin(MODELS_OK["RABL"])].itertuples()
          if len(RABL_TESTS) else []):
    _contrast(r.model, "E5-REAL", "lifted - grounded (H11a)", "beta_s", r.estimate, r.ci_lo, r.ci_hi, r.n, "T11b")
# driving: per-object share g of parallel vehicles by wheel contact
_DRIVE_TB = {"Virtual KITTI 2": "VK", "Argoverse 2": "AV2"}
_ok_drive = lambda ds, m: m in MODELS_OK.get(_DRIVE_TB.get(ds, ""), [])
for r in (X6_CONTACT.itertuples() if len(X6_CONTACT) else []):
    if _ok_drive(r.dataset, r.model):
        _level(r.model, r.dataset, f"contact {r.contact}", "grounded" if r.contact == "visible" else "contact hidden", "median_g",
               r.median_g, r.ci_lo, r.ci_hi, r.n, r.n_tracks, "T15a")
for r in (X6_CONTACT_TEST.itertuples() if len(X6_CONTACT_TEST) else []):
    if _ok_drive(r.dataset, r.model):
        _contrast(r.model, r.dataset, "contact hidden - visible", "median_g", r.diff_hidden_minus_visible, r.ci_lo, r.ci_hi,
                  r.n_hidden + r.n_visible, "T15a")
for r in (X6_MATCHED.itertuples() if len(X6_MATCHED) else []):
    if _ok_drive(r.dataset, r.model):
        _contrast(r.model, r.dataset, "contact hidden - visible, matched", "median_g", r.median_paired_diff, r.ci_lo, r.ci_hi,
                  r.n_matched, "T15a")

_lcols = ["model", "testbed", "condition", "support", "statistic", "estimate", "ci_lo", "ci_hi", "n", "n_clusters", "source",
          "implied_factor", "implied_lo", "implied_hi"]
SUPPORT_LEVELS = pd.DataFrame(SUPPORT_LEVEL_ROWS, columns=_lcols)
SUPPORT_CONTRASTS = pd.DataFrame(SUPPORT_CONTRAST_ROWS, columns=["model", "testbed", "contrast", "statistic", "estimate", "ci_lo",
                                                                 "ci_hi", "n", "direction", "source"])
_ci3 = lambda e, l, h: "–" if not np.isfinite(e) else (f"{e:.2f}" if not (np.isfinite(l) and np.isfinite(h)) else f"{e:.2f} [{l:.2f}, {h:.2f}]")
if len(SUPPORT_LEVELS):
    _sl = SUPPORT_LEVELS.assign(row=SUPPORT_LEVELS["testbed"] + ": " + SUPPORT_LEVELS["condition"] + " (" + SUPPORT_LEVELS["support"] + ")",
                                value=[_ci3(*v) for v in SUPPORT_LEVELS[["estimate", "ci_lo", "ci_hi"]].to_numpy(float)])
    _order = list(dict.fromkeys(_sl["row"]))
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print("Geometry share by ground support (beta_s on CONVOY-3D, CONVOY-REAL, E5 and E5-REAL; median per-object g in driving):")
        print(_sl.pivot_table(index="row", columns="model", values="value", aggfunc="first").reindex(_order)
              .rename(columns=lambda c: _PAPER_NAME.get(c, c)).to_string(na_rep="–"))
        _im = _sl[_sl["statistic"] == "beta_s"].assign(
            factor=[_ci3(*v) for v in _sl.loc[_sl["statistic"] == "beta_s", ["implied_factor", "implied_lo", "implied_hi"]].to_numpy(float)])
        print(f"\nDepth overestimate implied by each slope for an object keeping pace with the camera (s = {PREREG['implied_factor_s']}, "
              f"exp(beta x) with x = {X_IMPLIED:.2f}; 1 = correct depth, {np.exp(X_IMPLIED):.1f} = the geometric answer):")
        print(_im.pivot_table(index="row", columns="model", values="factor", aggfunc="first").reindex([o for o in _order if o in set(_im["row"])])
              .rename(columns=lambda c: _PAPER_NAME.get(c, c)).to_string(na_rep="–"))
if len(SUPPORT_CONTRASTS):
    _sc = SUPPORT_CONTRASTS.assign(row=SUPPORT_CONTRASTS["testbed"] + ": " + SUPPORT_CONTRASTS["contrast"],
                                   value=[f"{_ci3(e, l, h)} {d_}" if np.isfinite(e) else "–" for e, l, h, d_ in
                                          SUPPORT_CONTRASTS[["estimate", "ci_lo", "ci_hi", "direction"]].itertuples(index=False)])
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print("\nUnsupported or hidden-contact minus grounded, within each testbed (95% cluster-bootstrap intervals; direction from "
              "the interval):")
        print(_sc.pivot_table(index="row", columns="model", values="value", aggfunc="first").reindex(list(dict.fromkeys(_sc["row"])))
              .rename(columns=lambda c: _PAPER_NAME.get(c, c)).to_string(na_rep="–"))
    _t = SUPPORT_CONTRASTS[SUPPORT_CONTRASTS["direction"] != "NOT TESTABLE"]
    SUPPORT_COUNT = _t.groupby(["testbed", "contrast"], sort=False)["direction"].agg(
        higher=lambda v: int((v == "higher").sum()), no_clear_change=lambda v: int((v == "no clear change").sum()),
        lower=lambda v: int((v == "lower").sum())).reset_index()
    print("\nModels per direction, for each contrast:")
    print(SUPPORT_COUNT.to_string(index=False))
else:
    SUPPORT_COUNT = pd.DataFrame()
SUPPORT_LEVELS.to_csv(os.path.join(RESULTS_DIR, "support_levels.csv"), index=False)
SUPPORT_CONTRASTS.to_csv(os.path.join(RESULTS_DIR, "support_contrasts.csv"), index=False)

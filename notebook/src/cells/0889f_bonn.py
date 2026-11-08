# PLACEHOLDER
# 6.12 Bonn RGB-D Dynamic (v6.5, ground truth corrected in v6.6; exploratory, D56, D62): prevalence of the motion regimes
#      of real objects, and the per-object geometry share g of parallel objects by kind and by floor contact. Cluster =
#      image pair. No verdicts; nothing is reported if the ground truth fails its quality gate (Section 3.12).
BONN_PREV_ROWS, BONN_G_ROWS, BONN_TEST_ROWS = [], [], []
if len(MOVERS_BONN):
    _mk = MOVERS_BONN[MOVERS_BONN["measurable"].astype(bool)]
    for kind, g in [("all objects", _mk)] + list(_mk.groupby("cls")):
        BONN_PREV_ROWS.append(dict(kind=kind, n_objects=len(g), n_pairs=int(g["scene"].nunique()),
                                   frac_parallel=float((g["regime"] == "parallel").mean()),
                                   frac_intermediate=float((g["regime"] == "intermediate").mean()),
                                   frac_perpendicular=float((g["regime"] == "perpendicular").mean()),
                                   frac_no_static_solution=float(g["gt_no_static_solution"].astype(bool).mean()),
                                   frac_on_floor=float((g["floor"] == "on the floor").mean()),
                                   frac_off_floor=float((g["floor"] == "off the floor").mean()),
                                   median_parallax_px=float(g["gt_parallax_med"].median()), median_depth_m=float(g["gt_Z_true"].median())))
    for model in ([] if BONN_STATUS.get("suspect") else MODELS_OK.get("BONN", [])):    # v6.6: quality gate of the objects
        d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "BONN")].copy()
        if not len(d):
            continue
        d = d[d["measurable"].astype(bool) & (d["regime"] == "parallel") & (d["gt_parallax_med"] >= PREREG["parallax_min_px"])
              & np.isfinite(d["rho"]) & np.isfinite(d["rho_mono"]) & np.isfinite(d["gt_x_geo"])].copy()
        d["g"] = geo_share_obj(d["rho"], d["rho_mono"], d["gt_x_geo"])
        d = d[np.isfinite(d["g"]) & (np.abs(d["gt_x_geo"] - d["rho_mono"]) >= PREREG["x6_min_lever"])]
        d["cl"] = d["scene"]
        groups = [("kind", k, q) for k, q in d.groupby("cls")] + [("floor contact", f_, q) for f_, q in d.groupby("floor")
                                                                  if f_ != "undetermined"]
        for gtype, gname, q in groups:
            est, lo, hi = _cboot(q, _med_g, seed_for("bonn-g", model, gname)) if len(q) >= 8 else (_med_g(q), np.nan, np.nan)
            BONN_G_ROWS.append(dict(model=model, grouping=gtype, group=gname, n=len(q), n_pairs=int(q["cl"].nunique()),
                                    median_g=est, ci_lo=lo, ci_hi=hi, median_x_geo=float(q["gt_x_geo"].median()) if len(q) else np.nan))
        on, off = d[d["floor"] == "on the floor"], d[d["floor"] == "off the floor"]
        if len(on) >= 8 and len(off) >= 8:
            fn = lambda q: (float(np.median(q.loc[q["floor"] == "off the floor", "g"]) - np.median(q.loc[q["floor"] == "on the floor", "g"]))
                            if (q["floor"] == "off the floor").any() and (q["floor"] == "on the floor").any() else np.nan)
            est, lo, hi = _cboot(d[d["floor"].isin(["on the floor", "off the floor"])], fn, seed_for("bonn-diff", model))
        else:
            est = lo = hi = np.nan
        BONN_TEST_ROWS.append(dict(model=model, contrast="off the floor minus on the floor (median g)", n_off=len(off), n_on=len(on),
                                   estimate=est, ci_lo=lo, ci_hi=hi, direction=_dir_label(lo, hi) if np.isfinite(est) else "NOT TESTABLE"))
BONN_PREV = pd.DataFrame(BONN_PREV_ROWS)
BONN_G = pd.DataFrame(BONN_G_ROWS)
BONN_TESTS = pd.DataFrame(BONN_TEST_ROWS)
if len(BONN_PREV):
    with pd.option_context("display.width", 250):
        print("Bonn RGB-D Dynamic: motion regimes of real objects (parallel: |eps| below the flow-noise floor "
              f"{BONN_TAU_PAR:.2f} px):")
        print(BONN_PREV.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if len(BONN_G):
            print("\nMedian per-object geometry share g of parallel, identifiable objects (|x - rho_mono| >= 0.3), by kind and floor contact:")
            print(BONN_G.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
            print("\nObjects off the floor minus objects on it:")
            print(BONN_TESTS.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
        elif BONN_STATUS.get("suspect"):
            print("\nThe Bonn objects fail the quality gate (much of the static scene is flagged as moving), so no geometry "
                  "share is reported.")
        else:
            print("\nNo model has enough parallel Bonn objects for the geometry share in this run.")
else:
    print("Bonn RGB-D Dynamic is not part of this run:", BONN_STATUS["note"] or "no data")
for _n, _df in (("bonn_prevalence", BONN_PREV), ("bonn_geometry_share", BONN_G), ("bonn_floor_contrast", BONN_TESTS)):
    _df.to_csv(os.path.join(RESULTS_DIR, f"{_n}.csv"), index=False)

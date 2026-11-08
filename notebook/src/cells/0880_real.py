# PLACEHOLDER
# 6.6 Replication on real images (CONVOY-REAL: TUM backgrounds, motion-capture poses,
#     inserted movers with exact motion). Same protocol as E1 and E2.
REAL_E1_ROWS, REAL_E2_ROWS = [], []
_rb = BASE_RES[(BASE_RES["set"] == "REAL")] if len(BASE_RES) else pd.DataFrame()
if len(_rb):
    _rbp = _rb[e1_mask(_rb)]
    for lab, col in [("Classical triangulation, exact correspondences", "rho_classical_exact"),
                     (f"Classical triangulation, {FLOW_NAME} flow", "rho_classical_flow")]:
        r = slope_with_ci(_rbp["x_par"], _rbp[col], _rbp["scene"], seed_for("real-c", col))
        REAL_E1_ROWS.append(dict(model=lab, gate="-", n=r["n"], n_pairs=r["n_clusters"], beta_s=r["beta"],
                                 ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], r2=r["r2"], beta_mono=np.nan, verdict="control"))
    for meth in ["M1_exact", "M1_flow"]:
        r = stratified_recall_difference(_rb[_rb["measurable"].astype(bool)], f"recall_{meth}", seed_for("real-e2", meth))
        REAL_E2_ROWS.append(dict(model="(model-independent)", method=METHOD_LABELS[meth], **r, verdict=verdict_p2(r)))
for model in EVALUATED_MODELS:
    d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "REAL")]
    if not len(d):
        continue
    dp = d[e1_mask(d)]
    r = slope_with_ci(dp["x_par"], dp["rho"], dp["scene"], seed_for("real-e1", model), h0=0.5)
    rm = slope_with_ci(dp["x_par"], dp["rho_mono"], dp["scene"], seed_for("real-e1m", model))
    gate_ok = GATE_PASS.get((model, "REAL"), False)
    REAL_E1_ROWS.append(dict(model=model, gate="PASS" if gate_ok else "FAIL", n=r["n"], n_pairs=r["n_clusters"],
                             beta_s=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], r2=r["r2"], beta_mono=rm["beta"],
                             verdict=verdict_p1(r["beta"], r["ci_lo"], r["ci_hi"], r["r2"]) if gate_ok else "NOT INTERPRETABLE (gate)"))
    dm = d[d["measurable"].astype(bool)]
    for meth in ["M2_exact", "M2_flow", "M4", "M5"]:
        if f"recall_{meth}" not in dm or dm[f"recall_{meth}"].notna().sum() == 0:
            continue
        r = stratified_recall_difference(dm, f"recall_{meth}", seed_for("real-e2", model, meth))
        REAL_E2_ROWS.append(dict(model=model, method=METHOD_LABELS[meth], **r,
                                 verdict=("NOT INTERPRETABLE (gate)" if not gate_ok else
                                          "EXPLORATORY" if meth == "M5" else verdict_p2(r))))
REAL_E1 = pd.DataFrame(REAL_E1_ROWS, columns=["model", "gate", "n", "n_pairs", "beta_s", "ci_lo", "ci_hi", "r2", "beta_mono", "verdict"])
REAL_E2 = pd.DataFrame(REAL_E2_ROWS, columns=["model", "method", "diff", "ci_lo", "ci_hi", "rec_par", "rec_perp", "n_par", "n_perp", "verdict"])
print("CONVOY-REAL, E1 replication (purely parallel movers):")
print(REAL_E1.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
print("\nCONVOY-REAL, E2 replication:")
if len(REAL_E2):
    print(REAL_E2[["model", "method", "n_par", "n_perp", "rec_par", "rec_perp", "diff", "ci_lo", "ci_hi", "verdict"]]
          .to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
REAL_E1.to_csv(os.path.join(RESULTS_DIR, "real_E1.csv"), index=False)
REAL_E2.to_csv(os.path.join(RESULTS_DIR, "real_E2.csv"), index=False)
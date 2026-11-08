# PLACEHOLDER
# 6.5 A1 correspondence knockout (causal test of assumption A1). The second view is
#     re-rendered with every mover deleted, so the mover has no correspondence. If the
#     two-view depth of a parallel mover comes from cross-view correspondence, beta_s must
#     drop towards the single-image value; if it comes from monocular cues, nothing changes.
#     Pre-registered rule: A1 holds for a model if the paired-bootstrap CI of
#     beta_s(full) - beta_s(knockout) lies above 0.
A1_ROWS = []
for model in EVALUATED_MODELS:
    d = E1_DATA[model]
    d = d[np.isfinite(d["rho_ko"])]
    full = slope_boot(d["x_par"], d["rho"], d["scene"], seed_for("a1f", model))
    ko = slope_boot(d["x_par"], d["rho_ko"], d["scene"], seed_for("a1k", model))
    mono = slope_boot(d["x_par"], d["rho_mono"], d["scene"], seed_for("a1m", model))
    nb = min(len(full[1]), len(ko[1]))
    # paired bootstrap of the drop (same clips in both conditions): reuse the same seed
    T, _ = ClusterBootstrap(d["scene"].values, seed=seed_for("a1d", model)).linear(
        [np.ones(len(d)), d["x_par"].values, d["rho"].values, d["x_par"].values ** 2, (d["x_par"] * d["rho"]).values,
         d["rho_ko"].values, (d["x_par"] * d["rho_ko"]).values]) if len(d) >= 8 else (np.zeros((0, 7)), None)
    if len(T):
        with np.errstate(invalid="ignore", divide="ignore"):
            den = T[:, 0] * T[:, 3] - T[:, 1] ** 2
            bfull = (T[:, 0] * T[:, 4] - T[:, 1] * T[:, 2]) / den
            bko = (T[:, 0] * T[:, 6] - T[:, 1] * T[:, 5]) / den
        drop = bfull - bko
        dlo, dhi = np.nanpercentile(drop, [2.5, 97.5])
    else:
        dlo = dhi = np.nan
    ci = lambda bb: np.nanpercentile(bb, [2.5, 97.5]) if len(bb) > 50 else (np.nan, np.nan)
    flo, fhi = ci(full[1]); klo, khi = ci(ko[1]); mlo, mhi = ci(mono[1])
    A1_ROWS.append(dict(model=model, n=len(d), n_clips=int(d["scene"].nunique()),
                        beta_full=full[0], full_lo=flo, full_hi=fhi, beta_knockout=ko[0], ko_lo=klo, ko_hi=khi,
                        beta_mono=mono[0], mono_lo=mlo, mono_hi=mhi, drop=full[0] - ko[0], drop_lo=dlo, drop_hi=dhi,
                        median_abs_rho_change=float(np.median(np.abs(d["rho"] - d["rho_ko"]))) if len(d) else np.nan,
                        static_sensitivity=float(np.median(np.abs(d.loc[d["s"].abs() < 0.15, "rho"] - d.loc[d["s"].abs() < 0.15, "rho_ko"])))
                        if (d["s"].abs() < 0.15).sum() >= 3 else np.nan,
                        A1=("NOT INTERPRETABLE (gate)" if not interpretable(model, "MAIN")
                            else "HOLDS (deleting the correspondence lowers beta_s)" if np.isfinite(dlo) and dlo > 0
                            else "NOT DETECTED (depth does not depend on the correspondence)" if np.isfinite(dlo)
                            else "NOT TESTABLE")))
A1_TABLE = pd.DataFrame(A1_ROWS)
print(A1_TABLE.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
A1_TABLE.to_csv(os.path.join(RESULTS_DIR, "A1_knockout.csv"), index=False)
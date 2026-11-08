# PLACEHOLDER
# 6.4 E4 / Prediction 4: does the degeneracy persist over 2-8 frames on a straight camera
#     path, and break when the translation path bends? Multi-view models only. Each curved
#     clip is the twin of a straight clip (same street, movers, frames 0-1 and rotations), so
#     the straight-minus-curved difference is a paired comparison (cluster = twin).
E4_ROWS, E4_SUMMARY = [], []
for model in [m for m in EVALUATED_MODELS if MODEL_FAMILY[m] == "multiview"]:
    d = E4_RES[(E4_RES["model"] == model) & E4_RES["measurable"].astype(bool) & E4_RES["identifiable"].astype(bool)
               & (E4_RES["s"] < 1)].copy()
    d["twin"] = d["scene"] % N_E4
    res, n_k = {}, {}
    for K in RUN_CONFIG["e4_window_sizes"]:
        st = d[(d["path"] == "straight") & (d["K"] == K)][["twin", "mover", "x_par", "rho"]]
        cv = d[(d["path"] == "curved") & (d["K"] == K)][["twin", "mover", "rho", "gtlast_eps_med"]]
        m_ = st.merge(cv, on=["twin", "mover"], suffixes=("_st", "_cv"))
        b_st, b_cv, diff_b = paired_slopes(m_["x_par"], m_["rho_st"], m_["rho_cv"], m_["twin"], seed_for("e4", model, K))
        lo_st, hi_st = _ci(slope_boot(m_["x_par"], m_["rho_st"], m_["twin"], seed_for("e4s", model, K))[1])
        lo_cv, hi_cv = _ci(slope_boot(m_["x_par"], m_["rho_cv"], m_["twin"], seed_for("e4c", model, K))[1])
        dlo, dhi = _ci(diff_b)
        n_k[K] = len(m_)
        res[K] = dict(b_st=b_st, lo_st=lo_st, hi_st=hi_st, b_cv=b_cv, lo_cv=lo_cv, hi_cv=hi_cv,
                      diff=b_st - b_cv, dlo=dlo, dhi=dhi)
        for path, b, lo, hi in (("straight", b_st, lo_st, hi_st), ("curved", b_cv, lo_cv, hi_cv)):
            E4_ROWS.append(dict(model=model, path=path, K=K, n=len(m_), n_twins=int(m_["twin"].nunique()),
                                beta_s=b, ci_lo=lo, ci_hi=hi,
                                eps_first_last_median=float(m_["gtlast_eps_med"].median()) if path == "curved" and len(m_) else 0.0))
    Kmax = max(RUN_CONFIG["e4_window_sizes"])
    straight_in = all(PREREG["beta_support_range"][0] <= res[K]["b_st"] <= PREREG["beta_support_range"][1]
                      for K in RUN_CONFIG["e4_window_sizes"])
    straight_fail = any(res[K]["hi_st"] < PREREG["beta_ci_band"][0] or res[K]["lo_st"] > PREREG["beta_ci_band"][1]
                        for K in RUN_CONFIG["e4_window_sizes"])
    r8 = res[Kmax]
    v = ("NOT INTERPRETABLE (gate)" if not interpretable(model, "MAIN") else "NOT TESTABLE" if not np.isfinite(r8["b_st"]) else
         "FALSIFIED" if straight_fail else
         "SUPPORTED" if straight_in and r8["diff"] >= PREREG["e4_drop_min"] and r8["dlo"] > 0 else "PARTIAL")
    E4_SUMMARY.append(dict(model=model, **{f"straight_beta_K{K}": res[K]["b_st"] for K in RUN_CONFIG["e4_window_sizes"]},
                           **{f"curved_beta_K{K}": res[K]["b_cv"] for K in RUN_CONFIG["e4_window_sizes"]},
                           drop_K8=r8["diff"], drop_lo=r8["dlo"], drop_hi=r8["dhi"], n_K8=n_k[Kmax], verdict=v))
E4_TABLE = pd.DataFrame(E4_ROWS); E4_SUMMARY = pd.DataFrame(E4_SUMMARY)
# built-in check: at K = 2 a straight clip and its curved twin are the same two images
_k2 = E4_RES[E4_RES["K"] == 2].copy()
if len(_k2):
    _k2["twin"] = _k2["scene"] % N_E4
    _pv = _k2.pivot_table(index=["model", "twin", "mover"], columns="path", values="rho")
    if {"straight", "curved"} <= set(_pv.columns):
        print(f"Twin check (K = 2, identical inputs): max |rho_straight - rho_curved| = "
              f"{np.nanmax(np.abs(_pv['straight'] - _pv['curved'])):.2e}")
print(E4_TABLE.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
print()
print(E4_SUMMARY.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
E4_TABLE.to_csv(os.path.join(RESULTS_DIR, "E4_multiview.csv"), index=False)
E4_SUMMARY.to_csv(os.path.join(RESULTS_DIR, "E4_summary.csv"), index=False)
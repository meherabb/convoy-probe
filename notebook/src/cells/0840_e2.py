# PLACEHOLDER
# 6.2 E2 / Prediction 2: do dynamic masks miss parallel motion?
#     Recall at a 5% static false-positive rate, parallel (|eps| < 1 px) vs perpendicular
#     (eps >= 3 px), stratified by |s| tertile x mover-area tertile (matched |s| and size),
#     cluster bootstrap over clips. The attention-based mask of the locked list (Easi3R) is not
#     evaluated: its 3-frame version (v5 M3) and its 10-frame version (v6.0 E7) are withdrawn (D23, D33).
METHOD_LABELS = {"M1_exact": "M1 epipolar residual (exact corr.)", "M1_flow": "M1 epipolar residual (flow)",
                 "M2_exact": "M2 rigid-flow residual (exact corr.)", "M2_flow": "M2 rigid-flow residual (flow)",
                 "M4": "M4 model confidence", "M5": "M5 two-view vs single-image depth (exploratory remedy)"}
MODEL_METHODS = ["M2_exact", "M2_flow", "M4", "M5"]

def stratified_recall_difference(df, recall_col, seed):
    d = df[df["regime"].isin(["parallel", "perpendicular"]) & np.isfinite(df[recall_col])].copy()
    if d["regime"].nunique() < 2 or len(d) < 10:
        return dict(diff=np.nan, ci_lo=np.nan, ci_hi=np.nan, rec_par=np.nan, rec_perp=np.nan, n_par=0, n_perp=0)
    d["s_bin"] = pd.qcut(d["s"].abs().rank(method="first"), 3, labels=False)
    d["a_bin"] = pd.qcut(d["gt_area_px"].rank(method="first"), 3, labels=False)
    d["stratum"] = d["s_bin"] * 3 + d["a_bin"]
    cols = []
    for k in range(9):
        ink = (d["stratum"] == k).values
        par = ink & (d["regime"] == "parallel").values; perp = ink & (d["regime"] == "perpendicular").values
        rv = d[recall_col].values
        cols += [par.astype(float), rv * par, perp.astype(float), rv * perp]
    bs = ClusterBootstrap(d["scene"].values, seed=seed)
    T, P = bs.linear(cols)
    def _diff(t):
        t = t.reshape(-1, 9, 4)
        n_par, s_par, n_perp, s_perp = t[..., 0], t[..., 1], t[..., 2], t[..., 3]
        ok = (n_par > 0) & (n_perp > 0)
        w = np.where(ok, n_par + n_perp, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dk = np.where(ok, s_perp / np.maximum(n_perp, 1) - s_par / np.maximum(n_par, 1), 0.0)
            return (w * dk).sum(-1) / w.sum(-1)
    point = float(_diff(P[None])[0]); boots = _diff(T); boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(diff=point, ci_lo=float(lo), ci_hi=float(hi),
                rec_par=float(d.loc[d["regime"] == "parallel", recall_col].mean()),
                rec_perp=float(d.loc[d["regime"] == "perpendicular", recall_col].mean()),
                n_par=int((d["regime"] == "parallel").sum()), n_perp=int((d["regime"] == "perpendicular").sum()))

def verdict_p2(r):
    if not np.isfinite(r["diff"]):
        return "NOT TESTABLE"
    if r["ci_lo"] <= 0 <= r["ci_hi"] or r["ci_hi"] < 0:
        return "FALSIFIED"
    return "SUPPORTED" if r["diff"] >= PREREG["recall_diff_min"] else "PARTIAL"

E2_ROWS = []
_bm = BASE_RES[(BASE_RES["set"] == "MAIN") & BASE_RES["measurable"].astype(bool)]
for meth in ["M1_exact", "M1_flow"]:
    r = stratified_recall_difference(_bm, f"recall_{meth}", seed_for("e2", meth))
    E2_ROWS.append(dict(model="(model-independent)", method=METHOD_LABELS[meth], method_key=meth, **r,
                        auroc_par=float(_bm.loc[_bm["regime"] == "parallel", f"auroc_{meth}"].mean()),
                        auroc_perp=float(_bm.loc[_bm["regime"] == "perpendicular", f"auroc_{meth}"].mean())))
for model in EVALUATED_MODELS:
    d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN") & MOV_RES["measurable"].astype(bool)]
    for meth in MODEL_METHODS:
        if f"recall_{meth}" not in d or d[f"recall_{meth}"].notna().sum() == 0:
            continue
        r = stratified_recall_difference(d, f"recall_{meth}", seed_for("e2", model, meth))
        E2_ROWS.append(dict(model=model, method=METHOD_LABELS[meth], method_key=meth, **r,
                            auroc_par=float(d.loc[d["regime"] == "parallel", f"auroc_{meth}"].mean()),
                            auroc_perp=float(d.loc[d["regime"] == "perpendicular", f"auroc_{meth}"].mean())))
E2_TABLE = pd.DataFrame(E2_ROWS)
E2_TABLE["verdict"] = [("NOT INTERPRETABLE (gate)" if r["model"] != "(model-independent)" and not interpretable(r["model"], "MAIN")
                        else "EXPLORATORY" if r["method_key"] == "M5" else verdict_p2(r)) for r in E2_ROWS]
print(E2_TABLE[["model", "method", "n_par", "n_perp", "rec_par", "rec_perp", "diff", "ci_lo", "ci_hi",
                "auroc_par", "auroc_perp", "verdict"]].to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
E2_TABLE.to_csv(os.path.join(RESULTS_DIR, "E2_masks.csv"), index=False)

# recall as a function of the exact epipolar offset (for Figure 4)
EPS_BINS = [0, 0.25, 1, 2, 3, 6, 10, 1e9]
_rows = []
for model in ["(model-independent)"] + EVALUATED_MODELS:
    if model == "(model-independent)":
        src, meths = _bm, ["M1_exact", "M1_flow"]
    else:
        src = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN") & MOV_RES["measurable"].astype(bool)]
        meths = [m for m in MODEL_METHODS if f"recall_{m}" in src and src[f"recall_{m}"].notna().any()]
    src = src.assign(eps_bin=pd.cut(src["gt_eps_med"], EPS_BINS, right=False, labels=False))
    for meth in meths:
        for b, g in src.groupby("eps_bin"):
            _rows.append(dict(model=model, method_key=meth, eps_bin=int(b), eps_lo=EPS_BINS[int(b)],
                              eps_hi=EPS_BINS[int(b) + 1], n=len(g), recall=g[f"recall_{meth}"].mean(),
                              auroc=g[f"auroc_{meth}"].mean()))
E2_BY_EPS = pd.DataFrame(_rows)
E2_BY_EPS.to_csv(os.path.join(RESULTS_DIR, "E2_recall_by_eps.csv"), index=False)

# M5 (exploratory): recall on PARALLEL movers as a function of |x| = |log(1/(1-s))|, the depth
# discrepancy that the two-view prediction inherits and the single-image prediction does not
_rows = []
for model in EVALUATED_MODELS:
    d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN") & MOV_RES["measurable"].astype(bool)
                & (MOV_RES["motion_type"] == "parallel") & (MOV_RES["s"] < 1)]
    if "recall_M5" not in d or not len(d):
        continue
    for lo_, hi_ in [(0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 3.0)]:
        g = d[(d["x_par"].abs() >= lo_) & (d["x_par"].abs() < hi_)]
        _rows.append(dict(model=model, abs_x_lo=lo_, abs_x_hi=hi_, n=len(g), recall_M5=g["recall_M5"].mean(),
                          recall_M2_exact=g["recall_M2_exact"].mean()))
E2_M5_BY_X = pd.DataFrame(_rows)
if len(E2_M5_BY_X):
    print("\nM5 (exploratory remedy) vs M2 on parallel movers, by |x| = |-log(1-s)|:")
    print(E2_M5_BY_X.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
    E2_M5_BY_X.to_csv(os.path.join(RESULTS_DIR, "E2_M5_by_x.csv"), index=False)
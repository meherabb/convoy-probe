# PLACEHOLDER
# 6.3 E3 / Prediction 3: prior takeover. Prior weight per mover,
#     Zhat = w * Zhat_mono + (1 - w) * Z_geo  =>  w = (e^rho - e^x_geo) / (e^rho_mono - e^x_geo),
#     computed where the hypotheses are separable (|x_geo - rho_mono| > 0.10).
#     P3a: Spearman(w, |eps|) > 0.5.  P3b: at |eps| < 1 px, w(car) > w(crate).
def _spearman(c):
    if len(c["w"]) < 8:
        return np.nan
    return float(spearmanr(c["gt_eps_med"], c["w"]).correlation)

def _class_diff(c):
    a = c["w"][c["cls"] == "car"]; b = c["w"][c["cls"] != "car"]
    return float(np.median(a) - np.median(b)) if len(a) >= 3 and len(b) >= 3 else np.nan

E3_ROWS, E3_W = [], {}
for model in EVALUATED_MODELS:
    d = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "MAIN") & MOV_RES["measurable"].astype(bool)
                & (MOV_RES["s"] < 1) & np.isfinite(MOV_RES["gt_x_geo"])].copy()
    d["w"] = prior_weight(d)
    d = d[np.isfinite(d["w"])]
    E3_W[model] = d
    rs, rs_lo, rs_hi, _ = boot_stat(d, "scene", _spearman, seed_for("e3s", model), cols=["gt_eps_med", "w"])
    dp = d[d["regime"] == "parallel"]
    cd, cd_lo, cd_hi, _ = boot_stat(dp, "scene", _class_diff, seed_for("e3c", model), cols=["cls", "w"])
    med = {c: float(dp.loc[dp["cls"] == c, "w"].median()) if (dp["cls"] == c).sum() else np.nan
           for c in ["car", "crate", "crate_floating"]}
    v3a = ("SUPPORTED" if rs > PREREG["spearman_min"] and rs_lo > 0 else
           "FALSIFIED" if not (rs_lo > 0) else "PARTIAL") if np.isfinite(rs_lo) else "NOT TESTABLE"
    v3b = ("SUPPORTED" if cd_lo > 0 else "NOT SUPPORTED") if np.isfinite(cd_lo) else "NOT TESTABLE"
    if not interpretable(model, "MAIN"):
        v3a = v3b = "NOT INTERPRETABLE (gate)"
    dynamic_trained = MODEL_TRAINING[model].startswith("dynamic")
    lock_c = None
    if dynamic_trained and np.isfinite(cd_lo) and interpretable(model, "MAIN"):
        beats_all = all(np.isfinite(v) and v > 0.5 for v in med.values())
        lock_c = "FALSIFIED (c)" if beats_all and cd_lo <= 0 <= cd_hi else "not triggered"
    E3_ROWS.append(dict(model=model, training=MODEL_TRAINING[model], n_w=len(d), n_parallel=len(dp),
                        median_w_parallel=float(dp["w"].median()) if len(dp) else np.nan,
                        median_w_perp=float(d.loc[d["regime"] == "perpendicular", "w"].median()),
                        spearman_w_eps=rs, sp_lo=rs_lo, sp_hi=rs_hi, verdict_P3a=v3a,
                        w_car=med["car"], w_crate=med["crate"], w_crate_floating=med["crate_floating"],
                        class_diff=cd, cd_lo=cd_lo, cd_hi=cd_hi, verdict_P3b=v3b,
                        lock_condition_c=lock_c or ("n/a (not dynamic-trained)" if not dynamic_trained else
                                                    "n/a (gate)" if not interpretable(model, "MAIN") else "n/a (not testable)")))
E3_TABLE = pd.DataFrame(E3_ROWS)
print(E3_TABLE.drop(columns=["training"]).to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
E3_TABLE.to_csv(os.path.join(RESULTS_DIR, "E3_prior_weight.csv"), index=False)
_rows = []
for model, d in E3_W.items():
    d = d.assign(eps_bin=pd.cut(d["gt_eps_med"], EPS_BINS, right=False, labels=False))
    for b, g in d.groupby("eps_bin"):
        _rows.append(dict(model=model, eps_bin=int(b), eps_lo=EPS_BINS[int(b)], eps_hi=EPS_BINS[int(b) + 1], n=len(g),
                          w_median=g["w"].median(), w_q25=g["w"].quantile(0.25), w_q75=g["w"].quantile(0.75)))
E3_BY_EPS = pd.DataFrame(_rows)
E3_BY_EPS.to_csv(os.path.join(RESULTS_DIR, "E3_w_by_eps.csv"), index=False)
# PLACEHOLDER
# 6.7d Single-image protocol check (v6.2). The single-image reference of a multi-view model is the
#      model given one image. pi3 contradicted H11b in v6.1 with a very wide interval, so before any
#      single-image claim is made the same frames are also given as the image twice (the protocol
#      used for the CroCo models). For every multi-view model: H11b under both protocols on the same
#      crates, their per-crate agreement, the static-scene quality of both single-image depths, and
#      the CONVOY-REAL single-image slope under both. A model whose single-image results depend on
#      the protocol is reported as such and its single-image claims are not made.
PROTO_ROWS = []
_mv_models = [m for m in EVALUATED_MODELS if MODEL_FAMILY.get(m) == "multiview"]
for model in _mv_models:
    rec = dict(model=model)
    w = RABL_WIDE.get(model)
    d = RABL_RES[RABL_RES["model"] == model] if len(RABL_RES) else pd.DataFrame()
    if w is not None and len(d) and "rho_mono_dup" in d:
        dup = d.pivot_table(index="uid", columns="condition", values="rho_mono_dup", aggfunc="first")
        dup.columns = [f"rho_mono_dup_{c}" for c in dup.columns]
        s8 = w[w["ok_G"] & w["ok_F"] & w["contact_ok"]].merge(dup.reset_index(), on="uid", how="left")
        s8 = s8[np.isfinite(s8["rho_mono_G"]) & np.isfinite(s8["rho_mono_F"]) & np.isfinite(s8["rho_mono_dup_G"])
                & np.isfinite(s8["rho_mono_dup_F"])]
        if len(s8) >= 8 and s8["scene"].nunique() >= 5:
            for tag, cg, cf in (("one image", "rho_mono_G", "rho_mono_F"), ("image twice", "rho_mono_dup_G", "rho_mono_dup_F")):
                est, lo, hi, _ = boot_stat(s8, "scene", lambda q, cg=cg, cf=cf: float(np.median(q[cf]) - np.median(q[cg])),
                                           seed_for("proto", model, tag), cols=[cg, cf])
                key = "one" if tag == "one image" else "twice"
                rec.update({f"H11b_{key}": est, f"H11b_{key}_lo": lo, f"H11b_{key}_hi": hi})
            both = np.r_[s8["rho_mono_G"] - s8["rho_mono_dup_G"], s8["rho_mono_F"] - s8["rho_mono_dup_F"]]
            rec.update(n_crates=len(s8), median_abs_diff=float(np.median(np.abs(both))),
                       spearman=float(pd.Series(np.r_[s8["rho_mono_G"], s8["rho_mono_F"]]).corr(
                           pd.Series(np.r_[s8["rho_mono_dup_G"], s8["rho_mono_dup_F"]]), method="spearman")))
        im = IMG_RES[(IMG_RES["model"] == model) & (IMG_RES["set"] == "RABL")]
        if "mono_dup_absrel" in im:
            rec.update(static_absrel_one=float(im["mono_absrel"].median()), static_absrel_twice=float(im["mono_dup_absrel"].median()))
    rr = MOV_RES[(MOV_RES["model"] == model) & (MOV_RES["set"] == "REAL") & (MOV_RES["motion_type"] == "parallel")
                 & MOV_RES["measurable"].astype(bool) & MOV_RES["identifiable"].astype(bool)] if len(MOV_RES) else pd.DataFrame()
    if len(rr) and "rho_mono_dup" in rr:
        for key, col in (("one", "rho_mono"), ("twice", "rho_mono_dup")):
            sl = slope_with_ci(rr["x_par"], rr[col], rr["scene"], seed_for("proto-real", model, key))
            rec.update({f"real_mono_slope_{key}": sl["beta"], f"real_mono_r2_{key}": sl.get("r2", np.nan)})
    if "H11b_one" in rec:
        same_sign = np.sign(rec["H11b_one"]) == np.sign(rec["H11b_twice"])
        overlap = rec["H11b_one_lo"] <= rec["H11b_twice_hi"] and rec["H11b_twice_lo"] <= rec["H11b_one_hi"]
        rec["protocol"] = "consistent" if (same_sign and overlap) else "PROTOCOL-DEPENDENT"
    PROTO_ROWS.append(rec)
PROTOCOL_CHECK = pd.DataFrame(PROTO_ROWS)
PROTOCOL_DEPENDENT = set(PROTOCOL_CHECK.loc[PROTOCOL_CHECK.get("protocol", pd.Series(dtype=str)) == "PROTOCOL-DEPENDENT", "model"]) \
    if "protocol" in PROTOCOL_CHECK else set()
if len(PROTOCOL_CHECK):
    print("Single-image protocol check (multi-view models; E5-REAL crates and CONVOY-REAL movers):")
    with pd.option_context("display.width", 250):
        print(PROTOCOL_CHECK.to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
    PROTOCOL_CHECK.to_csv(os.path.join(RESULTS_DIR, "single_image_protocol_check.csv"), index=False)
    if PROTOCOL_DEPENDENT:
        print("Single-image results depend on the input protocol for: " + ", ".join({"pi3": "π³", "VGGT-1B": "VGGT"}.get(m, m) for m in sorted(PROTOCOL_DEPENDENT))
              + "; their single-image claims (H8, H11b) are reported but not made in the paper.")
else:
    print("No multi-view model in this run.")

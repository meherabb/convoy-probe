# PLACEHOLDER
# 6.7c Exploratory (v6.2, AMENDMENT_V6_2): do the masks follow the support cue? Recall of the
#      rigid-flow residual M2 (exact correspondences, at the image's 5% static false-positive rate) on
#      the same parallel movers in every ablation condition. A model that uses more geometry for a
#      mover leaves a smaller residual on it, so M2 should catch fewer movers when the contact is
#      hidden (G -> CH) or the mover is lifted (NS -> F). Paired by mover; cluster = base scene.
def _dir_label(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NOT TESTABLE"
    return "lower" if hi < 0 else "higher" if lo > 0 else "no clear change"

E5M_ROWS, E5M_TEST_ROWS = [], []
for model in MODELS_OK["ABL"]:
    w = E5_WIDE.get(model)
    if w is None:
        continue
    s5 = _sample(w, E5_CONDS, walled=True)[["uid"]]
    d = ABL_RES[(ABL_RES["model"] == model) & ABL_RES["uid"].isin(s5["uid"])]
    r = d.pivot_table(index=["uid", "scene"], columns="condition", values="recall_M2_exact", aggfunc="first").reset_index()
    r = r.dropna(subset=[c for c in E5_CONDS if c in r.columns])
    if len(r) < 8 or r["scene"].nunique() < 5:
        continue
    for c in E5_CONDS:
        est, lo, hi, _ = boot_stat(r, "scene", lambda q, c=c: float(np.mean(q[c])), seed_for("e5m", model, c), cols=[c])
        E5M_ROWS.append(dict(model=model, condition=c, label=ABLATION_DESIGN["labels"][c], n=len(r), recall_M2=est, ci_lo=lo, ci_hi=hi))
    for a, b, what in (("CH", "G", "contact hidden - grounded"), ("F", "NS", "lifted - grounded without shadow")):
        est, lo, hi, _ = boot_stat(r, "scene", lambda q, a=a, b=b: float(np.mean(q[a] - q[b])), seed_for("e5m-c", model, a, b),
                                   cols=[a, b])
        E5M_TEST_ROWS.append(dict(model=model, contrast=f"{a} - {b}", meaning=what, n=len(r), n_scenes=int(r["scene"].nunique()),
                                  estimate=est, ci_lo=lo, ci_hi=hi, direction=_dir_label(lo, hi)))
E5_MASKS = pd.DataFrame(E5M_ROWS)
E5_MASK_TESTS = pd.DataFrame(E5M_TEST_ROWS)
if len(E5_MASKS):
    print("Exploratory: M2 recall (exact correspondences) on the same parallel movers in each ablation condition:")
    print(E5_MASKS.pivot_table(index="model", columns="condition", values="recall_M2")[E5_CONDS]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nPaired changes (95% cluster-bootstrap intervals over base scenes):")
    print(E5_MASK_TESTS[["model", "contrast", "n", "estimate", "ci_lo", "ci_hi", "direction"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    E5_MASKS.to_csv(os.path.join(RESULTS_DIR, "E5_masks_by_condition.csv"), index=False)
    E5_MASK_TESTS.to_csv(os.path.join(RESULTS_DIR, "E5_masks_contrasts.csv"), index=False)
else:
    print("Masks by ablation condition: no interpretable model with enough movers in this run.")

# PLACEHOLDER
# 6.0 Static-scene validity gate (pre-registered). For every model and testbed: median over
#     images of the log-depth slope on static pixels and of AbsRel after scale alignment.
#     PASS: slope in [0.7, 1.3] and AbsRel <= 0.25. LEGACY-FLAT is the negative control.
TESTBED_NAMES = {"MAIN": "CONVOY-3D", "ABL": "CONVOY-3D ablation (E5)", "REAL": "CONVOY-REAL",
                 "RABL": "CONVOY-REAL contact ablation (E5-REAL)", "VK": "Virtual KITTI 2", "KT": "KITTI tracking",
                 "AV2": "Argoverse 2", "BSW": "CONVOY-3D baseline sweep (H12)", "WDR": "CONVOY-3D dose-response (H13)",
                 "BONN": "Bonn RGB-D Dynamic", "FLAT": "LEGACY-FLAT (control)"}
_rows = []
for (model, st), grp in IMG_RES.groupby(["model", "set"]):
    sl, ar = grp["slope"].median(), grp["absrel"].median()
    ok = (PREREG["gate_slope_range"][0] <= sl <= PREREG["gate_slope_range"][1]) and (ar <= PREREG["gate_absrel_max"])
    _rows.append(dict(model=model, testbed=TESTBED_NAMES.get(st, st), set=st, n_images=len(grp),
                      slope_median=sl, slope_q25=grp["slope"].quantile(0.25), slope_q75=grp["slope"].quantile(0.75),
                      absrel_median=ar, absrel_q25=grp["absrel"].quantile(0.25), absrel_q75=grp["absrel"].quantile(0.75),
                      mono_slope_median=grp["mono_slope"].median() if "mono_slope" in grp else np.nan,
                      gate="PASS" if ok else "FAIL"))
GATE = pd.DataFrame(_rows)
GATE_PASS = {(r.model, r.set): r.gate == "PASS" for r in GATE.itertuples()}
_order = ["MAIN", "ABL", "REAL", "RABL", "VK", "KT", "AV2", "BSW", "WDR", "BONN", "FLAT"]
GATE["set"] = pd.Categorical(GATE["set"], _order, ordered=True)
GATE = GATE.sort_values(["set", "model"]).reset_index(drop=True)
print(GATE[["testbed", "model", "n_images", "slope_median", "absrel_median", "mono_slope_median", "gate"]]
      .to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
_flat = GATE[GATE["set"] == "FLAT"]
_main = GATE[GATE["set"] == "MAIN"]
print(f"\nCONVOY-3D gate passed by {int((_main['gate'] == 'PASS').sum())}/{len(_main)} models; "
      f"LEGACY-FLAT control rejected for {int((_flat['gate'] == 'FAIL').sum())}/{len(_flat)} models "
      f"(expected: rejected for all).")
GATE.to_csv(os.path.join(RESULTS_DIR, "gate.csv"), index=False)
def interpretable(model, testbed):
    # a model x testbed combination is interpreted only if it passes the gate (Section 6.0)
    return bool(GATE_PASS.get((model, testbed), False))

# models whose results on each testbed are interpreted (and drawn in the figures)
MODELS_OK = {tb: [m for m in EVALUATED_MODELS if interpretable(m, tb)]
             for tb in ["MAIN", "ABL", "REAL", "RABL", "VK", "KT", "AV2", "BSW", "WDR", "BONN"]}
for _tb, _ok in MODELS_OK.items():
    _no = [m for m in EVALUATED_MODELS if m not in _ok and (m, _tb) in GATE_PASS]
    if _no:
        print(f"Not interpreted on {TESTBED_NAMES[_tb]} (gate FAIL): {', '.join(_no)}")

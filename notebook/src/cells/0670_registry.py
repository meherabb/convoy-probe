# PLACEHOLDER
# 4.5 Model registry and evaluation stages.
#     Every model is evaluated in one checkpointed stage per testbed ("eval:<model>:<testbed>",
#     Section 0.5), so a later session restores finished (model, testbed) results exactly and
#     runs only what is missing. A model whose stages can all be restored is never downloaded.
#     The load check (finite, positive depth from two views, one view and, for multi-view models,
#     a 3-frame window) runs in the master loop on the same loaded copy that is then evaluated, so
#     each model is loaded once per session (Section 5).
MODEL_CLASSES = [DUSt3RModel, MASt3RModel, MonST3RModel, VGGTModel, Pi3Model, MapAnythingModel]
VK_TRAINING_EXPOSURE = {"MASt3R": True, "VGGT-1B": True}   # Virtual KITTI listed in the published training data
MEASURE_VERSION = "m6.2"                             # version of the measurement code in Sections 4.6 and 5

_EVAL_PREREG = {k: PREREG[k] for k in ("erosion_px", "min_object_pixels", "mask_fpr", "static_depth_range_synth",
                                        "static_depth_range_real", "static_depth_range_vk")}

def eval_testbeds(cls):
    tbs = ["MAIN", "FLAT", "ABL"]
    if META_REAL:
        tbs.append("REAL")
    if META_RABL:
        tbs.append("RABL")
    if cls.family == "multiview":
        tbs.append("E4")
    if VK_STATUS["available"] and VK_PAIR_META:
        tbs.append("VK")
    if KT_STATUS["available"] and KT_PAIR_META:
        tbs.append("KT")
    if AV2_STATUS["available"] and AV2_PAIR_META:
        tbs.append("AV2")
    if AV2X_STATUS["available"] and AV2X_PAIR_META:
        tbs.append("AV2X")                            # v6.4: the further Argoverse 2 logs (pooled with AV2 in every analysis)
    tbs += ["BSW", "WDR"]                             # v6.5: the designed experiments (D55)
    if BONN_STATUS["available"] and BONN_PAIR_META:
        tbs.append("BONN")                            # v6.5: Bonn RGB-D Dynamic (D56)
    return tbs

def _eval_upstream(tb):
    return {"MAIN": [RENDER_FP["MAIN"], FLOW_FP.get("MAIN", "")], "FLAT": [RENDER_FP["FLAT"]], "ABL": [RENDER_FP["ABL"]],
            "REAL": [TUM_STAGES["REAL"].fp, FLOW_FP.get("REAL", "")], "RABL": [RABL_STAGE.fp], "E4": [RENDER_FP["E4"]],
            "VK": [VK_GT_STAGE.fp], "KT": [KT_STAGE.fp], "AV2": [AV2_STAGE.fp], "AV2X": [AV2X_STAGE.fp],
            "BSW": [RENDER_FP["BSW"]], "WDR": [RENDER_FP["WDR"]], "BONN": [BONN_STAGE.fp]}[tb]

def eval_stage(cls, tb):
    m = cls()                                         # constructor only: no download, no weights
    cfg = dict(model=m.name, checkpoint=m.checkpoint, input=list(m.input_size), precision=m.precision, testbed=tb,
               measure=MEASURE_VERSION, prereg=_EVAL_PREREG, window_sizes=list(RUN_CONFIG["e4_window_sizes"]))
    st = Stage(f"eval:{m.name}:{tb}", "eval", cfg, upstream=_eval_upstream(tb))
    st.out_dir = os.path.join(CACHE_DIR, m.name, tb)
    return st

EVAL_STAGES = {cls.name: {tb: eval_stage(cls, tb) for tb in eval_testbeds(cls)} for cls in MODEL_CLASSES}
MODELS, MODEL_STATUS = {}, {}

def _check_depth(out, ref, tag):
    d = out["depth"]
    assert d.shape == ref.shape[:2], f"{tag}: depth shape {d.shape}"
    fin = np.isfinite(d)
    assert fin.mean() > 0.99, f"{tag}: only {fin.mean():.1%} finite"
    assert np.median(d[fin]) > 0, f"{tag}: non-positive median depth"

def load_check(m):
    # one forward pass of every kind the evaluation uses, on a CONVOY-3D pair and an E4 window
    ims = [load_rgb(META_MAIN[0]["dir"], k) for k in range(2)]
    _check_depth(m.pair(ims[0], ims[1]), ims[0], "two views")
    _check_depth(m.mono(ims[0]), ims[0], "single image")
    if m.family == "multiview":
        e4 = [load_rgb(META_E4[0]["dir"], k) for k in range(3)]
        _check_depth(m.multi(e4), e4[0], "3-frame window")

for cls in MODEL_CLASSES:
    m = cls()
    MODEL_STATUS[m.name] = dict(model=m.name, family=m.family, training=m.training, checkpoint=m.checkpoint,
                                input=f"{m.input_size[0]}x{m.input_size[1]}", precision=m.precision, status="PENDING",
                                params_M=None, load_s=None, peak_gpu_GiB=None, check_s=None, note="")
    MODELS[m.name] = cls
print("Models and their evaluation stages:")
for cls in MODEL_CLASSES:
    st = EVAL_STAGES[cls.name]
    n_av = sum(s_.available() is not None for s_ in st.values())
    print(f"  {cls.name:<12} {', '.join(st)}  ({n_av}/{len(st)} restorable from checkpoints)")

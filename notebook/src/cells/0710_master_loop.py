# PLACEHOLDER
# 5.1 Master evaluation loop: one checkpointed stage per (model, testbed). A model is loaded only
#     if at least one of its stages must be computed. Its load check runs on the loaded copy, each
#     finished (model, testbed) stage is archived at once (checkpoints/ and convoy_checkpoints.zip), and
#     the model is released, with GPU and host memory handed back, before the next model is loaded.
#     A model is loaded only if enough RAM is free (MEMORY_GUARD_GIB); otherwise it is deferred,
#     reported, and evaluated by the next session that attaches this run's archive.
EXEMPLAR_MAIN = set(range(min(10, len(META_MAIN))))
EXEMPLAR_REAL = set(range(min(3, len(META_REAL))))
EXEMPLAR_ABL = set(range(min(6, N_ABL)))
EXEMPLAR_RABL = set(range(min(6, len({m["index"] for m in META_RABL}))))
EXEMPLAR_VK = set(range(min(8, len(VK_PAIR_META))))
EXEMPLAR_KT = set(range(min(6, len(KT_PAIR_META))))
EXEMPLAR_AV2 = set(range(min(6, len(AV2_PAIR_META))))
EXEMPLAR_BSW = set(range(min(4, N_BSW)))
EXEMPLAR_WDR = set(range(min(4, N_WDR)))
EXEMPLAR_BONN = set(range(min(6, len(BONN_PAIR_META))))

def _mono_dup_rows(m, rows, img0, gt, sm, masks):
    # v6.2 single-image protocol check (multi-view models): the image given twice instead of once
    if m.family != "multiview":
        return None
    od = m.multi([img0, img0])
    gd = scale_and_gate(od["depth"], gt, sm)
    for r, mk in zip(rows, masks):
        r["rho_mono_dup"] = mover_rho(od["depth"], gt, mk, gd["scale"])
    return gd

def _down2(a):
    a = np.asarray(a, np.float32)
    return cv2.resize(np.nan_to_num(a, nan=0.0, posinf=0.0), (a.shape[1] // 2, a.shape[0] // 2),
                      interpolation=cv2.INTER_AREA).astype(np.float16)

def _score_block(o, om, sc, pB, vis, p_flow, sm, masks, scale, mono_scale, seed):
    # per-mover recall / AUROC of the model-based masks M2 (exact and flow), M4 and M5
    dsc = o["depth"].astype(np.float64) * scale
    m2e = rigid_flow_residual_map(sc, dsc, pB)
    # M5 (exploratory remedy): disagreement between the two-view and the single-image depth of
    # the same model, |log(two-view) - log(single image)| after each is scale-aligned
    with np.errstate(divide="ignore", invalid="ignore"):
        m5 = np.abs(np.log(np.clip(dsc, 1e-6, None)) - np.log(np.clip(om["depth"].astype(np.float64) * mono_scale, 1e-6, None)))
    res = {"M2_exact": mask_scores(m2e, sm & vis, masks, seed), "M4": mask_scores(-o["conf"].astype(np.float64), sm, masks, seed),
           "M5": mask_scores(m5, sm, masks, seed)}
    if p_flow is not None:
        res["M2_flow"] = mask_scores(rigid_flow_residual_map(sc, dsc, p_flow), sm, masks, seed)
    return res, m2e, m5

def _mover_rows(set_, idx, masks, o, om, gt, sm, g, gm, scores, extra=None, ko=None, gk=None):
    rows = []
    for j, mk in enumerate(masks):
        row = dict(set=set_, scene=idx, mover=j, uid=f"{set_}-{idx:04d}-{j}", n_px=int(mk.sum()),
                   rho=mover_rho(o["depth"], gt, mk, g["scale"]), rho_mono=mover_rho(om["depth"], gt, mk, gm["scale"]),
                   rho_ko=mover_rho(ko["depth"], gt, mk, gk["scale"]) if ko is not None else np.nan,
                   conf_mover=float(np.median(o["conf"][mk])) if mk.any() else np.nan,
                   conf_static=float(np.median(o["conf"][sm])) if sm.any() else np.nan, **(extra or {}))
        for meth, lst in scores.items():
            row[f"recall_{meth}"] = lst[j]["recall"]; row[f"auroc_{meth}"] = lst[j]["auroc"]
        rows.append(row)
    return rows

def _img_row(set_, idx, g, gm, **kw):
    return dict(set=set_, scene=idx, **g, mono_scale=gm["scale"], mono_slope=gm["slope"], mono_absrel=gm["absrel"], **kw)

def _flat_gate(o, anchors):
    zp, zt = [], []
    for a in anchors:
        u, v = int(round(a["u"])), int(round(a["v"]))
        zp.append(float(np.median(o["depth"][max(v - 2, 0):v + 3, max(u - 2, 0):u + 3])))
        zt.append(a["Z"])
    zp, zt = np.asarray(zp), np.asarray(zt)
    ok = np.isfinite(zp) & (zp > 0)
    if ok.sum() < 4:
        return dict(scale=np.nan, slope=np.nan, absrel=np.nan, n_static=int(ok.sum()))
    s = float(np.median(zt[ok] / zp[ok]))
    return dict(scale=s, slope=float(np.polyfit(np.log(zt[ok]), np.log(zp[ok]), 1)[0]),
                absrel=float(np.mean(np.abs(s * zp[ok] - zt[ok]) / zt[ok])), n_static=int(ok.sum()))

# ---------------------------------------------------------------- one function per testbed
def _eval_main(m, name):
    img, mov, ex = [], [], {}
    for meta in META_MAIN:
        sc, d, idx = meta["scene"], meta["dir"], meta["index"]
        ims = [load_rgb(d, k) for k in range(2)]
        r0, r1 = load_frame(d, 0), load_frame(d, 1)
        masks, pB, vis = mover_masks(sc, r0, r1)
        sm = static_mask(r0, "synth"); gt = r0["depth"]
        o, om = m.pair(ims[0], ims[1]), m.mono(ims[0])
        has_ko = os.path.exists(os.path.join(d, "frame1_ko.png"))
        ko = m.pair(ims[0], load_rgb(d, 1, "_ko")) if has_ko else None
        g, gm = scale_and_gate(o["depth"], gt, sm), scale_and_gate(om["depth"], gt, sm)
        gk = scale_and_gate(ko["depth"], gt, sm) if has_ko else None
        img.append(_img_row("MAIN", idx, g, gm))
        scores, m2e, m5 = _score_block(o, om, sc, pB, vis, flow_to_pobs(load_flow(meta)), sm, masks, g["scale"], gm["scale"],
                                       seed_for(name, "MAIN", idx))
        mov += _mover_rows("MAIN", idx, masks, o, om, gt, sm, g, gm, scores, ko=ko, gk=gk)
        if idx in EXEMPLAR_MAIN:
            ex[f"MAIN-{idx}"] = dict(depth=_down2(o["depth"] * g["scale"]), mono=_down2(om["depth"] * gm["scale"]),
                                     m2=_down2(m2e), m5=_down2(m5), conf=_down2(o["conf"]),
                                     **({"ko": _down2(ko["depth"] * gk["scale"])} if has_ko else {}))
    return dict(images=img, movers=mov), ex

def _eval_flat(m, name):
    img = []
    for meta in META_FLAT:
        o = m.pair(load_rgb(meta["dir"], 0), load_rgb(meta["dir"], 1))
        img.append(dict(set="FLAT", scene=meta["index"], **_flat_gate(o, meta["anchors"])))
    return dict(images=img), {}

def _eval_real(m, name):
    img, mov, ex = [], [], {}
    for meta in META_REAL:
        sc, d, idx = meta["scene"], meta["dir"], meta["index"]
        ims = [load_rgb(d, k) for k in range(2)]
        r0, r1 = load_frame(d, 0), load_frame(d, 1)
        masks, pB, vis = mover_masks(sc, r0, r1)
        sm = static_mask(r0, "real"); gt = r0["depth"]
        o, om = m.pair(ims[0], ims[1]), m.mono(ims[0])
        g, gm = scale_and_gate(o["depth"], gt, sm), scale_and_gate(om["depth"], gt, sm)
        img.append(_img_row("REAL", idx, g, gm, seq=meta["seq"]))
        scores, m2e, m5 = _score_block(o, om, sc, pB, vis, flow_to_pobs(load_flow(meta)), sm, masks, g["scale"], gm["scale"],
                                       seed_for(name, "REAL", idx))
        rows = _mover_rows("REAL", idx, masks, o, om, gt, sm, g, gm, scores)
        gd = _mono_dup_rows(m, rows, ims[0], gt, sm, masks)
        if gd is not None:
            img[-1].update(mono_dup_slope=gd["slope"], mono_dup_absrel=gd["absrel"])
        mov += rows
        if idx in EXEMPLAR_REAL:
            ex[f"REAL-{idx}"] = dict(depth=_down2(o["depth"] * g["scale"]), mono=_down2(om["depth"] * gm["scale"]),
                                     m2=_down2(m2e), m5=_down2(m5), conf=_down2(o["conf"]))
    return dict(images=img, movers=mov), ex

def _eval_abl(m, name):
    # E5: every condition of every base scene; rows keyed by (uid, condition)
    img, mov, ex = [], [], {}
    for meta in META_ABL:
        sc, d, idx, cond = meta["scene"], meta["dir"], meta["index"], meta["condition"]
        ims = [load_rgb(d, k) for k in range(2)]
        r0, r1 = load_frame(d, 0), load_frame(d, 1)
        masks, pB, vis = mover_masks(sc, r0, r1)
        sm = static_mask(r0, "synth"); gt = r0["depth"]
        o, om = m.pair(ims[0], ims[1]), m.mono(ims[0])
        g, gm = scale_and_gate(o["depth"], gt, sm), scale_and_gate(om["depth"], gt, sm)
        img.append(_img_row("ABL", idx, g, gm, condition=cond))
        scores, m2e, m5 = _score_block(o, om, sc, pB, vis, None, sm, masks, g["scale"], gm["scale"], seed_for(name, "ABL", idx, cond))
        mov += _mover_rows("ABL", idx, masks, o, om, gt, sm, g, gm, scores, extra=dict(condition=cond))
        if idx in EXEMPLAR_ABL:
            ex[f"ABL-{idx}-{cond}"] = dict(depth=_down2(o["depth"] * g["scale"]), mono=_down2(om["depth"] * gm["scale"]))
    return dict(images=img, movers=mov), ex

def _eval_e4(m, name):
    rows = []
    for meta in META_E4:
        sc, d, idx = meta["scene"], meta["dir"], meta["index"]
        K = len(sc["cams"])
        ims = [load_rgb(d, k) for k in range(K)]
        r0, r1 = load_frame(d, 0), load_frame(d, 1)
        masks, _, _ = mover_masks(sc, r0, r1)
        sm = static_mask(r0, "synth")
        for Kw in RUN_CONFIG["e4_window_sizes"]:
            o = m.multi(ims[:Kw]); g = scale_and_gate(o["depth"], r0["depth"], sm)
            for j, mk in enumerate(masks):
                rows.append(dict(path=sc["camera"]["path"], scene=idx, mover=j, uid=f"E4-{idx:04d}-{j}", K=Kw,
                                 rho=mover_rho(o["depth"], r0["depth"], mk, g["scale"]), slope=g["slope"], absrel=g["absrel"]))
    return dict(e4=rows), {}

def _eval_vk(m, name):
    img, mov, ex = [], [], {}
    for meta in VK_PAIR_META:
        vsc = VK_SCENE_OBJS[meta["seq"]]; i, j = meta["frames"]; idx = meta["index"]
        gtp = vsc.pair_gt(i, j)
        A, B = vsc.frame_gt(i), vsc.frame_gt(j)
        masks = [erode_mask(gtp["mover_id"] == k, PREREG["erosion_px"]) & gtp["visible"] for k in range(len(gtp["movers"]))]
        sm = vk_static_mask(A["depth"], A["inst"]); gt = A["depth"]
        o, om = m.pair(A["rgb"], B["rgb"]), m.mono(A["rgb"])
        g, gm = scale_and_gate(o["depth"], gt, sm), scale_and_gate(om["depth"], gt, sm)
        img.append(_img_row("VK", idx, g, gm, seq=meta["seq"]))
        scores, m2e, m5 = _score_block(o, om, gtp["scene"], gtp["pB"], gtp["visible"], None, sm, masks, g["scale"], gm["scale"],
                                       seed_for(name, "VK", idx))
        mov += _mover_rows("VK", idx, masks, o, om, gt, sm, g, gm, scores)
        if idx in EXEMPLAR_VK:
            ex[f"VK-{idx}"] = dict(depth=_down2(o["depth"] * g["scale"]), mono=_down2(om["depth"] * gm["scale"]),
                                   m2=_down2(m2e), m5=_down2(m5))
    return dict(images=img, movers=mov), ex

def _eval_lidar(m, set_, metas, exemplars):
    # real driving with LiDAR (KITTI tracking, Argoverse 2; exploratory): sparse ground truth from the
    # nearest LiDAR return per pixel; no dense masks, so no mask scores
    img, mov, ex = [], [], {}
    for meta in metas:
        d, idx = meta["dir"], meta["index"]
        A = np.asarray(Image.open(os.path.join(d, "rgb0.png")).convert("RGB"))
        B = np.asarray(Image.open(os.path.join(d, "rgb1.png")).convert("RGB"))
        z = np.load(os.path.join(d, "gt.npz")); gt, sm = z["depth"], z["static"]
        masks = []
        for v in (meta["objects"] if "objects" in meta else meta["vehicles"]):      # Bonn (v6.5) stores objects
            mk = np.zeros(gt.shape, bool)
            px = np.asarray(v["pixels"], int).reshape(-1, 2)
            if len(px):
                mk[px[:, 1], px[:, 0]] = True
            masks.append(mk & np.isfinite(gt))
        o, om = m.pair(A, B), m.mono(A)
        g, gm = scale_and_gate(o["depth"], gt, sm), scale_and_gate(om["depth"], gt, sm)
        img.append(_img_row(set_, idx, g, gm, seq=meta["seq"]))
        mov += _mover_rows(set_, idx, masks, o, om, gt, sm, g, gm, {})
        if idx in exemplars:
            ex[f"{set_}-{idx}"] = dict(depth=_down2(o["depth"] * g["scale"]), mono=_down2(om["depth"] * gm["scale"]))
    return dict(images=img, movers=mov), ex

def _eval_kitti(m, name):
    return _eval_lidar(m, "KT", KT_PAIR_META, EXEMPLAR_KT)

def _eval_av2(m, name):
    return _eval_lidar(m, "AV2", AV2_PAIR_META, EXEMPLAR_AV2)

def _eval_av2x(m, name):
    # v6.4: the further Argoverse 2 logs, under the same set label (their pair indices continue those of AV2)
    return _eval_lidar(m, "AV2", AV2X_PAIR_META, set())

def _eval_designed(m, name, set_, metas, exemplars, mono_shared):
    # v6.5 designed experiments (D55): every level of every base scene; rows keyed by (uid, condition). In the baseline
    # sweep the first image is the same at every level, so its single-image prediction is computed once per base scene.
    img, mov, ex, mono_cache = [], [], {}, {}
    for meta in metas:
        sc, d, idx, cond = meta["scene"], meta["dir"], meta["index"], meta["condition"]
        ims = [load_rgb(d, k) for k in range(2)]
        r0, r1 = load_frame(d, 0), load_frame(d, 1)
        masks, pB, vis = mover_masks(sc, r0, r1)
        sm = static_mask(r0, "synth"); gt = r0["depth"]
        o = m.pair(ims[0], ims[1])
        if mono_shared:
            if idx not in mono_cache:
                mono_cache.clear(); mono_cache[idx] = m.mono(ims[0])
            om = mono_cache[idx]
        else:
            om = m.mono(ims[0])
        g, gm = scale_and_gate(o["depth"], gt, sm), scale_and_gate(om["depth"], gt, sm)
        img.append(_img_row(set_, idx, g, gm, condition=cond))
        scores, m2e, m5 = _score_block(o, om, sc, pB, vis, None, sm, masks, g["scale"], gm["scale"], seed_for(name, set_, idx, cond))
        mov += _mover_rows(set_, idx, masks, o, om, gt, sm, g, gm, scores, extra=dict(condition=cond))
        if idx in exemplars:
            ex[f"{set_}-{idx}-{cond}"] = dict(depth=_down2(o["depth"] * g["scale"]), mono=_down2(om["depth"] * gm["scale"]),
                                              m2=_down2(np.clip(m2e, 0.0, 6.0e4)))        # within the float16 range
    return dict(images=img, movers=mov), ex

def _eval_bsw(m, name):
    return _eval_designed(m, name, "BSW", META_BSW, EXEMPLAR_BSW, mono_shared=True)

def _eval_wdr(m, name):
    return _eval_designed(m, name, "WDR", META_WDR, EXEMPLAR_WDR, mono_shared=False)

def _eval_bonn(m, name):
    # v6.5 Bonn RGB-D Dynamic (D56): measured depth, objects from moving regions (Section 3.12); no dense masks
    return _eval_lidar(m, "BONN", BONN_PAIR_META, EXEMPLAR_BONN)

def _eval_rabl(m, name):
    # E5-REAL: both conditions of every real-background base pair; rows keyed by (uid, condition)
    img, mov, ex = [], [], {}
    for meta in META_RABL:
        sc, d, idx, cond = meta["scene"], meta["dir"], meta["index"], meta["condition"]
        ims = [load_rgb(d, k) for k in range(2)]
        r0, r1 = load_frame(d, 0), load_frame(d, 1)
        masks, pB, vis = mover_masks(sc, r0, r1)
        sm = static_mask(r0, "real"); gt = r0["depth"]
        o, om = m.pair(ims[0], ims[1]), m.mono(ims[0])
        g, gm = scale_and_gate(o["depth"], gt, sm), scale_and_gate(om["depth"], gt, sm)
        img.append(_img_row("RABL", idx, g, gm, condition=cond, seq=meta["seq"]))
        scores, m2e, m5 = _score_block(o, om, sc, pB, vis, None, sm, masks, g["scale"], gm["scale"], seed_for(name, "RABL", idx, cond))
        rows = _mover_rows("RABL", idx, masks, o, om, gt, sm, g, gm, scores, extra=dict(condition=cond))
        gd = _mono_dup_rows(m, rows, ims[0], gt, sm, masks)
        if gd is not None:
            img[-1].update(mono_dup_slope=gd["slope"], mono_dup_absrel=gd["absrel"])
        mov += rows
        if idx in EXEMPLAR_RABL:
            ex[f"RABL-{idx}-{cond}"] = dict(depth=_down2(o["depth"] * g["scale"]), mono=_down2(om["depth"] * gm["scale"]))
    return dict(images=img, movers=mov), ex

EVAL_FUNCTIONS = {"MAIN": _eval_main, "FLAT": _eval_flat, "REAL": _eval_real, "RABL": _eval_rabl, "ABL": _eval_abl,
                  "E4": _eval_e4, "VK": _eval_vk, "KT": _eval_kitti, "AV2": _eval_av2, "AV2X": _eval_av2x,
                  "BSW": _eval_bsw, "WDR": _eval_wdr, "BONN": _eval_bonn}

def _write_stage(st, tables, exemplars, seconds, status_record):
    shutil.rmtree(st.out_dir, ignore_errors=True); os.makedirs(st.out_dir)
    for k, rows in tables.items():
        pd.DataFrame(rows).to_csv(os.path.join(st.out_dir, f"{k}.csv"), index=False)
    np.savez_compressed(os.path.join(st.out_dir, "exemplars.npz"),
                        **{f"{k}|{kk}": vv for k, v in exemplars.items() for kk, vv in v.items()})
    with open(os.path.join(st.out_dir, "done.json"), "w") as f:
        json.dump(dict(stage=st.name, seconds=round(seconds, 1),
                       finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), f, indent=1)
    st.commit([st.out_dir], extra=dict(seconds=round(seconds, 1), status_record=status_record))

def evaluate_model(name, cls):
    stages, rec = EVAL_STAGES[name], MODEL_STATUS[name]
    todo = []
    for tb, st in stages.items():
        if not st.done():                    # prints RESTORED (inference skipped) or RUNNING for each testbed
            todo.append(tb)
    if not todo:
        st0 = stages["MAIN"]
        prev = (_read_json(st0.reg_path) or {}).get("extra", {}).get("status_record") or {}
        rec.update(status="PASS", params_M=prev.get("params_M"), load_s=prev.get("load_s"), peak_gpu_GiB=prev.get("peak_gpu_GiB"),
                   check_s=0.0, note=f"all {len(stages)} evaluation stages restored; not loaded")
        return
    release_memory()
    r = mem_report(f"before loading {name}")
    if r["available"] < MEMORY_GUARD_GIB:
        rec.update(status="DEFERRED", note=f"not loaded: {r['available']:.1f} GiB RAM free (< {MEMORY_GUARD_GIB} GiB); "
                                           f"{len(todo)} stage(s) left for the next session")
        print(f"[{name}] DEFERRED: {rec['note']}")
        return
    t0 = time.time()
    m = cls()
    try:
        m.load(); m.to_gpu()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        load_check(m)
        rec.update(status="PASS", params_M=round(m.n_params / 1e6, 1), load_s=round(m.load_seconds, 1),
                   check_s=round(time.time() - t0, 1),
                   note="loaded and ran" + (f"; {len(stages) - len(todo)} of {len(stages)} stages restored" if len(todo) < len(stages) else ""))
        print(f"[{name}] PASS: loaded ({rec['params_M']} M parameters, {rec['load_s']} s) and passed the load check")
        for tb in todo:
            t1 = time.time()
            tables, ex = EVAL_FUNCTIONS[tb](m, name)
            rec["peak_gpu_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0
            _write_stage(stages[tb], tables, ex, time.time() - t1, dict(rec))
            del tables, ex
            if tb == "VK":
                for _sc in VK_SCENE_OBJS.values():
                    _sc._cache.clear()
            release_memory()
    except Exception as e:
        rec.update(status="FAIL", note=f"{type(e).__name__}: {str(e)[:200]}")
        print(f"[{name}] FAIL: {rec['note']}")
    finally:
        m.unload(); del m
        release_memory()
        mem_report(f"after releasing {name}")

def _free_checkpoint_if_disk_low(cls, min_free_gib=15.0):
    # once a model's results are checkpointed its weights are no longer needed; free them only when
    # the disk is getting full (a re-run restores the results, not the weights)
    free = shutil.disk_usage(os.path.expanduser("~")).free / 2**30
    if free >= min_free_gib:
        return
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    d = os.path.join(hub, "models--" + cls.checkpoint.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print(f"[disk] freed the downloaded weights of {cls.checkpoint} ({free:.1f} GiB free before)")

with stage("master_evaluation"):
    for _name, _cls in MODELS.items():
        evaluate_model(_name, _cls)          # failures are reported and the model is excluded from every analysis;
        _free_checkpoint_if_disk_low(_cls, min_free_gib=1e9)   # weights are not needed once the results are archived
EVALUATED_MODELS = [n for n in MODELS if MODEL_STATUS[n]["status"] == "PASS"
                    and all(os.path.exists(os.path.join(st.out_dir, "done.json")) for st in EVAL_STAGES[n].values())]
MODEL_TABLE = pd.DataFrame(MODEL_STATUS.values())
print("\n" + MODEL_TABLE.to_string(index=False, na_rep="–"))
print("\n" + "=" * 88)
_n_restored = sum("not loaded" in r["note"] for r in MODEL_STATUS.values())
print(f"MODEL SUMMARY: {len(EVALUATED_MODELS)}/{len(MODEL_STATUS)} models evaluated: "
      f"{len(EVALUATED_MODELS) - _n_restored} ran on the GPU in this session, {_n_restored} restored from checkpoints")
print("=" * 88)
for k, r_ in MODEL_STATUS.items():
    print(f"  [{'PASS' if k in EVALUATED_MODELS else r_['status']}] {k:<12} testbeds: {', '.join(EVAL_STAGES[k])}"
          + ("" if k in EVALUATED_MODELS else f"  -> left out of every experiment: {r_['note']}"))
if any(r_["status"] == "DEFERRED" for r_ in MODEL_STATUS.values()):
    print("Deferred models: attach convoy_checkpoints.zip (or this notebook's output) to a new session and run all cells; "
          "every finished stage is restored and only the deferred models are evaluated.")
MODEL_TABLE.to_csv(os.path.join(LOG_DIR, "model_status.csv"), index=False)
print(f"\nEvaluated models ({len(EVALUATED_MODELS)}): {', '.join(EVALUATED_MODELS)}")
_et = stage_table("eval:")
if len(_et):
    print(_et.groupby("status").size().to_string())
    print(f"checkpoints/: {len(glob.glob(os.path.join(CKPT_DIR, '*.zip')))} stage archives, "
          f"{sum(os.path.getsize(p) for p in glob.glob(os.path.join(CKPT_DIR, '*.zip'))) / 2**20:.0f} MiB")
# every model result is in convoy_checkpoints.zip now: offer it (and copy it to the private dataset,
# if configured), so that a session lost during the analysis still leaves a complete resume point
print("Dataset copy:", backup_to_dataset([ZIP_PATH], "after the master evaluation"))
offer_downloads([ZIP_PATH])

# PLACEHOLDER
# 3.6 Optical flow used by (i) the classical triangulation positive control, (ii) the
#     flow-based dynamic masks of Section 4, and (iii) the ground truth of real walking people
#     (Section 3.7). RAFT-large (torchvision weights) is the estimator; if its weights cannot be
#     fetched, OpenCV's DIS flow is used. RAFT is loaded only when some flow must be computed;
#     flows restored from a checkpoint keep the estimator they were computed with.
FLOW = {"method": None, "model": None, "transforms": None, "preferred": FLOW_PREFERRED}

def _ensure_flow_model():
    if FLOW["method"] is not None:
        return
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        _w = Raft_Large_Weights.DEFAULT
        FLOW["model"] = raft_large(weights=_w, progress=False).eval()
        FLOW["transforms"] = _w.transforms()
        FLOW["method"] = f"RAFT-large (torchvision {_w.name})"
    except Exception as e:
        FLOW["method"] = "DIS (OpenCV, preset MEDIUM)"
        FLOW["fallback_reason"] = f"{type(e).__name__}: {str(e)[:160]}"
    print("Optical flow estimator loaded:", FLOW["method"])

def _flow_dis(a, b):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(ga, gb, None).astype(np.float32)

@torch.no_grad()
def _flow_raft(pairs):
    model = FLOW["model"].to(device)
    out = []
    for a, b in pairs:
        H, W = a.shape[:2]
        H8, W8 = (H + 7) // 8 * 8, (W + 7) // 8 * 8           # RAFT needs sizes divisible by 8
        pad = ((0, H8 - H), (0, W8 - W), (0, 0))
        ta = torch.from_numpy(np.pad(a, pad, mode="edge").copy()).permute(2, 0, 1)[None]
        tb = torch.from_numpy(np.pad(b, pad, mode="edge").copy()).permute(2, 0, 1)[None]
        ta, tb = FLOW["transforms"](ta, tb)
        flows = model(ta.to(device), tb.to(device), num_flow_updates=12)
        out.append(flows[-1][0].permute(1, 2, 0).float().cpu().numpy()[:H, :W])
    return out

def compute_flows(pairs, cache_paths):
    # pairs: list of (img_a, img_b) uint8; returns list of (H,W,2) flows a->b (cached as fp16)
    res, todo = [None] * len(pairs), []
    for n, p in enumerate(cache_paths):
        if os.path.exists(p):
            res[n] = np.load(p)["flow"].astype(np.float32)
        else:
            todo.append(n)
    if todo:
        _ensure_flow_model()
        if FLOW["model"] is not None:
            for start in range(0, len(todo), 16):
                chunk = todo[start:start + 16]
                for n, f in zip(chunk, _flow_raft([pairs[n] for n in chunk])):
                    res[n] = f
            FLOW["model"].to("cpu")
            release_memory() if "release_memory" in globals() else torch.cuda.empty_cache()
        else:
            for n in todo:
                res[n] = _flow_dis(*pairs[n])
        for n in todo:
            os.makedirs(os.path.dirname(cache_paths[n]), exist_ok=True)
            np.savez_compressed(cache_paths[n], flow=res[n].astype(np.float16))
    return res

def sample_map(arr, pts):
    # bilinear lookup of an (H,W[,C]) map at float pixel positions (N,2); NaN outside
    H, W = arr.shape[:2]
    x, y = pts[:, 0], pts[:, 1]
    ok = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    xc, yc = np.clip(x, 0, W - 1), np.clip(y, 0, H - 1)
    x0, y0 = np.floor(xc).astype(int), np.floor(yc).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = xc - x0, yc - y0
    if arr.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    v = (arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x1] * fx * (1 - fy)
         + arr[y1, x0] * (1 - fx) * fy + arr[y1, x1] * fx * fy)
    v = v.astype(np.float64)
    v[~ok] = np.nan
    return v

FLOW_FP = {}

def flow_stage(name, metas, upstream):
    # flows of frames 0 -> 1 for every meta (one checkpointed stage); returns {meta dir: flow file}
    st = Stage(f"flow:{name}", "flow", dict(preferred=FLOW_PREFERRED), upstream=upstream)
    FLOW_FP[name] = st.fp
    paths = {m["dir"]: os.path.join(FLOW_DIR, name.lower(), f"{os.path.basename(m['dir'])}_01.npz") for m in metas}
    method_file = os.path.join(FLOW_DIR, f"method_{name.lower()}.json")
    if not st.done():
        pairs = [(load_rgb(m["dir"], 0), load_rgb(m["dir"], 1)) for m in metas]
        compute_flows(pairs, [paths[m["dir"]] for m in metas])
        with open(method_file, "w") as f:
            json.dump({"method": FLOW["method"], "fallback_reason": FLOW.get("fallback_reason")}, f)
        st.commit(list(paths.values()) + [method_file])
    with open(method_file) as f:
        return paths, json.load(f)["method"]

with stage("flow"):
    _t0 = time.time()
    FLOW_CACHE, _m1 = flow_stage("MAIN", META_MAIN, [RENDER_FP["MAIN"]])
    FLOW_METHOD = {"MAIN": _m1}
    if META_REAL:
        _p, _m2 = flow_stage("REAL", META_REAL, [TUM_STAGES["REAL"].fp])
        FLOW_CACHE.update(_p); FLOW_METHOD["REAL"] = _m2
FLOW["method_used"] = FLOW_METHOD
print(f"flows ready for {len(FLOW_CACHE)} pairs ({time.time() - _t0:.0f}s); estimator per set: {FLOW_METHOD}")
FLOW_NAME = sorted(set(FLOW_METHOD.values()))[0].split(" (")[0] if FLOW_METHOD else "flow"

def load_flow(meta):
    return np.load(FLOW_CACHE[meta["dir"]])["flow"].astype(np.float32)

# flow accuracy on CONVOY-3D static pixels (end-point error against the exact correspondence)
_epe = []
for meta in META_MAIN[: min(24, len(META_MAIN))]:
    sc = meta["scene"]; r0 = load_frame(meta["dir"], 0); r1 = load_frame(meta["dir"], 1)
    pB, _, _, vis = gt_correspondence(sc, r0, 0, 1, r1)
    H, W = r0["depth"].shape
    vv, uu = np.mgrid[0:H, 0:W]
    fl = load_flow(meta)
    gtf = pB - np.stack([uu, vv], -1)
    m = vis & (r0["mover_id"] < 0) & np.isfinite(r0["depth"])
    _epe.append(np.linalg.norm(fl[m] - gtf[m], axis=-1))
_epe = np.concatenate(_epe)
FLOW["epe_static_convoy3d_median"] = float(np.median(_epe))
FLOW["epe_static_convoy3d_p90"] = float(np.percentile(_epe, 90))
print(f"{FLOW_METHOD['MAIN']}: end-point error on CONVOY-3D static pixels: median "
      f"{FLOW['epe_static_convoy3d_median']:.2f} px, 90th percentile {FLOW['epe_static_convoy3d_p90']:.2f} px")

release_memory(); _ = mem_report("optical flow")

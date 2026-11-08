# PLACEHOLDER
# 3.2 Build the CONVOY-3D manifests and render everything. Each set is one checkpointed stage
#     (Section 0.5): it is restored from a previous run when its fingerprint matches, otherwise
#     sampled and rendered here and archived the moment it is complete.
#     MAIN      : the 240 v5 clips (specifications identical to v5); frames 0 and 1 are rendered
#     KNOCKOUT  : frame 1 of the first n MAIN clips re-rendered with every mover removed (A1)
#     E4        : 8-frame clips, straight vs curved camera path, purely parallel movers
#     FLAT      : LEGACY-FLAT control pairs
#     ABL       : E5 causal ablation, 72 base scenes x 5 conditions (Section 3.1.7)
#     Memory is released and reported after every set (renders are large, transient numpy arrays).
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def _save_frame(dirpath, k, r, tag=""):
    Image.fromarray(r["rgb"]).save(os.path.join(dirpath, f"frame{k}{tag}.png"))
    np.savez_compressed(os.path.join(dirpath, f"gt{k}{tag}.npz"), depth=r["depth"], sem=r["sem"],
                        mover_id=r["mover_id"].astype(np.int8), mover_shadow=r["mover_shadow"])

def load_frame(dirpath, k, tag=""):
    rgb = np.asarray(Image.open(os.path.join(dirpath, f"frame{k}{tag}.png")).convert("RGB"))
    z = np.load(os.path.join(dirpath, f"gt{k}{tag}.npz"))
    return dict(rgb=rgb, depth=z["depth"], sem=z["sem"], mover_id=z["mover_id"].astype(np.int16),
                mover_shadow=z["mover_shadow"])

def load_rgb(dirpath, k, tag=""):
    return np.asarray(Image.open(os.path.join(dirpath, f"frame{k}{tag}.png")).convert("RGB"))

def _specs(set_name, seeds, decks, n_frames, path, e4):
    # sample (or reload) the scene specifications of one set (deterministic)
    spec_file = os.path.join(RENDER_DIR, f"specs_{set_name}_{path}.json")
    if os.path.exists(spec_file):
        with open(spec_file) as f:
            blob = json.load(f)
        if blob.get("n") == len(seeds):
            return blob["scenes"], blob["unplaced"]
    scenes, unplaced = sample_clip_sequence(seeds, decks, n_frames, path, e4)
    scenes = [[scene_to_json(t) for t in s] if isinstance(s, list) else scene_to_json(s) for s in scenes]
    with open(spec_file, "w") as f:
        json.dump({"n": len(seeds), "scenes": scenes, "unplaced": unplaced}, f)
    return scenes, unplaced

def _render_job(job):
    d = job["dir"]; meta_path = os.path.join(d, "scene.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("complete"):
            return meta
    os.makedirs(d, exist_ok=True)
    scene = json.loads(json.dumps(job["scene"]))
    frames = job.get("frames") or list(range(len(scene["cams"])))
    renders = {k: render_frame(scene, k) for k in frames}
    for k, r in renders.items():
        _save_frame(d, k, r)
    if job.get("knockout"):
        _save_frame(d, 1, render_frame(scene, 1, include_movers=False), tag="_ko")
    gt01 = mover_gt_stats(scene, renders, 0, 1)
    last = max(frames)
    gtlast = mover_gt_stats(scene, renders, 0, last) if last > 1 else gt01
    meta = dict(scene=scene_to_json(scene), gt01=gt01, gt0last=gtlast, set=job["set"], index=job["index"],
                frames=frames, dir=d, complete=True)
    if job.get("eval_frame") is not None:
        m = job["eval_frame"]
        meta["eval_frame"] = m
        meta["gt_eval"] = mover_gt_stats(scene, renders, m, m + 1)
    if scene.get("condition"):
        meta["condition"] = scene["condition"]
        meta["contact_px"] = [[contact_pixels(scene, renders[k], k, j) for k in (0, 1)]
                              for j in range(len(scene["movers"]))]
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta

def _render_flat_job(job):
    d = job["dir"]; meta_path = os.path.join(d, "scene.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    os.makedirs(d, exist_ok=True)
    out = render_legacy_flat(job["seed"])
    for k in range(2):
        Image.fromarray(out["rgb"][k]).save(os.path.join(d, f"frame{k}.png"))
    meta = dict(anchors=out["anchors"], cams=out["cams"], set="FLAT", index=job["index"], dir=d, complete=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta

def _abl_job(job):
    # one condition of one ablation base scene
    base = job["scene"]
    walls = []
    if job["cond"] in ("CH", "FH"):
        walls = [w for w in (design_wall(base, j) for j in range(len(base["movers"]))) if w is not None]
    lifts = [float(np.random.default_rng(seed_for("abl-lift", job["index"], j)).uniform(*ABLATION_DESIGN["lift_m"]))
             for j in range(len(base["movers"]))]
    sc = ablation_variant(base, job["cond"], walls, lifts)
    sc["walls"] = walls
    return _render_job(dict(set="ABL", index=job["index"], scene=sc, frames=[0, 1], dir=job["dir"]))

def _load_metas(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*", "scene.json"))):
        with open(d) as f:
            m = json.load(f)
        m["dir"] = os.path.dirname(d)                # restored checkpoints may come from another path
        out.append(m)
    return out

def _run_jobs(fn, jobs):
    with ThreadPoolExecutor(max_workers=RUN_CONFIG["render_workers"]) as ex:
        return list(ex.map(fn, jobs))

N_MAIN = RUN_CONFIG["n_scenes_main"]; M_PER = RUN_CONFIG["movers_per_scene"]
N_E4 = RUN_CONFIG["n_clips_e4_per_path"]; K_E4 = RUN_CONFIG["e4_n_frames"]
N_ABL = RUN_CONFIG["n_scenes_ablation"]
_base_cfg = dict(W=RUN_CONFIG["W"], H=RUN_CONFIG["H"], movers_per_scene=M_PER, mover=MOVER_DESIGN, camera=CAMERA_DESIGN)
RENDER_STAGES = {
    "MAIN": Stage("render:MAIN", "render", dict(_base_cfg, n=N_MAIN, ko=RUN_CONFIG["n_scenes_knockout"], frames=[0, 1])),
    "E4": Stage("render:E4", "render", dict(_base_cfg, n=N_E4, k=K_E4)),
    "FLAT": Stage("render:FLAT", "render", dict(n=RUN_CONFIG["n_scenes_flat"])),
    "ABL": Stage("render:ABL", "render", dict(_base_cfg, n=N_ABL, design=ABLATION_DESIGN, band=PREREG["contact_band_m"])),
}
RENDER_FP = {k: v.fp for k, v in RENDER_STAGES.items()}
_ = _scene_box_arrays(dict(boxes=[]))                    # initialise the texture-id table
_render_log = {}

with stage("render"):
    st = RENDER_STAGES["MAIN"]
    if not st.done():
        deck = make_design_deck(N_MAIN * M_PER, seed_for("deck-main"))
        specs, unplaced = _specs("MAIN", [seed_for("main", i) for i in range(N_MAIN)],
                                 [deck[i * M_PER:(i + 1) * M_PER] for i in range(N_MAIN)], 3, "straight", False)
        _run_jobs(_render_job, [dict(set="MAIN", index=i, scene=specs[i], frames=[0, 1],
                                     knockout=i < RUN_CONFIG["n_scenes_knockout"],
                                     dir=os.path.join(RENDER_DIR, "main", f"{i:04d}")) for i in range(N_MAIN)])
        st.commit([os.path.join(RENDER_DIR, "main"), os.path.join(RENDER_DIR, "specs_MAIN_straight.json")],
                  extra=dict(unplaced=len(unplaced)))
    release_memory(); mem_report("render:MAIN")
    st = RENDER_STAGES["E4"]
    if not st.done():
        deck = make_design_deck(N_E4 * M_PER, seed_for("deck-e4"), eps_targets={0.0: 1.0}, p_beyond=0.0)
        # straight/curved TWINS: same street, movers and frames 0-1; only the later path differs
        specs, unplaced = _specs("E4", [seed_for("e4", i) for i in range(N_E4)],
                                 [deck[i * M_PER:(i + 1) * M_PER] for i in range(N_E4)], K_E4, "twin", True)
        _run_jobs(_render_job, [dict(set="E4", index=pi * N_E4 + i, scene=specs[i][pi],
                                     dir=os.path.join(RENDER_DIR, "e4", f"{path}_{i:04d}"))
                                for pi, path in enumerate(["straight", "curved"]) for i in range(N_E4)])
        st.commit([os.path.join(RENDER_DIR, "e4"), os.path.join(RENDER_DIR, "specs_E4_twin.json")],
                  extra=dict(unplaced=len(unplaced)))
    release_memory(); mem_report("render:E4")
    st = RENDER_STAGES["FLAT"]
    if not st.done():
        _run_jobs(_render_flat_job, [dict(set="FLAT", index=i, seed=seed_for("flat", i),
                                          dir=os.path.join(RENDER_DIR, "flat", f"{i:04d}"))
                                     for i in range(RUN_CONFIG["n_scenes_flat"])])
        st.commit([os.path.join(RENDER_DIR, "flat")])
    st = RENDER_STAGES["ABL"]
    if not st.done():
        deck = make_design_deck(N_ABL * M_PER, seed_for("deck-abl"), classes=ABLATION_DESIGN["classes"],
                                eps_targets={0.0: 1.0}, p_beyond=0.0)
        specs, unplaced = _specs("ABL", [seed_for("abl", i) for i in range(N_ABL)],
                                 [deck[i * M_PER:(i + 1) * M_PER] for i in range(N_ABL)], 2, "straight", False)
        _run_jobs(_abl_job, [dict(index=i, scene=specs[i], cond=c, dir=os.path.join(RENDER_DIR, "abl", f"{i:04d}_{c}"))
                             for i in range(N_ABL) for c in ABLATION_DESIGN["conditions"]])
        st.commit([os.path.join(RENDER_DIR, "abl"), os.path.join(RENDER_DIR, "specs_ABL_straight.json")],
                  extra=dict(unplaced=len(unplaced)))
    release_memory(); mem_report("render:ABL")

META_MAIN = sorted(_load_metas(os.path.join(RENDER_DIR, "main")), key=lambda m: m["index"])
META_E4 = sorted(_load_metas(os.path.join(RENDER_DIR, "e4")), key=lambda m: m["index"])
META_FLAT = sorted(_load_metas(os.path.join(RENDER_DIR, "flat")), key=lambda m: m["index"])
META_ABL = sorted(_load_metas(os.path.join(RENDER_DIR, "abl")),
                  key=lambda m: (m["index"], ABLATION_DESIGN["conditions"].index(m["condition"])))
for _k, _l in [("MAIN", META_MAIN), ("E4", META_E4), ("FLAT", META_FLAT), ("ABL", META_ABL)]:
    assert len(_l) > 0, f"no {_k} renders"

def _mover_table(metas):
    rows = []
    for meta in metas:
        sc = meta["scene"]
        gev = meta.get("gt_eval") or [{}] * len(sc["movers"])
        for j, (mv, g01, gl) in enumerate(zip(sc["movers"], meta["gt01"], meta["gt0last"])):
            r = {k: v for k, v in mv.items() if k not in ("delta", "centroid0", "dims", "tint", "box_ids", "bbox0")}
            r.update({f"gt_{k}": v for k, v in g01.items() if k != "mover"})
            r.update({f"gtlast_{k}": v for k, v in gl.items() if k != "mover"})
            r.update({f"gtev_{k}": v for k, v in gev[j].items() if k != "mover"})
            r.update(set=meta["set"], scene=meta["index"], scene_seed=sc["seed"], cam_kind=sc["camera"]["kind"],
                     path=sc["camera"]["path"], focal=sc["camera"]["focal"], baseline=sc["camera"]["baseline"],
                     cam_psi_deg=sc["camera"]["psi_deg"], omega_deg=sc["camera"]["omega_deg"],
                     uid=f"{meta['set']}-{meta['index']:04d}-{mv['idx']}")
            if meta.get("condition"):
                cp = meta["contact_px"][j]
                r.update(condition=meta["condition"], contact_px0=cp[0], contact_px1=cp[1],
                         has_wall=any(w["mover"] == j for w in sc.get("walls", [])))
            rows.append(r)
    return pd.DataFrame(rows)

def _regimes(df, prefix="gt_"):
    df["x_par"] = np.where(df["s"] < 1, -np.log(np.clip(1 - df["s"], 1e-9, None)), np.nan)
    df["regime"] = np.where(df[prefix + "eps_med"] < PREREG["eps_parallel_max"], "parallel",
                            np.where(df[prefix + "eps_med"] >= PREREG["eps_perp_min"], "perpendicular", "intermediate"))
    df["measurable"] = (df[prefix + "n_eroded"] >= PREREG["min_object_pixels"]) & (df[prefix + "visible_frac"] >= 0.85)
    df["identifiable"] = df[prefix + "parallax_med"] >= PREREG["parallax_min_px"]
    return df

MOVERS_MAIN = _regimes(_mover_table(META_MAIN))
MOVERS_E4 = _regimes(_mover_table(META_E4))
MOVERS_E4["twin"] = MOVERS_E4["scene"] % N_E4
MOVERS_ABL = _regimes(_mover_table(META_ABL))
for _n, _df in [("main", MOVERS_MAIN), ("e4", MOVERS_E4), ("ablation", MOVERS_ABL)]:
    _df.to_csv(os.path.join(MANIFEST_DIR, f"convoy3d_{_n}_movers.csv"), index=False)

print(stage_table("render:").to_string(index=False))
print(f"\nMAIN: {len(MOVERS_MAIN)} movers in {len(META_MAIN)} clips (design deck {N_MAIN * M_PER})")
print(MOVERS_MAIN.groupby(["cls", "regime"]).size().unstack(fill_value=0).to_string())
print("\nby camera kind:", MOVERS_MAIN.groupby("cam_kind").size().to_dict(),
      "| motion types:", MOVERS_MAIN.groupby("motion_type").size().to_dict(),
      "| s > 1 (no static solution):", int((MOVERS_MAIN["s"] > 1).sum()))
print(f"measurable (>= {PREREG['min_object_pixels']} eroded px, >= 85% visible): "
      f"{int(MOVERS_MAIN['measurable'].sum())}/{len(MOVERS_MAIN)};  identifiable "
      f"(parallax >= {PREREG['parallax_min_px']} px): {int(MOVERS_MAIN['identifiable'].sum())}/{len(MOVERS_MAIN)}")
print(f"E4: {len(MOVERS_E4)} movers ({MOVERS_E4.groupby('path').size().to_dict()})")
_abl_g = MOVERS_ABL[MOVERS_ABL["condition"] == "G"]
print(f"ABL: {len(_abl_g)} base movers ({_abl_g.groupby('cls').size().to_dict()}) in {N_ABL} scenes x "
      f"{len(ABLATION_DESIGN['conditions'])} conditions; walls designed for "
      f"{int(MOVERS_ABL[MOVERS_ABL['condition'] == 'CH']['has_wall'].sum())} of them")

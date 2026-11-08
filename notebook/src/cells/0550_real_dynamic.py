# PLACEHOLDER
# 3.7 Real moving people (TUM fr3 walking): how often does real motion stay on its epipolar line?
#     A pixel is MOVING if its measured depth or its flow disagrees with the rigid motion implied
#     by the camera pose (|dZ| > max(0.10 m, 8%) or flow residual > 3 px), with forward-backward
#     flow consistency, away from depth edges, inside a connected region of >= 400 px. For every
#     moving pixel: eps (exact epipolar offset of its flow correspondence under the true pose)
#     and Z_tri (static triangulation of that correspondence). Only the prevalence of the motion
#     regimes is reported; the models are not evaluated on these pairs (Section 0.2, D24).

def select_dynamic_pairs(seq, n_pairs, rng, b_range=(0.05, 0.25), max_angle=20.0):
    pairs, n = [], len(seq)
    starts = np.linspace(0, n - 2, num=max(3 * n_pairs, 6)).astype(int)
    rng.shuffle(starts)
    for i in starts:
        target = rng.uniform(*b_range)
        for j in range(i + 1, min(n, i + 120)):
            b, ang = baseline_and_angle(seq, i, j)
            if b >= target:
                if ang <= max_angle and b <= b_range[1] + 0.05:
                    pairs.append((int(i), int(j), float(b), float(ang)))
                break
        if len(pairs) >= n_pairs:
            break
    return sorted(pairs)

def real_dynamic_ground_truth(seq, i, j, f_ij, f_ji):
    cam = seq.cam; H, W = cam.H, cam.W
    di, dj = seq.depth(i), seq.depth(j)
    Ri, Ci = seq.pose(i); Rj, Cj = seq.pose(j)
    vv, uu = np.mgrid[0:H, 0:W]
    p = np.stack([uu, vv], -1).reshape(-1, 2).astype(np.float64)
    z = di.reshape(-1).astype(np.float64)
    valid = np.isfinite(z)
    X = cam.rays(p[:, 0], p[:, 1]) * np.where(valid, z, 1.0)[:, None]
    Xj = (X @ Ri.T + Ci - Cj) @ Rj
    with np.errstate(divide="ignore", invalid="ignore"):
        p_rig = cam.project(Xj)
    p_obs = p + f_ij.reshape(-1, 2)
    zj_obs = sample_map(np.nan_to_num(dj, nan=0.0), p_rig)
    zj_obs[zj_obs <= 0] = np.nan
    back = sample_map(f_ji, p_obs)
    fb_ok = np.linalg.norm(f_ij.reshape(-1, 2) + back, axis=1) < 1.0 + 0.05 * np.linalg.norm(f_ij.reshape(-1, 2), axis=1)
    flow_res = np.linalg.norm(p_obs - p_rig, axis=1)
    dz = np.abs(zj_obs - Xj[:, 2])
    depth_bad = dz > np.maximum(0.10, 0.08 * Xj[:, 2])
    vimg = np.isfinite(di).astype(np.float32)
    nvalid = cv2.boxFilter(vimg, -1, (3, 3), normalize=False)
    gy, gx = np.gradient(np.nan_to_num(di, nan=0.0))
    edge = (((np.hypot(gx, gy) > 0.08) & (nvalid >= 9)) | (nvalid < 7)).reshape(-1)
    inside = np.isfinite(p_obs).all(1) & (p_obs[:, 0] >= 0) & (p_obs[:, 0] <= W - 1) & (p_obs[:, 1] >= 0) & (p_obs[:, 1] <= H - 1)
    base = valid & fb_ok & ~edge & inside & np.isfinite(zj_obs)
    dyn = base & (depth_bad | (flow_res > 3.0))
    dyn_img = cv2.morphologyEx(dyn.reshape(H, W).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(dyn_img, connectivity=8)
    keep = np.zeros(n_cc, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 400
    dyn = keep[lab].reshape(-1) & base
    R21, t21 = relative_pose(Ri, Ci, Rj, Cj)
    F = fundamental_matrix(cam.K, cam.K, R21, t21)
    idx = np.nonzero(dyn)[0]
    eps = np.abs(epipolar_offset(p[idx], p_obs[idx], F))
    ztri, _ = triangulate_depth(p[idx], p_obs[idx], cam.K, cam.K, R21, t21)
    par = translational_parallax(p[idx], p_obs[idx], cam.K, cam.K, R21)
    return dict(eps=eps.astype(np.float32), z_tri=ztri.astype(np.float32), parallax=par.astype(np.float32))

DYN_DIR = os.path.join(RENDER_DIR, "realdyn")
PREV_TUM_PATH = os.path.join(DYN_DIR, "prevalence_tum.csv")
DYN_PAIRS_PATH = os.path.join(DYN_DIR, "tum_dynamic_pairs.csv")
with stage("real_dynamic_gt"):
    _st = TUM_STAGES["DYN"]
    if not _st.done():
        os.makedirs(DYN_DIR, exist_ok=True)
        _rows, _prev = [], []
        for name, seq in TUM.items():
            if DATASET_STATUS[name]["role"] != "dynamic":
                continue
            rng_d = np.random.default_rng(seed_for("dyn-pairs", name))
            pairs = select_dynamic_pairs(seq, RUN_CONFIG["n_dyn_candidate_pairs_per_seq"], rng_d)
            imgs = {k: seq.rgb(k) for k in sorted({x for p_ in pairs for x in p_[:2]})}
            fw = compute_flows([(imgs[i], imgs[j]) for i, j, _, _ in pairs],
                               [os.path.join(FLOW_DIR, "dyn", f"{name}_{i}_{j}.npz") for i, j, _, _ in pairs])
            bw = compute_flows([(imgs[j], imgs[i]) for i, j, _, _ in pairs],
                               [os.path.join(FLOW_DIR, "dyn", f"{name}_{j}_{i}.npz") for i, j, _, _ in pairs])
            eps_all, zt_all = [], []
            for pk, ((i, j, b, ang), f1, f2) in enumerate(zip(pairs, fw, bw)):
                g = real_dynamic_ground_truth(seq, i, j, f1, f2)
                eps_all.append(g["eps"]); zt_all.append(g["z_tri"])
                _rows.append(dict(seq=name, index=pk, i=i, j=j, baseline=b, rot_deg=ang, n_dyn=int(len(g["eps"])),
                                  n_parallel=int(np.sum(g["eps"] < PREREG["eps_parallel_max"])),
                                  n_perp=int(np.sum(g["eps"] >= PREREG["eps_perp_min"]))))
            e, zt = np.concatenate(eps_all), np.concatenate(zt_all)
            _prev.append(dict(dataset="TUM fr3 walking", seq=name, n_pairs=len(pairs), n_moving_pixels=int(len(e)),
                              frac_parallel=float(np.mean(e < PREREG["eps_parallel_max"])),
                              frac_intermediate=float(np.mean((e >= PREREG["eps_parallel_max"]) & (e < PREREG["eps_perp_min"]))),
                              frac_perpendicular=float(np.mean(e >= PREREG["eps_perp_min"])),
                              frac_parallel_no_static_solution=float(np.mean(zt[e < 1] <= 0)) if np.any(e < 1) else float("nan"),
                              median_eps=float(np.median(e))))
            Image.fromarray(seq.rgb(len(seq) // 2)).save(os.path.join(DYN_DIR, f"thumb_{name}.png"))
            del imgs
        pd.DataFrame(_rows).to_csv(DYN_PAIRS_PATH, index=False)
        pd.DataFrame(_prev).to_csv(PREV_TUM_PATH, index=False)
        if all(DATASET_STATUS[n]["downloaded"] for n, v in TUM_SEQUENCES.items() if v["role"] == "dynamic"):
            _st.commit([DYN_DIR], extra=dict(flow=FLOW.get("method")))
        else:
            print("The walking ground truth is incomplete (a dynamic TUM sequence is missing); it is not checkpointed.")

PREVALENCE = pd.read_csv(PREV_TUM_PATH) if os.path.exists(PREV_TUM_PATH) else pd.DataFrame()
DYN_PAIRS = pd.read_csv(DYN_PAIRS_PATH) if os.path.exists(DYN_PAIRS_PATH) else pd.DataFrame()
TUM_THUMBS = {os.path.basename(p)[len("thumb_"):-4]: p for p in sorted(glob.glob(os.path.join(DYN_DIR, "thumb_*.png")))}
if len(PREVALENCE):
    print(PREVALENCE.drop(columns=["dataset"]).to_string(index=False, na_rep="–", float_format=lambda v: f"{v:.3f}"))
    PREVALENCE.to_csv(os.path.join(RESULTS_DIR, "prevalence_tum.csv"), index=False)

FLOW["model"] = None                                 # RAFT is not used after this cell
release_memory(); _ = mem_report("TUM walking prevalence")

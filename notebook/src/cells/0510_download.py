# PLACEHOLDER
# 3.4.1 TUM RGB-D: download and extract the four sequences (resumable, retried, size-checked).
#       The TUM data are needed only to build CONVOY-REAL and the walking ground truth; when both
#       of those stages are restored from a previous run's checkpoints, nothing is downloaded.
import urllib.request, tarfile

TUM_BASE = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/"
TUM_SEQUENCES = {
    "fr3_structure_texture_far":  dict(file="rgbd_dataset_freiburg3_structure_texture_far.tgz", role="static"),
    "fr3_long_office_household":  dict(file="rgbd_dataset_freiburg3_long_office_household.tgz", role="static"),
    "fr3_walking_xyz":            dict(file="rgbd_dataset_freiburg3_walking_xyz.tgz", role="dynamic"),
    "fr3_walking_halfsphere":     dict(file="rgbd_dataset_freiburg3_walking_halfsphere.tgz", role="dynamic"),
}
TUM_FR3_INTRINSICS = dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6, W=640, H=480)
TUM_DEPTH_SCALE = 5000.0

def download_file(url, dest, retries=4, timeout=120):
    # streaming download with HTTP range resume; returns (ok, message)
    tmp = dest + ".part"
    msg = "not attempted"
    for attempt in range(1, retries + 1):
        try:
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) + have if total is not None else None
                if have and resp.status != 206:          # server ignored the range: restart
                    have = 0
                mode = "ab" if have else "wb"
                with open(tmp, mode) as out:
                    while True:
                        chunk = resp.read(1 << 22)
                        if not chunk:
                            break
                        out.write(chunk)
            size = os.path.getsize(tmp)
            if total is not None and size != total:
                raise IOError(f"incomplete: {size} of {total} bytes")
            os.replace(tmp, dest)
            return True, f"{size / 2**20:.0f} MiB"
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            time.sleep(min(30, 5 * attempt))
    return False, msg

# stages built from the TUM data (Sections 3.5 and 3.7); identity of the data = sequence names and URLs
TUM_STAGE_CFG = dict(base=TUM_BASE, sequences={k: v["file"] for k, v in TUM_SEQUENCES.items()},
                     intrinsics=TUM_FR3_INTRINSICS, depth_scale=TUM_DEPTH_SCALE)
TUM_STAGES = {
    "REAL": Stage("tum:REAL", "tum", dict(TUM_STAGE_CFG, n_pairs=RUN_CONFIG["n_real_pairs_per_seq"],
                                          movers=RUN_CONFIG["real_movers_per_pair"], s_range=MOVER_DESIGN["s_range_main"])),
    "DYN": Stage("tum:DYN", "tum", dict(TUM_STAGE_CFG, n_pairs=RUN_CONFIG["n_dyn_candidate_pairs_per_seq"],
                                        regimes=[PREREG["eps_parallel_max"], PREREG["eps_perp_min"]], flow=FLOW_PREFERRED)),
    "RABL": Stage("tum:RABL", "tum", dict(TUM_STAGE_CFG, n_pairs=RUN_CONFIG["n_rabl_pairs_per_seq"],
                                          movers=RUN_CONFIG["rabl_movers_per_pair"], design=_jsonable(RABL_DESIGN),
                                          s_range=MOVER_DESIGN["s_range_main"], band=PREREG["rabl_contact_band_m"])),
}
TUM_NEEDED = any(st.available() is None for st in TUM_STAGES.values())

DATASET_STATUS = {}
with stage("download_tum"):
    for name, spec in TUM_SEQUENCES.items():
        seq_dir = os.path.join(DATA_DIR, name)
        done_flag = os.path.join(seq_dir, ".extracted")
        t0 = time.time()
        if not TUM_NEEDED:
            DATASET_STATUS[name] = dict(role=spec["role"], downloaded=False, restored=True, n_rgb=0, n_depth=0,
                                        groundtruth=False, note="every stage built from it is restored from a checkpoint; "
                                        "not downloaded", seconds=0.0, path=seq_dir)
            continue
        if not os.path.exists(done_flag):
            tgz = os.path.join(DATA_DIR, spec["file"])
            ok, msg = (True, "cached") if os.path.exists(tgz) else download_file(TUM_BASE + spec["file"], tgz)
            if ok:
                try:
                    os.makedirs(seq_dir, exist_ok=True)
                    with tarfile.open(tgz) as tf:
                        members = [m for m in tf.getmembers() if m.isfile() and not os.path.isabs(m.name)
                                   and ".." not in m.name.split("/")]
                        for m in members:
                            parts = m.name.split("/", 1)
                            if len(parts) < 2:
                                continue
                            m.name = parts[1]                   # strip the top-level folder
                            try:
                                tf.extract(m, seq_dir, filter="data")
                            except TypeError:                   # Python without extraction filters
                                tf.extract(m, seq_dir)
                    open(done_flag, "w").write("ok")
                    os.remove(tgz)                              # keep disk usage low
                except Exception as e:
                    ok, msg = False, f"extraction failed: {type(e).__name__}: {e}"
        else:
            ok, msg = True, "already extracted"
        n_rgb = len(glob.glob(os.path.join(seq_dir, "rgb", "*.png"))) if ok else 0
        n_dep = len(glob.glob(os.path.join(seq_dir, "depth", "*.png"))) if ok else 0
        has_gt = os.path.exists(os.path.join(seq_dir, "groundtruth.txt"))
        ok = ok and n_rgb > 100 and n_dep > 100 and has_gt
        DATASET_STATUS[name] = dict(role=spec["role"], downloaded=bool(ok), restored=False, n_rgb=n_rgb, n_depth=n_dep,
                                    groundtruth=has_gt, note=msg, seconds=round(time.time() - t0, 1), path=seq_dir)

print(f"{'sequence':<30}{'role':<9}{'status':<10}{'rgb':>6}{'depth':>7}  note")
for k, v in DATASET_STATUS.items():
    status = "RESTORED" if v["restored"] else ("PASS" if v["downloaded"] else "FAIL")
    print(f"{k:<30}{v['role']:<9}{status:<10}{v['n_rgb']:>6}{v['n_depth']:>7}  {v['note']}")
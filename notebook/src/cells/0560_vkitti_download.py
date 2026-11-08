# PLACEHOLDER
# 3.8 Virtual KITTI 2 (Cabon et al., 2020): photo-realistic driving with exact depth, camera
#     extrinsics and per-vehicle 3-D poses, downloadable without login. The image archives are
#     several gigabytes each, so only the frames this run uses are fetched: the compact text
#     ground truth (vehicle and camera poses) is read first, the frame pairs are chosen
#     from it, and the matching image, depth and instance files are read from the remote tar
#     archives with HTTP range requests (one streamed pass over an archive is the fallback when
#     the server ignores ranges). The stage is one checkpoint. If any part fails, Virtual KITTI 2
#     is removed from every experiment, table and figure.
import tarfile as _tarfile
import urllib.error

VK_FILES = {"textgt": "vkitti_2.0.3_textgt.tar.gz", "rgb": "vkitti_2.0.3_rgb.tar", "depth": "vkitti_2.0.3_depth.tar",
            "instance": "vkitti_2.0.3_instanceSegmentation.tar"}
VK_SCENES = VK_DESIGN["scenes"][:RUN_CONFIG["n_vk_scenes"]]
VK_RAW = os.path.join(VK_DIR, "raw")
_VK_MEMBER = re.compile(r"(?:^|/)(Scene\d\d)/([A-Za-z0-9_-]+)/frames/([A-Za-z]+)/Camera_(\d)/[A-Za-z]+_(\d{5})\.(jpg|png)$")


class RangeUnsupported(Exception):
    pass


class RemoteTar:
    """Random access to an uncompressed tar archive over HTTP (range requests)."""

    def __init__(self, url, timeout=90, retries=4):
        self.url, self.timeout, self.retries = url, timeout, retries
        self.n_requests, self.n_bytes = 0, 0
        data, total = self._get(0, 511, want_total=True)
        self.size = total

    def _get(self, start, end, want_total=False):
        err = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if r.status != 206:
                        raise RangeUnsupported(f"HTTP {r.status} to a range request")
                    data = r.read()
                    cr = r.headers.get("Content-Range", "")
                self.n_requests += 1; self.n_bytes += len(data)
                if want_total:
                    return data, int(cr.rsplit("/", 1)[1])
                return data
            except RangeUnsupported:
                raise
            except Exception as e:
                err = e
                time.sleep(2 + 3 * attempt)
        raise IOError(f"range request failed: {type(err).__name__}: {err}")

    @staticmethod
    def parse_header(block):
        if len(block) < 512 or block[257:262] != b"ustar":
            return None
        chk = int(block[148:156].split(b"\0", 1)[0].strip() or b"0", 8)
        if chk != sum(block[:148]) + 8 * 32 + sum(block[156:512]):
            return None
        name = block[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
        prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
        if prefix:
            name = prefix + "/" + name
        sz = block[124:136]
        size = int.from_bytes(sz[1:], "big") if sz[0] & 0x80 else int(sz.split(b"\0", 1)[0].strip() or b"0", 8)
        return dict(name=name, size=size, type=chr(block[156]) if block[156] else "0")

    def header_at_or_after(self, offset, window=1 << 18, max_windows=24):
        o = max(0, offset - offset % 512)
        for _ in range(max_windows):
            if o >= self.size:
                return None
            buf = self._get(o, min(o + window, self.size) - 1)
            for b in range(0, len(buf) - 511, 512):
                h = self.parse_header(buf[b:b + 512])
                if h and h["type"] in "0\0":
                    return o + b, h
            o += window
        return None

    def next_header(self, off, h):
        return off + 512 + (h["size"] + 511) // 512 * 512

    def iter_members(self, start, end):
        # stream [start, end) and yield (name, bytes) of the regular files in it
        req = urllib.request.Request(self.url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status != 206:
                raise RangeUnsupported(f"HTTP {r.status} to a range request")
            yield from _iter_tar_stream(r)


def _read_exact(f, n):
    out = bytearray()
    while len(out) < n:
        chunk = f.read(n - len(out))
        if not chunk:
            break
        out += chunk
    return bytes(out)


def _iter_tar_stream(f):
    long_name = None
    while True:
        block = _read_exact(f, 512)
        if len(block) < 512 or block == b"\0" * 512:
            return
        h = RemoteTar.parse_header(block)
        if h is None:
            return
        data = _read_exact(f, h["size"])
        _read_exact(f, (512 - h["size"] % 512) % 512)
        if h["type"] == "L":
            long_name = data.split(b"\0", 1)[0].decode("utf-8", "replace"); continue
        if h["type"] in ("x", "g"):
            for rec in data.decode("utf-8", "replace").split("\n"):
                if " path=" in rec:
                    long_name = rec.split(" path=", 1)[1]
            continue
        name = long_name or h["name"]; long_name = None
        if h["type"] in "0\0":
            yield name, data


def _member_key(name):
    m = _VK_MEMBER.search(name)
    if not m:
        return None
    scene, variation, folder, cam, frame, _ = m.groups()
    return scene, variation, folder, int(cam), int(frame)


def vk_fetch_frames(url, wanted, label, log):
    # wanted: {(scene, frame)} for variation clone / camera 0 -> {(scene, frame): bytes}
    out = {}
    need_scenes = sorted({s for s, _ in wanted})
    try:
        rt = RemoteTar(url)
        # coarse probes: which directory lives where in the archive
        n_probe = int(np.clip(rt.size // (24 << 20), 64, 480))
        probes = []
        for off in np.linspace(0, max(rt.size - 1024, 0), n_probe).astype(np.int64):
            hit = rt.header_at_or_after(int(off))
            if hit:
                probes.append((hit[0], _member_key(hit[1]["name"]), hit[1]))
        probes = sorted({p[0]: p for p in probes}.values())

        def in_dir(key, scene):
            return key is not None and key[0] == scene and key[1] == VK_DESIGN["variation"] and key[3] == VK_DESIGN["camera"]

        for scene in need_scenes:
            idx = [i for i, p in enumerate(probes) if in_dir(p[1], scene)]
            if not idx:
                raise RangeUnsupported(f"{scene} not found by the probes")
            # start: bisect between the last probe before the directory and its first probe
            a = probes[idx[0] - 1][0] if idx[0] > 0 else 0
            b = probes[idx[0]][0]
            while b - a > (1 << 18):
                mid = (a + b) // 2
                hit = rt.header_at_or_after(mid)
                if hit is None or hit[0] >= b:
                    a = mid
                elif in_dir(_member_key(hit[1]["name"]), scene):
                    b = hit[0]
                else:
                    a = hit[0]
            off, h = rt.header_at_or_after(a)
            while not in_dir(_member_key(h["name"]), scene):
                off = rt.next_header(off, h)
                h = rt.parse_header(rt._get(off, off + 511)) if off + 512 <= rt.size else None
                if h is None:
                    raise IOError(f"{scene}: tar walk ended before the directory")
            start = off
            # end: bisect between the last probe inside and the first probe after the directory
            a = probes[idx[-1]][0]
            b = probes[idx[-1] + 1][0] if idx[-1] + 1 < len(probes) else rt.size
            while b - a > (1 << 18):
                mid = (a + b) // 2
                hit = rt.header_at_or_after(mid)
                if hit is None or hit[0] >= b:
                    b = mid
                elif in_dir(_member_key(hit[1]["name"]), scene):
                    a = hit[0]
                else:
                    b = hit[0]
            off, h = rt.header_at_or_after(a)
            while h is not None and in_dir(_member_key(h["name"]), scene):
                off = rt.next_header(off, h)
                h = rt.parse_header(rt._get(off, off + 511)) if off + 512 <= rt.size else None
            end = off
            todo = {f for s_, f in wanted if s_ == scene}
            for name, data in rt.iter_members(start, end):
                key = _member_key(name)
                if in_dir(key, scene) and key[4] in todo:
                    out[(scene, key[4])] = data
                    todo.discard(key[4])
                    if not todo:
                        break                               # every wanted frame of this scene is in hand
        log[label] = dict(mode="range requests", requests=rt.n_requests, MiB=round(rt.n_bytes / 2**20, 1))
    except (RangeUnsupported, IOError, ValueError, TypeError) as e:
        # fallback: one streamed pass over the whole archive, stopping when everything is found
        log[label] = dict(mode=f"streamed ({type(e).__name__}: {str(e)[:80]})")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            for name, data in _iter_tar_stream(r):
                key = _member_key(name)
                if key and key[1] == VK_DESIGN["variation"] and key[3] == VK_DESIGN["camera"] and (key[0], key[4]) in wanted:
                    out[(key[0], key[4])] = data
                    if len(out) == len(wanted):
                        break
    return out


# ---------------- text ground truth ----------------
VK_COLUMNS = {   # documented column order of the Virtual KITTI 2 text ground truth (header row first)
    "extrinsic": ["frame", "cameraID"] + [f"m{r}{c}" for r in range(4) for c in range(4)],
    "intrinsic": ["frame", "cameraID", "fx", "fy", "cx", "cy"],
    "pose": ["frame", "cameraID", "trackID", "alpha", "width", "height", "length", "world_space_X", "world_space_Y",
             "world_space_Z", "rotation_world_space_y", "rotation_world_space_x", "rotation_world_space_z",
             "camera_space_X", "camera_space_Y", "camera_space_Z", "rotation_camera_space_y",
             "rotation_camera_space_x", "rotation_camera_space_z"],
    "bbox": ["frame", "cameraID", "trackID", "left", "right", "top", "bottom", "number_pixels", "truncation_ratio",
             "occupancy_ratio", "isMoving"],
    "info": ["trackID", "label", "model", "color"],
}

def _read_vk_table(text, kind):
    lines = [l.split() for l in text.splitlines() if l.strip()]
    head, rows = lines[0], lines[1:]
    cols = VK_COLUMNS[kind]
    if head[0] not in ("frame", "trackID") or any(len(r) < len(cols) for r in rows):
        raise ValueError(f"{kind}.txt: unexpected layout (header {head[:4]})")
    df = pd.DataFrame([r[:len(cols)] for r in rows], columns=cols)
    for c in cols:
        if c not in ("label", "model", "color", "isMoving"):
            df[c] = pd.to_numeric(df[c])
    return df


def vk_parse_textgt(scene_dir):
    tg = {}
    for k in ("extrinsic", "intrinsic", "pose", "bbox", "info"):
        with open(os.path.join(scene_dir, f"{k}.txt")) as f:
            tg[k] = _read_vk_table(f.read(), k)
    for k in ("extrinsic", "intrinsic", "pose", "bbox"):
        tg[k] = tg[k][tg[k]["cameraID"].astype(int) == VK_DESIGN["camera"]].reset_index(drop=True)
    tg["bbox"]["isMoving"] = tg["bbox"]["isMoving"].astype(str).str.lower().isin(["true", "1"])
    return tg


def vk_camera_centres(tg, inverted=False):
    # camera centres C_k (world) for camera 0, assuming extrinsic = world-to-camera (or its inverse)
    E = tg["extrinsic"].sort_values("frame")
    M = E[[f"m{r}{c}" for r in range(4) for c in range(4)]].to_numpy(np.float64).reshape(-1, 4, 4)
    Rs, Cs = [], []
    for T in M:
        if inverted:
            T = np.linalg.inv(T)
        Rwc = T[:3, :3]; t = T[:3, 3]
        Rs.append(Rwc.T); Cs.append(-Rwc.T @ t)
    return E["frame"].to_numpy(int), np.array(Rs), np.array(Cs)


def vk_select(tg, scene, rng):
    # frame pairs chosen from the text ground truth alone (before any image is read)
    frames, Rs, Cs = vk_camera_centres(tg)
    fidx = {f: n for n, f in enumerate(frames)}
    W_full = int(tg["intrinsic"]["cx"].iloc[0] * 2 + 1)
    x0 = int(round(tg["intrinsic"]["cx"].iloc[0] - (VK_DESIGN["crop_wh"][0] - 1) / 2))
    bb = tg["bbox"]
    bb = bb.assign(cl=np.clip(bb["left"], x0, x0 + VK_DESIGN["crop_wh"][0] - 1),
                   cr=np.clip(bb["right"], x0, x0 + VK_DESIGN["crop_wh"][0] - 1))
    bb["area_crop"] = (bb["cr"] - bb["cl"]).clip(lower=0) * (bb["bottom"] - bb["top"]).clip(lower=0)
    mov = bb[bb["isMoving"] & (bb["area_crop"] >= VK_DESIGN["min_vehicle_px"])]
    frames_with_movers = set(mov["frame"].astype(int))
    def rot_deg(i, j):
        R = Rs[fidx[i]].T @ Rs[fidx[j]]
        return float(np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    def base(i, j):
        return float(np.linalg.norm(Cs[fidx[j]] - Cs[fidx[i]]))
    starts = [int(f) for f in frames if int(f) in frames_with_movers]
    rng.shuffle(starts)
    pairs, used = [], set()
    for i in starts:
        target = rng.uniform(*VK_DESIGN["pair_baseline_m"])
        for j in range(i + 1, i + VK_DESIGN["max_gap_frames"] + 1):
            if j not in fidx:
                break
            if base(i, j) >= target:
                if base(i, j) <= VK_DESIGN["pair_baseline_m"][1] + 0.5 and rot_deg(i, j) <= VK_DESIGN["max_rot_deg"] \
                   and not any(abs(i - u) < 3 for u in used):
                    pairs.append((int(i), int(j), base(i, j), rot_deg(i, j))); used.add(i)
                break
        if len(pairs) >= RUN_CONFIG["vk_pairs_per_scene"]:
            break
    return sorted(pairs), dict(x0=x0, width_full=W_full)


VK_STAGE = Stage("vk:data", "vk", dict(files=VK_FILES, scenes=VK_SCENES, design={k: v for k, v in VK_DESIGN.items() if k != "base_url"},
                                       pairs=RUN_CONFIG["vk_pairs_per_scene"]))
VK_SELECTION_PATH = os.path.join(VK_RAW, "selection.json")
VK_STATUS = dict(available=False, note="", download_log={})
with stage("download_vkitti"):
    if VK_STAGE.done():
        VK_STATUS.update(available=True, note=f"restored from checkpoint ({STAGE_LOG[-1]['source']})")
    else:
        try:
            shutil.rmtree(VK_RAW, ignore_errors=True); os.makedirs(VK_RAW, exist_ok=True)
            tgz = os.path.join(DATA_DIR, VK_FILES["textgt"])
            ok, msg = (True, "cached") if os.path.exists(tgz) else download_file(VK_DESIGN["base_url"] + VK_FILES["textgt"], tgz)
            if not ok:
                raise IOError(f"text ground truth: {msg}")
            pat = re.compile(r"(?:^|/)(Scene\d\d)/" + VK_DESIGN["variation"] + r"/(extrinsic|intrinsic|pose|bbox|info)\.txt$")
            with _tarfile.open(tgz) as tf:
                for m in tf.getmembers():
                    mm = pat.search(m.name)
                    if m.isfile() and mm and mm.group(1) in VK_SCENES:
                        d = os.path.join(VK_RAW, mm.group(1)); os.makedirs(d, exist_ok=True)
                        with tf.extractfile(m) as src, open(os.path.join(d, mm.group(2) + ".txt"), "wb") as out:
                            out.write(src.read())
            selection, wanted = {}, set()
            for scene in VK_SCENES:
                sd = os.path.join(VK_RAW, scene)
                if not all(os.path.exists(os.path.join(sd, f"{k}.txt")) for k in ("extrinsic", "intrinsic", "pose", "bbox", "info")):
                    raise IOError(f"{scene}: text ground truth incomplete")
                tg = vk_parse_textgt(sd)
                pairs, geo = vk_select(tg, scene, np.random.default_rng(seed_for("vk-select", scene)))
                selection[scene] = dict(pairs=pairs, **geo)
                wanted |= {(scene, f) for p in pairs for f in p[:2]}
            log = {}
            for kind, prefix, ext in (("rgb", "rgb", "jpg"), ("depth", "depth", "png"), ("instance", "inst", "png")):
                got = vk_fetch_frames(VK_DESIGN["base_url"] + VK_FILES[kind], wanted, kind, log)
                missing = wanted - set(got)
                if missing:
                    raise IOError(f"{kind}: {len(missing)} of {len(wanted)} frames not found in the archive")
                for (scene, f), data in got.items():
                    with open(os.path.join(VK_RAW, scene, f"{prefix}_{f:05d}.{ext}"), "wb") as out:
                        out.write(data)
            with open(VK_SELECTION_PATH, "w") as f:
                json.dump(dict(selection=selection, download_log=log), f, indent=1,
                          default=lambda o: o.item() if hasattr(o, "item") else str(o))
            VK_STAGE.commit([VK_RAW])
            VK_STATUS.update(available=True, note="downloaded", download_log=log)
        except Exception as e:
            VK_STATUS.update(available=False, note=f"{type(e).__name__}: {str(e)[:300]}")
            shutil.rmtree(VK_RAW, ignore_errors=True)

if VK_STATUS["available"]:
    with open(VK_SELECTION_PATH) as f:
        _sel = json.load(f)
    VK_SELECTION = _sel["selection"]
    VK_STATUS["download_log"] = _sel.get("download_log", {})
    print(f"Virtual KITTI 2: PASS ({VK_STATUS['note']})")
    for k, v in VK_STATUS["download_log"].items():
        print(f"    {k:<9} {v}")
    for sc_, v in VK_SELECTION.items():
        print(f"    {sc_}: {len(v['pairs'])} pairs")
else:
    VK_SELECTION = {}
    print(f"Virtual KITTI 2: FAIL -> removed from every experiment, table and figure ({VK_STATUS['note']})")
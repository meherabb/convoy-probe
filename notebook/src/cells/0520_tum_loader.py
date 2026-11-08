# PLACEHOLDER
# 3.4.2 TUM loader: timestamp association (|dt| <= 20 ms), pose interpolation from the
#       100 Hz motion-capture trajectory, and the ground-truth self-consistency test (A3).

def _read_tum_list(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            rows.append((float(p[0]), p[1]))
    return rows

def _read_tum_gt(path):
    a = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            a.append([float(v) for v in line.split()[:8]])
    return np.asarray(a, np.float64)

def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw); qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

def _slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)

class TUMSequence:
    def __init__(self, name, root, max_dt=0.02, max_gt_gap=0.05):
        self.name, self.root = name, root
        rgb = _read_tum_list(os.path.join(root, "rgb.txt"))
        dep = _read_tum_list(os.path.join(root, "depth.txt"))
        gt = _read_tum_gt(os.path.join(root, "groundtruth.txt"))
        self.cam = PinholeCamera(**TUM_FR3_INTRINSICS)
        dt = np.array([d[0] for d in dep]); gtt = gt[:, 0]
        self.frames = []
        for t, f in rgb:
            j = int(np.argmin(np.abs(dt - t)))
            if abs(dt[j] - t) > max_dt:
                continue
            k = int(np.searchsorted(gtt, t))
            if k == 0 or k >= len(gtt) or gtt[k] - gtt[k - 1] > max_gt_gap:
                continue
            a = (t - gtt[k - 1]) / (gtt[k] - gtt[k - 1])
            C = (1 - a) * gt[k - 1, 1:4] + a * gt[k, 1:4]
            q = _slerp(gt[k - 1, [4, 5, 6, 7]], gt[k, [4, 5, 6, 7]], a)
            self.frames.append(dict(t=t, rgb=os.path.join(root, f), depth=os.path.join(root, dep[j][1]),
                                    R=quat_to_R(*q), C=C))
        self.pose_inverted = False

    def __len__(self):
        return len(self.frames)

    def rgb(self, i):
        return np.asarray(Image.open(self.frames[i]["rgb"]).convert("RGB"))

    def depth(self, i):
        d = np.asarray(Image.open(self.frames[i]["depth"])).astype(np.float32) / TUM_DEPTH_SCALE
        d[d <= 0] = np.nan
        return d

    def pose(self, i):
        R, C = self.frames[i]["R"], self.frames[i]["C"]
        if self.pose_inverted:                      # only if the convention test demands it
            return R.T, -R.T @ C
        return R, C

def depth_warp_error(seq, i, j, stride=4, inverted=None):
    # median relative disagreement between frame j's measured depth and frame i's depth
    # warped into frame j with the ground-truth relative pose (static pixels dominate)
    if inverted is not None:
        old = seq.pose_inverted; seq.pose_inverted = inverted
    cam = seq.cam
    Ri, Ci = seq.pose(i); Rj, Cj = seq.pose(j)
    if inverted is not None:
        seq.pose_inverted = old
    di, dj = seq.depth(i), seq.depth(j)
    vv, uu = np.mgrid[0:cam.H:stride, 0:cam.W:stride]
    z = di[vv, uu]; ok = np.isfinite(z)
    Xc = cam.rays(uu[ok], vv[ok]) * z[ok][:, None]
    P = Xc @ Ri.T + Ci
    Xj = (P - Cj) @ Rj
    front = Xj[:, 2] > 0.1
    p = cam.project(Xj[front])
    inside = (p[:, 0] >= 0) & (p[:, 0] <= cam.W - 1) & (p[:, 1] >= 0) & (p[:, 1] <= cam.H - 1)
    zj = Xj[front][inside, 2]
    zo = dj[np.round(p[inside, 1]).astype(int), np.round(p[inside, 0]).astype(int)]
    m = np.isfinite(zo)
    if m.sum() < 200:
        return np.nan
    return float(np.median(np.abs(zo[m] - zj[m]) / zo[m]))

def baseline_and_angle(seq, i, j):
    Ri, Ci = seq.pose(i); Rj, Cj = seq.pose(j)
    ang = np.rad2deg(np.arccos(np.clip((np.trace(Ri.T @ Rj) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(Cj - Ci)), float(ang)

TUM = {}
A3_VALIDATION = {}
A3_PATH = os.path.join(LOG_DIR, "a3_validation.json")
if not TUM_NEEDED:
    for _st in TUM_STAGES.values():                  # restores the composites, walking GT and the A3 record
        assert _st.done(), f"{_st.name}: checkpoint vanished"
    with open(A3_PATH) as f:
        A3_VALIDATION = json.load(f)
for name, st in DATASET_STATUS.items():
    if not st["downloaded"]:
        continue
    seq = TUMSequence(name, st["path"])
    rng_v = np.random.default_rng(seed_for("a3", name))
    pairs = []
    for _ in range(400):
        i = int(rng_v.integers(0, len(seq) - 1))
        for j in range(i + 1, min(len(seq), i + 90)):
            b, ang = baseline_and_angle(seq, i, j)
            if b >= 0.08:
                if ang < 20:
                    pairs.append((i, j))
                break
        if len(pairs) >= 16:
            break
    errA = np.nanmedian([depth_warp_error(seq, i, j, inverted=False) for i, j in pairs]) if pairs else np.nan
    errB = np.nanmedian([depth_warp_error(seq, i, j, inverted=True) for i, j in pairs]) if pairs else np.nan
    seq.pose_inverted = bool(np.isfinite(errB) and errB < 0.5 * errA)
    err = errB if seq.pose_inverted else errA
    passed = bool(np.isfinite(err) and err <= 0.05)
    A3_VALIDATION[name] = dict(n_frames=len(seq), n_test_pairs=len(pairs), warp_err_documented=float(errA),
                               warp_err_inverted=float(errB), convention="inverted" if seq.pose_inverted else "documented",
                               median_rel_depth_err=float(err), passed=passed)
    st["validated"] = passed
    if passed:
        TUM[name] = seq

print(f"{'sequence':<28}{'frames':>7}{'pairs':>6}{'warp err (doc.)':>17}{'(inverted)':>12}  A3")
for k, v in A3_VALIDATION.items():
    print(f"{k:<28}{v['n_frames']:>7}{v['n_test_pairs']:>6}{v['warp_err_documented']:>17.3f}"
          f"{v['warp_err_inverted']:>12.3f}  {'PASS' if v['passed'] else 'FAIL'} ({v['convention']} convention)")
print("\nA3 criterion (pre-registered): median relative disagreement <= 5% between measured depth and "
      "depth warped with the motion-capture pose. Dynamic sequences pass on their static majority.")
if TUM_NEEDED:
    with open(A3_PATH, "w") as f:
        json.dump(A3_VALIDATION, f, indent=2)
else:
    print("(A3 record restored from the checkpoint of the stages built from these sequences)")
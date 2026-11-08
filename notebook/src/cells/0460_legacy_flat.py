# PLACEHOLDER
# 3.1.6 LEGACY-FLAT control: the compositing testbed of the previous notebook version,
#       reproduced exactly (smooth background pasted identically into both views, seven
#       static marker discs at their true projections, one 2-D object sprite). It is kept
#       only as a negative control: the static-scene validity gate (Section 6.0) must reject it.
from PIL import Image, ImageDraw

LEGACY_CAM = PinholeCamera(700.0, 700.0, 320.0, 240.0, 640, 480)

def _legacy_background(H, W, rng):
    lo = rng.integers(2, 6)
    base = rng.uniform(40, 200, size=(lo, lo, 3)).astype(np.float32)
    img = Image.fromarray(base.astype(np.uint8)).resize((W, H), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) + rng.normal(0, 2, size=(H, W, 3))
    return np.clip(arr, 0, 255).astype(np.uint8)

def _legacy_anchor(draw, uv, color, marker, r=6):
    c = tuple(int(v) for v in color); x, y = uv
    if marker == "dot":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
    elif marker == "cross":
        draw.line([x - r, y, x + r, y], fill=c, width=3); draw.line([x, y - r, x, y + r], fill=c, width=3)
    else:
        draw.ellipse([x - r, y - r, x + r, y + r], outline=c, width=3)

def render_legacy_flat(seed):
    rng = np.random.default_rng(seed)
    cam = LEGACY_CAM
    yaw = rng.uniform(-0.03, 0.03)
    R1 = rot_y(yaw); C1 = np.array([0.0, 0.0, rng.uniform(0.05, 0.30)])
    bg = _legacy_background(cam.H, cam.W, rng)
    im0, im1 = Image.fromarray(bg.copy()), Image.fromarray(bg.copy())
    d0, d1 = ImageDraw.Draw(im0), ImageDraw.Draw(im1)
    anchors = []
    for _ in range(7):
        X = np.array([rng.uniform(-18, 18) * rng.uniform(0.3, 1.0), rng.uniform(-4, 4), rng.uniform(4, 90)])
        col = rng.uniform(30, 230, 3); mk = rng.choice(["dot", "cross", "ring"])
        uv0 = cam.project(X); uv1 = cam.project(world_to_cam(X, R1, C1))
        in0 = 6 <= uv0[0] < cam.W - 6 and 6 <= uv0[1] < cam.H - 6
        in1 = 6 <= uv1[0] < cam.W - 6 and 6 <= uv1[1] < cam.H - 6
        if in0:
            _legacy_anchor(d0, uv0, col, mk)
        if in1:
            _legacy_anchor(d1, uv1, col, mk)
        if in0 and in1:
            anchors.append(dict(u=float(uv0[0]), v=float(uv0[1]), Z=float(X[2])))
    # one 2-D sprite moving parallel to the baseline (as in the old testbed)
    Zo = rng.uniform(6, 55); s = rng.uniform(-0.5, 0.9)
    Xo = np.array([rng.uniform(-0.3, 0.3) * Zo, rng.uniform(-0.1, 0.25) * Zo, Zo])
    for im, (R, C, off) in ((d0, (np.eye(3), np.zeros(3), 0.0)), (d1, (R1, C1, 1.0))):
        uv = cam.project(world_to_cam(Xo + off * s * (C1 - 0), R, C))
        half = max(4.0, cam.fx * 0.9 / Zo)
        im.rounded_rectangle([uv[0] - half, uv[1] - 0.62 * half, uv[0] + half, uv[1] + 0.62 * half],
                             radius=0.15 * half, fill=tuple(int(v) for v in rng.uniform(60, 220, 3)))
    return dict(rgb=[np.asarray(im0), np.asarray(im1)], anchors=anchors,
                cams=[dict(**cam.to_dict(), R=np.eye(3).tolist(), C=[0.0, 0.0, 0.0]),
                      dict(**cam.to_dict(), R=R1.tolist(), C=C1.tolist())])

print("LEGACY-FLAT generator ready (reproduces the previous testbed for the gate control).")
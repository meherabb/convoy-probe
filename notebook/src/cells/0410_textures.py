# PLACEHOLDER
# 3.1.1 Procedural textures with mip-map pyramids and a trilinear, footprint-aware sampler.
#       Textures are generated once from a fixed seed, so every run sees identical images.

def _tileable_noise(rng, size, base=4, octaves=6, persistence=0.55, aspect=(1, 1)):
    # zero-mean, unit-std fractal value noise that tiles seamlessly (size = (H, W) or int)
    H, W = (size, size) if np.isscalar(size) else size
    out = np.zeros((H, W), np.float32); amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh, gw = int(base * aspect[0] * 2 ** o), int(base * aspect[1] * 2 ** o)
        if gh > H or gw > W:
            break
        grid = rng.random((gh, gw)).astype(np.float32)
        big = np.tile(grid, (3, 3))
        up = cv2.resize(big, (3 * W, 3 * H), interpolation=cv2.INTER_CUBIC)
        out += amp * up[H:2 * H, W:2 * W]
        tot += amp; amp *= persistence
    out /= max(tot, 1e-6)
    return (out - out.mean()) / (out.std() + 1e-6)

def _mip_pyramid(img):
    pyr = [img.astype(np.float32)]
    while min(pyr[-1].shape[:2]) > 1:
        a = pyr[-1]
        h2, w2 = a.shape[0] // 2, a.shape[1] // 2
        a = a[:2 * h2, :2 * w2]
        pyr.append(0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]))
    return pyr

TEXTURES = {}

def _register(name, img, tile_m, wrap=(True, True)):
    # tile_m: physical width (metres) of the texture image
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    TEXTURES[name] = dict(pyr=_mip_pyramid(img), texel=float(tile_m) / img.shape[1], wrap=wrap,
                          shape=img.shape)

def _bilinear(img, x, y, wrap):
    H, W = img.shape[:2]
    x = x - 0.5; y = y - 0.5
    x0 = np.floor(x); y0 = np.floor(y)
    fx = (x - x0).astype(np.float32)[:, None]; fy = (y - y0).astype(np.float32)[:, None]
    x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)
    if wrap[0]:
        x0 %= W; x1 = (x0 + 1) % W
    else:
        x1 = np.clip(x0 + 1, 0, W - 1); x0 = np.clip(x0, 0, W - 1)
    if wrap[1]:
        y0 %= H; y1 = (y0 + 1) % H
    else:
        y1 = np.clip(y0 + 1, 0, H - 1); y0 = np.clip(y0, 0, H - 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)

def tex_sample(name, u_m, v_m, footprint_m):
    # trilinear lookup; u_m, v_m texture coordinates in metres, footprint_m metres per pixel
    T = TEXTURES[name]; pyr = T["pyr"]; texel = T["texel"]
    n_lev = len(pyr)
    lev = np.log2(np.maximum(np.asarray(footprint_m, np.float64) / texel, 1.0))
    lev = np.minimum(lev, n_lev - 1)
    l0 = np.floor(lev).astype(np.int32)
    fr = (lev - l0).astype(np.float32)
    out = np.empty((u_m.shape[0], pyr[0].shape[2]), np.float32)
    for L in np.unique(l0):
        sel = np.nonzero(l0 == L)[0]
        sc = texel * (2 ** L)
        c0 = _bilinear(pyr[L], u_m[sel] / sc, v_m[sel] / sc, T["wrap"])
        if L + 1 < n_lev:
            c1 = _bilinear(pyr[L + 1], u_m[sel] / (2 * sc), v_m[sel] / (2 * sc), T["wrap"])
            f = fr[sel, None]
            out[sel] = c0 * (1 - f) + c1 * f
        else:
            out[sel] = c0
    return out

def _build_textures(seed):
    rng = np.random.default_rng(seed)
    N = 512
    # asphalt: fine aggregate + patches
    n1, n2, n3 = _tileable_noise(rng, N, base=32), _tileable_noise(rng, N, base=4), _tileable_noise(rng, N, base=64)
    n4 = _tileable_noise(rng, N, base=12)
    g = 0.30 + 0.034 * n1 + 0.010 * n2 + 0.014 * n4 + 0.05 * (n3 > 2.2) - 0.04 * (n3 < -2.3)
    _register("asphalt", np.stack([g * 1.02, g * 1.0, g * 0.97], -1), tile_m=4.0)
    # paving slabs 0.5 m with grout
    yy, xx = np.mgrid[0:N, 0:N] / N * 3.0
    grout = ((np.mod(xx, 0.5) < 0.012) | (np.mod(yy, 0.5) < 0.012)).astype(np.float32)
    tile_id = (np.floor(xx / 0.5) * 7 + np.floor(yy / 0.5) * 13).astype(np.int64)
    tile_var = rng.uniform(-0.05, 0.05, tile_id.max() + 1)[tile_id]
    p = 0.60 + tile_var + 0.025 * _tileable_noise(rng, N, base=24)
    p = p * (1 - grout) + 0.36 * grout
    _register("paving", np.stack([p * 1.0, p * 0.98, p * 0.94], -1), tile_m=3.0)
    # facades: 8 m x 8 m tiles, window grid 4 x 3
    facade_styles = [
        dict(wall=(0.78, 0.70, 0.58), brick=False, win=(0.16, 0.20, 0.25), frame=(0.92, 0.92, 0.90)),
        dict(wall=(0.62, 0.30, 0.22), brick=True,  win=(0.14, 0.17, 0.22), frame=(0.88, 0.86, 0.82)),
        dict(wall=(0.58, 0.58, 0.56), brick=False, win=(0.20, 0.25, 0.30), frame=(0.30, 0.30, 0.32)),
        dict(wall=(0.70, 0.76, 0.80), brick=False, win=(0.12, 0.16, 0.22), frame=(0.95, 0.95, 0.95)),
        dict(wall=(0.84, 0.78, 0.64), brick=False, win=(0.18, 0.18, 0.20), frame=(0.45, 0.32, 0.22)),
        dict(wall=(0.42, 0.26, 0.20), brick=True,  win=(0.16, 0.18, 0.21), frame=(0.80, 0.78, 0.70)),
    ]
    yy, xx = np.mgrid[0:N, 0:N] / N * 8.0           # metres; xx along facade, yy downward
    for i, st in enumerate(facade_styles):
        wall = np.ones((N, N, 3), np.float32) * np.array(st["wall"], np.float32)
        wall *= (1 + 0.06 * _tileable_noise(rng, N, base=16))[..., None]
        if st["brick"]:
            row = np.floor(yy / 0.075)
            off = np.where(np.mod(row, 2) == 0, 0.0, 0.12)
            mortar = (np.mod(yy, 0.075) < 0.010) | (np.mod(xx + off, 0.24) < 0.010)
            bid = (np.floor((xx + off) / 0.24) * 31 + row * 17).astype(np.int64)
            bvar = rng.uniform(-0.08, 0.08, bid.max() + 1)[bid]
            wall = wall * (1 + bvar)[..., None]
            wall[mortar] = np.array([0.72, 0.70, 0.66], np.float32)
        cx_ = np.mod(xx, 2.0); cy_ = np.mod(yy, 8.0 / 3.0)
        win = (np.abs(cx_ - 1.0) < 0.55) & (np.abs(cy_ - 1.25) < 0.75)
        frame = (np.abs(cx_ - 1.0) < 0.62) & (np.abs(cy_ - 1.25) < 0.82) & ~win
        sill = (np.abs(cx_ - 1.0) < 0.70) & (np.abs(cy_ - 2.10) < 0.05)
        glass = np.array(st["win"], np.float32) * (1 + 0.5 * np.clip(1.3 - cy_, 0, 1))[..., None]
        glass = glass * (1 + 0.15 * _tileable_noise(rng, N, base=8))[..., None]
        img = wall.copy()
        img[frame] = np.array(st["frame"], np.float32)
        img[sill] = np.array(st["frame"], np.float32) * 0.9
        img[win] = glass[win]
        _register(f"facade{i}", img, tile_m=8.0)
    # wood crate planks 0.15 m
    N2 = 256
    yy, xx = np.mgrid[0:N2, 0:N2] / N2 * 1.2
    plank = np.floor(yy / 0.15).astype(np.int64)
    pc = rng.uniform(0.85, 1.15, plank.max() + 1)[plank]
    grain = _tileable_noise(rng, (N2, N2), base=4, aspect=(4, 1))
    w = (0.55 + 0.06 * grain) * pc
    gap = np.mod(yy, 0.15) < 0.008
    img = np.stack([w * 1.0, w * 0.74, w * 0.46], -1); img[gap] = 0.12
    _register("wood", img, tile_m=1.2)
    # corrugated container (grey, tinted at render time)
    yy, xx = np.mgrid[0:N2, 0:N2] / N2 * 2.4
    corr = 0.82 + 0.13 * np.sin(2 * np.pi * xx / 0.3)
    rust = _tileable_noise(rng, N2, base=8)
    c = corr * (1 - 0.10 * (rust > 1.4))
    img = np.stack([c, c, c], -1) * (1 + 0.04 * _tileable_noise(rng, N2, base=32))[..., None]
    _register("container", img, tile_m=2.4)
    # patterned crate: colour patchwork + stripes (texture-only object)
    yy, xx = np.mgrid[0:N2, 0:N2]
    cell = (yy // 32) * 8 + (xx // 32)
    cols = rng.uniform(0.15, 0.95, (64, 3))
    img = cols[cell].astype(np.float32)
    stripes = (np.mod(xx + yy, 24) < 6)
    img[stripes] *= 0.55
    img *= (1 + 0.08 * _tileable_noise(rng, N2, base=16))[..., None]
    _register("pattern", img, tile_m=1.0)
    # car paint micro-variation, glass, tyre, metal, foliage, bark
    _register("paint", np.repeat((1 + 0.04 * _tileable_noise(rng, 128, base=8))[..., None], 3, -1), tile_m=1.0)
    yy, xx = np.mgrid[0:128, 0:128] / 128.0
    gl = 0.11 + 0.10 * np.clip(1 - 1.6 * yy, 0, 1) + 0.008 * _tileable_noise(rng, 128, base=4)
    _register("glass", np.stack([gl * 0.92, gl * 1.0, gl * 1.12], -1), tile_m=0.6)
    # car body side profile (0.8 m tall): shoulder highlight, trim line, darker sill
    prof = np.ones(128, np.float32)
    yb = np.arange(128) / 128.0 * 0.8
    prof[yb < 0.04] = 1.18; prof[(yb >= 0.52) & (yb < 0.58)] = 0.45; prof[yb >= 0.58] = 0.78
    body = prof[:, None] * (1 + 0.02 * _tileable_noise(rng, 128, base=8))
    _register("carbody", np.repeat(body[..., None], 3, -1), tile_m=0.8)
    t = 0.10 + 0.03 * _tileable_noise(rng, 64, base=8)
    _register("tyre", np.stack([t, t, t], -1), tile_m=0.5)
    m = 0.55 + 0.04 * _tileable_noise(rng, 64, base=8)
    _register("metal", np.stack([m, m * 1.01, m * 1.03], -1), tile_m=1.0)
    f1, f2 = _tileable_noise(rng, N2, base=16), _tileable_noise(rng, N2, base=64)
    leaf = np.clip(0.30 + 0.10 * f1 + 0.08 * f2, 0.05, 0.6)
    img = np.stack([leaf * 0.45, leaf * 0.95, leaf * 0.30], -1)
    img[(f2 < -1.2)] *= 0.35
    _register("foliage", img, tile_m=2.0)
    bk = 0.30 + 0.06 * _tileable_noise(rng, (128, 128), base=4, aspect=(1, 8))
    _register("bark", np.stack([bk * 1.0, bk * 0.80, bk * 0.62], -1), tile_m=1.0)
    # distant skyline with alpha (1200 m wide, 300 m tall), wraps horizontally
    SW, SH = 1024, 256
    img = np.zeros((SH, SW, 4), np.float32)
    x = 0
    while x < SW:
        bw = int(rng.integers(12, 60)); bh = int(rng.integers(30, 200))
        tone = rng.uniform(0.50, 0.68)
        img[SH - bh:, x:x + bw, :3] = np.array([tone * 0.92, tone * 0.96, tone * 1.04])
        img[SH - bh:, x:x + bw, 3] = 1.0
        win = (np.mod(np.arange(SH)[:, None], 6) < 2) & (np.mod(np.arange(x, x + bw)[None, :], 5) < 2)
        reg = img[SH - bh:, x:x + bw, :3]
        reg[win[SH - bh:, :reg.shape[1]]] *= 0.9
        x += bw + int(rng.integers(0, 6))
    _register("skyline", img, tile_m=1200.0, wrap=(True, False))
    # clouds (sampled by ray direction, no parallax)
    cl = _tileable_noise(rng, (256, 1024), base=4, aspect=(1, 4))
    _register("clouds", np.repeat(np.clip((cl - 0.3) / 1.6, 0, 1)[..., None], 3, -1), tile_m=2 * np.pi)

with stage("textures"):
    _build_textures(seed_for("textures"))
print(f"{len(TEXTURES)} textures built:", ", ".join(f"{k}{list(v['shape'][:2])}" for k, v in TEXTURES.items()))
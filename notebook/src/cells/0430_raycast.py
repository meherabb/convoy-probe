# PLACEHOLDER
# 3.1.3 Analytic ray casting and shading. Ray directions are scaled so that their
#       camera-z component is 1; the ray parameter t is therefore the camera depth Z.
SUN_DIR = np.array([-0.35, -1.0, 0.45]) / np.linalg.norm([-0.35, -1.0, 0.45])   # towards the sun (y down)
SKY_HORIZON = np.array([0.80, 0.85, 0.90], np.float32)
SKY_ZENITH = np.array([0.40, 0.58, 0.84], np.float32)
_TEX_IDS = None


def _scene_box_arrays(scene):
    # stacked per-box arrays (cached on the scene dict); textures mapped to integer ids
    global _TEX_IDS
    if _TEX_IDS is None:
        _TEX_IDS = {name: i for i, name in enumerate(sorted(TEXTURES))}
    if "_arr" in scene:
        return scene["_arr"]
    B = scene["boxes"]
    arr = dict(
        center=np.array([b["center"] for b in B], np.float64).reshape(-1, 3),
        half=np.array([b["half"] for b in B], np.float64).reshape(-1, 3),
        yaw=np.array([b["yaw"] for b in B], np.float64),
        mover=np.array([b["mover"] for b in B], np.int32),
        tex=np.array([[_TEX_IDS[b["mat"][a][0]] for a in "xyz"] for b in B], np.int32).reshape(-1, 3),
        tint=np.array([[b["mat"][a][1] for a in "xyz"] for b in B], np.float32).reshape(-1, 3, 3),
    )
    scene["_arr"] = arr
    return arr

def _mover_offsets(scene, frame_k):
    deltas = np.array([m["delta"] for m in scene["movers"]], np.float64).reshape(-1, 3)
    return deltas * float(frame_k)

def cast_rays(scene, C, D, frame_k, include_movers=True, grid=None, pix=None, cam=None, R=None,
              only_movers=None):
    # C: (3,) ray origin; D: (N,3) world directions (camera z = 1).
    # grid=(W,H) when D is the full row-major pixel grid (fast culling); otherwise pix=(u,v)
    # arrays give each ray's pixel position for box culling.
    N = D.shape[0]
    t = np.full(N, np.inf); sem = np.zeros(N, np.int8)
    bid = np.full(N, -1, np.int32); aux = np.zeros(N, np.int32)
    st = scene["street"]; g = st["ground_y"]
    with np.errstate(divide="ignore", invalid="ignore"):
        if only_movers is None:
            tg = (g - C[1]) / D[:, 1]
            ok = (D[:, 1] > 1e-12) & (tg > 1e-6)
            t[ok] = tg[ok]
            xg = C[0] + tg * D[:, 0]
            sem[ok] = np.where(np.abs(xg[ok]) < st["road_half"], SEM["road"], SEM["sidewalk"])
            for code, side in ((1, "left"), (2, "right")):
                xp = st["facade_x_" + side]
                tf = (xp - C[0]) / D[:, 0]
                dir_ok = (D[:, 0] < 0) if side == "left" else (D[:, 0] > 0)
                ok = dir_ok & (tf > 1e-6) & (tf < t)
                if not ok.any():
                    continue
                idx_ok = np.nonzero(ok)[0]
                y = C[1] + tf[idx_ok] * D[idx_ok, 1]; z = C[2] + tf[idx_ok] * D[idx_ok, 2]
                blocks = np.asarray(st["blocks_" + side], np.float64)
                bi = np.searchsorted(blocks[:, 0], z, side="right") - 1
                inb = (bi >= 0) & (bi < len(blocks))
                bi_c = np.clip(bi, 0, len(blocks) - 1)
                inb &= z < blocks[bi_c, 1]
                inb &= (y >= g - blocks[bi_c, 2]) & (y <= g)
                sel = idx_ok[inb]
                t[sel] = tf[sel]; sem[sel] = SEM["facade"]; aux[sel] = code * 10000 + bi_c[inb]
            tb = (st["backdrop_z"] - C[2]) / D[:, 2]
            ok = (D[:, 2] > 1e-9) & (tb > 1e-6) & (tb < t)
            if ok.any():
                idx_ok = np.nonzero(ok)[0]
                x = C[0] + tb[idx_ok] * D[idx_ok, 0]; y = C[1] + tb[idx_ok] * D[idx_ok, 1]
                fp = tb[idx_ok] * np.linalg.norm(D[idx_ok], axis=1) / 600.0
                a = tex_sample("skyline", x + st["skyline_shift"], y - (g - 300.0), fp)[:, 3]
                sel = idx_ok[a > 0.5]
                t[sel] = tb[sel]; sem[sel] = SEM["backdrop"]
        arr = _scene_box_arrays(scene)
        offs = _mover_offsets(scene, frame_k)
        nb = arr["center"].shape[0]
        if grid is not None:
            Wg, Hg = grid
        for i in range(nb):
            mv = arr["mover"][i]
            if mv >= 0 and not include_movers:
                continue
            if only_movers is not None and mv not in only_movers:
                continue
            ctr = arr["center"][i] + (offs[mv] if mv >= 0 else 0.0)
            hb = arr["half"][i]
            # ---- cull with the projected bounding box ----
            if cam is not None and R is not None:
                cn = box_corners({"center": ctr, "half": hb, "yaw": arr["yaw"][i]})
                Xc = (cn - C) @ R
                if np.all(Xc[:, 2] < 0.05):
                    continue
                if np.all(Xc[:, 2] > 0.05):
                    uv = np.stack([cam.cx + cam.fx * Xc[:, 0] / Xc[:, 2], cam.cy + cam.fy * Xc[:, 1] / Xc[:, 2]], -1)
                    u0, v0 = np.floor(uv.min(0)) - 1; u1, v1 = np.ceil(uv.max(0)) + 1
                    if grid is not None:
                        u0, u1 = int(max(u0, 0)), int(min(u1, Wg - 1)); v0, v1 = int(max(v0, 0)), int(min(v1, Hg - 1))
                        if u1 < u0 or v1 < v0:
                            continue
                        sel = (np.arange(v0, v1 + 1)[:, None] * Wg + np.arange(u0, u1 + 1)[None, :]).ravel()
                    else:
                        sel = np.nonzero((pix[0] >= u0) & (pix[0] <= u1) & (pix[1] >= v0) & (pix[1] <= v1))[0]
                        if sel.size == 0:
                            continue
                else:
                    sel = np.arange(N)
            else:
                sel = np.arange(N)
            cy_, sy_ = np.cos(arr["yaw"][i]), np.sin(arr["yaw"][i])
            o = C - ctr
            ol = np.array([cy_ * o[0] - sy_ * o[2], o[1], sy_ * o[0] + cy_ * o[2]])
            Ds = D[sel]
            dl = np.stack([cy_ * Ds[:, 0] - sy_ * Ds[:, 2], Ds[:, 1], sy_ * Ds[:, 0] + cy_ * Ds[:, 2]], -1)
            dl = np.where(np.abs(dl) < 1e-12, 1e-12, dl)
            t1 = (-hb - ol) / dl; t2 = (hb - ol) / dl
            tn = np.minimum(t1, t2); tx = np.maximum(t1, t2)
            tmin = tn.max(1); tmax = tx.min(1)
            hit = (tmax >= tmin) & (tmin > 1e-6) & (tmin < t[sel])
            if not hit.any():
                continue
            hs = sel[hit]
            ax = tn[hit].argmax(1)
            sgn = (-np.sign(dl[hit, :][np.arange(hit.sum()), ax])).astype(np.int32)
            t[hs] = tmin[hit]
            sem[hs] = SEM["mover"] if mv >= 0 else SEM["static_box"]
            bid[hs] = i
            aux[hs] = ax * 2 + (sgn > 0)
    return dict(t=t, sem=sem, bid=bid, aux=aux)

def shade_rays(scene, C, D, hit, frame_k, f_mean):
    N = D.shape[0]
    t, sem, bid, aux = hit["t"], hit["sem"], hit["bid"], hit["aux"]
    col = np.zeros((N, 3), np.float32)
    nrm = np.zeros((N, 3), np.float64)
    dn = np.linalg.norm(D, axis=1)
    fin = np.isfinite(t)
    P = C + np.where(fin, t, 0.0)[:, None] * D
    fp_minor = np.where(fin, t, 0.0) * dn / f_mean
    st = scene["street"]; g = st["ground_y"]
    # ---- ground ----
    for code in (SEM["road"], SEM["sidewalk"]):
        s_ = np.nonzero(sem == code)[0]
        if s_.size == 0:
            continue
        x, z = P[s_, 0], P[s_, 2]
        cos_t = np.abs(D[s_, 1]) / dn[s_]
        fp = fp_minor[s_] / np.sqrt(np.maximum(cos_t, 0.02))
        if code == SEM["road"]:
            alb = tex_sample("asphalt", x, z, fp)
            fx_ = fp_minor[s_]; fz_ = fp_minor[s_] / np.maximum(cos_t, 0.02)
            paint = np.zeros(s_.size, np.float32)
            wR = st["road_half"]
            lines = [(1.75, True), (-1.75, True), (5.25, True), (-5.25, True), (wR - 0.25, False), (-(wR - 0.25), False)]
            for xl, dashed in lines:
                if abs(xl) > wR - 0.2 and dashed:
                    continue
                cov_x = np.clip((0.07 - np.abs(x - xl)) / np.maximum(fx_, 1e-4) + 0.5, 0, 1)
                if dashed:
                    ph = np.mod(z + 3.0 * (xl > 0), 9.0)
                    cz = np.clip(np.minimum(ph, 3.0 - ph) / np.maximum(fz_, 1e-4) + 0.5, 0, 1) * (ph < 3.0 + fz_)
                    cz = np.where(fz_ > 3.0, 1.0 / 3.0, cz)
                    cov_x = cov_x * cz
                paint = np.maximum(paint, cov_x.astype(np.float32))
            alb = alb * (1 - paint[:, None]) + np.array([0.84, 0.83, 0.78], np.float32) * paint[:, None]
        else:
            alb = tex_sample("paving", x, z, fp)
            curb = np.abs(np.abs(x) - st["road_half"]) < 0.15
            alb[curb] = alb[curb] * 0.72
        col[s_] = alb
        nrm[s_] = (0.0, -1.0, 0.0)
    # ---- soft contact shadows (furniture, parked cars, ground movers) ----
    gi = np.nonzero((sem == SEM["road"]) | (sem == SEM["sidewalk"]))[0]
    shadow_mover = np.zeros(N, bool)
    if gi.size and scene.get("shadows"):
        gx, gz = P[gi, 0], P[gi, 2]
        fac = np.ones(gi.size, np.float32)
        offs_s = _mover_offsets(scene, frame_k)
        for sh in scene["shadows"]:
            cx_, cz_ = sh["c"]; hx, hz = sh["h"]
            if sh["mover"] >= 0:
                cx_ += offs_s[sh["mover"]][0]; cz_ += offs_s[sh["mover"]][2]
            r = np.hypot(hx, hz) + 0.7
            q = np.nonzero((np.abs(gx - cx_) < r) & (np.abs(gz - cz_) < r))[0]
            if q.size == 0:
                continue
            c_, s_y = np.cos(sh["yaw"]), np.sin(sh["yaw"])
            dx, dz = gx[q] - cx_, gz[q] - cz_
            lx, lz = c_ * dx - s_y * dz, s_y * dx + c_ * dz
            ox, oz = np.maximum(np.abs(lx) - hx, 0), np.maximum(np.abs(lz) - hz, 0)
            dist = np.hypot(ox, oz)
            inner = np.clip(np.minimum(hx - np.abs(lx), hz - np.abs(lz)) / 0.3, 0, 1)
            amt = sh["k"] * np.clip(1 - dist / 0.6, 0, 1) ** 2 * (0.75 + 0.25 * inner)
            fac[q] *= (1 - amt).astype(np.float32)
            if sh["mover"] >= 0:
                shadow_mover[gi[q[amt > 0.02]]] = True
        col[gi] *= fac[:, None]
    # ---- facades ----
    s_ = np.nonzero(sem == SEM["facade"])[0]
    if s_.size:
        side = aux[s_] // 10000; bi = aux[s_] % 10000
        cos_t = np.abs(D[s_, 0]) / dn[s_]
        fp = fp_minor[s_] / np.sqrt(np.maximum(cos_t, 0.02))
        for sd_code, sd in ((1, "left"), (2, "right")):
            blocks = np.asarray(st["blocks_" + sd], np.float64)
            m = side == sd_code
            if not m.any():
                continue
            ii = s_[m]; b_i = bi[m]
            styles = blocks[b_i, 3].astype(np.int32); shift = blocks[b_i, 4]
            for sty in np.unique(styles):
                q = styles == sty
                jj = ii[q]
                col[jj] = tex_sample(f"facade{sty}", P[jj, 2] + shift[q], P[jj, 1] - g, fp[m][q])
            plinth = P[ii, 1] > g - 0.45
            col[ii[plinth]] *= 0.62
            nrm[ii] = (1.0, 0.0, 0.0) if sd == "left" else (-1.0, 0.0, 0.0)
    # ---- backdrop ----
    s_ = np.nonzero(sem == SEM["backdrop"])[0]
    if s_.size:
        fp = fp_minor[s_]
        col[s_] = tex_sample("skyline", P[s_, 0] + st["skyline_shift"], P[s_, 1] - (g - 300.0), fp)[:, :3]
        nrm[s_] = (0.0, -0.3, -1.0)
    # ---- boxes (static furniture and movers) ----
    s_ = np.nonzero(bid >= 0)[0]
    if s_.size:
        arr = _scene_box_arrays(scene)
        offs = _mover_offsets(scene, frame_k)
        b = bid[s_]
        mv = arr["mover"][b]
        ctr = arr["center"][b] + np.where(mv[:, None] >= 0, offs[np.maximum(mv, 0)] if len(offs) else 0.0, 0.0)
        hb = arr["half"][b]; yaw = arr["yaw"][b]
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        d = P[s_] - ctr
        lp = np.stack([cy_ * d[:, 0] - sy_ * d[:, 2], d[:, 1], sy_ * d[:, 0] + cy_ * d[:, 2]], -1) + hb
        ax = aux[s_] // 2; sg = np.where(aux[s_] % 2 == 1, 1.0, -1.0)
        u = np.where(ax == 0, lp[:, 2], lp[:, 0])
        v = np.where(ax == 1, lp[:, 2], lp[:, 1])
        nl = np.zeros((s_.size, 3)); nl[np.arange(s_.size), ax] = sg
        nw = np.stack([cy_ * nl[:, 0] + sy_ * nl[:, 2], nl[:, 1], -sy_ * nl[:, 0] + cy_ * nl[:, 2]], -1)
        cos_t = np.abs(np.sum(nw * D[s_], 1)) / dn[s_]
        fp = fp_minor[s_] / np.sqrt(np.maximum(cos_t, 0.02))
        tid = arr["tex"][b, ax]; tint = arr["tint"][b, ax]
        inv_tex = {v_: k for k, v_ in _TEX_IDS.items()}
        for tk in np.unique(tid):
            q = tid == tk
            col[s_[q]] = tex_sample(inv_tex[int(tk)], u[q], v[q], fp[q])[:, :3] * tint[q]
        nrm[s_] = nw
    # ---- lighting, haze, sky ----
    lam = np.clip(nrm @ SUN_DIR, 0, None).astype(np.float32)
    lit = col * (0.55 + 0.55 * lam)[:, None]
    back = sem == SEM["backdrop"]
    lit[back] = col[back]
    haze = (1 - np.exp(-np.where(fin, t, 0.0) / 900.0)).astype(np.float32)
    lit = lit * (1 - haze[:, None]) + SKY_HORIZON * haze[:, None]
    sk = np.nonzero(~fin)[0]
    if sk.size:
        Dn = D[sk] / dn[sk, None]
        elev = np.arcsin(np.clip(-Dn[:, 1], -1, 1))
        grad = np.clip(elev / 0.6, 0, 1) ** 0.7
        sky = SKY_HORIZON * (1 - grad[:, None]) + SKY_ZENITH * grad[:, None]
        az = np.arctan2(Dn[:, 0], Dn[:, 2]) + np.pi
        cl = tex_sample("clouds", az, (np.pi / 2 - elev) * 0.5, np.full(sk.size, 1.0 / f_mean))[:, 0]
        cl = cl * np.clip(elev / 0.05, 0, 1)
        lit[sk] = (sky * (1 - 0.8 * cl[:, None]) + 0.97 * 0.8 * cl[:, None]).astype(np.float32)
    return lit, shadow_mover

def render_frame(scene, frame_k, include_movers=True, aa=True, add_noise=True):
    camd = scene["cams"][frame_k]
    cam = PinholeCamera.from_dict(camd)
    R = np.asarray(camd["R"], np.float64); C = np.asarray(camd["C"], np.float64)
    W, H = cam.W, cam.H
    vv, uu = np.mgrid[0:H, 0:W]
    D = cam.rays(uu.ravel().astype(np.float64), vv.ravel().astype(np.float64)) @ R.T
    hit = cast_rays(scene, C, D, frame_k, include_movers, grid=(W, H), cam=cam, R=R)
    f_mean = 0.5 * (cam.fx + cam.fy)
    rgb, shadow_mover = shade_rays(scene, C, D, hit, frame_k, f_mean)
    t = hit["t"].reshape(H, W); sem = hit["sem"].reshape(H, W); bid = hit["bid"].reshape(H, W)
    if aa:
        key = sem.astype(np.int64) * 1000003 + bid
        e = np.zeros((H, W), bool)
        dh = key[:, 1:] != key[:, :-1]; dv = key[1:, :] != key[:-1, :]
        tt = np.where(np.isfinite(t), t, 1e9)
        dh |= np.abs(tt[:, 1:] / tt[:, :-1] - 1) > 0.03; dv |= np.abs(tt[1:, :] / tt[:-1, :] - 1) > 0.03
        e[:, 1:] |= dh; e[:, :-1] |= dh; e[1:, :] |= dv; e[:-1, :] |= dv
        ei = np.nonzero(e.ravel())[0]
        if ei.size:
            acc = rgb[ei].copy()
            for du, dv_ in ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)):
                pu = uu.ravel()[ei] + du; pv = vv.ravel()[ei] + dv_
                Ds = cam.rays(pu, pv) @ R.T
                h2 = cast_rays(scene, C, Ds, frame_k, include_movers, pix=(pu, pv), cam=cam, R=R)
                acc += shade_rays(scene, C, Ds, h2, frame_k, f_mean)[0]
            rgb[ei] = acc / 5.0
    rgb = rgb.reshape(H, W, 3)
    li = scene["light"]
    rgb = rgb * np.asarray(li["white_balance"], np.float32) * li["exposure"]
    if add_noise:
        nrng = np.random.default_rng(scene["seed"] * 31 + frame_k * 7 + (0 if include_movers else 1))
        rgb = rgb + nrng.normal(0, li["noise_sigma"], rgb.shape).astype(np.float32)
    rgb8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    arr = _scene_box_arrays(scene)
    mover_id = np.where(bid >= 0, arr["mover"][np.maximum(bid, 0)], -1).astype(np.int16)
    mover_id[sem != SEM["mover"]] = -1
    return dict(rgb=rgb8, depth=t.astype(np.float32), sem=sem, mover_id=mover_id, box_id=bid.astype(np.int32),
                mover_shadow=shadow_mover.reshape(H, W))

print("Ray caster ready: cast_rays, shade_rays, render_frame.")
# PLACEHOLDER
# 3.1.7 E5 causal ablation of ground contact. A base scene with three purely parallel movers is
#       rendered in five conditions that change ONE thing about the movers and nothing else:
#         G   grounded, with its soft contact shadow (the CONVOY-3D default)
#         NS  grounded, no contact shadow
#         CH  grounded and shadowed, behind a static wall that hides its ground contact in both views
#         F   lifted off the ground by 0.8-1.6 m (a floating car or crate), no shadow
#         FH  lifted, behind the same wall as in CH (control for the wall itself)
#       The motion (s, delta = s b, eps = 0), the camera, the street and every static object are
#       identical across conditions. The wall is designed by ray casting: it hides every visible
#       pixel of the mover up to the cover height (wheels of a car, lowest 35% of a crate) in both
#       frames; whether the contact really is hidden is then measured on the renders (Section 3.3).

def _rect_poly_xz(center, half_x, half_z, yaw):
    R = rot_y(yaw)
    return np.array([(np.asarray(center) + R @ np.array([sx * half_x, 0.0, sz * half_z]))[[0, 2]]
                     for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

def _sat_overlap(a, b):
    # separating-axis test for two convex quadrilaterals in the ground plane
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            if np.linalg.norm(n) == 0:
                continue
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True

def _mover_footprints(scene, j, k):
    m = scene["movers"][j]
    off = np.asarray(m["delta"]) * k
    return [_rect_poly_xz(np.asarray(scene["boxes"][i]["center"]) + off, scene["boxes"][i]["half"][0],
                          scene["boxes"][i]["half"][2], scene["boxes"][i]["yaw"]) for i in m["box_ids"]]

def _mover_height(scene, j):
    bx = [scene["boxes"][i] for i in scene["movers"][j]["box_ids"]]
    top = min(b["center"][1] - b["half"][1] for b in bx); bot = max(b["center"][1] + b["half"][1] for b in bx)
    return bot - top

def _band_rays(scene, j, k, cover_m):
    # camera centre and world directions of the visible pixels of mover j (rendered alone) whose 3-D
    # point lies at most cover_m above the ground, in frame k
    cam, R, C = frame_camera(scene, k)
    m = scene["movers"][j]
    off = np.asarray(m["delta"]) * k
    uv = _project_boxes([scene["boxes"][i] for i in m["box_ids"]], off, cam, R, C, znear=0.05)
    if uv is None:
        return C, np.zeros((0, 3))
    u0, v0 = np.maximum(np.floor(uv.min(0)).astype(int) - 1, 0)
    u1, v1 = np.minimum(np.ceil(uv.max(0)).astype(int) + 1, [cam.W - 1, cam.H - 1])
    vv, uu = np.mgrid[v0:v1 + 1, u0:u1 + 1]
    D = cam.rays(uu.ravel().astype(np.float64), vv.ravel().astype(np.float64)) @ R.T
    h = cast_rays(scene, C, D, k, True, pix=(uu.ravel(), vv.ravel()), cam=cam, R=R, only_movers=[j])
    hit = np.isfinite(h["t"])
    P = C + h["t"][hit, None] * D[hit]
    above = scene["street"]["ground_y"] - P[:, 1]
    return C, D[hit][above <= cover_m]

def design_wall(scene, j, cover_m=None):
    # static wall hiding the ground contact of mover j (grounded geometry) in frames 0 and 1, or None. cover_m (v6.5,
    # contact dose-response) sets the hidden height directly; by default the E5 cover (wheels of a car, 35% of a crate)
    D_ = ABLATION_DESIGN
    m = scene["movers"][j]
    g = scene["street"]["ground_y"]
    if cover_m is not None:
        cover = float(cover_m)
    else:
        cover = D_["contact_cover_car_m"] if m["cls"] == "car" else D_["contact_cover_crate_frac"] * _mover_height(scene, j)
    C0 = np.asarray(scene["cams"][0]["C"]); C1 = np.asarray(scene["cams"][1]["C"])
    c0, c1 = C0[[0, 2]], C1[[0, 2]]
    fps = [np.concatenate(_mover_footprints(scene, j, k)) for k in (0, 1)]
    u = fps[0].mean(0) - c0; u /= np.linalg.norm(u); v = np.array([-u[1], u[0]])
    d_near = min(((f - c0) @ u).min() for f in fps)
    Dw = d_near - D_["wall_gap_m"] - D_["wall_thickness_m"]
    if Dw < 2.0 or (c1 - c0) @ u > Dw - 1.0:
        return None
    lat, hts = [], []
    for k in (0, 1):
        C, rays = _band_rays(scene, j, k, cover)
        if len(rays) == 0:
            return None
        den = rays[:, [0, 2]] @ u
        t = (Dw - (C[[0, 2]] - c0) @ u) / den
        if np.any(den <= 0) or np.any(t <= 0):
            return None
        X = C + t[:, None] * rays
        lat.append((X[:, [0, 2]] - c0) @ v)
        hts.append(g - X[:, 1])
    lat, hts = np.concatenate(lat), np.concatenate(hts)
    h_wall = float(hts.max() + 0.03)
    if h_wall > D_["wall_max_cover_frac"] * max(_mover_height(scene, j), 1.0) + cover or h_wall > 2.5:
        return None
    half_w = float((lat.max() - lat.min()) / 2 + 0.3); lat_c = float((lat.max() + lat.min()) / 2)
    cxz = c0 + u * (Dw + D_["wall_thickness_m"] / 2) + v * lat_c
    yaw = float(np.arctan2(-v[1], v[0]))
    poly = _rect_poly_xz((cxz[0], 0.0, cxz[1]), half_w, D_["wall_thickness_m"] / 2, yaw)
    # the wall may not touch any mover (either frame) or static object
    for jj in range(len(scene["movers"])):
        for k in (0, 1):
            if any(_sat_overlap(poly, f) for f in _mover_footprints(scene, jj, k)):
                return None
    for b in scene["boxes"]:
        if b["kind"] == "static" and _sat_overlap(poly, _rect_poly_xz(b["center"], b["half"][0], b["half"][2], b["yaw"])):
            return None
    return dict(mover=j, center=[float(cxz[0]), float(g - h_wall / 2), float(cxz[1])],
                half=[half_w, h_wall / 2, D_["wall_thickness_m"] / 2], yaw=yaw, height=h_wall, cover_m=float(cover))

def ablation_variant(base, cond, walls, lifts):
    # one condition of a base scene (a new scene dict; the base is not modified)
    sc = json.loads(json.dumps(scene_to_json(base)))
    sc["condition"] = cond
    if cond in ("NS", "F", "FH"):
        sc["shadows"] = [s_ for s_ in sc["shadows"] if s_["mover"] < 0]
    if cond in ("F", "FH"):
        for j, mv in enumerate(sc["movers"]):
            h = lifts[j]
            for i in mv["box_ids"]:
                sc["boxes"][i]["center"][1] -= h
            mv["centroid0"][1] -= h
            mv["lift"] = float(h)
    for mv in sc["movers"]:
        mv["support"] = "lifted" if cond in ("F", "FH") else "grounded"
    if cond in ("CH", "FH"):
        for w in walls:
            sc["boxes"].append(_box(w["center"], w["half"], w["yaw"], _mat("paving", (0.86, 0.86, 0.84)), part="wall"))
            sc["shadows"].append(shadow_fp(w["center"][0], w["center"][2], w["half"][0], w["half"][2] + 0.08, w["yaw"], 0.45))
    sc.pop("_arr", None)
    return sc

def contact_pixels(scene, render, k, j):
    # number of visible pixels of mover j within PREREG['contact_band_m'] of the ground (frame k)
    cam, R, C = frame_camera(scene, k)
    m = render["mover_id"] == j
    if not m.any():
        return 0
    vv, uu = np.nonzero(m)
    D = cam.rays(uu.astype(np.float64), vv.astype(np.float64)) @ R.T
    P = C + render["depth"][vv, uu].astype(np.float64)[:, None] * D
    return int(np.sum(scene["street"]["ground_y"] - P[:, 1] <= PREREG["contact_band_m"]))

print("Ablation builders ready: design_wall, ablation_variant, contact_pixels.")
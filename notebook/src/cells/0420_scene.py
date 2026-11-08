# PLACEHOLDER
# 3.1.2 Scene description: street layout, static furniture, cars and crates as oriented boxes.
#       A scene is a JSON-serialisable dict; rendering (3.1.3) is a pure function of it.
SEM = {"sky": 0, "road": 1, "sidewalk": 2, "facade": 3, "backdrop": 4, "static_box": 5, "mover": 6}
SEM_NAMES = {v: k for k, v in SEM.items()}
CAR_COLORS = [(0.70, 0.07, 0.07), (0.10, 0.20, 0.52), (0.86, 0.86, 0.84), (0.13, 0.13, 0.15),
              (0.55, 0.57, 0.60), (0.12, 0.34, 0.18), (0.80, 0.58, 0.10), (0.35, 0.22, 0.45)]
CONTAINER_TINTS = [(0.85, 0.40, 0.12), (0.15, 0.32, 0.62), (0.20, 0.50, 0.25), (0.70, 0.12, 0.10),
                   (0.80, 0.78, 0.72)]


def _box(center, half, yaw, mat, kind="static", mover=-1, part=""):
    return dict(center=[float(v) for v in center], half=[float(v) for v in half], yaw=float(yaw),
                mat=mat, kind=kind, mover=int(mover), part=part)

def _mat(tex, tint=(1, 1, 1), x=None, y=None, z=None):
    # material per face axis (box-local x / y / z faces); default = same texture everywhere
    base = [tex, [float(t) for t in tint]]
    return {"x": x or base, "y": y or base, "z": z or base}

def car_boxes(origin, yaw, L, Wd, color, kind="static", mover=-1):
    # origin: ground-contact centre (world, y down). Returns the boxes of a low-poly car.
    Ry = rot_y(yaw)
    def place(local_c, half, mat, part):
        return _box(np.asarray(origin) + Ry @ np.asarray(local_c, np.float64), half, yaw, mat, kind, mover, part)
    paint = ["paint", list(color)]
    glass = ["glass", [1, 1, 1]]
    boxes = [
        place((0, -0.72, 0), (Wd / 2, 0.40, L / 2),
              {"x": ["carbody", list(color)], "y": paint, "z": ["carbody", list(color)]}, "body"),
        place((0, -1.40, -0.15 * L), (Wd / 2 * 0.90, 0.30, L / 2 * 0.52),
              {"x": glass, "y": paint, "z": glass}, "cabin"),
    ]
    for sx in (-1, 1):
        for sz in (-1, 1):
            boxes.append(place((sx * (Wd / 2 - 0.13), -0.33, sz * (L / 2 - 0.80)), (0.13, 0.33, 0.33),
                               _mat("tyre"), "wheel"))
        boxes.append(place((sx * (Wd / 2 - 0.30), -0.86, L / 2 + 0.012), (0.20, 0.07, 0.015),
                           _mat("metal", (1.6, 1.6, 1.45)), "headlight"))
        boxes.append(place((sx * (Wd / 2 - 0.30), -0.88, -L / 2 - 0.012), (0.20, 0.06, 0.015),
                           _mat("metal", (1.5, 0.15, 0.12)), "taillight"))
    return boxes

def crate_boxes(center, half, yaw, style, tint, kind="mover", mover=-1):
    tex = {"wood": "wood", "container": "container", "pattern": "pattern"}[style]
    return [_box(center, half, yaw, _mat(tex, tint), kind, mover, "crate")]

STREET_HALF_WIDTH = {"forward": (6.0, 9.0), "oblique": (9.0, 12.0), "side": (11.0, 14.0)}

def lane_centres(road_half):
    c = [1.75 + 3.5 * k for k in range(8) if 1.75 + 3.5 * k <= road_half - 1.2]
    return np.array(sorted([-v for v in c] + c))

def sample_street(rng, kind="forward", z_min=-60.0, z_max=420.0):
    # wider streets for oblique / side-looking cameras so that whole vehicles fit in view
    g = float(rng.uniform(*CAMERA_DESIGN["height_m"]))           # camera height = ground y
    wR = float(rng.uniform(*STREET_HALF_WIDTH[kind]))              # road half width
    side = float(rng.uniform(2.0, 4.0))                            # sidewalk width
    street = dict(ground_y=g, road_half=wR, facade_x_left=-(wR + side), facade_x_right=wR + side,
                  backdrop_z=450.0, skyline_shift=float(rng.uniform(0, 1200)))
    for sd in ("left", "right"):
        blocks, z = [], z_min
        while z < z_max:
            L = float(rng.uniform(10, 30))
            blocks.append([z, z + L, float(rng.uniform(9, 30)), int(rng.integers(0, 6)), float(rng.uniform(0, 8))])
            z += L
        street["blocks_" + sd] = blocks
    return street

def sample_static_furniture(rng, street, z_from=-25.0, z_to=140.0):
    g, wR = street["ground_y"], street["road_half"]
    boxes, shadows = [], []
    for sgn in (-1, 1):
        z = z_from + rng.uniform(0, 10)
        while z < z_to:                                           # street lights
            x = sgn * (wR + 0.6)
            boxes.append(_box((x, g - 3.5, z), (0.07, 3.5, 0.07), 0, _mat("metal"), part="pole"))
            shadows.append(shadow_fp(x, z, 0.07, 0.07, 0.0, 0.35))
            boxes.append(_box((x - sgn * 0.7, g - 6.95, z), (0.7, 0.05, 0.05), 0, _mat("metal"), part="arm"))
            boxes.append(_box((x - sgn * 1.35, g - 6.85, z), (0.28, 0.08, 0.16), 0,
                              _mat("metal", (1.5, 1.45, 1.2)), part="lamp"))
            z += rng.uniform(22, 32)
        z = z_from + rng.uniform(0, 8)
        while z < z_to:                                           # trees
            x = sgn * (wR + 1.4 + rng.uniform(0, 0.6))
            boxes.append(_box((x, g - 1.6, z), (0.16, 1.6, 0.16), 0, _mat("bark"), part="trunk"))
            shadows.append(shadow_fp(x, z, 0.9, 0.9, 0.0, 0.30))
            hc = rng.uniform(1.0, 1.5)
            boxes.append(_box((x, g - 3.2 - hc, z), (rng.uniform(1.1, 1.6), hc, rng.uniform(1.1, 1.6)),
                              rng.uniform(0, np.pi), _mat("foliage", (rng.uniform(0.8, 1.2),) * 3), part="canopy"))
            z += rng.uniform(9, 18)
        z = z_from + rng.uniform(0, 6)
        while z < z_to:                                           # parked cars
            if rng.random() < 0.35:
                L, Wd = rng.uniform(4.1, 4.8), rng.uniform(1.70, 1.85)
                col = CAR_COLORS[int(rng.integers(len(CAR_COLORS)))]
                yaw_p = 0.0 if rng.random() < 0.5 else np.pi
                boxes += car_boxes((sgn * (wR - 1.15), g, z), yaw_p, L, Wd, col)
                shadows.append(shadow_fp(sgn * (wR - 1.15), z, Wd / 2, L / 2, yaw_p, 0.60))
            z += rng.uniform(6.0, 7.5)
        z = z_from + rng.uniform(0, 12)
        while z < z_to:                                           # bins / boxes on the sidewalk
            if rng.random() < 0.5:
                tint = CONTAINER_TINTS[int(rng.integers(len(CONTAINER_TINTS)))]
                boxes.append(_box((sgn * (wR + 0.9), g - 0.55, z), (0.32, 0.55, 0.32), 0,
                                  _mat("container", tint), part="bin"))
                shadows.append(shadow_fp(sgn * (wR + 0.9), z, 0.32, 0.32, 0.0, 0.45))
            z += rng.uniform(8, 16)
    return boxes, shadows

def shadow_fp(cx, cz, hx, hz, yaw, strength, mover=-1):
    # soft contact shadow on the ground (world x-z footprint); movers' shadows move with them
    return dict(c=[float(cx), float(cz)], h=[float(hx), float(hz)], yaw=float(yaw), k=float(strength), mover=int(mover))

def box_corners(b, offset=np.zeros(3)):
    c = np.asarray(b["center"]) + offset; h = np.asarray(b["half"])
    sg = np.array(list(itertools.product((-1, 1), repeat=3)), np.float64)
    return c + (sg * h) @ rot_y(b["yaw"]).T

print("Scene description helpers ready (street, furniture, cars, crates).")
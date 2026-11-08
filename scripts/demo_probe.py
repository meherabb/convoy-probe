#!/usr/bin/env python3
"""The probe in one minute on the CPU: the geometry share of three predictors whose answer is known.

Purely parallel movers are drawn as in CONVOY-3D (s in [-0.6, 0.85], depth 5-45 m, three per scene)
and seen by two cameras with an arbitrary rotation. Three predictors give their depth:

* classical triangulation of the exact correspondence (Proposition 1: it returns Z / (1 - s));
* a single-image prior that ignores the second view (Z up to 5% noise);
* a planted mixture of the two with prior weight w = 0.5 (Lemma 1).

    python scripts/demo_probe.py [--movers 600] [--seed 0]
"""
import argparse
import sys

import _common  # noqa: F401
import numpy as np

from convoy import geometry as G
from convoy import probe
from convoy.registered import MOVER_DESIGN


def movers(n, rng):
    cam = G.PinholeCamera(700.0, 700.0, 319.5, 239.5, 640, 480)
    s_lo, s_hi = MOVER_DESIGN["s_range_main"]
    z_lo, z_hi = MOVER_DESIGN["depth_range_m"]
    rows = []
    while len(rows) < n:
        scene = len(rows) // 3
        R1, C1 = np.eye(3), np.zeros(3)
        R2 = G.rot_ypr(rng.uniform(-0.15, 0.15), rng.uniform(-0.03, 0.03))
        b = np.array([rng.uniform(-0.3, 0.3), 0.0, rng.uniform(0.8, 3.0)])            # mostly forward, as in driving
        C2 = C1 + b
        Z, s = rng.uniform(z_lo, z_hi), rng.uniform(s_lo, s_hi)
        X = G.cam_to_world(cam.rays(rng.uniform(80, 560), rng.uniform(260, 420)) * Z, R1, C1)
        q = G.world_to_cam(X + s * b, R2, C2)                                          # the mover, displaced by s b
        if q[2] <= 0.5:
            continue
        p1, p2 = cam.project(G.world_to_cam(X, R1, C1)), cam.project(q)
        R21, t21 = G.relative_pose(R1, C1, R2, C2)
        if G.translational_parallax(p1, p2, cam.K, cam.K, R21) < 5.0:                # not identifiable from two views
            continue
        Z_tri, _ = G.triangulate_depth(p1, p2, cam.K, cam.K, R21, t21)
        rows.append((scene, s, Z, float(Z_tri)))
    return np.array(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--movers", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    scene, s, Z, Z_tri = movers(args.movers, rng).T
    x = G.geometric_answer(s)
    print(f"{len(s)} purely parallel movers in {len(np.unique(scene))} scenes; "
          f"triangulation returns Z/(1-s) to {np.max(np.abs(np.log(Z_tri / Z) - x)):.1e} in log depth\n")
    prior = Z * np.exp(rng.normal(0, 0.05, len(Z)))
    w = 0.5
    predictors = {"classical triangulation": Z_tri, "single-image prior": prior, f"mixture, w = {w}": w * Z + (1 - w) * Z_tri}
    # Lemma 1 over the x of these movers maps a slope to a prior weight (the paper's table uses the x of its design)
    table = probe.slope_weight_table(x=x, weights=np.linspace(0, 1, 201))
    beta_w = probe.slope_weight_table(x=x, weights=[w])[0]["beta_ols_over_design"]
    print(f"{'predictor':<26}{'beta_s':>8}   95% interval     prior weight (Lemma 1)")
    for name, depth in predictors.items():
        r = probe.geometry_share(s, np.log(depth / Z), scene, seed=1, n_boot=2000)
        w_eff = probe.beta_to_effective_w(r["beta"], table)
        print(f"{name:<26}{r['beta']:>8.3f}   [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]   {w_eff:.2f}")
    print(f"\nLemma 1 over these movers: beta(w = {w}) = {beta_w:.3f}. The six networks of the paper give "
          "beta_s between -0.04 and 0.19 on CONVOY-3D.")
    h = G.twin_contact_height(1.6, s=0.5)
    print(f"Static twin (Proposition 3): a car moving with the camera at s = 0.5, camera 1.6 m above the road, is explained by "
          f"a twin twice its size whose contact lies {abs(float(h)):.1f} m below the road.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

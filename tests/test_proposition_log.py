"""The notebook's proposition checks (cell 2.2), run with the package's geometry, reproduce the run's log."""
import json
import math
import os

from helpers import RESULTS, run_cell

from convoy import geometry
from convoy.registered import seed_for


def test_proposition_checks_reproduce_the_log(tmp_path):
    env = {name: getattr(geometry, name) for name in geometry.__all__}
    ns = run_cell("0320_prop_checks.py", os=os, json=json, seed_for=seed_for, LOG_DIR=str(tmp_path), **env)
    new = ns["PROP_CHECKS"]                                     # the cell's own assertions have passed
    logged = json.loads((RESULTS / "logs" / "proposition_checks.json").read_text())
    assert new.keys() == logged.keys()
    for k, v in new.items():
        ref = logged[k]
        if isinstance(ref, int) or k.endswith(("_n_trials", "_fraction_s_gt_1_triangulating_behind_camera",
                                                 "_fraction_twin_below_road_for_0_lt_s_lt_1")):
            assert v == ref, k                                  # counts and fractions: exactly the run's
        elif abs(ref) < 1e-5:
            assert abs(v) < 1e-5, k                             # float64 noise: the size, not the digits
        else:
            assert math.isclose(v, ref, rel_tol=1e-6), k

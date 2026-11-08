#!/usr/bin/env python3
"""Re-run the numerical checks of the propositions on the CPU and compare them with the run's log.

The checks are those of notebook cell 2.2 (``notebook/src/cells/0320_prop_checks.py``), executed
unchanged but with the geometry of the ``convoy`` package, the same seeds and the same random
streams; the run's values are in ``results/v6.6/logs/proposition_checks.json``.

The notebook numbers the propositions 1-4; in the paper, the notebook's Proposition 3 (motion
inside the epipolar plane) is a statement in Sec. 3 and the notebook's Proposition 4 (the static
twin) is Proposition 3.

    python scripts/check_propositions.py
"""
import json
import math
import os
import sys
import tempfile
import time

import _common  # noqa: F401
import numpy as np

from convoy import geometry
from convoy.registered import seed_for

REPO = _common.REPO
CELL = REPO / "notebook" / "src" / "cells" / "0320_prop_checks.py"
LOGGED = REPO / "results" / "v6.6" / "logs" / "proposition_checks.json"
PAPER_NAME = {"P1": "Proposition 1", "P2": "Proposition 2", "P3": "Sec. 3 (epipolar plane)", "P4": "Proposition 3 (twin)"}


def run_cell(log_dir):
    env = {name: getattr(geometry, name) for name in geometry.__all__}
    env.update(np=np, os=os, json=json, seed_for=seed_for, LOG_DIR=log_dir, __name__="prop_checks")
    exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), env)   # asserts inside the cell must pass
    return env["PROP_CHECKS"]


def main():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        new = run_cell(d)
    logged = json.loads(LOGGED.read_text())
    assert new.keys() == logged.keys(), "the cell and the log have different checks"
    print(f"\n{'check':<58}{'this machine':>14}{'run log':>14}  agreement")
    exact = close = 0
    for k, v in new.items():
        ref = logged[k]
        same = v == ref
        tiny = isinstance(v, float) and abs(ref) < 1e-5 and abs(v) < 1e-5        # both at float64 noise level
        ok = same or (isinstance(v, float) and math.isclose(v, ref, rel_tol=1e-6)) or tiny
        exact += same
        close += ok
        tag = "identical" if same else ("float noise" if tiny else ("rel < 1e-6" if ok else "DIFFERENT"))
        print(f"{k:<58}{v:>14.4g}{ref:>14.4g}  {tag}   [{PAPER_NAME[k[:2]]}]")
    print(f"\n{exact} of {len(new)} values identical to the run, {close} of {len(new)} in agreement "
          f"({time.time() - t0:.0f} s); all assertions of the notebook cell passed.")
    return 0 if close == len(new) else 1


if __name__ == "__main__":
    sys.exit(main())

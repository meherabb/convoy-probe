"""The README shows the tables of the run as generated, its counts are current, and its code example runs."""
import re

import numpy as np
import pytest
from helpers import REPO

from convoy import claims, paper

README = (REPO / "README.md").read_text(encoding="utf-8")
MODEL_COLS = [paper.MODEL_NAMES[m] for m in paper.MODELS]


def test_tables_in_the_readme_are_the_generated_ones():
    blocks = [
        paper.to_markdown(["Test", "Quantity", "n (clusters)"] + MODEL_COLS, paper.registered_tests_table(latex=False)),
        paper.to_markdown(["Mask", "Corr.", "Parallel", "Perpendicular"], paper.masks_table(latex=False)),
        paper.to_markdown(["Corr.", "Regime", "n (tracks)", "Epipolar", "Twin", "Either"], paper.twin_table(latex=False)),
    ]
    for b in blocks:
        assert b in README
    h = paper.h14_row(latex=False)
    assert (f"H14 (Argoverse 2, optical flow, {h['n']} vehicles in {h['n_tracks']} tracks): twin minus epipolar recall "
            f"{h['estimate']} {h['ci']}, {h['verdict']}.") in README


def test_counts_in_the_readme():
    assert f"The {len(claims.CLAIMS)} numeric statements in the text of the paper" in README
    assert f"# {len(claims.CLAIMS)} numeric statements" in (REPO / "docs" / "REPRODUCE.md").read_text(encoding="utf-8")
    assert f"its {len(list((REPO / 'notebook' / 'src' / 'cells').iterdir()))} cells" in README


def _example():
    code = re.search(r"## Measuring another model.*?```python\n(.*?)```", README, re.S).group(1)
    ns = {}
    exec(compile(code, "README.md", "exec"), ns)
    return ns["measure"]


@pytest.mark.parametrize("model, expected", [("triangulation", 1.0), ("prior", 0.0)])
def test_readme_example_runs(model, expected):
    measure = _example()
    rng = np.random.default_rng(0)
    images = []
    for k in range(12):                                     # 12 image pairs with 3 parallel movers each
        gt = rng.uniform(5, 45, (60, 80))
        pred = gt / 3.0                                     # the model's depth is up to scale
        static = np.ones_like(gt, bool)
        movers = []
        for j in range(3):
            mask = np.zeros_like(static)
            mask[5 + 18 * j:15 + 18 * j, 10:40] = True
            s = rng.uniform(-0.6, 0.85)
            if model == "triangulation":
                pred[mask] = gt[mask] / (1 - s) / 3.0       # the static twin's depth (Proposition 1)
            static &= ~mask
            movers.append(dict(mask=mask, s=s))
        images.append(dict(pred=pred, gt=gt, static=static, scene=k, movers=movers))
    r = measure(images)
    assert r["n"] == 36 and r["n_clusters"] == 12
    assert r["beta"] == pytest.approx(expected, abs=1e-9)

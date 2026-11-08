"""The registered constants of the package are those of the notebook and of the executed run."""
import ast
import json

import pytest
from helpers import RESULTS, cell_source

import convoy.registered as reg

RUN_CONFIG = json.loads((RESULTS / "manifest" / "run_config.json").read_text())


def _notebook_literal(name):
    """A dictionary literal assigned at the top level of the run-configuration cell (cell 0.4)."""
    tree = ast.parse(cell_source("0100_run_config.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return json.loads(json.dumps(ast.literal_eval(node.value)))           # tuples -> lists, as in the run record
    raise KeyError(name)


@pytest.mark.parametrize("name", ["PREREG", "MOVER_DESIGN", "CAMERA_DESIGN", "TWIN_DESIGN", "FLOWTWIN_DESIGN"])
def test_constants_equal_the_run_record(name):
    assert getattr(reg, name) == RUN_CONFIG[name]
    assert getattr(reg, name) == _notebook_literal(name)


def test_bootstrap_sizes_and_seed():
    assert reg.N_BOOT == RUN_CONFIG["RUN_CONFIG"]["n_boot"] == 10_000
    assert "B = B or min(BOOT_N, 2000)" in cell_source("0887b_x6.py")            # exploratory medians
    assert reg.N_BOOT_EXPLORATORY == 2_000
    env = json.loads((RESULTS / "logs" / "env_log.json").read_text())
    assert reg.GLOBAL_SEED == env["global_seed"]
    assert f"GLOBAL_SEED = {reg.GLOBAL_SEED}" in cell_source("0080_seeds.py")


def test_registration_files_are_the_run_manifests():
    """registration/ holds the idea lock, the amendments and the decisions exactly as the run wrote them."""
    reg_dir = RESULTS.parents[1] / "registration"
    files = sorted(p.name for p in reg_dir.glob("*.json"))
    assert files == ["amendment_v6.json", "amendment_v6_1.json", "amendment_v6_2.json", "amendment_v6_3.json",
                     "amendment_v6_4.json", "amendment_v6_5.json", "amendment_v6_6.json", "decisions.json", "idea_lock.json"]
    for f in files:
        assert (reg_dir / f).read_bytes() == (RESULTS / "manifest" / f).read_bytes()

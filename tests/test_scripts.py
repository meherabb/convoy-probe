"""The command-line scripts run from a fresh clone and report agreement."""
import subprocess
import sys

from helpers import REPO


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_demo_probe():
    out = _run(REPO / "scripts" / "demo_probe.py", "--movers", "300")
    assert "classical triangulation      1.000" in out and "single-image prior" in out


def test_verify_paper_numbers():
    assert "ALL NUMBERS AGREE" in _run(REPO / "scripts" / "verify_paper_numbers.py")


def test_make_paper_tables(tmp_path):
    _run(REPO / "scripts" / "make_paper_tables.py", "--out", tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tab_alltests.tex", "tab_masks.tex", "tab_tests.tex", "tab_twin.tex",
                                                          "tab_twin_ci.tex", "tables.md"]
    assert (tmp_path / "tab_tests.tex").read_text().count("\\\\\n") == 11

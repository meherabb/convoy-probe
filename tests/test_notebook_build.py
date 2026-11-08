"""The shipped notebook is the one that produced the results and can be rebuilt from its cell sources."""
import json
import re
import subprocess
import sys

from helpers import CELLS, REPO, RESULTS

NOTEBOOK = REPO / "notebook" / "convoy_ambiguity.ipynb"


def test_rebuild_is_byte_identical(tmp_path):
    out = tmp_path / "convoy_ambiguity.ipynb"
    r = subprocess.run([sys.executable, str(REPO / "notebook" / "build_notebook.py"), "--out", str(out), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "identical to the shipped notebook" in r.stdout


def test_notebook_structure():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    assert len(cells) == len(list(CELLS.glob("*.py"))) + len(list(CELLS.glob("*.md"))) == 90
    for n, c in enumerate(cells, start=1):
        first = c["source"].split("\n", 1)[0]
        assert first in (f"# Cell No: {n:03d}", f"<!-- Cell No: {n:03d} -->")
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None             # shipped without outputs
    assert NOTEBOOK.stat().st_size < 1_000_000                                     # Kaggle upload limit
    assert "Version 6.6" in cells[0]["source"]


def test_notebook_is_anonymous():
    sources = [c["source"] for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]]
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z.]{2,}")
    assert not any(email.search(s) for s in sources)                               # no e-mail address
    text = "\n".join(sources)
    assert 'get_secret("KAGGLE_KEY")' in text                                      # credentials only from Kaggle Secrets
    assert not re.search(r'KAGGLE_(USERNAME|KEY)"\]\s*=\s*"', text)                # and never written as literals


def test_notebook_pins_the_model_code():
    src = (CELLS / "0610_repos.py").read_text(encoding="utf-8")
    repos = json.loads((RESULTS / "manifest" / "repos.json").read_text())
    for name, info in repos.items():
        assert info["ok"] and info["note"] == "pinned commit"
        assert re.search(rf'"{re.escape(name)}":.*?commit="{info["commit"]}[0-9a-f]{{28}}"', src, re.S), name

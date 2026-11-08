"""results/v6.6 is the unmodified output of the executed notebook run."""
import csv
import hashlib
import json
import re

from helpers import REPO, RESULTS

#: files of this directory that the run's inventory cannot list (the inventory itself, and the guide to the directory)
NOT_IN_INVENTORY = {"manifest/file_inventory.csv", "README.md"}


def _inventory():
    with open(RESULTS / "manifest" / "file_inventory.csv", newline="", encoding="utf-8") as f:
        return {r["path"]: r for r in csv.DictReader(f)}


def test_every_file_matches_the_run_checksums():
    inv = _inventory()
    files = sorted(p for p in RESULTS.rglob("*") if p.is_file())
    assert len(files) > 250
    for p in files:
        rel = p.relative_to(RESULTS).as_posix()
        if rel in NOT_IN_INVENTORY:
            continue
        assert rel in inv, f"{rel} is not part of the run's outputs"
        assert int(inv[rel]["bytes"]) == p.stat().st_size, rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == inv[rel]["sha256"], rel


def test_every_result_and_figure_of_the_run_is_here():
    """All results, manifests, logs and figure PNGs of the run are included (the per-model cache is not)."""
    inv = _inventory()
    for rel in inv:
        top = rel.split("/")[0]
        if top in ("results", "manifest", "logs") or (top == "tables" and rel.endswith(".csv")) or \
                (top == "figures" and rel.endswith(".png")):
            assert (RESULTS / rel).exists(), rel


def test_figures_have_captions_in_the_manifest():
    figs = json.loads((RESULTS / "manifest" / "figures.json").read_text())
    names = {f.stem for f in (RESULTS / "figures").glob("*.png")}
    listed = {f["figure"] for f in figs}
    assert len(names) == 19 and names == listed, names ^ listed
    assert all(f["caption"] and f["bold_text"] == 0 for f in figs)


def test_readme_links_resolve():
    for md in [REPO / "README.md", *REPO.glob("docs/*.md"), *REPO.glob("*/README.md"), RESULTS / "README.md"]:
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (md.parent / target).exists(), f"{md.relative_to(REPO)}: broken link {target}"

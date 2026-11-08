"""The README's images and links resolve, and its badges and diagram show the current counts of the checks."""
import re

import pytest
from helpers import REPO

from convoy import claims, paper

README = (REPO / "README.md").read_text(encoding="utf-8")
DOCS = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md")), REPO / "registration" / "README.md",
        REPO / "notebook" / "README.md", REPO / "results" / "v6.6" / "README.md"]
ASSETS = REPO / "docs" / "assets"


def _table_rows():
    R = paper.Results()
    return (len(paper.registered_tests_table(R)) + len(paper.all_tests_table(R)) + len(paper.masks_table(R))
            + 2 * len(paper.twin_table(R)))


def test_every_image_of_the_readme_is_in_the_repository():
    paths = re.findall(r'<img src="([^"]+)"', README)
    assert paths
    assert all(not p.startswith(("http://", "https://")) for p in paths), "the README loads no external images"
    missing = [p for p in paths if not (REPO / p).is_file()]
    assert missing == []


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_relative_links_resolve(doc):
    text = doc.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'<a href="([^"]+)"', text)
    local = [t.split("#")[0] for t in targets if not t.startswith(("http://", "https://", "#", "mailto:"))]
    missing = [t for t in local if t and not (doc.parent / t).exists()]
    assert missing == []


def test_badges_show_the_current_counts():
    n, rows = len(claims.CLAIMS), _table_rows()
    assert f"{n}/{n} verified" in (ASSETS / "badges" / "paper_numbers.svg").read_text(encoding="utf-8")
    assert f"{rows}/{rows} verified" in (ASSETS / "badges" / "table_rows.svg").read_text(encoding="utf-8")


def test_study_diagram_shows_the_current_counts():
    svg = (ASSETS / "study.svg").read_text(encoding="utf-8")
    n_files = len(paper.Results().rows("manifest/file_inventory.csv"))
    assert f"{len(claims.CLAIMS)} numeric statements and {_table_rows()} table rows" in svg
    assert f"({n_files} files, SHA-256 checked)" in svg

#!/usr/bin/env python3
"""Write the tables of the paper from the run outputs, as LaTeX rows and as Markdown.

    python scripts/make_paper_tables.py                     # writes build/paper_tables/
    python scripts/make_paper_tables.py --results PATH      # another run's outputs/ directory
    python scripts/make_paper_tables.py --print             # Markdown to stdout

The LaTeX files hold the table bodies exactly as in the paper (``tests/test_paper_tables.py``
compares them row by row).
"""
import argparse
import sys
from pathlib import Path

import _common  # noqa: F401

from convoy import paper

MODEL_COLS = [paper.MODEL_NAMES[m] for m in paper.MODELS]


def build(R):
    tables = {}
    h = paper.h14_row(R)
    t1 = paper.to_latex_rows(paper.registered_tests_table(R))
    t1.append(r"H14 & \multicolumn{8}{l}{Twin minus epipolar recall of parallel vehicles with optical flow (Argoverse~2, "
              rf"{h['n']} vehicles, {h['n_tracks']} tracks): {h['estimate']} {h['ci']}}}\\")
    tables["tab_tests"] = ("Table 1: registered tests", t1,
                           paper.to_markdown(["Test", "Quantity", "n (clusters)"] + MODEL_COLS,
                                             paper.registered_tests_table(R, latex=False)))
    tables["tab_alltests"] = ("Table 6: registered contrasts with 95% intervals",
                              paper.to_latex_rows(paper.all_tests_table(R)),
                              paper.to_markdown(["Test"] + MODEL_COLS, paper.all_tests_table(R, latex=False)))
    tables["tab_masks"] = ("Table 2: dynamic-mask recall at 5% static false positives", paper.to_latex_rows(paper.masks_table(R)),
                           paper.to_markdown(["Mask", "Corr.", "Parallel", "Perpendicular"], paper.masks_table(R, latex=False)))
    for key, title, iv in (("tab_twin", "Table 3: static-twin test on Argoverse 2", False),
                           ("tab_twin_ci", "Table 7: static-twin test with 95% intervals", True)):
        rows = paper.twin_table(R, intervals=iv)
        tables[key] = (title, paper.to_latex_rows(rows),
                       paper.to_markdown(["Corr.", "Regime", "n (tracks)", "Epipolar", "Twin", "Either"],
                                         paper.twin_table(R, latex=False, intervals=iv)))
    return tables, h


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=None, help="results directory (default: results/v6.6)")
    ap.add_argument("--out", default=str(_common.REPO / "build" / "paper_tables"))
    ap.add_argument("--print", action="store_true", help="print the Markdown tables instead of writing files")
    args = ap.parse_args(argv)
    R = paper.Results(args.results) if args.results else paper.Results()
    tables, _ = build(R)
    h = paper.h14_row(R, latex=False)
    md = []
    for key, (title, _tex, mdt) in tables.items():
        md += [f"### {title}", "", mdt, ""]
        if key == "tab_tests":
            md += [f"H14 (Argoverse 2, optical flow, {h['n']} vehicles in {h['n_tracks']} tracks): twin minus epipolar "
                   f"recall {h['estimate']} {h['ci']}, {h['verdict']}.", ""]
    if args.print:
        print("\n".join(md))
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key, (_title, tex, _md) in tables.items():
        (out / f"{key}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {len(tables)} LaTeX table bodies and tables.md to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

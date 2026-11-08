#!/usr/bin/env python3
"""Recompute every number of the paper from the run outputs and report any difference.

    python scripts/verify_paper_numbers.py                      # all claims and tables against results/v6.6
    python scripts/verify_paper_numbers.py --paper main.tex     # also check that every quoted fragment is in the paper
    python scripts/verify_paper_numbers.py --results PATH       # another run's outputs/ directory
    python scripts/verify_paper_numbers.py -v                   # list every claim

The claims are defined in ``src/convoy/claims.py`` (text of the paper) and ``src/convoy/paper.py``
(tables). Exit status 0 if everything agrees.
"""
import argparse
import sys
from pathlib import Path

import _common  # noqa: F401

from convoy import claims, paper


def _table_checks(R):
    """The tables, rebuilt; a table that cannot be rebuilt (missing file, inconsistent n) is reported."""
    out = []
    builders = [("Table 1", lambda: paper.registered_tests_table(R)), ("Table 1, H14", lambda: [paper.h14_row(R)]),
                ("Table 6", lambda: paper.all_tests_table(R)), ("Table 2", lambda: paper.masks_table(R)),
                ("Table 3", lambda: paper.twin_table(R)),
                ("Table 7", lambda: paper.twin_table(R, intervals=True))]
    for name, fn in builders:
        try:
            out.append((name, len(fn()), None))
        except Exception as e:                         # noqa: BLE001
            out.append((name, 0, f"{type(e).__name__}: {e}"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=None)
    ap.add_argument("--paper", default=None, help="LaTeX source of the paper (main.tex)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    R = paper.Results(args.results) if args.results else paper.Results()
    print(f"results: {R.root}\n")
    res = claims.check(R)
    where = None
    for c, v, ok in res:
        if args.verbose or not ok:
            if c.where != where:
                where = c.where
                print(where)
            print(f"  {'ok ' if ok else 'BAD'} {c.value:<48} {'' if ok else '-> ' + v}")
    n_bad = sum(not ok for _c, _v, ok in res)
    print(f"\ntext: {len(res) - n_bad} of {len(res)} numbers agree with the run outputs")
    tbl = _table_checks(R)
    for name, n, err in tbl:
        print(f"  {name:<28} {'BAD ' + err if err else f'{n} rows rebuilt'}")
    n_bad += sum(err is not None for _n, _k, err in tbl)
    if args.paper:
        tex = Path(args.paper).read_text(encoding="utf-8")
        missing = claims.quotes_in(tex)
        for c in missing:
            print(f"  not in the paper ({c.where}): {c.quote}")
        rows = [r for _, rows in [("t1", paper.to_latex_rows(paper.registered_tests_table(R))),
                                  ("all", paper.to_latex_rows(paper.all_tests_table(R))),
                                  ("masks", paper.to_latex_rows(paper.masks_table(R))),
                                  ("twin", paper.to_latex_rows([r[1:] for r in paper.twin_table(R)])),
                                  ("twin-ci", paper.to_latex_rows([r[1:] for r in paper.twin_table(R, intervals=True)]))]
                for r in rows]
        lines = [line.strip() for line in tex.splitlines()]
        absent = [r for r in rows if not any(line.endswith(r) for line in lines)]
        for r in absent:
            print(f"  table row not in the paper: {r}")
        print(f"paper: {len(claims.CLAIMS) - len(missing)} of {len(claims.CLAIMS)} quotes and "
              f"{len(rows) - len(absent)} of {len(rows)} table rows found in {args.paper}")
        n_bad += len(missing) + len(absent)
    print("\nALL NUMBERS AGREE" if n_bad == 0 else f"\n{n_bad} DIFFERENCE(S)")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

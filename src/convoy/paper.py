"""Tables of the paper, generated from the run outputs in ``results/v6.6``.

Every table of the paper is rebuilt here from the result files of the executed notebook, with the
rounding used in the paper; ``tests/test_paper_tables.py`` checks the rows against the LaTeX of the
paper and ``scripts/make_paper_tables.py`` writes them as LaTeX and Markdown.

The results directory is ``results/v6.6`` of this repository; set ``CONVOY_RESULTS`` to read
another run (for example the ``outputs/`` folder of a new notebook run).
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

__all__ = [
    "MODELS", "MODEL_NAMES", "results_dir", "Results", "fmt", "fmt_ci", "fmt_range", "registered_tests_table",
    "all_tests_table", "masks_table", "twin_table", "to_latex_rows", "to_markdown",
]

#: Model keys of the run, in the paper's order, and their names in the paper.
MODELS = ["DUSt3R", "MASt3R", "MonST3R", "VGGT-1B", "pi3", "MapAnything"]
MODEL_NAMES = {"DUSt3R": "DUSt3R", "MASt3R": "MASt3R", "MonST3R": "MonST3R", "VGGT-1B": "VGGT", "pi3": "π³",
               "MapAnything": "MapAnything"}


def results_dir() -> Path:
    """``$CONVOY_RESULTS`` if set, else ``results/v6.6`` of the repository that contains this package."""
    env = os.environ.get("CONVOY_RESULTS")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "results" / "v6.6"
        if (cand / "manifest" / "run_config.json").exists():
            return cand
    raise FileNotFoundError("results/v6.6 not found; set CONVOY_RESULTS to the outputs directory of a run")


class Results:
    """Read-only access to the CSV and JSON files of one run (``results/``, ``tables/``, ``manifest/``, ``logs/``)."""

    def __init__(self, root=None):
        self.root = Path(root) if root is not None else results_dir()
        self._cache = {}

    def path(self, rel) -> Path:
        return self.root / rel

    def rows(self, rel):
        """Rows of a CSV file as dictionaries of strings (read once)."""
        if ("csv", rel) not in self._cache:
            with open(self.path(rel), newline="", encoding="utf-8") as f:
                self._cache["csv", rel] = tuple(csv.DictReader(f))
        return self._cache["csv", rel]

    def json(self, rel):
        """Content of a JSON file (read once)."""
        if ("json", rel) not in self._cache:
            with open(self.path(rel), encoding="utf-8") as f:
                self._cache["json", rel] = json.load(f)
        return self._cache["json", rel]

    def select(self, rel, **where):
        """Rows of a CSV whose columns equal the given values (strings), in file order."""
        return [r for r in self.rows(rel) if all(r.get(k) == v for k, v in where.items())]

    def one(self, rel, **where):
        rs = self.select(rel, **where)
        if len(rs) != 1:
            raise KeyError(f"{rel}: {len(rs)} rows match {where}")
        return rs[0]

    def prediction(self, prefix, model):
        """Row of ``results/prediction_summary.csv`` (every registered and locked test)."""
        rs = [r for r in self.rows("results/prediction_summary.csv") if r["prediction"].startswith(prefix)
              and r["model"] == model]
        if len(rs) != 1:
            raise KeyError(f"prediction_summary: {len(rs)} rows for {prefix!r} / {model!r}")
        return rs[0]


# ---- number formatting as in the paper -------------------------------------------------------------------------

def fmt(v, nd=2, latex=False, keep_sign=False):
    """Round to ``nd`` decimals; ``-0.00`` is written ``0.00`` unless ``keep_sign`` (then one more decimal)."""
    v = float(v)
    s = f"{v:.{nd}f}"
    if float(s) == 0 and s.startswith("-"):
        s = f"{v:.{nd + 1}f}" if keep_sign else s[1:]
    return s.replace("-", "$-$") if latex else s.replace("-", "−")


def fmt_ci(est, lo, hi, nd=2, latex=False):
    """``est [lo, hi]``; a bound that rounds to zero keeps its sign with one more decimal (as in the paper)."""
    return f"{fmt(est, nd, latex)} [{fmt(lo, nd, latex, keep_sign=True)}, {fmt(hi, nd, latex, keep_sign=True)}]"


def fmt_range(values, nd=2, latex=False, sep="--"):
    vals = [float(v) for v in values]
    return f"{fmt(min(vals), nd, latex)}{sep}{fmt(max(vals), nd, latex)}"


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# ---- Table 1 and supplementary Table 6: the registered tests ----------------------------------------------------

#: (row key, prefix in prediction_summary.csv, quantity in LaTeX, quantity in plain text, n source)
REGISTERED_ROWS = [
    ("P1", "P1 degeneracy (CONVOY-3D)", r"$\bs$ on CONVOY-3D (predicted $0.8$--$1.2$)",
     "β_s on CONVOY-3D (predicted 0.8–1.2)", ("results/E1_beta_s.csv", None, "n_clips")),
    ("A1", "A1 correspondence", r"knockout: $\bs$(full) $-$ $\bs$(mover deleted)",
     "knockout: β_s(full) − β_s(mover deleted)", ("results/A1_knockout.csv", None, "n_clips")),
    ("H5", "H5", r"lifting: $\beta(\mathrm F)-\beta(\mathrm{NS})$", "lifting: β(F) − β(NS)",
     ("results/E5_contrasts.csv", "H5", "n_scenes")),
    ("H6", "H6", r"hiding the contact: $\beta(\mathrm{CH})-\beta(\mathrm G)$", "hiding the contact: β(CH) − β(G)",
     ("results/E5_contrasts.csv", "H6", "n_scenes")),
    ("H7", "H7", r"interaction: $(\beta_{\mathrm{CH}}-\beta_{\mathrm G})-(\beta_{\mathrm{FH}}-\beta_{\mathrm F})$",
     "interaction: (β_CH − β_G) − (β_FH − β_F)", ("results/E5_contrasts.csv", "H7", "n_scenes")),
    ("H8", "H8", r"one view: $\mathrm{med}\,\rhom(\mathrm F)-\mathrm{med}\,\rhom(\mathrm{NS})$",
     "one view: med ρ_mono(F) − med ρ_mono(NS)", ("results/E5_contrasts.csv", "H8", "n_scenes")),
    ("H11a", "H11a", r"real, lifting: $\beta(\mathrm F)-\beta(\mathrm G)$", "real, lifting: β(F) − β(G)",
     ("results/E5real_contrasts.csv", "H11a", "n_pairs")),
    ("H11b", "H11b", r"real, one view: $\mathrm{med}\,\rhom(\mathrm F)-\mathrm{med}\,\rhom(\mathrm G)$",
     "real, one view: med ρ_mono(F) − med ρ_mono(G)", ("results/E5real_contrasts.csv", "H11b", "n_pairs")),
    ("H12", "H12", r"baseline: $\beta(\times3)-\beta(\times0.5)$", "baseline: β(×3) − β(×0.5)",
     ("results/BSW_tests.csv", "H12", "n_scenes")),
    ("H13", "H13", r"15\,cm wall: $\beta(\text{wall})-\beta(\text{no wall})$", "15 cm wall: β(wall) − β(no wall)",
     ("results/WDR_tests.csv", "H13", "n_scenes")),
]


def _sample_size(R, source, model):
    rel, test, ccol = source
    where = dict(model=model) if test is None else dict(model=model, test=test)
    r = R.one(rel, **where)
    return int(r["n"]), int(r[ccol])


def _cell(row, key, latex):
    est = fmt(row["estimate"], latex=latex)
    v = row["verdict"]
    if key == "P1":                     # every model falsifies P1; the caption says so
        return est
    if v.startswith("SUPPORTED") or v.startswith("HOLDS"):
        return rf"\textbf{{{est}}}" if latex else f"**{est}**"
    if v == "CONTRADICTED":
        return est + (r"$^\dagger$" if latex else "†")
    if v == "NO PREDICTION":
        return f"({est})"
    return est


def registered_tests_table(R=None, latex=True):
    """Table 1 (``tab:tests``): one row per registered test, estimate per model with its verdict marked."""
    R = R or Results()
    rows = []
    for key, prefix, q_tex, q_txt, source in REGISTERED_ROWS:
        sizes = {_sample_size(R, source, m) for m in MODELS}
        if len(sizes) != 1:
            raise ValueError(f"{key}: the sample differs between models: {sizes}")
        (n, nc), = sizes
        for m in MODELS:                 # the summary and the source file must agree on n
            if int(R.prediction(prefix, m)["n"]) != n:
                raise ValueError(f"{key} / {m}: n differs between prediction_summary.csv and {source[0]}")
        rows.append([key, q_tex if latex else q_txt, f"{n} ({nc})"]
                    + [_cell(R.prediction(prefix, m), key, latex) for m in MODELS])
    return rows


def h14_row(R=None, latex=True):
    """The H14 line of Table 1 (Argoverse 2, optical flow; one test, not per model)."""
    R = R or Results()
    h = R.json("results/twin_flow_thresholds.json")["H14"]
    p = [r for r in R.rows("results/prediction_summary.csv") if r["prediction"].startswith("H14")]
    assert len(p) == 1 and int(p[0]["n"]) == h["n"] and p[0]["verdict"] == h["verdict"]
    est = fmt(h["estimate"], latex=latex)
    est = (rf"\textbf{{{est}}}" if latex else f"**{est}**") if h["verdict"] == "SUPPORTED" else est
    return dict(n=h["n"], n_tracks=h["n_tracks"], estimate=est, ci=f"[{fmt(h['ci_lo'])}, {fmt(h['ci_hi'])}]",
                verdict=h["verdict"])


def all_tests_table(R=None, latex=True):
    """Supplementary Table 6 (``tab:alltests``): estimate and 95% interval of every registered test and model."""
    R = R or Results()
    rows = []
    for key, prefix, *_ in REGISTERED_ROWS:
        rows.append([key] + [fmt_ci(r["estimate"], r["ci_lo"], r["ci_hi"], latex=latex)
                             for r in (R.prediction(prefix, m) for m in MODELS)])
    return rows


# ---- Table 2: dynamic masks --------------------------------------------------------------------------------------

def masks_table(R=None, latex=True):
    """Table 2 (``tab:masks``): recall at 5% static false positives (CONVOY-3D, parallel and perpendicular movers)."""
    R = R or Results()
    rows = R.rows("results/E2_masks.csv")
    dash = "--" if latex else "–"

    def by_key(k):
        return [r for r in rows if r["method_key"] == k and r["model"] in MODELS]

    def single(k):
        r, = [r for r in rows if r["method_key"] == k and r["model"] == "(model-independent)"]
        return fmt(r["rec_par"], latex=latex), fmt(r["rec_perp"], latex=latex)

    def rng(k):
        rs = by_key(k)
        assert len(rs) == len(MODELS), k
        return (fmt_range([r["rec_par"] for r in rs], latex=latex, sep=dash),
                fmt_range([r["rec_perp"] for r in rs], latex=latex, sep=dash))

    raft = "RAFT"
    return [["M1 epipolar residual", "exact", *single("M1_exact")],
            ["M1 epipolar residual", raft, *single("M1_flow")],
            ["M2 rigid-flow residual", "exact", *rng("M2_exact")],
            ["M2 rigid-flow residual", raft, *rng("M2_flow")],
            ["M4 model confidence", dash, *rng("M4")]]


# ---- Table 3 and supplementary Table 7: the static-twin test ----------------------------------------------------

def twin_table(R=None, latex=True, intervals=False):
    """Table 3 (``tab:twin``) or, with ``intervals``, supplementary Table 7 (``tab:twin-ci``)."""
    R = R or Results()
    out = []
    for corr, rel in (("labels", "results/twin_recall_by_regime.csv"), ("flow", "results/twin_flow_recall_by_regime.csv")):
        for r in R.rows(rel):
            cells = []
            for c in ("det_epipolar", "det_twin", "det_either"):
                cells.append(fmt_ci(r[c], r[c + "_lo"], r[c + "_hi"], latex=latex) if intervals else fmt(r[c], latex=latex))
            out.append([corr, r["regime"], f"{int(r['n'])} ({int(r['n_tracks'])})", *cells])
    return out


# ---- output --------------------------------------------------------------------------------------------------------

def to_latex_rows(rows):
    return [" & ".join(r) + r"\\" for r in rows]


def to_markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)

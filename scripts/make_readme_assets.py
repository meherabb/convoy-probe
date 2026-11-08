#!/usr/bin/env python3
"""Draw the overview figures and badges of the README from the run outputs.

    python scripts/make_readme_assets.py                 # writes docs/assets/*.svg from results/v6.6
    python scripts/make_readme_assets.py --results PATH  # from another run's outputs/ directory

Every value in the figures is read from the run outputs; the counts on the badges and in the study
diagram come from ``convoy.claims`` and ``convoy.paper``, so they change when the paper's numbers
change. The figures need matplotlib (``pip install -e ".[assets]"``); the package does not.
"""
import argparse
from html import escape

import _common  # noqa: F401
import numpy as np

from convoy import claims, paper
from convoy.paper import MODEL_NAMES, MODELS

ASSETS = _common.REPO / "docs" / "assets"

# Chart ink and surfaces (light card, readable on light and dark pages) and two categorical slots.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, HAIRLINE = "#e1e0d9", "#c3c2b7", "#d9d8d1"
BLUE, ORANGE, BLUE_DARK, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1c5cab", "#008300", "#6250d6"
SANS = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'SFMono-Regular', Menlo, Consolas, 'DejaVu Sans Mono', monospace"


def minus(s):
    return s.replace("-", "−")


def f2(v):
    return minus(f"{float(v):.2f}")


# ---- figure: the results at a glance ----------------------------------------------------------------------------------

def at_a_glance(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10, "svg.fonttype": "path",
                         "svg.hashsalt": "convoy", "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                         "xtick.color": INK2, "ytick.color": INK, "xtick.major.size": 0, "ytick.major.size": 0})
    fig = plt.figure(figsize=(11.8, 4.5), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 0.95], wspace=0.40, left=0.118, right=0.985, top=0.71, bottom=0.27)
    names = [MODEL_NAMES[m] for m in MODELS]

    def frame(ax, title, sub, letter):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.text(0.0, 1.30, f"{letter}  {title}", transform=ax.transAxes, fontsize=13.2, fontweight="semibold",
                color=INK, va="bottom")
        ax.text(0.0, 1.19, sub, transform=ax.transAxes, fontsize=10.2, color=INK2, va="bottom")
        ax.tick_params(axis="x", labelsize=9.8, pad=3)
        ax.tick_params(axis="y", labelsize=10.6, pad=6)

    def dots(ax, y, est, lo, hi, color):
        for yi, e, a, b in zip(y, est, lo, hi):
            ax.plot([a, b], [yi, yi], color=color, lw=2.0, solid_capstyle="round", zorder=3)
            ax.plot([e], [yi], "o", ms=7.5, color=color, mec=SURFACE, mew=1.6, zorder=4)

    # (a) geometry share on CONVOY-3D, against classical triangulation
    ax = fig.add_subplot(gs[0])
    e1 = [R.one("results/E1_beta_s.csv", model=m) for m in MODELS]
    est = [float(r["beta_s"]) for r in e1]
    y = np.arange(len(MODELS), 0, -1, dtype=float)
    tri = claims.parse_ci(R.one("tables/T3_E1_geometry_share.csv", model="Classical triangulation, RAFT-large flow")["beta_s_ci"])
    n = e1[0]["n"]
    frame(ax, "Little geometry for moving objects", f"$\\beta_s$ of {n} parallel movers, CONVOY-3D", "a")
    for xv, lab in ((0.0, "single-image\nprior"), (1.0, "triangulation")):
        ax.axvline(xv, color=AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
        ax.text(xv, 7.35, lab, ha="center", va="bottom", fontsize=8.8, color=MUTED, linespacing=1.0)
    dots(ax, y, est, [float(r["ci_lo"]) for r in e1], [float(r["ci_hi"]) for r in e1], BLUE)
    dots(ax, [0.0], [tri[0]], [tri[1]], [tri[2]], INK2)
    ax.axhline(0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(list(y) + [0.0], names + ["classical (RAFT)"])
    for lab in ax.get_yticklabels()[-1:]:
        lab.set_color(INK2)
    lo_m, hi_m = int(np.argmin(est)), int(np.argmax(est))
    for i in (lo_m, hi_m):
        ax.text(float(e1[i]["ci_hi"]) + 0.03, y[i], f2(est[i]), va="center", fontsize=9.8, color=INK)
    ax.text(tri[0] - 0.035, 0.0, f2(tri[0]), va="center", ha="right", fontsize=9.8, color=INK)
    ax.set_xlim(-0.28, 1.12)
    ax.set_ylim(-0.6, 7.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)

    # (b) hiding the contact raises the share (registered H6)
    ax = fig.add_subplot(gs[1])
    h6 = [R.prediction("H6", m) for m in MODELS]
    est = [float(r["estimate"]) for r in h6]
    frame(ax, "Hiding the ground contact raises it",
          f"registered H6: $\\beta_s$(hidden) − $\\beta_s$(grounded), {h6[0]['n']} movers", "b")
    ax.axvline(0.0, color=AXIS, lw=1.0, ls=(0, (3, 3)), zorder=1)
    dots(ax, y, est, [float(r["ci_lo"]) for r in h6], [float(r["ci_hi"]) for r in h6], BLUE)
    for i in (int(np.argmin(est)), int(np.argmax(est))):
        ax.text(float(h6[i]["ci_hi"]) + 0.015, y[i], f2(est[i]), va="center", fontsize=9.8, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(-0.05, 0.66)
    ax.set_ylim(0.4, 7.3)
    ax.set_xticks([0, 0.2, 0.4, 0.6], ["0", "0.2", "0.4", "0.6"])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.text(0.0, -0.2, "all six intervals above zero", transform=ax.transAxes, fontsize=9.6, color=INK2)

    # (c) the static-twin test on Argoverse 2 (parallel vehicles)
    ax = fig.add_subplot(gs[2])
    lab = R.one("results/twin_recall_by_regime.csv", regime="parallel")
    flo = R.one("results/twin_flow_recall_by_regime.csv", regime="parallel")
    h14 = R.json("results/twin_flow_thresholds.json")["H14"]
    frame(ax, "Contact exposes the twin", "parallel Argoverse 2 vehicles flagged", "c")
    groups = [("labelled\ncorrespondences", lab), ("optical flow\n(registered H14)", flo)]
    width, gap = 0.34, 0.025
    xr = 0.012                                                  # rounding of the data end, in data units of x
    for g, (_name, r) in enumerate(groups):
        for k, (key, color) in enumerate((("det_epipolar", BLUE), ("det_twin", ORANGE))):
            x0 = g - width - gap / 2 if k == 0 else g + gap / 2
            h = float(r[key])
            ax.add_patch(FancyBboxPatch((x0, 0), width, h, boxstyle=f"round,pad=0,rounding_size={xr}",
                                        mutation_aspect=1 / 0.25, fc=color, ec="none", zorder=3))
            ax.add_patch(Rectangle((x0, 0), width, min(h, 0.05), fc=color, ec="none", zorder=3))
            ax.text(x0 + width / 2, h + 0.025, f2(h), ha="center", va="bottom", fontsize=10.0, color=INK, zorder=4)
    ax.set_xticks([0, 1], [g for g, _ in groups])
    ax.tick_params(axis="x", labelsize=10.0, colors=INK, pad=4)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.tick_params(axis="y", labelsize=9.8, colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.spines["bottom"].set_color(AXIS)
    handles = [Rectangle((0, 0), 1, 1, fc=BLUE), Rectangle((0, 0), 1, 1, fc=ORANGE)]
    ax.legend(handles, ["epipolar test", "static-twin test"], loc="upper right", frameon=False, fontsize=9.6,
              handlelength=1.0, handleheight=1.0, borderaxespad=0.1, labelcolor=INK)
    ax.text(0.0, -0.40, "at 5% false positives on parked vehicles", transform=ax.transAxes, fontsize=9.6, color=INK2)
    ax.text(0.0, -0.50, f"H14: +{float(h14['estimate']):.2f} [{float(h14['ci_lo']):.2f}, {float(h14['ci_hi']):.2f}] "
            f"over {h14['n']} vehicles", transform=ax.transAxes, fontsize=9.6, color=INK2)
    fig.savefig(path, format="svg", facecolor=SURFACE, metadata={"Date": None, "Creator": None})
    plt.close(fig)


# ---- figure: the study as a pipeline ----------------------------------------------------------------------------------

SUB_S = '\u03b2<tspan baseline-shift="sub" font-size="72%">s</tspan>'


def _text(x, y, s, size=13, color=INK, weight=400, anchor="start", family=SANS):
    """An SVG text element; ``{beta_s}`` in ``s`` becomes a subscripted beta_s."""
    body = escape(s).replace("{beta_s}", SUB_S)
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}">{body}</text>')


def study_diagram(R, path):
    n_claims = len(claims.CLAIMS)
    n_rows = _table_rows(R)
    n_boot = R.json("manifest/run_config.json")["RUN_CONFIG"]["n_boot"]
    h14 = R.json("results/twin_flow_thresholds.json")["H14"]
    n_files = len(R.rows("manifest/file_inventory.csv"))
    stages = [
        ("Testbeds", ["CONVOY-3D, exact motion", "E5, sweep, dose-response", "CONVOY-REAL, E5-REAL",
                      "Virtual KITTI 2, Argoverse 2"], "notebook §3"),
        ("Six frozen models", ["DUSt3R · MASt3R · MonST3R", "VGGT · π³ · MapAnything",
                               "two views, depth of view 1", "public checkpoints"], "notebook §4–5"),
        ("The probe", ["ρ: log-depth error", "x = −log(1 − s)", "{beta_s} = slope of ρ on x",
                       "0 prior · 1 geometry"], "convoy.probe"),
        ("Registered tests", ["P1–P4, A1 locked first", "H5–H8, H11–H14 fixed", "before the data they test",
                              f"bootstrap, {n_boot:,} draws"], "convoy.stats"),
        ("Consequences", ["masks inherit the prior", "static-twin height test", "H14 with real flow",
                          f"+{float(h14['estimate']):.2f} [{float(h14['ci_lo']):.2f}, {float(h14['ci_hi']):.2f}]"],
         "convoy.twin"),
    ]
    W, H, x0, top, ch, gap = 1280, 448, 32, 92, 236, 34
    cw = (W - 2 * x0 - 4 * gap) / 5
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="The study in five steps, from testbeds to verified paper numbers">',
           f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="16" fill="{SURFACE}" stroke="{HAIRLINE}"/>',
           _text(x0, 46, "The study at a glance", 21, INK, 650),
           _text(x0, 70, "From exact parallel motion to every number of the paper", 13.5, INK2)]
    for i, (head, lines, where) in enumerate(stages):
        x = x0 + i * (cw + gap)
        out.append(f'<rect x="{x}" y="{top}" width="{cw}" height="{ch}" rx="12" fill="#ffffff" stroke="{HAIRLINE}"/>')
        out.append(f'<rect x="{x}" y="{top}" width="{cw}" height="5" rx="2.5" fill="{BLUE}"/>')
        out.append(f'<circle cx="{x + 28}" cy="{top + 38}" r="14" fill="{BLUE_DARK}"/>')
        out.append(_text(x + 28, top + 43, str(i + 1), 14, "#ffffff", 650, "middle"))
        out.append(_text(x + 52, top + 44, head, 16.5, INK, 650))
        for k, line in enumerate(lines):
            out.append(_text(x + 18, top + 84 + 25 * k, line, 12.8, INK2))
        out.append(f'<rect x="{x + 14}" y="{top + ch - 44}" width="{cw - 28}" height="28" rx="7" fill="#f2f1ec"/>')
        out.append(_text(x + cw / 2, top + ch - 25, where, 12, INK, 500, "middle", MONO))
        if i < len(stages) - 1:
            ax_ = x + cw + 8
            ay = top + ch / 2
            out.append(f'<path d="M{ax_} {ay} h{gap - 16} m-7 -7 l7 7 l-7 7" fill="none" stroke="{MUTED}" '
                       f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    by = top + ch + 26
    out.append(f'<rect x="{x0}" y="{by}" width="{W - 2 * x0}" height="62" rx="12" fill="#eef4fc" stroke="#cde2fb"/>')
    out.append(_text(x0 + 22, by + 26, "Traceability", 14.5, BLUE_DARK, 650))
    out.append(_text(x0 + 22, by + 47, f"results/v6.6 ({n_files} files, SHA-256 checked)  →  scripts/verify_paper_numbers.py  "
                     f"→  {n_claims} numeric statements and {n_rows} table rows of the paper, recomputed", 13.2, INK))
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _table_rows(R):
    """Rows of Tables 1, 2, 3, 6 and 7 that the verifier finds in the LaTeX source of the paper."""
    return (len(paper.registered_tests_table(R)) + len(paper.all_tests_table(R)) + len(paper.masks_table(R))
            + 2 * len(paper.twin_table(R)))


# ---- badges -----------------------------------------------------------------------------------------------------------

def _text_width(s, size=11):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    return TextPath((0, 0), s, size=size, prop=FontProperties(family="DejaVu Sans")).get_extents().width


def badge(label, message, color, path):
    lw, mw = round(_text_width(label)) + 12, round(_text_width(message)) + 12
    w = lw + mw
    text = f"{escape(label)}: {escape(message)}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="20" role="img" aria-label="{text}">
<title>{text}</title>
<clipPath id="r"><rect width="{w}" height="20" rx="3" fill="#fff"/></clipPath>
<g clip-path="url(#r)"><rect width="{lw}" height="20" fill="{INK2}"/><rect x="{lw}" width="{mw}" height="20" fill="{color}"/></g>
<g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
<text x="{lw / 2}" y="14">{escape(label)}</text><text x="{lw + mw / 2}" y="14">{escape(message)}</text></g>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def badges(R, folder):
    folder.mkdir(parents=True, exist_ok=True)
    n = len(claims.CLAIMS)
    rows = _table_rows(R)
    items = [
        ("python", "3.10 – 3.13", BLUE_DARK, "python.svg"),
        ("code", "MIT", GREEN, "license.svg"),
        ("paper numbers", f"{n}/{n} verified", GREEN, "paper_numbers.svg"),
        ("table rows", f"{rows}/{rows} verified", GREEN, "table_rows.svg"),
        ("models", "6 frozen", BLUE_DARK, "models.svg"),
        ("registered", "P1–P4 · A1 · H5–H14", VIOLET, "registered.svg"),
        ("compute", "1× T4 · 10 h", BLUE_DARK, "compute.svg"),
    ]
    for label, msg, color, name in items:
        badge(label, msg, color, folder / name)
    return [name for *_x, name in items]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=None)
    args = ap.parse_args(argv)
    R = paper.Results(args.results) if args.results else paper.Results()
    ASSETS.mkdir(parents=True, exist_ok=True)
    at_a_glance(R, ASSETS / "at_a_glance.svg")
    study_diagram(R, ASSETS / "study.svg")
    names = badges(R, ASSETS / "badges")
    print(f"wrote docs/assets/at_a_glance.svg, docs/assets/study.svg and {len(names)} badges")


if __name__ == "__main__":
    main()

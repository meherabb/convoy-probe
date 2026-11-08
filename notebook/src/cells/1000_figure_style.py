# PLACEHOLDER
# 10.0 Figure style. Exact CVPR sizes: double-column figures 5.5 x 1.8 in, single-column
#      2.8 x 2.0 in, saved WITHOUT a tight bounding box so the files have exactly these sizes.
#      Text is set for print at 100% scale (5-7 pt), in regular weight only: no bold anywhere,
#      panel labels included (mathtext bold is mapped to the regular face as well). Fonts are
#      embedded as TrueType (Type 42). Every figure passes an automatic layout check before it is
#      saved: no text outside the canvas, no overlapping text, no legend covering plotted data.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm
from matplotlib.lines import Line2D
from matplotlib.text import Text
from IPython.display import display, Image as _IPyImage

FIG_DOUBLE, FIG_SINGLE = (5.5, 1.8), (2.8, 2.0)
_installed = {f.name for f in _fm.fontManager.ttflist}
FIG_FONT = next((f for f in ["Arial", "Helvetica", "Liberation Sans", "TeX Gyre Heros", "Nimbus Sans", "FreeSans", "DejaVu Sans"]
                 if f in _installed), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FIG_FONT, "DejaVu Sans"], "font.weight": "normal",
    "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5, "axes.titleweight": "normal",
    "axes.labelweight": "normal", "figure.titleweight": "normal",
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.6, "legend.frameon": True,
    "legend.framealpha": 0.92, "legend.facecolor": "white", "legend.edgecolor": "none", "legend.fancybox": False,
    "legend.borderpad": 0.25,
    "legend.title_fontsize": 5.6, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.2, "legend.columnspacing": 0.8, "legend.labelspacing": 0.25,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 1.5,
    "axes.titlepad": 2.5, "lines.linewidth": 0.9, "lines.markersize": 3.0, "lines.markeredgewidth": 0.5,
    "errorbar.capsize": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": False,
    "mathtext.fontset": "custom", "mathtext.rm": FIG_FONT, "mathtext.it": f"{FIG_FONT}:italic",
    "mathtext.bf": FIG_FONT, "mathtext.sf": FIG_FONT, "mathtext.cal": FIG_FONT, "mathtext.default": "regular",
    "image.interpolation": "nearest", "savefig.facecolor": "white", "figure.facecolor": "white",
})
MODEL_COLORS = {"DUSt3R": "#0072B2", "MASt3R": "#56B4E9", "MonST3R": "#D55E00",
                "VGGT-1B": "#009E73", "pi3": "#CC79A7", "MapAnything": "#E69F00"}
MODEL_MARKERS = {"DUSt3R": "o", "MASt3R": "o", "MonST3R": "o", "VGGT-1B": "s", "pi3": "s", "MapAnything": "s"}
MODEL_LABELS = {"pi3": "π³", "VGGT-1B": "VGGT"}
CONTROL_COLOR, MONO_COLOR, GRID_COLOR = "#000000", "#8C8C8C", "#D9D9D9"
SUPPORT_SHADE = "#E8F2E8"
PAR_SHADE, PERP_SHADE = "#E6EEF7", "#FBEDE6"
REGIME_COLORS = {"parallel": "#0072B2", "intermediate": "#999999", "perpendicular": "#D55E00"}
_extra_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
for _i, _m in enumerate([m for m in EVALUATED_MODELS if m not in MODEL_COLORS]):
    MODEL_COLORS[_m] = _extra_palette[_i % len(_extra_palette)]
    MODEL_MARKERS[_m] = "o" if MODEL_FAMILY.get(_m) == "croco" else "s"
mcol = lambda m: MODEL_COLORS.get(m, "#666666")
mlab = lambda m: MODEL_LABELS.get(m, m)

def gate_caption_note(*testbeds):
    # caption sentence naming the models left out of a figure because they fail the static-scene gate
    parts = []
    for tb in testbeds:
        no = [mlab(m) for m in EVALUATED_MODELS if m not in MODELS_OK[tb] and (m, tb) in GATE_PASS]
        if no:
            parts.append(f"{', '.join(no)} ({TESTBED_NAMES[tb]})")
    return (" Not shown because they fail the static-scene gate: " + "; ".join(parts) + ".") if parts else ""

def panel_label(ax, s, dx=-0.02, dy=1.02, ha="right", fig=None):
    # "(a)", "(b)", ... in regular weight, slightly larger than the body text
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=7, fontweight="normal", va="bottom", ha=ha)

def fig_panel_label(fig, x, y, s):
    fig.text(x, y, s, fontsize=7, fontweight="normal", va="top", ha="left")

# v6.5 (D60): every panel carries a short title "(a) Title" in regular weight, matching the bold lead of its caption
PANEL_TITLE_FS = 6.2

def panel_title(ax, label, title, x=0.0, pad=2.5, loc="left"):
    ax.set_title(f"({label}) {title}" if label else title, fontsize=PANEL_TITLE_FS, loc=loc, pad=pad, x=x)

def fig_panel_title(fig, x, y, label, title, ha="left"):
    fig.text(x, y, f"({label}) {title}" if label else title, fontsize=PANEL_TITLE_FS, va="top", ha=ha)

def title_axes(fig, rect, label, title, x=0.0):
    # an invisible axes that only carries a panel title: an image or schematic panel then shares the title line of the
    # plot panels whose axes have the same top
    a = fig.add_axes(rect)
    a.set_xticks([]); a.set_yticks([])
    a.set_axis_off()
    panel_title(a, label, title, x=x)
    return a

def empty_panel(ax, msg):
    # a panel without data in this run says so instead of showing empty axes
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=5.2, color="#7F7F7F")
    ax.set_yticks([])

def _text_polygon(t, rend):
    # exact outline of a text in display units: the axis-aligned box for 0/90 degree text, and
    # the rotated rectangle (same centre, unrotated width/height) for slanted tick labels
    bb = t.get_window_extent(rend)
    box = np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
    ang = np.deg2rad(t.get_rotation())
    s = t.get_text()
    if abs(np.sin(2 * ang)) < 1e-6 or "\n" in s or "$" in s:
        return box
    w, h, _ = rend.get_text_width_height_descent(s, t.get_fontproperties(), ismath=False)
    c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
    u, v = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
    return np.array([c + sx * w / 2 * u + sy * h / 2 * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

def _polygons_overlap(a, b, tol=1.0):
    # separating-axis test for two convex quadrilaterals; overlap must exceed tol pixels on every axis
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]]); nn = np.linalg.norm(n)
            if nn == 0:
                continue
            n = n / nn
            pa, pb = a @ n, b @ n
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= tol:
                return False
    return True

def _data_points(ax):
    # display coordinates of everything plotted in an axes (line vertices densified along segments,
    # markers, scatter offsets); reference lines are included, images and shaded spans are not
    pts = []
    for ln in ax.get_lines():
        if not ln.get_visible():
            continue
        if not all(ln.get_transform().contains_branch_seperately(ax.transData)):
            continue                                       # axhline / axvline reference lines span the axes
        xy = np.asarray(ln.get_xydata(), float)
        xy = xy[np.isfinite(xy).all(1)] if xy.ndim == 2 else np.zeros((0, 2))
        if not len(xy):
            continue
        P = ln.get_transform().transform(xy)
        if len(P) > 1 and ln.get_linestyle() not in ("None", "", " ", "none"):
            t = np.linspace(0, 1, 24)[:, None, None]
            P = np.concatenate([P, (P[:-1][None] * (1 - t) + P[1:][None] * t).reshape(-1, 2)])
        pts.append(P)
    for coll in ax.collections:
        if not coll.get_visible():
            continue
        try:
            offs = np.asarray(coll.get_offsets(), float)
            if len(offs) and np.isfinite(offs).all():
                pts.append(coll.get_offset_transform().transform(offs))
            for path in coll.get_paths():                 # errorbar whiskers (LineCollection)
                v = path.vertices
                if len(v) == 2 and np.isfinite(v).all():
                    P = coll.get_transform().transform(v)
                    t = np.linspace(0, 1, 12)[:, None]
                    pts.append(P[0] * (1 - t) + P[1] * t)
        except Exception:
            pass
    return np.concatenate(pts) if pts else np.zeros((0, 2))

def _legend_hits(leg, axes, rend):
    bb = leg.get_window_extent(rend).expanded(1.0, 1.0)
    n = 0
    for ax in axes:
        P = _data_points(ax)
        if len(P):
            n += int(((P[:, 0] > bb.x0 + 1) & (P[:, 0] < bb.x1 - 1) & (P[:, 1] > bb.y0 + 1) & (P[:, 1] < bb.y1 - 1)).sum())
    return n

_LEG_LOCS = ["upper left", "upper right", "lower left", "lower right", "center left", "center right", "upper center",
             "lower center", "center"]

def _relocate_legends(fig):
    # an axes legend that covers plotted data is moved to the first standard position inside the axes where it
    # covers none (it stays where it was if every position covers some)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible() or _legend_hits(leg, [ax], rend) == 0:
            continue
        best = (None, _legend_hits(leg, [ax], rend))
        for loc in _LEG_LOCS:
            leg.set_bbox_to_anchor(ax.bbox)
            if hasattr(leg, "set_loc"):
                leg.set_loc(loc)
            else:
                leg._loc = leg.codes[loc]
            fig.canvas.draw()
            h = _legend_hits(leg, [ax], rend)
            if h < best[1]:
                best = (loc, h)
            if h == 0:
                break
        if best[0] is not None and _legend_hits(leg, [ax], rend) != 0:
            if hasattr(leg, "set_loc"):
                leg.set_loc(best[0])
            else:
                leg._loc = leg.codes[best[0]]

def _legend_overlaps(fig, rend):
    out = []
    legends = [(ax.get_legend(), [ax]) for ax in fig.axes if ax.get_legend() is not None]
    legends += [(lg, fig.axes) for lg in fig.legends]
    for leg, axes in legends:
        if not leg.get_visible():
            continue
        bb = leg.get_window_extent(rend).expanded(1.0, 1.0)
        for ax in axes:
            P = _data_points(ax)
            if not len(P):
                continue
            inside = (P[:, 0] > bb.x0 + 1) & (P[:, 0] < bb.x1 - 1) & (P[:, 1] > bb.y0 + 1) & (P[:, 1] < bb.y1 - 1)
            if inside.any():
                out.append((leg.get_texts()[0].get_text()[:16] if leg.get_texts() else "legend", int(inside.sum())))
    return out

def _layout_check(fig):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    hidden = set()
    for ax in fig.axes:                      # tick labels outside the view limits are not drawn
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tk.get_loc()
                if loc is None or not (lo - 1e-9 * abs(hi - lo) <= loc <= hi + 1e-9 * abs(hi - lo)):
                    hidden.update([id(tk.label1), id(tk.label2)])
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and id(t) not in hidden
             and t.get_window_extent(rend).width > 0]
    bold = [t.get_text()[:20] for t in texts if t.get_fontweight() not in ("normal", 400, "regular", "book")]
    boxes = [t.get_window_extent(rend) for t in texts]
    outside = [t.get_text() for t, b in zip(texts, boxes) if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1]
    polys = [_text_polygon(t, rend) for t in texts]
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ix = min(a.x1, b.x1) - max(a.x0, b.x0); iy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ix > 1.0 and iy > 1.0 and _polygons_overlap(polys[i], polys[j]):
                overlaps.append((texts[i].get_text()[:20], texts[j].get_text()[:20]))
    return outside, overlaps, _legend_overlaps(fig, rend), bold

FIGURE_MANIFEST = []

# ---- captions (v6.3): every panel of a caption opens with a bold lead phrase ----------------------
# Captions are written in light Markdown: "(a) **Lead.** text". The notebook shows them rendered;
# figures.json keeps the Markdown, a plain-text and a LaTeX version, and captions.tex (Section 13)
# holds ready-to-paste \caption{...} lines with \textbf lead-ins. The figure graphics themselves stay in
# regular weight (D32).
def cap(label, lead, text=""):
    return f"({label}) **{lead}.** {text}".rstrip()

_TEX_TOKENS = [(r"\brho - rho_mono = w \(x - rho_mono\)", r"$\\rho-\\rho_{\\mathrm{mono}}=w\\,(x-\\rho_{\\mathrm{mono}})$"),
               (r"\bw = c_support d\^2/\(d\^2 \+ kappa\^2\)", r"$w=c_{\\mathrm{support}}\\,d^2/(d^2+\\kappa^2)$"),
               (r"\|x - rho_mono\|", r"$|x-\\rho_{\\mathrm{mono}}|$"), (r"\|eps\|", r"$|\\varepsilon|$"),
               (r"(?<![\w$])-s/\(1-s\)", r"$-s/(1-s)$"), (r"\bH s/\(1-s\)", r"$H s/(1-s)$"),
               (r"0\.85 (<=|≤) s < 1\.15", r"$0.85 \\leq s < 1.15$"), (r"0\.85 (<=|≤) s < 1\b", r"$0.85 \\leq s < 1$"),
               (r"\bbeta_s\b", r"$\\beta_s$"), (r"\bbeta_geo\b", r"$\\beta_{\\mathrm{geo}}$"), (r"\bbeta\(w\)", r"$\\beta(w)$"),
               (r"\bx_geo\b", r"$x_{\\mathrm{geo}}$"), (r"\bZ_geo\b", r"$Z_{\\mathrm{geo}}$"), (r"\brho_mono\b", r"$\\rho_{\\mathrm{mono}}$"),
               (r"\bx = -log\(1-s\)", r"$x=-\\log(1-s)$"), (r"\bX/\(1-s\)", r"$X/(1-s)$"), (r"\bZ/\(1-s\)", r"$Z/(1-s)$"),
               (r"\bs b\b", r"$s\\,\\mathbf{b}$"), (r"(?<![\w$])s >= 1\b", r"$s \\geq 1$"), (r"(?<![\w$])s < 1\b", r"$s < 1$"),
               (r" <= ", r" $\\leq$ "), (r" >= ", r" $\\geq$ "), (r"\bkappa\b", r"$\\kappa$"), (r"\beps\b", r"$\\varepsilon$"),
               (r"\brho\b", r"$\\rho$")]
_TEX_CHARS = [("π³", r"$\pi^3$"), ("β_s", r"$\beta_s$"), ("β", r"$\beta$"), ("ε", r"$\varepsilon$"), ("ρ", r"$\rho$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
              ("−", "$-$"), ("–", "--"), ("×", r"$\times$"), ("°", r"$^\circ$"), ("δ", r"$\delta$"), ("≈", r"$\approx$"),
              ("·", r"$\cdot$"), ("σ", r"$\sigma$"), ("ω", r"$\omega$"), ("ψ", r"$\psi$"), ("³", r"$^3$"), ("²", r"$^2$")]

def caption_plain(s):
    return re.sub(r"\*\*(.+?)\*\*", r"\1", s)

def caption_tex(s):
    out = re.sub(r"\*\*(.+?)\*\*", lambda m_: "\x01" + m_.group(1) + "\x02", s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
        out = out.replace(a, b)
    def _outside_math(fn, text):
        parts = re.split(r"(\$[^$]*\$)", text)              # math segments stay as they are
        return "".join(fn(p_) if k % 2 == 0 else p_ for k, p_ in enumerate(parts))
    for pat, rep in _TEX_TOKENS:
        out = _outside_math(lambda t, pat=pat, rep=rep: re.sub(pat, rep, t), out)
    for a, b in _TEX_CHARS:
        out = _outside_math(lambda t, a=a, b=b: t.replace(a, b), out)
    parts = re.split(r"(\$[^$]*\$)", out)
    for k in range(0, len(parts), 2):
        parts[k] = parts[k].replace("_", r"\_").replace("<", "$<$").replace(">", "$>$").replace("^", r"\^{}")
        parts[k] = re.sub(r"(?<=[0-9)])-(?=[0-9(])", "--", parts[k])          # number and panel ranges: en dash
    return "".join(parts).replace("\x01", r"\textbf{").replace("\x02", "}")

def _show_caption(title, caption):
    try:
        from IPython.display import Markdown as _Md
        display(_Md(f"**{title}.** {caption}"))
    except ImportError:
        print(f"{title}. {caption_plain(caption)}")

def save_figure(fig, fig_id, title, question, source, placement, data, caption=""):
    _relocate_legends(fig)
    outside, overlaps, legend_hits, bold = _layout_check(fig)
    w, h = fig.get_size_inches()
    pdf, png = os.path.join(FIGURE_DIR, f"{fig_id}.pdf"), os.path.join(FIGURE_DIR, f"{fig_id}.png")
    fig.savefig(pdf); fig.savefig(png, dpi=300)
    with open(os.path.join(FIGURE_DIR, f"{fig_id}_data.json"), "w") as f:
        json.dump(data, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    FIGURE_MANIFEST.append(dict(figure=fig_id, title=title, question=question, source=source, placement=placement,
                                caption=caption, caption_plain=caption_plain(caption),
                                caption_tex=f"\\textbf{{{caption_tex(title)}.}} {caption_tex(caption)}",
                                size_in=f"{w:.2f} x {h:.2f}", text_outside=len(outside),
                                text_overlaps=len(overlaps), legend_over_data=len(legend_hits), bold_text=len(bold),
                                pdf=pdf, png=png))
    plt.close(fig)                       # the figure is shown from its saved PNG, not drawn a second time (D32)
    del fig
    display(_IPyImage(filename=png, width=int(round(w * 150))))
    release_memory()
    ok = not outside and not overlaps and not legend_hits and not bold
    qa = "PASS" if ok else (f"CHECK (outside: {outside[:3]}, overlaps: {overlaps[:3]}, legend over data: {legend_hits[:3]}, "
                            f"bold: {bold[:3]})")
    print(f"{fig_id}: {w:.2f} x {h:.2f} in, font {FIG_FONT}, layout QA {qa}")
    if caption:
        _show_caption(title, caption)

def clean_image_axes(a, frame=True, lw=0.4):
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(frame); sp.set_linewidth(lw)

def tag(a, s, loc="lower left", color="white", fs=5.0):
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.04, "bottom") if "lower" in loc else (0.96, "top")
    a.text(x, y, s, transform=a.transAxes, fontsize=fs, color=color, va=va, ha=ha,
           bbox=dict(boxstyle="square,pad=0.14", fc="black", ec="none", alpha=0.55))

def crop_around(img, bbox, out_hw, pad=0.35):
    # crop an image around a bounding box (u0, v0, u1, v1) with relative padding, fixed aspect
    H, W = img.shape[:2]
    u0, v0, u1, v1 = bbox
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    h = (v1 - v0) * (1 + 2 * pad); w = (u1 - u0) * (1 + 2 * pad)
    ar = out_hw[1] / out_hw[0]
    if w / max(h, 1) > ar:
        h = w / ar
    else:
        w = h * ar
    h, w = min(h, H), min(w, W)
    a0 = int(np.clip(cu - w / 2, 0, W - w)); b0 = int(np.clip(cv - h / 2, 0, H - h))
    return img[b0:b0 + int(h), a0:a0 + int(w)]

def mask_bbox(mask, margin=0):
    vv, uu = np.nonzero(mask)
    return (uu.min() - margin, vv.min() - margin, uu.max() + margin, vv.max() + margin) if len(vv) else None

def depth_to_rgb(d, vmin=None, vmax=None, cmap="turbo"):
    # colour encodes log depth between the limits vmin, vmax (log metres): near = warm, far = cool;
    # missing or infinite depth (sky) is white
    ld = np.log(np.clip(np.nan_to_num(d, nan=np.nan, posinf=np.nan), 1e-3, None))
    fin = np.isfinite(ld)
    vmin = np.nanpercentile(ld[fin], 2) if vmin is None else vmin
    vmax = np.nanpercentile(ld[fin], 98) if vmax is None else vmax
    x = 1.0 - np.clip((ld - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgb = plt.get_cmap(cmap)(np.clip(np.nan_to_num(x, nan=0.0), 0.04, 0.96))[..., :3]
    rgb[~fin] = 1.0
    return rgb

def ci_err(est, lo, hi):
    # asymmetric error-bar lengths, or None when the interval is missing
    return [[est - lo], [hi - est]] if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(est) else None

print(f"Figure style ready: double column {FIG_DOUBLE} in, single column {FIG_SINGLE} in, font {FIG_FONT}, regular weight only.")

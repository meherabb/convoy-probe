#!/usr/bin/env python3
"""Build the Kaggle notebook from its cell sources and validate it.

The notebook ``convoy_ambiguity.ipynb`` is generated from the files in ``src/cells/`` (sorted by
file name; ``*.py`` become code cells, ``*.md`` markdown cells). Building it again reproduces the
shipped notebook byte for byte, which ``--check`` verifies.

    python notebook/build_notebook.py                 # writes notebook/build/convoy_ambiguity.ipynb
    python notebook/build_notebook.py --check         # rebuilds and compares with the shipped notebook
    python notebook/build_notebook.py --out PATH      # writes to PATH

Validation: sequential cell numbering, every code cell compiles, nbformat schema (if installed),
no bold text drawn into figures, no e-mail address in any cell, and none of the terms listed in
the environment variable ``NOTEBOOK_DENYLIST`` (comma-separated; use it to keep identifying
strings out of an anonymous release).
"""
import argparse
import glob
import json
import os
import py_compile
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CELL_DIR = os.path.join(HERE, "src", "cells")
SHIPPED = os.path.join(HERE, "convoy_ambiguity.ipynb")
MAX_BYTES = 1_000_000          # Kaggle upload limit the notebook was designed for


def load_cells(cell_dir=CELL_DIR):
    files = sorted(glob.glob(os.path.join(cell_dir, "*.py")) + glob.glob(os.path.join(cell_dir, "*.md")),
                   key=os.path.basename)
    cells = []
    for n, path in enumerate(files, start=1):
        with open(path, encoding="utf-8") as f:
            src = f.read().rstrip("\n")
        if path.endswith(".py"):
            assert src.startswith("# PLACEHOLDER\n"), f"{path}: a code cell must start with '# PLACEHOLDER'"
            src = src.replace("# PLACEHOLDER", f"# Cell No: {n:03d}", 1)
            kind = "code"
        else:
            assert src.startswith("<!-- Cell No: PLACEHOLDER -->\n"), f"{path}: a markdown cell must start with the placeholder"
            src = src.replace("<!-- Cell No: PLACEHOLDER -->", f"<!-- Cell No: {n:03d} -->", 1)
            kind = "markdown"
        cells.append((os.path.basename(path), kind, src))
    return cells


def to_notebook(cells):
    nb_cells = []
    for n, (_name, kind, src) in enumerate(cells, start=1):
        cid = f"cell-{n:03d}"                        # nbformat 4.5 cell ids
        if kind == "code":
            nb_cells.append({"cell_type": "code", "execution_count": None, "id": cid, "metadata": {}, "outputs": [],
                             "source": src})
        else:
            nb_cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": src})
    return {
        "cells": nb_cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"file_extension": ".py", "mimetype": "text/x-python", "name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def validate(nb, cells):
    nums = []
    for c in nb["cells"]:
        first = c["source"].split("\n", 1)[0]
        pat = r"# Cell No: (\d{3})" if c["cell_type"] == "code" else r"<!-- Cell No: (\d{3}) -->"
        m = re.fullmatch(pat, first)
        assert m, f"bad first line: {first!r}"
        nums.append(int(m.group(1)))
    assert nums == list(range(1, len(nb["cells"]) + 1)), "cell numbering is not sequential"
    for name, kind, src in cells:                   # every code cell must compile
        if kind != "code":
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(src)
            tmp = f.name
        try:
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError as e:
            raise SystemExit(f"compile error in {name}: {e}") from e
        finally:
            os.remove(tmp)
    try:
        import nbformat
        nbformat.validate(nbformat.from_dict(nb))
    except ImportError:
        pass
    # figure text is drawn in regular weight (bold LaTeX is allowed in caption strings only)
    draw = re.compile(r"set_(x|y)?label|set_title|\.text\(|legend\(|label=|tag\(|suptitle")
    for name, kind, src in cells:
        if kind != "code":
            continue
        if re.search(r"fontweight\s*=\s*[\"']bold|weight\s*=\s*[\"']bold", src):
            raise SystemExit(f"bold text found in {name}")
        for line in src.split("\n"):
            if draw.search(line) and re.search(r"\\\\textbf|\\\\mathbf|\\textbf|\\mathbf", line):
                raise SystemExit(f"bold text found in {name}: {line.strip()[:80]}")
    # anonymity: no e-mail address and none of the user-supplied identifying terms
    terms = [t.strip() for t in os.environ.get("NOTEBOOK_DENYLIST", "").split(",") if t.strip()]
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z.]{2,}"
    if terms:
        pattern += "|" + "|".join(re.escape(t) for t in terms)
    anon = re.compile(pattern, re.I)
    for name, _kind, src in cells:
        hit = anon.search(src)
        if hit:
            raise SystemExit(f"identifying string in {name}: {hit.group(0)!r}")
    return nums


def build(out):
    cells = load_cells()
    nb = to_notebook(cells)
    nums = validate(nb, cells)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, separators=(",", ":"))     # compact: below 1 MB for Kaggle
    n_code = sum(1 for c in cells if c[1] == "code")
    size = os.path.getsize(out)
    print(f"wrote {out}: {len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown), "
          f"numbering 001-{nums[-1]:03d} OK; {size:,} bytes")
    if size >= MAX_BYTES:
        raise SystemExit(f"notebook too large: {size:,} bytes (limit {MAX_BYTES:,})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "build", "convoy_ambiguity.ipynb"))
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the shipped notebook")
    args = ap.parse_args(argv)
    out = build(args.out)
    if args.check:
        with open(out, "rb") as a, open(SHIPPED, "rb") as b:
            same = a.read() == b.read()
        print("identical to the shipped notebook" if same else "DIFFERS from the shipped notebook")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

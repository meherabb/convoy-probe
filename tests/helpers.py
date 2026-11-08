"""Load definitions from the notebook's cell sources so that tests can compare them with the package."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CELLS = REPO / "notebook" / "src" / "cells"
RESULTS = REPO / "results" / "v6.6"


def cell_source(name: str) -> str:
    return (CELLS / name).read_text(encoding="utf-8")


def cell_definitions(name: str, names=None, imports=True, **env):
    """Execute only the imports and the top-level ``def``/``class``/assignments of ``names`` in a cell.

    The rest of the cell (which loads data, runs models or draws figures) is skipped, so a
    function of the notebook can be called exactly as written, with the globals it needs in ``env``.
    """
    tree = ast.parse(cell_source(name), filename=name)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if imports:
                keep.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names):
            keep.append(node)
        elif isinstance(node, ast.Assign) and names is not None and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets):
            keep.append(node)
    ns = dict(np=np, pd=pd, __name__=f"notebook_{Path(name).stem}")
    ns.update(env)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(CELLS / name), "exec"), ns)
    return ns


def run_cell(name: str, **env):
    """Execute a whole cell in a fresh namespace holding ``env`` (for cells without heavy dependencies)."""
    ns = dict(np=np, pd=pd, __name__=f"notebook_{Path(name).stem}")
    ns.update(env)
    exec(compile(cell_source(name), str(CELLS / name), "exec"), ns)
    return ns

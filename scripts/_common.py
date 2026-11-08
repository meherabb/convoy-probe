"""Shared set-up of the scripts: make ``convoy`` importable from a fresh clone without installing it."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
try:
    import convoy  # noqa: F401
except ImportError:                                   # running from a clone without ``pip install -e .``
    sys.path.insert(0, str(REPO / "src"))

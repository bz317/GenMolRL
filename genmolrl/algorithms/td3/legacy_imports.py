"""Compatibility imports for the existing custom PGFS TD3 implementation."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_legacy_pgfs_on_path() -> Path:
    root = Path(__file__).resolve().parents[4] / "designing-new-molecules"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root

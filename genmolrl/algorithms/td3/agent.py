"""TD3 agent adapter.

The first GenMolRL migration reuses the existing custom PGFS TD3 implementation
while moving launch/config/env ownership into GenMolRL.
"""

from __future__ import annotations

from genmolrl.algorithms.td3.legacy_imports import ensure_legacy_pgfs_on_path

ensure_legacy_pgfs_on_path()

from src.models.pgfs.train.td3_agent import TD3Agent  # noqa: E402

__all__ = ["TD3Agent"]

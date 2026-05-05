"""Replay buffer adapter for PGFS TD3."""

from __future__ import annotations

from genmolrl.algorithms.td3.legacy_imports import ensure_legacy_pgfs_on_path

ensure_legacy_pgfs_on_path()

from src.models.pgfs.train.replay_buffer import ReplayBuffer  # noqa: E402

__all__ = ["ReplayBuffer"]

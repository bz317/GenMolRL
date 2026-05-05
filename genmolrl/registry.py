"""Gymnasium registration for GenMolRL environments."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register


ENV_ID = "GenMolRL-MoleculeDesign-v0"


def register_envs() -> None:
    """Register environments once."""
    if ENV_ID in gym.envs.registry:
        return
    register(
        id=ENV_ID,
        entry_point="genmolrl.envs.molecule_design_env:MoleculeDesignEnv",
        max_episode_steps=5,
    )

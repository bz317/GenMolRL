"""Lazy adapter for the hierarchical Bi-PPO trainer."""

from __future__ import annotations


class PPOBiAdapter:
    """Adapter boundary that avoids importing torch until needed."""

    start_strategy = "random_pool"
    action_representation = "multidiscrete_hierarchical"
    algorithm = "ppo_bi"

    @staticmethod
    def train(config: dict, experiment_name: str) -> None:
        from genmolrl.algorithms.ppo_bi.train import train

        train(config, experiment_name)

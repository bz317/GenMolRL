"""Shared W&B metric setup."""

from __future__ import annotations

import wandb


PPO_COMPATIBLE_METRICS = [
    "training/*",
    "train/*",
    "eval/*",
    "training/total_reward_each_episode",
    "train/mean_reward",
    "eval/total_reward_each_episode",
    "eval/mean_reward",
    "eval/mean_ep_length",
    "avg_reward",
    "total_qed",
    "avg_qed",
    "max_qed",
    "episode_length",
    "episode",
    "total_steps",
    "overall_max_qed",
    "total_episodes",
    "cumulative_reward",
    "steps_done",
    "reward_per_step",
    "qed_per_step",
    "critic_loss",
    "actor_loss",
    "temperature",
]


def define_ppo_compatible_metrics() -> None:
    if wandb.run is None:
        return
    wandb.define_metric("train/global_step")
    for metric in PPO_COMPATIBLE_METRICS:
        wandb.define_metric(metric, step_metric="train/global_step")

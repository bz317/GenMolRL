"""Shared SB3 callbacks."""

from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb


class EpisodeWandbCallback(BaseCallback):
    """Log PPO-compatible episode metrics from SB3 info dicts."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.episode_rewards: deque[float] = deque(maxlen=window)
        self.cumulative_reward = 0.0
        self.overall_max_qed = 0.0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue
            reward = float(ep.get("r", 0.0))
            length = int(ep.get("l", 0))
            qed = float(info.get("QED", 0.0))
            self.total_episodes += 1
            self.episode_rewards.append(reward)
            self.cumulative_reward += reward
            self.overall_max_qed = max(self.overall_max_qed, qed)
            step = int(self.num_timesteps)
            wandb.log(
                {
                    "train/global_step": step,
                    "training/total_reward_each_episode": reward,
                    "train/mean_reward": float(np.mean(self.episode_rewards)),
                    "avg_reward": reward / max(length, 1),
                    "episode_length": length,
                    "episode": self.total_episodes,
                    "total_steps": step,
                    "total_episodes": self.total_episodes,
                    "cumulative_reward": self.cumulative_reward,
                    "max_qed": qed,
                    "overall_max_qed": self.overall_max_qed,
                },
                step=step,
            )
        return True

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


class EvaluationWandbCallback(BaseCallback):
    """Evaluate the current SB3 policy and log PPO-compatible eval metrics."""

    def __init__(self, eval_env, *, eval_freq: int = 10000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.last_eval_step = 0
        self.eval_count = 0

    def _base_eval_env(self):
        return self.eval_env.envs[0].unwrapped

    def _num_eval_episodes(self) -> int:
        base_env = self._base_eval_env()
        if hasattr(base_env.start_strategy, "num_starts"):
            return int(base_env.start_strategy.num_starts())
        return int(len(base_env.reactant_keys))

    def _reset_eval_cycle(self) -> None:
        base_env = self._base_eval_env()
        if hasattr(base_env.start_strategy, "reset_cycle"):
            base_env.start_strategy.reset_cycle()

    def _predict(self, obs):
        try:
            mask = self.eval_env.envs[0].unwrapped.action_masks()
            action, _ = self.model.predict(obs, deterministic=True, action_masks=np.asarray([mask]))
        except TypeError:
            action, _ = self.model.predict(obs, deterministic=True)
        return action

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = int(self.num_timesteps)
        self.eval_count += 1

        episode_rewards = []
        episode_lengths = []
        episode_final_qeds = []
        episode_max_qeds = []
        self._reset_eval_cycle()
        n_eval_episodes = self._num_eval_episodes()
        obs = self.eval_env.reset()

        for episode_idx in range(1, n_eval_episodes + 1):
            done = False
            episode_reward = 0.0
            episode_length = 0
            max_qed = 0.0
            final_qed = 0.0
            while not done:
                action = self._predict(obs)
                obs, rewards, dones, infos = self.eval_env.step(action)
                reward = float(np.asarray(rewards).reshape(-1)[0])
                info = infos[0] if infos else {}
                done = bool(np.asarray(dones).reshape(-1)[0])
                episode_reward += reward
                episode_length += 1
                q = float(info.get("QED", 0.0))
                max_qed = max(max_qed, q)
                final_qed = q

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_final_qeds.append(final_qed)
            episode_max_qeds.append(max_qed)
            wandb.log(
                {
                    "train/global_step": int(self.num_timesteps),
                    "eval/episode": episode_idx,
                    "eval/total_reward_each_episode": episode_reward,
                    "eval/source_train_global_step": int(self.num_timesteps),
                },
                step=int(self.num_timesteps),
            )

        wandb.log(
            {
                "train/global_step": int(self.num_timesteps),
                "eval/mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                "eval/std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
                "eval/mean_ep_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                "eval/mean_final_qed": float(np.mean(episode_final_qeds)) if episode_final_qeds else 0.0,
                "eval/max_qed": float(np.max(episode_max_qeds)) if episode_max_qeds else 0.0,
                "eval/max_episode_qed": float(np.max(episode_max_qeds)) if episode_max_qeds else 0.0,
                "eval/n_molecules": n_eval_episodes,
                "eval_count": self.eval_count,
            },
            step=int(self.num_timesteps),
        )
        return True

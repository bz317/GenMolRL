"""Custom PGFS TD3 training entry point."""

from __future__ import annotations

import os

import numpy as np
import torch

import wandb
from genmolrl.algorithms.common import env_kwargs, init_wandb, set_seed
from genmolrl.algorithms.td3.agent import TD3Agent
from genmolrl.algorithms.td3.knn import KNNWrapper
from genmolrl.algorithms.td3.random_selector import NoValidActionError, select_random_action
from genmolrl.algorithms.td3.replay_buffer import ReplayBuffer
from genmolrl.config import project_root
from genmolrl.logging.wandb_metrics import define_ppo_compatible_metrics
from genmolrl.registry import ENV_ID, register_envs

import gymnasium as gym  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_td3_env(config: dict, *, eval_env: bool = False):
    register_envs()
    kwargs = env_kwargs(config, eval_env=eval_env)
    kwargs["algorithm_family"] = "td3_pgfs"
    kwargs["append_action_mask_to_obs"] = False
    env = gym.make(ENV_ID, **kwargs)
    return KNNWrapper(env)


def _to_r2_tensor(env, r2):
    if isinstance(r2, torch.Tensor):
        return r2
    if r2 is None:
        return torch.zeros((1, env.unwrapped.observation_space.shape[0]), device=device)
    return torch.tensor(env.unwrapped.reactants[r2], dtype=torch.float32, device=device).unsqueeze(0)


def _has_real_action(env, smiles: str | None) -> bool:
    if not smiles:
        return False
    mask_kind = getattr(env.unwrapped.mask_provider, "mode", "r2_available")
    return bool(env.unwrapped.reaction_manager.feasible_first_reactant_templates(smiles, kind=mask_kind))


def train(config: dict, experiment_name: str):
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)
    config = dict(config)
    config["algorithm"] = "TD3"
    config.setdefault("env", {})["algorithm_family"] = "td3_pgfs"
    run = init_wandb(config, "TD3", experiment_name)
    define_ppo_compatible_metrics()

    env = _make_td3_env(config, eval_env=False)
    td3_cfg = config["td3"]
    train_cfg = config["training"]
    agent = TD3Agent(
        env,
        float(td3_cfg.get("actor_lr", 1e-4)),
        float(td3_cfg.get("critic_lr", 3e-4)),
        float(td3_cfg.get("gamma", 0.99)),
        float(td3_cfg.get("tau", 0.005)),
        float(td3_cfg.get("policy_noise", 0.2)),
        float(td3_cfg.get("noise_std", 0.1)),
        float(td3_cfg.get("noise_clip", 0.2)),
        int(td3_cfg.get("policy_freq", 2)),
        float(td3_cfg.get("initial_temperature", 1.0)),
        float(td3_cfg.get("min_temperature", 0.25)),
        int(train_cfg.get("start_timesteps", 10000)),
        int(train_cfg.get("total_timesteps", 1_000_000)),
    )
    replay_buffer = ReplayBuffer(
        env.unwrapped.observation_space.shape[0],
        env.unwrapped.action_space.n,
        env.unwrapped.observation_space.shape[0],
        int(td3_cfg.get("buffer_size", 500000)),
    )

    max_timesteps = int(train_cfg.get("total_timesteps", 1_000_000))
    start_timesteps = int(train_cfg.get("start_timesteps", 10000))
    batch_size = int(td3_cfg.get("batch_size", 64))
    save_freq = int(config["callbacks"].get("model_save_freq", 5000))
    checkpoint_dir = project_root() / "runs" / run.id / "td3_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_done = 0
    episode_count = 0
    completed_rewards: list[float] = []
    cumulative_reward = 0.0
    overall_max_qed = 0.0
    while steps_done < max_timesteps:
        episode_count += 1
        state, info = env.reset(seed=seed + episode_count)
        done = False
        episode_reward = 0.0
        episode_len = 0
        max_qed = 0.0
        while not done and steps_done < max_timesteps:
            if not _has_real_action(env, info.get("SMILES")) and not env.unwrapped.use_stop_action:
                break
            steps_done += 1
            episode_len += 1
            if steps_done < start_timesteps:
                env.disable()
                try:
                    action = select_random_action(env, info["SMILES"])
                except NoValidActionError:
                    steps_done -= 1
                    episode_len -= 1
                    break
            else:
                env.enable()
                action = agent.get_action(state)
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)
            replay_buffer.add(
                info.get("SMILES"),
                state,
                action[0],
                _to_r2_tensor(env, action[1]),
                reward,
                next_info.get("SMILES"),
                next_state,
                done,
            )
            state = next_state
            info = next_info
            episode_reward += float(reward)
            max_qed = max(max_qed, float(info.get("QED", 0.0)))
            wandb.log(
                {
                    "train/global_step": steps_done,
                    "steps_done": steps_done,
                    "reward_per_step": float(reward),
                    "qed_per_step": float(info.get("QED", 0.0)),
                    "episode": episode_count,
                },
                step=steps_done,
            )
            if steps_done >= start_timesteps and replay_buffer.size() >= batch_size:
                metrics = agent.train(replay_buffer, batch_size)
                metrics.update({"train/global_step": steps_done, "steps_done": steps_done})
                wandb.log(metrics, step=steps_done)
            if steps_done % save_freq == 0:
                agent.save_model(str(checkpoint_dir / f"checkpoint_{steps_done}.tar"), steps_done, episode_count, replay_buffer)
        completed_rewards.append(episode_reward)
        cumulative_reward += episode_reward
        overall_max_qed = max(overall_max_qed, max_qed)
        wandb.log(
            {
                "train/global_step": steps_done,
                "training/total_reward_each_episode": episode_reward,
                "train/mean_reward": float(np.mean(completed_rewards[-100:])),
                "avg_reward": episode_reward / max(episode_len, 1),
                "episode_length": episode_len,
                "episode": episode_count,
                "total_steps": steps_done,
                "total_episodes": episode_count,
                "cumulative_reward": cumulative_reward,
                "max_qed": max_qed,
                "overall_max_qed": overall_max_qed,
            },
            step=steps_done,
        )
    agent.save_model(str(checkpoint_dir / "final_model.pth"), steps_done, episode_count, replay_buffer)
    return agent

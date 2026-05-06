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


def _num_eval_episodes(eval_env) -> int:
    if hasattr(eval_env.unwrapped.start_strategy, "num_starts"):
        return int(eval_env.unwrapped.start_strategy.num_starts())
    return int(len(eval_env.unwrapped.reactant_keys))


def _reset_eval_cycle(eval_env) -> None:
    if hasattr(eval_env.unwrapped.start_strategy, "reset_cycle"):
        eval_env.unwrapped.start_strategy.reset_cycle()


def _evaluate_td3(agent, eval_env, *, seed: int, eval_count: int, steps_done: int) -> None:
    previous_env = agent.env
    agent.env = eval_env
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_final_qeds: list[float] = []
    episode_max_qeds: list[float] = []
    try:
        _reset_eval_cycle(eval_env)
        n_eval_episodes = _num_eval_episodes(eval_env)
        for episode_idx in range(1, n_eval_episodes + 1):
            state, info = eval_env.reset(seed=seed + 1_000_000 + eval_count * 10_000 + episode_idx)
            done = False
            episode_reward = 0.0
            episode_len = 0
            max_qed = float(info.get("QED", 0.0))
            final_qed = max_qed
            while not done:
                if not _has_real_action(eval_env, info.get("SMILES")) and not eval_env.unwrapped.use_stop_action:
                    break
                eval_env.enable()
                action = agent.get_action(state, evaluate=True)
                state, reward, terminated, truncated, info = eval_env.step(action)
                done = bool(terminated or truncated)
                episode_reward += float(reward)
                episode_len += 1
                final_qed = float(info.get("QED", 0.0))
                max_qed = max(max_qed, final_qed)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_len)
            episode_final_qeds.append(final_qed)
            episode_max_qeds.append(max_qed)
            wandb.log(
                {
                    "train/global_step": steps_done,
                    "eval/episode": episode_idx,
                    "eval/total_reward_each_episode": episode_reward,
                    "eval/source_train_global_step": steps_done,
                },
                step=steps_done,
            )
        wandb.log(
            {
                "train/global_step": steps_done,
                "eval/mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                "eval/std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
                "eval/mean_ep_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                "eval/mean_final_qed": float(np.mean(episode_final_qeds)) if episode_final_qeds else 0.0,
                "eval/max_qed": float(np.max(episode_max_qeds)) if episode_max_qeds else 0.0,
                "eval/max_episode_qed": float(np.max(episode_max_qeds)) if episode_max_qeds else 0.0,
                "eval/n_molecules": n_eval_episodes,
                "eval_count": eval_count,
            },
            step=steps_done,
        )
    finally:
        agent.env = previous_env


def train(config: dict, experiment_name: str):
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)
    config = dict(config)
    config["algorithm"] = "TD3"
    config.setdefault("env", {})["algorithm_family"] = "td3_pgfs"
    run = init_wandb(config, "TD3", experiment_name)
    define_ppo_compatible_metrics()

    env = _make_td3_env(config, eval_env=False)
    eval_env = _make_td3_env(config, eval_env=True)
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
    eval_freq = int(train_cfg.get("eval_freq", 10000))
    warmup_stop_probability = float(td3_cfg.get("warmup_stop_probability", 0.1))
    checkpoint_dir = project_root() / "runs" / run.id / "td3_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_done = 0
    episode_count = 0
    completed_rewards: list[float] = []
    cumulative_reward = 0.0
    overall_max_qed = 0.0
    eval_count = 0
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
                    action = select_random_action(
                        env,
                        info["SMILES"],
                        stop_probability=warmup_stop_probability,
                    )
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
            if eval_freq > 0 and steps_done >= start_timesteps and steps_done % eval_freq == 0:
                eval_count += 1
                _evaluate_td3(
                    agent,
                    eval_env,
                    seed=seed,
                    eval_count=eval_count,
                    steps_done=steps_done,
                )
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

"""Custom PGFS TD3 training entry point."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

import wandb
from genmolrl.algorithms.common import env_kwargs, init_wandb, set_seed
from genmolrl.algorithms.td3.constants import TD3_UNI_DISCRETE_ACTION_DESIGN
from genmolrl.algorithms.td3.agent import TD3Agent
from genmolrl.algorithms.td3.knn import KNNWrapper
from genmolrl.algorithms.td3.mask_kind import td3_template_mask_kind
from genmolrl.algorithms.td3.random_selector import NoValidActionError, select_random_action
from genmolrl.algorithms.td3.replay_buffer import ReplayBuffer
from genmolrl.config import project_root
from genmolrl.logging.wandb_metrics import define_ppo_compatible_metrics
from genmolrl.registry import ENV_ID, register_envs

import gymnasium as gym  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _td3_fp_dim(env) -> int:
    """Width of the morgan fingerprint that backs both observations and the
    continuous R2 vector. Disentangled from ``observation_space.shape`` so the
    R2 head stays at 1024 dims when the action mask gets appended to the obs.
    """
    base = getattr(env.unwrapped, "base_obs_dim", None)
    if base is not None:
        return int(base)
    return int(env.unwrapped.observation_space.shape[0])


def _td3_r2_vec_dim(env) -> int:
    if getattr(env.unwrapped, "action_design", "") == TD3_UNI_DISCRETE_ACTION_DESIGN:
        return 0
    return _td3_fp_dim(env)


def _make_td3_env(config: dict, *, eval_env: bool = False):
    register_envs()
    kwargs = env_kwargs(config, eval_env=eval_env)
    kwargs["algorithm_family"] = "td3_pgfs"
    # Append the per-step action mask to the observation, matching PPO/A2C.
    # Without this, the TD3 actor's f_net only sees the morgan fingerprint and
    # has to *infer* which templates are feasible from the fingerprint alone;
    # the Stop slot is always feasible, so its logit gets gradient signal
    # from every state in the batch while each template logit only gets
    # signal from states where that template is feasible. Including the mask
    # gives the actor explicit per-state feasibility, the same input PPO/A2C
    # already get. The continuous R2 head still emits a 1024-dim fingerprint
    # vector (see ``_td3_fp_dim``) so this change does not affect the R2
    # storage / KNN logic for bi reactions.
    kwargs["append_action_mask_to_obs"] = True
    env = gym.make(ENV_ID, **kwargs)
    if getattr(env.unwrapped, "action_design", "") == TD3_UNI_DISCRETE_ACTION_DESIGN:
        return env
    return KNNWrapper(env)


def _to_r2_tensor(env, r2):
    if getattr(env.unwrapped, "action_design", "") == TD3_UNI_DISCRETE_ACTION_DESIGN:
        return torch.zeros((1, 0), device=device)
    if isinstance(r2, torch.Tensor):
        return r2
    if r2 is None:
        return torch.zeros((1, _td3_fp_dim(env)), device=device)
    return torch.tensor(env.unwrapped.reactants[r2], dtype=torch.float32, device=device).unsqueeze(0)


def _has_real_action(env, smiles: str | None, *, template_mask_kind: str | None = None) -> bool:
    if not smiles:
        return False
    kind = td3_template_mask_kind(env, override=template_mask_kind)
    return bool(env.unwrapped.reaction_manager.feasible_first_reactant_templates(smiles, kind=kind))


def _num_eval_episodes(eval_env) -> int:
    if hasattr(eval_env.unwrapped.start_strategy, "num_starts"):
        return int(eval_env.unwrapped.start_strategy.num_starts())
    return int(len(eval_env.unwrapped.reactant_keys))


def _reset_eval_cycle(eval_env) -> None:
    if hasattr(eval_env.unwrapped.start_strategy, "reset_cycle"):
        eval_env.unwrapped.start_strategy.reset_cycle()


def _evaluate_td3(
    agent,
    eval_env,
    *,
    seed: int,
    eval_count: int,
    steps_done: int,
    template_mask_kind: str | None = None,
) -> None:
    previous_env = agent.env
    agent.env = eval_env
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_reaction_lengths: list[int] = []
    episode_start_qeds: list[float] = []
    episode_final_qeds: list[float] = []
    episode_final_delta_qeds: list[float] = []
    episode_max_qeds: list[float] = []
    episode_stopped: list[float] = []
    try:
        _reset_eval_cycle(eval_env)
        n_eval_episodes = _num_eval_episodes(eval_env)
        for episode_idx in range(1, n_eval_episodes + 1):
            state, info = eval_env.reset(seed=seed + 1_000_000 + eval_count * 10_000 + episode_idx)
            done = False
            episode_reward = 0.0
            episode_len = 0
            reaction_len = 0
            start_qed = float(info.get("QED", 0.0))
            max_qed = start_qed
            final_qed = max_qed
            stopped = False
            while not done:
                if not _has_real_action(eval_env, info.get("SMILES"), template_mask_kind=template_mask_kind) and not eval_env.unwrapped.use_stop_action:
                    break
                if hasattr(eval_env, "enable"):
                    eval_env.enable()
                action = agent.get_action(state, evaluate=True)
                selected_template = int(action[0].detach().reshape(-1).argmax().item())
                selected_stop = eval_env.unwrapped.use_stop_action and selected_template == eval_env.unwrapped.num_templates
                state, reward, terminated, truncated, info = eval_env.step(action)
                done = bool(terminated or truncated)
                episode_reward += float(reward)
                episode_len += 1
                if selected_stop or info.get("stop"):
                    stopped = True
                else:
                    reaction_len += 1
                final_qed = float(info.get("QED", 0.0))
                max_qed = max(max_qed, final_qed)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_len)
            episode_reaction_lengths.append(reaction_len)
            episode_start_qeds.append(start_qed)
            episode_final_qeds.append(final_qed)
            episode_final_delta_qeds.append(final_qed - start_qed)
            episode_max_qeds.append(max_qed)
            episode_stopped.append(float(stopped))
            wandb.log(
                {
                    "train/global_step": steps_done,
                    "eval/episode": episode_idx,
                    "eval/total_reward_each_episode": episode_reward,
                    "eval/final_delta_qed_each_episode": final_qed - start_qed,
                    "eval/reaction_length_each_episode": reaction_len,
                    "eval/stopped_each_episode": float(stopped),
                    "eval/source_train_global_step": steps_done,
                },
                step=steps_done,
            )
        final_delta_array = np.asarray(episode_final_delta_qeds, dtype=np.float32)
        wandb.log(
            {
                "train/global_step": steps_done,
                "eval/mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                "eval/std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
                "eval/mean_ep_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                "eval/mean_reaction_length": float(np.mean(episode_reaction_lengths)) if episode_reaction_lengths else 0.0,
                "eval/stop_rate": float(np.mean(episode_stopped)) if episode_stopped else 0.0,
                "eval/mean_start_qed": float(np.mean(episode_start_qeds)) if episode_start_qeds else 0.0,
                "eval/mean_final_qed": float(np.mean(episode_final_qeds)) if episode_final_qeds else 0.0,
                "eval/mean_final_delta_qed": float(np.mean(episode_final_delta_qeds)) if episode_final_delta_qeds else 0.0,
                "eval/positive_delta_fraction": float(np.mean(final_delta_array > 0.0)) if episode_final_delta_qeds else 0.0,
                "eval/negative_delta_fraction": float(np.mean(final_delta_array < 0.0)) if episode_final_delta_qeds else 0.0,
                "eval/zero_delta_fraction": float(np.mean(final_delta_array == 0.0)) if episode_final_delta_qeds else 0.0,
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
    env_cfg = config.get("env", {})
    if env_cfg.get("action_design") == TD3_UNI_DISCRETE_ACTION_DESIGN and config.get("reaction_mode") != "uni":
        raise ValueError(f"{TD3_UNI_DISCRETE_ACTION_DESIGN} requires reaction_mode: uni")
    run = init_wandb(config, "TD3", experiment_name)
    define_ppo_compatible_metrics()

    env = _make_td3_env(config, eval_env=False)
    eval_env = _make_td3_env(config, eval_env=True)
    td3_cfg = config["td3"]
    train_cfg = config["training"]
    r2_vec_dim = _td3_r2_vec_dim(env)
    mk = td3_cfg.get("template_mask_kind")
    template_mask_kind = mk if isinstance(mk, str) else None
    # Opt-in SAC-discrete-style soft actor-critic update. Default False keeps
    # existing TD3 runs untouched; setting ``td3.entropy_regularization=true``
    # in the YAML switches to the entropy-regularized actor/critic loss.
    entropy_regularization = bool(td3_cfg.get("entropy_regularization", False))
    entropy_alpha = float(td3_cfg.get("entropy_alpha", 0.2))
    # Optional automatic alpha tuning. When True, ``entropy_alpha`` is the
    # initial value of a learnable alpha and ``target_entropy`` is the
    # per-state entropy the tuner aims for (good default ≈ 0.5-0.6 nats for
    # uni mode where mean log(N_feasible) ≈ 1.08).
    auto_tune_alpha = bool(td3_cfg.get("auto_tune_alpha", False))
    target_entropy = float(td3_cfg.get("target_entropy", 0.5))
    # When set, per-state target = target_entropy_ratio * log(N_feasible(s)),
    # which scales the entropy budget with the per-state action count and is
    # the recommended path for masked discrete tasks. When None/null in the
    # YAML, the agent falls back to the fixed-nats ``target_entropy`` above.
    _ratio_cfg = td3_cfg.get("target_entropy_ratio", None)
    target_entropy_ratio = None if _ratio_cfg is None else float(_ratio_cfg)
    alpha_lr = float(td3_cfg.get("alpha_lr", 3e-4))

    # Network width / activation are configurable so TD3 can mirror PPO/A2C's
    # SB3 default ``[64, 64]`` Tanh policy/critic. When a key is omitted from
    # the YAML the agent falls back to the legacy ``[256, 128, 128]`` ReLU
    # actor / ``[256, 64, 16]`` ReLU critic / ``[256, 256, 167]`` ReLU R2 head
    # so existing TD3 configs reproduce bit-for-bit.
    def _hidden_dims(key):
        value = td3_cfg.get(key)
        if value is None:
            return None
        return [int(x) for x in value]

    actor_hidden_dims = _hidden_dims("actor_hidden_dims")
    critic_hidden_dims = _hidden_dims("critic_hidden_dims")
    pi_hidden_dims = _hidden_dims("pi_hidden_dims")
    activation = td3_cfg.get("activation")
    if isinstance(activation, str):
        activation = activation.lower()

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
        template_mask_kind=template_mask_kind,
        entropy_regularization=entropy_regularization,
        entropy_alpha=entropy_alpha,
        auto_tune_alpha=auto_tune_alpha,
        target_entropy=target_entropy,
        target_entropy_ratio=target_entropy_ratio,
        alpha_lr=alpha_lr,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        pi_hidden_dims=pi_hidden_dims,
        activation=activation,
    )
    replay_buffer = ReplayBuffer(
        env.unwrapped.observation_space.shape[0],
        env.unwrapped.action_space.n,
        r2_vec_dim,
        int(td3_cfg.get("buffer_size", 500000)),
    )

    max_timesteps = int(train_cfg.get("total_timesteps", 1_000_000))
    start_timesteps = int(train_cfg.get("start_timesteps", 10000))
    batch_size = int(td3_cfg.get("batch_size", 64))
    save_freq = int(config["callbacks"].get("model_save_freq", 100000))
    eval_freq = int(train_cfg.get("eval_freq", 10000))
    warmup_stop_probability = float(td3_cfg.get("warmup_stop_probability", 0.1))
    # Optional epsilon-greedy template exploration during the post-warmup phase.
    # Defaults are 0/0/0 so existing TD3 runs and other algorithms are unaffected.
    training_random_action_prob = float(td3_cfg.get("training_random_action_prob", 0.0))
    training_random_action_min_prob = float(
        td3_cfg.get("training_random_action_min_prob", training_random_action_prob)
    )
    training_random_action_decay_steps = int(td3_cfg.get("training_random_action_decay_steps", 0))
    save_replay_buffer_in_checkpoints = bool(td3_cfg.get("save_replay_buffer_in_checkpoints", False))
    checkpoint_dir = project_root() / "runs" / run.id / "td3_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_done = 0
    episode_count = 0
    completed_rewards: list[float] = []
    cumulative_reward = 0.0
    overall_max_qed = 0.0
    eval_count = 0
    max_dead_start_resamples = int(td3_cfg.get("max_dead_start_resamples", 4096))
    while steps_done < max_timesteps:
        episode_count += 1
        state, info = env.reset(seed=seed + episode_count)
        if not env.unwrapped.use_stop_action:
            resamples = 0
            while not _has_real_action(env, info.get("SMILES"), template_mask_kind=template_mask_kind) and resamples < max_dead_start_resamples:
                resamples += 1
                episode_count += 1
                state, info = env.reset(seed=seed + episode_count)
            if not _has_real_action(env, info.get("SMILES"), template_mask_kind=template_mask_kind):
                steps_done += 1
                wandb.log(
                    {
                        "train/global_step": steps_done,
                        "steps_done": steps_done,
                        "training/dead_start_skip": 1.0,
                    },
                    step=steps_done,
                )
                continue
        done = False
        episode_reward = 0.0
        episode_len = 0
        max_qed = 0.0
        while not done and steps_done < max_timesteps:
            steps_done += 1
            episode_len += 1
            if steps_done < start_timesteps:
                if hasattr(env, "disable"):
                    env.disable()
                try:
                    action = select_random_action(
                        env,
                        info["SMILES"],
                        stop_probability=warmup_stop_probability,
                        template_mask_kind=template_mask_kind,
                    )
                except NoValidActionError:
                    steps_done -= 1
                    episode_len -= 1
                    break
            else:
                eps = training_random_action_prob
                if training_random_action_decay_steps > 0 and training_random_action_prob > 0.0:
                    progress = (steps_done - start_timesteps) / float(
                        training_random_action_decay_steps
                    )
                    progress = max(0.0, min(1.0, progress))
                    eps = (
                        training_random_action_prob
                        + (training_random_action_min_prob - training_random_action_prob) * progress
                    )
                use_random = eps > 0.0 and random.random() < eps
                if use_random:
                    if hasattr(env, "disable"):
                        env.disable()
                    try:
                        action = select_random_action(
                            env,
                            info["SMILES"],
                            stop_probability=0.0,
                            template_mask_kind=template_mask_kind,
                        )
                    except NoValidActionError:
                        steps_done -= 1
                        episode_len -= 1
                        break
                else:
                    if hasattr(env, "enable"):
                        env.enable()
                    action = agent.get_action(state)
                wandb.log(
                    {
                        "train/global_step": steps_done,
                        "train/eps_random_action": float(eps),
                        "train/used_random_action": float(use_random),
                    },
                    step=steps_done,
                )
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
                    template_mask_kind=template_mask_kind,
                )
            if steps_done % save_freq == 0:
                agent.save_model(
                    str(checkpoint_dir / f"checkpoint_{steps_done}.tar"),
                    steps_done,
                    episode_count,
                    replay_buffer,
                    include_replay_buffer=save_replay_buffer_in_checkpoints,
                )
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
    agent.save_model(
        str(checkpoint_dir / "final_model.pth"),
        steps_done,
        episode_count,
        replay_buffer,
        include_replay_buffer=save_replay_buffer_in_checkpoints,
    )
    return agent

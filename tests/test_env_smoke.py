"""Smoke tests for the unified GenMolRL environment."""

from __future__ import annotations

import gymnasium as gym

from genmolrl.algorithms.common import env_kwargs
from genmolrl.config import load_config
from genmolrl.registry import ENV_ID, register_envs


def test_ppo_uni_env_reset():
    register_envs()
    cfg = load_config("configs/ppo_uni_masked_delta_qed.yaml")
    env = gym.make(ENV_ID, **env_kwargs(cfg))
    obs, info = env.reset(seed=42)
    assert obs.shape[0] == 1040
    assert info["SMILES"]
    assert env.unwrapped.action_masks().shape[0] == 16
    env.close()


def test_td3_td3_template_mask_kind_matches_yaml_masking():
    register_envs()
    cfg = load_config("configs/td3_uni_continuous_masked_delta_qed.yaml")
    kwargs = env_kwargs(cfg)
    kwargs["algorithm_family"] = "td3_pgfs"
    # Mirror the TD3 train flow: action mask is appended to the observation.
    kwargs["append_action_mask_to_obs"] = True
    env = gym.make(ENV_ID, **kwargs)
    from genmolrl.algorithms.td3.mask_kind import td3_template_mask_kind

    assert td3_template_mask_kind(env) == env.unwrapped.mask_provider.mode
    env.close()


def test_td3_uni_discrete_env_reset():
    register_envs()
    cfg = load_config("configs/td3_uni_discrete_masked_delta_qed.yaml")
    kwargs = env_kwargs(cfg)
    kwargs["algorithm_family"] = "td3_pgfs"
    kwargs["append_action_mask_to_obs"] = True
    env = gym.make(ENV_ID, **kwargs)
    obs, info = env.reset(seed=42)
    # 1024-dim morgan FP + 16-dim action mask (15 templates + Stop).
    assert obs.shape[0] == 1024 + 16
    assert env.unwrapped.base_obs_dim == 1024
    assert info["SMILES"]
    assert env.unwrapped.action_design == "td3_uni_discrete"
    env.close()


def test_td3_uni_env_reset():
    register_envs()
    cfg = load_config("configs/td3_uni_continuous_masked_delta_qed.yaml")
    kwargs = env_kwargs(cfg)
    kwargs["algorithm_family"] = "td3_pgfs"
    kwargs["append_action_mask_to_obs"] = True
    env = gym.make(ENV_ID, **kwargs)
    obs, info = env.reset(seed=42)
    assert obs.shape[0] == 1024 + 16
    assert env.unwrapped.base_obs_dim == 1024
    assert info["SMILES"]
    env.close()


def test_td3_agent_dim_disentanglement():
    """When ``append_action_mask_to_obs`` inflates the observation, the TD3
    agent must keep its R2 head at the morgan-FP width (not the obs width)."""
    register_envs()
    from genmolrl.algorithms.td3.agent import TD3Agent
    from genmolrl.algorithms.td3.train import _td3_fp_dim

    # --- Discrete uni: R2 dim is always 0 ---
    cfg_d = load_config("configs/td3_uni_discrete_masked_delta_qed.yaml")
    kwargs_d = env_kwargs(cfg_d)
    kwargs_d["algorithm_family"] = "td3_pgfs"
    kwargs_d["append_action_mask_to_obs"] = True
    env_d = gym.make(ENV_ID, **kwargs_d)
    agent_d = TD3Agent(env_d, max_timesteps=10, start_timesteps=1)
    assert agent_d.state_dim == 1024 + 16
    assert agent_d.continuous_r2_dim == 0
    assert _td3_fp_dim(env_d) == 1024
    env_d.close()

    # --- Continuous PGFS-R2: R2 dim must be 1024 (FP), NOT 1040 (obs) ---
    cfg_c = load_config("configs/td3_uni_continuous_masked_delta_qed.yaml")
    kwargs_c = env_kwargs(cfg_c)
    kwargs_c["algorithm_family"] = "td3_pgfs"
    kwargs_c["append_action_mask_to_obs"] = True
    env_c = gym.make(ENV_ID, **kwargs_c)
    agent_c = TD3Agent(env_c, max_timesteps=10, start_timesteps=1)
    assert agent_c.state_dim == 1024 + 16
    assert agent_c.continuous_r2_dim == 1024  # critical: NOT 1040
    assert _td3_fp_dim(env_c) == 1024
    env_c.close()


def test_td3_agent_arch_alignment_with_ppo():
    """The TD3 YAMLs now configure ``[64, 64]`` Tanh actor + critic to mirror
    PPO/A2C's SB3 ``MlpPolicy`` defaults. This test guards both:
      1. The new YAML keys (``actor_hidden_dims`` / ``critic_hidden_dims``
         / ``activation``) flow through to the actual ``nn.Module`` graph.
      2. The legacy (unconfigured) path still produces the original
         ``[256, 128, 128]`` ReLU actor, so any caller that omits these
         keys reproduces the pre-fix behavior bit-for-bit.
    """
    import torch.nn as nn

    register_envs()
    from genmolrl.algorithms.td3.agent import TD3Agent

    # --- Aligned config: [64, 64] Tanh, both heads ---
    cfg = load_config("configs/td3_uni_discrete_masked_delta_qed.yaml")
    assert cfg["td3"]["actor_hidden_dims"] == [64, 64]
    assert cfg["td3"]["critic_hidden_dims"] == [64, 64]
    assert cfg["td3"]["activation"] == "tanh"

    kwargs = env_kwargs(cfg)
    kwargs["algorithm_family"] = "td3_pgfs"
    kwargs["append_action_mask_to_obs"] = True
    env = gym.make(ENV_ID, **kwargs)
    agent = TD3Agent(
        env,
        max_timesteps=10,
        start_timesteps=1,
        actor_hidden_dims=cfg["td3"]["actor_hidden_dims"],
        critic_hidden_dims=cfg["td3"]["critic_hidden_dims"],
        activation=cfg["td3"]["activation"],
    )

    actor_layers = list(agent.actor.f_net.network)
    actor_linears = [m for m in actor_layers if isinstance(m, nn.Linear)]
    assert [l.out_features for l in actor_linears] == [64, 64, agent.template_dim]
    assert any(isinstance(m, nn.Tanh) for m in actor_layers), "Tanh activation missing in actor"
    assert not any(isinstance(m, nn.ReLU) for m in actor_layers), "ReLU should not appear when activation=tanh"

    critic_layers = list(agent.critic1.network)
    critic_linears = [m for m in critic_layers if isinstance(m, nn.Linear)]
    assert [l.out_features for l in critic_linears] == [64, 64, 1]
    assert any(isinstance(m, nn.Tanh) for m in critic_layers), "Tanh activation missing in critic"
    env.close()

    # --- Legacy path: no knobs set => [256, 128, 128] ReLU actor ---
    legacy_kwargs = env_kwargs(cfg)
    legacy_kwargs["algorithm_family"] = "td3_pgfs"
    legacy_kwargs["append_action_mask_to_obs"] = True
    legacy_env = gym.make(ENV_ID, **legacy_kwargs)
    legacy_agent = TD3Agent(legacy_env, max_timesteps=10, start_timesteps=1)
    legacy_layers = list(legacy_agent.actor.f_net.network)
    legacy_linears = [m for m in legacy_layers if isinstance(m, nn.Linear)]
    assert [l.out_features for l in legacy_linears] == [256, 128, 128, legacy_agent.template_dim]
    assert any(isinstance(m, nn.ReLU) for m in legacy_layers), "Legacy default should still be ReLU"
    legacy_env.close()


def test_non_neural_search_smoke():
    from genmolrl.algorithms.search import train

    for mode, cfg_path in [
        ("random_search", "configs/random_search_uni_delta_qed.yaml"),
        ("greedy_search", "configs/greedy_search_uni_delta_qed.yaml"),
        ("exhausted_search", "configs/exhausted_search_uni_delta_qed.yaml"),
    ]:
        cfg = load_config(cfg_path)
        cfg["search"]["max_paths"] = 1
        cfg["search"]["max_attempts"] = 5
        cfg["max_episode_len"] = 2
        cfg["search"]["max_starts"] = 1
        cfg["search"]["use_wandb"] = False
        cfg["search"]["results_file"] = f"runs/{mode}_test_smoke.txt"
        result = train(cfg, f"{mode}_smoke", mode=mode)
        assert result["saved_paths"] >= 1
        assert result["results_file"].endswith(".txt")


def test_random_and_greedy_search_smoke():
    test_non_neural_search_smoke()

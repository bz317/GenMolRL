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

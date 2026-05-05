"""Smoke tests for the unified GenMolRL environment."""

from __future__ import annotations

import gymnasium as gym

from genmolrl.algorithms.common import env_kwargs
from genmolrl.config import load_config
from genmolrl.registry import ENV_ID, register_envs


def test_ppo_uni_env_reset():
    register_envs()
    cfg = load_config("GenMolRL/configs/ppo_uni_masked_delta_qed.yaml")
    env = gym.make(ENV_ID, **env_kwargs(cfg))
    obs, info = env.reset(seed=42)
    assert obs.shape[0] == 1040
    assert info["SMILES"]
    assert env.unwrapped.action_masks().shape[0] == 16
    env.close()


def test_td3_uni_env_reset():
    register_envs()
    cfg = load_config("GenMolRL/configs/td3_uni_masked_delta_qed.yaml")
    kwargs = env_kwargs(cfg)
    kwargs["algorithm_family"] = "td3_pgfs"
    kwargs["append_action_mask_to_obs"] = False
    env = gym.make(ENV_ID, **kwargs)
    obs, info = env.reset(seed=42)
    assert obs.shape[0] == 1024
    assert info["SMILES"]
    env.close()


def test_random_and_greedy_search_smoke():
    from genmolrl.algorithms.search import train

    for mode, cfg_path in [
        ("random_search", "GenMolRL/configs/random_search_uni_delta_qed.yaml"),
        ("greedy_search", "GenMolRL/configs/greedy_search_uni_delta_qed.yaml"),
    ]:
        cfg = load_config(cfg_path)
        cfg["search"]["max_paths"] = 1
        cfg["search"]["max_attempts"] = 5
        cfg["search"]["max_steps"] = 2
        cfg["search"]["use_wandb"] = False
        cfg["search"]["results_file"] = f"GenMolRL/runs/{mode}_test_smoke.txt"
        result = train(cfg, f"{mode}_smoke", mode=mode)
        assert result["saved_paths"] >= 1
        assert result["results_file"].endswith(".txt")

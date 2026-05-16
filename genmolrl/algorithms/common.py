"""Shared trainer helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

import wandb
from genmolrl.config import project_root, resolve_path
from genmolrl.logging.callbacks import EpisodeWandbCallback, EvaluationWandbCallback
from genmolrl.logging.wandb_metrics import define_ppo_compatible_metrics
from genmolrl.registry import ENV_ID, register_envs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir(run_id: str) -> Path:
    path = project_root() / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_wandb(config: dict, algorithm: str, experiment_name: str):
    project = os.getenv("WANDB_PROJECT", config.get("project", "GenMolRL"))
    init_kw = {
        "project": project,
        "name": experiment_name,
        "job_type": f"train-{algorithm.lower()}",
        "save_code": True,
        "resume": "allow" if config.get("wandb_resume") else "never",
        "config": config,
    }
    entity = os.getenv("WANDB_ENTITY") or config.get("entity")
    if entity:
        init_kw["entity"] = entity
    run = wandb.init(**init_kw)
    define_ppo_compatible_metrics()
    return run


def env_kwargs(config: dict, *, eval_env: bool = False) -> dict:
    dataset = config["dataset"]
    env_cfg = config["env"]
    max_episode_len = config.get("max_episode_len", env_cfg.get("max_episode_len", env_cfg.get("max_steps", 5)))
    algorithm_family = env_cfg["algorithm_family"]
    # For `sb3_multidiscrete` (bi-reaction) the R2 axis is sized by the
    # reactant pool, so the eval env MUST share the training pool or the
    # MultiDiscrete([T+1, R2]) action head shape will mismatch. Source the
    # held-out start molecules from `dataset.test_file` separately via
    # `start_pool_file`. Other families (sb3_discrete, td3_pgfs,
    # graphtransrl, graphtransppo) keep the legacy behaviour where the
    # eval env loads `test_file` directly as its reactant pool.
    # ``dataset.eval_r2_pool`` selects the R(2) candidate pool the eval env
    # builds its ReactionManager / FAISS index on. ``test`` (default) swaps
    # ``reactant_file`` to ``test_file`` at eval time — disjoint test R(2)s,
    # matches PGFS and PPO-Bi ``eval_r2_pool=test`` (the encoder-style
    # convention used by all GenMolRL methods). ``train`` keeps the training
    # reactant pool at eval (apples-to-apples vs PPO-Bi ``eval_r2_pool=train``
    # / legacy ``r2_arch=lookup`` baselines); R(1) still iterates the test
    # pool via ``start_pool_file``. ``sb3_multidiscrete`` overrides to
    # ``train`` regardless because its action head is sized by the training
    # reactant pool.
    eval_r2_pool = str(dataset.get("eval_r2_pool", "test")).lower()
    if eval_r2_pool not in {"test", "train"}:
        raise ValueError(
            f"dataset.eval_r2_pool must be 'test' or 'train', got {eval_r2_pool!r}"
        )
    if eval_env and algorithm_family == "sb3_multidiscrete":
        reactant_file = dataset["training_file"]
        start_pool_file = dataset.get("test_file")
    elif eval_env and eval_r2_pool == "train":
        reactant_file = dataset["training_file"]
        start_pool_file = dataset.get("test_file")
    else:
        reactant_file = dataset["training_file"] if not eval_env else dataset.get("test_file")
        start_pool_file = None
    if reactant_file is None:
        raise KeyError("dataset.test_file must be set for evaluation environments")
    kwargs = {
        "reactant_file": resolve_path(reactant_file),
        "template_file": resolve_path(dataset["templates_file"]),
        "reaction_mode": config["reaction_mode"],
        "algorithm_family": algorithm_family,
        "action_design": env_cfg.get("action_design", "discrete"),
        "masking": config["masking"],
        "reward": config["reward"],
        "max_steps": max_episode_len,
        "use_stop_action": env_cfg.get("use_stop_action", True),
        "stop_early_penalty": env_cfg.get("stop_early_penalty", 0.0),
        "stop_penalty_until_step": env_cfg.get("stop_penalty_until_step", -1),
        "invalid_reaction_penalty": env_cfg.get("invalid_reaction_penalty", -1.0),
        "reward_round_digits": env_cfg.get("reward_round_digits"),
        "info_qed_round_digits": env_cfg.get("info_qed_round_digits"),
        "render_mode": "human" if eval_env else None,
        "append_action_mask_to_obs": env_cfg.get("append_action_mask_to_obs"),
    }
    if start_pool_file is not None:
        kwargs["start_pool_file"] = resolve_path(start_pool_file)
    if eval_env:
        kwargs["start_strategy"] = "cycle_pool"
    elif dataset.get("fixed_start_smiles"):
        kwargs["start_strategy"] = "fixed"
        kwargs["fixed_start_smiles"] = dataset["fixed_start_smiles"]
    else:
        kwargs["start_strategy"] = dataset.get("start_strategy", "random_pool")
        if dataset.get("start_smiles_file"):
            kwargs["start_smiles_file"] = resolve_path(dataset["start_smiles_file"])
    return kwargs


def make_envs(config: dict, seed: int):
    register_envs()
    train_env = make_vec_env(
        ENV_ID,
        n_envs=int(config["training"].get("n_envs", 1)),
        env_kwargs=env_kwargs(config, eval_env=False),
        monitor_dir=str(project_root() / "runs" / "monitors" / "train"),
        seed=seed,
    )
    eval_env = make_vec_env(
        ENV_ID,
        n_envs=1,
        env_kwargs=env_kwargs(config, eval_env=True),
        monitor_dir=str(project_root() / "runs" / "monitors" / "eval"),
        seed=seed + 1,
    )
    return train_env, eval_env


def sb3_callbacks(config: dict, run_id: str, eval_env=None):
    paths = run_dir(run_id)
    ckpt_dir = paths / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        WandbCallback(
            model_save_path=str(paths / "wandb_model"),
            model_save_freq=int(config["callbacks"].get("model_save_freq", 100000)),
            gradient_save_freq=int(config["callbacks"].get("gradient_save_freq", 100)),
            log="all",
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=int(config["callbacks"].get("model_save_freq", 100000)),
            save_path=str(ckpt_dir),
            name_prefix=config.get("algorithm", "model").lower(),
        ),
        EpisodeWandbCallback(),
    ]
    if eval_env is not None:
        callbacks.append(
            EvaluationWandbCallback(
                eval_env,
                eval_freq=int(config["training"].get("eval_freq", 10000)),
            )
        )
    return CallbackList(callbacks)

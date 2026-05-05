"""PPO training entry point."""

from __future__ import annotations

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

import wandb
from genmolrl.algorithms.common import init_wandb, make_envs, sb3_callbacks, set_seed


def train(config: dict, experiment_name: str):
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)
    config = dict(config)
    config["algorithm"] = "PPO"
    run = init_wandb(config, "PPO", experiment_name)
    train_env, _eval_env = make_envs(config, seed)

    ppo_cfg = config["ppo"]
    cls = MaskablePPO if config["masking"] != "none" else PPO
    model = cls(
        "MlpPolicy",
        train_env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        tensorboard_log=str(run.dir),
        verbose=1,
        seed=seed,
    )
    model.learn(
        total_timesteps=int(config["training"].get("total_timesteps", 1_000_000)),
        callback=sb3_callbacks(config, run.id),
        progress_bar=bool(ppo_cfg.get("progress_bar", False)),
    )
    model.save(str(wandb.run.dir + "/final_model"))
    return model

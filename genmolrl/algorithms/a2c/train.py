"""A2C training entry point."""

from __future__ import annotations

from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import wandb
from genmolrl.algorithms.a2c.policies import MaskedActorCriticPolicy
from genmolrl.algorithms.common import init_wandb, make_envs, sb3_callbacks, set_seed


def train(config: dict, experiment_name: str):
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)
    config = dict(config)
    config["algorithm"] = "A2C"
    run = init_wandb(config, "A2C", experiment_name)
    train_env, eval_env = make_envs(config, seed)

    a2c_cfg = config["a2c"]
    policy = MaskedActorCriticPolicy if config["masking"] != "none" else "MlpPolicy"
    policy_kwargs = {}
    if policy is MaskedActorCriticPolicy:
        policy_kwargs = {"optimizer_class": RMSpropTFLike, "optimizer_kwargs": {"eps": 1e-5}}
    model = A2C(
        policy,
        train_env,
        learning_rate=float(a2c_cfg.get("learning_rate", 7e-4)),
        n_steps=int(a2c_cfg.get("n_steps", 5)),
        gamma=float(a2c_cfg.get("gamma", 0.99)),
        gae_lambda=float(a2c_cfg.get("gae_lambda", 1.0)),
        ent_coef=float(a2c_cfg.get("ent_coef", 0.0)),
        vf_coef=float(a2c_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(a2c_cfg.get("max_grad_norm", 0.5)),
        tensorboard_log=str(run.dir),
        verbose=1,
        seed=seed,
        device=a2c_cfg.get("device", "cpu"),
        policy_kwargs=policy_kwargs,
    )
    model.learn(
        total_timesteps=int(config["training"].get("total_timesteps", 1_000_000)),
        callback=sb3_callbacks(config, run.id, eval_env),
        progress_bar=bool(a2c_cfg.get("progress_bar", False)),
    )
    model.save(str(wandb.run.dir + "/final_model"))
    return model

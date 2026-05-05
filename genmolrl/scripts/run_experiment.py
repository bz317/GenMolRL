"""Unified GenMolRL experiment launcher."""

from __future__ import annotations

import argparse
import os

from genmolrl.config import load_config


def _trainer(algorithm: str):
    if algorithm == "ppo":
        from genmolrl.algorithms.ppo.train import train
    elif algorithm == "a2c":
        from genmolrl.algorithms.a2c.train import train
    elif algorithm == "td3":
        from genmolrl.algorithms.td3.train import train
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GenMolRL experiment")
    parser.add_argument("--algorithm", choices=["ppo", "a2c", "td3"], required=True)
    parser.add_argument("--reaction-mode", choices=["uni", "bi"])
    parser.add_argument("--masking", choices=["substructure", "reaction_valid", "r2_available", "none"])
    parser.add_argument("--reward", choices=["delta_qed", "final_qed"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-name")
    args = parser.parse_args()

    config = load_config(args.config)
    config["algorithm"] = args.algorithm.upper()
    if args.reaction_mode:
        config["reaction_mode"] = args.reaction_mode
    if args.masking:
        config["masking"] = args.masking
    if args.reward:
        config["reward"] = args.reward

    experiment_name = args.experiment_name or config.get(
        "experiment_name",
        f"{args.algorithm.upper()}_{config['reaction_mode']}_{config['reward']}",
    )
    os.environ.setdefault("WANDB_PROJECT", config.get("project", "MolSynthRL"))
    _trainer(args.algorithm)(config, experiment_name)


if __name__ == "__main__":
    main()

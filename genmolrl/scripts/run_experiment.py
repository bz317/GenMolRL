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
    elif algorithm in {"random_search", "greedy_search", "exhausted_search"}:
        from genmolrl.algorithms.search import train as search_train

        def train(config: dict, experiment_name: str):
            return search_train(config, experiment_name, mode=algorithm)
    elif algorithm == "graphtransrl":
        from genmolrl.methods.graphtransrl_adapter import GraphTransRLAdapter

        train = GraphTransRLAdapter.train
    elif algorithm == "graphtransppo":
        from genmolrl.methods.graphtransppo_adapter import GraphTransPPOAdapter

        train = GraphTransPPOAdapter.train
    elif algorithm == "ppo_bi":
        from genmolrl.methods.ppo_bi_adapter import PPOBiAdapter

        train = PPOBiAdapter.train
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GenMolRL experiment")
    parser.add_argument(
        "--algorithm",
        choices=[
            "ppo",
            "a2c",
            "td3",
            "random_search",
            "greedy_search",
            "exhausted_search",
            "graphtransrl",
            "graphtransppo",
            "ppo_bi",
        ],
        required=True,
    )
    parser.add_argument("--reaction-mode", choices=["uni", "bi"])
    parser.add_argument("--masking", choices=["substructure", "reaction_valid", "r2_available", "none"])
    parser.add_argument("--reward", choices=["delta_qed", "final_qed"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--max-episode-len", type=int, help="Override maximum trajectory length")
    parser.add_argument("--training-file", help="Override dataset.training_file from the config")
    parser.add_argument("--test-file", help="Override dataset.test_file from the config")
    parser.add_argument("--templates-file", help="Override dataset.templates_file from the config")
    parser.add_argument(
        "--greedy-mode",
        choices=["best_action", "positive_delta_only"],
        help="Override search.greedy_mode for greedy_search",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["algorithm"] = args.algorithm.upper()
    if args.reaction_mode:
        config["reaction_mode"] = args.reaction_mode
    if args.masking:
        config["masking"] = args.masking
    if args.reward:
        config["reward"] = args.reward
    if args.max_episode_len is not None:
        config["max_episode_len"] = args.max_episode_len
    dataset_overrides = {
        "training_file": args.training_file,
        "test_file": args.test_file,
        "templates_file": args.templates_file,
    }
    if any(value is not None for value in dataset_overrides.values()):
        config.setdefault("dataset", {})
        for key, value in dataset_overrides.items():
            if value is not None:
                config["dataset"][key] = value
    if args.greedy_mode is not None:
        config.setdefault("search", {})["greedy_mode"] = args.greedy_mode

    experiment_name = args.experiment_name or config.get(
        "experiment_name",
        f"{args.algorithm.upper()}_{config['reaction_mode']}_{config['reward']}",
    )
    os.environ.setdefault("WANDB_PROJECT", config.get("project", "GenMolRL"))
    _trainer(args.algorithm)(config, experiment_name)


if __name__ == "__main__":
    main()

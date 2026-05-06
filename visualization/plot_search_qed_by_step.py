#!/usr/bin/env python3
"""Plot QED distributions by reaction step for search baselines.

The plot uses one final/best molecule per test start. Greedy/random have one
saved final trajectory per start molecule. Exhaustive search can have many
paths and intermediate points per start, so it contributes only the best QED
available for that start across steps 1-5. Starts whose best option is taking no
action are shown in the step-0 column.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("Exhaustive", "Greedy", "Random", "PPO", "A2C")
COLORS = {
    "Exhaustive": "#4C78A8",
    "Greedy": "#F58518",
    "Random": "#54A24B",
    "PPO": "#B279A2",
    "A2C": "#E45756",
}


def parse_exhaustive_best_action_steps(
    path: Path,
    max_step: int,
) -> tuple[dict[int, list[float]], dict[str, int]]:
    values = {step: [] for step in range(0, max_step + 1)}
    best_by_start: dict[str, tuple[int, float]] = {}
    start_qed_by_start: dict[str, float] = {}
    path_start_by_id: dict[int, str] = {}
    in_steps = False
    header: list[str] | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[steps]":
                in_steps = True
                continue
            if not in_steps or not line:
                continue

            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            row = dict(zip(header, parts))

            try:
                path_id = int(row["path_id"])
                step = int(row["step"])
                qed = float(row["qed"])
            except (KeyError, ValueError):
                continue
            if step == 0:
                start_smiles = row["product"]
                path_start_by_id[path_id] = start_smiles
                start_qed_by_start.setdefault(start_smiles, qed)
                continue

            if not 1 <= step <= max_step:
                continue

            start_smiles = path_start_by_id.get(path_id)
            if start_smiles is None:
                continue
            current_best = best_by_start.get(start_smiles)
            if current_best is None or qed > current_best[1]:
                best_by_start[start_smiles] = (step, qed)

    no_action_best = 0
    for start_smiles, start_qed in start_qed_by_start.items():
        best = best_by_start.get(start_smiles)
        if best is None or start_qed >= best[1]:
            no_action_best += 1
            values[0].append(start_qed)
            continue
        step, qed = best
        values[step].append(qed)

    return values, {
        "starts_seen": len(start_qed_by_start),
        "no_action_best": no_action_best,
        "action_best": sum(len(values[step]) for step in values),
    }


def parse_trajectory_finals(path: Path, max_step: int) -> dict[int, list[float]]:
    values = {step: [] for step in range(0, max_step + 1)}
    in_trajectories = False
    header: list[str] | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[trajectories]":
                in_trajectories = True
                continue
            if not in_trajectories or not line:
                continue

            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            row = dict(zip(header, parts))

            try:
                step = int(row["num_reactions"])
                qed = float(row["final_qed"])
            except (KeyError, ValueError):
                continue
            if 1 <= step <= max_step:
                values[step].append(qed)

    return values


def parse_cached_policy_eval(path: Path, max_step: int) -> dict[int, list[float]] | None:
    if not path.is_file():
        return None
    values = {step: [] for step in range(0, max_step + 1)}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                step = int(row["num_reactions"])
                qed = float(row["final_qed"])
            except (KeyError, ValueError):
                continue
            if 0 <= step <= max_step:
                values[step].append(qed)
    return values


def evaluate_sb3_policy(
    *,
    repo_root: Path,
    method: str,
    model_path: Path,
    config_path: Path,
    cache_path: Path,
    max_step: int,
) -> dict[int, list[float]]:
    cached = parse_cached_policy_eval(cache_path, max_step)
    if cached is not None:
        return cached

    sys.path.insert(0, str(repo_root))
    from genmolrl.algorithms.common import env_kwargs
    from genmolrl.config import load_config
    from genmolrl.envs.molecule_design_env import MoleculeDesignEnv

    config = load_config(config_path)
    config["algorithm"] = method
    env = MoleculeDesignEnv(**env_kwargs(config, eval_env=True))
    env.start_strategy.reset_cycle()
    n_eval = env.start_strategy.num_starts()

    if method == "PPO":
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(model_path, device="cpu")
    elif method == "A2C":
        from stable_baselines3 import A2C

        # Import registers the custom policy class needed to unpickle the model.
        import genmolrl.algorithms.a2c.policies  # noqa: F401

        model = A2C.load(model_path, device="cpu")
    else:
        raise ValueError(f"Unsupported SB3 method: {method}")

    values = {step: [] for step in range(0, max_step + 1)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "episode", "start_smiles", "final_smiles", "initial_qed", "final_qed", "num_reactions"])
        for episode in range(n_eval):
            obs, info = env.reset()
            start_smiles = str(info["SMILES"])
            initial_qed = float(info["QED"])
            final_smiles = start_smiles
            final_qed = initial_qed
            num_reactions = 0
            done = False

            while not done:
                if method == "PPO":
                    action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                if info.get("stop") or info.get("reaction_failed") or info.get("bad_template_index") is not None:
                    final_smiles = str(info.get("SMILES") or final_smiles)
                    final_qed = float(info.get("QED", final_qed))
                    break
                num_reactions = int(info.get("step", num_reactions))
                final_smiles = str(info["SMILES"])
                final_qed = float(info["QED"])

            if 0 <= num_reactions <= max_step:
                values[num_reactions].append(final_qed)
            writer.writerow([method, episode, start_smiles, final_smiles, initial_qed, final_qed, num_reactions])

    env.close()
    return values


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def write_summary_csv(
    output_path: Path,
    grouped_values: dict[str, dict[int, list[float]]],
    max_step: int,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "step", "n", "mean", "median", "q1", "q3", "min", "max"])
        for method in METHODS:
            for step in range(0, max_step + 1):
                values = grouped_values[method][step]
                if values:
                    writer.writerow(
                        [
                            method,
                            step,
                            len(values),
                            sum(values) / len(values),
                            median(values),
                            quantile(values, 0.25),
                            quantile(values, 0.75),
                            min(values),
                            max(values),
                        ]
                    )
                else:
                    writer.writerow([method, step, 0, "", "", "", "", "", ""])


def plot_boxplots(
    output_path: Path,
    grouped_values: dict[str, dict[int, list[float]]],
    max_step: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        max_step + 1,
        figsize=(3.3 * (max_step + 1), 4.8),
        sharey=True,
        constrained_layout=True,
    )
    if max_step == 0:
        axes = [axes]

    for axis_index, axis in enumerate(axes):
        step = axis_index
        present = [
            (method, grouped_values[method][step])
            for method in METHODS
            if grouped_values[method][step]
        ]
        data = [values for _, values in present]
        all_positions = [0.72, 0.86, 1.0, 1.14, 1.28]
        position_by_method = dict(zip(METHODS, all_positions))
        positions = [position_by_method[method] for method, _ in present]
        box = axis.boxplot(
            data,
            positions=positions,
            widths=0.12,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch, (method, _) in zip(box["boxes"], present):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.72)

        if step == 0:
            axis.set_title("No action")
        else:
            axis.set_title(f"After {step} reaction{'s' if step > 1 else ''}")
        axis.set_xlabel("Method")
        axis.set_xlim(0.62, 1.38)
        axis.set_xticks(all_positions)
        axis.set_xticklabels(METHODS)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        for method in METHODS:
            idx = position_by_method[method]
            values = grouped_values[method][step]
            axis.text(
                idx,
                0.02,
                f"n={len(values)}\nvalues",
                ha="center",
                va="bottom",
                fontsize=8,
                transform=axis.get_xaxis_transform(),
            )

    axes[0].set_ylabel("QED")
    fig.suptitle(
        "Final-molecule QED distributions by reaction count",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--exhaustive",
        type=Path,
        default=repo_root / "runs/exhausted_search_uni_results_5_len.txt",
    )
    parser.add_argument(
        "--greedy",
        type=Path,
        default=repo_root / "runs/greedy_search_uni_results.txt",
    )
    parser.add_argument(
        "--random",
        type=Path,
        default=repo_root / "runs/random_search_uni_results.txt",
    )
    parser.add_argument(
        "--ppo-model",
        type=Path,
        default=repo_root / "runs/3zd846ff/wandb_model/model.zip",
    )
    parser.add_argument(
        "--a2c-model",
        type=Path,
        default=repo_root / "runs/en3i9xg8/wandb_model/model.zip",
    )
    parser.add_argument(
        "--ppo-config",
        type=Path,
        default=repo_root / "configs/ppo_uni_masked_delta_qed.yaml",
    )
    parser.add_argument(
        "--a2c-config",
        type=Path,
        default=repo_root / "configs/a2c_uni_masked_delta_qed.yaml",
    )
    parser.add_argument(
        "--ppo-cache",
        type=Path,
        default=repo_root / "visualization/ppo_3zd846ff_eval_trajectories.csv",
    )
    parser.add_argument(
        "--a2c-cache",
        type=Path,
        default=repo_root / "visualization/a2c_en3i9xg8_eval_trajectories.csv",
    )
    parser.add_argument("--max-step", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "visualization/search_qed_by_step_boxplots.png",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=repo_root / "visualization/search_qed_by_step_summary.csv",
    )
    parser.add_argument(
        "--exhaustive-stats",
        type=Path,
        default=repo_root / "visualization/search_qed_exhaustive_best_action_stats.txt",
    )
    args = parser.parse_args()

    if args.max_step < 1 or args.max_step > 5:
        raise ValueError("--max-step must be between 1 and 5")

    exhaustive_values, exhaustive_stats = parse_exhaustive_best_action_steps(
        args.exhaustive,
        args.max_step,
    )
    grouped_values = {
        "Exhaustive": exhaustive_values,
        "Greedy": parse_trajectory_finals(args.greedy, args.max_step),
        "Random": parse_trajectory_finals(args.random, args.max_step),
        "PPO": evaluate_sb3_policy(
            repo_root=repo_root,
            method="PPO",
            model_path=args.ppo_model,
            config_path=args.ppo_config,
            cache_path=args.ppo_cache,
            max_step=args.max_step,
        ),
        "A2C": evaluate_sb3_policy(
            repo_root=repo_root,
            method="A2C",
            model_path=args.a2c_model,
            config_path=args.a2c_config,
            cache_path=args.a2c_cache,
            max_step=args.max_step,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.exhaustive_stats.parent.mkdir(parents=True, exist_ok=True)
    plot_boxplots(args.output, grouped_values, args.max_step)
    write_summary_csv(args.summary_csv, grouped_values, args.max_step)
    args.exhaustive_stats.write_text(
        "\n".join(f"{key}: {value}" for key, value in exhaustive_stats.items()) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.exhaustive_stats}")
    print(
        "Exhaustive starts best by taking no action: "
        f"{exhaustive_stats['no_action_best']} / {exhaustive_stats['starts_seen']}"
    )


if __name__ == "__main__":
    main()

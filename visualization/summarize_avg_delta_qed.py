#!/usr/bin/env python3
"""Write average delta-QED table for search, PPO, and A2C methods.

The default path choices intentionally mirror `plot_search_qed_by_step.py`.
For exhaustive search, one value is reported per test start: the best reachable
QED across all saved paths, or the start QED if no action improves it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median

from plot_search_qed_by_step import METHODS, evaluate_sb3_policy


def _section_rows(path: Path, section_name: str):
    in_section = False
    header: list[str] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == f"[{section_name}]":
                in_section = True
                header = None
                continue
            if in_section and line.startswith("[") and line.endswith("]"):
                break
            if not in_section or not line:
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            yield dict(zip(header, parts))


def parse_search_trajectory_deltas(path: Path) -> list[float]:
    deltas: list[float] = []
    for row in _section_rows(path, "trajectories"):
        try:
            deltas.append(float(row["final_qed"]) - float(row["initial_qed"]))
        except (KeyError, ValueError):
            continue
    return deltas


def parse_exhaustive_best_deltas(path: Path, max_step: int) -> list[float]:
    start_qed_by_start: dict[str, float] = {}
    best_qed_by_start: dict[str, float] = {}
    path_start_by_id: dict[int, str] = {}

    for row in _section_rows(path, "steps"):
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
            best_qed_by_start.setdefault(start_smiles, qed)
            continue
        if not 1 <= step <= max_step:
            continue

        start_smiles = path_start_by_id.get(path_id)
        if start_smiles is None:
            continue
        best_qed_by_start[start_smiles] = max(best_qed_by_start.get(start_smiles, qed), qed)

    return [
        best_qed_by_start.get(start_smiles, start_qed) - start_qed
        for start_smiles, start_qed in start_qed_by_start.items()
    ]


def parse_policy_cache_deltas(path: Path) -> list[float]:
    deltas: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                deltas.append(float(row["final_qed"]) - float(row["initial_qed"]))
            except (KeyError, ValueError):
                continue
    return deltas


def ensure_policy_cache(
    *,
    repo_root: Path,
    method: str,
    model_path: Path,
    config_path: Path,
    cache_path: Path,
    max_step: int,
) -> None:
    if cache_path.is_file():
        return
    evaluate_sb3_policy(
        repo_root=repo_root,
        method=method,
        model_path=model_path,
        config_path=config_path,
        cache_path=cache_path,
        max_step=max_step,
    )


def summarize(deltas: list[float]) -> dict[str, float | int | str]:
    if not deltas:
        return {
            "n": 0,
            "avg_delta_qed": "",
            "median_delta_qed": "",
            "min_delta_qed": "",
            "max_delta_qed": "",
            "positive_delta_fraction": "",
            "negative_delta_fraction": "",
            "zero_delta_fraction": "",
        }
    return {
        "n": len(deltas),
        "avg_delta_qed": mean(deltas),
        "median_delta_qed": median(deltas),
        "min_delta_qed": min(deltas),
        "max_delta_qed": max(deltas),
        "positive_delta_fraction": sum(v > 0 for v in deltas) / len(deltas),
        "negative_delta_fraction": sum(v < 0 for v in deltas) / len(deltas),
        "zero_delta_fraction": sum(v == 0 for v in deltas) / len(deltas),
    }


def format_value(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_table(output_path: Path, summaries: dict[str, dict[str, float | int | str]]) -> None:
    columns = [
        "method",
        "n",
        "avg_delta_qed",
        "median_delta_qed",
        "min_delta_qed",
        "max_delta_qed",
        "positive_delta_fraction",
        "negative_delta_fraction",
        "zero_delta_fraction",
    ]
    rows = []
    for method in METHODS:
        summary = summaries[method]
        rows.append([method] + [format_value(summary[column]) for column in columns[1:]])

    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]
    lines = ["\t".join(column.ljust(widths[index]) for index, column in enumerate(columns))]
    lines.extend(
        "\t".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        default=repo_root / "visualization/avg_delta_qed_table.txt",
    )
    args = parser.parse_args()

    if args.max_step < 1 or args.max_step > 5:
        raise ValueError("--max-step must be between 1 and 5")

    ensure_policy_cache(
        repo_root=repo_root,
        method="PPO",
        model_path=args.ppo_model,
        config_path=args.ppo_config,
        cache_path=args.ppo_cache,
        max_step=args.max_step,
    )
    ensure_policy_cache(
        repo_root=repo_root,
        method="A2C",
        model_path=args.a2c_model,
        config_path=args.a2c_config,
        cache_path=args.a2c_cache,
        max_step=args.max_step,
    )

    deltas_by_method = {
        "Exhaustive": parse_exhaustive_best_deltas(args.exhaustive, args.max_step),
        "Greedy": parse_search_trajectory_deltas(args.greedy),
        "Random": parse_search_trajectory_deltas(args.random),
        "PPO": parse_policy_cache_deltas(args.ppo_cache),
        "A2C": parse_policy_cache_deltas(args.a2c_cache),
    }
    summaries = {method: summarize(deltas_by_method[method]) for method in METHODS}
    write_table(args.output, summaries)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Delete intermediate checkpoints under runs/, keeping multiples of INTERVAL steps.

Keeps SB3 CheckpointCallback zip files *_N_steps.zip when N % interval == 0,
TD3 checkpoint_N.tar likewise, GraphTransRL model_step_N.pt likewise.
Preserves wandb_model/model.zip and final_model / best_model files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _candidate_files(runs_root: Path) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    for path in runs_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if "checkpoints" in path.parts and name.endswith("_steps.zip"):
            m = re.fullmatch(r".+_(\d+)_steps\.zip", name)
            if m:
                out.append((path, int(m.group(1))))
        elif "td3_checkpoints" in path.parts:
            if name.startswith("checkpoint_") and name.endswith(".tar"):
                m = re.fullmatch(r"checkpoint_(\d+)\.tar", name)
                if m:
                    out.append((path, int(m.group(1))))
        elif name.startswith("model_step_") and name.endswith(".pt"):
            m = re.fullmatch(r"model_step_(\d+)\.pt", name)
            if m:
                out.append((path, int(m.group(1))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Path to GenMolRL runs directory",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100_000,
        help="Retain checkpoints only at this timestep multiple (default: 100000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be removed without deleting",
    )
    args = parser.parse_args()
    runs_root = args.runs_dir.resolve()
    if not runs_root.is_dir():
        raise SystemExit(f"Not a directory: {runs_root}")

    to_remove: list[Path] = []
    for path, steps in _candidate_files(runs_root):
        if steps <= 0:
            continue
        if steps % args.interval != 0:
            to_remove.append(path)

    to_remove.sort()
    nbytes = sum(p.stat().st_size for p in to_remove)

    print(f"Candidates to delete: {len(to_remove)} ({nbytes / 1e9:.3f} GiB)")
    for p in to_remove:
        prefix = "(dry-run) " if args.dry_run else ""
        print(f"  {prefix}{p}")

    if args.dry_run or not to_remove:
        return

    for p in to_remove:
        p.unlink()
    print(f"Deleted {len(to_remove)} files.")


if __name__ == "__main__":
    main()

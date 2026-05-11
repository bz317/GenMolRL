"""Smoke test: prove streaming exhausted_search writes leaves to disk.

Runs the SearchRunner.exhausted_search code path with the strictest possible
shape (depth = 1, 2 starts) so every DFS branch reaches a leaf in seconds.

Expectations:
  - both sidecar files (.steps.tmp / .trajectories.tmp) exist mid-run and grow
  - after both starts complete, the consolidated results file contains
    [summary] / [trajectories] / [steps] sections with > 0 rows in each
  - the sidecar files are cleaned up at the end
  - stdout prints per-start `[exhausted] starts=...` heartbeats

Run with:
    cd GenMolRL && WANDB_MODE=disabled PYTHONPATH=. PYTHONUNBUFFERED=1 \\
        python -u tools/smoke_stream_exhausted_bi.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from genmolrl.algorithms.search import SearchRunner


def main() -> None:
    cfg_path = ROOT / "configs" / "exhausted_search_bi_delta_qed.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["max_episode_len"] = 1
    cfg["search"]["max_starts"] = 2
    cfg["search"]["max_paths"] = None
    cfg["search"]["max_reactions"] = None
    cfg["search"]["max_r2_per_template"] = None
    cfg["search"]["use_wandb"] = False
    cfg["search"]["progress_every_starts"] = 1
    cfg["search"]["results_file"] = "runs/_smoke_exhausted_bi_stream.txt"

    print(f"Loaded cfg from {cfg_path}")
    print(f"max_episode_len={cfg['max_episode_len']} max_starts={cfg['search']['max_starts']}")

    t0 = time.monotonic()
    runner = SearchRunner(cfg, experiment_name="smoke_bi", mode="exhausted_search")
    print(f"runner init took {time.monotonic() - t0:.1f}s; result_file={runner.result_file}")
    print(f"sidecars: {runner._stream_paths}  {runner._stream_trajs}")
    print(f"sidecar exists pre-run: steps={Path(runner._stream_paths).exists()}  trajs={Path(runner._stream_trajs).exists()}")

    t0 = time.monotonic()
    summary = runner.run()
    elapsed = time.monotonic() - t0
    print(f"\nrun() returned after {elapsed:.1f}s")
    print(f"summary={summary}")

    result_path = runner.result_file
    text = result_path.read_text()

    steps_tmp = Path(runner._stream_paths)
    trajs_tmp = Path(runner._stream_trajs)
    print(f"sidecars after run: steps_exists={steps_tmp.exists()} trajs_exists={trajs_tmp.exists()}")

    assert "[summary]" in text, "missing [summary] section"
    assert "[trajectories]" in text, "missing [trajectories] section"
    assert "[steps]" in text, "missing [steps] section"
    assert text.index("[summary]") < text.index("[trajectories]") < text.index("[steps]")

    n_traj_data_rows = 0
    n_step_data_rows = 0
    sect = None
    for line in text.splitlines():
        if line.startswith("[trajectories]"):
            sect = "traj"
            continue
        if line.startswith("[steps]"):
            sect = "steps"
            continue
        if not line or line.startswith("path_id"):
            continue
        if sect == "traj":
            n_traj_data_rows += 1
        elif sect == "steps":
            n_step_data_rows += 1
    print(f"data rows: trajectories={n_traj_data_rows} steps={n_step_data_rows}")
    assert n_traj_data_rows > 0, "no trajectory rows written"
    assert n_step_data_rows > 0, "no step rows written"

    assert not steps_tmp.exists(), f"steps sidecar not cleaned: {steps_tmp}"
    assert not trajs_tmp.exists(), f"trajs sidecar not cleaned: {trajs_tmp}"

    print("OK streaming path works at depth=1, 2 starts")


if __name__ == "__main__":
    main()

"""Sweep the bi test set and report per-molecule template feasibility:

  metric A — R1-feasible templates per molecule (FULL test set sweep)
      counts how many of the 102 bi-templates have an R1 SMARTS that matches
      the current molecule. This is what `feasible_first_reactant_templates`
      returns and what the action mask exposes to the policy. Cheap (~ms/mol).

  metric B — templates with at least one valid (R1, R2) → product (SAMPLE)
      strictly ≤ metric A. For each R1-feasible template we enumerate every
      R2 in the test reactant bank and check whether `apply_reaction` returns
      any non-empty product. Slow (~tens of ms to seconds per molecule), so
      we run it on a random sample of `--sample-b` molecules (default 500).

The script reuses SearchRunner so all caps and data plumbing match the real
exhaustive-search job.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np  # noqa: E402

from genmolrl.algorithms.search import SearchRunner  # noqa: E402
from genmolrl.config import load_config  # noqa: E402


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _summary(name: str, arr: np.ndarray, n_total: int, bins: list[int]) -> None:
    _print(f"  swept {len(arr)} {name}")
    _print(f"  molecules with 0 feasible/productive  : {int((arr == 0).sum())}")
    _print(f"  molecules with ≥1 feasible/productive : {int((arr > 0).sum())}")
    _print(f"  mean                                  : {arr.mean():.3f}")
    _print(f"  median                                : {int(np.median(arr))}")
    _print(f"  min / max                             : {int(arr.min())} / {int(arr.max())}")
    _print(f"  std                                   : {arr.std():.3f}")
    _print(
        f"  percentiles 25/50/75/90/99            : "
        f"{int(np.percentile(arr, 25))} / "
        f"{int(np.percentile(arr, 50))} / "
        f"{int(np.percentile(arr, 75))} / "
        f"{int(np.percentile(arr, 90))} / "
        f"{int(np.percentile(arr, 99))}"
    )
    _print("  histogram:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = int(((arr >= lo) & (arr < hi)).sum())
        bar = "#" * int(60 * in_bin / max(1, n_total))
        _print(f"    [{lo:>4d}, {hi:>4d}): {in_bin:>6d}  {bar}")
    in_bin = int((arr >= bins[-1]).sum())
    _print(f"    [{bins[-1]:>4d},   ∞): {in_bin:>6d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-b", type=int, default=500,
                        help="Random sample size for the expensive d1 branching metric "
                             "(0 = skip metric B).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config("configs/exhausted_search_bi_delta_qed.yaml")
    cfg.setdefault("search", {})
    cfg["search"]["use_wandb"] = False
    cfg["search"]["overwrite_results"] = False
    cfg["search"]["results_file"] = "/tmp/audit_template_feasibility.txt"

    _print("Building SearchRunner ...")
    t0 = time.time()
    runner = SearchRunner(cfg, mode="exhausted_search", experiment_name="audit_feasibility")
    _print(f"  built in {time.time() - t0:.1f}s")
    _print(f"  bi templates available     : {len(runner.manager.templates)}")
    _print(f"  test reactant bank size    : {len(runner.reactant_keys)}")
    _print()

    keys = runner.reactant_keys
    n_total = len(keys)

    # ----------------------------------------------------------------
    # Metric A — R1-feasibility on the FULL test set (cheap)
    # ----------------------------------------------------------------
    _print(f"=== Metric A: R1-feasible templates per molecule ({n_total} starts) ===")
    t0 = time.time()
    n_feasible = np.zeros(n_total, dtype=np.int32)
    progress_every = max(1, n_total // 10)
    n_err = 0
    for i, s in enumerate(keys):
        try:
            n_feasible[i] = len(runner._valid_template_indices(s))
        except Exception:
            n_feasible[i] = -1
            n_err += 1
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            est_total = elapsed * n_total / (i + 1)
            _print(f"    progress: {i + 1}/{n_total}  elapsed={elapsed:.1f}s  est_total={est_total:.0f}s")
    dt = time.time() - t0
    _print(f"  swept {n_total} molecules in {dt:.1f}s ({dt / n_total * 1e3:.2f} ms/mol)")
    _print(f"  molecules that errored     : {n_err}")
    ok_a = n_feasible[n_feasible >= 0]
    _summary("starts", ok_a, n_total, bins=[0, 1, 2, 3, 5, 10, 20, 40, 80, 102])

    # ----------------------------------------------------------------
    # Metric B — d1 branching (after R2 enumeration) on a random sample
    # ----------------------------------------------------------------
    if args.sample_b <= 0:
        _print("\n(metric B skipped: --sample-b 0)")
        return

    _print()
    n_sample = min(args.sample_b, n_total)
    rng = random.Random(args.seed)
    sample_indices = rng.sample(range(n_total), n_sample)
    _print(f"=== Metric B: productive templates per molecule (sample of {n_sample} starts) ===")
    _print("    Each call applies RDKit reactions against every R2 in the test bank.")

    n_productive = np.zeros(n_sample, dtype=np.int32)
    n_branches = np.zeros(n_sample, dtype=np.int64)
    progress_every_b = max(1, n_sample // 20)
    t0 = time.time()
    for j, i in enumerate(sample_indices):
        s = keys[i]
        try:
            children = runner._all_next(s)
        except Exception:
            n_productive[j] = -1
            n_branches[j] = -1
            continue
        n_branches[j] = len(children)
        n_productive[j] = len({c[0] for c in children})
        if (j + 1) % progress_every_b == 0:
            elapsed = time.time() - t0
            est_total = elapsed * n_sample / (j + 1)
            _print(
                f"    progress: {j + 1}/{n_sample}  elapsed={elapsed:.1f}s  "
                f"est_total={est_total:.0f}s  "
                f"running_mean_branches={n_branches[: j + 1][n_branches[: j + 1] >= 0].mean():.2f}"
            )
    dt = time.time() - t0
    _print(f"  swept {n_sample} molecules in {dt:.1f}s ({dt / n_sample * 1e3:.1f} ms/mol)")
    ok_b = n_productive[n_productive >= 0]
    branches_ok = n_branches[n_branches >= 0]
    _print("  -- productive templates per molecule --")
    _summary("starts", ok_b, n_sample, bins=[0, 1, 2, 3, 5, 10, 20, 40, 80, 102])
    _print("  -- d1 branches per molecule (products after R2 enumeration) --")
    _summary("starts", branches_ok, n_sample, bins=[0, 1, 2, 5, 10, 50, 100, 500, 1000, 5000])
    _print(f"  sum d1 branches in sample             : {int(branches_ok.sum()):,}")
    _print(f"  extrapolated to full test set         : "
           f"~{int(branches_ok.mean() * n_total):,} d1 reactions across {n_total} starts")

    if branches_ok.mean() > 0:
        b = float(branches_ok.mean())
        per_start_d5 = b + b**2 + b**3 + b**4 + b**5
        total_d5 = per_start_d5 * n_total
        _print()
        _print("  back-of-envelope depth-5 reaction count (constant-branching assumption):")
        _print(f"    per start ≈ b + b² + b³ + b⁴ + b⁵ = "
               f"{b:.2f} + {b**2:.1f} + {b**3:.0f} + {b**4:.0f} + {b**5:.0f} = {per_start_d5:,.0f}")
        _print(f"    total    ≈ per_start × {n_total} = {total_d5:,.0f} reactions")
        _print("    (this is a rough upper bound — typical depth>1 branching may shrink as")
        _print("     molecules grow / become saturated)")


if __name__ == "__main__":
    main()

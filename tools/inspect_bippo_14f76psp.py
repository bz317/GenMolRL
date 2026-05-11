"""Inspect Bi-PPO run 14f76psp from wandb for masking/eval diagnostics.

We probe two questions:
  1. Is `masking: reaction_valid` actually being honoured? Symptoms of failure
     would be a high share of -1.0 step rewards (invalid_reaction_penalty),
     proving the policy is picking templates the mask said were valid but the
     env still rejects.
  2. Are eval metrics flat after some step?

We pull every reward/loss series we know the trainer emits, summarise their
distribution over time, and dump quantiles + windowed means so the user can
spot the plateau or the -1 saturation.
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict

import wandb

RUN = "boqiaoz-cambridge/GenMolRL_Bi/runs/14f76psp"


def main() -> None:
    api = wandb.Api()
    run = api.run(RUN)
    print(f"run.name        = {run.name}")
    print(f"run.state       = {run.state}")
    print(f"run.created_at  = {run.created_at}")
    print(f"run.summary keys (sorted, first 60):")
    skeys = sorted(run.summary.keys())
    for k in skeys[:60]:
        try:
            v = run.summary[k]
            if isinstance(v, (int, float)):
                print(f"  {k:40s} = {v}")
        except Exception:
            pass
    print(f"... {len(skeys)} total summary keys")

    cfg = dict(run.config)
    cfg_keys = ["masking", "reaction_mode", "reward", "max_episode_len", "training", "env", "ppo"]
    print("\nrun.config (selected):")
    for k in cfg_keys:
        if k in cfg:
            print(f"  {k} = {cfg[k]}")

    print("\nScanning full history (this can take ~30s)...")
    rows = list(run.scan_history())
    print(f"history rows = {len(rows)}")

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        step = row.get("global_step") or row.get("_step")
        if step is None:
            continue
        for key, value in row.items():
            if key in {"_step", "_timestamp", "_runtime", "global_step"}:
                continue
            if isinstance(value, (int, float)) and not math.isnan(value):
                series[key].append((int(step), float(value)))

    print(f"\nseries discovered ({len(series)}): first 40 names:")
    for name in sorted(series.keys())[:40]:
        print(f"  {name:50s} n={len(series[name])}")

    interesting_keys = [
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/loss",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/explained_variance",
        "train/learning_rate",
        "eval/mean_reward",
        "eval/mean_ep_length",
        "eval/qed",
        "eval/mean_qed",
        "eval/best_qed",
        "env/episode_reward",
        "env/episode_length",
    ]
    candidates = []
    for k in interesting_keys:
        if k in series:
            candidates.append(k)
    # Add anything that looks rewardy/qed/eval-ish if not already listed.
    for k in sorted(series.keys()):
        kl = k.lower()
        if ("reward" in kl or "qed" in kl or "eval" in kl) and k not in candidates:
            candidates.append(k)

    print("\n=== Summary of key series ===")
    for k in candidates:
        pts = series[k]
        if not pts:
            continue
        pts.sort()
        steps = [s for s, _ in pts]
        vals = [v for _, v in pts]
        print(
            f"\n[{k}]  n={len(vals)}  step={steps[0]}..{steps[-1]}  "
            f"first10_mean={sum(vals[:10]) / max(1, len(vals[:10])):.4f}  "
            f"last10_mean={sum(vals[-10:]) / max(1, len(vals[-10:])):.4f}  "
            f"min={min(vals):.4f}  max={max(vals):.4f}"
        )
        n = len(vals)
        if n >= 8:
            chunk = max(1, n // 8)
            buckets = [vals[i:i + chunk] for i in range(0, n, chunk)]
            means = [sum(b) / len(b) for b in buckets]
            bucket_steps = [steps[i] for i in range(0, n, chunk)]
            for s, m in zip(bucket_steps, means):
                print(f"    step ~{s:>8d}  mean={m:.4f}")

    # Drill into invalid-penalty signal: count -1.0 rewards in train rollouts.
    invalid_threshold = -0.999
    for k in ["env/last_reward", "env/step_reward", "train/last_reward", "rollout/last_reward"]:
        if k in series:
            vals = [v for _, v in series[k]]
            frac = sum(1 for v in vals if v <= invalid_threshold) / max(1, len(vals))
            print(f"\n[{k}]  fraction <= -0.999 (invalid penalty proxy) = {frac:.3f}  over n={len(vals)}")

    if "rollout/ep_rew_mean" in series:
        pts = sorted(series["rollout/ep_rew_mean"])
        if pts:
            n = len(pts)
            cuts = [n // 4, n // 2, 3 * n // 4, n - 1]
            print("\nrollout/ep_rew_mean quartiles:")
            for c in cuts:
                if 0 <= c < n:
                    s, v = pts[c]
                    print(f"  q@{c}/{n}  step={s:>8d}  ep_rew_mean={v:.4f}")


if __name__ == "__main__":
    main()

"""Smoke test for Bi-PPO: both architectures × both masking strategies.

Runs four 16-step rollouts and asserts that **every recorded reaction
transition has reward > invalid_reaction_penalty**. This is the operational
definition of the user's requirement that ``reaction_valid`` masking yields
zero -1 rewards (we additionally check ``substructure``, which relies on the
rejection-sampling backstop).

Configurations exercised:

  - policy_arch=hierarchical, masking=reaction_valid (true-valid mask, zero
    -1 by construction)
  - policy_arch=hierarchical, masking=substructure  (pattern mask + rejection)
  - policy_arch=multidiscrete, masking=reaction_valid (true-valid union mask +
    joint rejection)
  - policy_arch=multidiscrete, masking=substructure  (pattern union mask +
    joint rejection)

To keep the smoke fast we set ``r2_resample_retries`` low and use 16-step
rollouts. The smoke uses the production data (data/Bi/*) but only touches a
handful of unique states.

Run::

    WANDB_MODE=disabled PYTHONPATH=. python -u tools/smoke_bi_ppo.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

import math

import numpy as np

from genmolrl.algorithms.ppo_bi.train import BiPPO
from genmolrl.config import load_config


CASES = [
    ("hierarchical", "reaction_valid"),
    ("hierarchical", "substructure"),
    ("multidiscrete", "reaction_valid"),
    ("multidiscrete", "substructure"),
]


def run_case(policy_arch: str, masking: str, n_steps: int = 16) -> None:
    print(f"\n=========================================")
    print(f"CASE  policy_arch={policy_arch:<14s}  masking={masking}")
    print(f"=========================================")
    config = load_config("configs/ppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = masking
    config.setdefault("ppo_bi", {})
    config["ppo_bi"]["policy_arch"] = policy_arch
    # Cap retries so a hung loop fails fast instead of stalling smoke. The
    # multidiscrete + substructure case may need more retries because R2
    # is broad and (T, R2) joint rejection rate is higher.
    config["ppo_bi"]["r2_resample_retries"] = 16
    config["ppo_bi"]["n_steps"] = n_steps
    config["ppo_bi"]["batch_size"] = 8
    config["ppo_bi"]["n_epochs"] = 1
    config.setdefault("training", {})
    config["training"]["total_timesteps"] = n_steps

    t0 = time.monotonic()
    trainer = BiPPO(config)
    print(
        f"  trainer: arch={trainer.policy_arch}  masking={trainer.masking}  "
        f"r2_mask_kind={trainer.r2_mask_kind}  device={trainer.device}  "
        f"num_T={trainer.num_templates}  num_R2={trainer.num_reactants}"
    )

    rollout, last_value = trainer.collect_rollout(
        n_steps, base_step=0, log_episodes=False
    )
    elapsed = time.monotonic() - t0
    n = len(rollout)
    n_stop = sum(1 for tr in rollout if tr.is_stop)
    rewards = np.array([tr.reward for tr in rollout]) if rollout else np.array([0.0])
    rxn_rewards = np.array([tr.reward for tr in rollout if not tr.is_stop])
    print(
        f"  rollout: n={n}  reactions={n - n_stop}  stops={n_stop}  "
        f"rejections={trainer._rejection_total}  invalid_count={trainer._invalid_reaction_count}  "
        f"reward[min/mean/max]=[{rewards.min():.3f}/{rewards.mean():.3f}/{rewards.max():.3f}]"
        f"  elapsed={elapsed:.1f}s"
    )

    # Core assertion: every reaction transition is non-penalty.
    near_penalty = [
        float(r)
        for r in rxn_rewards
        if math.isclose(float(r), trainer.invalid_reaction_penalty, abs_tol=1e-6)
    ]
    assert not near_penalty, (
        f"  FAIL: {len(near_penalty)} reaction transitions equal the "
        f"invalid_reaction_penalty ({trainer.invalid_reaction_penalty})!"
    )
    assert trainer._invalid_reaction_count == 0, (
        f"  FAIL: trainer._invalid_reaction_count={trainer._invalid_reaction_count}"
    )

    # Sanity: PPO update step runs.
    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)
    bad = {k: v for k, v in metrics.items() if not math.isfinite(float(v)) and k != "train/explained_variance"}
    assert not bad, f"  FAIL: non-finite update metrics: {bad}"
    print(f"  update OK: loss={metrics['train/loss']:.4f}  pg={metrics['train/policy_loss']:.4f}  ent={metrics['train/entropy']:.4f}")
    print(f"  PASS")


def main() -> None:
    for arch, mask in CASES:
        # reaction_valid is expensive; use a smaller per-case n_steps for those.
        steps = 8 if mask == "reaction_valid" else 16
        run_case(arch, mask, n_steps=steps)
    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()

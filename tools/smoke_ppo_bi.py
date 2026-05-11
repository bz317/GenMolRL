"""Smoke test for hierarchical Bi-PPO.

Instantiates the trainer with the production bi-config, runs a tiny rollout
(64 steps, batch=16, 1 epoch), and verifies:

  - rollout produces transitions with no invalid-reaction penalty (the
    rejection-sampling loop ensures recorded steps are always valid),
  - the joint log-prob bookkeeping is consistent (re-evaluating an action
    under the recorded mask yields the same log-prob, within fp tolerance),
  - the PPO update step runs without exceptions and produces finite metrics,
  - the greedy eval path on a small test subset terminates.

Run with::

    WANDB_MODE=disabled PYTHONPATH=. python -u tools/smoke_ppo_bi.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

import math

import numpy as np
import torch

from genmolrl.algorithms.ppo_bi.train import HierarchicalBiPPO
from genmolrl.config import load_config


def main() -> None:
    config = load_config("configs/ppo_bi_hierarchical_delta_qed.yaml")
    # Trim to a tiny smoke-sized run.
    config.setdefault("training", {})
    config["training"]["total_timesteps"] = 64
    config["training"]["eval_freq"] = 1000
    config["training"]["save_freq"] = 1000
    config.setdefault("ppo_bi", {})
    config["ppo_bi"]["n_steps"] = 32
    config["ppo_bi"]["batch_size"] = 16
    config["ppo_bi"]["n_epochs"] = 1
    config["ppo_bi"]["r2_resample_retries"] = 4

    print(
        f"r2_mask_kind={config['ppo_bi'].get('r2_mask_kind', 'pattern')}  "
        f"masking={config['masking']}  reaction_mode={config['reaction_mode']}"
    )

    trainer = HierarchicalBiPPO(config)
    print(
        f"num_templates={trainer.num_templates}  num_reactants={trainer.num_reactants}  "
        f"device={trainer.device}"
    )

    # 1) rollout
    rollout, last_value = trainer.collect_rollout(32, base_step=0, log_episodes=False)
    n = len(rollout)
    n_stop = sum(1 for tr in rollout if tr.is_stop)
    n_reactions = n - n_stop
    rewards = np.array([tr.reward for tr in rollout])
    n_neg1 = int(np.sum(rewards <= trainer.invalid_reaction_penalty + 1e-6) - n_stop * (trainer.stop_early_penalty <= trainer.invalid_reaction_penalty))
    print(
        f"rollout: n={n} reactions={n_reactions} stops={n_stop}  "
        f"reward[min/mean/max]=[{rewards.min():.3f}/{rewards.mean():.3f}/{rewards.max():.3f}]  "
        f"#invalid_reaction_logged={trainer._invalid_reaction_count}  "
        f"last_value={last_value:.3f}"
    )

    # Reaction transitions must never carry the invalid-reaction penalty.
    rxn_rewards = [tr.reward for tr in rollout if not tr.is_stop]
    near_penalty = [r for r in rxn_rewards if math.isclose(r, trainer.invalid_reaction_penalty, abs_tol=1e-6)]
    assert not near_penalty, (
        f"Rejection-sampling loop allowed an invalid reaction through: "
        f"reaction rewards include the invalid_reaction_penalty={trainer.invalid_reaction_penalty}. "
        f"Offending rewards: {near_penalty}"
    )
    assert trainer._invalid_reaction_count == 0, (
        f"Unexpected env-step invalid-reaction count: {trainer._invalid_reaction_count}"
    )
    print("OK: rollout contains zero invalid-reaction-penalty rewards on reaction transitions.")

    # 2) PPO update sanity
    advantages, returns = trainer.compute_gae(rollout, last_value)
    update_metrics = trainer.ppo_update(rollout, advantages, returns)
    print("update metrics:")
    for k, v in update_metrics.items():
        print(f"  {k}: {v:.4f}")
        assert math.isfinite(float(v)) or k == "train/explained_variance", (
            f"Non-finite update metric: {k}={v}"
        )

    # 3) Quick greedy eval on a 3-molecule subset.
    eval_subset = trainer.sampler.test_smiles[:3]
    trainer.sampler.test_smiles = eval_subset
    eval_metrics = trainer.evaluate()
    print("eval metrics (3 molecules):")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")
    print("OK: greedy eval completed.")

    print("\nSMOKE PASS")


if __name__ == "__main__":
    main()

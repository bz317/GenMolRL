"""Fast Bi-PPO smoke: both architectures × both masking strategies on a
subsampled reactant pool.

The full data/Bi/reactants_train.pkl has 116k entries; computing the
``bi_r2_valid_mask`` for even one cold ``(state, T)`` pair costs ~5 minutes
on CPU because RDKit ``apply_reaction`` is called on every pattern-matching
R2 (~10k–80k per template). For a smoke we therefore patch the trainer's
``reaction_manager.reactants`` to a 512-entry subsample of the same pool —
this exercises every code path (true-validity mask, pattern mask, rejection
sampling, joint sampling, evaluate_minibatch, PPO update) but completes in
seconds.

Run::

    WANDB_MODE=disabled PYTHONPATH=. python -u tools/smoke_bi_ppo_fast.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np

from genmolrl.algorithms.ppo_bi.train import BiPPO
from genmolrl.chem.reaction_manager import ReactionManager
from genmolrl.config import load_config


SUBSAMPLE = 512  # reactant pool size for the smoke


def make_trainer(policy_arch: str, masking: str, n_steps: int) -> BiPPO:
    config = load_config("configs/ppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = masking
    config.setdefault("ppo_bi", {})
    config["ppo_bi"]["policy_arch"] = policy_arch
    config["ppo_bi"]["r2_resample_retries"] = 32
    config["ppo_bi"]["n_steps"] = n_steps
    config["ppo_bi"]["batch_size"] = 8
    config["ppo_bi"]["n_epochs"] = 1
    config.setdefault("training", {})
    config["training"]["total_timesteps"] = n_steps

    trainer = BiPPO(config)

    # Subsample the reactant pool to keep the smoke fast. Re-use the same
    # `match_template`-driven valid-R2 cache structure (clear the caches so
    # they recompute against the smaller pool).
    rng = random.Random(0)
    full_keys = list(trainer.reaction_manager.reactants.keys())
    chosen = set(rng.sample(full_keys, k=min(SUBSAMPLE, len(full_keys))))
    new_pool = {k: trainer.reaction_manager.reactants[k] for k in full_keys if k in chosen}
    trainer.reaction_manager.reactants = new_pool
    trainer.reaction_manager.template_mask_cache.clear()
    trainer.reaction_manager.valid_reactants_cache.clear()
    trainer.reaction_manager._bi_r2_valid_cache = {}
    trainer.reactant_keys = list(new_pool.keys())
    trainer.num_reactants = len(trainer.reactant_keys)

    # Rebuild the policy so its R2 embedding has the right size.
    from genmolrl.algorithms.ppo_bi.policy import BiPolicy
    import torch

    trainer.policy = BiPolicy(
        num_templates=trainer.num_templates,
        num_reactants=trainer.num_reactants,
        conditional_r2=(trainer.policy_arch == "hierarchical"),
        obs_dim=1024,
        trunk_hidden=256,
        template_embed_dim=64,
        r2_embed_dim=64,
    ).to(trainer.device)
    trainer.optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=3e-4)
    return trainer


def run_case(policy_arch: str, masking: str, n_steps: int = 16) -> None:
    print(f"\n===== CASE  policy_arch={policy_arch:<14s}  masking={masking}  =====")
    t0 = time.monotonic()
    trainer = make_trainer(policy_arch, masking, n_steps=n_steps)
    print(
        f"  trainer: arch={trainer.policy_arch}  masking={trainer.masking}  "
        f"r2_mask_kind={trainer.r2_mask_kind}  device={trainer.device}  "
        f"num_T={trainer.num_templates}  num_R2={trainer.num_reactants}  "
        f"(pool subsampled from full bi train pool)"
    )

    rollout, last_value = trainer.collect_rollout(
        n_steps, base_step=0, log_episodes=False
    )
    n = len(rollout)
    n_stop = sum(1 for tr in rollout if tr.is_stop)
    rewards = np.array([tr.reward for tr in rollout]) if rollout else np.array([0.0])
    rxn_rewards = np.array([tr.reward for tr in rollout if not tr.is_stop])
    elapsed = time.monotonic() - t0
    print(
        f"  rollout: n={n}  reactions={n - n_stop}  stops={n_stop}  "
        f"rejections={trainer._rejection_total}  invalid_count={trainer._invalid_reaction_count}  "
        f"reward[min/mean/max]=[{rewards.min():.3f}/{rewards.mean():.3f}/{rewards.max():.3f}]  "
        f"elapsed={elapsed:.1f}s"
    )

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

    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)
    bad = {
        k: v
        for k, v in metrics.items()
        if not math.isfinite(float(v)) and k != "train/explained_variance"
    }
    assert not bad, f"  FAIL: non-finite update metrics: {bad}"
    print(
        f"  update OK: loss={metrics['train/loss']:.4f}  "
        f"pg={metrics['train/policy_loss']:.4f}  ent={metrics['train/entropy']:.4f}"
    )
    print("  PASS")


def main() -> None:
    cases = [
        ("hierarchical", "reaction_valid", 16),
        ("hierarchical", "substructure", 16),
        ("multidiscrete", "reaction_valid", 16),
        ("multidiscrete", "substructure", 16),
    ]
    for arch, mask, steps in cases:
        run_case(arch, mask, n_steps=steps)
    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()

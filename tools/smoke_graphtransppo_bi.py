"""Fast GraphTransPPO-Bi smoke: both policy archs × reaction_valid + r2_available.

Mirrors :mod:`tools.smoke_bi_ppo_fast` for the graph-encoder variant. The
full bi reactant pool (~116k SMILES) makes mask construction expensive
under ``reaction_valid``, so we subsample the pool to 512 entries and
rebuild the policy's R2 embedding accordingly. This exercises every
code path that differs between BiPPO and GraphTransBiPPO:

  * graph batching (``batch_from_smiles``) in ``_encode_smiles``
  * the new ``GraphTransBiPolicy`` heads on top of a 128-d pooled trunk
  * hierarchical (T → R2|T) vs multidiscrete (independent T, R2) sampling
  * ``reaction_valid`` (zero -1 contract) vs ``r2_available`` (-1 allowed)
  * ``compute_gae`` + ``ppo_update`` (graph-aware ``_evaluate_minibatch``)

Run::

    WANDB_MODE=disabled PYTHONPATH=. python -u tools/smoke_graphtransppo_bi.py
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

from genmolrl.algorithms.graphtransppo_bi.train import GraphTransBiPPO
from genmolrl.config import load_config


SUBSAMPLE = 512


def make_trainer(policy_arch: str, masking: str, n_steps: int) -> GraphTransBiPPO:
    config = load_config("configs/graphtransppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = masking
    config.setdefault("graphtransppo_bi", {})
    config["graphtransppo_bi"]["policy_arch"] = policy_arch
    # Force the legacy lookup R2 head — see the matching note in
    # ``tools/smoke_bi_ppo_fast.py``. The encoder path is exercised by
    # ``tools/smoke_r2_arch.py`` which builds GraphTransBiPPO directly without
    # the post-init reactant-pool surgery this smoke test performs.
    config["graphtransppo_bi"]["r2_arch"] = "lookup"
    config["graphtransppo_bi"]["r2_resample_retries"] = 32
    config["graphtransppo_bi"]["n_steps"] = n_steps
    config["graphtransppo_bi"]["batch_size"] = 8
    config["graphtransppo_bi"]["n_epochs"] = 1
    # Small graph net to keep the smoke fast — encoder shape is the same.
    config["graphtransppo_bi"]["num_emb"] = 16
    config["graphtransppo_bi"]["num_layers"] = 1
    config["graphtransppo_bi"]["num_heads"] = 1
    config["graphtransppo_bi"]["device"] = "cpu"
    config.setdefault("training", {})
    config["training"]["total_timesteps"] = n_steps

    trainer = GraphTransBiPPO(config)

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

    # Rebuild the policy so its R2 embedding matches the subsampled pool.
    method_cfg = trainer._method_cfg(config)
    import torch
    trainer.policy = trainer._build_policy(method_cfg)
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
        f"trunk_dim={trainer.policy.trunk_dim}"
    )

    rollout, last_value = trainer.collect_rollout(
        n_steps, base_step=0, log_episodes=False
    )
    n = len(rollout)
    n_stop = sum(1 for tr in rollout if tr.is_stop)
    rewards = np.array([tr.reward for tr in rollout]) if rollout else np.array([0.0])
    elapsed = time.monotonic() - t0
    print(
        f"  rollout: n={n}  reactions={n - n_stop}  stops={n_stop}  "
        f"rejections={trainer._rejection_total}  invalid_count={trainer._invalid_reaction_count}  "
        f"reward[min/mean/max]=[{rewards.min():.3f}/{rewards.mean():.3f}/{rewards.max():.3f}]  "
        f"elapsed={elapsed:.1f}s"
    )

    if masking == "reaction_valid":
        # README contract: reaction_valid must never emit -1 transitions
        # (hierarchical: by mask construction; multidiscrete: by joint
        # rejection sampling). Any leak here is a regression.
        rxn_rewards = [tr.reward for tr in rollout if not tr.is_stop]
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
    # For r2_available we explicitly allow -1 leaks (pattern-only mask),
    # so we don't assert on the invalid count — just sanity-check the
    # rollout shape and the update finite-ness below.

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
        ("hierarchical", "r2_available", 16),
        ("multidiscrete", "reaction_valid", 16),
        ("multidiscrete", "r2_available", 16),
    ]
    for arch, mask, steps in cases:
        run_case(arch, mask, n_steps=steps)
    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()

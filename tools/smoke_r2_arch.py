"""Fast smoke test for the new ``r2_arch`` knob on BiPPO / GraphTransBiPPO.

Verifies four things that the design hinges on:

  1. ``r2_arch='lookup'`` is bit-identical to the legacy BiPolicy — the
     policy has an ``r2_embed`` table and no ``r2_encoder``; ``evaluate()``
     keeps the active pool at the train pool (no swap).
  2. ``r2_arch='encoder'`` swaps the pool to the *test* pool inside
     ``evaluate()`` and restores the *train* pool afterwards. The cached
     ``r2_keys`` matrix has the right shape on each side.
  3. ``collect_rollout``, ``ppo_update``, and ``evaluate`` all run to
     completion in encoder mode without dimension mismatches, with the
     train pool used for rollout / update and the test pool used for eval.
  4. Both BiPPO (Morgan-FP trunk) and GraphTransBiPPO (GraphTransformer
     trunk) work in encoder mode — i.e. the trainer refactor cleanly
     supports both R1 encoder types.

Each test subsamples the train and test reactant pools to keep the smoke
fast (the full ``reaction_valid`` mask construction is otherwise expensive
over the 116k train pool).

Run::

    WANDB_MODE=disabled PYTHONPATH=. python -u tools/smoke_r2_arch.py
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
import torch

from genmolrl.algorithms.ppo_bi.train import BiPPO
from genmolrl.config import load_config


TRAIN_SUBSAMPLE = 256  # cheap reaction_valid mask construction
TEST_SUBSAMPLE = 128   # smaller than train so we can distinguish pool sizes


def _subsample_pool(pool: dict, n: int, rng: random.Random) -> dict:
    keys = list(pool.keys())
    chosen = set(rng.sample(keys, k=min(n, len(keys))))
    return {k: pool[k] for k in keys if k in chosen}


def make_bi_trainer(*, r2_arch: str, policy_arch: str, masking: str) -> BiPPO:
    """Build a tiny BiPPO trainer for smoke-test purposes.

    The reactant_train and reactant_test pickle files are loaded *first*,
    subsampled in-memory, written to a temp dict, and only then handed to
    the trainer — so that ``BiPPO.__init__`` builds the pre-computed
    ``_train_r2_fps`` / ``_eval_r2_fps`` tensors against the small pools.
    """
    config = load_config("configs/ppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = masking
    config.setdefault("ppo_bi", {})
    config["ppo_bi"]["policy_arch"] = policy_arch
    config["ppo_bi"]["r2_arch"] = r2_arch
    config["ppo_bi"]["r2_resample_retries"] = 16
    config["ppo_bi"]["n_steps"] = 8
    config["ppo_bi"]["batch_size"] = 4
    config["ppo_bi"]["n_epochs"] = 1
    config["ppo_bi"]["trunk_hidden"] = 64
    config["ppo_bi"]["template_embed_dim"] = 16
    config["ppo_bi"]["r2_embed_dim"] = 16
    config["ppo_bi"]["r2_encoder_hidden"] = 64
    config["ppo_bi"]["device"] = "cpu"
    config.setdefault("training", {})
    config["training"]["total_timesteps"] = 8

    import pickle
    from genmolrl.config import resolve_path

    with open(resolve_path(config["dataset"]["training_file"]), "rb") as f:
        train_full = pickle.load(f)
    with open(resolve_path(config["dataset"]["test_file"]), "rb") as f:
        test_full = pickle.load(f)
    rng = random.Random(0)
    train_small = _subsample_pool(train_full, TRAIN_SUBSAMPLE, rng)
    test_small = _subsample_pool(test_full, TEST_SUBSAMPLE, rng)

    # Write to scratch files (the trainer reads via resolve_path → file).
    scratch = Path("/tmp/genmolrl_smoke_r2_arch")
    scratch.mkdir(parents=True, exist_ok=True)
    train_path = scratch / "reactants_train.pkl"
    test_path = scratch / "reactants_test.pkl"
    with open(train_path, "wb") as f:
        pickle.dump(train_small, f)
    with open(test_path, "wb") as f:
        pickle.dump(test_small, f)
    config["dataset"]["training_file"] = str(train_path)
    config["dataset"]["test_file"] = str(test_path)

    return BiPPO(config)


def case_lookup_backcompat() -> None:
    print("\n===== CASE  r2_arch=lookup  policy_arch=hierarchical  masking=r2_available  =====")
    t0 = time.monotonic()
    trainer = make_bi_trainer(
        r2_arch="lookup", policy_arch="hierarchical", masking="r2_available"
    )
    # Pool wiring sanity: lookup aliases train ↔ eval to the same objects.
    assert trainer._eval_reaction_manager is trainer._train_reaction_manager, (
        "lookup mode must alias eval pool to train pool (same manager object)"
    )
    assert trainer._train_r2_fps is None and trainer._eval_r2_fps is None, (
        "lookup mode must not allocate r2_fps tensors"
    )
    # Policy sanity: r2_embed table, no encoder.
    assert trainer.policy.r2_embed is not None and trainer.policy.r2_encoder is None
    print(
        f"  trainer: r2_arch={trainer.r2_arch} eval_role={trainer._eval_pool_role} "
        f"|train|={trainer._train_num_reactants} |test alias|={trainer._eval_num_reactants}"
    )

    rollout, last_value = trainer.collect_rollout(8, base_step=0, log_episodes=False)
    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)
    eval_m = trainer.evaluate()
    # After evaluate(), the active pool must be restored to train.
    assert trainer.reaction_manager is trainer._train_reaction_manager
    assert trainer.num_reactants == trainer._train_num_reactants

    elapsed = time.monotonic() - t0
    bad = {
        k: v for k, v in metrics.items()
        if not math.isfinite(float(v)) and k != "train/explained_variance"
    }
    assert not bad, f"  FAIL: non-finite update metrics: {bad}"
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} eval_mean_reward={eval_m['eval/mean_reward']:.4f} "
        f"elapsed={elapsed:.1f}s"
    )
    print("  PASS")


def case_encoder_pool_swap() -> None:
    print("\n===== CASE  r2_arch=encoder  policy_arch=hierarchical  masking=r2_available  =====")
    t0 = time.monotonic()
    trainer = make_bi_trainer(
        r2_arch="encoder", policy_arch="hierarchical", masking="r2_available"
    )
    # Pool wiring sanity: encoder must have a DISTINCT eval manager and r2_fps.
    assert trainer._eval_reaction_manager is not trainer._train_reaction_manager
    assert trainer._train_r2_fps is not None and trainer._eval_r2_fps is not None
    assert trainer._train_r2_fps.shape[0] == trainer._train_num_reactants
    assert trainer._eval_r2_fps.shape[0] == trainer._eval_num_reactants
    # Policy sanity: r2_encoder MLP, no r2_embed table.
    assert trainer.policy.r2_embed is None and trainer.policy.r2_encoder is not None
    print(
        f"  trainer: r2_arch={trainer.r2_arch} eval_role={trainer._eval_pool_role} "
        f"|train|={trainer._train_num_reactants} |test|={trainer._eval_num_reactants}"
    )

    # Active pool at __init__ must be train.
    assert trainer.reaction_manager is trainer._train_reaction_manager
    assert trainer.num_reactants == trainer._train_num_reactants

    rollout, last_value = trainer.collect_rollout(8, base_step=0, log_episodes=False)
    # After rollout, active pool must still be train.
    assert trainer.reaction_manager is trainer._train_reaction_manager
    # Every R2 action collected during rollout indexes into the TRAIN pool, so
    # the r2_action values must all live in [0, train_num_reactants) (STOP rows
    # get R2_PAD=-1, which we ignore here).
    r2_actions = [tr.r2_action for tr in rollout if not tr.is_stop]
    assert all(0 <= a < trainer._train_num_reactants for a in r2_actions), (
        f"  FAIL: rollout R2 indices outside train pool: "
        f"{[a for a in r2_actions if not (0 <= a < trainer._train_num_reactants)]}"
    )

    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)
    # After update, still on train pool.
    assert trainer.reaction_manager is trainer._train_reaction_manager

    # Eval: monkey-patch _greedy_trajectory to record the active pool size
    # observed inside the loop, so we can prove the swap actually happened.
    observed_pool_sizes: list[int] = []
    real_greedy = trainer._greedy_trajectory

    def _patched(start_smiles: str):
        observed_pool_sizes.append(trainer.num_reactants)
        return real_greedy(start_smiles)

    trainer._greedy_trajectory = _patched  # type: ignore[assignment]
    eval_m = trainer.evaluate()
    trainer._greedy_trajectory = real_greedy  # type: ignore[assignment]

    # The observed pool sizes inside eval must all equal the TEST pool size.
    assert observed_pool_sizes, "evaluate() did not produce any trajectories"
    assert all(n == trainer._eval_num_reactants for n in observed_pool_sizes), (
        f"  FAIL: pool not swapped during evaluate(). "
        f"Observed sizes inside eval: {set(observed_pool_sizes)}, "
        f"expected all == |test|={trainer._eval_num_reactants}."
    )

    # After evaluate(), the active pool must be restored to train.
    assert trainer.reaction_manager is trainer._train_reaction_manager
    assert trainer.num_reactants == trainer._train_num_reactants

    elapsed = time.monotonic() - t0
    bad = {
        k: v for k, v in metrics.items()
        if not math.isfinite(float(v)) and k != "train/explained_variance"
    }
    assert not bad, f"  FAIL: non-finite update metrics: {bad}"
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} eval_mean_reward={eval_m['eval/mean_reward']:.4f} "
        f"eval_pool_observed={set(observed_pool_sizes)} elapsed={elapsed:.1f}s"
    )
    print("  PASS")


def case_encoder_multidiscrete() -> None:
    print("\n===== CASE  r2_arch=encoder  policy_arch=multidiscrete  masking=r2_available  =====")
    t0 = time.monotonic()
    trainer = make_bi_trainer(
        r2_arch="encoder", policy_arch="multidiscrete", masking="r2_available"
    )
    rollout, last_value = trainer.collect_rollout(8, base_step=0, log_episodes=False)
    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)
    eval_m = trainer.evaluate()
    elapsed = time.monotonic() - t0
    bad = {
        k: v for k, v in metrics.items()
        if not math.isfinite(float(v)) and k != "train/explained_variance"
    }
    assert not bad, f"  FAIL: non-finite update metrics: {bad}"
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} eval_mean_reward={eval_m['eval/mean_reward']:.4f} "
        f"elapsed={elapsed:.1f}s"
    )
    print("  PASS")


def case_graphtransbi_encoder() -> None:
    print("\n===== CASE  GraphTransBiPPO  r2_arch=encoder  policy_arch=hierarchical  =====")
    from genmolrl.algorithms.graphtransppo_bi.train import GraphTransBiPPO

    config = load_config("configs/graphtransppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = "r2_available"
    config.setdefault("graphtransppo_bi", {})
    config["graphtransppo_bi"]["policy_arch"] = "hierarchical"
    config["graphtransppo_bi"]["r2_arch"] = "encoder"
    config["graphtransppo_bi"]["r2_resample_retries"] = 8
    config["graphtransppo_bi"]["n_steps"] = 8
    config["graphtransppo_bi"]["batch_size"] = 4
    config["graphtransppo_bi"]["n_epochs"] = 1
    config["graphtransppo_bi"]["num_emb"] = 16
    config["graphtransppo_bi"]["num_layers"] = 1
    config["graphtransppo_bi"]["num_heads"] = 1
    config["graphtransppo_bi"]["template_embed_dim"] = 16
    config["graphtransppo_bi"]["r2_embed_dim"] = 16
    config["graphtransppo_bi"]["r2_encoder_hidden"] = 64
    config["graphtransppo_bi"]["device"] = "cpu"
    config.setdefault("training", {})
    config["training"]["total_timesteps"] = 8

    import pickle
    from genmolrl.config import resolve_path

    with open(resolve_path(config["dataset"]["training_file"]), "rb") as f:
        train_full = pickle.load(f)
    with open(resolve_path(config["dataset"]["test_file"]), "rb") as f:
        test_full = pickle.load(f)
    rng = random.Random(0)
    train_small = _subsample_pool(train_full, TRAIN_SUBSAMPLE, rng)
    test_small = _subsample_pool(test_full, TEST_SUBSAMPLE, rng)
    scratch = Path("/tmp/genmolrl_smoke_r2_arch_gtb")
    scratch.mkdir(parents=True, exist_ok=True)
    train_path = scratch / "reactants_train.pkl"
    test_path = scratch / "reactants_test.pkl"
    with open(train_path, "wb") as f:
        pickle.dump(train_small, f)
    with open(test_path, "wb") as f:
        pickle.dump(test_small, f)
    config["dataset"]["training_file"] = str(train_path)
    config["dataset"]["test_file"] = str(test_path)

    t0 = time.monotonic()
    trainer = GraphTransBiPPO(config)
    assert trainer.r2_arch == "encoder"
    assert trainer.policy.r2_encoder is not None and trainer.policy.r2_embed is None
    rollout, last_value = trainer.collect_rollout(8, base_step=0, log_episodes=False)
    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)
    eval_m = trainer.evaluate()
    elapsed = time.monotonic() - t0
    bad = {
        k: v for k, v in metrics.items()
        if not math.isfinite(float(v)) and k != "train/explained_variance"
    }
    assert not bad, f"  FAIL: non-finite update metrics: {bad}"
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} eval_mean_reward={eval_m['eval/mean_reward']:.4f} "
        f"|train|={trainer._train_num_reactants} |test|={trainer._eval_num_reactants} "
        f"elapsed={elapsed:.1f}s"
    )
    print("  PASS")


def main() -> None:
    case_lookup_backcompat()
    case_encoder_pool_swap()
    case_encoder_multidiscrete()
    case_graphtransbi_encoder()
    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()

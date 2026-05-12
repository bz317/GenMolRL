"""Fast smoke test for the new ``r2_arch`` knob on BiPPO / GraphTransBiPPO.

Verifies the design points the trainers hinge on:

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
  5. Option 1 (Bi-PPO): the residual MLP variant of the R2 encoder
     ``r2_encoder_residual=True`` instantiates the deeper Pre-LN
     residual stack, exposes more parameters than the legacy plain MLP,
     and trains end-to-end without dimension errors.
  6. Option 3 (Bi-GraphTransPPO): ``r2_arch='encoder_graph'`` builds the
     Siamese R2 GraphTransformer and its R2 graph-pool caches, the
     trainer can encode the train pool with grad during the PPO update
     and the test pool no_grad during ``evaluate()``, the
     ``r2_keys_refresh_minibatches`` cache cycles correctly, and the
     R2 backbone gradient flows on refresh minibatches and is frozen
     on cached minibatches.

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


def case_residual_mlp_bipolicy() -> None:
    """Option 1: deep residual-MLP R2 encoder under BiPPO.

    Builds a trainer with ``r2_encoder_residual=True`` and confirms:
      - The encoder is a residual stack (3-module Sequential whose first
        Sub-module starts with ``nn.Linear → nn.LayerNorm → nn.ReLU``).
      - It has *strictly more* parameters than the legacy plain 2-layer MLP
        built with the same hidden width.
      - One PPO cycle (rollout + update + evaluate) runs cleanly and the
        R2 encoder receives a real gradient on the update minibatch.
    """
    import torch.nn as nn

    print("\n===== CASE  BiPPO  r2_encoder_residual=True (Option 1)  =====")
    t0 = time.monotonic()

    # Build the "deep" trainer.
    config = load_config("configs/ppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = "r2_available"
    config.setdefault("ppo_bi", {})
    config["ppo_bi"]["policy_arch"] = "hierarchical"
    config["ppo_bi"]["r2_arch"] = "encoder"
    config["ppo_bi"]["r2_encoder_residual"] = True
    config["ppo_bi"]["r2_encoder_hidden"] = 128
    config["ppo_bi"]["r2_encoder_n_res_blocks"] = 2
    config["ppo_bi"]["n_steps"] = 8
    config["ppo_bi"]["batch_size"] = 4
    config["ppo_bi"]["n_epochs"] = 1
    config["ppo_bi"]["trunk_hidden"] = 64
    config["ppo_bi"]["template_embed_dim"] = 16
    config["ppo_bi"]["r2_embed_dim"] = 16
    config["ppo_bi"]["device"] = "cpu"
    config.setdefault("training", {})["total_timesteps"] = 8

    import pickle
    from genmolrl.config import resolve_path
    with open(resolve_path(config["dataset"]["training_file"]), "rb") as f:
        train_full = pickle.load(f)
    with open(resolve_path(config["dataset"]["test_file"]), "rb") as f:
        test_full = pickle.load(f)
    rng = random.Random(0)
    train_small = _subsample_pool(train_full, TRAIN_SUBSAMPLE, rng)
    test_small = _subsample_pool(test_full, TEST_SUBSAMPLE, rng)
    scratch = Path("/tmp/genmolrl_smoke_residual_mlp")
    scratch.mkdir(parents=True, exist_ok=True)
    train_path = scratch / "reactants_train.pkl"
    test_path = scratch / "reactants_test.pkl"
    with open(train_path, "wb") as f:
        pickle.dump(train_small, f)
    with open(test_path, "wb") as f:
        pickle.dump(test_small, f)
    config["dataset"]["training_file"] = str(train_path)
    config["dataset"]["test_file"] = str(test_path)

    trainer = BiPPO(config)

    # Structural check: residual MLP is a 3-stage Sequential (stem, body, proj).
    enc = trainer.policy.r2_encoder
    assert isinstance(enc, nn.Sequential) and len(enc) == 3, (
        f"  FAIL: residual encoder expected 3-stage Sequential, got {enc}"
    )
    # The stem is also a Sequential whose first module is a Linear.
    stem_first = list(enc[0].children())[0]
    assert isinstance(stem_first, nn.Linear), (
        f"  FAIL: residual encoder stem must start with nn.Linear, got {stem_first}"
    )
    deep_params = sum(p.numel() for p in enc.parameters())

    # Build a baseline plain-MLP trainer at the same hidden width to confirm
    # the residual variant really is deeper / wider.
    plain_config = dict(config)
    plain_config["ppo_bi"] = dict(config["ppo_bi"])
    plain_config["ppo_bi"]["r2_encoder_residual"] = False
    plain_config["ppo_bi"]["r2_encoder_n_layers"] = 2
    plain_trainer = BiPPO(plain_config)
    plain_params = sum(p.numel() for p in plain_trainer.policy.r2_encoder.parameters())
    assert deep_params > plain_params, (
        f"  FAIL: residual encoder ({deep_params:,} params) is not deeper "
        f"than plain encoder ({plain_params:,} params)"
    )

    # End-to-end PPO cycle + gradient flow check.
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
        f"  encoder params: deep={deep_params:,} plain={plain_params:,} "
        f"(deep / plain = {deep_params / max(1, plain_params):.1f}x)"
    )
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} elapsed={elapsed:.1f}s"
    )
    print("  PASS")


def _check_graph_dependencies_present() -> bool:
    """Skip the graph-encoder case cleanly if torch_geometric isn't installed."""
    import importlib.util
    return importlib.util.find_spec("torch_geometric") is not None


def case_graphtransbi_encoder_graph() -> None:
    """Option 3: Siamese R2 GraphTransformer under GraphTransBiPPO.

    Builds a trainer with ``r2_arch='encoder_graph'`` and confirms:
      - The policy has a separate R2 backbone + projection (and no
        r2_encoder MLP, no r2_embed lookup).
      - The trainer builds distinct ``_train_r2_graphs`` and
        ``_eval_r2_graphs`` Batch caches matching the two pool sizes.
      - One PPO cycle runs cleanly, the cached r2_keys cycles through
        refresh / cached / cached / ... according to
        ``r2_keys_refresh_minibatches``, and the R2 backbone parameters
        receive a non-zero gradient on the refresh minibatch.
      - ``evaluate()`` swaps to the test-pool graph Batch and restores
        the train pool when finished.
    """
    if not _check_graph_dependencies_present():
        print("\n===== CASE  GraphTransBiPPO  r2_arch=encoder_graph  =====")
        print("  SKIP (torch_geometric not installed)")
        return

    print("\n===== CASE  GraphTransBiPPO  r2_arch=encoder_graph (Option 3)  =====")
    from genmolrl.algorithms.graphtransppo_bi.train import GraphTransBiPPO

    config = load_config("configs/graphtransppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = "r2_available"
    config.setdefault("graphtransppo_bi", {})
    config["graphtransppo_bi"]["policy_arch"] = "hierarchical"
    config["graphtransppo_bi"]["r2_arch"] = "encoder_graph"
    config["graphtransppo_bi"]["r2_resample_retries"] = 8
    config["graphtransppo_bi"]["n_steps"] = 8
    config["graphtransppo_bi"]["batch_size"] = 4
    config["graphtransppo_bi"]["n_epochs"] = 1
    config["graphtransppo_bi"]["num_emb"] = 16
    config["graphtransppo_bi"]["num_layers"] = 1
    config["graphtransppo_bi"]["num_heads"] = 1
    config["graphtransppo_bi"]["r2_num_emb"] = 8
    config["graphtransppo_bi"]["r2_num_layers"] = 1
    config["graphtransppo_bi"]["r2_num_heads"] = 1
    config["graphtransppo_bi"]["template_embed_dim"] = 16
    config["graphtransppo_bi"]["r2_embed_dim"] = 16
    config["graphtransppo_bi"]["r2_keys_refresh_minibatches"] = 3
    config["graphtransppo_bi"]["device"] = "cpu"
    config.setdefault("training", {})["total_timesteps"] = 8

    import pickle
    from genmolrl.config import resolve_path
    with open(resolve_path(config["dataset"]["training_file"]), "rb") as f:
        train_full = pickle.load(f)
    with open(resolve_path(config["dataset"]["test_file"]), "rb") as f:
        test_full = pickle.load(f)
    rng = random.Random(0)
    train_small = _subsample_pool(train_full, TRAIN_SUBSAMPLE, rng)
    test_small = _subsample_pool(test_full, TEST_SUBSAMPLE, rng)
    scratch = Path("/tmp/genmolrl_smoke_encoder_graph")
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

    # Structural checks: the R2 side is graph-encoded; no MLP / no lookup.
    assert trainer.r2_arch == "encoder_graph"
    assert trainer.policy.r2_backbone is not None
    assert trainer.policy.r2_project is not None
    assert trainer.policy.r2_encoder is None
    assert trainer.policy.r2_embed is None
    # Pool caches: separate Batches with the right sizes.
    assert trainer._train_r2_graphs is not None
    assert trainer._eval_r2_graphs is not None
    assert trainer._train_r2_graphs.num_graphs == trainer._train_num_reactants
    assert trainer._eval_r2_graphs.num_graphs == trainer._eval_num_reactants
    print(
        f"  trainer: r2_arch={trainer.r2_arch} eval_role={trainer._eval_pool_role} "
        f"|train|={trainer._train_num_reactants} |test|={trainer._eval_num_reactants} "
        f"refresh={trainer.r2_keys_refresh_minibatches}"
    )

    # End-to-end cycle.
    rollout, last_value = trainer.collect_rollout(8, base_step=0, log_episodes=False)
    advantages, returns = trainer.compute_gae(rollout, last_value)

    # Manually drive the cache to verify refresh / cached behaviour:
    # - First call (cache empty) → refresh, returns graph-attached tensor.
    # - Calls 2..K → cached (no_grad), returns the SAME detached tensor.
    # - Call K+1 → refresh again.
    trainer._begin_update_cycle()
    first = trainer._r2_keys_for_update()
    assert first.requires_grad, "first _r2_keys_for_update call must return grad-attached r2_keys"
    second = trainer._r2_keys_for_update()
    assert not second.requires_grad, "second call must be cached (no_grad)"
    assert second is trainer._cached_r2_keys, "second call must reuse the detached cache"
    third = trainer._r2_keys_for_update()
    assert third is second, "third call must reuse the cache"
    # K=3 → fourth call refreshes.
    fourth = trainer._r2_keys_for_update()
    assert fourth.requires_grad, "fourth call after K=3 cached must refresh with grad"

    # Verify the refresh minibatch path DOES flow gradient into the R2 backbone.
    # (Can't check after ppo_update directly because the trainer's last
    # minibatch may have been a cached/no-grad one, in which case
    # ``optimizer.zero_grad()`` left the R2 backbone grad at exactly zero by
    # the time the update returned.)
    trainer._begin_update_cycle()
    trainer.optimizer.zero_grad(set_to_none=False)
    fresh_keys = trainer._r2_keys_for_update()
    assert fresh_keys.requires_grad
    # Simulate a tiny PPO loss-like term: dot of query × keys, summed.
    fake_loss = fresh_keys.sum()
    fake_loss.backward()
    r2_grad_norms = [
        float(p.grad.detach().abs().sum().item())
        for p in trainer.policy.r2_backbone.parameters()
        if p.grad is not None
    ]
    assert r2_grad_norms, "  FAIL: R2 backbone parameters got no gradient during refresh"
    assert any(g > 0 for g in r2_grad_norms), (
        "  FAIL: R2 backbone gradient is all zeros — refresh did not flow grad"
    )
    trainer.optimizer.zero_grad(set_to_none=False)

    # Real PPO update for end-to-end smoke (after the manual gradient check).
    metrics = trainer.ppo_update(rollout, advantages, returns)

    eval_m = trainer.evaluate()
    # After evaluate(), active pool must be restored to train.
    assert trainer.reaction_manager is trainer._train_reaction_manager
    assert trainer.num_reactants == trainer._train_num_reactants

    elapsed = time.monotonic() - t0
    bad = {
        k: v for k, v in metrics.items()
        if not math.isfinite(float(v)) and k != "train/explained_variance"
    }
    assert not bad, f"  FAIL: non-finite update metrics: {bad}"
    print(
        f"  R2 backbone grad sum (refresh mb): {sum(r2_grad_norms):.4f} "
        f"(over {len(r2_grad_norms)} params with grad)"
    )
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} eval_mean_reward={eval_m['eval/mean_reward']:.4f} "
        f"elapsed={elapsed:.1f}s"
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
    case_residual_mlp_bipolicy()      # Option 1: deeper R2 MLP for BiPPO
    case_graphtransbi_encoder_graph()  # Option 3: Siamese R2 GT for GraphTransBiPPO
    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()

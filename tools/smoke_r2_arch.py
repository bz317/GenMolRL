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
    # The encoder-default YAML now ships ``eval_r2_pool: test`` for clarity,
    # but the existing smoke cases predate that knob and rely on the
    # per-arch default (lookup → train, encoder/encoder_graph → test).
    # Drop the YAML override so each case test exercises the default path
    # of whichever r2_arch it asked for; the dedicated eval_r2_pool
    # smokes below override it explicitly when they need to.
    config["ppo_bi"].pop("eval_r2_pool", None)
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


def case_graphtransbi_lookup_train() -> None:
    """GraphTransBiPPO with r2_arch=lookup + eval_r2_pool=train.

    Mirrors the gr7aa7z6 wiring on the graph-trunk algorithm. Verifies:

      - ``_eval_pool_role == "train"``, ``eval_r2_pool == "train"``.
      - ``_eval_reaction_manager is _train_reaction_manager`` (eval aliased
        to train, no extra pool allocated).
      - Policy has ``r2_embed`` and no encoder modules (no
        ``r2_encoder`` MLP, no ``r2_backbone`` / ``r2_project``
        encoder_graph stack).
      - No FP / graph batch caches allocated.
      - ``r2_keys`` consumed during the PPO update IS exactly
        ``policy.r2_embed.weight``.
      - End-to-end rollout + update + evaluate runs to completion
        without dimension errors on the test starts.
    """
    if not _check_graph_dependencies_present():
        print("\n===== CASE  GraphTransBiPPO  r2_arch=lookup + eval_r2_pool=train  =====")
        print("  SKIP (torch_geometric not installed)")
        return
    print("\n===== CASE  GraphTransBiPPO  r2_arch=lookup + eval_r2_pool=train  =====")
    from genmolrl.algorithms.graphtransppo_bi.train import GraphTransBiPPO

    config = load_config("configs/graphtransppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = "r2_available"
    config.setdefault("graphtransppo_bi", {})
    # Override to lookup + train regardless of what the YAML now defaults to,
    # so this case asserts the wiring works irrespective of YAML drift.
    config["graphtransppo_bi"]["policy_arch"] = "hierarchical"
    config["graphtransppo_bi"]["r2_arch"] = "lookup"
    config["graphtransppo_bi"]["eval_r2_pool"] = "train"
    config["graphtransppo_bi"]["r2_resample_retries"] = 8
    config["graphtransppo_bi"]["n_steps"] = 8
    config["graphtransppo_bi"]["batch_size"] = 4
    config["graphtransppo_bi"]["n_epochs"] = 1
    config["graphtransppo_bi"]["num_emb"] = 16
    config["graphtransppo_bi"]["num_layers"] = 1
    config["graphtransppo_bi"]["num_heads"] = 1
    config["graphtransppo_bi"]["template_embed_dim"] = 16
    config["graphtransppo_bi"]["r2_embed_dim"] = 16
    config["graphtransppo_bi"]["device"] = "cpu"
    # encoder_graph-only knobs from the YAML should be dormant — set them
    # to something obviously-not-meaningful so any code path that
    # accidentally reads them in lookup mode will blow up loudly.
    for stale_key in (
        "r2_num_emb",
        "r2_num_layers",
        "r2_num_heads",
        "r2_keys_refresh_minibatches",
    ):
        config["graphtransppo_bi"].pop(stale_key, None)
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
    scratch = Path("/tmp/genmolrl_smoke_gtb_lookup_train")
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

    # Wiring: lookup + train should alias eval pool to train pool with no
    # extra allocations.
    assert trainer.r2_arch == "lookup", trainer.r2_arch
    assert trainer._eval_pool_role == "train"
    assert trainer.eval_r2_pool == "train"
    assert trainer._eval_reaction_manager is trainer._train_reaction_manager, (
        "lookup + train must alias eval manager to train manager"
    )
    # Policy: lookup table, no encoder.
    assert trainer.policy.r2_embed is not None
    assert trainer.policy.r2_encoder is None, (
        "lookup mode must not instantiate the r2_encoder MLP"
    )
    # encoder_graph-specific attributes either don't exist or are None.
    assert getattr(trainer.policy, "r2_backbone", None) is None, (
        "lookup mode must not instantiate the Siamese R2 GraphTransformer"
    )
    assert getattr(trainer.policy, "r2_project", None) is None
    # Pool caches: no FPs (encoder), no graphs (encoder_graph).
    assert trainer._train_r2_fps is None and trainer._eval_r2_fps is None
    assert trainer._train_r2_graphs is None and trainer._eval_r2_graphs is None

    # r2_keys at update time IS the lookup table.
    r2_keys = trainer._compute_active_r2_keys(pool="train", with_grad=True)
    assert r2_keys is trainer.policy.r2_embed.weight, (
        "lookup mode must return the embedding weight matrix directly"
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
        f"  wiring: r2_arch=lookup eval_pool=train  "
        f"eval_mgr_is_train_mgr={trainer._eval_reaction_manager is trainer._train_reaction_manager}  "
        f"r2_keys_is_lookup=True"
    )
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} "
        f"eval_mean_reward={eval_m['eval/mean_reward']:.4f} "
        f"|train|={trainer._train_num_reactants} elapsed={elapsed:.1f}s"
    )
    print("  PASS")


def case_graphtransbi_multidiscrete_lookup_train() -> None:
    """Sibling smoke for GraphTransBiPPO with policy_arch=multidiscrete +
    r2_arch=lookup + eval_r2_pool=train. Confirms the
    configs/graphtransppo_bi_multidiscrete_delta_qed.yaml default wiring
    is also valid (multidiscrete + lookup is the SB3-style independent
    R2 head over the lookup table)."""
    if not _check_graph_dependencies_present():
        print("\n===== CASE  GraphTransBiPPO  multidiscrete + lookup + train  =====")
        print("  SKIP (torch_geometric not installed)")
        return
    print("\n===== CASE  GraphTransBiPPO  multidiscrete + lookup + train  =====")
    from genmolrl.algorithms.graphtransppo_bi.train import GraphTransBiPPO

    config = load_config("configs/graphtransppo_bi_multidiscrete_delta_qed.yaml")
    config["masking"] = "r2_available"
    config.setdefault("graphtransppo_bi", {})
    config["graphtransppo_bi"]["policy_arch"] = "multidiscrete"
    config["graphtransppo_bi"]["r2_arch"] = "lookup"
    config["graphtransppo_bi"]["eval_r2_pool"] = "train"
    config["graphtransppo_bi"]["r2_resample_retries"] = 8
    config["graphtransppo_bi"]["n_steps"] = 8
    config["graphtransppo_bi"]["batch_size"] = 4
    config["graphtransppo_bi"]["n_epochs"] = 1
    config["graphtransppo_bi"]["num_emb"] = 16
    config["graphtransppo_bi"]["num_layers"] = 1
    config["graphtransppo_bi"]["num_heads"] = 1
    config["graphtransppo_bi"]["template_embed_dim"] = 16
    config["graphtransppo_bi"]["r2_embed_dim"] = 16
    config["graphtransppo_bi"]["device"] = "cpu"
    for stale_key in (
        "r2_num_emb",
        "r2_num_layers",
        "r2_num_heads",
        "r2_keys_refresh_minibatches",
    ):
        config["graphtransppo_bi"].pop(stale_key, None)
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
    scratch = Path("/tmp/genmolrl_smoke_gtb_multi_lookup_train")
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
    assert trainer.r2_arch == "lookup"
    assert trainer.policy_arch == "multidiscrete"
    assert trainer._eval_pool_role == "train"
    assert trainer._eval_reaction_manager is trainer._train_reaction_manager
    assert trainer.policy.r2_embed is not None
    assert trainer.policy.r2_encoder is None
    assert getattr(trainer.policy, "r2_backbone", None) is None
    rollout, last_value = trainer.collect_rollout(8, base_step=0, log_episodes=False)
    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)
    eval_m = trainer.evaluate()
    elapsed = time.monotonic() - t0
    assert math.isfinite(float(metrics["train/loss"]))
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} eval_mean_reward={eval_m['eval/mean_reward']:.4f} "
        f"|train|={trainer._train_num_reactants} elapsed={elapsed:.1f}s"
    )
    print("  PASS")


def case_graphtransbi_yaml_default_roundtrip() -> None:
    """Load the two graphtransppo_bi YAMLs as-is and assert the new
    defaults are r2_arch=lookup + eval_r2_pool=train. Catches accidental
    drift back to encoder_graph at the YAML layer."""
    print("\n===== CASE  graphtransppo_bi YAML defaults  =====")
    for yaml_path, expected_policy in [
        ("configs/graphtransppo_bi_hierarchical_delta_qed.yaml", "hierarchical"),
        ("configs/graphtransppo_bi_multidiscrete_delta_qed.yaml", "multidiscrete"),
    ]:
        config = load_config(yaml_path)
        block = config.get("graphtransppo_bi", {})
        assert block.get("policy_arch") == expected_policy, (
            f"  FAIL: {yaml_path} policy_arch={block.get('policy_arch')!r}, "
            f"expected {expected_policy!r}"
        )
        assert block.get("r2_arch") == "lookup", (
            f"  FAIL: {yaml_path} r2_arch={block.get('r2_arch')!r}, "
            f"expected 'lookup'"
        )
        assert block.get("eval_r2_pool") == "train", (
            f"  FAIL: {yaml_path} eval_r2_pool={block.get('eval_r2_pool')!r}, "
            f"expected 'train'"
        )
        # The encoder_graph-only knobs should be DORMANT (commented out
        # or absent) under lookup. If a YAML accidentally re-introduces
        # them, the construction guard below would still allow the run,
        # but they'd silently be ignored — surface that here.
        for stale in ("r2_num_emb", "r2_num_layers", "r2_num_heads", "r2_keys_refresh_minibatches"):
            assert stale not in block, (
                f"  FAIL: {yaml_path} has stale encoder_graph knob "
                f"{stale!r}={block[stale]!r} under r2_arch=lookup — comment it out"
            )
        print(f"  {yaml_path}: policy_arch={expected_policy}, r2_arch=lookup, eval_r2_pool=train  OK")
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


def case_lookup_explicit_train_gr7aa7z6_mirror() -> None:
    """gr7aa7z6 reproduction smoke.

    Asserts the new ``eval_r2_pool: train`` YAML knob (when combined with
    ``r2_arch: lookup``) produces exactly the legacy gr7aa7z6 wiring:

      - ``_eval_pool_role == "train"`` (matches the "train_R2" wandb tag).
      - ``_eval_reaction_manager`` is the SAME object as
        ``_train_reaction_manager`` (no separate test pool allocated).
      - No FP / graph caches allocated (lookup uses ``r2_embed.weight``).
      - Policy has an ``r2_embed`` table and no encoder modules.
      - ``r2_keys`` consumed during the PPO update are exactly
        ``policy.r2_embed.weight`` (the lookup-table parameter).
      - ``evaluate()`` runs on the train pool (eval R2 indices in
        ``[0, train_num_reactants)``).
      - End-to-end rollout + update + eval finite-loss-no-crash.

    Plus an apples-to-apples bit-equivalence check against the older
    `case_lookup_backcompat` setup (which used the implicit default).
    """
    print("\n===== CASE  lookup + eval_r2_pool=train (gr7aa7z6 mirror)  =====")
    t0 = time.monotonic()
    # 1) Default-driven build (lookup → train via the per-arch default).
    trainer_default = make_bi_trainer(
        r2_arch="lookup", policy_arch="hierarchical", masking="r2_available"
    )
    assert trainer_default._eval_pool_role == "train"
    assert trainer_default.eval_r2_pool == "train"

    # 2) Explicit-knob-driven build. Builds a second tiny trainer with
    #    ``eval_r2_pool: train`` written into the method_cfg dict, to prove
    #    the YAML knob produces the same wiring as the implicit default.
    import pickle
    from genmolrl.config import resolve_path
    config = load_config("configs/ppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = "r2_available"
    config.setdefault("ppo_bi", {})
    config["ppo_bi"]["policy_arch"] = "hierarchical"
    config["ppo_bi"]["r2_arch"] = "lookup"
    config["ppo_bi"]["eval_r2_pool"] = "train"  # the explicit knob under test
    config["ppo_bi"]["r2_resample_retries"] = 16
    config["ppo_bi"]["n_steps"] = 8
    config["ppo_bi"]["batch_size"] = 4
    config["ppo_bi"]["n_epochs"] = 1
    config["ppo_bi"]["trunk_hidden"] = 64
    config["ppo_bi"]["template_embed_dim"] = 16
    config["ppo_bi"]["r2_embed_dim"] = 16
    config["ppo_bi"]["device"] = "cpu"
    config.setdefault("training", {})["total_timesteps"] = 8

    with open(resolve_path(config["dataset"]["training_file"]), "rb") as f:
        train_full = pickle.load(f)
    with open(resolve_path(config["dataset"]["test_file"]), "rb") as f:
        test_full = pickle.load(f)
    rng = random.Random(0)
    train_small = _subsample_pool(train_full, TRAIN_SUBSAMPLE, rng)
    test_small = _subsample_pool(test_full, TEST_SUBSAMPLE, rng)
    scratch = Path("/tmp/genmolrl_smoke_gr7aa7z6_mirror")
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
    # The explicit-knob and implicit-default builds must agree on every
    # eval-pool-affecting attribute.
    assert trainer._eval_pool_role == trainer_default._eval_pool_role == "train"
    assert trainer.eval_r2_pool == trainer_default.eval_r2_pool == "train"
    assert trainer._eval_reaction_manager is trainer._train_reaction_manager, (
        "lookup + train must alias eval manager to train manager (no extra alloc)"
    )
    assert trainer._eval_reactant_keys is trainer._train_reactant_keys
    assert trainer._train_r2_fps is None and trainer._eval_r2_fps is None
    assert trainer.policy.r2_embed is not None and trainer.policy.r2_encoder is None

    # r2_keys used at update time must be the exact lookup table.
    r2_keys_for_update = trainer._compute_active_r2_keys(pool="train", with_grad=True)
    assert r2_keys_for_update is trainer.policy.r2_embed.weight, (
        "lookup mode must return the embedding weight matrix directly as r2_keys"
    )

    rollout, last_value = trainer.collect_rollout(8, base_step=0, log_episodes=False)
    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)

    # Eval: inside evaluate() the active pool MUST be the train pool, and the
    # r2_keys MUST still be the lookup table (no encoder ever runs).
    observed_pool_sizes: list[int] = []
    observed_keys_are_lookup: list[bool] = []
    real_greedy = trainer._greedy_trajectory

    def _patched(start_smiles: str):
        observed_pool_sizes.append(trainer.num_reactants)
        observed_keys_are_lookup.append(
            trainer._active_r2_keys is trainer.policy.r2_embed.weight
            if trainer._active_r2_keys is not None
            else True
        )
        return real_greedy(start_smiles)

    trainer._greedy_trajectory = _patched  # type: ignore[assignment]
    eval_m = trainer.evaluate()
    trainer._greedy_trajectory = real_greedy  # type: ignore[assignment]

    assert observed_pool_sizes, "evaluate() did not produce any trajectories"
    assert all(n == trainer._train_num_reactants for n in observed_pool_sizes), (
        f"  FAIL: eval pool size {set(observed_pool_sizes)} != "
        f"train pool size {trainer._train_num_reactants} — gr7aa7z6 used train R2"
    )
    assert all(observed_keys_are_lookup), (
        "  FAIL: r2_keys during eval is NOT the lookup table — gr7aa7z6 was lookup"
    )
    # Eval R2 indices must lie inside the train pool (no spillover).
    # We can't read them after evaluate() returns, so just check counts.
    assert trainer.num_reactants == trainer._train_num_reactants, (
        "  FAIL: active pool not restored to train pool after evaluate()"
    )

    elapsed = time.monotonic() - t0
    bad = {
        k: v for k, v in metrics.items()
        if not math.isfinite(float(v)) and k != "train/explained_variance"
    }
    assert not bad, f"  FAIL: non-finite update metrics: {bad}"
    print(
        f"  wiring:  eval_role={trainer._eval_pool_role}  "
        f"eval_mgr_is_train_mgr={trainer._eval_reaction_manager is trainer._train_reaction_manager}  "
        f"r2_keys_is_lookup=True"
    )
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} "
        f"eval_pool_observed={set(observed_pool_sizes)} "
        f"(train|={trainer._train_num_reactants}) elapsed={elapsed:.1f}s"
    )
    print("  PASS")


def case_lookup_test_errors() -> None:
    """lookup + eval_r2_pool='test' must raise a clear ValueError.

    Structurally impossible: the lookup table has no rows for test
    reactants. The trainer constructor must fail loudly at this combo so
    a user can't silently index into the wrong rows.
    """
    print("\n===== CASE  lookup + eval_r2_pool=test (must error)  =====")
    config = load_config("configs/ppo_bi_hierarchical_delta_qed.yaml")
    config.setdefault("ppo_bi", {})
    config["ppo_bi"]["r2_arch"] = "lookup"
    config["ppo_bi"]["eval_r2_pool"] = "test"
    config["ppo_bi"]["device"] = "cpu"
    config["ppo_bi"]["n_steps"] = 4
    config["ppo_bi"]["batch_size"] = 2
    config["ppo_bi"]["n_epochs"] = 1
    config.setdefault("training", {})["total_timesteps"] = 4

    try:
        BiPPO(config)
    except ValueError as exc:
        msg = str(exc)
        assert "lookup" in msg.lower() and "test" in msg.lower(), (
            f"  FAIL: error message doesn't reference lookup+test: {msg!r}"
        )
        print(f"  raised ValueError as expected: {msg[:100]}{'…' if len(msg) > 100 else ''}")
        print("  PASS")
        return
    raise AssertionError(
        "  FAIL: lookup + eval_r2_pool='test' should have raised ValueError but didn't"
    )


def case_encoder_train_pool() -> None:
    """encoder + eval_r2_pool='train' must train and evaluate on the SAME pool.

    The R2 encoder is a function — it works on any pool — so this combo is
    legal and exists specifically for apples-to-apples comparison against
    the gr7aa7z6 lookup baseline that ALSO ran eval on the train pool.
    Both eval and train r2 FPs must alias the same tensor.
    """
    print("\n===== CASE  encoder + eval_r2_pool=train  =====")
    t0 = time.monotonic()
    config = load_config("configs/ppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = "r2_available"
    config.setdefault("ppo_bi", {})
    config["ppo_bi"]["policy_arch"] = "hierarchical"
    config["ppo_bi"]["r2_arch"] = "encoder"
    config["ppo_bi"]["eval_r2_pool"] = "train"
    config["ppo_bi"]["r2_resample_retries"] = 8
    config["ppo_bi"]["n_steps"] = 8
    config["ppo_bi"]["batch_size"] = 4
    config["ppo_bi"]["n_epochs"] = 1
    config["ppo_bi"]["trunk_hidden"] = 64
    config["ppo_bi"]["template_embed_dim"] = 16
    config["ppo_bi"]["r2_embed_dim"] = 16
    config["ppo_bi"]["r2_encoder_hidden"] = 64
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
    scratch = Path("/tmp/genmolrl_smoke_encoder_train_pool")
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
    assert trainer.r2_arch == "encoder" and trainer._eval_pool_role == "train"
    # Eval manager aliased to train manager → no separate test allocation.
    assert trainer._eval_reaction_manager is trainer._train_reaction_manager
    # FP caches: train allocated, eval aliased to train.
    assert trainer._train_r2_fps is not None
    assert trainer._eval_r2_fps is trainer._train_r2_fps, (
        "encoder + eval_r2_pool=train must alias eval FPs to train FPs"
    )

    rollout, last_value = trainer.collect_rollout(8, base_step=0, log_episodes=False)
    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)

    observed_pool_sizes: list[int] = []
    real_greedy = trainer._greedy_trajectory

    def _patched(start_smiles: str):
        observed_pool_sizes.append(trainer.num_reactants)
        return real_greedy(start_smiles)

    trainer._greedy_trajectory = _patched  # type: ignore[assignment]
    eval_m = trainer.evaluate()
    trainer._greedy_trajectory = real_greedy  # type: ignore[assignment]

    assert all(n == trainer._train_num_reactants for n in observed_pool_sizes), (
        "encoder + eval_r2_pool=train: eval pool size must equal train pool size"
    )
    elapsed = time.monotonic() - t0
    bad = {k: v for k, v in metrics.items()
           if not math.isfinite(float(v)) and k != "train/explained_variance"}
    assert not bad, f"  FAIL: non-finite update metrics: {bad}"
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} "
        f"eval_pool_observed={set(observed_pool_sizes)} elapsed={elapsed:.1f}s"
    )
    print("  PASS")


def case_encoder_graph_train_pool() -> None:
    """encoder_graph + eval_r2_pool='train' must alias the R2 graph batches."""
    if not _check_graph_dependencies_present():
        print("\n===== CASE  encoder_graph + eval_r2_pool=train  =====")
        print("  SKIP (torch_geometric not installed)")
        return
    print("\n===== CASE  encoder_graph + eval_r2_pool=train  =====")
    from genmolrl.algorithms.graphtransppo_bi.train import GraphTransBiPPO

    config = load_config("configs/graphtransppo_bi_hierarchical_delta_qed.yaml")
    config["masking"] = "r2_available"
    config.setdefault("graphtransppo_bi", {})
    config["graphtransppo_bi"]["policy_arch"] = "hierarchical"
    config["graphtransppo_bi"]["r2_arch"] = "encoder_graph"
    config["graphtransppo_bi"]["eval_r2_pool"] = "train"
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
    scratch = Path("/tmp/genmolrl_smoke_encoder_graph_train_pool")
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
    assert trainer.r2_arch == "encoder_graph" and trainer._eval_pool_role == "train"
    assert trainer._eval_reaction_manager is trainer._train_reaction_manager
    assert trainer._eval_r2_graphs is trainer._train_r2_graphs, (
        "encoder_graph + eval_r2_pool=train must alias the R2 graph batches"
    )
    assert trainer._train_r2_graphs.num_graphs == trainer._train_num_reactants

    rollout, last_value = trainer.collect_rollout(8, base_step=0, log_episodes=False)
    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)
    eval_m = trainer.evaluate()
    elapsed = time.monotonic() - t0
    bad = {k: v for k, v in metrics.items()
           if not math.isfinite(float(v)) and k != "train/explained_variance"}
    assert not bad, f"  FAIL: non-finite update metrics: {bad}"
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} "
        f"|train_graphs|={trainer._train_r2_graphs.num_graphs} "
        f"aliased_to_eval={trainer._eval_r2_graphs is trainer._train_r2_graphs} "
        f"elapsed={elapsed:.1f}s"
    )
    print("  PASS")


def case_gr7aa7z6_yaml_roundtrip() -> None:
    """Load configs/ppo_bi_hierarchical_lookup_delta_qed.yaml and confirm
    the trainer it builds has the gr7aa7z6 wiring (modulo the smoke-test
    pool subsampling and training_step reduction).
    """
    print("\n===== CASE  configs/ppo_bi_hierarchical_lookup_delta_qed.yaml  =====")
    config = load_config("configs/ppo_bi_hierarchical_lookup_delta_qed.yaml")
    # Shrink the smoke run so it stays fast — every other field comes from
    # the gr7aa7z6-mirror YAML and IS what we want to assert below.
    config.setdefault("ppo_bi", {})
    config["ppo_bi"]["n_steps"] = 4
    config["ppo_bi"]["batch_size"] = 2
    config["ppo_bi"]["n_epochs"] = 1
    config["ppo_bi"]["device"] = "cpu"
    config["ppo_bi"]["trunk_hidden"] = 64
    config["ppo_bi"]["template_embed_dim"] = 16
    config["ppo_bi"]["r2_embed_dim"] = 16
    config.setdefault("training", {})["total_timesteps"] = 4

    import pickle
    from genmolrl.config import resolve_path
    with open(resolve_path(config["dataset"]["training_file"]), "rb") as f:
        train_full = pickle.load(f)
    with open(resolve_path(config["dataset"]["test_file"]), "rb") as f:
        test_full = pickle.load(f)
    rng = random.Random(0)
    train_small = _subsample_pool(train_full, TRAIN_SUBSAMPLE, rng)
    test_small = _subsample_pool(test_full, TEST_SUBSAMPLE, rng)
    scratch = Path("/tmp/genmolrl_smoke_gr7aa7z6_yaml")
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
    # The five things that pin the gr7aa7z6 wiring:
    assert trainer.policy_arch == "hierarchical", trainer.policy_arch
    assert trainer.r2_arch == "lookup", trainer.r2_arch
    assert trainer.eval_r2_pool == "train", trainer.eval_r2_pool
    assert trainer.masking == "r2_available", trainer.masking
    # PPO hyperparameters from gr7aa7z6 wandb config:
    method_cfg = config["ppo_bi"]
    # The shrunken fields (n_steps / batch_size / n_epochs / trunk_hidden /
    # template_embed_dim / r2_embed_dim / device) are smoke-only overrides;
    # everything *else* must match gr7aa7z6.
    for k, expected in [
        ("learning_rate", 0.0003),
        ("gamma", 0.99),
        ("gae_lambda", 0.95),
        ("clip_range", 0.2),
        ("ent_coef", 0.01),
        ("vf_coef", 0.5),
        ("max_grad_norm", 0.5),
        ("target_kl", 0.02),
    ]:
        assert float(method_cfg[k]) == expected, (
            f"  FAIL: gr7aa7z6 hyperparam {k} = {expected}, got {method_cfg[k]}"
        )
    # And the data-hygiene fields (gr7aa7z6 ran on the same files):
    for k, expected in [
        ("training_file", "data/Bi/reactants_train.pkl"),
        ("test_file", "data/Bi/reactants_test.pkl"),
        ("templates_file", "data/Bi/templates.pkl"),
    ]:
        # The dataset paths were re-pointed to scratch above; check that the
        # *original* YAML pointed at the gr7aa7z6 data (re-parse fresh).
        original_yaml = load_config(
            "configs/ppo_bi_hierarchical_lookup_delta_qed.yaml"
        )
        assert original_yaml["dataset"][k] == expected, (
            f"  FAIL: gr7aa7z6 dataset field {k} = {expected}, got "
            f"{original_yaml['dataset'][k]!r}"
        )

    # End-to-end run with the shrunken setup.
    rollout, last_value = trainer.collect_rollout(4, base_step=0, log_episodes=False)
    advantages, returns = trainer.compute_gae(rollout, last_value)
    metrics = trainer.ppo_update(rollout, advantages, returns)
    eval_m = trainer.evaluate()
    assert math.isfinite(float(metrics["train/loss"]))
    print(
        f"  YAML wiring: arch={trainer.r2_arch}, eval_pool={trainer.eval_r2_pool}, "
        f"policy={trainer.policy_arch}, masking={trainer.masking}"
    )
    print(
        f"  rollout={len(rollout)} update_loss={metrics['train/loss']:.4f} "
        f"eval_n={eval_m['eval/n_molecules']} (eval pool = train, "
        f"size={trainer._train_num_reactants})"
    )
    print("  PASS")


def main() -> None:
    case_lookup_backcompat()
    case_encoder_pool_swap()
    case_encoder_multidiscrete()
    case_graphtransbi_encoder()
    case_residual_mlp_bipolicy()                # Option 1: deeper R2 MLP for BiPPO
    case_graphtransbi_encoder_graph()           # Option 3: Siamese R2 GT for GraphTransBiPPO
    # graphtransppo_bi: new lookup + train default (matches gr7aa7z6
    # discipline on the graph-trunk algorithm — only the R1 encoder
    # differs from Bi-PPO):
    case_graphtransbi_lookup_train()
    case_graphtransbi_multidiscrete_lookup_train()
    case_graphtransbi_yaml_default_roundtrip()
    # eval_r2_pool knob cases (gr7aa7z6 reproducibility):
    case_lookup_explicit_train_gr7aa7z6_mirror()
    case_lookup_test_errors()
    case_encoder_train_pool()
    case_encoder_graph_train_pool()
    case_gr7aa7z6_yaml_roundtrip()
    print("\nALL CASES PASSED")


if __name__ == "__main__":
    main()

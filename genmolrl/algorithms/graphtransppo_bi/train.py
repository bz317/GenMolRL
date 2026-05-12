"""PPO trainer for bi-reaction MultiDiscrete with a GraphTransformer R1 encoder.

``GraphTransBiPPO`` extends :class:`genmolrl.algorithms.ppo_bi.train.BiPPO`
by swapping the Morgan-fingerprint trunk for the ``GraphTransformer``
backbone used in GraphTransRL / GraphTransPPO. Every other PPO concern
(rollout collection, action sampling control flow for hierarchical /
multidiscrete, masking semantics for substructure / r2_available /
reaction_valid, joint rejection sampling, GAE, clipped surrogate, value
clipping, target_kl early stop, explained-variance reporting,
episode-level logging, checkpointing) is inherited verbatim from
``BiPPO``.

This is achieved by overriding the following small hook methods that
``BiPPO`` exposes:

  - ``_method_cfg``             → read from ``config['graphtransppo_bi']``.
  - ``_build_policy``           → instantiate :class:`GraphTransBiPolicy`.
  - ``_encode_smiles``          → build a graph batch and run the
                                   R1 GraphTransformer.
  - ``_supported_r2_archs``     → add the new ``'encoder_graph'`` option.
  - ``_init_extra_pool_data``   → cache R2 graph batches for both pools
                                   when ``r2_arch='encoder_graph'``.
  - ``_r2_pool_data_for``       → return the right pool data type
                                   (Batch under encoder_graph, FP tensor
                                   under encoder, ignored under lookup).

Three R2-side architectures are supported (selected via
``graphtransppo_bi.r2_arch``):

  - ``lookup`` (legacy): ``nn.Embedding(num_R2, r2_embed_dim)`` — fixed
    train pool, eval forced to the same pool.
  - ``encoder`` (mid-tier): MLP over Morgan FPs — asymmetric two-tower
    (R1 = graph, R2 = fingerprint). Train and test pools share the
    encoder weights.
  - ``encoder_graph`` (Option 3, current default): Siamese
    GraphTransformer over the R2 molecular graph + projection. Both
    towers are graph-encoded; defaults are tuned smaller than the R1
    side to keep the per-PPO-minibatch pool-encoding cost manageable
    over the ~116k candidate pool.
"""

from __future__ import annotations

import importlib.util

import torch

from genmolrl.algorithms.common import init_wandb
from genmolrl.algorithms.graphtransppo_bi.policy import GraphTransBiPolicy
from genmolrl.algorithms.graphtransrl.graph_transformer import batch_from_smiles
from genmolrl.algorithms.ppo_bi.train import BiPPO, run_training_loop


def require_graphtransppo_bi_dependencies() -> None:
    """Verify torch_geometric is importable before building the policy.

    Matches the check in ``graphtransppo.train`` so the failure mode is
    a clear ImportError at trainer construction rather than at the first
    ``batch_from_smiles`` call inside the rollout.
    """
    if importlib.util.find_spec("torch_geometric") is None:
        raise ImportError(
            "GraphTransPPO-Bi requires torch_geometric. Install it in the "
            "active environment, e.g. `python -m pip install torch-geometric "
            "-f https://data.pyg.org/whl/torch-2.3.0+cu121.html` for the "
            "current torch==2.3.0+cu121 environment."
        )


class GraphTransBiPPO(BiPPO):
    """Bi-PPO with a GraphTransformer encoder for the R1 trunk.

    Inherits everything from :class:`BiPPO` except the three small encoder
    hooks. The ``policy_arch`` ∈ {``hierarchical``, ``multidiscrete``}
    switch from ``BiPolicy`` carries over unchanged; the masking-mode
    contract (``substructure`` / ``r2_available`` allow -1, ``reaction_
    valid`` guarantees zero -1, with rejection sampling under
    multidiscrete + reaction_valid) is fully shared with the fingerprint
    trainer.

    Backward compatibility: ``BiPPO`` and ``ppo_bi_multidiscrete_*.yaml``
    runs are completely unaffected — they read ``config['ppo_bi']`` and
    instantiate ``BiPolicy`` via the default hook implementations.
    """

    def __init__(self, config: dict):
        require_graphtransppo_bi_dependencies()
        super().__init__(config)
        # ``encoder_graph`` mode is the only setting where per-minibatch R2
        # pool encoding becomes a real bottleneck (the Siamese R2
        # GraphTransformer is run over ~116k candidate graphs every call).
        # We refresh r2_keys WITH grad every ``r2_keys_refresh_minibatches``
        # PPO minibatches and reuse a detached copy in between; the R1 trunk,
        # template head, value head, and R2 query head still get gradient
        # every minibatch — only the Siamese R2 backbone's gradient signal
        # is downsampled. Trade-off: bigger refresh interval → faster
        # wall-clock but sparser gradient flow into the R2 backbone.
        # Default 8 is roughly "twice per PPO epoch" at the standard
        # 32-minibatches-per-epoch setting.
        method_cfg = self._method_cfg(config)
        self.r2_keys_refresh_minibatches = max(
            1, int(method_cfg.get("r2_keys_refresh_minibatches", 8))
        )
        self._cached_r2_keys: torch.Tensor | None = None
        self._minibatches_since_r2_refresh: int = 0

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _method_cfg(self, config: dict) -> dict:
        """Read the graph-trainer config block.

        Falls back to ``config['ppo_bi']`` for shared PPO knobs
        (learning_rate, n_steps, batch_size, ...) so a user can lift an
        existing ppo_bi YAML and only add the graph-specific encoder
        fields to a ``graphtransppo_bi:`` block.
        """
        return config.get("graphtransppo_bi", config.get("ppo_bi", {}))

    def _supported_r2_archs(self) -> set[str]:
        """Extend the base set with the Siamese-graph R2 encoder option."""
        return {"lookup", "encoder", "encoder_graph"}

    def _build_policy(self, method_cfg: dict) -> torch.nn.Module:
        return GraphTransBiPolicy(
            num_templates=self.num_templates,
            num_reactants=self.num_reactants,
            conditional_r2=(self.policy_arch == "hierarchical"),
            num_emb=int(method_cfg.get("num_emb", 64)),
            num_layers=int(method_cfg.get("num_layers", 3)),
            num_heads=int(method_cfg.get("num_heads", 2)),
            template_embed_dim=int(method_cfg.get("template_embed_dim", 64)),
            r2_embed_dim=int(method_cfg.get("r2_embed_dim", 64)),
            r2_arch=self.r2_arch,
            r2_encoder_hidden=method_cfg.get("r2_encoder_hidden"),
            r2_fp_dim=int(method_cfg.get("r2_fp_dim", 1024)),
            # Siamese R2 GraphTransformer knobs (Option 3). Ignored when
            # r2_arch != 'encoder_graph'. Defaults are intentionally
            # smaller than the R1 backbone because the R2 pool is
            # re-encoded per PPO minibatch over ~116k candidates.
            r2_num_emb=int(method_cfg.get("r2_num_emb", 32)),
            r2_num_layers=int(method_cfg.get("r2_num_layers", 1)),
            r2_num_heads=int(method_cfg.get("r2_num_heads", 2)),
        ).to(self.device)

    def _encode_smiles(self, smiles_list: list[str]) -> torch.Tensor:
        """Build a graph batch and run the GraphTransformer trunk.

        The trainer's call sites (``_sample_action_*``, ``collect_rollout``
        bootstrap, ``_evaluate_minibatch``) hit this single hook, so the
        rest of the BiPPO core is unchanged when the encoder switches.
        ``cond`` is an all-ones placeholder matching GraphTransPPO; this
        is the input to the conditioning MLP of the backbone and could
        later carry e.g. the current episode step depth.
        """
        graph = batch_from_smiles(smiles_list, device=self.device)
        cond = torch.ones((len(smiles_list), 1), device=self.device)
        return self.policy.forward_trunk(graph, cond)

    # ------------------------------------------------------------------
    # R2-graph pool caches (Option 3)
    # ------------------------------------------------------------------

    def _init_extra_pool_data(self) -> None:
        """Pre-build R2 graph batches for both pools under ``encoder_graph``.

        Called by ``BiPPO.__init__`` after the reaction managers and
        ``self._train_reactant_keys`` / ``self._eval_reactant_keys`` are
        populated. We build the Batch once per pool because the R2 SMILES
        list is fixed for the whole run; per-PPO-minibatch we re-run the
        R2 GraphTransformer over the cached Batch to produce ``r2_keys``.
        Building once amortises the SMILES-parsing + bond-feature cost,
        which is otherwise repeated thousands of times.

        Memory cost for a ~116k pool with average ~20 atoms each: ~150-300
        MB on GPU. Trade-off accepted because the alternative (per-step
        SMILES parsing) is two-orders-of-magnitude slower.

        No-op in ``r2_arch in {'lookup', 'encoder'}`` because those archs
        consume ``r2_embed.weight`` / Morgan-FP tensors that the base
        class already initialised.

        When ``_eval_pool_role == "train"`` (i.e. ``eval_r2_pool: train``
        in the YAML — the apples-to-apples comparison against the
        lookup / gr7aa7z6 baseline) the two graph batches are identical,
        so the eval-side cache is just a view onto the train-side cache
        — no second SMILES-parsing pass.
        """
        if self.r2_arch != "encoder_graph":
            self._train_r2_graphs = None
            self._eval_r2_graphs = None
            return
        self._train_r2_graphs = batch_from_smiles(
            self._train_reactant_keys, device=self.device
        )
        if self._eval_pool_role == "train":
            self._eval_r2_graphs = self._train_r2_graphs
        else:
            self._eval_r2_graphs = batch_from_smiles(
                self._eval_reactant_keys, device=self.device
            )

    def _r2_pool_data_for(self, pool: str):
        """Return graph Batch under ``encoder_graph``, else delegate to base.

        The trainer's :meth:`BiPPO._compute_active_r2_keys` calls this hook
        when the active arch is *not* ``lookup``; we route to the right
        pool's cached Batch when ``r2_arch='encoder_graph'`` and otherwise
        fall through to the Morgan-FP tensors already handled by
        ``BiPPO._r2_pool_data_for``.
        """
        if self.r2_arch == "encoder_graph":
            if pool == "train":
                return self._train_r2_graphs
            if pool == "eval":
                return self._eval_r2_graphs
            raise ValueError(f"pool must be 'train' or 'eval', got {pool!r}")
        return super()._r2_pool_data_for(pool)

    # ------------------------------------------------------------------
    # PPO update amortisation (Option 3 — Siamese R2 encoder is expensive)
    # ------------------------------------------------------------------

    def _begin_update_cycle(self) -> None:
        """Drop the cached r2_keys at the top of every :meth:`ppo_update`.

        Each PPO cycle starts with no cache, so the very first minibatch's
        ``_r2_keys_for_update`` triggers a fresh, gradient-attached pool
        encoding. The cache then refills until
        ``r2_keys_refresh_minibatches`` minibatches have been served.
        """
        self._cached_r2_keys = None
        self._minibatches_since_r2_refresh = 0

    def _r2_keys_for_update(self) -> torch.Tensor:
        """Cached-refresh policy for the Siamese R2 GraphTransformer.

        For ``r2_arch in {'lookup', 'encoder'}`` fall straight back to the
        base class (cheap encoders, no caching needed).

        For ``r2_arch='encoder_graph'``:

          - On the first minibatch of each PPO cycle (and every
            ``r2_keys_refresh_minibatches`` minibatches thereafter),
            encode the full R2 pool through the Siamese R2
            GraphTransformer **with gradients**. The R2 backbone
            receives gradient signal from this minibatch's PG / value
            loss via backprop through ``r2_keys``.
          - On intermediate minibatches, return a *detached* copy of the
            previous fresh encoding. Gradient still flows into the R1
            trunk / template head / value head / R2 query head, but the
            Siamese R2 backbone is held fixed for the next K-1 steps.

        This caps the per-update R2-encoding cost at roughly
        ``n_epochs * minibatches_per_epoch / K`` full-pool forwards
        instead of one per minibatch — a 10-30× wall-clock saving for
        K=8 at the standard PPO config, in exchange for a similarly
        sparser R2-backbone gradient.
        """
        if self.r2_arch != "encoder_graph":
            return super()._r2_keys_for_update()

        needs_refresh = (
            self._cached_r2_keys is None
            or self._minibatches_since_r2_refresh >= self.r2_keys_refresh_minibatches
        )
        if needs_refresh:
            fresh = self._compute_active_r2_keys(pool="train", with_grad=True)
            # Keep a detached snapshot for reuse on intermediate minibatches.
            # We can't reuse ``fresh`` directly across minibatches because
            # its computation graph is freed once the caller's loss.backward()
            # runs — that's why we detach for the cache and *return* the
            # graph-attached tensor only on the refresh minibatch.
            self._cached_r2_keys = fresh.detach()
            self._minibatches_since_r2_refresh = 1
            return fresh

        self._minibatches_since_r2_refresh += 1
        return self._cached_r2_keys


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def train(config: dict, experiment_name: str) -> None:
    """Entry point invoked by the unified launcher.

    Mirrors :func:`genmolrl.algorithms.ppo_bi.train.train` but tags the
    wandb run with the ``graphtransppo_bi`` algorithm string and reuses
    the shared ``run_training_loop`` so the rollout / eval / checkpoint
    cadence stays bit-identical between the two bi trainers.
    """
    require_graphtransppo_bi_dependencies()
    trainer = GraphTransBiPPO(config)
    run = init_wandb(config, "graphtransppo_bi", experiment_name)
    run_training_loop(trainer, run, config, experiment_name)


__all__ = ["GraphTransBiPPO", "train"]

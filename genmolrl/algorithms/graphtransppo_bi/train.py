"""PPO trainer for bi-reaction MultiDiscrete with a GraphTransformer R1 encoder.

``GraphTransBiPPO`` extends :class:`genmolrl.algorithms.ppo_bi.train.BiPPO`
by swapping the Morgan-fingerprint trunk for the ``GraphTransformer``
backbone used in GraphTransRL / GraphTransPPO. **Only the R1 encoder
changes** — every other PPO concern (rollout collection, action sampling
control flow for hierarchical / multidiscrete, masking semantics for
substructure / r2_available / reaction_valid, joint rejection sampling,
GAE, clipped surrogate, value clipping, target_kl early stop,
explained-variance reporting, episode-level logging, checkpointing) is
inherited verbatim from ``BiPPO``.

This is achieved by overriding three small hook methods that ``BiPPO``
exposes:

  - ``_method_cfg``     → read from ``config['graphtransppo_bi']``.
  - ``_build_policy``   → instantiate :class:`GraphTransBiPolicy`.
  - ``_encode_smiles``  → build a graph batch and run the GraphTransformer.

The R2 side is **not** graph-encoded: it stays as ``nn.Embedding(num_R2,
r2_embed_dim)`` for the same per-step cost reason as ``BiPolicy``
(running the GraphTransformer over ~116k candidate R2s per step is
infeasible). See the README's "Bi-GraphTransPPO" section for the rationale
and for the (future) Siamese R2 graph encoder variant.
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

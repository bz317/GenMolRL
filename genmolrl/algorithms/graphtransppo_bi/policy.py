"""Actor-critic policy for GraphTransPPO-Bi with switchable action distribution.

GraphTransBiPolicy = BiPolicy with the R1 trunk replaced by the
``GraphTransformer`` encoder used in GraphTransRL / GraphTransPPO. The
action heads (template, value, R2 query, R2 embedding/encoder, template
embedding) are structurally identical to
:class:`genmolrl.algorithms.ppo_bi.policy.BiPolicy`; only the encoder of
the current molecule R1 changes, from a 1024-d Morgan fingerprint passed
through an MLP to the molecular graph passed through a
``GraphTransformer``.

The R2 side supports the same two architectures as ``BiPolicy``:

  - ``r2_arch='lookup'`` (legacy default): a learned
    ``nn.Embedding(num_reactants, r2_embed_dim)`` lookup table. The R2
    pool is fixed at training time — see ``BiPolicy`` docstring for
    details.
  - ``r2_arch='encoder'``: a Morgan-FP MLP encoder shared between train
    and eval, so the trainer can swap the active R2 pool (train ↔ test)
    between rollout and evaluation. The R1 side stays graph-based; only
    the R2 side becomes vocabulary-free. This is the standard
    "two-tower retrieval with a learned encoder" parameterisation
    applied per modality (R1 = graph, R2 = fingerprint).

Both ``conditional_r2={True, False}`` are supported via the same flag as
``BiPolicy``:

  - ``conditional_r2=True``  → hierarchical: π(T, R2 | R1) = π_T(T | R1) ·
    π_R2(R2 | R1, T). The R2 query reads a learned per-template
    embedding alongside the trunk features.
  - ``conditional_r2=False`` → multidiscrete: π(T, R2 | R1) = π_T(T | R1) ·
    π_R2(R2 | R1). T and R2 are sampled independently given the graph
    embedding of R1.

The ``forward_trunk(graph_batch, cond)`` signature differs from
``BiPolicy.forward_trunk(fps)`` — the trainer's ``_encode_smiles`` hook
hides this from the rollout / update code paths.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from genmolrl.algorithms.graphtransrl.graph_transformer import (
    GraphFeatureSpec,
    GraphTransformer,
)


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Module:
    """Local ReLU MLP factory matching the one in ``ppo_bi.policy``.

    Inlined (not imported from ``ppo_bi.policy``) so the two policy modules
    stay independently importable — a downstream user might want to load
    one without paying the import cost / dependencies of the other.
    """
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class GraphTransBiPolicy(nn.Module):
    """Bi-reaction actor-critic with a GraphTransformer encoder for R1.

    Action space: ``MultiDiscrete([num_templates + 1, N_pool])`` where the
    last template index is the STOP action. ``N_pool`` is the cardinality of
    the *currently active* R2 pool — training pool during rollout/update,
    optionally the test pool during evaluation when ``r2_arch='encoder'``.

    Args:
        num_templates: number of reaction templates (excluding STOP).
        num_reactants: size of the training-time R2 pool. Only used to size
            the ``r2_embed`` table in ``r2_arch='lookup'`` mode; ignored in
            ``r2_arch='encoder'`` mode.
        conditional_r2: if True the R2 head consumes a per-template
            embedding (hierarchical mode); if False it only consumes the
            shared trunk (multidiscrete mode).
        num_emb: GraphTransformer embedding width (matches GraphTransRL /
            GraphTransPPO default). The pooled trunk feature dim is
            ``2 * num_emb``.
        num_layers: GraphTransformer layer count.
        num_heads: GraphTransformer attention heads.
        template_embed_dim: embedding dim for the T-conditioning vector
            consumed by the R2 head; ignored when ``conditional_r2=False``.
        r2_embed_dim: dimension of the R2 query target vectors.
        r2_arch: ``'lookup'`` (legacy ``nn.Embedding``) or ``'encoder'``
            (Morgan-FP MLP → r2_embed_dim, vocabulary-free).
        r2_encoder_hidden: hidden width of the R2 encoder MLP. Only used
            when ``r2_arch='encoder'``; defaults to ``r2_fp_dim // 2``.
        r2_fp_dim: input fingerprint dim for the R2 encoder MLP.
            Defaults to 1024 (Morgan FP length used in ``ppo_bi``).
    """

    def __init__(
        self,
        num_templates: int,
        num_reactants: int,
        *,
        conditional_r2: bool = True,
        num_emb: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
        template_embed_dim: int = 64,
        r2_embed_dim: int = 64,
        r2_arch: str = "lookup",
        r2_encoder_hidden: int | None = None,
        r2_fp_dim: int = 1024,
    ):
        super().__init__()
        self.num_templates = int(num_templates)
        self.num_reactants = int(num_reactants)
        self.action_dim_t = self.num_templates + 1  # includes STOP
        self.stop_index = self.num_templates
        self.conditional_r2 = bool(conditional_r2)
        self.r2_embed_dim = int(r2_embed_dim)

        self.r2_arch = str(r2_arch).lower()
        if self.r2_arch not in {"lookup", "encoder"}:
            raise ValueError(
                f"r2_arch must be 'lookup' or 'encoder', got {self.r2_arch!r}"
            )

        spec = GraphFeatureSpec()
        self.backbone = GraphTransformer(
            spec.node_dim,
            spec.edge_dim,
            spec.cond_dim,
            num_emb=num_emb,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        # ``GraphTransformer.forward(...)`` pools to a 2*num_emb graph-level
        # feature (mean-pooled atom nodes + the conditioning virtual node),
        # which becomes the trunk feature consumed by every head.
        graph_dim = num_emb * 2
        self.trunk_dim = graph_dim

        self.template_head = nn.Linear(graph_dim, self.action_dim_t)
        self.value_head = nn.Linear(graph_dim, 1)

        # The template_embed module is always allocated so the checkpoint
        # key set is stable across modes; under multidiscrete it has no
        # gradient signal because the R2 head never indexes it.
        self.template_embed = nn.Embedding(self.num_templates, template_embed_dim)
        if self.conditional_r2:
            self.r2_query_head = _mlp(
                graph_dim + template_embed_dim,
                graph_dim,
                r2_embed_dim,
                n_layers=2,
            )
        else:
            self.r2_query_head = _mlp(
                graph_dim,
                graph_dim,
                r2_embed_dim,
                n_layers=2,
            )

        if self.r2_arch == "lookup":
            self.r2_embed = nn.Embedding(self.num_reactants, r2_embed_dim)
            self.r2_encoder = None
            self.r2_fp_dim = None
        else:
            self.r2_fp_dim = int(r2_fp_dim)
            r2_hidden = (
                int(r2_encoder_hidden)
                if r2_encoder_hidden is not None
                else max(self.r2_fp_dim // 2, r2_embed_dim)
            )
            self.r2_encoder = _mlp(
                self.r2_fp_dim, r2_hidden, r2_embed_dim, n_layers=2
            )
            self.r2_embed = None

    # ------------------------------------------------------------------
    # Encoder + heads
    # ------------------------------------------------------------------

    def forward_trunk(
        self, graph_batch, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the pooled graph-level trunk features for a batch.

        ``graph_batch`` is a ``torch_geometric.data.Batch`` built by
        :func:`genmolrl.algorithms.graphtransrl.graph_transformer.batch_from_smiles`.
        ``cond`` is the optional conditioning vector consumed by the
        backbone (the trainer passes an all-ones placeholder, matching
        the GraphTransPPO convention).
        """
        _, graph_embeddings = self.backbone(graph_batch, cond)
        return graph_embeddings

    def template_logits(self, trunk_feats: torch.Tensor) -> torch.Tensor:
        return self.template_head(trunk_feats)

    def value(self, trunk_feats: torch.Tensor) -> torch.Tensor:
        return self.value_head(trunk_feats).squeeze(-1)

    def encode_r2_pool(self, r2_fps: torch.Tensor | None) -> torch.Tensor:
        """Return ``r2_keys`` of shape ``(N_pool, r2_embed_dim)`` for the active pool.

        Same semantics as :meth:`BiPolicy.encode_r2_pool`:
        - ``r2_arch='lookup'``: returns ``self.r2_embed.weight`` (``r2_fps``
          ignored, table fixed to training pool).
        - ``r2_arch='encoder'``: applies the R2 MLP to ``r2_fps``; the
          output is differentiable so PPO updates flow into the encoder.
        """
        if self.r2_arch == "lookup":
            return self.r2_embed.weight
        if r2_fps is None:
            raise ValueError(
                "r2_arch='encoder' requires r2_fps to compute r2_keys"
            )
        return self.r2_encoder(r2_fps)

    def r2_logits(
        self,
        trunk_feats: torch.Tensor,
        t_idx: torch.Tensor | None = None,
        r2_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """R2 logits over the current R2 pool.

        Same signature as :meth:`BiPolicy.r2_logits`. The trainer passes
        ``t_idx`` only under the hierarchical architecture; under
        multidiscrete it passes ``None`` and the query depends solely on
        the graph-encoded trunk features. ``r2_keys`` follows the same
        rules as in ``BiPolicy``: required in encoder mode, defaults to
        ``self.r2_embed.weight`` in lookup mode.
        """
        if self.conditional_r2:
            if t_idx is None:
                raise ValueError(
                    "conditional_r2=True requires t_idx for r2_logits()"
                )
            tmpl_vec = self.template_embed(t_idx)
            query = self.r2_query_head(torch.cat([trunk_feats, tmpl_vec], dim=-1))
        else:
            query = self.r2_query_head(trunk_feats)
        if r2_keys is None:
            if self.r2_arch == "encoder":
                raise ValueError(
                    "r2_arch='encoder' requires r2_keys for r2_logits() — "
                    "compute via policy.encode_r2_pool(r2_fps) and pass in."
                )
            r2_keys = self.r2_embed.weight
        return query @ r2_keys.T


__all__ = ["GraphTransBiPolicy"]

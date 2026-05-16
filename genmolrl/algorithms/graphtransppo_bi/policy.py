"""Actor-critic policy for GraphTransPPO-Bi with switchable action distribution.

GraphTransBiPolicy = BiPolicy with the R1 trunk replaced by the
``GraphTransformer`` encoder used in GraphTransRL / GraphTransPPO. The
action heads (template, value, R2 query, R2 embedding/encoder, template
embedding) are structurally identical to
:class:`genmolrl.algorithms.ppo_bi.policy.BiPolicy`; only the encoder of
the current molecule R1 changes, from a 1024-d Morgan fingerprint passed
through an MLP to the molecular graph passed through a
``GraphTransformer``.

The R2 side supports four architectures:

  - ``r2_arch='lookup'`` (legacy default): a learned
    ``nn.Embedding(num_reactants, r2_embed_dim)`` lookup table. The R2
    pool is fixed at training time — see ``BiPolicy`` docstring for
    details.
  - ``r2_arch='encoder'``: a Morgan-FP MLP encoder shared between train
    and eval, so the trainer can swap the active R2 pool (train ↔ test)
    between rollout and evaluation. The R1 side stays graph-based; only
    the R2 side stays fingerprint-based — the asymmetric "two-tower
    retrieval" variant that ``ppo_bi`` already uses.
  - ``r2_arch='encoder_graph'`` (Option 3 upgrade): a **two-tower**
    GraphTransformer for the R2 graph followed by a small projection to
    ``r2_embed_dim``. The R2 backbone has the same architecture *family*
    as the R1 backbone but **independent weights** (defaults are tuned
    smaller — ``r2_num_emb=32``, 1 layer — to keep the per-PPO-minibatch
    cost manageable over the ~116k candidate pool).
  - ``r2_arch='encoder_graph_shared'`` (true Siamese / weight-tied):
    R2 is encoded by the **same** ``GraphTransformer`` module instance
    as R1 (``self.backbone``). Only one linear projection
    (``2 * num_emb → r2_embed_dim``) is allocated on top. The shared
    encoder receives gradient signal from every R1 forward (template /
    value / R2-query heads) **and** every R2 pool encoding, so it
    learns faster and the R1/R2 dot-product head sees comparably
    encoded vectors. Per-PPO-minibatch pool-encoding cost is higher
    than ``encoder_graph`` (the R1 backbone is bigger than the default
    R2 backbone), so the trainer's existing
    ``r2_keys_refresh_minibatches`` caching applies here too. The
    ``r2_num_emb`` / ``r2_num_layers`` / ``r2_num_heads`` knobs are
    **ignored** under this arch because the architecture is fixed to
    the R1 backbone — set them under ``encoder_graph`` instead.

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
        r2_arch: ``'lookup'`` (legacy ``nn.Embedding``), ``'encoder'``
            (Morgan-FP MLP → r2_embed_dim, vocabulary-free),
            ``'encoder_graph'`` (two-tower R2 GraphTransformer with
            **independent** weights + projection — Option 3 upgrade),
            or ``'encoder_graph_shared'`` (true Siamese: R2 reuses
            ``self.backbone`` and only adds a projection).
        r2_encoder_hidden: hidden width of the R2 encoder MLP. Only used
            when ``r2_arch='encoder'``; defaults to ``r2_fp_dim // 2``.
        r2_fp_dim: input fingerprint dim for the R2 encoder MLP.
            Defaults to 1024 (Morgan FP length used in ``ppo_bi``).
            Ignored when ``r2_arch='encoder_graph'``.
        r2_num_emb: GraphTransformer embedding width for the R2
            backbone (used only when ``r2_arch='encoder_graph'``).
            Defaults to a smaller value than the R1 backbone because
            the R2 pool is encoded per-PPO-minibatch over ~116k
            candidates — keeping num_emb modest is the single biggest
            cost lever. The pooled R2 graph-level feature has width
            ``2 * r2_num_emb``.
        r2_num_layers: GraphTransformer layer count for the R2 backbone.
            Same cost trade-off as ``r2_num_emb``.
        r2_num_heads: GraphTransformer attention head count for the R2
            backbone.
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
        r2_num_emb: int = 32,
        r2_num_layers: int = 1,
        r2_num_heads: int = 2,
    ):
        super().__init__()
        self.num_templates = int(num_templates)
        self.num_reactants = int(num_reactants)
        self.action_dim_t = self.num_templates + 1  # includes STOP
        self.stop_index = self.num_templates
        self.conditional_r2 = bool(conditional_r2)
        self.r2_embed_dim = int(r2_embed_dim)

        self.r2_arch = str(r2_arch).lower()
        if self.r2_arch not in {"lookup", "encoder", "encoder_graph", "encoder_graph_shared"}:
            raise ValueError(
                f"r2_arch must be 'lookup', 'encoder', 'encoder_graph', or "
                f"'encoder_graph_shared', got {self.r2_arch!r}"
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

        # R2 encoder: four mutually-exclusive variants matching the four
        # supported r2_arch values. The same ``encode_r2_pool`` API hides the
        # variation from the trainer; only the input data type differs (none /
        # tensor / Batch). Attributes that don't apply to the active arch are
        # set to None to keep the checkpoint key set obvious. Under
        # ``encoder_graph_shared`` only ``self.r2_project`` is added — the
        # R2 encoder is ``self.backbone`` itself (weight-tied).
        self.r2_embed = None
        self.r2_encoder = None
        self.r2_backbone = None
        self.r2_project = None
        self.r2_fp_dim = None
        self.r2_num_emb = None
        if self.r2_arch == "lookup":
            self.r2_embed = nn.Embedding(self.num_reactants, r2_embed_dim)
        elif self.r2_arch == "encoder":
            self.r2_fp_dim = int(r2_fp_dim)
            r2_hidden = (
                int(r2_encoder_hidden)
                if r2_encoder_hidden is not None
                else max(self.r2_fp_dim // 2, r2_embed_dim)
            )
            self.r2_encoder = _mlp(
                self.r2_fp_dim, r2_hidden, r2_embed_dim, n_layers=2
            )
        elif self.r2_arch == "encoder_graph":
            # Two-tower R2 GraphTransformer + linear projection. Same
            # architecture family as the R1 backbone but with **independent**
            # weights — disentangling lets us shrink the R2 side aggressively
            # (defaults: 1 layer, num_emb=32) to keep the per-PPO-minibatch
            # pool-encoding cost manageable. The pooled graph feature has
            # width ``2 * r2_num_emb`` (atoms via global_mean_pool + the
            # conditioning virtual node), which is then projected to
            # ``r2_embed_dim`` so the dot-product head sees the same key
            # dim regardless of which R2 arch is active.
            self.r2_num_emb = int(r2_num_emb)
            self.r2_backbone = GraphTransformer(
                spec.node_dim,
                spec.edge_dim,
                spec.cond_dim,
                num_emb=self.r2_num_emb,
                num_layers=int(r2_num_layers),
                num_heads=int(r2_num_heads),
            )
            self.r2_project = nn.Linear(self.r2_num_emb * 2, r2_embed_dim)
        else:
            # ``encoder_graph_shared``: true Siamese / weight-tied. R2 is
            # encoded by the **same** ``self.backbone`` module that encodes
            # R1, so the encoder learns from gradient signal flowing in
            # from both sides of the retrieval head. Only a single linear
            # projection ``2 * num_emb → r2_embed_dim`` is allocated on top
            # so the dot-product head sees a consistent key dimension.
            # The R1 backbone here is the bigger ``num_emb=64`` / 3-layer
            # default, which gives R2 a much stronger encoder for free at
            # the cost of a more expensive per-PPO-minibatch pool forward
            # — but the trainer's ``r2_keys_refresh_minibatches`` caching
            # already exists to amortise that.
            self.r2_project = nn.Linear(graph_dim, r2_embed_dim)

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

    def encode_r2_pool(self, r2_pool_data) -> torch.Tensor:
        """Return ``r2_keys`` of shape ``(N_pool, r2_embed_dim)`` for the active pool.

        Generalises :meth:`BiPolicy.encode_r2_pool` to four encoder
        variants — the trainer passes a different ``r2_pool_data`` type
        for each arch and this method dispatches on ``self.r2_arch``:

        - ``r2_arch='lookup'``: returns ``self.r2_embed.weight``
          (``r2_pool_data`` ignored, table fixed to training pool).
        - ``r2_arch='encoder'``: expects ``r2_pool_data`` to be a
          ``(N_pool, r2_fp_dim)`` Morgan-FP tensor and applies the R2 MLP.
        - ``r2_arch='encoder_graph'``: expects ``r2_pool_data`` to be a
          ``torch_geometric.data.Batch`` of R2 molecular graphs, runs the
          INDEPENDENT-weights R2 GraphTransformer + linear projection.
          ``cond`` is an all-ones placeholder matching the R1
          convention; could later carry pool-side context (e.g. R2
          cluster id) the same way R1 could carry episode-step depth.
        - ``r2_arch='encoder_graph_shared'``: same input type as
          ``encoder_graph``, but the R2 GraphTransformer IS
          ``self.backbone`` (weight-tied to the R1 encoder), followed
          by ``self.r2_project``.

        The output is differentiable in all three encoder modes so PPO
        gradients flow into whichever module is acting as the R2
        encoder.
        """
        if self.r2_arch == "lookup":
            return self.r2_embed.weight
        if r2_pool_data is None:
            raise ValueError(
                f"r2_arch={self.r2_arch!r} requires r2_pool_data to compute r2_keys"
            )
        if self.r2_arch == "encoder":
            return self.r2_encoder(r2_pool_data)
        # encoder_graph[_shared]: r2_pool_data is a torch_geometric Batch.
        n_graphs = int(r2_pool_data.num_graphs)
        cond = torch.ones((n_graphs, 1), device=r2_pool_data.x.device)
        backbone = self.r2_backbone if self.r2_arch == "encoder_graph" else self.backbone
        _, graph_embeds = backbone(r2_pool_data, cond)
        return self.r2_project(graph_embeds)

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
            if self.r2_arch != "lookup":
                raise ValueError(
                    f"r2_arch={self.r2_arch!r} requires r2_keys for r2_logits() — "
                    "compute via policy.encode_r2_pool(...) and pass in."
                )
            r2_keys = self.r2_embed.weight
        return query @ r2_keys.T


__all__ = ["GraphTransBiPolicy"]

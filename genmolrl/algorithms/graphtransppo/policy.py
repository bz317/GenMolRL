"""Actor-critic policy for GraphTransPPO.

Reuses the :class:`GraphTransformer` backbone shipped with GraphTransRL and
adds a per-graph scalar value head so PPO can baseline trajectory rewards
with V(s). Kept as a separate class (instead of mutating
:class:`GraphTransRLPolicy`) so the existing GraphTransRL trainer stays
bit-for-bit unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from genmolrl.algorithms.graphtransrl.graph_transformer import (
    GraphFeatureSpec,
    GraphTransformer,
    mlp,
)


class GraphTransPPOPolicy(nn.Module):
    """Graph-transformer actor-critic over Stop + reaction-template actions."""

    def __init__(
        self,
        num_templates: int,
        *,
        num_emb: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
    ):
        super().__init__()
        spec = GraphFeatureSpec()
        self.backbone = GraphTransformer(
            spec.node_dim,
            spec.edge_dim,
            spec.cond_dim,
            num_emb=num_emb,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        graph_dim = num_emb * 2
        self.stop_head = mlp(graph_dim, num_emb, 1, 1)
        self.template_head = mlp(graph_dim, num_emb, num_templates, 1)
        self.value_head = mlp(graph_dim, num_emb, 1, 1)
        self.num_templates = int(num_templates)

    def actor_critic(
        self, graph_batch, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, graph_embeddings = self.backbone(graph_batch, cond)
        logits = torch.cat(
            [self.template_head(graph_embeddings), self.stop_head(graph_embeddings)],
            dim=-1,
        )
        value = self.value_head(graph_embeddings).squeeze(-1)
        return logits, value

    def forward(
        self, graph_batch, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor_critic(graph_batch, cond)


__all__ = ["GraphTransPPOPolicy"]

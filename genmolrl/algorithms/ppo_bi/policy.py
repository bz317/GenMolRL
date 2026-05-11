"""Actor-critic policy for Bi-PPO with switchable action distribution.

Supports two parameterisations of the bi-reaction MultiDiscrete action space::

    a = (T, R2),   T ∈ {0, …, num_templates - 1, STOP},   R2 ∈ {0, …, num_reactants - 1}

  - ``hierarchical`` (autoregressive):
        π(a | s) = π_T(T | s) · π_R2(R2 | s, T)
    The R2 query depends on the embedding of the sampled T, so the policy can
    learn template-conditional reactant preferences.

  - ``multidiscrete`` (independent):
        π(a | s) = π_T(T | s) · π_R2(R2 | s)
    The R2 query depends only on the shared trunk; T and R2 are sampled
    independently (matching what SB3 ``MaskablePPO`` does on a flat
    MultiDiscrete action space, but kept under the same trainer for parity).

In both modes the R2 logits are produced as a dot product against a learned
per-reactant embedding so the parameter count grows linearly with the pool
size. ``forward_trunk`` returns shared features once per state; the trainer
drives sampling, masking, and log-prob accounting.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Module:
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class BiPolicy(nn.Module):
    """Actor-critic with a template head and an R2 head (T-conditional or not).

    Action space: ``MultiDiscrete([num_templates + 1, num_reactants])`` where the
    last template index is the STOP action.

    Args:
        num_templates: number of reaction templates (excluding STOP).
        num_reactants: size of the R2 reactant pool.
        conditional_r2: if True the R2 head consumes a per-template embedding
            (autoregressive / hierarchical mode); if False the R2 head only
            consumes the shared trunk (independent / multidiscrete mode).
        obs_dim: input feature dim (Morgan FP length, default 1024).
        trunk_hidden: hidden width of the shared trunk and heads.
        template_embed_dim: embedding dimension for the T-conditioning vector
            consumed by the R2 head; ignored when ``conditional_r2=False``.
        r2_embed_dim: embedding dimension for the per-reactant query targets.
    """

    def __init__(
        self,
        num_templates: int,
        num_reactants: int,
        *,
        conditional_r2: bool = True,
        obs_dim: int = 1024,
        trunk_hidden: int = 256,
        template_embed_dim: int = 64,
        r2_embed_dim: int = 64,
    ):
        super().__init__()
        self.num_templates = int(num_templates)
        self.num_reactants = int(num_reactants)
        self.action_dim_t = self.num_templates + 1  # includes STOP
        self.stop_index = self.num_templates
        self.conditional_r2 = bool(conditional_r2)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
        )
        self.template_head = nn.Linear(trunk_hidden, self.action_dim_t)
        self.value_head = nn.Linear(trunk_hidden, 1)

        if self.conditional_r2:
            self.template_embed = nn.Embedding(self.num_templates, template_embed_dim)
            self.r2_query_head = _mlp(
                trunk_hidden + template_embed_dim,
                trunk_hidden,
                r2_embed_dim,
                n_layers=2,
            )
        else:
            # Independent R2 head: same query architecture but no T input. The
            # template_embed module is still allocated so checkpoints have a
            # stable key set across modes; it just has no gradient signal in
            # multidiscrete mode (we don't index it). Kept tiny in that case.
            self.template_embed = nn.Embedding(self.num_templates, template_embed_dim)
            self.r2_query_head = _mlp(
                trunk_hidden,
                trunk_hidden,
                r2_embed_dim,
                n_layers=2,
            )
        self.r2_embed = nn.Embedding(self.num_reactants, r2_embed_dim)

    def forward_trunk(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def template_logits(self, trunk_feats: torch.Tensor) -> torch.Tensor:
        return self.template_head(trunk_feats)

    def value(self, trunk_feats: torch.Tensor) -> torch.Tensor:
        return self.value_head(trunk_feats).squeeze(-1)

    def r2_logits(
        self, trunk_feats: torch.Tensor, t_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """R2 logits over the full reactant pool.

        When ``conditional_r2=True`` ``t_idx`` must be a LongTensor of shape
        ``(batch,)`` containing valid template indices (excluding STOP). When
        ``conditional_r2=False`` ``t_idx`` is ignored — the R2 distribution
        is conditioned only on the state, matching independent MultiDiscrete
        sampling.
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
        return query @ self.r2_embed.weight.T


__all__ = ["BiPolicy"]

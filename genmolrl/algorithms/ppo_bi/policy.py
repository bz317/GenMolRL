"""Actor-critic policy for Bi-PPO with switchable action distribution.

Supports two parameterisations of the bi-reaction MultiDiscrete action space::

    a = (T, R2),   T ∈ {0, …, num_templates - 1, STOP},   R2 ∈ {0, …, N_pool - 1}

  - ``hierarchical`` (autoregressive):
        π(a | s) = π_T(T | s) · π_R2(R2 | s, T)
    The R2 query depends on the embedding of the sampled T, so the policy can
    learn template-conditional reactant preferences.

  - ``multidiscrete`` (independent):
        π(a | s) = π_T(T | s) · π_R2(R2 | s)
    The R2 query depends only on the shared trunk; T and R2 are sampled
    independently (matching what SB3 ``MaskablePPO`` does on a flat
    MultiDiscrete action space, but kept under the same trainer for parity).

R2 representation (``r2_arch``):

  - ``lookup`` (legacy, default): a learned ``nn.Embedding(num_reactants,
    r2_embed_dim)`` table. The R2 pool is fixed at training time; the
    embedding row for the i-th reactant SMILES is keyed by its training-pool
    index. **Cannot** be swapped between pools at eval time without
    breaking the index mapping. Bit-identical to the original BiPolicy.

  - ``encoder``: an MLP that maps a Morgan fingerprint to an R2 key
    vector. The pool is no longer baked into the parameters — any R2
    SMILES can be embedded by computing ``encoder(morgan_fp(SMILES))``.
    The trainer can swap the active pool (train ↔ test) between rollout
    and evaluation; both pools use the same encoder weights, so the
    R2 distribution generalises to unseen reactants in the chemistry
    sense (same fingerprint → same key vector). When
    ``r2_encoder_residual=True`` (the new default for Bi-PPO YAMLs)
    the encoder is a deeper residual MLP (Option 1 upgrade): a 1024 →
    1024 stem, ``r2_encoder_n_res_blocks`` Pre-LN residual blocks of
    width 1024, then a 1024 → ``r2_embed_dim`` projection — roughly
    ~5.3M parameters versus ~558k for the legacy 2-layer MLP. With
    ``r2_encoder_residual=False`` the encoder falls back to the
    original plain 2-layer MLP so older YAMLs stay bit-identical.

``forward_trunk`` returns shared features once per state; the trainer
drives sampling, masking, and log-prob accounting. ``r2_logits`` is the
single call site that needs to know the active R2 pool: in encoder mode
the trainer pre-computes ``r2_keys = encode_r2_pool(r2_fps)`` and passes
them in; in lookup mode the trainer either passes ``None`` (defaulting to
``r2_embed.weight``) or passes the same weight matrix for symmetry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Module:
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class _ResBlock(nn.Module):
    """Pre-LN residual block: ``x → x + Linear(ReLU(Linear(LN(x))))``.

    Two Linears per block at the same hidden width so a deep stack composes
    cleanly without the residual path having to project. This is the same
    pattern as the FFN blocks in a Transformer (Pre-LN variant), which is
    the most stable choice for deeper-than-2-layer MLPs trained with PPO
    advantages (small batches → noisy gradients).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        return x + h


def _residual_mlp(
    in_dim: int,
    hidden: int,
    out_dim: int,
    *,
    n_res_blocks: int = 2,
) -> nn.Module:
    """Deep residual-MLP factory used by the R2 encoder under Option 1.

    Layout for ``n_res_blocks = K``::

        Linear(in_dim → hidden)
        LayerNorm(hidden) ; ReLU                # stem
        ResBlock(hidden) × K                    # body
        Linear(hidden → out_dim)                # projection

    Total Linears = ``2 + 2 * K`` (stem + 2 per block + projection); with
    ``hidden=1024`` and ``K=2`` that's ~5.3M parameters, an order of
    magnitude above the legacy 2-layer MLP (~558k params at 1024→512→64).
    Used when ``r2_encoder_residual=True`` is set on the policy; the
    legacy plain ``_mlp`` is retained for backward-compatible
    ``r2_encoder_residual=False`` runs.
    """
    if n_res_blocks <= 0:
        return _mlp(in_dim, hidden, out_dim, n_layers=2)
    stem = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
    )
    body = nn.Sequential(*[_ResBlock(hidden) for _ in range(n_res_blocks)])
    proj = nn.Linear(hidden, out_dim)
    return nn.Sequential(stem, body, proj)


class BiPolicy(nn.Module):
    """Actor-critic with a template head and an R2 head (T-conditional or not).

    Action space: ``MultiDiscrete([num_templates + 1, N_pool])`` where the
    last template index is the STOP action and ``N_pool`` is the cardinality
    of the *currently active* R2 pool (training pool during rollout/update;
    optionally the test pool during evaluation, see ``r2_arch='encoder'``).

    Args:
        num_templates: number of reaction templates (excluding STOP).
        num_reactants: size of the training-time R2 pool. Only used to size
            the ``r2_embed`` table in ``r2_arch='lookup'`` mode; ignored in
            ``r2_arch='encoder'`` mode (the encoder works on any pool).
        conditional_r2: if True the R2 head consumes a per-template embedding
            (autoregressive / hierarchical mode); if False the R2 head only
            consumes the shared trunk (independent / multidiscrete mode).
        obs_dim: input feature dim (Morgan FP length, default 1024).
        trunk_hidden: hidden width of the shared trunk and heads.
        template_embed_dim: embedding dimension for the T-conditioning vector
            consumed by the R2 head; ignored when ``conditional_r2=False``.
        r2_embed_dim: dimension of the R2 query target vectors.
        r2_arch: ``'lookup'`` (legacy fixed-pool ``nn.Embedding``) or
            ``'encoder'`` (Morgan-FP MLP → r2_embed_dim, vocabulary-free).
        r2_encoder_hidden: hidden width of the R2 encoder MLP. Only used when
            ``r2_arch='encoder'``; defaults to ``obs_dim // 2`` in plain-MLP
            mode and ``obs_dim`` (1024) in residual-MLP mode.
        r2_encoder_n_layers: number of Linear layers in the plain-MLP variant
            of the R2 encoder (used only when ``r2_encoder_residual=False``).
            Defaults to 2 for backward compatibility.
        r2_encoder_residual: if True, build the R2 encoder as a deep
            Pre-LN residual MLP (Option 1 upgrade): 1 stem Linear +
            ``r2_encoder_n_res_blocks`` residual blocks + 1 projection
            Linear. If False, use the plain 2-layer MLP (legacy
            behaviour, bit-identical to the original encoder).
        r2_encoder_n_res_blocks: number of residual blocks in the deep
            R2 encoder. Each block adds 2 Linears + LayerNorm at the
            same hidden width. Only used when
            ``r2_encoder_residual=True``. Defaults to 2.
        r2_fp_dim: input fingerprint dim for the R2 encoder MLP. Defaults to
            ``obs_dim`` because the same Morgan-FP layout is used for R1 and R2.
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
        r2_arch: str = "lookup",
        r2_encoder_hidden: int | None = None,
        r2_encoder_n_layers: int = 2,
        r2_encoder_residual: bool = False,
        r2_encoder_n_res_blocks: int = 2,
        r2_fp_dim: int | None = None,
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

        if self.r2_arch == "lookup":
            self.r2_embed = nn.Embedding(self.num_reactants, r2_embed_dim)
            self.r2_encoder = None
            self.r2_fp_dim = None
        else:
            # ``encoder``: R2 key = encoder(morgan_fp(R2_smiles)). The encoder is
            # the SAME network at training and evaluation, so swapping the active
            # R2 pool (train ↔ test) only changes which fingerprints get fed
            # in; the parameters are shared and any unseen pool generalises
            # in the standard "two-tower retrieval with a learned encoder"
            # sense. Two variants share this branch:
            #
            #   - r2_encoder_residual=False (legacy): plain 2-layer MLP
            #     (in_dim → hidden → r2_embed_dim). Bit-identical to the
            #     original Bi-PPO encoder so older runs reproduce.
            #
            #   - r2_encoder_residual=True (Option 1 upgrade, current
            #     default in ppo_bi YAMLs): deep Pre-LN residual MLP
            #     (in_dim → hidden, K residual blocks at hidden, hidden →
            #     r2_embed_dim). Defaults are tuned to ~5.3M parameters
            #     at hidden=1024 and K=2; layer norm + skip connections
            #     make the deeper stack stable under PPO advantage noise.
            self.r2_fp_dim = int(r2_fp_dim) if r2_fp_dim is not None else int(obs_dim)
            self.r2_encoder_residual = bool(r2_encoder_residual)
            if self.r2_encoder_residual:
                # Wider default for the residual variant so each block has
                # enough capacity to do useful nonlinear refinement on top
                # of the skip-connected pass-through.
                default_hidden = self.r2_fp_dim
                r2_hidden = (
                    int(r2_encoder_hidden)
                    if r2_encoder_hidden is not None
                    else max(default_hidden, r2_embed_dim)
                )
                self.r2_encoder = _residual_mlp(
                    self.r2_fp_dim,
                    r2_hidden,
                    r2_embed_dim,
                    n_res_blocks=int(r2_encoder_n_res_blocks),
                )
            else:
                r2_hidden = (
                    int(r2_encoder_hidden)
                    if r2_encoder_hidden is not None
                    else max(self.r2_fp_dim // 2, r2_embed_dim)
                )
                self.r2_encoder = _mlp(
                    self.r2_fp_dim,
                    r2_hidden,
                    r2_embed_dim,
                    n_layers=int(r2_encoder_n_layers),
                )
            self.r2_embed = None

    def forward_trunk(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def template_logits(self, trunk_feats: torch.Tensor) -> torch.Tensor:
        return self.template_head(trunk_feats)

    def value(self, trunk_feats: torch.Tensor) -> torch.Tensor:
        return self.value_head(trunk_feats).squeeze(-1)

    def encode_r2_pool(self, r2_fps: torch.Tensor | None) -> torch.Tensor:
        """Return ``r2_keys`` of shape ``(N_pool, r2_embed_dim)`` for the active pool.

        In ``r2_arch='lookup'`` mode the static ``r2_embed.weight`` is returned
        directly — ``r2_fps`` is ignored (the table is fixed to the training
        pool). In ``r2_arch='encoder'`` mode the MLP is applied to the pool
        fingerprints; the trainer must supply ``r2_fps`` (e.g. the train pool
        FPs at rollout time, the test pool FPs at eval time). The output is
        differentiable in encoder mode so PPO gradients flow into the encoder.
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

        When ``conditional_r2=True`` ``t_idx`` must be a LongTensor of shape
        ``(batch,)`` containing valid template indices (excluding STOP). When
        ``conditional_r2=False`` ``t_idx`` is ignored — the R2 distribution
        is conditioned only on the state, matching independent MultiDiscrete
        sampling.

        ``r2_keys`` is the ``(N_pool, r2_embed_dim)`` matrix that scores the
        R2 axis. In ``r2_arch='lookup'`` mode passing ``None`` defaults to
        ``self.r2_embed.weight``; in ``r2_arch='encoder'`` mode the trainer
        must pre-compute and pass it (typically from
        :meth:`encode_r2_pool`).
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


__all__ = ["BiPolicy"]

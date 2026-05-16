"""Hand-rolled PPO trainer for the bi-reaction MultiDiscrete action space.

Supports two policy architectures and three masking strategies. Each masking
strategy preserves its README-canonical contract: ``substructure`` and
``r2_available`` are pattern-only and may emit ``invalid_reaction_penalty``
(-1) when RDKit fails to produce a sanitised product despite the pattern
match; ``reaction_valid`` is the strict mode that **guarantees zero -1
rewards in recorded transitions**.

Policy architectures (``policy_arch``):

  - ``hierarchical`` (autoregressive):
        a = (T, R2),   π(a | s) = π_T(T | s) · π_R2(R2 | s, T)
    The R2 head is conditioned on the sampled template via a learned per-T
    embedding, so the policy can learn template-conditional reactant
    preferences. R2 masking is per-(state, T).

  - ``multidiscrete`` (independent):
        a = (T, R2),   π(a | s) = π_T(T | s) · π_R2(R2 | s)
    T and R2 are sampled independently (the SB3 MaskablePPO MultiDiscrete
    parameterisation, but kept under the same trainer for parity). R2 masking
    is per-state (union over valid templates).

Masking strategies (driven by ``config['masking']``, semantics match the
README's `Masking Modes` section):

  - ``substructure`` (pattern-only, -1 allowed):
        * Template axis: R1 first-reactant substructure match only. No R2
          inspection, no ``apply_reaction``. Uni and bi are treated the same
          way — bit-identical to the legacy ``template_substructure_mask``.
        * R2 axis: pattern-match candidate set (``ReactionManager.r2_mask``
          or its union over valid T). Pattern matching does *not* guarantee
          a sanitised product, so any sampled (T, R2) that fails
          ``apply_reaction`` is recorded as a regular transition with
          ``reward = invalid_reaction_penalty``. **No rejection sampling.**

  - ``r2_available`` (pattern + R2 availability, -1 allowed):
        * Template axis: R1 first-reactant match plus, for bi templates,
          at least one R2 in the pool pattern-matches the template's
          second-reactant slot.
        * R2 axis: pattern-match set. As with ``substructure``, sanitisation
          failures surface as ``invalid_reaction_penalty``. **No rejection
          sampling.**

  - ``reaction_valid`` (zero -1, slow):
        * Template axis: R1 match + RDKit produces a sanitised product. For
          uni this is ``apply_reaction(state, T, None)`` succeeds (legacy
          uni semantics, unchanged). For bi this is ∃ R2 such that
          ``apply_reaction(state, T, R2)`` succeeds.
        * R2 axis:
            - hierarchical: per-(state, T) RDKit-validated set from
              :meth:`ReactionManager.bi_r2_valid_mask`. Every sampled
              (T, R2) is guaranteed to produce a sanitised product —
              **zero -1 by mask construction**. No rejection sampling
              needed.
            - multidiscrete: per-state R2 mask = union over valid T of the
              per-(state, T) RDKit-validated set. Joint (T, R2) may still
              be an invalid pair (R2 valid for some *other* T), so
              **rejection sampling at sampling time** enforces zero -1 by
              retrying until a valid joint is found (or, on retry
              exhaustion, the action falls through to STOP).

Architecture mirrors ``genmolrl.algorithms.graphtransppo.train`` to keep the
PPO accounting (GAE, clipped surrogate, value clipping, target_kl early stop,
explained-variance, KL log, etc.) identical and source-of-truth comparable.
Uni-mode PPO continues to flow through ``genmolrl.algorithms.ppo.train``
unchanged — this trainer is opted into via the ``--algorithm ppo_bi`` registry
entry, so existing uni runs are bit-for-bit unaffected.
"""

from __future__ import annotations

import pickle
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED

import wandb

from genmolrl.algorithms.common import init_wandb, run_dir, set_seed
from genmolrl.algorithms.ppo_bi.policy import BiPolicy
from genmolrl.chem.fingerprints import morgan_fp_array
from genmolrl.chem.reaction_manager import BI_TYPE, ReactionManager
from genmolrl.config import resolve_path

STOP_NAME = "Stop"
R2_PAD = -1  # sentinel R2 index for STOP transitions


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _reactant_smiles(data: Any) -> list[str]:
    if isinstance(data, dict):
        return [str(k) for k in data.keys()]
    if isinstance(data, (list, tuple, set)):
        return [str(x) for x in data]
    raise ValueError("Reactant file must contain a dict or sequence of SMILES.")


def _qed(smiles: str, *, round_digits: int | None = None) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    value = float(QED.qed(mol))
    return round(value, round_digits) if round_digits is not None else value


def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var_y = float(np.var(returns))
    if var_y == 0.0:
        return float("nan")
    return float(1.0 - np.var(returns - values) / var_y)


@dataclass
class _Transition:
    smiles: str
    t_action: int
    r2_action: int  # R2_PAD when is_stop is True
    log_pi_old: float
    value: float
    reward: float
    done: bool
    is_stop: bool
    template_mask: torch.Tensor  # bool [num_templates + 1], CPU
    r2_mask: torch.Tensor | None  # bool [num_reactants], CPU, None for STOP


class _StartSampler:
    def __init__(self, train_smiles: list[str], test_smiles: list[str], seed: int):
        if not train_smiles:
            raise ValueError("ppo_bi training requires at least one training molecule.")
        if not test_smiles:
            raise ValueError("ppo_bi evaluation requires at least one test molecule.")
        self.train_smiles = list(train_smiles)
        self.test_smiles = list(test_smiles)
        self.rng = random.Random(seed)

    def sample_train(self) -> str:
        return self.rng.choice(self.train_smiles)

    def eval_starts(self) -> list[str]:
        return list(self.test_smiles)


class BiPPO:
    """PPO trainer for the bi-reaction MultiDiscrete([T+1, R2]) action space.

    Supports both ``hierarchical`` and ``multidiscrete`` policy architectures
    via ``config['ppo_bi']['policy_arch']`` (default: ``hierarchical``). The
    sampling, masking, and log-prob accounting differ between architectures
    but the PPO core (GAE, clipped objective, value clipping, target_kl early
    stop, explained variance, etc.) is shared.
    """

    def __init__(self, config: dict):
        if config.get("reaction_mode", "uni") != "bi":
            raise ValueError(
                "ppo_bi is the bi-reaction trainer and requires reaction_mode: bi. "
                "Use --algorithm ppo for uni mode (its behaviour is unchanged)."
            )
        self.config = config
        training_cfg = config.get("training", {})
        self.seed = int(config.get("seed", training_cfg.get("seed", 0)))
        set_seed(self.seed)

        dataset = config["dataset"]
        self.train_reactants = _load_pickle(resolve_path(dataset["training_file"]))
        self.test_reactants = _load_pickle(resolve_path(dataset["test_file"]))
        self.templates_raw = _load_pickle(resolve_path(dataset["templates_file"]))
        self.train_smiles = _reactant_smiles(self.train_reactants)
        self.test_smiles = _reactant_smiles(self.test_reactants)

        self.reaction_mode = "bi"
        self.masking = config.get("masking", "reaction_valid")
        if self.masking not in {"reaction_valid", "r2_available", "substructure"}:
            raise ValueError(
                f"Unsupported masking for ppo_bi: {self.masking!r}. "
                "Use 'reaction_valid' (zero-failure, slow) or 'substructure' "
                "(pattern-match, fast with rejection backstop)."
            )
        self.reward_name = config.get("reward", "delta_qed")
        if self.reward_name != "delta_qed":
            raise ValueError("ppo_bi currently supports reward: delta_qed only")

        env_cfg = config.get("env", {})
        self.max_episode_len = int(
            config.get("max_episode_len", env_cfg.get("max_episode_len", 5))
        )
        self.use_stop_action = bool(env_cfg.get("use_stop_action", True))
        self.qed_round_digits = env_cfg.get(
            "info_qed_round_digits", env_cfg.get("reward_round_digits")
        )
        self.invalid_reaction_penalty = float(
            env_cfg.get("invalid_reaction_penalty", -1.0)
        )
        self.stop_early_penalty = float(env_cfg.get("stop_early_penalty", 0.0))
        self.stop_penalty_until_step = int(env_cfg.get("stop_penalty_until_step", -1))

        # Build the *training-pool* reaction manager. This is the
        # reaction_manager / reactant_keys / num_reactants that the rollout
        # loop and the PPO update use. The active-pool attributes are
        # initialised to this train pool below; ``evaluate()`` temporarily
        # swaps them to the eval-pool versions when r2_arch='encoder'.
        train_manager_source = (
            self.train_reactants
            if isinstance(self.train_reactants, dict)
            else {s: None for s in self.train_smiles}
        )
        self._train_reaction_manager = ReactionManager(
            self.templates_raw, train_manager_source
        )
        self._train_reaction_manager.templates = (
            self._train_reaction_manager.templates_for_mode("bi")
        )
        self._train_reaction_manager.template_keys = list(
            self._train_reaction_manager.templates.keys()
        )
        self._train_reaction_manager.template_mask_cache.clear()
        self._train_reaction_manager._bi_r2_valid_cache = {}

        self.num_templates = len(self._train_reaction_manager.templates)
        self.stop_index = self.num_templates
        self._train_reactant_keys = list(self._train_reaction_manager.reactants.keys())
        self._train_num_reactants = len(self._train_reactant_keys)

        # Active-pool aliases. The rollout / update / sampling code paths
        # below read these (NOT the underscore-prefixed pool-specific
        # attributes), so pool swapping is a 3-attribute rebind in
        # ``evaluate()``.
        self.reaction_manager = self._train_reaction_manager
        self.reactant_keys = self._train_reactant_keys
        self.num_reactants = self._train_num_reactants

        method_cfg = self._method_cfg(config)
        self.device = torch.device(
            method_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_arch = str(method_cfg.get("policy_arch", "hierarchical")).lower()
        if self.policy_arch not in {"hierarchical", "multidiscrete"}:
            raise ValueError(
                f"policy_arch must be 'hierarchical' or 'multidiscrete', got "
                f"{self.policy_arch!r}"
            )

        # R2 representation. ``lookup`` is the legacy fixed-pool
        # ``nn.Embedding`` (bit-identical to the original BiPolicy). ``encoder``
        # is a Morgan-FP MLP shared between train and eval. ``encoder_graph``
        # (in GraphTransBiPPO) is a Siamese R2 GraphTransformer + projection.
        #
        # Subclasses may extend ``_supported_r2_archs`` to add new values
        # (e.g. ``GraphTransBiPPO`` adds ``'encoder_graph'`` for the Siamese
        # GraphTransformer R2 encoder under Option 3). The validation is kept
        # here in the base so misspelled YAMLs fail loudly at construction
        # time, but the supported set is overridable.
        self.r2_arch = str(method_cfg.get("r2_arch", "lookup")).lower()
        supported_archs = self._supported_r2_archs()
        if self.r2_arch not in supported_archs:
            raise ValueError(
                f"r2_arch must be one of {sorted(supported_archs)}, "
                f"got {self.r2_arch!r}"
            )

        # ``eval_r2_pool`` chooses which R2 pool the policy draws from
        # during evaluate() — either the training pool ("train") or the
        # test pool ("test"). Compatibility matrix:
        #
        #   r2_arch=lookup        + eval_r2_pool=train → OK (gr7aa7z6 baseline).
        #   r2_arch=lookup        + eval_r2_pool=test  → ERROR (no test rows).
        #   r2_arch=encoder       + eval_r2_pool=train → OK (FP encoder on train pool).
        #   r2_arch=encoder       + eval_r2_pool=test  → OK (current encoder default).
        #   r2_arch=encoder_graph + eval_r2_pool=train → OK (graph encoder on train pool).
        #   r2_arch=encoder_graph + eval_r2_pool=test  → OK (current encoder_graph default).
        #
        # If unset, the default preserves prior implicit behaviour:
        # lookup → train, every other arch → test. Internally we map the
        # YAML "train"/"test" to the legacy role names "train"/"eval" so
        # ``_swap_active_pool`` and ``_compute_active_r2_keys`` keep their
        # existing call sites — only the binding changes.
        _default_eval_pool = "train" if self.r2_arch == "lookup" else "test"
        _eval_pool_yaml = str(
            method_cfg.get("eval_r2_pool", _default_eval_pool)
        ).lower()
        _yaml_to_internal = {"train": "train", "test": "eval", "eval": "eval"}
        if _eval_pool_yaml not in _yaml_to_internal:
            raise ValueError(
                "eval_r2_pool must be 'train' or 'test', got "
                f"{_eval_pool_yaml!r}"
            )
        self._eval_pool_role = _yaml_to_internal[_eval_pool_yaml]
        # Public, YAML-style spelling for logging / experiment tags.
        self.eval_r2_pool = "train" if self._eval_pool_role == "train" else "test"

        # lookup + test is structurally impossible — the embedding table is
        # sized to the training pool and has no rows for test reactants.
        # Fail loudly at construction time rather than silently index into
        # the wrong row at eval.
        if self.r2_arch == "lookup" and self._eval_pool_role == "eval":
            raise ValueError(
                "r2_arch='lookup' is incompatible with eval_r2_pool='test': "
                "the nn.Embedding(num_reactants, r2_embed_dim) table is sized "
                "to the TRAINING pool — there are no rows for test reactants. "
                "Use r2_arch='encoder', 'encoder_graph', or "
                "'encoder_graph_shared' to evaluate on test R2s, or set "
                "eval_r2_pool='train' to keep the legacy (gr7aa7z6) "
                "behaviour where eval draws R2 from the train pool."
            )

        # Derive the R2-axis mask source from the masking mode (the README
        # contract). reaction_valid → RDKit-validated set (zero -1 guarantee);
        # substructure / r2_available → pattern-match set (-1 surfaces when
        # RDKit fails despite the pattern match). Advanced users can override
        # via the explicit `ppo_bi.r2_mask_kind` key (rarely needed).
        masking_to_r2 = {
            "reaction_valid": "true_valid",
            "substructure": "pattern",
            "r2_available": "pattern",
        }
        self.r2_mask_kind = str(
            method_cfg.get("r2_mask_kind", masking_to_r2.get(self.masking, "pattern"))
        ).lower()
        if self.r2_mask_kind not in {"pattern", "true_valid"}:
            raise ValueError(
                f"r2_mask_kind must be 'pattern' or 'true_valid', got {self.r2_mask_kind!r}"
            )

        # reaction_valid is the only masking mode that promises zero -1
        # rewards. For hierarchical the per-(state, T) RDKit-validated mask
        # makes the promise hold by construction (no rejection needed). For
        # multidiscrete the joint (T, R2) is not guaranteed valid even with
        # per-axis true-valid masks, so we need rejection sampling to enforce
        # the promise. ``substructure`` / ``r2_available`` are pattern-only —
        # they intentionally allow apply_reaction failures to surface as -1.
        self._enforce_zero_invalid = self.masking == "reaction_valid"
        self._needs_joint_rejection = (
            self._enforce_zero_invalid and self.policy_arch == "multidiscrete"
        )

        self.policy = self._build_policy(method_cfg)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(method_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(method_cfg.get("weight_decay", 0.0)),
            eps=float(method_cfg.get("adam_eps", 1e-5)),
        )

        # Build the eval-pool reaction manager. Source depends on the
        # ``eval_r2_pool`` knob (NOT on ``r2_arch``):
        #
        #   - ``_eval_pool_role == "train"``: eval shares the train pool.
        #     Aliases the eval-pool attributes at the train pool objects;
        #     ``_swap_active_pool('eval')`` is then a structural no-op. This
        #     is the gr7aa7z6 baseline under lookup, and also a legal
        #     "evaluate on the same pool you trained on" mode under encoder
        #     / encoder_graph for apples-to-apples comparison against
        #     lookup.
        #
        #   - ``_eval_pool_role == "eval"``: build a separate
        #     ``ReactionManager`` from ``data/Bi/reactants_test.pkl``. Only
        #     legal when ``r2_arch != 'lookup'`` (the lookup-table guard
        #     above already rules this combination out).
        if self._eval_pool_role == "train":
            self._eval_reaction_manager = self._train_reaction_manager
            self._eval_reactant_keys = self._train_reactant_keys
            self._eval_num_reactants = self._train_num_reactants
        else:
            test_manager_source = (
                self.test_reactants
                if isinstance(self.test_reactants, dict)
                else {s: None for s in self.test_smiles}
            )
            self._eval_reaction_manager = ReactionManager(
                self.templates_raw, test_manager_source
            )
            self._eval_reaction_manager.templates = (
                self._eval_reaction_manager.templates_for_mode("bi")
            )
            self._eval_reaction_manager.template_keys = list(
                self._eval_reaction_manager.templates.keys()
            )
            self._eval_reaction_manager.template_mask_cache.clear()
            self._eval_reaction_manager._bi_r2_valid_cache = {}
            self._eval_reactant_keys = list(
                self._eval_reaction_manager.reactants.keys()
            )
            self._eval_num_reactants = len(self._eval_reactant_keys)

        # Pre-compute Morgan fingerprints for both pools as torch tensors,
        # used by ``r2_arch='encoder'``. These feed the R2 encoder MLP at
        # every sampling / update call; building them once at init avoids
        # per-step morgan_fp overhead.
        #
        # When ``_eval_pool_role == "train"`` the two pools are identical,
        # so the eval-side cache is just a view onto the train-side cache.
        # When ``_eval_pool_role == "eval"`` we build a separate test-pool
        # FP tensor. Subclasses (e.g. GraphTransBiPPO) extend this with
        # their own pool-specific caches via ``_init_extra_pool_data``.
        if self.r2_arch == "encoder":
            train_fp_np = np.stack(
                [morgan_fp_array(s) for s in self._train_reactant_keys], axis=0
            )
            self._train_r2_fps = torch.from_numpy(train_fp_np).float().to(self.device)
            if self._eval_pool_role == "train":
                self._eval_r2_fps = self._train_r2_fps
            else:
                test_fp_np = np.stack(
                    [morgan_fp_array(s) for s in self._eval_reactant_keys], axis=0
                )
                self._eval_r2_fps = torch.from_numpy(test_fp_np).float().to(self.device)
        else:
            self._train_r2_fps = None
            self._eval_r2_fps = None

        # Subclass hook for arch-specific pool caches (e.g. the R2 graph
        # Batches that ``GraphTransBiPPO`` needs for ``r2_arch='encoder_graph'``).
        # Default is a no-op so the base trainer's behaviour is unchanged.
        self._init_extra_pool_data()

        # Cached r2_keys for the active scope. Populated by
        # ``_compute_active_r2_keys`` at the start of each rollout / eval
        # sweep (no_grad) and recomputed per minibatch inside the PPO update
        # (with_grad), so MLP weight updates flow through into r2_keys.
        self._active_r2_keys: torch.Tensor | None = None

        # PPO knobs (defaults match graphtransppo/MaskablePPO).
        self.gamma = float(method_cfg.get("gamma", 0.99))
        self.gae_lambda = float(method_cfg.get("gae_lambda", 0.95))
        self.clip_range = float(method_cfg.get("clip_range", 0.2))
        clip_vf = method_cfg.get("clip_range_vf", None)
        self.clip_range_vf = float(clip_vf) if clip_vf is not None else None
        self.vf_coef = float(method_cfg.get("vf_coef", 0.5))
        self.ent_coef = float(method_cfg.get("ent_coef", 0.0))
        self.max_grad_norm = float(method_cfg.get("max_grad_norm", 0.5))
        self.target_kl = method_cfg.get("target_kl", None)
        if self.target_kl is not None:
            self.target_kl = float(self.target_kl)
        self.normalize_advantage = bool(method_cfg.get("normalize_advantage", True))

        self.n_steps = int(method_cfg.get("n_steps", training_cfg.get("n_steps", 2048)))
        self.minibatch = int(method_cfg.get("batch_size", training_cfg.get("batch_size", 64)))
        self.n_epochs = int(method_cfg.get("n_epochs", 10))

        # Joint-rejection cap used ONLY when self._needs_joint_rejection (i.e.
        # multidiscrete + reaction_valid). Each retry runs one RDKit
        # ``apply_reaction`` call; with reaction_valid masks the joint
        # rejection rate is moderate (most (T, R2) pairs in the union mask
        # are valid). substructure / r2_available do NOT use this; they take
        # the first sample and let -1 surface naturally.
        self.r2_resample_retries = int(method_cfg.get("r2_resample_retries", 16))

        self.sampler = _StartSampler(self.train_smiles, self.test_smiles, self.seed)

        self._current_smiles: str | None = None
        self._current_react_steps: int = 0

        self._ep_reward_window: deque[float] = deque(maxlen=100)
        self._ep_length_window: deque[int] = deque(maxlen=100)
        self._total_episodes: int = 0
        self._cumulative_reward: float = 0.0
        # invalid_reaction_count is the cumulative number of -1 transitions
        # recorded by the rollout. Must stay 0 under masking=reaction_valid
        # (enforced by `_sample_action_*` + the assert in the rollout). Under
        # masking=substructure / r2_available it grows naturally — the
        # README contract allows pattern-match leaks to surface as -1.
        self._invalid_reaction_count: int = 0
        self._stop_event_count: int = 0
        self._rejection_total: int = 0
        self._sample_calls: int = 0

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    # Subclasses (e.g. GraphTransBiPPO) override these three small methods to
    # swap the encoder while reusing the entire PPO core (rollout loop,
    # masking, hierarchical/multidiscrete sampling, rejection logic, GAE,
    # clipped surrogate, value clipping, target_kl early stop, etc.). The
    # default implementations below preserve the original BiPolicy +
    # Morgan-fingerprint trainer behaviour bit-for-bit.

    def _method_cfg(self, config: dict) -> dict:
        """Return the method-specific config block (overridable by subclasses).

        Default reads ``config['ppo_bi']`` and falls back to ``config['ppo']``
        for shared PPO knobs. Subclasses (e.g. GraphTransBiPPO) point this at
        their own config block but keep the PPO defaults compatible.
        """
        return config.get("ppo_bi", config.get("ppo", {}))

    def _supported_r2_archs(self) -> set[str]:
        """Return the set of legal ``r2_arch`` values for this trainer.

        The base BiPPO trainer supports the Morgan-FP-only set
        ``{'lookup', 'encoder'}``. Subclasses can extend it to include
        encoder variants their policy understands (e.g. GraphTransBiPPO
        adds ``'encoder_graph'`` for the Siamese GraphTransformer R2
        encoder under Option 3). Overriding this is preferred over
        re-implementing the YAML validation in each subclass.
        """
        return {"lookup", "encoder"}

    def _init_extra_pool_data(self) -> None:
        """Hook for arch-specific pool-data caches built at init time.

        Default is a no-op (the base trainer only needs the Morgan-FP
        tensors built unconditionally above). Subclasses override this
        to allocate their own caches; for instance
        ``GraphTransBiPPO._init_extra_pool_data`` builds
        ``_train_r2_graphs`` and ``_eval_r2_graphs`` (torch_geometric
        ``Batch`` objects) when ``r2_arch='encoder_graph'``.
        """
        return None

    def _build_policy(self, method_cfg: dict) -> torch.nn.Module:
        """Construct the policy module (overridable by subclasses).

        Default builds ``BiPolicy`` (Morgan-FP MLP trunk + BiPolicy heads) and
        moves it to ``self.device``. Subclasses can return any module that
        exposes ``forward_trunk``, ``template_logits``, ``value``, and
        ``r2_logits`` with the same signatures so the rollout loop and PPO
        update are encoder-agnostic.
        """
        return BiPolicy(
            num_templates=self.num_templates,
            num_reactants=self.num_reactants,
            conditional_r2=(self.policy_arch == "hierarchical"),
            obs_dim=1024,
            trunk_hidden=int(method_cfg.get("trunk_hidden", 256)),
            template_embed_dim=int(method_cfg.get("template_embed_dim", 64)),
            r2_embed_dim=int(method_cfg.get("r2_embed_dim", 64)),
            r2_arch=self.r2_arch,
            r2_encoder_hidden=method_cfg.get("r2_encoder_hidden"),
            # Option 1: residual MLP for the R2 encoder. Defaults preserve
            # the legacy plain 2-layer MLP behaviour so older YAMLs reproduce
            # bit-for-bit; new ppo_bi YAMLs flip ``r2_encoder_residual: true``
            # to opt into the deeper variant.
            r2_encoder_n_layers=int(method_cfg.get("r2_encoder_n_layers", 2)),
            r2_encoder_residual=bool(method_cfg.get("r2_encoder_residual", False)),
            r2_encoder_n_res_blocks=int(method_cfg.get("r2_encoder_n_res_blocks", 2)),
        ).to(self.device)

    def _encode_smiles(self, smiles_list: list[str]) -> torch.Tensor:
        """Map a list of SMILES to trunk features ``z(R1) ∈ R^{B x trunk_dim}``.

        Default builds a Morgan-fingerprint batch and runs ``policy.
        forward_trunk(fps)``. Subclasses override this to plug in any other
        encoder (e.g. ``GraphTransBiPPO`` runs the GraphTransformer over the
        molecular graphs) without changing the rollout loop or PPO update.
        """
        fps = np.stack([morgan_fp_array(s) for s in smiles_list], axis=0)
        return self.policy.forward_trunk(torch.from_numpy(fps).to(self.device))

    # ------------------------------------------------------------------
    # Active-pool helpers (r2_arch='encoder' swaps train ↔ test at eval)
    # ------------------------------------------------------------------

    def _swap_active_pool(self, pool: str) -> None:
        """Point ``self.reaction_manager`` / ``reactant_keys`` / ``num_reactants``
        at the named pool.

        Only ``r2_arch='encoder'`` mode has distinct train and eval pools; in
        ``lookup`` mode both attributes alias the train pool and this call is
        a structural no-op. Callers MUST restore the previous pool by calling
        ``_swap_active_pool('train')`` after the eval section (see
        :meth:`evaluate`); we keep the swap explicit (rather than a context
        manager) so the sampling functions don't have to thread a pool
        argument through the rollout loop's hot path.
        """
        if pool == "train":
            self.reaction_manager = self._train_reaction_manager
            self.reactant_keys = self._train_reactant_keys
            self.num_reactants = self._train_num_reactants
        elif pool == "eval":
            self.reaction_manager = self._eval_reaction_manager
            self.reactant_keys = self._eval_reactant_keys
            self.num_reactants = self._eval_num_reactants
        else:
            raise ValueError(f"pool must be 'train' or 'eval', got {pool!r}")

    def _r2_pool_data_for(self, pool: str):
        """Return the input passed to ``policy.encode_r2_pool`` for ``pool``.

        Default returns the pre-computed Morgan-FP tensor for the named
        pool — this is what ``r2_arch='encoder'`` consumes. Subclasses
        override this to return arch-specific data; e.g.
        ``GraphTransBiPPO._r2_pool_data_for`` returns a torch_geometric
        ``Batch`` when ``r2_arch='encoder_graph'`` so the policy's
        Siamese R2 GraphTransformer can encode the pool directly from
        graphs. Called by :meth:`_compute_active_r2_keys` only when
        ``r2_arch != 'lookup'``.
        """
        if pool == "train":
            return self._train_r2_fps
        if pool == "eval":
            return self._eval_r2_fps
        raise ValueError(f"pool must be 'train' or 'eval', got {pool!r}")

    def _compute_active_r2_keys(self, *, pool: str, with_grad: bool) -> torch.Tensor:
        """Return ``r2_keys`` for the named pool, honouring the grad context.

        In ``r2_arch='lookup'`` mode this returns ``self.policy.r2_embed.weight``
        regardless of pool (the embedding is fixed to the train pool by design).
        Otherwise the policy's R2 encoder is applied to the pool input data
        returned by :meth:`_r2_pool_data_for` — Morgan-FP tensor under
        ``r2_arch='encoder'``, torch_geometric ``Batch`` under
        ``r2_arch='encoder_graph'`` in subclasses. ``with_grad=False`` is used
        for rollouts and evaluation (one forward pass per sweep, no autograd);
        ``with_grad=True`` is used inside the PPO update so gradients flow
        into the encoder each minibatch.
        """
        if self.r2_arch == "lookup":
            return self.policy.r2_embed.weight
        pool_data = self._r2_pool_data_for(pool)
        if pool_data is None:
            raise RuntimeError(
                f"r2_arch={self.r2_arch!r} but {pool} pool data is not initialised."
            )
        if with_grad:
            return self.policy.encode_r2_pool(pool_data)
        with torch.no_grad():
            return self.policy.encode_r2_pool(pool_data)

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def _template_mask(self, smiles: str, *, force_stop: bool = False) -> torch.Tensor:
        mask = torch.zeros(self.num_templates + 1, dtype=torch.bool, device=self.device)
        if not force_stop:
            template_mask = (
                self.reaction_manager.get_mask(smiles, kind=self.masking).to(self.device)
                > 0.5
            )
            mask[: self.num_templates] = template_mask
        if self.use_stop_action:
            mask[self.stop_index] = True
        return mask

    def _r2_mask_per_template(self, smiles: str, t_idx: int) -> torch.Tensor:
        """Per-(state, T) R2 mask used by the hierarchical architecture.

        ``r2_mask_kind='true_valid'`` returns the RDKit-validated set
        (:meth:`ReactionManager.bi_r2_valid_mask`) so sampling cannot pick a
        ``(T, R2)`` that would fail ``apply_reaction``. ``r2_mask_kind=
        'pattern'`` returns the pattern-match set; the rejection loop in
        ``_sample_action_hierarchical`` removes the rare sanitisation
        failures.
        """
        if self.r2_mask_kind == "true_valid":
            mask_np = self.reaction_manager.bi_r2_valid_mask(smiles, t_idx)
        else:
            mask_np = self.reaction_manager.r2_mask(t_idx)
        return torch.from_numpy(mask_np.astype(np.bool_)).to(self.device)

    def _r2_mask_per_state(
        self, smiles: str, valid_t_indices: list[int]
    ) -> torch.Tensor:
        """Per-state R2 mask used by the multidiscrete architecture.

        It is the union of the per-template R2 masks over all valid (non-STOP)
        templates. This is the largest mask that is consistent with the
        independent-sampling assumption π(R2 | s) (no T conditioning). It does
        NOT by itself guarantee a valid joint (T, R2) — that's the job of
        rejection sampling in ``_sample_action_multidiscrete``.
        """
        union = np.zeros(self.num_reactants, dtype=np.bool_)
        for t in valid_t_indices:
            if t < 0 or t >= self.num_templates:
                continue
            if self.r2_mask_kind == "true_valid":
                m = self.reaction_manager.bi_r2_valid_mask(smiles, t)
            else:
                m = self.reaction_manager.r2_mask(t)
            union |= m.astype(np.bool_)
        return torch.from_numpy(union).to(self.device)

    # ------------------------------------------------------------------
    # Single-state forward + sampling
    # ------------------------------------------------------------------

    def _fp(self, smiles: str) -> torch.Tensor:
        """Backward-compatible single-SMILES Morgan-fingerprint helper.

        Kept for callers outside this module that still expect the
        fingerprint-tensor API. The trainer itself now goes through
        ``_encode_smiles`` so a graph-aware subclass needs no extra hooks.
        """
        return torch.from_numpy(morgan_fp_array(smiles)).to(self.device)

    def _sample_action(
        self, smiles: str, *, force_stop: bool = False, deterministic: bool = False
    ) -> tuple[int, int, float, float, torch.Tensor, torch.Tensor | None, str | None]:
        """Dispatch to the per-architecture sampler."""
        self._sample_calls += 1
        if self.policy_arch == "hierarchical":
            return self._sample_action_hierarchical(
                smiles, force_stop=force_stop, deterministic=deterministic
            )
        return self._sample_action_multidiscrete(
            smiles, force_stop=force_stop, deterministic=deterministic
        )

    # ------------------------------------------------------------------
    # Hierarchical sampling: T then R2|T
    # ------------------------------------------------------------------

    def _sample_action_hierarchical(
        self, smiles: str, *, force_stop: bool, deterministic: bool
    ) -> tuple[int, int, float, float, torch.Tensor, torch.Tensor | None, str | None]:
        """Sample (T, R2) autoregressively.

        With ``masking=reaction_valid`` the per-(state, T) R2 mask is
        RDKit-validated, so ``apply_reaction`` is *guaranteed* to succeed —
        we still run it once to get the product SMILES. With ``substructure``
        / ``r2_available`` the R2 mask is pattern-only; ``apply_reaction``
        may fail and the trainer returns ``product=None``, which the rollout
        loop records as an ``invalid_reaction_penalty`` transition.
        """
        trunk = self._encode_smiles([smiles])
        tmpl_logits = self.policy.template_logits(trunk)
        value = float(self.policy.value(trunk).item())

        tmpl_mask = self._template_mask(smiles, force_stop=force_stop)
        if not bool(tmpl_mask.any()):
            return self._stop_return(value, tmpl_mask)

        masked_tmpl_logits = tmpl_logits[0].masked_fill(~tmpl_mask, -1e9)
        tmpl_dist = torch.distributions.Categorical(logits=masked_tmpl_logits)
        t_t = (
            torch.argmax(masked_tmpl_logits)
            if deterministic
            else tmpl_dist.sample()
        )
        t_idx = int(t_t.item())
        log_pi_t = float(tmpl_dist.log_prob(t_t).item())

        if t_idx == self.stop_index:
            return self._stop_return(value, tmpl_mask, log_pi_t=log_pi_t, t_idx=t_idx)

        r2_mask = self._r2_mask_per_template(smiles, t_idx)
        if not bool(r2_mask.any()):
            return self._stop_return(value, tmpl_mask, log_pi_t=log_pi_t)

        r2_logits_all = self.policy.r2_logits(
            trunk,
            torch.tensor([t_idx], device=self.device, dtype=torch.long),
            r2_keys=self._active_r2_keys,
        )[0]
        masked_r2_logits = r2_logits_all.masked_fill(~r2_mask, -1e9)
        r2_dist = torch.distributions.Categorical(logits=masked_r2_logits)
        r2_t = (
            torch.argmax(masked_r2_logits)
            if deterministic
            else r2_dist.sample()
        )
        r2_idx = int(r2_t.item())
        log_pi_r2 = float(r2_dist.log_prob(r2_t).item())

        template = self.reaction_manager.templates[t_idx]
        product = self.reaction_manager.apply_reaction(
            smiles, template, self.reactant_keys[r2_idx]
        )
        if product is None and self._enforce_zero_invalid:
            # reaction_valid + hierarchical: the mask is exact, so this is a
            # logic bug or an RDKit pathology. Bump a counter for visibility
            # and fall back to STOP rather than emit a -1 the contract forbids.
            self._invalid_reaction_count += 1
            return self._stop_return(value, tmpl_mask, log_pi_t=log_pi_t)

        # product may be None here only when masking ∈ {substructure,
        # r2_available}; the rollout loop will record an invalid-penalty
        # transition. We still return the action and log_pi so the policy
        # learns from the failure signal.
        return (
            t_idx,
            r2_idx,
            log_pi_t + log_pi_r2,
            value,
            tmpl_mask.detach().to("cpu"),
            r2_mask.detach().to("cpu"),
            product,
        )

    # ------------------------------------------------------------------
    # Multidiscrete sampling: independent T and R2 with joint rejection
    # ------------------------------------------------------------------

    def _sample_action_multidiscrete(
        self, smiles: str, *, force_stop: bool, deterministic: bool
    ) -> tuple[int, int, float, float, torch.Tensor, torch.Tensor | None, str | None]:
        """Sample (T, R2) independently from the per-axis masked distributions.

        With ``masking=reaction_valid`` the per-state R2 mask is the union of
        the per-(state, T) RDKit-validated sets. The independent-sampling
        assumption means the joint ``(T_sampled, R2_sampled)`` might pair an
        R2 with a template it doesn't actually work for; ``_needs_joint_
        rejection`` is True in this case and the loop below retries the joint
        until a valid pair is found (or budget exhausted → fall through to
        STOP). With ``substructure`` / ``r2_available`` no rejection happens:
        we take the first sample and let ``invalid_reaction_penalty`` surface
        if ``apply_reaction`` fails (this is the README contract for those
        masking modes).
        """
        trunk = self._encode_smiles([smiles])
        tmpl_logits = self.policy.template_logits(trunk)
        value = float(self.policy.value(trunk).item())

        tmpl_mask = self._template_mask(smiles, force_stop=force_stop)
        if not bool(tmpl_mask.any()):
            return self._stop_return(value, tmpl_mask)

        valid_t = [
            int(i)
            for i in torch.where(tmpl_mask[: self.num_templates])[0].tolist()
        ]
        masked_tmpl_logits = tmpl_logits[0].masked_fill(~tmpl_mask, -1e9)
        tmpl_dist = torch.distributions.Categorical(logits=masked_tmpl_logits)

        if not valid_t:
            # Only STOP is available.
            t_t = torch.tensor(self.stop_index, device=self.device, dtype=torch.long)
            return (
                self.stop_index,
                R2_PAD,
                float(tmpl_dist.log_prob(t_t).item()),
                value,
                tmpl_mask.detach().to("cpu"),
                None,
                None,
            )

        state_r2_mask = self._r2_mask_per_state(smiles, valid_t)
        if not bool(state_r2_mask.any()):
            return self._stop_return(value, tmpl_mask)
        r2_logits = self.policy.r2_logits(trunk, None, r2_keys=self._active_r2_keys)[0]
        masked_r2_logits = r2_logits.masked_fill(~state_r2_mask, -1e9)
        r2_dist = torch.distributions.Categorical(logits=masked_r2_logits)

        # One-shot fast path for substructure / r2_available: take a single
        # sample, run apply_reaction, return whatever it produces (product
        # may be None → rollout records the -1 transition).
        if not self._needs_joint_rejection:
            t_t = (
                torch.argmax(masked_tmpl_logits)
                if deterministic
                else tmpl_dist.sample()
            )
            t_idx = int(t_t.item())
            log_pi_t = float(tmpl_dist.log_prob(t_t).item())
            if t_idx == self.stop_index:
                return (
                    self.stop_index,
                    R2_PAD,
                    log_pi_t,
                    value,
                    tmpl_mask.detach().to("cpu"),
                    None,
                    None,
                )

            r2_t = (
                torch.argmax(masked_r2_logits)
                if deterministic
                else r2_dist.sample()
            )
            r2_idx = int(r2_t.item())
            log_pi_r2 = float(r2_dist.log_prob(r2_t).item())
            template = self.reaction_manager.templates[t_idx]
            product = self.reaction_manager.apply_reaction(
                smiles, template, self.reactant_keys[r2_idx]
            )
            return (
                t_idx,
                r2_idx,
                log_pi_t + log_pi_r2,
                value,
                tmpl_mask.detach().to("cpu"),
                state_r2_mask.detach().to("cpu"),
                product,
            )

        # reaction_valid path: rejection-sample the joint until valid or budget
        # exhausted. The recorded log_pi is taken against the original per-axis
        # distributions so the PPO importance ratio still corresponds to the
        # policy's parameterisation (the implicit truncation by the rejection
        # set has bounded mass and is treated as a small ignorable bias).
        tried_pairs: set[tuple[int, int]] = set()
        retries = 0
        for _ in range(max(1, self.r2_resample_retries) + 1):
            t_t = (
                torch.argmax(masked_tmpl_logits)
                if deterministic
                else tmpl_dist.sample()
            )
            t_idx = int(t_t.item())
            log_pi_t = float(tmpl_dist.log_prob(t_t).item())
            if t_idx == self.stop_index:
                return (
                    self.stop_index,
                    R2_PAD,
                    log_pi_t,
                    value,
                    tmpl_mask.detach().to("cpu"),
                    None,
                    None,
                )

            r2_t = (
                torch.argmax(masked_r2_logits)
                if deterministic
                else r2_dist.sample()
            )
            r2_idx = int(r2_t.item())
            log_pi_r2 = float(r2_dist.log_prob(r2_t).item())

            pair = (t_idx, r2_idx)
            if pair in tried_pairs:
                if deterministic:
                    break
                retries += 1
                continue
            tried_pairs.add(pair)

            template = self.reaction_manager.templates[t_idx]
            product = self.reaction_manager.apply_reaction(
                smiles, template, self.reactant_keys[r2_idx]
            )
            if product is not None:
                self._rejection_total += retries
                return (
                    t_idx,
                    r2_idx,
                    log_pi_t + log_pi_r2,
                    value,
                    tmpl_mask.detach().to("cpu"),
                    state_r2_mask.detach().to("cpu"),
                    product,
                )
            retries += 1

        self._rejection_total += retries
        # Could not find a valid joint within budget → fall through to STOP
        # rather than violate the reaction_valid contract.
        return self._stop_return(value, tmpl_mask)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stop_return(
        self,
        value: float,
        tmpl_mask: torch.Tensor,
        *,
        log_pi_t: float = 0.0,
        t_idx: int | None = None,
    ) -> tuple[int, int, float, float, torch.Tensor, torch.Tensor | None, str | None]:
        """Build the canonical 'fall back to STOP' return tuple."""
        return (
            t_idx if t_idx is not None else self.stop_index,
            R2_PAD,
            log_pi_t,
            value,
            tmpl_mask.detach().to("cpu"),
            None,
            None,
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        n_steps: int,
        *,
        base_step: int = 0,
        log_episodes: bool = True,
    ) -> tuple[list[_Transition], float]:
        self.policy.eval()
        transitions: list[_Transition] = []
        if self._current_smiles is None:
            self._current_smiles = self.sampler.sample_train()
            self._current_react_steps = 0

        ep_reward = 0.0
        ep_length = 0
        steps_taken = 0

        with torch.no_grad():
            # Cache r2_keys for the entire rollout. Policy weights are frozen
            # during rollout (PPO updates happen after), so r2_keys is constant
            # and computing it once amortises the encoder forward over n_steps.
            # In lookup mode this is just a view onto ``r2_embed.weight``.
            self._active_r2_keys = self._compute_active_r2_keys(
                pool="train", with_grad=False
            )
            while steps_taken < n_steps:
                current = self._current_smiles
                react_steps = self._current_react_steps

                at_max = react_steps >= self.max_episode_len
                if at_max and not self.use_stop_action:
                    if transitions:
                        transitions[-1].done = True
                    self._end_episode(
                        ep_reward, ep_length, base_step + steps_taken, log_episodes
                    )
                    ep_reward, ep_length = 0.0, 0
                    continue

                (
                    t_idx,
                    r2_idx,
                    log_pi,
                    value,
                    tmpl_mask,
                    r2_mask,
                    product,
                ) = self._sample_action(current, force_stop=at_max)

                if t_idx < 0:
                    if transitions:
                        transitions[-1].done = True
                    self._end_episode(
                        ep_reward, ep_length, base_step + steps_taken, log_episodes
                    )
                    ep_reward, ep_length = 0.0, 0
                    continue

                done = False
                is_stop = t_idx == self.stop_index
                reward = 0.0
                if is_stop:
                    self._stop_event_count += 1
                    done = True
                    if (
                        self.stop_penalty_until_step >= 0
                        and react_steps < self.stop_penalty_until_step
                    ):
                        reward = self.stop_early_penalty
                elif product is None:
                    # apply_reaction failed despite the mask passing the
                    # candidate. With masking ∈ {substructure, r2_available}
                    # this is the README contract (pattern-only masks are
                    # allowed to leak): record the transition with
                    # invalid_reaction_penalty so the policy learns from the
                    # failure. With masking=reaction_valid this branch is
                    # unreachable because `_sample_action_*` already routes
                    # any pathological non-product case through STOP at the
                    # source. The assert below makes that contract explicit.
                    assert not self._enforce_zero_invalid, (
                        "reaction_valid produced an invalid action — this "
                        "should be unreachable; the per-arch sampler must "
                        "fall through to STOP rather than route here."
                    )
                    self._invalid_reaction_count += 1
                    reward = self.invalid_reaction_penalty
                    done = True
                else:
                    prev_qed = _qed(current, round_digits=self.qed_round_digits)
                    next_qed = _qed(product, round_digits=self.qed_round_digits)
                    reward = float(next_qed - prev_qed)
                    self._current_smiles = product
                    self._current_react_steps = react_steps + 1
                    if self._current_react_steps >= self.max_episode_len:
                        done = True

                transitions.append(
                    _Transition(
                        smiles=current,
                        t_action=t_idx,
                        r2_action=r2_idx if not is_stop else R2_PAD,
                        log_pi_old=log_pi,
                        value=value,
                        reward=reward,
                        done=done,
                        is_stop=is_stop,
                        template_mask=tmpl_mask,
                        r2_mask=r2_mask,
                    )
                )
                steps_taken += 1
                ep_reward += reward
                ep_length += 1

                if done:
                    self._end_episode(
                        ep_reward, ep_length, base_step + steps_taken, log_episodes
                    )
                    ep_reward, ep_length = 0.0, 0

        last_value = 0.0
        if transitions and not transitions[-1].done:
            with torch.no_grad():
                trunk = self._encode_smiles([self._current_smiles])
                last_value = float(self.policy.value(trunk).item())
        # Drop the cached keys so a stale tensor doesn't accidentally survive
        # into the next phase (PPO update recomputes per-minibatch, eval
        # recomputes once at the start of evaluate()).
        self._active_r2_keys = None
        return transitions, last_value

    def _end_episode(
        self,
        ep_reward: float,
        ep_length: int,
        step: int,
        log: bool,
    ) -> None:
        if ep_length > 0:
            self._ep_reward_window.append(float(ep_reward))
            self._ep_length_window.append(int(ep_length))
            self._total_episodes += 1
            self._cumulative_reward += float(ep_reward)
            if log and wandb.run is not None:
                wandb.log(
                    {
                        "train/global_step": int(step),
                        "train/episode_reward": float(ep_reward),
                        "train/episode_length": int(ep_length),
                        "train/mean_reward": float(np.mean(self._ep_reward_window)),
                        "train/mean_ep_length": float(np.mean(self._ep_length_window)),
                        "train/total_episodes": float(self._total_episodes),
                        "train/invalid_reaction_count": float(self._invalid_reaction_count),
                        "train/stop_event_count": float(self._stop_event_count),
                        "train/rejection_total": float(self._rejection_total),
                        "cumulative_reward": float(self._cumulative_reward),
                    },
                    step=int(step),
                )
        self._current_smiles = self.sampler.sample_train()
        self._current_react_steps = 0

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rollout: list[_Transition],
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(rollout)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            non_terminal = 0.0 if rollout[t].done else 1.0
            next_value = (
                last_value if t == n - 1 else (0.0 if rollout[t].done else rollout[t + 1].value)
            )
            delta = rollout[t].reward + self.gamma * next_value * non_terminal - rollout[t].value
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        values = np.array([tr.value for tr in rollout], dtype=np.float32)
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _r2_keys_for_update(self) -> torch.Tensor:
        """Hook: return ``r2_keys`` for the current PPO minibatch.

        Default recomputes from scratch *with grad* each call — fine for
        cheap encoders (``lookup`` is a free view onto ``r2_embed.weight``;
        ``encoder`` is a single MLP forward over ~116k FPs, milliseconds
        on GPU). Subclasses with expensive pool-encoding paths can
        override this to amortise: e.g. ``GraphTransBiPPO`` under
        ``r2_arch='encoder_graph'`` refreshes the Siamese R2
        GraphTransformer's keys only every ``r2_keys_refresh_minibatches``
        and reuses a detached cache in between. See
        :meth:`_begin_update_cycle` for the per-update reset hook.
        """
        return self._compute_active_r2_keys(pool="train", with_grad=True)

    def _begin_update_cycle(self) -> None:
        """Hook called once at the top of every :meth:`ppo_update`.

        Default is a no-op. Subclasses use this to reset any per-update
        state — e.g. ``GraphTransBiPPO`` invalidates its
        ``_cached_r2_keys`` here so the next ``_r2_keys_for_update`` call
        triggers a fresh, gradient-attached pool encoding.
        """
        return None

    def _evaluate_minibatch(
        self,
        smiles_batch: list[str],
        t_actions: torch.Tensor,
        r2_actions: torch.Tensor,
        is_stop: torch.Tensor,
        tmpl_masks: torch.Tensor,
        r2_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_pi_new, entropy_per_sample, values)`` for the minibatch.

        Uses the same architecture (conditional vs unconditional R2 head) as
        the trainer was configured with. The masks recorded at rollout time
        are reused verbatim so the update sees the same distribution that the
        old log-probs were taken under.
        """
        trunk = self._encode_smiles(smiles_batch)
        tmpl_logits = self.policy.template_logits(trunk)
        values = self.policy.value(trunk)

        tmpl_logits = tmpl_logits.masked_fill(~tmpl_masks.to(self.device), -1e9)
        tmpl_dist = torch.distributions.Categorical(logits=tmpl_logits)
        log_pi_t = tmpl_dist.log_prob(t_actions)

        # R2 component: zero contribution for STOP rows, otherwise log π(R2|...).
        # The PPO update always runs on transitions collected from the train
        # pool. ``_r2_keys_for_update`` is the hook subclasses use to amortise
        # expensive encoders; the base trainer just delegates straight to
        # ``_compute_active_r2_keys(pool='train', with_grad=True)``.
        r2_keys = self._r2_keys_for_update()
        if self.policy_arch == "hierarchical":
            safe_t = torch.where(is_stop, torch.zeros_like(t_actions), t_actions)
            r2_logits = self.policy.r2_logits(trunk, safe_t, r2_keys=r2_keys)
        else:
            r2_logits = self.policy.r2_logits(trunk, None, r2_keys=r2_keys)
        r2_logits = r2_logits.masked_fill(~r2_masks.to(self.device), -1e9)
        r2_dist = torch.distributions.Categorical(logits=r2_logits)
        safe_r2 = torch.where(r2_actions < 0, torch.zeros_like(r2_actions), r2_actions)
        log_pi_r2_raw = r2_dist.log_prob(safe_r2)
        log_pi_r2 = torch.where(is_stop, torch.zeros_like(log_pi_r2_raw), log_pi_r2_raw)

        log_pi = log_pi_t + log_pi_r2

        ent_t = tmpl_dist.entropy()
        ent_r2_raw = r2_dist.entropy()
        ent_r2 = torch.where(is_stop, torch.zeros_like(ent_r2_raw), ent_r2_raw)
        entropy = ent_t + ent_r2
        return log_pi, entropy, values

    def ppo_update(
        self,
        rollout: list[_Transition],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        self.policy.train()
        # Subclass hook: invalidate any per-update caches (e.g. the
        # encoder_graph r2_keys cache in GraphTransBiPPO) so the next
        # minibatch starts from a fresh, gradient-attached encoding.
        self._begin_update_cycle()
        n = len(rollout)
        old_values = np.array([tr.value for tr in rollout], dtype=np.float32)
        adv_norm = advantages.copy()
        if self.normalize_advantage and n > 1:
            adv_norm = (adv_norm - adv_norm.mean()) / (adv_norm.std() + 1e-8)

        smiles_all = [tr.smiles for tr in rollout]
        t_actions_all = np.array([tr.t_action for tr in rollout], dtype=np.int64)
        r2_actions_all = np.array([tr.r2_action for tr in rollout], dtype=np.int64)
        is_stop_all = np.array([tr.is_stop for tr in rollout], dtype=np.bool_)
        log_pi_old_all = np.array([tr.log_pi_old for tr in rollout], dtype=np.float32)
        template_masks_all = torch.stack([tr.template_mask for tr in rollout]).bool()
        zero_r2 = torch.zeros(self.num_reactants, dtype=torch.bool)
        r2_masks_all = torch.stack(
            [(tr.r2_mask if tr.r2_mask is not None else zero_r2) for tr in rollout]
        ).bool()

        idx = np.arange(n)
        loss_acc, pg_acc, v_acc, ent_acc, clip_frac_acc, kl_acc = [], [], [], [], [], []
        epochs_done = 0
        last_kl = 0.0
        early_stopped = False
        for epoch in range(self.n_epochs):
            np.random.shuffle(idx)
            kl_epoch: list[float] = []
            for start in range(0, n, self.minibatch):
                mb = idx[start : start + self.minibatch]
                if len(mb) == 0:
                    continue
                mb_smiles = [smiles_all[i] for i in mb]
                mb_t = torch.as_tensor(t_actions_all[mb], device=self.device)
                mb_r2 = torch.as_tensor(r2_actions_all[mb], device=self.device)
                mb_is_stop = torch.as_tensor(is_stop_all[mb], device=self.device)
                mb_log_pi_old = torch.as_tensor(log_pi_old_all[mb], device=self.device)
                mb_adv = torch.as_tensor(adv_norm[mb], device=self.device)
                mb_ret = torch.as_tensor(returns[mb], device=self.device)
                mb_old_v = torch.as_tensor(old_values[mb], device=self.device)
                mb_tmpl_mask = template_masks_all[mb]
                mb_r2_mask = r2_masks_all[mb]

                log_pi_new, entropy_per_sample, values = self._evaluate_minibatch(
                    mb_smiles, mb_t, mb_r2, mb_is_stop, mb_tmpl_mask, mb_r2_mask
                )

                ratio = torch.exp(log_pi_new - mb_log_pi_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                if self.clip_range_vf is None:
                    v_loss = F.mse_loss(values, mb_ret)
                else:
                    v_clipped = mb_old_v + torch.clamp(
                        values - mb_old_v, -self.clip_range_vf, self.clip_range_vf
                    )
                    v_loss = torch.max((values - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()

                entropy = entropy_per_sample.mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_pi_new - mb_log_pi_old
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio).item()
                    clip_frac = float((torch.abs(ratio - 1.0) > self.clip_range).float().mean().item())

                kl_epoch.append(approx_kl)
                loss_acc.append(float(loss.detach().cpu().item()))
                pg_acc.append(float(pg_loss.detach().cpu().item()))
                v_acc.append(float(v_loss.detach().cpu().item()))
                ent_acc.append(float(entropy.detach().cpu().item()))
                clip_frac_acc.append(clip_frac)
                kl_acc.append(approx_kl)

            epochs_done = epoch + 1
            mean_kl_epoch = float(np.mean(kl_epoch)) if kl_epoch else 0.0
            last_kl = mean_kl_epoch
            if self.target_kl is not None and mean_kl_epoch > 1.5 * self.target_kl:
                early_stopped = True
                break

        ev = _explained_variance(old_values, returns.astype(np.float32))
        return {
            "train/loss": float(np.mean(loss_acc)) if loss_acc else 0.0,
            "train/policy_loss": float(np.mean(pg_acc)) if pg_acc else 0.0,
            "train/value_loss": float(np.mean(v_acc)) if v_acc else 0.0,
            "train/entropy": float(np.mean(ent_acc)) if ent_acc else 0.0,
            "train/approx_kl": float(np.mean(kl_acc)) if kl_acc else 0.0,
            "train/clip_fraction": float(np.mean(clip_frac_acc)) if clip_frac_acc else 0.0,
            "train/epochs_done": float(epochs_done),
            "train/early_stop_kl": float(last_kl),
            "train/early_stopped": float(1.0 if early_stopped else 0.0),
            "train/explained_variance": float(ev),
            "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    # ------------------------------------------------------------------
    # Greedy evaluation
    # ------------------------------------------------------------------

    def _greedy_trajectory(self, start_smiles: str) -> tuple[float, float, int, float]:
        self.policy.eval()
        current = str(start_smiles)
        start_qed = _qed(current, round_digits=self.qed_round_digits)
        max_qed = start_qed
        total_reward = 0.0
        react_steps = 0
        with torch.no_grad():
            for _ in range(self.max_episode_len + int(self.use_stop_action)):
                at_max = react_steps >= self.max_episode_len
                if at_max and not self.use_stop_action:
                    break
                (
                    t_idx,
                    _r2_idx,
                    _,
                    _,
                    _tmpl_mask,
                    _,
                    product,
                ) = self._sample_action(current, force_stop=at_max, deterministic=True)
                if t_idx < 0 or t_idx == self.stop_index:
                    break
                if product is None:
                    break
                next_qed = _qed(product, round_digits=self.qed_round_digits)
                total_reward += float(
                    next_qed - _qed(current, round_digits=self.qed_round_digits)
                )
                current = product
                react_steps += 1
                max_qed = max(max_qed, next_qed)
        final_qed = _qed(current, round_digits=self.qed_round_digits)
        return total_reward, final_qed - start_qed, react_steps, max_qed

    def evaluate(self) -> dict[str, float]:
        self.policy.eval()
        rewards: list[float] = []
        deltas: list[float] = []
        lengths: list[int] = []
        max_qeds: list[float] = []
        # Swap the active pool to the eval pool (no-op in lookup mode — same
        # train pool — but a real swap to the disjoint test pool in encoder
        # mode). The cached r2_keys is computed once for the eval sweep; like
        # the rollout cache, this amortises the encoder forward over all test
        # starts. The try/finally guarantees we restore the train pool even
        # if a trajectory raises.
        self._swap_active_pool(self._eval_pool_role)
        prev_active_keys = self._active_r2_keys
        try:
            with torch.no_grad():
                self._active_r2_keys = self._compute_active_r2_keys(
                    pool=self._eval_pool_role, with_grad=False
                )
            for s in self.sampler.eval_starts():
                r, d, l, mq = self._greedy_trajectory(s)
                rewards.append(r)
                deltas.append(d)
                lengths.append(l)
                max_qeds.append(mq)
        finally:
            self._swap_active_pool("train")
            self._active_r2_keys = prev_active_keys
        if not rewards:
            return {
                "eval/mean_reward": 0.0,
                "eval/avg_delta_qed": 0.0,
                "eval/mean_final_delta_qed": 0.0,
                "eval/mean_ep_length": 0.0,
                "eval/max_qed": float("nan"),
                "eval/n_molecules": 0,
            }
        return {
            "eval/mean_reward": float(np.mean(rewards)),
            "eval/avg_delta_qed": float(np.mean(deltas)),
            "eval/mean_final_delta_qed": float(np.mean(deltas)),
            "eval/mean_ep_length": float(np.mean(lengths)),
            "eval/max_qed": float(np.max(max_qeds)),
            "eval/n_molecules": len(rewards),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "config": self.config,
                "policy_arch": self.policy_arch,
            },
            path,
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def run_training_loop(trainer: BiPPO, run, config: dict, experiment_name: str) -> None:
    """Shared PPO training loop used by both BiPPO and its graph-aware subclass.

    The loop is encoder-agnostic; ``trainer`` only needs to expose
    ``collect_rollout``, ``compute_gae``, ``ppo_update``, ``evaluate``, and
    ``save`` plus the cumulative-counter attributes referenced below. The
    extracted helper is so ``GraphTransBiPPO`` can reuse the exact same
    rollout / eval / checkpoint cadence as the fingerprint trainer without
    having to copy-paste the body.
    """
    training_cfg = config.get("training", {})
    total_timesteps = int(training_cfg.get("total_timesteps", 1_000_000))
    eval_freq = int(training_cfg.get("eval_freq", 10_000))
    save_freq = int(training_cfg.get("save_freq", 100_000))
    n_steps = trainer.n_steps

    out_dir = run_dir(run.id if run is not None else experiment_name)
    best_eval = -float("inf")

    global_step = 0
    last_eval_bucket = -1
    last_save_bucket = -1
    while global_step < total_timesteps:
        rollout, last_value = trainer.collect_rollout(
            n_steps, base_step=global_step, log_episodes=True
        )
        advantages, returns = trainer.compute_gae(rollout, last_value)
        update_metrics = trainer.ppo_update(rollout, advantages, returns)
        global_step += len(rollout)

        rewards = [tr.reward for tr in rollout]
        stop_frac = float(np.mean([tr.is_stop for tr in rollout])) if rollout else 0.0
        rollout_metrics = {
            "train/global_step": global_step,
            "train/rollout_steps": float(len(rollout)),
            "train/rollout_mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "train/rollout_stop_fraction": stop_frac,
            "train/rollout_invalid_count_cum": float(trainer._invalid_reaction_count),
            "train/rejection_count_cum": float(trainer._rejection_total),
        }
        wandb.log({**rollout_metrics, **update_metrics}, step=global_step)

        bucket = global_step // eval_freq
        if eval_freq > 0 and bucket > last_eval_bucket:
            last_eval_bucket = bucket
            eval_metrics = trainer.evaluate()
            eval_metrics["train/global_step"] = global_step
            wandb.log(eval_metrics, step=global_step)
            if eval_metrics["eval/mean_reward"] > best_eval:
                best_eval = eval_metrics["eval/mean_reward"]
                trainer.save(out_dir / "best_model.pt")

        sbucket = global_step // save_freq
        if save_freq > 0 and sbucket > last_save_bucket:
            last_save_bucket = sbucket
            trainer.save(out_dir / f"model_step_{global_step}.pt")

    final_eval = trainer.evaluate()
    final_eval["train/global_step"] = global_step
    wandb.log(final_eval, step=global_step)
    trainer.save(out_dir / "final_model.pt")
    if run is not None:
        run.finish()


def train(config: dict, experiment_name: str) -> None:
    trainer = BiPPO(config)
    run = init_wandb(config, "ppo_bi", experiment_name)
    run_training_loop(trainer, run, config, experiment_name)


__all__ = ["BiPPO", "train", "run_training_loop"]

"""KNN action wrapper for PGFS-style TD3 second-reactant selection.

Two scoring modes are supported:

* ``score_mode="reactant_distance"`` (legacy default): pick the top-k closest
  R(2) candidates by L2 distance to the continuous policy output, then score
  each candidate with ``QED(R2) - distance(a, R2)`` and ε-greedy-argmax. This
  is a fast surrogate that does NOT require running the forward reaction.
* ``score_mode="product"`` (PGFS-faithful, Algorithm 1 lines 12-14): for each
  of the top-k closest R(2) candidates, run the forward reaction
  ``ForwardReaction(R(1), T, R(2))`` and score the resulting **product** with
  the env's reward function (e.g. ΔQED). Argmax over the scored products is
  returned. ε is forced to 0 here because PGFS uses a hard argmax over
  products.

The selection knobs (``score_mode``, ``random_epsilon``, ``top_k``) default to
the legacy reactant-distance behaviour so existing TD3 configurations are
unaffected. The PGFS-faithful Bi-TD3 config sets
``td3.knn_score_mode: product`` and ``td3.knn_random_epsilon: 0.0`` explicitly.
"""

from __future__ import annotations

import random

import faiss
import faiss.contrib.torch_utils  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED


_VALID_SCORE_MODES = ("reactant_distance", "product")


class KNNWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env,
        enabled: bool = True,
        *,
        score_mode: str = "reactant_distance",
        random_epsilon: float = 0.3,
        top_k: int = 5,
    ):
        super().__init__(env)
        self.enabled = enabled
        if score_mode not in _VALID_SCORE_MODES:
            raise ValueError(
                f"score_mode must be one of {_VALID_SCORE_MODES}, got {score_mode!r}"
            )
        self.score_mode = score_mode
        self.random_epsilon = float(random_epsilon)
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")
        self.reactants = self.env.unwrapped.reactants
        self.knn_indices = {}
        self.res = None
        self.use_gpu = False
        if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
            try:
                self.res = faiss.StandardGpuResources()
                self.use_gpu = True
            except Exception:
                self.res = None
                self.use_gpu = False

    def _initialize_index_for_template(self, template_index: int) -> None:
        if template_index in self.knn_indices:
            return
        if template_index >= len(self.env.unwrapped.templates):
            return
        if self.env.unwrapped.templates[template_index]["type"] != "bimolecular":
            return
        valid_reactants = self.env.unwrapped.reaction_manager.get_valid_reactants(template_index)
        if not valid_reactants:
            return

        fps = torch.stack([torch.as_tensor(self.reactants[reactant], dtype=torch.float32) for reactant in valid_reactants])
        if self.use_gpu:
            fps = fps.cuda()
            index = faiss.GpuIndexFlatL2(self.res, fps.shape[1])
            index.add(fps.cpu().numpy())
        else:
            index = faiss.IndexFlatL2(fps.shape[1])
            index.add(fps.numpy())
        self.knn_indices[template_index] = index

    def action(self, action):
        template_one_hot, reactant_vector = action
        template_index = torch.argmax(template_one_hot).item()
        if not self.enabled:
            return template_index, (reactant_vector if isinstance(reactant_vector, str) else None)
        if template_index >= len(self.env.unwrapped.templates):
            return template_index, None

        reactant_vector = (reactant_vector >= 0).float()
        self._initialize_index_for_template(template_index)
        return self._process_knn_search(
            self.knn_indices.get(template_index), reactant_vector, template_index
        )

    def _process_knn_search(self, knn_index, reactant_vector, template_index: int):
        if knn_index is None or torch.all(reactant_vector.eq(0)):
            return template_index, None
        k = min(self.top_k, int(knn_index.ntotal))
        if k <= 0:
            return template_index, None
        query_np = reactant_vector.detach().cpu().numpy().astype(np.float32, copy=False).reshape(1, -1)
        distances, indices = knn_index.search(query_np, k)
        valid_reactants = self.env.unwrapped.reaction_manager.get_valid_reactants(template_index)
        top_reactants = [valid_reactants[int(idx)] for idx in indices[0] if 0 <= int(idx) < len(valid_reactants)]
        if not top_reactants:
            return template_index, None

        if self.score_mode == "product":
            return template_index, self._select_by_product_reward(template_index, top_reactants)
        return template_index, self._select_by_reactant_distance(top_reactants, distances[0])

    def _select_by_reactant_distance(self, top_reactants, distances) -> str | None:
        """Legacy surrogate: score each R(2) by ``QED(R2) - L2(a, R2)`` with ε-random pick."""
        if self.random_epsilon > 0.0 and random.random() < self.random_epsilon:
            return random.choice(top_reactants)
        best_reactant = None
        best_score = -np.inf
        for pos, reactant in enumerate(top_reactants):
            mol = Chem.MolFromSmiles(reactant)
            qed = QED.qed(mol) if mol else 0.0
            score = qed - float(distances[pos])
            if score > best_score:
                best_score = score
                best_reactant = reactant
        return best_reactant

    def _select_by_product_reward(self, template_index: int, top_reactants) -> str | None:
        """PGFS Algorithm 1 lines 12-14: forward-react each top-k R(2) candidate
        and argmax the resulting product's reward.

        Uses ``env.reward_fn.step_reward(R(1), product)`` so that whatever the
        env's reward function is (ΔQED, final QED, ...) drives the selection
        — this keeps the kNN scoring consistent with the actual env reward
        the critic is being trained against. Candidates whose forward
        reaction returns ``None`` are skipped. If every candidate fails, the
        first top-k entry is returned so the downstream env.step still has a
        well-formed action (it will produce ``invalid_reaction_penalty`` if
        the same reaction fails there, which is the correct learning signal).
        """
        unwrapped = self.env.unwrapped
        reaction_manager = unwrapped.reaction_manager
        reward_fn = unwrapped.reward_fn
        current_state = unwrapped.current_state
        template = unwrapped.templates[template_index]
        best_reactant = None
        best_score = -np.inf
        valid_seen = False
        for reactant in top_reactants:
            product = reaction_manager.apply_reaction(current_state, template, reactant)
            if product is None:
                continue
            valid_seen = True
            score = float(reward_fn.step_reward(current_state, product))
            if score > best_score:
                best_score = score
                best_reactant = reactant
        if best_reactant is not None:
            return best_reactant
        # All k forward reactions failed. Return the closest reactant so the
        # env step records an explicit invalid_reaction transition (the same
        # signal a random rollout would generate for these states).
        return top_reactants[0] if not valid_seen else best_reactant

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False


__all__ = ["KNNWrapper"]

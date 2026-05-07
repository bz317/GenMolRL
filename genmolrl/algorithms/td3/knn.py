"""KNN action wrapper for PGFS-style TD3 second-reactant selection."""

from __future__ import annotations

import random

import faiss
import faiss.contrib.torch_utils  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED


class KNNWrapper(gym.ActionWrapper):
    def __init__(self, env, enabled: bool = True):
        super().__init__(env)
        self.enabled = enabled
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
        return self._process_knn_search(self.knn_indices.get(template_index), reactant_vector, template_index)

    def _process_knn_search(self, knn_index, reactant_vector, template_index: int, k: int = 5, epsilon: float = 0.3):
        if knn_index is None or torch.all(reactant_vector.eq(0)):
            return template_index, None
        k = min(k, int(knn_index.ntotal))
        if k <= 0:
            return template_index, None
        query_np = reactant_vector.detach().cpu().numpy().astype(np.float32, copy=False).reshape(1, -1)
        distances, indices = knn_index.search(query_np, k)
        valid_reactants = self.env.unwrapped.reaction_manager.get_valid_reactants(template_index)
        top_reactants = [valid_reactants[int(idx)] for idx in indices[0] if 0 <= int(idx) < len(valid_reactants)]
        if not top_reactants:
            return template_index, None
        if random.random() < epsilon:
            return template_index, random.choice(top_reactants)

        best_reactant = None
        best_score = -np.inf
        for pos, reactant in enumerate(top_reactants):
            mol = Chem.MolFromSmiles(reactant)
            qed = QED.qed(mol) if mol else 0.0
            score = qed - distances[0][pos]
            if score > best_score:
                best_score = score
                best_reactant = reactant
        return template_index, best_reactant

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False


__all__ = ["KNNWrapper"]

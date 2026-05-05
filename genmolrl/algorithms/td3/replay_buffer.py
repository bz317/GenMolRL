"""Replay buffer for the self-contained PGFS-style TD3 trainer."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim: int, template_dim: int, r2_vec_dim: int, capacity: int = int(1e6)):
        self.capacity = capacity
        self.count = 0
        self.index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_smiles = [None] * capacity
        self.state_tensors = torch.zeros((capacity, state_dim), device=self.device)
        self.templates = torch.zeros((capacity, template_dim), device=self.device)
        self.r2_vectors = torch.zeros((capacity, r2_vec_dim), device=self.device)
        self.rewards = torch.zeros((capacity, 1), device=self.device)
        self.next_state_smiles = [None] * capacity
        self.next_state_tensors = torch.zeros((capacity, state_dim), device=self.device)
        self.not_dones = torch.zeros((capacity, 1), device=self.device)

    def add(self, state_smiles, state_obs, template, r2_vector, reward, next_state_smiles, next_state_obs, done):
        idx = self.index
        self.state_smiles[idx] = state_smiles
        self.state_tensors[idx] = torch.as_tensor(state_obs, dtype=torch.float32, device=self.device)
        self.templates[idx] = template.clone().detach()
        self.r2_vectors[idx] = r2_vector.clone().detach()
        self.rewards[idx] = torch.as_tensor([reward], dtype=torch.float32, device=self.device)
        self.next_state_smiles[idx] = next_state_smiles
        self.next_state_tensors[idx] = torch.as_tensor(next_state_obs, dtype=torch.float32, device=self.device)
        self.not_dones[idx] = torch.as_tensor([1.0 - done], dtype=torch.float32, device=self.device)

        self.index = (idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.count < batch_size:
            raise ValueError("Not enough elements in the buffer to sample")
        indices = np.random.randint(0, self.count, size=batch_size)
        return (
            [self.state_smiles[i] for i in indices],
            self.state_tensors[indices],
            self.templates[indices],
            self.r2_vectors[indices],
            self.rewards[indices],
            [self.next_state_smiles[i] for i in indices],
            self.next_state_tensors[indices],
            self.not_dones[indices],
        )

    def size(self) -> int:
        return self.count

    def clear(self) -> None:
        self.count = 0
        self.index = 0


__all__ = ["ReplayBuffer"]

"""Neural networks for the self-contained PGFS-style TD3 trainer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 128]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PiNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 167]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(combined_input))


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, template_dim: int, action_dim: int):
        super().__init__()
        self.f_net = FNetwork(state_dim, template_dim)
        self.pi_net = PiNetwork(state_dim + template_dim, action_dim)
        self.logits: torch.Tensor | None = None

    def forward(
        self,
        state: torch.Tensor,
        template_mask_info=None,
        temperature: float = 1.0,
        evaluate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.logits = self.f_net(state)
        template_one_hot, template_types = self._apply_template_mask(template_mask_info, temperature, evaluate)
        template_indices = template_one_hot.argmax(dim=-1)

        if template_types is not None:
            template_indices = template_indices.to(template_types.device)
            is_bimolecular = template_types[template_indices] == 1
        else:
            is_bimolecular = torch.zeros_like(template_indices, dtype=torch.bool)

        r2_vector = torch.zeros(state.size(0), self.pi_net.network[-1].out_features, device=state.device)
        if is_bimolecular.any():
            bimolecular_states = torch.cat((state[is_bimolecular], template_one_hot[is_bimolecular]), dim=-1)
            r2_vector[is_bimolecular] = self.pi_net(bimolecular_states)
        return template_one_hot, r2_vector

    def _apply_template_mask(self, template_mask_info, temperature: float, evaluate: bool):
        if self.logits is None:
            raise RuntimeError("Actor logits are not initialized.")
        if template_mask_info is None:
            if evaluate:
                selected_templates = self.logits.argmax(dim=-1)
                return F.one_hot(selected_templates, num_classes=self.logits.size(1)).float(), None
            return F.gumbel_softmax(self.logits, tau=temperature, hard=True), None

        template_mask, template_types = template_mask_info
        masked_logits = self.logits + (1 - template_mask) * (-1e9)
        if evaluate:
            selected_templates = masked_logits.argmax(dim=-1)
            return F.one_hot(selected_templates, num_classes=masked_logits.size(1)).float(), template_types
        return F.gumbel_softmax(masked_logits, tau=temperature, hard=True), template_types


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, template_dim: int, r2_vec_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 64, 16]
        layers: list[nn.Module] = []
        prev_dim = state_dim + template_dim + r2_vec_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

    def forward(self, state: torch.Tensor, template: torch.Tensor, r2_vector: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, template, r2_vector], dim=-1))

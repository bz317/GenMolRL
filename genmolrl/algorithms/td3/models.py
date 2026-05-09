"""Neural networks for the self-contained PGFS-style TD3 trainer.

Both width and activation are configurable so the agent can mirror PPO/A2C's
SB3 default ``[64, 64]`` Tanh policy. The legacy defaults (``[256, 128, 128]``
ReLU actor / ``[256, 64, 16]`` ReLU critic / ``[256, 256, 167]`` ReLU R2 head)
are preserved so unconfigured callers reproduce the original behavior bit-for-bit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}


def resolve_activation(activation) -> type[nn.Module]:
    """Resolve a string / class into an ``nn.Module`` activation factory."""
    if activation is None:
        return nn.ReLU
    if isinstance(activation, str):
        key = activation.lower()
        if key not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown TD3 activation: {activation!r}. "
                f"Choose from: {sorted(_ACTIVATIONS)}."
            )
        return _ACTIVATIONS[key]
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation
    raise TypeError(f"Unsupported activation specifier: {activation!r}")


def apply_td3_template_mask(
    logits: torch.Tensor,
    template_mask_info,
    *,
    temperature: float,
    evaluate: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Masked template softmax / Gumbel-Softmax (hard); optional ``template_types`` for bi heads."""
    if template_mask_info is None:
        if evaluate:
            selected_templates = logits.argmax(dim=-1)
            return F.one_hot(selected_templates, num_classes=logits.size(1)).float(), None
        return F.gumbel_softmax(logits, tau=temperature, hard=True), None
    template_mask, template_types = template_mask_info
    masked_logits = logits + (1 - template_mask) * (-1e9)
    if evaluate:
        selected_templates = masked_logits.argmax(dim=-1)
        return F.one_hot(selected_templates, num_classes=masked_logits.size(1)).float(), template_types
    return F.gumbel_softmax(masked_logits, tau=temperature, hard=True), template_types


class FNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation=None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 128]
        act_cls = resolve_activation(activation)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), act_cls()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._activation_cls = act_cls
        self._init_weights()

    def _init_weights(self) -> None:
        nonlin = "relu" if self._activation_cls is nn.ReLU else "linear"
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlin)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PiNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation=None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 167]
        act_cls = resolve_activation(activation)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), act_cls()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._activation_cls = act_cls
        self._init_weights()

    def _init_weights(self) -> None:
        nonlin = "relu" if self._activation_cls is nn.ReLU else "linear"
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlin)

    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(combined_input))


class ActorNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        template_dim: int,
        action_dim: int,
        f_hidden_dims: list[int] | None = None,
        pi_hidden_dims: list[int] | None = None,
        activation=None,
    ):
        super().__init__()
        self.f_net = FNetwork(state_dim, template_dim, hidden_dims=f_hidden_dims, activation=activation)
        self.pi_net = PiNetwork(
            state_dim + template_dim, action_dim, hidden_dims=pi_hidden_dims, activation=activation
        )
        self.logits: torch.Tensor | None = None

    def forward(
        self,
        state: torch.Tensor,
        template_mask_info=None,
        temperature: float = 1.0,
        evaluate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.logits = self.f_net(state)
        template_one_hot, template_types = apply_td3_template_mask(
            self.logits, template_mask_info, temperature=temperature, evaluate=evaluate
        )
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


class ActorNetworkUniDiscrete(nn.Module):
    """Uni-only TD3 actor: template logits only (no continuous R2 / Pi head)."""

    def __init__(
        self,
        state_dim: int,
        template_dim: int,
        f_hidden_dims: list[int] | None = None,
        activation=None,
    ):
        super().__init__()
        self.f_net = FNetwork(state_dim, template_dim, hidden_dims=f_hidden_dims, activation=activation)
        self.logits: torch.Tensor | None = None

    def forward(
        self,
        state: torch.Tensor,
        template_mask_info=None,
        temperature: float = 1.0,
        evaluate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.logits = self.f_net(state)
        template_one_hot, _template_types = apply_td3_template_mask(
            self.logits, template_mask_info, temperature=temperature, evaluate=evaluate
        )
        r2_vector = state.new_zeros(state.size(0), 0)
        return template_one_hot, r2_vector


class CriticNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        template_dim: int,
        r2_vec_dim: int,
        hidden_dims: list[int] | None = None,
        activation=None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 64, 16]
        act_cls = resolve_activation(activation)
        layers: list[nn.Module] = []
        prev_dim = state_dim + template_dim + r2_vec_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), act_cls()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._activation_cls = act_cls
        self._init_weights()

    def _init_weights(self) -> None:
        nonlin = "relu" if self._activation_cls is nn.ReLU else "linear"
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlin)

    def forward(self, state: torch.Tensor, template: torch.Tensor, r2_vector: torch.Tensor) -> torch.Tensor:
        if r2_vector.shape[-1] == 0:
            return self.network(torch.cat([state, template], dim=-1))
        return self.network(torch.cat([state, template, r2_vector], dim=-1))

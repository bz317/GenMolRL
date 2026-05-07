"""Random action selector used during TD3 warm-up."""

from __future__ import annotations

import random

import torch

from genmolrl.algorithms.td3.constants import TD3_UNI_DISCRETE_ACTION_DESIGN
from genmolrl.algorithms.td3.mask_kind import td3_template_mask_kind

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoValidActionError(ValueError):
    """Raised when a TD3 state has no real reaction action available."""


def _continuous_r2_placeholder_dim(env) -> int:
    if getattr(env.unwrapped, "action_design", "") == TD3_UNI_DISCRETE_ACTION_DESIGN:
        return 0
    return int(env.unwrapped.observation_space.shape[0])


def stop_action(env):
    n_actions = int(env.unwrapped.action_space.n)
    template_one_hot = torch.zeros(n_actions, device=device)
    template_one_hot[-1] = 1
    template_one_hot = template_one_hot.unsqueeze(0)
    d = _continuous_r2_placeholder_dim(env)
    r2 = torch.zeros(d, device=device).unsqueeze(0)
    return template_one_hot, r2


def select_random_action(env, smiles_string, *, stop_probability: float = 0.0, template_mask_kind: str | None = None):
    rm = env.unwrapped.reaction_manager
    templates = env.unwrapped.templates
    mask_kind = td3_template_mask_kind(env, override=template_mask_kind)
    feasible = rm.feasible_first_reactant_templates(smiles_string, kind=mask_kind)
    if feasible and getattr(env.unwrapped, "use_stop_action", False) and random.random() < float(stop_probability):
        return stop_action(env)
    if not feasible:
        mask = rm.get_mask(smiles_string, kind=mask_kind)
        if mask is None or int(mask.sum().item()) == 0:
            if getattr(env.unwrapped, "use_stop_action", False):
                return stop_action(env)
            raise NoValidActionError("No valid templates found for the given SMILES string.")
        if getattr(env.unwrapped, "use_stop_action", False):
            return stop_action(env)
        raise NoValidActionError("No feasible template: bimolecular partners missing for all applicable templates.")

    selected_idx = random.choice(feasible)
    n_actions = int(env.unwrapped.action_space.n)
    template_one_hot = torch.zeros(n_actions, device=device)
    template_one_hot[selected_idx] = 1
    template_one_hot = template_one_hot.unsqueeze(0)

    if templates[selected_idx]["type"] == "bimolecular":
        r2 = random.choice(rm.get_valid_reactants(selected_idx))
    else:
        r2 = torch.zeros(_continuous_r2_placeholder_dim(env), device=device).unsqueeze(0)
    return template_one_hot, r2


__all__ = ["NoValidActionError", "select_random_action", "stop_action"]

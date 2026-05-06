"""Random action selector used during TD3 warm-up."""

from __future__ import annotations

import random

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoValidActionError(ValueError):
    """Raised when a TD3 state has no real reaction action available."""


def stop_action(env):
    n_actions = int(env.unwrapped.action_space.n)
    template_one_hot = torch.zeros(n_actions, device=device)
    template_one_hot[-1] = 1
    template_one_hot = template_one_hot.unsqueeze(0)
    r2 = torch.zeros(env.unwrapped.observation_space.shape[0], device=device).unsqueeze(0)
    return template_one_hot, r2


def select_random_action(env, smiles_string):
    rm = env.unwrapped.reaction_manager
    templates = env.unwrapped.templates
    mask_kind = getattr(env.unwrapped.mask_provider, "mode", "r2_available")
    feasible = rm.feasible_first_reactant_templates(smiles_string, kind=mask_kind)
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
        r2 = torch.zeros(env.unwrapped.observation_space.shape[0], device=device).unsqueeze(0)
    return template_one_hot, r2


__all__ = ["NoValidActionError", "select_random_action", "stop_action"]

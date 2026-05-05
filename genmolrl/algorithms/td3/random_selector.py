"""Random action selector used during TD3 warm-up."""

from __future__ import annotations

import random

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_random_action(env, smiles_string):
    rm = env.unwrapped.reaction_manager
    templates = env.unwrapped.templates
    feasible = rm.feasible_first_reactant_templates(smiles_string)
    if not feasible:
        mask = rm.get_mask(smiles_string)
        if mask is None or int(mask.sum().item()) == 0:
            raise ValueError("No valid templates found for the given SMILES string.")
        raise ValueError("No feasible template: bimolecular partners missing for all applicable templates.")

    selected_idx = random.choice(feasible)
    template_one_hot = torch.zeros(len(templates), device=device)
    template_one_hot[selected_idx] = 1
    template_one_hot = template_one_hot.unsqueeze(0)

    if templates[selected_idx]["type"] == "bimolecular":
        r2 = random.choice(rm.get_valid_reactants(selected_idx))
    else:
        r2 = torch.zeros(env.unwrapped.observation_space.shape[0], device=device).unsqueeze(0)
    return template_one_hot, r2


__all__ = ["select_random_action"]

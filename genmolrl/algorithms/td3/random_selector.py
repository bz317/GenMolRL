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
    # R2 is the fingerprint of the second reactant — fixed at the morgan-FP
    # width (``base_obs_dim``), not the observation width (which now includes
    # the appended action mask). Falls back to obs dim for legacy callers.
    base = getattr(env.unwrapped, "base_obs_dim", None)
    if base is not None:
        return int(base)
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
    # Drop bimolecular templates whose R(2) pool is empty for the loaded
    # reactant set. Such templates can still be picked by the learned actor
    # (the env returns ``invalid_reaction_penalty`` and the policy learns
    # from the signal), but the random warmup writes a real ``(T, R2)``
    # transition into the replay buffer and therefore needs a sample-able
    # R(2). This mirrors PGFS's pool-side filter on ``pool_R(2)[T]`` and is
    # required under ``masking: none`` / ``masking: substructure`` where
    # ``feasible_first_reactant_templates`` does NOT itself filter on R(2)
    # availability. Under ``masking: r2_available`` / ``reaction_valid``
    # this filter is a no-op (those masks already require R(2)).
    if feasible:
        feasible = [
            idx
            for idx in feasible
            if templates[idx].get("type") != "bimolecular"
            or rm.get_valid_reactants(idx)
        ]
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

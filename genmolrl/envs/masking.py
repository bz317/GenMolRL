"""Mask providers for SB3 and TD3 policies."""

from __future__ import annotations

import numpy as np


class MaskProvider:
    def __init__(self, mode: str = "substructure", use_stop_action: bool = True):
        aliases = {
            "current": "substructure",
            "executable": "r2_available",
            "ppo_original": "reaction_valid",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"substructure", "reaction_valid", "r2_available", "none"}:
            raise ValueError(f"Unsupported masking mode: {mode}")
        self.mode = mode
        self.use_stop_action = bool(use_stop_action)

    def template_mask_with_stop(self, reaction_manager, smiles: str | None) -> np.ndarray:
        mask = reaction_manager.get_mask(smiles, kind=self.mode).cpu().numpy().astype(np.int8)
        if self.use_stop_action:
            mask = np.concatenate([mask, np.ones(1, dtype=np.int8)])
        return mask

    def multidiscrete_mask(self, reaction_manager, smiles: str | None) -> np.ndarray:
        template_mask = self.template_mask_with_stop(reaction_manager, smiles)
        # Preserve current factorized MultiDiscrete behavior: R2 is not conditioned on T.
        r2_mask = np.ones(len(reaction_manager.reactants), dtype=np.int8)
        return np.concatenate([template_mask, r2_mask])

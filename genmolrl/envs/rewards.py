"""Reward functions for molecule design."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import QED


def qed(smiles: str | None) -> float:
    if not smiles:
        return 0.0
    mol = Chem.MolFromSmiles(smiles)
    return float(QED.qed(mol)) if mol is not None else 0.0


class RewardFunction:
    def __init__(
        self,
        reward_type: str = "delta_qed",
        invalid_penalty: float = -1.0,
        round_digits: int | None = None,
    ):
        if reward_type not in {"delta_qed", "final_qed"}:
            raise ValueError(f"Unsupported reward type: {reward_type}")
        self.reward_type = reward_type
        self.invalid_penalty = float(invalid_penalty)
        self.round_digits = round_digits

    def _maybe_round(self, value: float) -> float:
        if self.round_digits is None:
            return float(value)
        return float(round(value, int(self.round_digits)))

    def step_reward(self, previous_smiles: str | None, current_smiles: str | None) -> float:
        if not current_smiles:
            return self.invalid_penalty
        current_qed = qed(current_smiles)
        if self.reward_type == "final_qed":
            return self._maybe_round(current_qed)
        return self._maybe_round(current_qed - qed(previous_smiles))

    def stop_reward(self, *, current_step: int, stop_early_penalty: float, stop_penalty_until_step: int) -> float:
        if stop_penalty_until_step > 0 and current_step <= stop_penalty_until_step:
            return float(stop_early_penalty)
        return 0.0

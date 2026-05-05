"""Start-state selection strategies."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from genmolrl.config import resolve_path


def load_start_file(path: str | None, reactants: dict) -> list[str]:
    if not path:
        return []
    starts = []
    for line in Path(resolve_path(path)).read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token and token in reactants and token not in starts:
            starts.append(token)
    return starts


class StartStrategy:
    def __init__(self, strategy: str = "random_pool", fixed_start_smiles: str | None = None, start_smiles_file: str | None = None):
        if strategy not in {"random_pool", "cycle_file", "fixed", "learned_policy"}:
            raise ValueError(f"Unsupported start_strategy: {strategy}")
        self.strategy = strategy
        self.fixed_start_smiles = fixed_start_smiles
        self.start_smiles_file = start_smiles_file
        self._cycle_idx = 0
        self._starts: list[str] = []

    def initialize(self, reactants: dict) -> None:
        if self.strategy == "fixed":
            if self.fixed_start_smiles not in reactants:
                raise ValueError(f"fixed_start_smiles not in reactant pool: {self.fixed_start_smiles}")
            self._starts = [str(self.fixed_start_smiles)]
        elif self.strategy == "cycle_file":
            self._starts = load_start_file(self.start_smiles_file, reactants)
            if not self._starts:
                raise ValueError(f"No usable starts from {self.start_smiles_file!r}")
        else:
            self._starts = list(reactants.keys())

    def sample(self, env) -> str:
        if self.strategy == "learned_policy":
            raise NotImplementedError("learned_policy is reserved for SynFlowNet-style adapters.")
        if self.strategy in {"fixed", "cycle_file"}:
            smiles = self._starts[self._cycle_idx % len(self._starts)]
            if self.strategy == "cycle_file":
                self._cycle_idx += 1
            return smiles
        return str(env.np_random.choice(self._starts))

    @staticmethod
    def validate(smiles: str) -> bool:
        return Chem.MolFromSmiles(smiles) is not None

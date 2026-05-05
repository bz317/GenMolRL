"""Placeholder adapter for future REINVENT scaffold-decorator integration."""

from __future__ import annotations


class ScaffoldDecoratorAdapter:
    """Documents the future scaffold-conditioned SMILES generation boundary."""

    start_strategy = "scaffold_file"
    action_representation = "smiles_token"

    def sample(self, n: int):
        raise NotImplementedError("Scaffold-decorator integration is reserved for a later GenMolRL phase.")

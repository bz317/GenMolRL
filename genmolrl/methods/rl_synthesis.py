"""Common synthesis-method interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class GenerationResult:
    smiles: str | None
    reward: float | None = None
    metadata: dict[str, Any] | None = None


class MoleculeGenerator(Protocol):
    def sample(self, n: int) -> list[GenerationResult]:
        """Generate candidate molecules."""

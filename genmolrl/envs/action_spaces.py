"""Action-space helpers."""

from __future__ import annotations

from dataclasses import dataclass

from gymnasium import spaces


@dataclass(frozen=True)
class ActionSpaceSpec:
    family: str
    reaction_mode: str
    action_design: str = "discrete"
    use_stop_action: bool = True

    def build(self, num_templates: int, num_reactants: int):
        if self.family == "sb3_multidiscrete":
            return spaces.MultiDiscrete([num_templates + int(self.use_stop_action), num_reactants])
        if self.family in {"sb3_discrete", "td3_pgfs"}:
            return spaces.Discrete(num_templates + int(self.use_stop_action))
        raise ValueError(f"Unsupported action-space family: {self.family}")

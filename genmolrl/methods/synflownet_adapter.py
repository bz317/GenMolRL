"""Placeholder adapter for future SynFlowNet integration."""

from __future__ import annotations


class SynFlowNetAdapter:
    """Documents the future adapter boundary without importing SynFlowNet eagerly."""

    start_strategy = "learned_policy"
    action_representation = "reaction_graph_action"

    def sample(self, n: int):
        raise NotImplementedError("SynFlowNet integration is reserved for a later GenMolRL phase.")

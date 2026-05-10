"""Lazy adapter for GenMolRL's GraphTransPPO method."""

from __future__ import annotations


class GraphTransPPOAdapter:
    """Adapter boundary that avoids importing graph dependencies until needed."""

    start_strategy = "supplied"
    action_representation = "reaction_graph_action"
    algorithm = "graphtransppo"

    @staticmethod
    def train(config: dict, experiment_name: str) -> None:
        from genmolrl.algorithms.graphtransppo.train import train

        train(config, experiment_name)

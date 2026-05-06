"""Lazy adapter for GenMolRL's GraphTransRL method."""

from __future__ import annotations


class GraphTransRLAdapter:
    """Adapter boundary that avoids importing graph dependencies until needed."""

    start_strategy = "supplied"
    action_representation = "reaction_graph_action"
    algorithm = "graphtransrl"

    @staticmethod
    def train(config: dict, experiment_name: str) -> None:
        from genmolrl.algorithms.graphtransrl.train import train

        train(config, experiment_name)

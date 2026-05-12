"""Lazy adapter for the GraphTransPPO-Bi trainer.

The import is deferred so users who never select ``graphtransppo_bi`` do
not have to install ``torch_geometric`` to use the rest of the CLI.
"""

from __future__ import annotations


class GraphTransPPOBiAdapter:
    """Adapter boundary that avoids importing graph dependencies until needed."""

    start_strategy = "random_pool"
    action_representation = "multidiscrete_graph"
    algorithm = "graphtransppo_bi"

    @staticmethod
    def train(config: dict, experiment_name: str) -> None:
        from genmolrl.algorithms.graphtransppo_bi.train import train

        train(config, experiment_name)

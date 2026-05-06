"""Graph-transformer policy used by GraphTransRL.

This module is self-contained inside GenMolRL. The policy makes decisions from
molecular graph embeddings rather than fixed Morgan fingerprints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from rdkit import Chem

try:  # pragma: no cover - exercised in integration smoke when PyG is available.
    import torch_geometric.data as gd
    import torch_geometric.nn as gnn
    from torch_geometric.utils import add_self_loops
except Exception:  # pragma: no cover
    gd = None
    gnn = None
    add_self_loops = None


ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "other"]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def mlp(n_in: int, n_hid: int, n_out: int, n_layer: int, act=nn.LeakyReLU) -> nn.Sequential:
    dims = [n_in] + [n_hid] * n_layer + [n_out]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def _atom_features(atom: Chem.Atom) -> list[float]:
    symbol = atom.GetSymbol()
    symbol_idx = ATOM_TYPES.index(symbol) if symbol in ATOM_TYPES else ATOM_TYPES.index("other")
    one_hot = [0.0] * len(ATOM_TYPES)
    one_hot[symbol_idx] = 1.0
    return one_hot + [
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()) / 4.0,
        float(atom.GetDegree()) / 4.0,
    ]


def _bond_features(bond: Chem.Bond) -> list[float]:
    one_hot = [0.0] * len(BOND_TYPES)
    if bond.GetBondType() in BOND_TYPES:
        one_hot[BOND_TYPES.index(bond.GetBondType())] = 1.0
    return one_hot + [float(bond.GetIsConjugated()), float(bond.IsInRing())]


@dataclass(frozen=True)
class GraphFeatureSpec:
    node_dim: int = len(ATOM_TYPES) + 4
    edge_dim: int = len(BOND_TYPES) + 2
    cond_dim: int = 1


def smiles_to_data(smiles: str, *, device: torch.device | None = None):
    if gd is None:
        raise ImportError("torch_geometric is required for GraphTransRL.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    node_features = [_atom_features(atom) for atom in mol.GetAtoms()]
    if not node_features:
        node_features = [[0.0] * GraphFeatureSpec.node_dim]
    edge_index: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])
    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.zeros((0, GraphFeatureSpec.edge_dim), dtype=torch.float32)
    data = gd.Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
    )
    return data.to(device) if device is not None else data


def batch_from_smiles(smiles_batch: list[str], *, device: torch.device):
    return gd.Batch.from_data_list([smiles_to_data(smiles, device=device) for smiles in smiles_batch])


class GraphTransformer(nn.Module):
    """Small graph transformer for molecule-level action scoring."""

    def __init__(
        self,
        x_dim: int,
        e_dim: int,
        g_dim: int = 1,
        num_emb: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
    ):
        super().__init__()
        if gnn is None or add_self_loops is None:
            raise ImportError("torch_geometric is required for GraphTransRL.")
        self.num_layers = int(num_layers)
        self.x2h = mlp(x_dim, num_emb, num_emb, 2)
        self.e2h = mlp(e_dim, num_emb, num_emb, 2)
        self.c2h = mlp(max(1, g_dim), num_emb, num_emb, 2)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "gen": gnn.GENConv(num_emb, num_emb, num_layers=1, aggr="add", norm=None),
                        "attn": gnn.TransformerConv(num_emb * 2, num_emb, edge_dim=num_emb, heads=num_heads, concat=False),
                        "norm1": gnn.LayerNorm(num_emb, affine=False),
                        "ff": mlp(num_emb, num_emb * 4, num_emb, 1),
                        "norm2": gnn.LayerNorm(num_emb, affine=False),
                        "cscale": nn.Linear(num_emb, num_emb * 2),
                    }
                )
            )

    def forward(self, graph_batch, cond: torch.Tensor | None):
        x = self.x2h(graph_batch.x)
        edge_attr = self.e2h(graph_batch.edge_attr)
        cond_input = cond if cond is not None else torch.ones((graph_batch.num_graphs, 1), device=x.device)
        c = self.c2h(cond_input)

        num_nodes = graph_batch.x.shape[0]
        u = torch.arange(num_nodes, device=x.device)
        v = graph_batch.batch + num_nodes
        aug_edge_index = torch.cat([graph_batch.edge_index, torch.stack([u, v]), torch.stack([v, u])], dim=1)
        virtual_edges = torch.zeros((num_nodes * 2, edge_attr.shape[1]), device=x.device)
        if virtual_edges.numel():
            virtual_edges[:, 0] = 1.0
        aug_edge_attr = torch.cat([edge_attr, virtual_edges], dim=0)
        aug_edge_index, aug_edge_attr = add_self_loops(aug_edge_index, aug_edge_attr, fill_value="mean")
        aug_batch = torch.cat([graph_batch.batch, torch.arange(c.shape[0], device=x.device)], dim=0)
        h = torch.cat([x, c], dim=0)

        for layer in self.layers:
            h_norm = layer["norm1"](h, aug_batch)
            agg = layer["gen"](h_norm, aug_edge_index, aug_edge_attr)
            update = layer["attn"](torch.cat([h_norm, agg], dim=1), aug_edge_index, aug_edge_attr)
            scale_shift = layer["cscale"](c[aug_batch])
            scale, shift = scale_shift[:, : update.shape[1]], scale_shift[:, update.shape[1] :]
            h = h + update * scale + shift
            h = h + layer["ff"](layer["norm2"](h, aug_batch))

        node_embeddings = h[:-c.shape[0]]
        graph_embeddings = torch.cat([gnn.global_mean_pool(node_embeddings, graph_batch.batch), h[-c.shape[0] :]], dim=1)
        return node_embeddings, graph_embeddings


class GraphTransRLPolicy(nn.Module):
    """Graph-transformer policy over Stop + reaction-template actions.

    There is intentionally no AddFirstReactant head: the start molecule is supplied
    by the training/evaluation sampler, not learned by the policy.
    """

    def __init__(
        self,
        num_templates: int,
        *,
        num_emb: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
    ):
        super().__init__()
        spec = GraphFeatureSpec()
        self.backbone = GraphTransformer(
            spec.node_dim,
            spec.edge_dim,
            spec.cond_dim,
            num_emb=num_emb,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        graph_dim = num_emb * 2
        self.stop_head = mlp(graph_dim, num_emb, 1, 1)
        self.template_head = mlp(graph_dim, num_emb, num_templates, 1)

    def forward(self, graph_batch, cond: torch.Tensor | None = None) -> torch.Tensor:
        _, graph_embeddings = self.backbone(graph_batch, cond)
        return torch.cat([self.template_head(graph_embeddings), self.stop_head(graph_embeddings)], dim=-1)


__all__ = ["GraphTransRLPolicy", "batch_from_smiles", "smiles_to_data"]

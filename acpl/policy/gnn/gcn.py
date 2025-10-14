# acpl/policy/gnn/gcn.py
from __future__ import annotations

import torch
from torch import nn


def _make_undirected_with_self_loops(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Return an undirected, self-loop-augmented, coalesced edge_index (2, E').
    Assumes input edge_index has shape (2, E).
    """
    ei = edge_index.to(device=device, dtype=torch.long)
    src, dst = ei[0], ei[1]

    # Add reverse edges to ensure undirected
    rev = torch.stack([dst, src], dim=0)

    # Add self loops
    loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    loops = torch.stack([loops, loops], dim=0)

    full = torch.cat([ei, rev, loops], dim=1)

    # Coalesce (unique by pairs), keeping lexicographic order (dst-major is fine)
    keys = full[0] * num_nodes + full[1]  # (2,E) -> linearized keys
    uniq = torch.unique(keys, sorted=True)
    # Rebuild from uniq keys
    undirected = torch.stack([uniq // num_nodes, uniq % num_nodes], dim=0)
    return undirected


def _normalized_weights(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute symmetric GCN normalization weights:
        w_e = 1 / sqrt(deg[src] * deg[dst])
    where degrees are computed on the given edge_index (which should include self loops).
    """
    src, dst = edge_index[0], edge_index[1]
    deg = torch.bincount(dst, minlength=num_nodes).clamp(min=1)
    inv_sqrt = deg.to(torch.float32).pow(-0.5)
    return inv_sqrt[src] * inv_sqrt[dst]


class GCNLayer(nn.Module):
    """
    One GCN layer with sum aggregation and symmetric normalization:
        H = Ď^{-1/2} Â Ď^{-1/2} X W + b
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    @torch.no_grad()
    def _check(self, x: torch.Tensor, edge_index: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (N, F), got shape {tuple(x.shape)}")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, E)")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self._check(x, edge_index)
        device = x.device
        n = x.size(0)

        # Undirected + self loops + coalesce
        undirected = _make_undirected_with_self_loops(edge_index, n, device)

        # Transform then aggregate with normalized weights
        h = self.lin(x)  # (N, out_dim)
        src, dst = undirected[0], undirected[1]
        w = _normalized_weights(undirected, n).to(h.dtype)  # (E',)

        out = torch.zeros_like(h)
        out.index_add_(0, dst, h.index_select(0, src) * w.unsqueeze(1))
        return out


class TwoLayerGCN(nn.Module):
    """
    Minimal 2-layer GCN for node embeddings.

    Args
    ----
    in_dim : int
        Input feature size.
    hidden_dim : int
        Hidden/output feature size H.
    dropout : float
        Dropout after the first layer.
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (N, F)
            Node features.
        edge_index : torch.Tensor, shape (2, E)
            Graph edges (undirected or directed). Undirected & self loops are ensured internally.

        Returns
        -------
        z : torch.Tensor, shape (N, H)
            Node embeddings.
        """
        z = self.gcn1(x, edge_index)
        z = self.act(z)
        z = self.drop(z)
        z = self.gcn2(z, edge_index)
        return z

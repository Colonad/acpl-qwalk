# acpl/policy/gnn/graphsage.py
# Mean-pooling GraphSAGE encoder (Phase B3)
# SPDX-License-Identifier: MIT
"""
GraphSAGE (mean) encoder for ACPL-QWALK
=======================================

This module provides a permutation-equivariant GraphSAGE encoder with mean
aggregation, designed to feed node embeddings z_v into the temporal controller
of our adaptive coin policy. It is aligned with the theory report (§18) and the
Phase B3 checklist:

- Mean aggregation over neighbors (degree-robust)
- Nodewise update on [h_v || mean_N(v)] (Eq. 21–23 in the PDF)
- Optional edge features in the message MLP
- Residual connections when shapes match
- Layer/Graph normalization + dropout
- Deterministic degree handling (safe mean for deg=0 with self-loops toggle)
- Pure PyTorch; no hard dependency on torch_geometric
- Permutation-equivariant by construction (shared weights + set-mean)

Typical usage (inside policy builder):

    encoder = GraphSAGE(
        in_dim=F,
        hidden_dim=H,
        out_dim=H,
        num_layers=2,
        edge_dim=E_edge,          # None if no edge features
        norm="layer",             # {"layer","graph","none"}
        dropout=0.1,
        act="gelu",
        residual=True,
        add_self_loops=True,
    )
    Z = encoder(x, edge_index, edge_attr=edge_attr)   # -> [N, out_dim]

Inputs
------
- x:          [N, F]  float32 node features (degree, PE, coords, etc.)
- edge_index: [2, E]  int64 COO list of directed edges/arcs (u -> v).
                      For undirected simple graphs, include both directions.
- edge_attr:  [E, De] optional float32 edge features (orientation/phase/etc.)

Outputs
-------
- z:          [N, D] node embeddings (D = out_dim)

Determinism notes
-----------------
- Aggregation uses index_add and degree-based mean; no randomness inside.
- add_self_loops=True ensures deg(v)>=1 for stable means (can be disabled).

References: “GNN-Based Adaptive Coin Policy Learning for DTQWs” (PDF sections:
18.1–18.11) and Phase B3 checklist entry for graphsage.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

# ---------- small utils ----------


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "prelu":
        return nn.PReLU()
    raise ValueError(f"Unknown activation: {name}")


class GraphNorm(nn.Module):
    """
    GraphNorm (lightweight) — normalize per graph in a batch-less setting
    (single-graph usage). Equivalent to LayerNorm over feature channels but with
    an affine transform. If you later batch multiple graphs, prefer to call this
    per-graph or switch to LayerNorm.

    y = (x - mean) / sqrt(var + eps) * w + b
    """

    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C]
        mean = x.mean(dim=0, keepdim=True)
        var = (x - mean).pow(2).mean(dim=0, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight + self.bias
        return y


def _build_norm(kind: Literal["layer", "graph", "none"], dim: int) -> nn.Module:
    k = kind.lower()
    if k == "none":
        return nn.Identity()
    if k == "layer":
        return nn.LayerNorm(dim)
    if k == "graph":
        return GraphNorm(dim)
    raise ValueError(f"Unknown norm kind: {kind}")


def _maybe_cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat([a, b], dim=-1)


def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Add self loops (v->v) when not present. Keeps dtype/device of edge_index.
    """
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
    self_loops = torch.stack([self_loops, self_loops], dim=0)  # [2, N]
    return torch.cat([edge_index, self_loops], dim=1)


def _segment_mean(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute mean over incoming edges per target node with stable degree handling.
    src:   [E, D]
    index: [E] (dst indices)
    Return: [N, D]
    """
    N, D = num_nodes, src.size(-1)
    out = src.new_zeros((N, D))
    out.index_add_(0, index, src)
    deg = src.new_zeros((N,), dtype=torch.long)
    deg.index_add_(0, index, torch.ones_like(index, dtype=torch.long))
    # avoid divide-by-zero; will be handled by add_self_loops or clamped
    deg = deg.clamp_min(1).to(out.dtype)
    out = out / deg.unsqueeze(-1)
    return out


# ---------- GraphSAGE convolution (mean) ----------


class SAGEConvMean(nn.Module):
    """
    One GraphSAGE (mean) layer with optional edge feature conditioning.

    Messages:
        m_{u->v} = phi_msg( h_u , e_{uv} )
        (identity on h_u if edge_dim is None)
    Aggregate:
        m̄_v = mean_{u in N(v)} m_{u->v}
    Update:
        h'_v = act( Norm( W [ h_v || m̄_v ] + b ) )
        (residual added by caller if dims match)

    Shapes:
        h:        [N, Din]
        edge_idx: [2, E] (u, v)
        e_attr:   [E, De] or None
        out:      [N, Dout]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        edge_dim: int | None = None,
        act: str = "gelu",
        norm: Literal["layer", "graph", "none"] = "layer",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        # Message transform for neighbor states (and edge features).
        msg_in = in_dim + (edge_dim or 0)
        if edge_dim is None:
            # Identity message (no parameters)
            self.msg_mlp = nn.Identity()
            self._uses_edge = False
            self._msg_out_dim = in_dim
        else:
            self._uses_edge = True
            hidden = max(in_dim, out_dim)
            self.msg_mlp = nn.Sequential(
                nn.Linear(msg_in, hidden),
                _activation(act),
                nn.Linear(hidden, in_dim),
            )
            self._msg_out_dim = in_dim  # keep same dim as h_u for stable means

        # Update on concatenation [h_v || mean_m]
        upd_in = in_dim + self._msg_out_dim
        self.upd_lin = nn.Linear(upd_in, out_dim)
        self.norm = _build_norm(norm, out_dim)
        self.act = _activation(act)
        self.drop = nn.Dropout(dropout)

        # init
        nn.init.xavier_uniform_(self.upd_lin.weight)
        if self.upd_lin.bias is not None:
            nn.init.zeros_(self.upd_lin.bias)
        if isinstance(self.msg_mlp, nn.Sequential):
            for m in self.msg_mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        h: torch.Tensor,  # [N, Din]
        edge_index: torch.Tensor,  # [2, E] (u, v)
        edge_attr: torch.Tensor | None = None,  # [E, De]
        *,
        num_nodes: int | None = None,
    ) -> torch.Tensor:
        if h.dim() != 2:
            raise ValueError(f"h must be 2D [N, Din], got {h.shape}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must be [2, E], got {edge_index.shape}")
        if (self._uses_edge and edge_attr is None) or (
            edge_attr is not None and not self._uses_edge
        ):
            raise ValueError(
                "edge_attr usage mismatch: set edge_dim when constructing layer to use edge_attr."
            )

        N = h.size(0) if num_nodes is None else int(num_nodes)
        device = h.device
        u, v = edge_index  # sources, targets

        # Gather neighbor states
        h_u = h.index_select(0, u)  # [E, Din]
        if self._uses_edge:
            if edge_attr.dim() != 2 or edge_attr.size(0) != edge_index.size(1):
                raise ValueError(f"edge_attr must be [E, De], got {edge_attr.shape}")
            m_in = torch.cat([h_u, edge_attr], dim=-1)  # [E, Din+De]
            m = self.msg_mlp(m_in)  # [E, Din]
        else:
            m = h_u  # identity messages

        # Mean aggregate into each destination v
        mean_m = _segment_mean(m, v, N)  # [N, Din]

        # Update with concatenation
        upd_in = torch.cat([h, mean_m], dim=-1)  # [N, Din+Din]
        out = self.upd_lin(upd_in)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return out


# ---------- Stacked GraphSAGE encoder ----------


@dataclass
class GraphSAGEConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    num_layers: int = 2
    edge_dim: int | None = None
    norm: Literal["layer", "graph", "none"] = "layer"
    dropout: float = 0.1
    act: str = "gelu"
    residual: bool = True
    add_self_loops: bool = True


class GraphSAGE(nn.Module):
    """
    Stacked GraphSAGE (mean) encoder.

    L layers with mean aggregation and nodewise updates. Residual connections
    are added when the feature sizes match (Din == Dout for that block).
    Self-loops may be added to ensure non-empty neighborhoods (stable means).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        *,
        edge_dim: int | None = None,
        norm: Literal["layer", "graph", "none"] = "layer",
        dropout: float = 0.1,
        act: str = "gelu",
        residual: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = nn.ModuleList()
        for l in range(num_layers):
            in_d = dims[l]
            out_d = dims[l + 1]
            # Use edge features only in first k layers (here: all layers if provided)
            layers.append(
                SAGEConvMean(
                    in_dim=in_d,
                    out_dim=out_d,
                    edge_dim=edge_dim,
                    act=act,
                    norm=norm,
                    dropout=(
                        dropout if l < num_layers - 1 else 0.0
                    ),  # no dropout on final by default
                )
            )
        self.layers = layers
        self.residual = residual
        self.add_self_loops = add_self_loops

        # Fast flags
        self._uses_edge = edge_dim is not None

    def forward(
        self,
        x: torch.Tensor,  # [N, F]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor | None = None,  # [E, De]
    ) -> torch.Tensor:
        if self._uses_edge and edge_attr is None:
            raise ValueError("edge_attr required but was None (edge_dim set at init).")
        if (not self._uses_edge) and (edge_attr is not None):
            # Silently ignore extra edge_attr to be forgiving (or raise if you prefer).
            edge_attr = None

        N = x.size(0)
        ei = edge_index
        if self.add_self_loops:
            ei = _add_self_loops(ei, N)

            # If edge features are used, append zeros for self-loop attrs (neutral)
            if self._uses_edge:
                De = edge_attr.size(-1)
                zeros = edge_attr.new_zeros((N, De))
                edge_attr = torch.cat([edge_attr, zeros], dim=0)

        h = x
        for l, layer in enumerate(self.layers):
            h_next = layer(h, ei, edge_attr=edge_attr, num_nodes=N)
            if self.residual and h_next.shape == h.shape:
                h_next = h_next + h
            h = h_next
        return h


# ---------- convenience builder ----------


def build_graphsage(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    *,
    edge_dim: int | None = None,
    norm: Literal["layer", "graph", "none"] = "layer",
    dropout: float = 0.1,
    act: str = "gelu",
    residual: bool = True,
    add_self_loops: bool = True,
) -> GraphSAGE:
    """
    Convenience factory mirroring config files.
    """
    return GraphSAGE(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        edge_dim=edge_dim,
        norm=norm,
        dropout=dropout,
        act=act,
        residual=residual,
        add_self_loops=add_self_loops,
    )


__all__ = [
    "GraphSAGE",
    "GraphSAGEConfig",
    "build_graphsage",
    "SAGEConvMean",
]
# EOF

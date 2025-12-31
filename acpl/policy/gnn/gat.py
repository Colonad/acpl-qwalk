# acpl/policy/gnn/gat.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
import math
from .segment import segment_sum

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utilities (scatter/segment ops with no extra dependencies)
# ---------------------------------------------------------------------------


def _add_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Add i->i self edges if missing.
    edge_index: [2, E]
    """
    device = edge_index.device
    self_loops = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
    self_loops = self_loops.unsqueeze(0).repeat(2, 1)  # [[0..N-1],[0..N-1]]
    return torch.cat([edge_index, self_loops], dim=1)


def _segment_softmax(values: Tensor, seg_ids: Tensor) -> Tensor:
    """
    Softmax over segments (edges grouped by seg_ids). Stable, no torch_scatter dependency.
    values: [E, H] or [E] scores to normalize within each segment id.
    seg_ids: [E] long integer ids (e.g., destination node indices)

    Returns:
        softmax values of the same shape as `values`.
    """
    # Sort by segment id so we can do run-length operations
    order = torch.argsort(seg_ids)
    seg_ids_sorted = seg_ids[order]
    vals_sorted = values[order]

    # Compute per-segment max for numerical stability
    if vals_sorted.dim() == 1:
        # [E]
        max_buf = torch.full_like(vals_sorted, -float("inf"))
        max_buf.index_reduce_(0, seg_ids_sorted, vals_sorted, reduce="amax", include_self=False)
        # gather each edge's segment max
        seg_max = max_buf[seg_ids_sorted]
        exps = torch.exp(vals_sorted - seg_max)
        denom = torch.zeros_like(vals_sorted)
        denom.index_reduce_(0, seg_ids_sorted, exps, reduce="sum", include_self=False)
        denom = denom[seg_ids_sorted]
        out_sorted = exps / (denom + 1e-12)
    else:
        # [E, H]
        E, H = vals_sorted.shape
        max_buf = torch.full(
            (seg_ids_sorted.max().item() + 1, H),
            -float("inf"),
            device=vals_sorted.device,
            dtype=vals_sorted.dtype,
        )
        max_buf.index_reduce_(0, seg_ids_sorted, vals_sorted, reduce="amax", include_self=False)
        seg_max = max_buf[seg_ids_sorted]
        exps = torch.exp(vals_sorted - seg_max)
        denom = torch.zeros_like(max_buf)
        denom.index_reduce_(0, seg_ids_sorted, exps, reduce="sum", include_self=False)
        denom = denom[seg_ids_sorted]
        out_sorted = exps / (denom + 1e-12)

    # Invert permutation
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(order.numel(), device=order.device)
    return out_sorted[inv_order]


def _maybe_graph_norm(x: Tensor, batch: Tensor | None) -> Tensor:
    """
    Graph-wise feature normalization (mean/std per graph).
    If `batch` is None, we fall back to global normalization across nodes.
    x: [N, C]
    batch: [N] with graph ids in [0..B-1] or None
    """
    if batch is None:
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5)

    # Compute means per graph
    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    C = x.size(1)
    device, dtype = x.device, x.dtype

    # mean
    sum_buf = torch.zeros(B, C, device=device, dtype=dtype)
    sum_buf.index_add_(0, batch, x)
    count = torch.zeros(B, 1, device=device, dtype=dtype)
    count.index_add_(0, batch, torch.ones_like(x[:, :1]))
    mean = sum_buf / (count.clamp_min_(1.0))

    # variance
    xc = x - mean[batch]
    var_buf = torch.zeros(B, C, device=device, dtype=dtype)
    var_buf.index_add_(0, batch, xc * xc)
    var = var_buf / (count.clamp_min_(1.0))

    return xc / torch.sqrt(var + 1e-5)


def _activation(name: str, dim: int | None = None) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "gelu":
        return nn.GELU()
    if n == "prelu":
        return nn.PReLU(num_parameters=dim or 1, init=0.25)
    raise ValueError(f"Unsupported activation: {name}")


# ---------------------------------------------------------------------------
# GAT Convolution (multi-head, edge-aware, attention on incoming edges)
# ---------------------------------------------------------------------------


class GATConv(nn.Module):
    """
    Multi-head Graph Attention with optional edge features.

    Shapes
    ------
    x: [N, in_dim]
    edge_index: [2, E] with (src, dst)
    edge_attr (optional): [E, edge_dim]
    Returns: [N, heads*out_dim] if concat=True else [N, out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int = 4,
        edge_dim: int | None = None,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout_attn: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.heads = int(heads)
        self.edge_dim = edge_dim
        self.concat = bool(concat)
        self.negative_slope = float(negative_slope)
        self.dropout_attn = float(dropout_attn)
        self.add_self_loops = bool(add_self_loops)

        # Linear projection of node features to multi-head subspaces
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)

        # Attention vectors: per-head src & dst terms (size out_dim per head)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))

        # Optional edge attention term
        if edge_dim is not None and edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, heads * out_dim, bias=False)
            self.att_edge = nn.Parameter(torch.empty(heads, out_dim))
        else:
            self.lin_edge = None
            self.att_edge = None

        self.attn_drop = nn.Dropout(dropout_attn) if dropout_attn > 0 else nn.Identity()
        if bias:
            if self.concat:
                self.bias = nn.Parameter(torch.zeros(heads * out_dim))
            else:
                self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.att_src, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.att_dst, gain=math.sqrt(2))
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight, gain=math.sqrt(2))
        if self.att_edge is not None:
            nn.init.xavier_uniform_(self.att_edge, gain=math.sqrt(2))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        N = x.size(0)
        if self.add_self_loops:
            edge_index = _add_self_loops(edge_index, N)

            # If edge features are used, pad zeros for self-loops
            if edge_attr is not None and self.lin_edge is not None:
                E = edge_index.size(1)
                old_E = edge_attr.size(0)
                pad = E - old_E
                if pad > 0:
                    zeros = torch.zeros(
                        pad, edge_attr.size(1), device=edge_attr.device, dtype=edge_attr.dtype
                    )
                    edge_attr = torch.cat([edge_attr, zeros], dim=0)

        src, dst = edge_index[0], edge_index[1]

        # Project node (and optional edge) features
        x_proj = self.lin(x).view(N, self.heads, self.out_dim)  # [N, H, D]
        x_src = x_proj[src]  # [E, H, D]
        x_dst = x_proj[dst]  # [E, H, D]

        # Compute unnormalized attention logits e_ij per edge/head
        # e_ij = LeakyReLU(a_src·x_i + a_dst·x_j (+ a_edge·e_ij))
        att = (x_src * self.att_src).sum(-1) + (x_dst * self.att_dst).sum(-1)  # [E, H]
        if self.lin_edge is not None and edge_attr is not None:
            e_proj = self.lin_edge(edge_attr).view(-1, self.heads, self.out_dim)  # [E, H, D]
            att = att + (e_proj * self.att_edge).sum(-1)  # [E, H]

        att = F.leaky_relu(att, negative_slope=self.negative_slope)

        # Normalize over incoming edges for each destination node
        # (i.e., neighbors j -> i aggregate into i; softmax over edges with same dst)
        alpha = _segment_softmax(att, dst)  # [E, H]
        alpha = self.attn_drop(alpha)

        # Message passing: out_i = sum_{j in N(i)} alpha_ji * x_src(j)
        m = x_src * alpha.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros_like(x_proj)
        out = segment_sum(m, dst, N)  # sum over incoming edges grouped by dst

        # Concat or average heads
        if self.concat:
            out = out.reshape(N, self.heads * self.out_dim)  # [N, H*D]
        else:
            out = out.mean(dim=1)  # [N, D]

        if self.bias is not None:
            out = out + self.bias

        return out


# ---------------------------------------------------------------------------
# GAT block and full encoder
# ---------------------------------------------------------------------------


class _Norm(nn.Module):
    def __init__(self, dim: int, kind: str):
        super().__init__()
        k = kind.lower()
        self.kind = k
        if k == "layer":
            self.norm = nn.LayerNorm(dim)
        elif k in ("graph", "none"):
            self.norm = None
        else:
            raise ValueError(f"Unknown norm kind: {kind}")

    def forward(self, x: Tensor, batch: Tensor | None) -> Tensor:
        if self.kind == "layer":
            return self.norm(x)
        if self.kind == "graph":
            return _maybe_graph_norm(x, batch)
        return x  # "none"


class GATBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int,
        edge_dim: int | None,
        concat: bool,
        act: str,
        norm: str,
        dropout: float,
        residual: bool,
        add_self_loops: bool,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.gat = GATConv(
            in_dim=in_dim,
            out_dim=out_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=concat,
            dropout_attn=attention_dropout,
            add_self_loops=add_self_loops,
            bias=True,
        )
        self.concat = concat
        self.out_dim = out_dim * heads if concat else out_dim

        self.act = _activation(act, dim=self.out_dim)
        self.norm = _Norm(self.out_dim, norm)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.use_res = bool(residual)
        if self.use_res:
            if in_dim != self.out_dim:
                self.res_proj = nn.Linear(in_dim, self.out_dim, bias=False)
            else:
                self.res_proj = nn.Identity()

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, batch: Tensor | None
    ) -> Tensor:
        y = self.gat(x, edge_index, edge_attr)  # [N, C_out]
        if self.use_res:
            x = self.res_proj(x) + y
        else:
            x = y
        x = self.norm(x, batch)
        x = self.act(x)
        x = self.drop(x)
        return x


@dataclass
class GATConfig:
    in_dim: int
    out_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    edge_dim: int | None = None
    norm: str = "layer"  # "layer" | "graph" | "none"
    dropout: float = 0.1
    attention_dropout: float = 0.0
    act: str = "gelu"  # "relu" | "gelu" | "prelu"
    residual: bool = True
    add_self_loops: bool = True
    concat: bool = True  # concat heads in hidden layers
    final_concat: bool = False  # set False to average heads on last layer


class GAT(nn.Module):
    """
    Full multi-layer GAT encoder.

    Forward signature matches GraphSAGE encoder:
        forward(x, edge_index, edge_attr=None, batch=None) -> Tensor [N, out_dim]
    """

    def __init__(self, cfg: GATConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers = []
        in_dim = cfg.in_dim
        for li in range(cfg.num_layers):
            is_last = li == (cfg.num_layers - 1)
            out_dim = cfg.out_dim if is_last else cfg.hidden_dim
            concat = cfg.concat if not is_last else cfg.final_concat

            block = GATBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                heads=cfg.heads,
                edge_dim=cfg.edge_dim,
                concat=concat,
                act=cfg.act,
                norm=cfg.norm,
                dropout=cfg.dropout if not is_last else 0.0,  # last layer: keep features crisp
                residual=cfg.residual,
                add_self_loops=cfg.add_self_loops,
                attention_dropout=cfg.attention_dropout,
            )
            layers.append(block)
            in_dim = block.out_dim  # next input dim is current output dim

        self.layers = nn.ModuleList(layers)

    @property
    def output_dim(self) -> int:
        last: GATBlock = self.layers[-1]
        return last.out_dim

    def reset_parameters(self) -> None:
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                try:
                    m.reset_parameters()  # type: ignore[attr-defined]
                except Exception:
                    pass

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : [N, in_dim]
        edge_index : [2, E]
        edge_attr : [E, edge_dim] or None
        batch : [N] or None (graph ids for graph-wise norm)

        Returns
        -------
        Tensor : [N, output_dim]
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
        return x


# ---------------------------------------------------------------------------
# Builder (used by encoder_factory)
# ---------------------------------------------------------------------------


def build_gat(
    *,
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    heads: int = 4,
    edge_dim: int | None = None,
    norm: str = "layer",
    dropout: float = 0.1,
    attention_dropout: float = 0.0,
    act: str = "gelu",
    residual: bool = True,
    add_self_loops: bool = True,
    concat: bool = True,
    final_concat: bool = False,
) -> GAT:
    """
    Convenience builder mirroring GATConfig fields.
    """
    cfg = GATConfig(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        edge_dim=edge_dim,
        norm=norm,
        dropout=dropout,
        attention_dropout=attention_dropout,
        act=act,
        residual=residual,
        add_self_loops=add_self_loops,
        concat=concat,
        final_concat=final_concat,
    )
    return GAT(cfg)

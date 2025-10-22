# acpl/policy/gnn/gin.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

# -----------------------------------------------------------------------------
# Small utilities (kept consistent with our SAGE/GAT modules)
# -----------------------------------------------------------------------------


def _activation(name: str, dim: int | None = None) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "gelu":
        return nn.GELU()
    if n == "prelu":
        return nn.PReLU(num_parameters=dim or 1, init=0.25)
    if n in ("", "identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def _maybe_graph_norm(x: Tensor, batch: Tensor | None) -> Tensor:
    """
    Graph-wise feature normalization (mean/std per graph).
    If `batch` is None, fall back to global normalization across nodes.
    x: [N, C], batch: [N] with graph ids in [0..B-1] or None.
    """
    if batch is None:
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5)

    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    C = x.size(1)
    device, dtype = x.device, x.dtype

    # mean
    sum_buf = torch.zeros(B, C, device=device, dtype=dtype)
    sum_buf.index_add_(0, batch, x)
    count = torch.zeros(B, 1, device=device, dtype=dtype)
    count.index_add_(0, batch, torch.ones_like(x[:, :1]))
    mean = sum_buf / count.clamp_min_(1.0)

    # var
    xc = x - mean[batch]
    var_buf = torch.zeros(B, C, device=device, dtype=dtype)
    var_buf.index_add_(0, batch, xc * xc)
    var = var_buf / count.clamp_min_(1.0)

    return xc / torch.sqrt(var + 1e-5)


def _add_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Add i->i self edges if missing. edge_index: [2, E]
    """
    device = edge_index.device
    i = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
    self_loops = torch.stack([i, i], dim=0)  # [2, N]
    return torch.cat([edge_index, self_loops], dim=1)


def _mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    *,
    act: str = "relu",
    norm: str = "layer",
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Build a robust two+ layer MLP with per-layer norm/act/dropout.
    """
    dims = [in_dim, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        # On last layer, we usually keep it linear; keep act/norm optional but default off
        is_last = i == len(dims) - 2
        if not is_last:
            if norm == "layer":
                layers.append(nn.LayerNorm(dims[i + 1]))
            elif norm in ("graph", "none"):
                pass  # handled outside or no-op
            else:
                raise ValueError(f"Unknown MLP norm kind: {norm}")
            layers.append(_activation(act, dim=dims[i + 1]))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# GIN / GINE Convolution
# -----------------------------------------------------------------------------


class GINConv(nn.Module):
    """
    Graph Isomorphism Network (GIN) convolution with optional edge features (GINE).

    Aggregate messages by summation, then apply an MLP:
        h_i' = MLP( (1 + eps) * h_i  +  SUM_{j in N(i)} ψ(h_j, e_{ji}) )

    Where ψ(h_j, e_{ji}) = h_j for vanilla GIN,
    or ψ(h_j, e_{ji}) = h_j + φ(e_{ji}) for GINE with an edge encoder φ.

    Args
    ----
    in_dim:         input node feature dim
    out_dim:        output node feature dim
    eps_init:       initial value of learnable epsilon (1 + eps scales self features)
    eps_learnable:  if True, epsilon is a learnable Parameter
    edge_dim:       optional edge feature dimension; if set, activates GINE-style update
    edge_embed_dim: hidden dim for edge encoder (defaults to in_dim if None)
    add_self_loops: whether to add self-loops to edge_index internally
    mlp_hidden:     hidden dims for the MLP head
    mlp_act/norm/dropout: MLP building knobs
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        eps_init: float = 0.0,
        eps_learnable: bool = True,
        edge_dim: int | None = None,
        edge_embed_dim: int | None = None,
        add_self_loops: bool = True,
        mlp_hidden: Sequence[int] = (64, 64),
        mlp_act: str = "relu",
        mlp_norm: str = "layer",
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.edge_dim = edge_dim
        self.add_self_loops = bool(add_self_loops)

        # Learnable epsilon (or fixed buffer)
        if eps_learnable:
            self.eps = nn.Parameter(torch.tensor(float(eps_init)))
        else:
            self.register_buffer("eps", torch.tensor(float(eps_init)))

        # Optional edge encoder (GINE)
        if edge_dim is not None and edge_dim > 0:
            edim = int(edge_embed_dim) if edge_embed_dim is not None else self.in_dim
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, edim, bias=True),
                _activation(mlp_act, dim=edim),
                nn.Linear(edim, self.in_dim, bias=True),
            )
        else:
            self.edge_encoder = None

        # Post-aggregation MLP
        # Input to MLP is in_dim (self+neigh sum), output is out_dim.
        self.mlp = _mlp(
            in_dim=self.in_dim,
            hidden_dims=mlp_hidden,
            out_dim=self.out_dim,
            act=mlp_act,
            norm=mlp_norm,
            dropout=mlp_dropout,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if isinstance(self.eps, nn.Parameter):
            with torch.no_grad():
                self.eps.fill_(self.eps.detach())
        # Init MLP linears
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Edge encoder
        if self.edge_encoder is not None:
            for m in self.edge_encoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """
        x: [N, C_in]
        edge_index: [2, E] (src, dst)
        edge_attr: [E, edge_dim] or None
        returns: [N, C_out]
        """
        N, Cin = x.shape
        assert Cin == self.in_dim, f"Expected node dim {self.in_dim}, got {Cin}"
        if self.add_self_loops:
            edge_index = _add_self_loops(edge_index, N)
            if edge_attr is not None and self.edge_encoder is not None:
                # Pad zeros for self-loop edge features
                E_new = edge_index.size(1)
                pad = E_new - edge_attr.size(0)
                if pad > 0:
                    zeros = torch.zeros(
                        pad, edge_attr.size(1), device=edge_attr.device, dtype=edge_attr.dtype
                    )
                    edge_attr = torch.cat([edge_attr, zeros], dim=0)

        src, dst = edge_index[0], edge_index[1]

        # Vanilla message: m_ji = x_j
        msg = x[src]  # [E, C_in]
        # Edge-enhanced (GINE) message if requested
        if self.edge_encoder is not None and edge_attr is not None:
            msg = msg + self.edge_encoder(edge_attr)  # [E, C_in]

        # Sum aggregation of incoming messages per destination node
        agg = torch.zeros_like(x)  # [N, C_in]
        agg.index_add_(0, dst, msg)

        # Combine self and neighborhood with learnable epsilon
        out = (1.0 + self.eps) * x + agg  # [N, C_in]

        # Apply MLP head
        out = self.mlp(out)  # [N, C_out]
        return out


# -----------------------------------------------------------------------------
# GIN Block and full encoder
# -----------------------------------------------------------------------------


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
        return x


class GINBlock(nn.Module):
    """
    A residual GIN block:
        y = GINConv(x)
        y = Residual(x, y)
        y = Norm(y)
        y = Act(y)
        y = Dropout(y)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        eps_init: float,
        eps_learnable: bool,
        edge_dim: int | None,
        edge_embed_dim: int | None,
        add_self_loops: bool,
        mlp_hidden: Sequence[int],
        mlp_act: str,
        mlp_norm: str,
        mlp_dropout: float,
        block_norm: str,
        block_act: str,
        block_dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()
        self.conv = GINConv(
            in_dim=in_dim,
            out_dim=out_dim,
            eps_init=eps_init,
            eps_learnable=eps_learnable,
            edge_dim=edge_dim,
            edge_embed_dim=edge_embed_dim,
            add_self_loops=add_self_loops,
            mlp_hidden=mlp_hidden,
            mlp_act=mlp_act,
            mlp_norm=mlp_norm,
            mlp_dropout=mlp_dropout,
        )

        self.norm = _Norm(out_dim, block_norm)
        self.act = _activation(block_act, dim=out_dim)
        self.drop = nn.Dropout(block_dropout) if block_dropout > 0 else nn.Identity()

        self.use_res = bool(residual)
        if self.use_res:
            if in_dim != out_dim:
                self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_proj = nn.Identity()

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None, batch: Tensor | None
    ) -> Tensor:
        y = self.conv(x, edge_index, edge_attr)  # [N, C_out]
        if self.use_res:
            x = self.res_proj(x) + y
        else:
            x = y
        x = self.norm(x, batch)
        x = self.act(x)
        x = self.drop(x)
        return x


# -----------------------------------------------------------------------------
# Public config + model
# -----------------------------------------------------------------------------


@dataclass
class GINConfig:
    in_dim: int
    out_dim: int
    hidden_dim: int = 128
    num_layers: int = 3

    # GIN/GINE specifics
    eps_init: float = 0.0
    eps_learnable: bool = True
    edge_dim: int | None = None
    edge_embed_dim: int | None = None
    add_self_loops: bool = True

    # MLP inside each GINConv
    mlp_hidden: Sequence[int] = (64, 64)
    mlp_act: str = "relu"  # inside MLP
    mlp_norm: str = "layer"  # inside MLP ("layer"|"graph"|"none")
    mlp_dropout: float = 0.0

    # Block-level wrappers (post-conv)
    norm: str = "layer"  # "layer" | "graph" | "none"
    act: str = "gelu"  # outer activation after block norm
    dropout: float = 0.1
    residual: bool = True


class GIN(nn.Module):
    """
    Multi-layer GIN / GINE encoder.

    Forward signature:
        forward(x, edge_index, edge_attr=None, batch=None) -> Tensor [N, out_dim]
    """

    def __init__(self, cfg: GINConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers = []
        in_dim = cfg.in_dim
        for li in range(cfg.num_layers):
            is_last = li == (cfg.num_layers - 1)
            out_dim = cfg.out_dim if is_last else cfg.hidden_dim

            block = GINBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                eps_init=cfg.eps_init,
                eps_learnable=cfg.eps_learnable,
                edge_dim=cfg.edge_dim,
                edge_embed_dim=cfg.edge_embed_dim,
                add_self_loops=cfg.add_self_loops,
                mlp_hidden=cfg.mlp_hidden,
                mlp_act=cfg.mlp_act,
                mlp_norm=cfg.mlp_norm,
                mlp_dropout=cfg.mlp_dropout,
                block_norm=cfg.norm,
                block_act=(cfg.act if not is_last else "identity"),  # keep final layer crisp
                block_dropout=(cfg.dropout if not is_last else 0.0),
                residual=cfg.residual,
            )
            layers.append(block)
            in_dim = out_dim

        self.layers = nn.ModuleList(layers)

    @property
    def output_dim(self) -> int:
        return self.layers[-1].conv.out_dim  # type: ignore[attr-defined]

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
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
        return x


# -----------------------------------------------------------------------------
# Convenience builder (for encoder factory / YAML configs)
# -----------------------------------------------------------------------------


def build_gin(
    *,
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    eps_init: float = 0.0,
    eps_learnable: bool = True,
    edge_dim: int | None = None,
    edge_embed_dim: int | None = None,
    add_self_loops: bool = True,
    mlp_hidden: Sequence[int] = (64, 64),
    mlp_act: str = "relu",
    mlp_norm: str = "layer",
    mlp_dropout: float = 0.0,
    norm: str = "layer",
    act: str = "gelu",
    dropout: float = 0.1,
    residual: bool = True,
) -> GIN:
    cfg = GINConfig(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        eps_init=eps_init,
        eps_learnable=eps_learnable,
        edge_dim=edge_dim,
        edge_embed_dim=edge_embed_dim,
        add_self_loops=add_self_loops,
        mlp_hidden=mlp_hidden,
        mlp_act=mlp_act,
        mlp_norm=mlp_norm,
        mlp_dropout=mlp_dropout,
        norm=norm,
        act=act,
        dropout=dropout,
        residual=residual,
    )
    return GIN(cfg)

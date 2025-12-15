# acpl/policy/gnn/gcn.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from contextlib import nullcontext

# AMP autocast — supports new (torch.amp) and old (torch.cuda.amp) APIs
try:
    # New-style API: torch.amp.autocast(device_type=..., enabled=...)
    from torch.amp import autocast as _autocast  # type: ignore[attr-defined]

    def _autocast_ctx(device_type: str, enabled: bool = True):
        return _autocast(device_type=device_type, enabled=enabled)

except Exception:
    try:
        # Older API: torch.cuda.amp.autocast(enabled=...) (CUDA-only)
        from torch.cuda.amp import autocast as _cuda_autocast  # type: ignore

        def _autocast_ctx(device_type: str, enabled: bool = True):
            return _cuda_autocast(enabled=enabled) if device_type == "cuda" else nullcontext()

    except Exception:
        def _autocast_ctx(device_type: str, enabled: bool = True):
            return nullcontext()



# --- torch.compile / Dynamo safety -----------------------------------------
# Inductor can crash if a compiled segment receives a sparse tensor input.
# We keep sparse adjacency build+spmm in eager by disabling compilation for _spmm.
try:
    # Newer PyTorch
    from torch.compiler import disable as _compile_disable  # type: ignore
except Exception:
    try:
        import torch._dynamo as _dynamo  # type: ignore
        _compile_disable = _dynamo.disable
    except Exception:
        def _compile_disable(fn):  # type: ignore
            return fn
# ---------------------------------------------------------------------------



# ----------------------------- helpers (runtime-only) ----------------------------- #


def _activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "prelu":
        return nn.PReLU()
    if n == "gelu":
        return nn.GELU()
    if n == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation '{name}'")


def _add_self_loops(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    self_loop_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (edge_index_with_loops, edge_weight_with_loops).
    """
    device = edge_index.device
    dtype_ei = edge_index.dtype
    diag = torch.arange(num_nodes, device=device, dtype=dtype_ei)
    self_edges = torch.stack([diag, diag], dim=0)  # (2, N)

    if edge_weight is None:
        ew = torch.ones(edge_index.shape[1], device=device, dtype=torch.float32)
    else:
        ew = edge_weight.to(torch.float32)

    self_w = torch.full((num_nodes,), float(self_loop_weight), device=device, dtype=ew.dtype)
    ei = torch.cat([edge_index, self_edges], dim=1)
    ew = torch.cat([ew, self_w], dim=0)
    return ei, ew


def _normalize_renorm(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalized weights for Â = D^{-1/2}(A+I)D^{-1/2}.
    """
    ei, ew = _add_self_loops(edge_index, edge_weight, num_nodes, self_loop_weight=1.0)
    row = ei[0]
    col = ei[1]

    deg = torch.bincount(row, weights=ew, minlength=num_nodes).to(torch.float32)
    deg_inv_sqrt = (deg + eps).pow(-0.5)

    norm_w = ew * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return ei, norm_w


@_compile_disable
def _spmm(ei: torch.Tensor, ew: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Sparse COO matmul: (N×N)_sparse @ (N×F)_dense.

    Important: torch.sparse.mm on CUDA does NOT support float16,
    and autocast may try to run it in half. We therefore:
      - upcast inputs to float32,
      - disable autocast just for this op,
      - cast back to the original dtype of x afterwards.
    """
    n = x.size(0)
    device = x.device
    orig_dtype = x.dtype

    # Always compute in float32 for the sparse matmul
    x32 = x.to(torch.float32)
    ew32 = ew.to(torch.float32)

    A = torch.sparse_coo_tensor(ei, ew32, size=(n, n), device=device).coalesce()

    if device.type == "cuda":
        # Run sparse.mm outside autocast to prevent it from going to half
        with _autocast_ctx("cuda", enabled=False):
            out32 = torch.sparse.mm(A, x32)
    else:
        out32 = torch.sparse.mm(A, x32)


    # Cast back to the original dtype (fp16/bf16/etc.) if needed
    if orig_dtype != torch.float32:
        return out32.to(orig_dtype)
    return out32



# --------------------------------- layer & model --------------------------------- #


class _GCNLayer(nn.Module):
    """
    One renormalized GCN layer:
        H^{l+1} = Norm( act( (Â H^l) W ) ) + residual
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        layernorm: bool = True,
        residual: bool = True,
        dropedge: float = 0.0,
    ):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
        self.act = _activation(activation)
        self.do = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(out_dim) if layernorm else nn.Identity()
        self.residual = bool(residual and in_dim == out_dim)
        self.dropedge = float(dropedge)

        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(
        self,
        x: torch.Tensor,  # (N, Din)
        edge_index: torch.Tensor,  # (2, E)
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = x.size(0)
        ei, ew = _normalize_renorm(edge_index, edge_weight, N)

        # DropEdge (only non-self edges)
        if self.training and self.dropedge > 0.0:
            r, c = ei[0], ei[1]
            is_self = r == c
            keep = torch.ones_like(ew, dtype=torch.bool)
            nz = (~is_self).nonzero(as_tuple=False).flatten()
            if nz.numel() > 0:
                rnd = torch.rand(nz.numel(), device=ew.device)
                keep[nz] = rnd > self.dropedge
            ei = ei[:, keep]
            ew = ew[keep]

        h = _spmm(ei, ew, x)  # (N, Din)
        y = self.lin(h)  # (N, Dout)
        y = self.act(y)
        y = self.do(y)
        y = self.norm(y)
        if self.residual:
            y = y + x
        return y


@dataclass
class GCNConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    activation: str = "gelu"
    dropout: float = 0.1
    layernorm: bool = True
    residual: bool = True
    dropedge: float = 0.0


class GCN(nn.Module):
    """
    Two-layer GCN encoder (renormalized adjacency, sum aggregation).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        layernorm: bool = True,
        residual: bool = True,
        dropedge: float = 0.0,
    ):
        super().__init__()
        self.cfg = GCNConfig(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            layernorm=layernorm,
            residual=residual,
            dropedge=dropedge,
        )
        self.gcn1 = _GCNLayer(
            in_dim,
            hidden_dim,
            activation=activation,
            dropout=dropout,
            layernorm=layernorm,
            residual=False,
            dropedge=dropedge,
        )
        self.gcn2 = _GCNLayer(
            hidden_dim,
            out_dim,
            activation=activation,
            dropout=dropout,
            layernorm=layernorm,
            residual=False,
            dropedge=dropedge,
        )
        self.out_norm = nn.LayerNorm(out_dim) if layernorm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,  # (N, Fin)
        edge_index: torch.Tensor,  # (2, E)
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.gcn1(x, edge_index, edge_weight)
        h = self.gcn2(h, edge_index, edge_weight)
        return self.out_norm(h)

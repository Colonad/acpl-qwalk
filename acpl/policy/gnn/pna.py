# acpl/policy/gnn/pna.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

# -----------------------------------------------------------------------------
# Shared utilities (kept consistent with our other Phase B3 GNN modules)
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


# -----------------------------------------------------------------------------
# Segment reductions (sum/mean/std/min/max) without external deps
# -----------------------------------------------------------------------------


def _segment_sum(src: Tensor, index: Tensor, n_nodes: int) -> Tensor:
    out = torch.zeros(n_nodes, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def _segment_count(index: Tensor, n_nodes: int, *, device=None, dtype=None) -> Tensor:
    device = device if device is not None else index.device
    dtype = dtype if dtype is not None else torch.float32
    ones = torch.ones(index.size(0), 1, device=device, dtype=dtype)
    out = torch.zeros(n_nodes, 1, device=device, dtype=dtype)
    out.index_add_(0, index, ones)
    return out


def _segment_mean(src: Tensor, index: Tensor, n_nodes: int) -> Tensor:
    s = _segment_sum(src, index, n_nodes)
    c = _segment_count(index, n_nodes, device=src.device, dtype=src.dtype)
    return s / c.clamp_min_(1.0)


def _segment_var(src: Tensor, index: Tensor, n_nodes: int) -> Tensor:
    # Var = E[x^2] - (E[x])^2
    s = _segment_sum(src, index, n_nodes)
    sq = _segment_sum(src * src, index, n_nodes)
    c = _segment_count(index, n_nodes, device=src.device, dtype=src.dtype)
    mean = s / c.clamp_min_(1.0)
    mean_sq = sq / c.clamp_min_(1.0)
    var = (mean_sq - mean * mean).clamp_min_(0.0)
    return var


def _segment_min(src: Tensor, index: Tensor, n_nodes: int) -> Tensor:
    # Use scatter_reduce_ if available (PyTorch >= 2.0), otherwise fallback.
    if hasattr(torch.Tensor, "scatter_reduce_"):
        out = torch.full((n_nodes, src.size(-1)), float("inf"), device=src.device, dtype=src.dtype)
        # Create expanded index for broadcasting along feature dim
        idx = index.view(-1, 1).expand(-1, src.size(-1))
        out.scatter_reduce_(0, idx, src, reduce="amin")
        return out
    # Fallback: loop over unique indices (OK for moderate graphs used in tests)
    out = torch.full((n_nodes, src.size(-1)), float("inf"), device=src.device, dtype=src.dtype)
    uniq = torch.unique(index)
    for u in uniq:
        m = index == u
        out[int(u.item())] = src[m].amin(dim=0)
    return out


def _segment_max(src: Tensor, index: Tensor, n_nodes: int) -> Tensor:
    if hasattr(torch.Tensor, "scatter_reduce_"):
        out = torch.full((n_nodes, src.size(-1)), -float("inf"), device=src.device, dtype=src.dtype)
        idx = index.view(-1, 1).expand(-1, src.size(-1))
        out.scatter_reduce_(0, idx, src, reduce="amax")
        return out
    out = torch.full((n_nodes, src.size(-1)), -float("inf"), device=src.device, dtype=src.dtype)
    uniq = torch.unique(index)
    for u in uniq:
        m = index == u
        out[int(u.item())] = src[m].amax(dim=0)
    return out


# -----------------------------------------------------------------------------
# PNA Convolution (multi-aggregators + degree-aware scalers + edge encoder)
# -----------------------------------------------------------------------------


class PNAConv(nn.Module):
    """
    Principal Neighbourhood Aggregation (PNA) layer.

    Core idea:
        1) Project node features to a hidden space.
        2) Compute messages from neighbors (optionally edge-conditioned).
        3) Aggregate with multiple reducers (e.g., mean/sum/std/max/min).
        4) Apply degree-aware scalers (identity/amplification/attenuation/linear).
        5) Concatenate all aggregator×scaler outputs and mix to out_dim.

    Notation:
        x: [N, C_in], edge_index: [2, E] (src, dst), edge_attr: [E, E_dim] or None
        out: [N, C_out]

    Args
    ----
    in_dim:            input node feature dimension
    out_dim:           output node feature dimension
    hidden_dim:        hidden projection dimension used for messages/aggregation
    aggregators:       list of aggregators: {"sum","mean","std","min","max"}
    scalers:           list of scalers: {"identity","amplification","attenuation","linear"}
    deg_histogram:     histogram over node degrees (1D, length >= max_degree+1). Used
                       to compute reference averages (mean degree, mean log-degree).
    edge_dim:          optional edge feature dim to enable edge-conditioned messages
    edge_embed_dim:    hidden dim for edge encoder (defaults to hidden_dim if None)
    add_self_loops:    whether to add self loops internally (recommended True)
    msg_dropout:       dropout applied to pre-aggregation messages
    mix_act/norm/dropout: mixing head MLP controls (outside block wrappers)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int = 128,
        aggregators: Sequence[str] = ("mean", "sum", "std"),
        scalers: Sequence[str] = ("identity", "amplification", "attenuation"),
        deg_histogram: Tensor | Sequence[float] | Sequence[int] | None = None,
        edge_dim: int | None = None,
        edge_embed_dim: int | None = None,
        add_self_loops: bool = True,
        msg_dropout: float = 0.0,
        mix_act: str = "relu",
        mix_norm: str = "layer",
        mix_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.add_self_loops = bool(add_self_loops)

        # Aggregators + scalers
        self.aggregators = [a.lower() for a in aggregators]
        self.scalers = [s.lower() for s in scalers]
        for a in self.aggregators:
            if a not in {"sum", "mean", "std", "min", "max"}:
                raise ValueError(f"Unsupported aggregator: {a}")
        for s in self.scalers:
            if s not in {"identity", "amplification", "attenuation", "linear"}:
                raise ValueError(f"Unsupported scaler: {s}")

        # Degree priors (used by scalers). Expect a histogram over integer degrees.
        if deg_histogram is None:
            # Robust default to avoid div-by-zero: pretend mean degree ≈ 1
            self.register_buffer("deg_hist", torch.tensor([1.0, 1.0]))
        else:
            dh = torch.as_tensor(deg_histogram, dtype=torch.float32)
            if dh.dim() != 1 or dh.numel() < 2:
                raise ValueError("deg_histogram must be a 1D tensor/seq with length >= 2.")
            self.register_buffer("deg_hist", dh)

        # Pre-projection for node features prior to messaging
        self.pre = nn.Linear(self.in_dim, self.hidden_dim, bias=True)

        # Optional edge encoder (edge-conditioned messages)
        if edge_dim is not None and edge_dim > 0:
            edim = int(edge_embed_dim) if edge_embed_dim is not None else self.hidden_dim
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, edim, bias=True),
                _activation("relu", dim=edim),
                nn.Linear(edim, self.hidden_dim, bias=True),
            )
        else:
            self.edge_encoder = None

        # Mixer: concatenated aggregators×scalers → out_dim
        cat_dim = self.hidden_dim * (len(self.aggregators) * len(self.scalers))
        self.mix = nn.Sequential(
            nn.Linear(cat_dim, out_dim, bias=True),
            (nn.LayerNorm(out_dim) if mix_norm == "layer" else nn.Identity()),
            _activation(mix_act, dim=out_dim),
            (nn.Dropout(mix_dropout) if mix_dropout > 0 else nn.Identity()),
        )

        self.msg_drop = nn.Dropout(msg_dropout) if msg_dropout > 0 else nn.Identity()

        self.reset_parameters()

    # ---- degree statistics for scalers ------------------------------------- #

    @property
    def _avg_deg(self) -> float:
        hist = self.deg_hist  # [D_max+1]
        idx = torch.arange(hist.numel(), device=hist.device, dtype=hist.dtype)
        total = hist.sum().clamp_min_(1.0)
        return float((idx * hist).sum() / total)

    @property
    def _avg_log_deg(self) -> float:
        hist = self.deg_hist
        idx = torch.arange(hist.numel(), device=hist.device, dtype=hist.dtype)
        # +1 per PNA to avoid log(0)
        total = hist.sum().clamp_min_(1.0)
        return float((hist * (idx + 1).log()).sum() / total)

    def _scaler(self, name: str, deg: Tensor) -> Tensor:
        """
        Return per-node scaling factors given in-degree tensor deg: [N, 1] (float).
        """
        if name == "identity":
            return torch.ones_like(deg)
        if name == "linear":
            return (deg / max(self._avg_deg, 1e-6)).clamp_min(1e-6)
        if name == "amplification":
            # log-degree ratio ≥ 1 when deg >= avg (amplifies high degree)
            num = (deg + 1.0).log()
            return (num / max(self._avg_log_deg, 1e-6)).clamp_min(1e-6)
        if name == "attenuation":
            # inverse of amplification (down-weight high degree)
            num = (deg + 1.0).log()
            return max(self._avg_log_deg, 1e-6) / num.clamp_min(1e-6)
        raise ValueError(f"Unknown scaler: {name}")

    # ---- initialization ----------------------------------------------------- #

    def reset_parameters(self) -> None:
        for lin in [self.pre]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        if self.edge_encoder is not None:
            for m in self.edge_encoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for m in self.mix.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---- forward ------------------------------------------------------------ #

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """
        x: [N, C_in]
        edge_index: [2, E]  (src, dst)
        edge_attr: [E, E_dim] or None
        """
        N, Cin = x.shape
        assert Cin == self.in_dim, f"Expected node dim {self.in_dim}, got {Cin}"

        if self.add_self_loops:
            edge_index = _add_self_loops(edge_index, N)
            if edge_attr is not None and self.edge_encoder is not None:
                E_new = edge_index.size(1)
                pad = E_new - edge_attr.size(0)
                if pad > 0:
                    zeros = torch.zeros(
                        pad, edge_attr.size(1), device=edge_attr.device, dtype=edge_attr.dtype
                    )
                    edge_attr = torch.cat([edge_attr, zeros], dim=0)

        src, dst = edge_index[0], edge_index[1]

        # Pre-project node features into hidden message space
        h = self.pre(x)  # [N, H]
        msg = h[src]  # [E, H]

        # Edge-conditioned messages (if enabled)
        if self.edge_encoder is not None and edge_attr is not None:
            msg = msg + self.edge_encoder(edge_attr)

        msg = self.msg_drop(msg)

        # In-degree per destination node, as float [N, 1]
        deg = _segment_count(dst, N, device=x.device, dtype=x.dtype)  # [N,1]
        degf = deg  # already float dtype

        # Apply each aggregator to build [N,H] tensors
        aggs: list[Tensor] = []
        for a in self.aggregators:
            if a == "sum":
                aggs.append(_segment_sum(msg, dst, N))
            elif a == "mean":
                aggs.append(_segment_mean(msg, dst, N))
            elif a == "std":
                # sqrt(var + eps)
                var = _segment_var(msg, dst, N)
                aggs.append(torch.sqrt(var + 1e-5))
            elif a == "min":
                aggs.append(_segment_min(msg, dst, N))
            elif a == "max":
                aggs.append(_segment_max(msg, dst, N))
            else:
                raise RuntimeError(f"Unexpected aggregator: {a}")

        # Degree-aware scalers; concatenate (aggregator × scaler)
        outs: list[Tensor] = []
        for A in aggs:
            for s in self.scalers:
                scale = self._scaler(s, degf)  # [N,1]
                outs.append(A * scale)

        out_cat = (
            torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        )  # [N, H * (|aggs|*|scalers|)]
        out = self.mix(out_cat)  # [N, C_out]
        return out


# -----------------------------------------------------------------------------
# PNA Block (residual + norm + act + dropout)
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


class PNABlock(nn.Module):
    """
    Residual PNA block:
        y = PNAConv(x)
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
        hidden_dim: int,
        aggregators: Sequence[str],
        scalers: Sequence[str],
        deg_histogram: Tensor,
        edge_dim: int | None,
        edge_embed_dim: int | None,
        add_self_loops: bool,
        msg_dropout: float,
        block_norm: str,
        block_act: str,
        block_dropout: float,
        residual: bool,
        mix_act: str,
        mix_norm: str,
        mix_dropout: float,
    ) -> None:
        super().__init__()

        self.conv = PNAConv(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg_histogram=deg_histogram,
            edge_dim=edge_dim,
            edge_embed_dim=edge_embed_dim,
            add_self_loops=add_self_loops,
            msg_dropout=msg_dropout,
            mix_act=mix_act,
            mix_norm=mix_norm,
            mix_dropout=mix_dropout,
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
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        batch: Tensor | None,
    ) -> Tensor:
        y = self.conv(x, edge_index, edge_attr)
        if self.use_res:
            x = self.res_proj(x) + y
        else:
            x = y
        x = self.norm(x, batch)
        x = self.act(x)
        x = self.drop(x)
        return x


# -----------------------------------------------------------------------------
# Public config + encoder
# -----------------------------------------------------------------------------


@dataclass
class PNAConfig:
    in_dim: int
    out_dim: int
    hidden_dim: int = 128
    num_layers: int = 3

    # Degree-aware knobs
    aggregators: Sequence[str] = ("mean", "sum", "std", "max", "min")
    scalers: Sequence[str] = ("identity", "amplification", "attenuation", "linear")
    deg_histogram: Tensor | Sequence[float] | Sequence[int] | None = None

    # Edge features
    edge_dim: int | None = None
    edge_embed_dim: int | None = None
    add_self_loops: bool = True

    # Message/mix internals
    msg_dropout: float = 0.0
    mix_act: str = "relu"
    mix_norm: str = "layer"
    mix_dropout: float = 0.0

    # Block-level wrappers
    norm: str = "layer"  # "layer" | "graph" | "none"
    act: str = "gelu"
    dropout: float = 0.1
    residual: bool = True


class PNA(nn.Module):
    """
    Multi-layer PNA encoder.

    Forward:
        forward(x, edge_index, edge_attr=None, batch=None) -> [N, out_dim]
    """

    def __init__(self, cfg: PNAConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Materialize a degree histogram buffer for all blocks
        if cfg.deg_histogram is None:
            deg_hist = torch.tensor([1.0, 1.0])
        else:
            deg_hist = torch.as_tensor(cfg.deg_histogram, dtype=torch.float32)
        self.register_buffer("deg_hist", deg_hist)

        layers = []
        in_dim = cfg.in_dim
        for li in range(cfg.num_layers):
            is_last = li == cfg.num_layers - 1
            out_dim = cfg.out_dim if is_last else cfg.hidden_dim

            block = PNABlock(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=cfg.hidden_dim,
                aggregators=cfg.aggregators,
                scalers=cfg.scalers,
                deg_histogram=self.deg_hist,
                edge_dim=cfg.edge_dim,
                edge_embed_dim=cfg.edge_embed_dim,
                add_self_loops=cfg.add_self_loops,
                msg_dropout=cfg.msg_dropout,
                block_norm=cfg.norm,
                block_act=(cfg.act if not is_last else "identity"),
                block_dropout=(cfg.dropout if not is_last else 0.0),
                residual=cfg.residual,
                mix_act=cfg.mix_act,
                mix_norm=cfg.mix_norm,
                mix_dropout=cfg.mix_dropout,
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
# Convenience factory (for YAML/registry integration)
# -----------------------------------------------------------------------------


def build_pna(
    *,
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    aggregators: Sequence[str] = ("mean", "sum", "std", "max", "min"),
    scalers: Sequence[str] = ("identity", "amplification", "attenuation", "linear"),
    deg_histogram: Tensor | Sequence[float] | Sequence[int] | None = None,
    edge_dim: int | None = None,
    edge_embed_dim: int | None = None,
    add_self_loops: bool = True,
    msg_dropout: float = 0.0,
    mix_act: str = "relu",
    mix_norm: str = "layer",
    mix_dropout: float = 0.0,
    norm: str = "layer",
    act: str = "gelu",
    dropout: float = 0.1,
    residual: bool = True,
) -> PNA:
    cfg = PNAConfig(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        aggregators=aggregators,
        scalers=scalers,
        deg_histogram=deg_histogram,
        edge_dim=edge_dim,
        edge_embed_dim=edge_embed_dim,
        add_self_loops=add_self_loops,
        msg_dropout=msg_dropout,
        mix_act=mix_act,
        mix_norm=mix_norm,
        mix_dropout=mix_dropout,
        norm=norm,
        act=act,
        dropout=dropout,
        residual=residual,
    )
    return PNA(cfg)

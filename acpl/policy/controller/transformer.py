# acpl/policy/controller/transformer.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

__all__ = [
    "TemporalTransformerConfig",
    "TemporalTransformer",
]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _activation(name: str, dim: int | None = None) -> nn.Module:
    n = (name or "gelu").lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "gelu":
        return nn.GELU()
    if n == "swish" or n == "silu":
        return nn.SiLU()
    if n == "prelu":
        return nn.PReLU(num_parameters=dim or 1, init=0.25)
    if n in ("identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation '{name}'")


class DropPath(nn.Module):
    """
    Stochastic depth (a.k.a. DropPath). Per-sample path dropping.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        p = self.drop_prob
        if not self.training or p <= 0.0:
            return x
        # x: (B, S, D) or (S, B, D) — broadcast over batch/sequence
        keep = 1.0 - p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device, dtype=x.dtype) < keep).to(x.dtype)
        return x * mask / keep


def _init_linear(m: nn.Linear, *, kaiming: bool = False) -> None:
    if kaiming:
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    else:
        nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


# -----------------------------------------------------------------------------
# Time embeddings (sinusoidal or learned)
# -----------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """
    Classic transformer PE but time-only. Returns shape (T, P).
    """

    def __init__(self, dim: int, max_T: int = 4096):
        super().__init__()
        self.dim = int(dim)
        self.max_T = int(max_T)
        pe = torch.zeros(self.max_T, self.dim)
        position = torch.arange(0, self.max_T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / self.dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, T: int) -> Tensor:
        if T > self.max_T:
            # Extend on the fly if needed (no gradient)
            with torch.no_grad():
                old_T = self.max_T
                self.max_T = int(T * 1.5)
                pe = torch.zeros(self.max_T, self.dim, device=self.pe.device)
                position = torch.arange(
                    0, self.max_T, dtype=torch.float32, device=self.pe.device
                ).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.pe.device)
                    * (-math.log(10_000.0) / self.dim)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.pe = pe
        return self.pe[:T]


class LearnedTimeEmbedding(nn.Module):
    """
    Learned time embedding. Returns (T, P).
    """

    def __init__(self, dim: int, max_T: int = 2048):
        super().__init__()
        self.emb = nn.Embedding(max_T, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, T: int) -> Tensor:
        idx = torch.arange(T, device=self.emb.weight.device)
        return self.emb(idx)


# -----------------------------------------------------------------------------
# ALiBi: attention linear biases (length-extrapolation friendly)
# -----------------------------------------------------------------------------


def _alibi_slopes(n_heads: int) -> Tensor:
    """
    Slopes as in ALiBi (Press et al.).
    """

    def get_slopes(n):
        # same recipe as reference implementation
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        slopes = [start * ratio**i for i in range(n)]
        return torch.tensor(slopes, dtype=torch.float32)

    if math.log2(n_heads).is_integer():
        slopes = get_slopes(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = torch.cat(
            [
                get_slopes(closest_power_of_2),
                get_slopes(2 * closest_power_of_2)[0::2][: (n_heads - closest_power_of_2)],
            ]
        )
    return slopes  # [H]


def build_alibi_bias(n_heads: int, T: int, device=None, dtype=None, causal: bool = True) -> Tensor:
    """
    Returns bias tensor shaped [H, T, T], added to attention logits.
    ALiBi uses (i-j) with slope per head, negative for j>i in causal mode.
    """
    slopes = _alibi_slopes(n_heads).to(device=device, dtype=dtype)
    pos = torch.arange(T, device=device, dtype=torch.int64)
    # distances j->i: (i - j)
    dist = pos.unsqueeze(0) - pos.unsqueeze(1)  # [T, T]
    if causal:
        dist = dist.triu()  # ensure we don't add positive bias to future
    bias = slopes.view(n_heads, 1, 1) * dist.to(dtype or torch.float32)  # [H, T, T]
    # Note: when used with SDPA, bias should be added to attn logits pre-softmax.
    return bias


# -----------------------------------------------------------------------------
# Transformer blocks
# -----------------------------------------------------------------------------


class MultiheadSelfAttention(nn.Module):
    """
    Self-attention with optional ALiBi bias and causal masking.
    Uses PyTorch SDPA when available.
    Inputs are (B, T, D).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        assert dim % n_heads == 0, "model_dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=use_bias)
        self.out_proj = nn.Linear(dim, dim, bias=use_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        _init_linear(self.qkv)
        _init_linear(self.out_proj)

    def forward(
        self,
        x: Tensor,  # (B, T, D)
        *,
        causal: bool,
        alibi_bias: Tensor | None = None,  # (H, T, T) or None
        key_padding_mask: Tensor | None = None,  # (B, T) True for PAD
    ) -> Tensor:
        B, T, D = x.shape
        H = self.n_heads
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H, T, Dh)
        def split_heads(t: Tensor) -> Tensor:
            return t.view(B, T, H, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Build attn mask
        attn_mask = None  # what we'll pass to SDPA
        if causal:
            # (T, T) bool – True means "masked" (upper triangle)
            causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(diagonal=1)
        else:
            causal_mask = None

        # Merge ALiBi (float additive bias) with causal (bool) if present
        # SDPA will broadcast a bool [T, T], or will accept a float [B, H, T, T].
        if alibi_bias is not None:
            # alibi_bias: [H, T, T]  → expand to [B, H, T, T]
            float_bias = alibi_bias.unsqueeze(0).expand(B, -1, -1, -1).to(q.dtype)
            if causal_mask is not None:
                # convert causal bool to -inf/0 then add to ALiBi
                causal_inf = torch.zeros(T, T, device=x.device, dtype=q.dtype).masked_fill(
                    causal_mask, float("-inf")
                )
                float_bias = float_bias + causal_inf
            attn_mask = float_bias  # final float mask [B, H, T, T]
        else:
            # No ALiBi: we can just pass the bool [T, T] causal mask (or None)
            attn_mask = causal_mask  # either None or bool [T, T]

        y = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,  # bool [T,T] OR float [B,H,T,T]
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,  # we've handled causality in attn_mask
        )

        # (B, H, T, Dh) -> (B, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out_proj(y)
        y = self.proj_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float, act: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = _activation(act, dim=hidden)
        self.drop = nn.Dropout(dropout)
        _init_linear(self.fc1, kaiming=True)
        _init_linear(self.fc2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre/Post-norm transformer block with DropPath and optional residual gating.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        norm_first: bool = True,
        residual_gating: bool = False,
        droppath: float = 0.0,
        act: str = "gelu",
    ):
        super().__init__()
        self.norm_first = bool(norm_first)
        self.residual_gating = bool(residual_gating)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(
            dim, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout
        )
        self.drop_path1 = DropPath(droppath) if droppath > 0 else nn.Identity()
        self.gate1 = nn.Parameter(torch.ones(1)) if residual_gating else None

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(mlp_ratio * dim)
        self.mlp = MLP(dim, hidden, dropout=dropout, act=act)
        self.drop_path2 = DropPath(droppath) if droppath > 0 else nn.Identity()
        self.gate2 = nn.Parameter(torch.ones(1)) if residual_gating else None

    def forward(
        self,
        x: Tensor,  # (B, T, D)
        *,
        causal: bool,
        alibi_bias: Tensor | None,
        key_padding_mask: Tensor | None,
    ) -> Tensor:
        if self.norm_first:
            # SA
            y = self.norm1(x)
            y = self.attn(
                y, causal=causal, alibi_bias=alibi_bias, key_padding_mask=key_padding_mask
            )
            y = self.drop_path1(y)
            if self.gate1 is not None:
                x = x + self.gate1 * y
            else:
                x = x + y
            # MLP
            y = self.norm2(x)
            y = self.mlp(y)
            y = self.drop_path2(y)
            if self.gate2 is not None:
                x = x + self.gate2 * y
            else:
                x = x + y
            return x

        # Post-norm variant
        y = self.attn(x, causal=causal, alibi_bias=alibi_bias, key_padding_mask=key_padding_mask)
        y = self.drop_path1(y)
        if self.gate1 is not None:
            x = self.norm1(x + self.gate1 * y)
        else:
            x = self.norm1(x + y)

        y = self.mlp(x)
        y = self.drop_path2(y)
        if self.gate2 is not None:
            x = self.norm2(x + self.gate2 * y)
        else:
            x = self.norm2(x + y)
        return x


# -----------------------------------------------------------------------------
# Public config
# -----------------------------------------------------------------------------


@dataclass
class TemporalTransformerConfig:
    # I/O
    in_dim: int  # node embedding dim + optional external tpe dim (if concat_external_tpe=True)
    model_dim: int = 256
    out_dim: int | None = None  # if None, equals model_dim
    num_layers: int = 6
    num_heads: int = 8

    # Feedforward / regularization
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    droppath: float = 0.0  # last layer gets droppath, linearly scaled across depth

    # Norm/topology
    norm_first: bool = True
    residual_gating: bool = False

    # Time embeddings
    use_internal_time: bool = True  # if True, controller generates its own time embeddings
    time_embed_type: Literal["sinusoidal", "learned"] = "sinusoidal"
    time_embed_dim: int = 64
    time_max_len: int = 4096  # cap for internal time embeddings

    # Input wiring
    concat_external_tpe: bool = True  # if you pass a tpe tensor, concat it to z per time-step
    project_in: bool = True  # project (z||tpe) into model_dim
    final_norm: bool = True  # add final LayerNorm
    causal: bool = True  # use causal mask over time
    use_alibi: bool = True  # add ALiBi bias for length-extrapolation


# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer controller (Phase B3).

    Primary contract (drop-in for the tiny controller):
        forward(z, tpe) -> (T, N, D)

    Inputs
    ------
    z   : (N, Dz)                 node embeddings (from GNN encoder)
    tpe : (T, Pt) or None         time positional embeddings/features
                                  - If None and cfg.use_internal_time=True,
                                    we create time embeddings internally.

    Output
    ------
    H   : (T, N, D_out)           time-major hidden states

    Notes
    -----
    - Processes each node independently across time (batch==N).
    - Builds a sequence x_t(v) = concat(z[v], time_emb[t], external_tpe[t]?).
    - Robust to variable T (supports causal masking and ALiBi).
    """

    def __init__(self, cfg: TemporalTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.model_dim = int(cfg.model_dim)
        self.out_dim = int(cfg.out_dim or cfg.model_dim)

        # Internal time embedding if requested
        if cfg.use_internal_time:
            if cfg.time_embed_type == "sinusoidal":
                self.time_emb = SinusoidalTimeEmbedding(cfg.time_embed_dim, max_T=cfg.time_max_len)
            elif cfg.time_embed_type == "learned":
                self.time_emb = LearnedTimeEmbedding(cfg.time_embed_dim, max_T=cfg.time_max_len)
            else:
                raise ValueError(f"Unknown time_embed_type: {cfg.time_embed_type}")
            internal_dim = cfg.time_embed_dim
        else:
            self.time_emb = None
            internal_dim = 0

        # Input projection
        proj_in_dim = cfg.in_dim
        if cfg.concat_external_tpe:
            # When caller provides tpe (T, Pt), we concat it at runtime; the
            # linear expects z||tpe, so include Pt in cfg.in_dim upstream.
            # If you are using internal-only time embeddings, keep cfg.in_dim = Dz.
            pass
        else:
            # no external tpe concatenation; only z (Dz) is expected externally
            proj_in_dim = cfg.in_dim + internal_dim

        if cfg.project_in:
            self.in_proj = nn.Linear(
                proj_in_dim if cfg.concat_external_tpe else proj_in_dim, self.model_dim
            )
            _init_linear(self.in_proj, kaiming=True)
        else:
            # require dims match
            if proj_in_dim != self.model_dim:
                raise ValueError("project_in=False requires in_dim (+internal_tpe?) == model_dim")
            self.in_proj = nn.Identity()

        # Blocks
        dpr = torch.linspace(
            0, cfg.droppath, steps=cfg.num_layers
        ).tolist()  # stochastic depth schedule
        blocks = []
        for i in range(cfg.num_layers):
            blocks.append(
                TransformerBlock(
                    dim=self.model_dim,
                    n_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    attn_dropout=cfg.attn_dropout,
                    dropout=cfg.dropout,
                    norm_first=cfg.norm_first,
                    residual_gating=cfg.residual_gating,
                    droppath=dpr[i],
                    act="gelu",
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.final_norm = nn.LayerNorm(self.model_dim) if cfg.final_norm else nn.Identity()

        # Optional output projection if out_dim differs
        self.out_proj = (
            nn.Linear(self.model_dim, self.out_dim)
            if self.out_dim != self.model_dim
            else nn.Identity()
        )
        if isinstance(self.out_proj, nn.Linear):
            _init_linear(self.out_proj)

    @torch.no_grad()
    def _build_alibi(self, T: int, n_heads: int, device, dtype) -> Tensor | None:
        if not self.cfg.use_alibi:
            return None
        return build_alibi_bias(n_heads, T, device=device, dtype=dtype, causal=self.cfg.causal)

    def _prepare_sequence(
        self,
        z: Tensor,  # (N, Dz)
        tpe: Tensor | None,  # (T, Pt) or None
    ) -> Tensor:
        """
        Build (B, T, Din) = (N, T, concat[z, tpe?, internal?])
        """
        N, Dz = z.shape
        if tpe is None and not self.cfg.use_internal_time and self.cfg.concat_external_tpe:
            raise ValueError(
                "concat_external_tpe=True but no tpe provided and internal time disabled."
            )

        # Time embeddings: external (given by caller) and/or internal
        if tpe is not None and self.cfg.concat_external_tpe:
            T, Pt = tpe.shape
            t_ext = tpe  # (T, Pt)
        else:
            # We still need T; infer from internal time embedding only
            if self.cfg.use_internal_time:
                # we defer T until forward where tpe_external may be None
                t_ext = None
            else:
                # Degenerate case: no time features at all -> T must come from tpe
                raise ValueError(
                    "Cannot infer sequence length T without internal or external time embeddings."
                )

        # Determine T and build internal time embedding if requested
        if tpe is not None:
            T = tpe.size(0)
        else:
            # internal only
            T = 1  # temporary; we’ll fix below if internal time used with explicit T
        if self.cfg.use_internal_time:
            t_int = self.time_emb(T)  # (T, Pi)
        else:
            t_int = None

        # Expand z across time: (N, T, Dz)
        if tpe is not None:
            T = tpe.size(0)
        elif self.cfg.use_internal_time:
            T = t_int.size(0)
        else:
            raise AssertionError("Unreachable: T resolution failed.")
        z_rep = z.unsqueeze(1).expand(N, T, Dz)

        parts = [z_rep]  # list of (N, T, *)
        if t_ext is not None and self.cfg.concat_external_tpe:
            parts.append(t_ext.unsqueeze(0).expand(N, T, t_ext.size(-1)))
        if t_int is not None and not self.cfg.concat_external_tpe:
            # We use internal time only when external is not concatenated,
            # because in the usual config 'in_dim' already includes external Pt.
            parts.append(t_int.unsqueeze(0).expand(N, T, t_int.size(-1)))

        x = torch.cat(parts, dim=-1)  # (N, T, Din)
        return x

    def forward(
        self,
        z: Tensor,  # (N, Dz)
        tpe: Tensor | None = None,  # (T, Pt) or None
        key_padding_mask: Tensor | None = None,  # (N, T) True for PAD positions
    ) -> Tensor:
        device = z.device
        dtype = z.dtype

        # Prepare input sequence (N, T, Din)
        x = self._prepare_sequence(z, tpe)
        N, T, Din = x.shape

        # Project to model dim
        x = self.in_proj(x)  # (N, T, Dm)

        # Build ALiBi bias (H, T, T)
        alibi = self._build_alibi(T, self.blocks[0].attn.n_heads, device=device, dtype=dtype)

        # Process blocks
        for blk in self.blocks:
            x = blk(
                x,  # (N, T, D)
                causal=self.cfg.causal,
                alibi_bias=alibi,
                key_padding_mask=key_padding_mask,
            )

        x = self.final_norm(x)  # (N, T, Dm)
        x = self.out_proj(x)  # (N, T, Do)

        # Return (T, N, Do) to match the original TinyTimeTransformer interface
        return x.transpose(0, 1).contiguous()

# acpl/policy/controller/tiny_transformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


__all__ = ["TinyTimeTransformer", "TinyTimeTransformerConfig"]


@dataclass
class TinyTimeTransformerConfig:
    in_dim: int           # node emb + time PE
    model_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    layernorm: bool = True
    residual: bool = True


class TinyTimeTransformer(nn.Module):
    """
    Very small Transformer over **time** per node (independent across nodes).
    It processes the sequence x_t(v) = concat(z[v], tpe[t]).

    Shapes
    ------
    Input:  z (N, Dz), tpe (T, Pt)
    Output: H (T, N, Dm)
    """
    def __init__(self, cfg: TinyTimeTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.in_dim, cfg.model_dim, bias=True)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.mlp_ratio * cfg.model_dim),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,   # (B, S, D)
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.out_norm = nn.LayerNorm(cfg.model_dim) if cfg.layernorm else nn.Identity()

        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor, tpe: torch.Tensor) -> torch.Tensor:
        N, Dz = z.shape
        T, Pt = tpe.shape
        # (N, T, Dz+Pt)
        x = torch.cat([z.unsqueeze(1).expand(N, T, Dz),
                       tpe.unsqueeze(0).expand(N, T, Pt)], dim=-1)
        x = self.proj(x)           # (N, T, Dm)
        y = self.enc(x)            # (N, T, Dm)
        y = self.out_norm(y)
        return y.transpose(0, 1).contiguous()   # (T, N, Dm)

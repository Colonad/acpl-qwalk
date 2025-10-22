# acpl/policy/encoder_factory.py
# SPDX-License-Identifier: MIT
"""
Encoder factory for ACPL policies.

Builds node encoders (GNNs) from a config dict or a short name.
Currently supports GraphSAGE (mean) from Phase B3.

Schema (minimal):
    {
      "name": "graphsage",
      "in_dim": <int>,          # required by caller (passed in)
      "out_dim": <int>,         # required by caller (passed in)
      "hidden_dim": 128,
      "num_layers": 2,
      "edge_dim": null,
      "norm": "layer",          # "layer" | "graph" | "none"
      "dropout": 0.1,
      "act": "gelu",            # "relu" | "gelu" | "prelu"
      "residual": true,
      "add_self_loops": true
    }

Usage:
    from acpl.policy.encoder_factory import create_encoder

    enc = create_encoder(
        cfg={"name":"graphsage","hidden_dim":128,"num_layers":2},
        in_dim=x.size(-1),
        out_dim=embed_dim,
    )
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from acpl.policy.gnn.graphsage import build_graphsage

_NAMES = {"graphsage"}


def _pop_with_default(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    return cfg[key] if key in cfg and cfg[key] is not None else default


def create_encoder(
    cfg: Mapping[str, Any] | str,
    *,
    in_dim: int,
    out_dim: int,
) -> nn.Module:
    """
    Create an encoder from a config mapping or short name.

    Parameters
    ----------
    cfg : Mapping or str
        Either a config dict with "name", or the string "graphsage".
    in_dim : int
        Input feature dimension (must be provided by caller).
    out_dim : int
        Output embedding dimension (must be provided by caller).

    Returns
    -------
    nn.Module
        A node encoder (e.g., GraphSAGE).
    """
    if isinstance(cfg, str):
        name = cfg.lower()
        params: dict[str, Any] = {}
    else:
        name = str(cfg.get("name", "graphsage")).lower()
        params = dict(cfg)

    if name not in _NAMES:
        raise ValueError(f"Unknown encoder '{name}'. Supported: {sorted(_NAMES)}")

    # GraphSAGE (mean) parameters with safe defaults
    hidden_dim = int(_pop_with_default(params, "hidden_dim", max(in_dim, out_dim)))
    num_layers = int(_pop_with_default(params, "num_layers", 2))
    edge_dim: int | None = params.get("edge_dim")
    if edge_dim is not None:
        edge_dim = int(edge_dim)

    norm = str(_pop_with_default(params, "norm", "layer")).lower()
    dropout = float(_pop_with_default(params, "dropout", 0.1))
    act = str(_pop_with_default(params, "act", "gelu")).lower()
    residual = bool(_pop_with_default(params, "residual", True))
    add_self_loops = bool(_pop_with_default(params, "add_self_loops", True))

    encoder: nn.Module = build_graphsage(
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
    return encoder

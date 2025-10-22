# acpl/policy/gnn/__init__.py
# SPDX-License-Identifier: MIT

"""
GNN policy encoders.
Exports the mean-pooling GraphSAGE used in Phase B3.
"""

from .graphsage import (
    GraphSAGE,
    GraphSAGEConfig,
    SAGEConvMean,
    build_graphsage,
)

__all__ = [
    "GraphSAGE",
    "GraphSAGEConfig",
    "SAGEConvMean",
    "build_graphsage",
]

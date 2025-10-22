# tests/test_features_errors.py
import pytest
import torch

from acpl.data.features import (
    FeatureSpec,
    build_arc_features,
    build_node_features,
    laplacian_positional_encoding,
    node_features_line,
    random_walk_structural_encoding,
)
from acpl.data.graphs import grid_graph, line_graph


def test_node_features_line_rejects_wrong_coord_shape():
    edge_index, degrees, coords, _ = line_graph(5)
    # Make coords (N,2) instead of (N,1)
    bad_coords = torch.cat([coords, coords], dim=1)
    with pytest.raises(ValueError):
        _ = node_features_line(degrees, bad_coords)


def test_build_node_features_odd_sinusoidal_dims_raises():
    edge_index, degrees, coords, _ = grid_graph(3, 3)
    spec = FeatureSpec(
        use_degree=True,
        degree_norm="inv_sqrt",
        use_coords=True,
        use_sinusoidal_coords=True,
        sinusoidal_dims=3,  # <-- must be even; should raise
        use_lap_pe=False,
        use_rwse=False,
    )
    with pytest.raises(ValueError):
        _ = build_node_features(edge_index, degrees, coords, spec=spec)


def test_build_node_features_invalid_degree_norm_raises():
    edge_index, degrees, coords, _ = line_graph(6)
    spec = FeatureSpec(
        use_degree=True,
        degree_norm="totally_wrong_mode",  # <-- invalid; should raise
        use_coords=False,
        use_lap_pe=False,
        use_rwse=False,
    )
    with pytest.raises(ValueError):
        _ = build_node_features(edge_index, degrees, coords, spec=spec)


def test_build_arc_features_bad_edge_index_shape_raises():
    edge_index, degrees, coords, _ = grid_graph(3, 3)
    # Corrupt edge_index to wrong leading dimension (3, A)
    bad_edge_index = torch.cat([edge_index, edge_index[:1]], dim=0)
    with pytest.raises(ValueError):
        _ = build_arc_features(bad_edge_index, coords)


def test_lap_pe_bad_edge_index_shape_raises():
    edge_index, degrees, coords, _ = grid_graph(3, 3)
    N = degrees.numel()
    bad_edge_index = torch.cat([edge_index, edge_index[:1]], dim=0)  # (3, A)
    with pytest.raises(ValueError):
        _ = laplacian_positional_encoding(bad_edge_index, N, k=3)


def test_rwse_bad_edge_index_shape_raises():
    edge_index, degrees, coords, _ = line_graph(7)
    bad_edge_index = torch.cat([edge_index, edge_index[:1]], dim=0)  # (3, A)
    with pytest.raises(ValueError):
        _ = random_walk_structural_encoding(bad_edge_index, degrees.numel(), K=3)

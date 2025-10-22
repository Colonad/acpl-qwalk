# tests/test_features.py
import torch

from acpl.data.features import (
    FeatureSpec,
    build_arc_features,
    build_node_features,
    laplacian_positional_encoding,
    node_features_line,
    normalize_degree,
    random_walk_structural_encoding,
)
from acpl.data.graphs import grid_graph, line_graph


def _ortho_check(M: torch.Tensor, atol=1e-5):
    # Check columns of M are approximately orthonormal: M^T M ≈ I
    QTQ = M.T @ M  # (k,k)
    I = torch.eye(QTQ.size(0), dtype=QTQ.dtype, device=QTQ.device)
    assert torch.allclose(QTQ, I, atol=atol)


def test_node_features_line_on_path_graph():
    # Path of length 5: degrees = [1,2,2,2,1], coords in [0,1]
    edge_index, degrees, coords, arc_slices = line_graph(5, seed=0)
    X = node_features_line(degrees, coords)
    assert X.shape == (5, 2)

    # Column 0 is inv_sqrt(deg), column 1 is normalized x
    inv_sqrt = 1.0 / torch.sqrt(torch.clamp(degrees.to(torch.float32), min=1.0))
    assert torch.allclose(X[:, 0], inv_sqrt, atol=1e-6)
    assert (X[:, 1] >= 0).all() and (X[:, 1] <= 1).all()

    # Sanity: endpoints have larger inv_sqrt than middle nodes (1 > 1/sqrt(2))
    assert X[0, 0] > X[1, 0]
    assert X[-1, 0] > X[-2, 0]


def test_lappe_shapes_determinism_and_orthogonality():
    # Small 3x3 grid, k=4 LapPE (symmetric normalized Laplacian)
    edge_index, degrees, coords, arc_slices = grid_graph(3, 3, seed=123)
    N = degrees.numel()
    k = 4

    # Deterministic with fixed seed and random_sign=True
    PE1 = laplacian_positional_encoding(edge_index, N, k, mode="sym", random_sign=True, seed=42)
    PE2 = laplacian_positional_encoding(edge_index, N, k, mode="sym", random_sign=True, seed=42)
    assert PE1.shape == (N, k)
    assert torch.allclose(PE1, PE2, atol=1e-7)

    # Orthogonality of columns (for sym Laplacian)
    _ortho_check(PE1, atol=1e-5)

    # If we fix random_sign=False, sign should be consistent as well
    PE3 = laplacian_positional_encoding(edge_index, N, k, mode="sym", random_sign=False)
    PE4 = laplacian_positional_encoding(edge_index, N, k, mode="sym", random_sign=False)
    assert torch.allclose(PE3, PE4, atol=1e-7)


def test_rwse_basic_properties():
    # On a grid without self-loops, diag(P) = 0; later powers may become >0.
    edge_index, degrees, coords, arc_slices = grid_graph(3, 3, seed=0)
    N = degrees.numel()
    K = 4
    RW = random_walk_structural_encoding(edge_index, N, K)
    assert RW.shape == (N, K)

    # Values are probabilities: in [0,1]
    assert (RW >= 0).all()
    assert (RW <= 1).all()

    # First column is diag(P) which must be zero (no self-loops)
    assert torch.allclose(RW[:, 0], torch.zeros(N, dtype=RW.dtype), atol=1e-7)

    # There should be some positive return probability by k=2 or later
    assert (RW[:, 1:].sum(dim=1) > 0).any()


def test_arc_features_geometry_and_shapes_2d():
    # Grid with 2D coords → arc features contain direction(2) + distance(1) + angle sin/cos(2) = 5 dims
    edge_index, degrees, coords, arc_slices = grid_graph(3, 3, seed=0)
    A = edge_index.shape[1]
    F = build_arc_features(
        edge_index, coords, use_direction=True, use_distance=True, use_angle2d=True
    )
    assert F.shape == (A, 5)

    # Direction rows should be unit norm wherever distance > 0
    dir_part = F[:, :2]
    dist = F[:, 2]
    # avoid division by zero; for grid edges, dist>0 always
    norms = torch.linalg.norm(dir_part, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    # Angle encoding: sin^2 + cos^2 ≈ 1
    sin = F[:, 3]
    cos = F[:, 4]
    unit_circle = sin * sin + cos * cos
    assert torch.allclose(unit_circle, torch.ones_like(unit_circle), atol=1e-6)


def test_build_node_features_block_indexing_and_shapes():
    # Combine multiple blocks and verify index ranges are consistent
    edge_index, degrees, coords, arc_slices = grid_graph(3, 3, seed=0)
    spec = FeatureSpec(
        use_degree=True,
        degree_norm="inv_sqrt",
        degree_onehot_K=3,
        use_coords=True,
        use_sinusoidal_coords=True,
        sinusoidal_dims=4,
        use_lap_pe=True,
        lap_pe_k=3,
        lap_pe_norm="sym",
        lap_pe_random_sign=False,
        use_rwse=True,
        rwse_K=3,
        seed=999,
    )
    X, idx = build_node_features(edge_index, degrees, coords, spec=spec)
    N = degrees.numel()

    # Check shapes
    start_end = []
    for key in ["deg_norm", "deg_onehot", "coords", "coords_sin", "lap_pe", "rwse"]:
        if key in idx:
            s, e = idx[key]
            assert 0 <= s <= e <= X.shape[1]
            start_end.append((s, e))
            # slice exists and has expected N rows
            assert X[:, s:e].shape[0] == N

    # Ensure blocks do not overlap and cover exactly the used width
    start_end_sorted = sorted(start_end, key=lambda t: t[0])
    for (s1, e1), (s2, e2) in zip(start_end_sorted, start_end_sorted[1:], strict=False):
        assert e1 == s2  # contiguous packing

    # Sanity: degree norm column equals normalize_degree(...)
    s_deg, e_deg = idx["deg_norm"]
    expected = normalize_degree(degrees, mode="inv_sqrt").view(-1, 1)
    assert torch.allclose(X[:, s_deg:e_deg], expected, atol=1e-6)

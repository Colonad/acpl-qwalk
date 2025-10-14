# tests/test_shift_unitarity.py
import numpy as np
import pytest

from acpl.sim.portmap import make_flipflop_portmap
from acpl.sim.shift import (
    apply_shift_numpy,
    apply_shift_torch,
    build_shift_index,
    build_shift_scipy,
    build_shift_torch,
    is_permutation_matrix_numpy,
)

# Optional dependencies
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None


def path_edges(n: int) -> np.ndarray:
    """Undirected path 0-1-...-(n-1) as (2, E) array."""
    u = np.arange(n - 1, dtype=np.int64)
    v = u + 1
    return np.stack([u, v], axis=0)


def triangle_edges() -> np.ndarray:
    """Undirected triangle on nodes {0,1,2} as (2, 3) array."""
    return np.array([[0, 1, 0], [1, 2, 2]], dtype=np.int64)


def test_shift_index_matches_rev_and_is_involution():
    # Use a small path so sizes are tiny but nontrivial.
    pm = make_flipflop_portmap(path_edges(5))  # E=4, A=8
    dest = build_shift_index(pm)

    # Must equal pm.rev
    assert np.array_equal(dest, pm.rev)

    # Applying S twice should be identity on arcs (involution).
    A = pm.num_arcs
    psi = np.arange(A, dtype=np.int64)
    psi1 = apply_shift_numpy(psi, dest)
    psi2 = apply_shift_numpy(psi1, dest)
    assert np.array_equal(psi2, psi)


@pytest.mark.parametrize("edge_index", [path_edges(4), triangle_edges()])
def test_shift_numpy_vs_mapping(edge_index):
    pm = make_flipflop_portmap(edge_index)
    dest = build_shift_index(pm)

    # 1-D and 2-D cases should both permute along axis 0.
    A = pm.num_arcs
    psi1d = np.random.randn(A).astype(np.float64)
    psi2d = np.random.randn(A, 3).astype(np.float64)

    out1d = apply_shift_numpy(psi1d, dest)
    out2d = apply_shift_numpy(psi2d, dest)

    # Direct check: out[a] == psi[rev[a]]
    assert np.allclose(out1d, psi1d[pm.rev])
    assert np.allclose(out2d, psi2d[pm.rev, :])


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
@pytest.mark.parametrize("edge_index", [path_edges(4), triangle_edges()])
def test_shift_torch_sparse_matches_index_select(edge_index):
    pm = make_flipflop_portmap(edge_index)
    S = build_shift_torch(pm, device=None, dtype=None)
    A = pm.num_arcs

    # Sparse permutation matrix should be square and coalesced.
    assert tuple(S.shape) == (A, A)
    if hasattr(S, "is_coalesced"):
        assert S.is_coalesced()

    # Compare S @ psi with index_select using the mapping.
    psi = torch.randn(A, dtype=torch.float32)
    dest = torch.from_numpy(pm.rev.astype(np.int64))
    out_mat = torch.sparse.mm(S, psi.view(-1, 1)).view(-1)
    out_idx = apply_shift_torch(psi, dest)
    assert torch.allclose(out_mat, out_idx, atol=0, rtol=0)

    # 2-D case
    psi2 = torch.randn(A, 2, dtype=torch.float32)
    out_mat2 = torch.sparse.mm(S, psi2)
    out_idx2 = apply_shift_torch(psi2, dest)
    assert torch.allclose(out_mat2, out_idx2, atol=0, rtol=0)


@pytest.mark.skipif(sp is None, reason="SciPy not available")
@pytest.mark.parametrize("edge_index", [path_edges(5), triangle_edges()])
def test_shift_scipy_is_permutation_and_applies(edge_index):
    pm = make_flipflop_portmap(edge_index)
    S = build_shift_scipy(pm)

    # Check permutation properties (one 1 per row/col).
    assert is_permutation_matrix_numpy(S)

    # Applying S is the same as indexing by rev.
    A = pm.num_arcs
    x = np.random.randn(A).astype(np.float64)
    y = (S @ x).astype(np.float64)
    assert np.allclose(y, x[pm.rev])

    # Unitarity for permutation: S^T S = I
    prod_sts = (S.T @ S).tocoo()
    assert prod_sts.shape == (A, A)
    # Compare against an identity via row/col sums (cheap & robust).
    ones = np.ones(A, dtype=np.float64)
    assert np.allclose((S.T @ S) @ ones, ones)
    assert np.allclose((S @ S.T) @ ones, ones)


def test_is_permutation_matrix_numpy_dense_true_and_false():
    # True case: a 4x4 permutation (swap 0<->1, 2<->3).
    P = np.zeros((4, 4), dtype=np.int64)
    P[1, 0] = 1
    P[0, 1] = 1
    P[3, 2] = 1
    P[2, 3] = 1
    assert is_permutation_matrix_numpy(P)

    # False: row with two ones, another with zero.
    Q = P.copy()
    Q[0, 0] = 1
    assert not is_permutation_matrix_numpy(Q)


def _line_edge_index(n: int) -> np.ndarray:
    """Undirected line graph: 0-1-2-...-(n-1), returned as (2, E) numpy array."""
    edges = [(i, i + 1) for i in range(n - 1)]
    uu = np.array([u for (u, v) in edges], dtype=np.int64)
    vv = np.array([v for (u, v) in edges], dtype=np.int64)
    return np.stack([uu, vv], axis=0)  # (2, E)


def test_shift_torch_is_permutation_and_unitary():
    ei = _line_edge_index(6)  # small, heterogeneous degrees at interior vs ends
    pm = make_flipflop_portmap(ei)
    s = build_shift_torch(pm)  # sparse COO (A x A), values in {0,1}

    # Basic shape check
    a = pm.num_arcs
    assert s.shape == (a, a)
    assert s.is_coalesced()

    # Convert to dense (tiny sizes in tests) and check permutation properties
    dense = s.to_dense()
    assert dense.dtype.is_floating_point
    # Binary 0/1 entries
    assert torch.all((dense == 0) | (dense == 1))
    # One-hot rows and columns
    row_sums = dense.sum(dim=1)
    col_sums = dense.sum(dim=0)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))
    assert torch.allclose(col_sums, torch.ones_like(col_sums))
    # Unitarity for permutation: S^T S = I
    prod = dense.T @ dense
    eye_mat = torch.eye(a, dtype=dense.dtype, device=dense.device)
    assert torch.allclose(prod, eye_mat)


@pytest.mark.skipif(sp is None, reason="SciPy not available")
def test_shift_scipy_is_permutation_and_unitary():
    ei = _line_edge_index(7)  # different size than previous test
    pm = make_flipflop_portmap(ei)
    s = build_shift_scipy(pm)  # scipy.sparse.coo_matrix

    # Basic shape
    a = pm.num_arcs
    assert s.shape == (a, a)

    # Binary values
    assert np.all((s.data == 0) | (s.data == 1))

    # Row/col sums should be 1
    row_sums = np.asarray(s.sum(axis=1)).ravel()
    col_sums = np.asarray(s.sum(axis=0)).ravel()
    assert np.allclose(row_sums, np.ones_like(row_sums))
    assert np.allclose(col_sums, np.ones_like(col_sums))

    # Unitarity (for permutation): S^T S = I (in sparse)
    prod = (s.T @ s).tocoo()
    # Compare structurally to identity: each diagonal 1, no off-diagonals
    assert prod.shape == (a, a)
    # Build an identity to compare
    from scipy.sparse import eye as sp_eye  # local import to keep test fast if scipy present

    eye_sparse = sp_eye(a, format="coo", dtype=s.dtype)
    # Same coordinates and data (allow reordering by sorting indices)
    prod_tuples = sorted(
        zip(prod.row.tolist(), prod.col.tolist(), prod.data.tolist(), strict=False)
    )
    eye_tuples = sorted(
        zip(
            eye_sparse.row.tolist(), eye_sparse.col.tolist(), eye_sparse.data.tolist(), strict=False
        )
    )
    assert prod_tuples == eye_tuples

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

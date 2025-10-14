# acpl/data/features.py
from __future__ import annotations

import numpy as np

__all__ = [
    "node_features_line",
    "laplacian_pe",
    "degree_role_onehot",
    "hypercube_bitstrings",
    "safe_concat",
]


# ---------------------------------------------------------------------
# Phase-A line features (unchanged)
# ---------------------------------------------------------------------


def node_features_line(
    degrees: np.ndarray,
    coords: np.ndarray,
    *,
    normalize_degree: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Build node features for the Phase-A line graph.

    Features (per node)
    -------------------
    1) degree (optionally normalized by max degree)
    2) 1-D coordinate in [0, 1] (already normalized)

    Parameters
    ----------
    degrees : (n,) int64
    coords  : (n,1) float
    normalize_degree : bool
    dtype : np.dtype

    Returns
    -------
    (n, 2) array
    """
    if degrees.ndim != 1:
        raise ValueError(f"degrees must be 1-D (got shape {degrees.shape})")
    if coords.ndim != 2 or coords.shape[1] != 1:
        raise ValueError(f"coords must have shape (n, 1) (got {coords.shape})")
    if degrees.shape[0] != coords.shape[0]:
        raise ValueError(
            f"degrees and coords must have the same rows: {degrees.shape[0]} != {coords.shape[0]}"
        )

    n = degrees.shape[0]

    # Degree feature (optionally normalized)
    deg = degrees.astype(np.float64, copy=False)
    if normalize_degree:
        max_deg = float(deg.max()) if n > 0 else 0.0
        scale = max(max_deg, 1.0)  # avoid divide-by-zero
        deg = deg / scale

    # Coordinate feature (already normalized to [0,1] by graphs.py)
    coord = coords.astype(np.float64, copy=False).reshape(n)

    feats = np.stack([deg, coord], axis=1).astype(dtype, copy=False)
    return feats


# ---------------------------------------------------------------------
# Laplacian positional encodings (NumPy-only, deterministic sign)
# ---------------------------------------------------------------------


def _deterministic_sign(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Fix the sign of an eigenvector deterministically:
    - Prefer sign such that the sum is nonnegative.
    - If sum ~ 0, flip to make the first nonzero entry positive.
    """
    s = float(x.sum())
    if abs(s) > eps:
        return x if s >= 0.0 else -x
    nz = np.where(np.abs(x) > eps)[0]
    if nz.size == 0:
        return x  # zero vector; nothing to fix
    return x if x[nz[0]] >= 0.0 else -x


def laplacian_pe(
    edge_index: np.ndarray,
    n: int | None = None,
    *,
    k: int = 8,
    normalized: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Compute the first k Laplacian eigenvectors (skip the trivial constant one).

    - If normalized=True, use the symmetric normalized Laplacian:
        L = I - D^{-1/2} A D^{-1/2}
      Otherwise, use the combinatorial Laplacian:
        L = D - A
    - Eigenvectors have deterministic sign.

    Parameters
    ----------
    edge_index : (2, E) int64
        Undirected, canonically coalesced (u < v) is preferred but not required.
    n : int, optional
        Number of nodes. If None, inferred as 1 + max(edge_index).
    k : int
        Number of non-trivial eigenvectors to return.
    normalized : bool
        Use symmetric normalized Laplacian if True.
    dtype : np.dtype
        Output dtype.

    Returns
    -------
    (n, k) float32/float64 (dtype)
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")

    if edge_index.size == 0:
        n = 0 if n is None else int(n)
    else:
        n_infer = int(edge_index.max()) + 1
        n = n_infer if n is None else int(n)

    if n <= 0:
        return np.zeros((0, k), dtype=dtype)

    # Build dense adjacency (NumPy-only for portability/simplicity).
    a = np.zeros((n, n), dtype=np.float64)
    if edge_index.size:
        u = edge_index[0].astype(np.int64, copy=False)
        v = edge_index[1].astype(np.int64, copy=False)
        a[u, v] = 1.0
        a[v, u] = 1.0

    d = a.sum(axis=1)
    if normalized:
        with np.errstate(divide="ignore"):
            inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        inv_sqrt[d <= 0.0] = 0.0
        d_inv = np.diag(inv_sqrt)
        # L = I - D^{-1/2} A D^{-1/2}
        lap = np.eye(n, dtype=np.float64) - d_inv @ a @ d_inv
    else:
        # L = D - A
        lap = np.diag(d) - a

    # Eigen-decomposition (symmetric)
    w, vecs = np.linalg.eigh(lap)
    # Sort by eigenvalue ascending
    order = np.argsort(w)
    w = w[order]
    vecs = vecs[:, order]

    # Skip the trivial constant eigenvector (index 0) if present
    start = 1 if w.size > 0 else 0
    # Take up to k vectors
    take = max(0, min(k, vecs.shape[1] - start))
    if take == 0:
        return np.zeros((n, 0), dtype=dtype)

    pe = vecs[:, start : start + take]
    # Deterministic sign per vector
    for j in range(pe.shape[1]):
        pe[:, j] = _deterministic_sign(pe[:, j])

    return pe.astype(dtype, copy=False)


# ---------------------------------------------------------------------
# Role encodings (simple degree one-hot)
# ---------------------------------------------------------------------


def degree_role_onehot(
    degrees: np.ndarray,
    *,
    max_deg: int | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Degree-based role encoding: one-hot bins over {0,1,...,max_deg}.

    Parameters
    ----------
    degrees : (n,) int64
    max_deg : int, optional
        If None, use degrees.max(). Caps the number of columns to max_deg+1.
    dtype : np.dtype

    Returns
    -------
    (n, max_deg+1) array
    """
    if degrees.ndim != 1:
        raise ValueError(f"degrees must be 1-D (got {degrees.shape})")
    d = degrees.astype(np.int64, copy=False)
    n = d.shape[0]
    dmax = int(d.max()) if n > 0 else 0
    bins = max_deg if max_deg is not None else dmax
    bins = max(0, int(bins))
    out = np.zeros((n, bins + 1), dtype=dtype)
    # clamp to [0, D] then set one-hot
    d_clamp = np.clip(d, 0, bins)
    rows = np.arange(n, dtype=np.int64)
    out[rows, d_clamp] = 1.0
    return out


# ---------------------------------------------------------------------
# Bitstrings for hypercubes (2^dim nodes, lexicographic order)
# ---------------------------------------------------------------------


def hypercube_bitstrings(
    dim: int,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Return bitstring features for Q_dim with nodes labeled 0..2^dim-1.

    Parameters
    ----------
    dim : int
        Hypercube dimension (>= 0).
    dtype : np.dtype

    Returns
    -------
    (2^dim, dim) array with entries in {0,1} (as floats).
    """
    if dim < 0:
        raise ValueError("dim must be >= 0")
    n = 1 << dim
    if n == 0:
        return np.zeros((0, dim), dtype=dtype)
    # Build by bit masking
    bits = np.zeros((n, dim), dtype=np.uint8)
    for b in range(dim):
        mask = 1 << (dim - 1 - b)  # MSB first for a nice lexicographic layout
        bits[:, b] = ((np.arange(n, dtype=np.int64) & mask) > 0).astype(np.uint8)
    return bits.astype(dtype, copy=False)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def safe_concat(
    *arrays: np.ndarray,
    axis: int = 1,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Concatenate feature blocks that may be None or have zero columns.

    - Skips None inputs.
    - Casts to `dtype`.
    - If all inputs are None/empty, returns shape (n, 0).

    All inputs must share the same number of rows.
    """
    blocks: list[np.ndarray] = []
    n: int | None = None
    for arr in arrays:
        if arr is None:
            continue
        if arr.ndim != 2:
            raise ValueError("all arrays must be 2-D")
        if n is None:
            n = arr.shape[0]
        elif arr.shape[0] != n:
            raise ValueError("all arrays must have the same number of rows")
        if arr.shape[1] == 0:
            continue
        blocks.append(arr.astype(dtype, copy=False))

    if n is None:
        return np.zeros((0, 0), dtype=dtype)
    if not blocks:
        return np.zeros((n, 0), dtype=dtype)

    return np.concatenate(blocks, axis=axis)

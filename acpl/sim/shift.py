# acpl/sim/shift.py
from __future__ import annotations

import numpy as np

from .portmap import PortMap

try:  # optional torch support
    import torch
except Exception:  # pragma: no cover - torch may be absent
    torch = None  # type: ignore[assignment]

try:  # optional scipy (useful for debugging / CPU checks)
    import scipy.sparse as sp
except Exception:  # pragma: no cover - scipy may be absent
    sp = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Core constructors
# ---------------------------------------------------------------------------


def build_shift_index(pm: PortMap) -> np.ndarray:
    """
    Return an integer mapping 'dest' such that, for an arc-indexed state vector psi:
        psi_next = psi[dest]
    implements the flip-flop shift S (i.e., maps each arc to its reverse arc).

    Parameters
    ----------
    pm : PortMap
        Port map with 'rev' pairing (u->v) <-> (v->u).

    Returns
    -------
    dest : np.ndarray (num_arcs,), dtype=int64
        For every column index 'a', the destination row index is dest[a] = pm.rev[a].
        Equivalently, S has ones at coordinates (row=dest[a], col=a).
    """
    # Copy to avoid accidental in-place modification later.
    return pm.rev.copy()


def build_shift_torch(
    pm: PortMap,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Build the flip-flop shift as a Torch sparse COO permutation matrix s.

    The convention is column-vector multiplication: psi_next = s @ psi.
    Therefore, s[row, col] = 1 when row = pm.rev[col].
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is not available, cannot build a torch sparse S.")

    a = pm.num_arcs
    cols = torch.arange(a, device=device, dtype=torch.int64)
    rows = torch.as_tensor(pm.rev, device=device, dtype=torch.int64)

    if dtype is None:
        dtype = torch.float32

    indices = torch.stack([rows, cols], dim=0)  # shape (2, A)
    values = torch.ones(a, device=device, dtype=dtype)
    shape = (a, a)
    s = torch.sparse_coo_tensor(indices, values, shape, device=device, dtype=dtype)
    # Ensure canonical (coalesced) form.
    s = s.coalesce()
    return s


def build_shift_scipy(pm: PortMap, dtype: np.dtype = np.float32) -> sp.coo_matrix:
    """
    Build the flip-flop shift as a SciPy sparse COO permutation matrix s.

    psi_next = s @ psi (column-vector convention).
    """
    if sp is None:  # pragma: no cover
        raise RuntimeError("SciPy is not available, cannot build a SciPy sparse S.")

    a = pm.num_arcs
    rows = pm.rev
    cols = np.arange(a, dtype=np.int64)
    data = np.ones(a, dtype=dtype)
    return sp.coo_matrix((data, (rows, cols)), shape=(a, a))


# ---------------------------------------------------------------------------
# Convenience (vector-only) applications
# ---------------------------------------------------------------------------


def apply_shift_numpy(psi: np.ndarray, dest: np.ndarray) -> np.ndarray:
    """
    Apply the shift to a numpy state vector using the index mapping.

    Parameters
    ----------
    psi : np.ndarray (A,) or (A, k)
        Arc-indexed state (real or complex). If 2-D, columns are treated as
        independent vectors.
    dest : np.ndarray (A,), dtype=int64
        Mapping as returned by `build_shift_index`.

    Returns
    -------
    psi_next : np.ndarray
        psi_next = psi[dest] (row-wise for 1-D; along axis=0 for 2-D).
    """
    if psi.ndim == 1:
        return psi[dest]
    if psi.ndim == 2:
        return psi[dest, :]
    raise ValueError(f"psi must be 1-D or 2-D (got ndim={psi.ndim})")


def apply_shift_torch(
    psi: torch.Tensor,
    dest: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the shift to a torch state vector using an index mapping tensor.

    Parameters
    ----------
    psi : torch.Tensor (A,) or (A, k)
        Arc-indexed state tensor on some device (real or complex).
    dest : torch.Tensor (A,), dtype=torch.long
        Destination index mapping (e.g., torch.from_numpy(pm.rev)) on the same device.

    Returns
    -------
    psi_next : torch.Tensor
        psi_next = psi.index_select(0, dest). For 2-D input, selection is
        along dim=0 (rows).
    """
    if psi.ndim == 1:
        return psi.index_select(0, dest)
    if psi.ndim == 2:
        return psi.index_select(0, dest)
    raise ValueError(f"psi must be 1-D or 2-D (got ndim={psi.ndim})")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def is_permutation_matrix_numpy(mat: object) -> bool:
    """
    Quick check: matrix should be square, binary {0,1}, and each row/col sums to 1.
    Works for numpy arrays or scipy.sparse matrices.
    """
    # Accept dense ndarray
    if isinstance(mat, np.ndarray):
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            return False
        if not np.all((mat == 0) | (mat == 1)):
            return False
        row_ok = np.allclose(mat.sum(axis=1), 1)
        col_ok = np.allclose(mat.sum(axis=0), 1)
        return bool(row_ok and col_ok)

    # Accept scipy sparse
    if sp is not None and sp.issparse(mat):  # pragma: no cover (simple)
        if mat.shape[0] != mat.shape[1]:
            return False
        ones = np.ones(mat.shape[0], dtype=np.float32)
        row_ok = np.allclose(np.asarray(mat @ ones).ravel(), ones)
        col_ok = np.allclose(np.asarray(mat.T @ ones).ravel(), ones)
        # Check binary data
        if hasattr(mat, "data"):
            data_ok = np.all((mat.data == 0) | (mat.data == 1))
        else:
            data_ok = True
        return bool(row_ok and col_ok and data_ok)

    return False

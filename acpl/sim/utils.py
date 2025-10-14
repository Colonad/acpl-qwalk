# acpl/sim/utils.py
from __future__ import annotations

from dataclasses import dataclass

try:  # Torch is required for simulator/backprop.
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("acpl.sim.utils requires PyTorch to be installed.") from exc

from .portmap import PortMap

# -----------------------------------------------------------------------------
# DType / device helpers
# -----------------------------------------------------------------------------


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """
    Map a real dtype to its complex counterpart.
    float32 -> complex64, float64 -> complex128 (defaults to complex64 otherwise).
    """
    if real_dtype in (torch.float32, torch.complex64):
        return torch.complex64
    if real_dtype in (torch.float64, torch.complex128):
        return torch.complex128
    return torch.complex64


def canonical_device_dtype(
    *tensors: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.device, torch.dtype]:
    """
    Infer (device, dtype) from provided tensors, optionally overridden by args.
    If no tensors are given, fall back to (cpu, float32).
    """
    dev = device
    dt = dtype
    for t in tensors:
        if dev is None:
            dev = t.device
        if dt is None:
            dt = t.dtype
    if dev is None:
        dev = torch.device("cpu")
    if dt is None:
        dt = torch.float32
    return dev, dt


def ensure_complex(x: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Ensure a complex dtype for a state/coin tensor, preserving device and layout.
    """
    if x.is_complex():
        if dtype is None or x.dtype == dtype:
            return x
        return x.to(dtype=dtype)
    cdt = complex_dtype_for(dtype or x.dtype)
    return x.to(dtype=cdt)


# -----------------------------------------------------------------------------
# PortMap → torch index tensors
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PortMapTensors:
    """Torch views of PortMap indexing arrays (long dtype)."""

    tail: torch.Tensor  # (A,)
    head: torch.Tensor  # (A,)
    rev: torch.Tensor  # (A,)
    node_ptr: torch.Tensor  # (V+1,)
    node_arcs: torch.Tensor  # (A,)


def portmap_tensors(pm: PortMap, *, device: torch.device | None = None) -> PortMapTensors:
    """
    Convert numpy arrays in PortMap to torch.LongTensor on the requested device.
    """
    dev = device or torch.device("cpu")
    tl = torch.from_numpy(pm.tail).to(device=dev, dtype=torch.long)
    hd = torch.from_numpy(pm.head).to(device=dev, dtype=torch.long)
    rv = torch.from_numpy(pm.rev).to(device=dev, dtype=torch.long)
    nptr = torch.from_numpy(pm.node_ptr).to(device=dev, dtype=torch.long)
    narcs = torch.from_numpy(pm.node_arcs).to(device=dev, dtype=torch.long)
    return PortMapTensors(tl, hd, rv, nptr, narcs)


def degrees_from_portmap(pm: PortMap, *, device: torch.device | None = None) -> torch.Tensor:
    """
    Return degrees (outgoing ports) per vertex as a LongTensor of shape (V,).
    """
    pt = portmap_tensors(pm, device=device)
    return (pt.node_ptr[1:] - pt.node_ptr[:-1]).to(dtype=torch.long)


# -----------------------------------------------------------------------------
# State helpers
# -----------------------------------------------------------------------------


def normalize_state(psi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize a state vector along axis 0 (arc axis).
    Accepts shapes (A,) or (A, B). Returns a tensor with the same shape.
    """
    if psi.ndim == 1:
        norm2 = torch.dot(psi.conj(), psi).real
        denom = torch.sqrt(norm2 + eps)
        return psi / denom
    if psi.ndim == 2:
        norm2 = (psi.conj() * psi).real.sum(dim=0, keepdim=True)  # (1, B)
        denom = torch.sqrt(norm2 + eps)
        return psi / denom
    raise ValueError(f"psi must be 1-D or 2-D (got ndim={psi.ndim})")


# -----------------------------------------------------------------------------
# Partial trace over the coin → position distribution P(v)
# -----------------------------------------------------------------------------


def partial_trace_position(psi: torch.Tensor, pm: PortMap) -> torch.Tensor:
    """
    Sum magnitudes over ports (arcs) per vertex to obtain position probabilities P.
    psi: (A,) or (A,B) complex (or real) wavefunction.
    Returns:
        (V,) for 1-D input, or (V,B) for 2-D input.
    """
    dev = psi.device
    pt = portmap_tensors(pm, device=dev)
    num_arcs = pm.num_arcs
    num_nodes = pm.num_nodes

    if psi.ndim == 1:
        if psi.numel() != num_arcs:
            raise ValueError(f"psi shape mismatch: expected (A,), got {tuple(psi.shape)}")
        # Group arcs by tail vertex
        psi_lumped = psi.index_select(0, pt.node_arcs)  # (A,)
        magsq = (psi_lumped.conj() * psi_lumped).real  # (A,)

        # Build per-arc vertex ids: repeat each vertex id by its degree
        deg = pt.node_ptr[1:] - pt.node_ptr[:-1]  # (V,)
        arc_vid = torch.arange(num_nodes, device=dev).repeat_interleave(deg)  # (A,)

        p = torch.zeros(num_nodes, device=dev, dtype=magsq.dtype)  # (V,)
        p.index_add_(0, arc_vid, magsq)
        return p

    if psi.ndim == 2:
        a0, b = psi.shape
        if a0 != num_arcs:
            raise ValueError(f"psi shape mismatch: expected (A,B), got {tuple(psi.shape)}")
        psi_lumped = psi.index_select(0, pt.node_arcs)  # (A,B)
        magsq = (psi_lumped.conj() * psi_lumped).real  # (A,B)

        # Per-arc vertex ids (A,)
        deg = pt.node_ptr[1:] - pt.node_ptr[:-1]  # (V,)
        arc_vid = torch.arange(num_nodes, device=dev).repeat_interleave(deg)  # (A,)

        # Accumulate along dim 0 into (V,B)
        p = torch.zeros(num_nodes, b, device=dev, dtype=magsq.dtype)
        p.index_add_(0, arc_vid, magsq)
        return p

    raise ValueError(f"psi must be 1-D or 2-D (got ndim={psi.ndim}).")


# -----------------------------------------------------------------------------
# Block-diagonal builders for arbitrary degree sets (sparse COO)
# -----------------------------------------------------------------------------


def _invert_permutation(idx: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), device=idx.device, dtype=idx.dtype)
    return inv


def coerce_coin_blocks_list(
    c_blocks: torch.Tensor | list[torch.Tensor],
    pm: PortMap,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> list[torch.Tensor]:
    """
    Ensure a list[V] of per-vertex coin blocks with shapes (d_v, d_v) and complex dtype.
    - If `c_blocks` is (V,2,2) and all degrees are 2, convert to list.
    - If `c_blocks` is already a list, validate shapes and coerce dtype/device.
    """
    dev, dt = canonical_device_dtype(device=device, dtype=dtype)
    pt = portmap_tensors(pm, device=dev)
    v = pm.num_nodes
    deg_v = (pt.node_ptr[1:] - pt.node_ptr[:-1]).tolist()

    if isinstance(c_blocks, torch.Tensor):
        if c_blocks.ndim != 3 or c_blocks.shape[0] != v or c_blocks.shape[-2:] != (2, 2):
            raise ValueError("tensor c_blocks must have shape (V, 2, 2) for degree-2 graphs.")
        if any(d != 2 for d in deg_v):
            raise ValueError(
                "tensor c_blocks provided but graph has mixed degrees; pass a list instead."
            )
        lst = [ensure_complex(c_blocks[i].to(device=dev, dtype=dt)) for i in range(v)]
        return lst

    if not isinstance(c_blocks, (list, tuple)) or len(c_blocks) != v:
        raise ValueError("c_blocks must be a list of length V with per-vertex (d_v, d_v) tensors.")

    out: list[torch.Tensor] = []
    for i in range(v):
        dv = deg_v[i]
        m = ensure_complex(c_blocks[i].to(device=dev, dtype=dt))
        if m.ndim != 2 or m.shape != (dv, dv):
            raise ValueError(f"c_blocks[{i}] must have shape ({dv}, {dv}), got {tuple(m.shape)}")
        out.append(m)
    return out


def build_blkdiag_lumped_sparse(
    c_blocks: list[torch.Tensor],
    pm: PortMap,
) -> torch.Tensor:
    """
    Build a sparse COO block-diagonal matrix (A x A) in the vertex-lumped arc order.

    In the lumped order, arcs belonging to the same vertex are contiguous, so
    the block-diagonal is simply the concatenation of per-vertex diagonal blocks.
    """
    pt = portmap_tensors(pm, device=c_blocks[0].device)
    dev = c_blocks[0].device
    cdt = c_blocks[0].dtype
    a = pm.num_arcs

    # Offsets of each vertex's arc segment in the lumped order: [0, d0, d0+d1, ...]
    deg = (pt.node_ptr[1:] - pt.node_ptr[:-1]).to(dtype=torch.long)
    offsets = torch.zeros(pm.num_nodes + 1, device=dev, dtype=torch.long)
    offsets[1:] = torch.cumsum(deg, dim=0)
    # Collect indices/data
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    data: list[torch.Tensor] = []

    for v in range(pm.num_nodes):
        dv = int(deg[v].item())
        if dv == 0:
            continue
        off = int(offsets[v].item())
        # Dense block indices in local coordinates
        r = torch.arange(dv, device=dev)
        c = torch.arange(dv, device=dev)
        rr = r.repeat_interleave(dv)  # (dv*dv,)
        cc = c.repeat(dv)  # (dv*dv,)
        # Shift to global lumped coordinates
        rows.append(rr + off)
        cols.append(cc + off)
        data.append(c_blocks[v].reshape(-1))

    if not rows:
        # Empty graph (degenerate)
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), device=dev, dtype=torch.long),
            torch.empty((0,), device=dev, dtype=cdt),
            (a, a),
            device=dev,
            dtype=cdt,
        ).coalesce()

    rows_t = torch.cat(rows, dim=0)
    cols_t = torch.cat(cols, dim=0)
    data_t = torch.cat(data, dim=0).to(dtype=cdt)

    idx = torch.stack([rows_t, cols_t], dim=0)
    mat = torch.sparse_coo_tensor(idx, data_t, (a, a), device=dev, dtype=cdt)
    return mat.coalesce()


def build_blkdiag_arc_sparse(
    c_blocks: list[torch.Tensor],
    pm: PortMap,
) -> torch.Tensor:
    """
    Build a sparse COO block-diagonal matrix (A x A) in the original arc order.

    This is obtained from the lumped form by applying the permutation induced by
    `pt.node_arcs`: M_arc = P^T M_lumped P, which is equivalent to remapping the
    row/col indices via the permutation array.
    """
    pt = portmap_tensors(pm, device=c_blocks[0].device)
    dev = c_blocks[0].device
    cdt = c_blocks[0].dtype

    m_lumped = build_blkdiag_lumped_sparse(c_blocks, pm).coalesce()
    if m_lumped._nnz() == 0:
        return m_lumped

    # node_arcs maps lumped index -> arc index
    perm = pt.node_arcs  # (A,)
    row_l, col_l = m_lumped.indices()
    dat = m_lumped.values()

    row_a = perm.index_select(0, row_l)
    col_a = perm.index_select(0, col_l)
    idx_a = torch.stack([row_a, col_a], dim=0)

    a = pm.num_arcs
    m_arc = torch.sparse_coo_tensor(idx_a, dat.to(dtype=cdt), (a, a), device=dev, dtype=cdt)
    return m_arc.coalesce()

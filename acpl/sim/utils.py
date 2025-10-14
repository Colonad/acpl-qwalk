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

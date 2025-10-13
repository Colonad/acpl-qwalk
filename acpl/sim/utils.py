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


def partial_trace_position(
    psi: torch.Tensor,
    pm: PortMap,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute position distribution P(v) by tracing out the coin/port.

    In arc basis where each vertex v has a contiguous block of outgoing arcs
    (as given by pm.node_arcs[ pm.node_ptr[v]:pm.node_ptr[v+1] ]), the position
    probability is the squared magnitude summed over that block.

    Parameters
    ----------
    psi : torch.Tensor
        Arc-indexed state of shape (A,) or (A, B). Complex dtype recommended.
        If (A, B), columns are treated as independent states (minibatch).
    pm : PortMap
        Port map with CSR fields `node_ptr` and `node_arcs`.
    device : optional
        Device to place intermediate tensors. Defaults to psi.device.

    Returns
    -------
    torch.Tensor
        If psi is (A,), returns (V,).
        If psi is (A, B), returns (V, B).
        Each column sums to ~1 if psi is normalized.
    """
    dev = device or psi.device
    pt = portmap_tensors(pm, device=dev)
    num_arcs = pm.num_arcs

    if psi.ndim == 1:
        if psi.numel() != num_arcs:
            raise ValueError(f"psi shape mismatch: expected (A,), got {tuple(psi.shape)}")
        # Gather psi in node-lumped order
        psi_lumped = psi.index_select(0, pt.node_arcs)  # (A,)
        magsq = (psi_lumped.conj() * psi_lumped).real
        # Sum segments per vertex using prefix sums
        starts = pt.node_ptr[:-1]
        ends = pt.node_ptr[1:]
        # Vectorized segment sums via cumulative sum
        csum = torch.cat(
            [torch.zeros(1, device=dev, dtype=magsq.dtype), magsq.cumsum(0)],
            dim=0,
        )
        p = csum[ends] - csum[starts]
        return p

    if psi.ndim == 2:
        if psi.size(0) != num_arcs:
            raise ValueError(f"psi shape mismatch: expected (A, B), got {tuple(psi.shape)}")
        batch = psi.size(1)
        psi_lumped = psi.index_select(0, pt.node_arcs)  # (A, B)
        magsq = (psi_lumped.conj() * psi_lumped).real
        csum = torch.cat(
            [torch.zeros(1, batch, device=dev, dtype=magsq.dtype), magsq.cumsum(0)],
            dim=0,
        )  # (A+1, B)
        starts = pt.node_ptr[:-1].unsqueeze(1)  # (V,1)
        ends = pt.node_ptr[1:].unsqueeze(1)  # (V,1)
        p = csum.gather(0, ends) - csum.gather(0, starts)  # (V, B)
        return p.squeeze(1) if p.shape[1] == 1 else p

    raise ValueError(f"psi must be 1-D or 2-D (got ndim={psi.ndim})")

# acpl/sim/step.py
from __future__ import annotations

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("acpl.sim.step requires PyTorch to be installed.") from exc

from .portmap import PortMap
from .shift import apply_shift_torch
from .utils import complex_dtype_for, portmap_tensors

# -----------------------------------------------------------------------------
# Coin application: psi' = ( ⊕_v C_v ) psi
# -----------------------------------------------------------------------------


def apply_coin_blockdiag(
    psi: torch.Tensor,
    c_blocks: torch.Tensor,
    pm: PortMap,
) -> torch.Tensor:
    """
    Apply the vertex-local coin blocks (block-diagonal) to an arc-indexed state.

    Parameters
    ----------
    psi : torch.Tensor
        Arc-indexed state, shape (A,) or (A, B), complex preferred.
    c_blocks : torch.Tensor
        Per-vertex coin blocks. For Phase A (degree-2 graphs), shape (V, 2, 2).
        For future mixed-degree support, blocks must match each vertex degree.
    pm : PortMap
        Provides the CSR mapping from vertices to their outgoing arcs.

    Returns
    -------
    torch.Tensor
        State with coins applied, same shape as `psi`.
    """
    if psi.ndim not in (1, 2):
        raise ValueError(f"psi must be 1-D or 2-D (got ndim={psi.ndim})")

    dev = psi.device
    pt = portmap_tensors(pm, device=dev)
    num_arcs = pm.num_arcs
    num_nodes = pm.num_nodes

    if psi.size(0) != num_arcs:
        want = "(A,)" if psi.ndim == 1 else "(A, B)"
        raise ValueError(f"psi shape mismatch: expected {want}, got {tuple(psi.shape)}")

    # Ensure complex dtype (coins are complex in general).
    cdt = psi.dtype if psi.is_complex() else complex_dtype_for(psi.dtype)
    psi = psi.to(dtype=cdt)
    c_blocks = c_blocks.to(device=dev, dtype=cdt)

    # Gather arcs grouped by tail vertex: contiguous by vertex
    psi_lumped = psi.index_select(0, pt.node_arcs)

    # Fast degree-2 path (Phase A)
    deg_v = torch.diff(pt.node_ptr)  # (V,)
    if torch.all(deg_v == 2) and c_blocks.shape[-2:] == (2, 2) and c_blocks.size(0) == num_nodes:
        if psi.ndim == 1:
            x = psi_lumped.view(num_nodes, 2)  # (V, 2)
            y = torch.matmul(c_blocks, x.unsqueeze(-1)).squeeze(-1)  # (V, 2)
            psi_coin = y.view(num_arcs)
        else:
            batch = psi.size(1)
            x = psi_lumped.view(num_nodes, 2, batch)  # (V, 2, B)
            # (V, 2, 2) @ (V, 2, B) -> (V, 2, B)
            y = torch.matmul(c_blocks, x)
            psi_coin = y.view(num_arcs, batch)
    else:
        # Mixed-degree fallback: loop per vertex (correct, not the Phase A hot path).
        if psi.ndim == 1:
            out = torch.empty_like(psi, dtype=cdt)
            for v in range(num_nodes):
                s, e = pt.node_ptr[v].item(), pt.node_ptr[v + 1].item()
                dv = e - s
                if c_blocks.size(0) != num_nodes or c_blocks[v].shape[-2:] != (dv, dv):
                    raise ValueError(
                        f"c_blocks for vertex {v} must be ({dv},{dv}), "
                        f"got {tuple(c_blocks[v].shape[-2:])}"
                    )
                out[pt.node_arcs[s:e]] = (c_blocks[v] @ psi_lumped[s:e]).to(cdt)
            psi_coin = out
        else:
            batch = psi.size(1)
            out = torch.empty_like(psi, dtype=cdt)
            for v in range(num_nodes):
                s, e = pt.node_ptr[v].item(), pt.node_ptr[v + 1].item()
                dv = e - s
                if c_blocks.size(0) != num_nodes or c_blocks[v].shape[-2:] != (dv, dv):
                    raise ValueError(
                        f"c_blocks for vertex {v} must be ({dv},{dv}), "
                        f"got {tuple(c_blocks[v].shape[-2:])}"
                    )
                # (dv, dv) @ (dv, B) -> (dv, B)
                out[pt.node_arcs[s:e], :] = (c_blocks[v] @ psi_lumped[s:e, :]).to(cdt)
            psi_coin = out

    # Undo the vertex-lumped permutation to revert to arc order.
    inv = torch.empty_like(pt.node_arcs)
    inv[pt.node_arcs] = torch.arange(num_arcs, device=dev, dtype=torch.long)
    if psi.ndim == 1:
        return psi_coin.index_select(0, inv)
    return psi_coin.index_select(0, inv)


# -----------------------------------------------------------------------------
# One DTQW step: psi <- S @ blkdiag(C) @ psi
# -----------------------------------------------------------------------------


def step(
    psi: torch.Tensor,
    pm: PortMap,
    c_blocks: torch.Tensor,
    *,
    dest: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    One DTQW step with flip-flop shift:

        psi <- S @ ( ⊕_v C_v ) @ psi
    """
    dev = psi.device
    # Apply coin (block-diagonal)
    psi_after_coin = apply_coin_blockdiag(psi, c_blocks, pm)

    # Apply shift via index-select mapping (fastest for permutations).
    if dest is None:
        dest = torch.from_numpy(pm.rev).to(device=dev, dtype=torch.long)
    psi_next = apply_shift_torch(psi_after_coin, dest)
    return psi_next

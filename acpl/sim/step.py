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
# Helpers
# -----------------------------------------------------------------------------


def _is_tensor_blocks(c_blocks: object) -> bool:
    return isinstance(c_blocks, torch.Tensor)


def _is_list_blocks(c_blocks: object) -> bool:
    return isinstance(c_blocks, (list, tuple)) and all(
        isinstance(x, torch.Tensor) for x in c_blocks
    )


def _invert_permutation(idx: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), device=idx.device, dtype=idx.dtype)
    return inv


# -----------------------------------------------------------------------------
# Coin application: psi' = ( ⊕_v C_v ) psi
# -----------------------------------------------------------------------------


def apply_coin_blockdiag(
    psi: torch.Tensor,
    c_blocks: torch.Tensor | list[torch.Tensor],
    pm: PortMap,
) -> torch.Tensor:
    """
    Apply the vertex-local coin blocks (block-diagonal) to an arc-indexed state.

    Parameters
    ----------
    psi : torch.Tensor
        Arc-indexed state, shape (A,) or (A, B), complex preferred.
    c_blocks : torch.Tensor | list[torch.Tensor]
        - Phase A: (V, 2, 2) tensor for degree-2 graphs.
        - Phase B (mixed degrees): list of length V; item v has shape (d_v, d_v).
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

    # Gather arcs grouped by tail vertex: contiguous by vertex
    psi_lumped = psi.index_select(0, pt.node_arcs)

    deg_v = torch.diff(pt.node_ptr)  # (V,)
    all_deg2 = bool(torch.all(deg_v == 2).item())

    # ----------------------- Fast degree-2 path -----------------------
    if _is_tensor_blocks(c_blocks) and all_deg2:
        cb = c_blocks.to(device=dev, dtype=cdt)
        if cb.ndim != 3 or cb.shape != (num_nodes, 2, 2):
            raise ValueError("For degree-2 graphs, c_blocks must have shape (V, 2, 2).")

        if psi.ndim == 1:
            x = psi_lumped.view(num_nodes, 2)  # (V, 2)
            y = torch.matmul(cb, x.unsqueeze(-1)).squeeze(-1)  # (V, 2)
            psi_coin = y.view(num_arcs)
        else:
            bsz = psi.size(1)
            x = psi_lumped.view(num_nodes, 2, bsz)  # (V, 2, B)
            # (V, 2, 2) @ (V, 2, B) -> (V, 2, B)
            y = torch.matmul(cb, x)
            psi_coin = y.view(num_arcs, bsz)

    # ----------------- Mixed-degree (grouped by degree) ---------------
    else:
        # Normalize c_blocks to list[V] for mixed path. If user passed a tensor
        # with shapes (V, d, d) but degrees vary (unsupported tensor), we require list.
        if _is_tensor_blocks(c_blocks):
            raise ValueError(
                "Mixed-degree graphs require c_blocks as a list of per-vertex tensors."
            )
        if not _is_list_blocks(c_blocks) or len(c_blocks) != num_nodes:
            raise ValueError("c_blocks must be a list of length V with (d_v, d_v) tensors.")

        # Check shapes and dtype, build degree->vertex indices
        groups: dict[int, list[int]] = {}
        for v in range(num_nodes):
            s = int(pt.node_ptr[v].item())
            e = int(pt.node_ptr[v + 1].item())
            dv = e - s
            m = c_blocks[v].to(device=dev, dtype=cdt)
            if m.ndim != 2 or m.shape != (dv, dv):
                raise ValueError(
                    f"c_blocks[{v}] must have shape ({dv}, {dv}), got {tuple(m.shape)}"
                )
            c_blocks[v] = m  # ensure on correct device/dtype
            groups.setdefault(dv, []).append(v)

        # Output buffer
        out = torch.empty_like(psi, dtype=cdt)

        # For each degree group, batch the vertices for matmul
        for deg, vs in groups.items():
            vs_t = torch.tensor(vs, device=dev, dtype=torch.long)
            # Arc segment for all these vertices in the lumped order
            # Build a gather index of length sum(deg) = deg * |vs|
            starts = pt.node_ptr[vs_t]  # (|vs|,)
            seg_idx = torch.arange(deg, device=dev).repeat(len(vs))  # (deg*|vs|,)
            # Map (vertex, local_port) -> global lumped arc index
            base = starts.repeat_interleave(deg)  # (deg*|vs|,)
            gl_idx = base + seg_idx  # (deg*|vs|,)

            if psi.ndim == 1:
                x = psi_lumped.index_select(0, gl_idx).view(len(vs), deg)  # (|vs|, deg)
                # Stack blocks -> (|vs|, deg, deg)
                cst = torch.stack([c_blocks[v] for v in vs], dim=0)
                # (|vs|, deg, deg) @ (|vs|, deg, 1) -> (|vs|, deg, 1)
                y = torch.matmul(cst, x.unsqueeze(-1)).squeeze(-1)  # (|vs|, deg)
                # Scatter back
                out.index_copy_(0, pt.node_arcs[gl_idx], y.view(-1))
            else:
                bsz = psi.size(1)
                x = psi_lumped.index_select(0, gl_idx).view(len(vs), deg, bsz)  # (|vs|,deg,B)
                cst = torch.stack([c_blocks[v] for v in vs], dim=0)  # (|vs|,deg,deg)
                y = torch.matmul(cst, x)  # (|vs|,deg,B)
                out.index_copy_(0, pt.node_arcs[gl_idx], y.view(-1, bsz))

        psi_coin = out

    # Undo the vertex-lumped permutation to revert to arc order.
    inv = _invert_permutation(pt.node_arcs)
    if psi.ndim == 1:
        return psi_coin.index_select(0, inv)
    return psi_coin.index_select(0, inv)


# -----------------------------------------------------------------------------
# One DTQW step: psi <- S @ blkdiag(C) @ psi
# -----------------------------------------------------------------------------


def step(
    psi: torch.Tensor,
    pm: PortMap,
    c_blocks: torch.Tensor | list[torch.Tensor],
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

from __future__ import annotations

import torch


def sum_by_dst_sorted(
    edge_index: torch.Tensor,
    msg: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Deterministic sum aggregation of edge messages into destination nodes.

    Parameters
    ----------
    edge_index : (2, E) long
        edge_index[0]=src, edge_index[1]=dst
    msg : (E, D) tensor
        message per edge
    num_nodes : int
        number of nodes N

    Returns
    -------
    out : (N, D) tensor
        out[dst] = sum_{edges to dst} msg[e]
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must be torch.long")
    if msg.ndim < 1:
        raise ValueError("msg must have shape (E, ...)")

    src = edge_index[0]
    dst = edge_index[1]
    E = dst.numel()
    if msg.shape[0] != E:
        raise ValueError(f"msg first dim must be E={E}, got {msg.shape[0]}")

    # Sort by (dst, src) to make reduction order deterministic.
    # key = dst * N + src fits in int64 for your N (<= few 1e6 easily).
    key = dst * int(num_nodes) + src
    perm = torch.argsort(key, stable=True)
    dst_s = dst.index_select(0, perm)
    msg_s = msg.index_select(0, perm)

    # Build rowptr via counts per dst (bincount is deterministic)
    counts = torch.bincount(dst_s, minlength=int(num_nodes))
    rowptr = torch.empty(int(num_nodes) + 1, device=counts.device, dtype=torch.long)
    rowptr[0] = 0
    rowptr[1:] = torch.cumsum(counts, dim=0)

    out = msg_s.new_zeros((int(num_nodes),) + msg_s.shape[1:])

    # Deterministic per-node reduction by contiguous slice sum (no atomics)
    # N in your project is small enough (<=256 typical); this is fast enough and test-safe.
    for v in range(int(num_nodes)):
        a = int(rowptr[v].item())
        b = int(rowptr[v + 1].item())
        if b > a:
            out[v] = msg_s[a:b].sum(dim=0)

    return out

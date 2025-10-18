# acpl/sim/portmap.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

__all__ = [
    "PortMap",
    "build_portmap",
    "undirected_bidir",
]

TensorLike = Union[torch.Tensor, np.ndarray]
EdgeIndexLike = Union[TensorLike, Iterable[tuple[int, int]]]


# ------------------------------- Utilities --------------------------------- #


def _to_edge_index_2xE(
    edge_index: EdgeIndexLike,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """
    Convert various edge_index shapes/types to torch.LongTensor with shape (2, E).

    Accepts:
      - torch.Tensor of shape (2, E) or (E, 2)
      - np.ndarray of shape (2, E) or (E, 2)
      - iterable of (u, v) pairs

    Returns:
      torch.LongTensor (2, E) on `device`, dtype `dtype`
    """
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index.to(dtype=dtype, device=device)
        if ei.ndim != 2:
            raise ValueError("edge_index tensor must be 2-D")
        if ei.shape[0] == 2:
            return ei
        if ei.shape[1] == 2:
            return ei.T.contiguous()
        raise ValueError("edge_index must have shape (2, E) or (E, 2)")
    elif isinstance(edge_index, np.ndarray):
        arr = np.asarray(edge_index, dtype=np.int64)
        if arr.ndim != 2:
            raise ValueError("edge_index array must be 2-D")
        if arr.shape[0] == 2:
            pass
        elif arr.shape[1] == 2:
            arr = arr.T
        else:
            raise ValueError("edge_index must have shape (2, E) or (E, 2)")
        return torch.as_tensor(arr, dtype=dtype, device=device)
    else:
        # Iterable of pairs
        pairs = list(edge_index)  # exhaust
        if len(pairs) == 0:
            return torch.empty((2, 0), dtype=dtype, device=device)
        arr = np.array(pairs, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Iterable edge_index must yield (u, v) pairs")
        return torch.as_tensor(arr.T, dtype=dtype, device=device)


def undirected_bidir(
    edge_index: EdgeIndexLike,
    num_nodes: int | None = None,
    *,
    coalesce: bool = True,
    allow_self_loops: bool = False,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Ensure an undirected graph is represented with BOTH directions (u,v) and (v,u).

    - Removes self-loops unless allow_self_loops=True.
    - Optionally coalesces duplicated edges (multi-edges) into stable
      pairs by multiplicity (keeps counts).
    - Returns (edge_index_directed, num_nodes) where the directed edge_index is (2, E_dir).

    Notes on multigraph pairing:
      For each unordered pair {u,v}, we count multiplicity m. We then generate
      m directed copies in each direction and *pair them by rank*; this is
      critical for a deterministic flip-flop mapping downstream.

    Returns:
      directed_edge_index: (2, E_dir)
      num_nodes: inferred if not provided
    """
    ei = _to_edge_index_2xE(edge_index, device=device, dtype=torch.long)
    if num_nodes is None:
        # Infer from max index
        num_nodes = (ei.max().item() + 1) if ei.numel() > 0 else 0

    # Drop invalid edges and normalize type
    u, v = ei[0], ei[1]
    mask_valid = (u >= 0) & (u < num_nodes) & (v >= 0) & (v < num_nodes)
    if not allow_self_loops:
        mask_valid = mask_valid & (u != v)
    if mask_valid.sum().item() != ei.shape[1]:
        u = u[mask_valid]
        v = v[mask_valid]

    if u.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device), num_nodes

    # We need a deterministic multiplicity-aware expansion in both directions.
    # Strategy:
    #  1) Build dictionary for unordered edge -> list of (u,v) occurrences (stable order).
    #  2) For each unordered key, sort its directional occurrences by (src,dst) to stabilize,
    #     then pair by rank to create symmetric directed multiedges.
    # Step (1): stable bucket by key=(min(u,v), max(u,v))
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()

    buckets: dict[tuple[int, int], dict[str, list]] = {}
    for i in range(u_np.shape[0]):
        a, b = int(u_np[i]), int(v_np[i])
        key = (a, b) if a <= b else (b, a)
        if key not in buckets:
            buckets[key] = {"uv": [], "vu": []}
        if a <= b:
            buckets[key]["uv"].append((a, b))
        else:
            buckets[key]["vu"].append((a, b))  # store as (src, dst) even if it's "reverse"

    # Step (2): for each key, match counts in both directions.
    dir_src: list[int] = []
    dir_dst: list[int] = []

    for (a, b), parts in buckets.items():
        # Collect lists of occurrences in both directions
        fwd = parts["uv"]
        rev = [(y, x) for (x, y) in parts["vu"]]  # convert to consistent orientation (a,b)
        # Stabilize order within each by lexicographic tie-break
        fwd.sort()  # already all (a,b)
        rev.sort()  # all (a,b) after conversion

        m_fwd = len(fwd)
        m_rev = len(rev)
        m = max(m_fwd, m_rev)

        # Pad the smaller side by repeating the last element (if coalesce=False we keep true multiplicity)
        if coalesce:
            # Coalesce duplicates: treat as single edge in undirected sense
            m = 1

        # Now emit m directed instances in BOTH directions, pairing by rank.
        for k in range(m):
            # source instances
            # If one side had 0, synthesize a mirror to keep pairing stable
            f = fwd[min(k, max(m_fwd - 1, 0))] if m_fwd > 0 else (a, b)
            r = rev[min(k, max(m_rev - 1, 0))] if m_rev > 0 else (a, b)

            # Emit (a,b) and (b,a)
            dir_src.append(f[0])
            dir_dst.append(f[1])
            dir_src.append(f[1])
            dir_dst.append(f[0])

    directed = torch.tensor([dir_src, dir_dst], dtype=torch.long, device=device)
    return directed, num_nodes


# ------------------------------ PortMap class ------------------------------- #


@dataclass(frozen=True)
class PortMap:
    """
    Flip-flop port mapping of an undirected (multi)graph expressed over oriented arcs.

    Storage conventions:
      - We materialize a *directed* multigraph of size A arcs, arranged in CSR blocks by source node.
      - For an undirected edge with multiplicity m, we create m arcs (u->v) and m arcs (v->u).
      - The flip-flop shift S maps arc i to its reverse arc rev[i].
      - CSR structure (node_ptr) exposes per-node slices; within each node, outgoing arcs are
        sorted deterministically by (dst, local multiplicity rank).

    Attributes
    ----------
    num_nodes : int
    num_arcs  : int
    src, dst  : (A,) long tensors
    node_ptr  : (N+1,) long CSR pointer (arcs grouped by src node)
    port      : (A,) long local port index in [0, deg(src)-1]
    rev       : (A,) long index of reverse arc: applying flip-flop shift is psi[rev]
    deg       : (N,) long out-degree (undirected degree in this construction)
    """

    num_nodes: int
    num_arcs: int
    src: torch.Tensor  # (A,)
    dst: torch.Tensor  # (A,)
    node_ptr: torch.Tensor  # (N+1,)
    port: torch.Tensor  # (A,)
    rev: torch.Tensor  # (A,)
    deg: torch.Tensor  # (N,)

    # ----------------------------- Constructors ----------------------------- #

    @staticmethod
    def from_undirected(
        edge_index: EdgeIndexLike,
        num_nodes: int | None = None,
        *,
        allow_self_loops: bool = False,
        coalesce: bool = False,
        device: torch.device | None = None,
    ) -> PortMap:
        """
        Build a PortMap from an undirected edge list.

        Parameters
        ----------
        edge_index : (2,E) or (E,2) or iterable of (u,v)
            Undirected edges (order doesn't matter); multigraphs allowed.
        num_nodes : int, optional
            If omitted, inferred from max node id.
        allow_self_loops : bool
            If True, self-loop arcs are allowed and their reverse is themselves.
        coalesce : bool
            If True, coalesces multi-edges into a single flip-flop pair per unordered pair.
        device : torch.device, optional

        Returns
        -------
        PortMap
        """
        directed, n = undirected_bidir(
            edge_index,
            num_nodes=num_nodes,
            coalesce=coalesce,
            allow_self_loops=allow_self_loops,
            device=device,
        )
        if directed.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            node_ptr = torch.zeros(n + 1, dtype=torch.long, device=device)
            deg = torch.zeros(n, dtype=torch.long, device=device)
            return PortMap(
                num_nodes=n,
                num_arcs=0,
                src=empty,
                dst=empty,
                node_ptr=node_ptr,
                port=empty,
                rev=empty,
                deg=deg,
            )

        src = directed[0]
        dst = directed[1]
        device = src.device

        # CSR by src (deterministic: lexicographic by (src, dst))
        order = torch.lexsort((dst, src)) if hasattr(torch, "lexsort") else None
        if order is None:
            # Fallback for PyTorch without lexsort: stable argsort by src then dst
            _, order = torch.sort(src * (directed.shape[1] + 1) + dst)
        src = src[order]
        dst = dst[order]

        # Build CSR
        n = int(n)  # ensure python int
        counts = torch.bincount(src, minlength=n)
        node_ptr = torch.zeros(n + 1, dtype=torch.long, device=device)
        node_ptr[1:] = torch.cumsum(counts, dim=0)

        # Local port indices (0..deg(u)-1) in CSR order
        port = torch.empty_like(src)
        # For reverse lookup (v->u) find index: we will fill rev later
        # We also need a local multiplicity rank per (u,v) to pair with reverse (v,u).
        # Build per-node neighbor buckets to compute local ranks.
        start = node_ptr[:-1]
        end = node_ptr[1:]
        # local neighbor occurrence rank
        # We compute rank by scanning each node's outgoing slice.
        for u in range(n):
            s, e = int(start[u]), int(end[u])
            if s == e:
                continue
            dst_slice = dst[s:e]
            # Count ranks per identical dst within the slice
            # We can do this with a small hashmap (dst_id -> running count)
            seen: dict[int, int] = {}
            for i in range(s, e):
                v = int(dst[i])
                rank = seen.get(v, 0)
                port[i] = rank  # temporary: rank among edges to same v
                seen[v] = rank + 1

        # Now we need to convert the temporary "rank-among-(u->v)" to a true local port index
        # in [0, deg(u)-1]. We choose a canonical interleave by sorting by (dst, rank) so ports
        # are stable and match (v->u) pairing by rank.
        final_src = src.clone()
        final_dst = dst.clone()
        # Build a permutation within each node slice by (dst, rank)
        # We can reconstruct rank again but also reuse above by viewing `port` as rank_uv
        rank_uv = port.clone()
        perm = torch.arange(src.numel(), device=device)
        for u in range(n):
            s, e = int(start[u]), int(end[u])
            if s == e:
                continue
            sort_keys = torch.stack([final_dst[s:e], rank_uv[s:e]], dim=0)  # (2, len)
            # Sort lexicographically by dst then rank_uv
            # Implement lexsort by combining keys: dst*(max_rank+1)+rank
            # Compute max rank per u (<= multiplicity of its most frequent neighbor)
            max_rank = int(rank_uv[s:e].max().item()) if (e - s) > 0 else 0
            key = sort_keys[0] * (max_rank + 1) + sort_keys[1]
            rel = torch.argsort(key, stable=True)
            # Apply within-slice permutation
            final_dst[s:e] = final_dst[s:e][rel]
            final_src[s:e] = final_src[s:e][rel]
            rank_uv[s:e] = rank_uv[s:e][rel]
            perm[s:e] = perm[s:e][rel]

        # Recompute true local port = 0..deg(u)-1 as plain position within CSR slice
        port = torch.empty_like(final_src)
        for u in range(n):
            s, e = int(start[u]), int(end[u])
            if s == e:
                continue
            port[s:e] = torch.arange(e - s, device=device, dtype=torch.long)

        src = final_src
        dst = final_dst

        # Degrees
        deg = counts

        # Build reverse-arc indices rev[i], using rank pairing:
        # The arc u->v at rank r (within the ordered-by-(dst,rank) slice) must map to arc v->u at *the same rank r*
        # within v's slice among neighbors equal to u. To compute this efficiently:
        #  - Build CSR again (we have it) and, for each node v, a dictionary from neighbor u to the start index
        #    and ranks encountered.
        #  - Then rev[idx(u->v, r)] = idx(v->u, r) found by scanning v's slice and counting occurrences of u.
        A = src.numel()
        rev = torch.empty(A, dtype=torch.long, device=device)

        # Precompute for each node v: mapping neighbor->list of positions in that slice
        neighbor_positions: dict[int, dict[int, list]] = {}
        for v in range(n):
            s, e = int(start[v]), int(end[v])
            if s == e:
                continue
            dsv = dst[s:e]
            bucket: dict[int, list] = {}
            for pos, vv in enumerate(dsv.tolist()):
                bucket.setdefault(vv, []).append(s + pos)
            neighbor_positions[v] = bucket

        # For each u slice, align ports by neighbor v
        for u in range(n):
            s, e = int(start[u]), int(end[u])
            if s == e:
                continue
            # For this CSR slice, arcs are sorted by (dst, local-rank). So arcs to same v appear contiguously
            # with increasing rank. We map rank r at (u->v) to the r-th position among arcs where src=v, dst=u.
            v_to_pos = neighbor_positions  # alias
            # Prepare counters for each neighbor to pop ranks in order
            counters: dict[int, int] = {}
            for i in range(s, e):
                v = int(dst[i])
                r = counters.get(v, 0)
                # Positions in v's slice where dst==u are neighbor_positions[v].get(u, [])
                pos_list = v_to_pos.get(v, {}).get(u, [])
                if r >= len(pos_list):
                    raise RuntimeError(
                        f"Inconsistent multigraph pairing between {u} and {v} (rank {r} not found)."
                    )
                j = pos_list[r]
                rev[i] = j
                counters[v] = r + 1

        # Validate invariants
        _validate_portmap_invariants(n, src, dst, node_ptr, port, rev, deg)

        return PortMap(
            num_nodes=n,
            num_arcs=A,
            src=src,
            dst=dst,
            node_ptr=node_ptr,
            port=port,
            rev=rev,
            deg=deg,
        )

    # ------------------------------- Methods -------------------------------- #

    def to_shift_permutation(self) -> torch.Tensor:
        """
        Return a length-A tensor `perm` such that applying the flip-flop shift S is:
            psi_next = psi[perm]
        This yields a (deterministic) permutation (i.e., unitary) over arcs.
        """
        return self.rev.clone()

    def to_shift_sparse(self) -> torch.Tensor:
        """
        Return a sparse COO permutation matrix S of shape (A, A) with ones at (rev[i], i).
        """
        A = self.num_arcs
        if A == 0:
            idx = torch.empty((2, 0), dtype=torch.long, device=self.src.device)
            val = torch.empty((0,), dtype=torch.complex64, device=self.src.device)
            return torch.sparse_coo_tensor(idx, val, (0, 0)).coalesce()
        rows = self.rev
        cols = torch.arange(A, device=self.src.device, dtype=torch.long)
        idx = torch.stack([rows, cols], dim=0)
        val = torch.ones(A, dtype=torch.complex64, device=self.src.device)
        return torch.sparse_coo_tensor(idx, val, (A, A)).coalesce()

    def arc_slice(self, u: int) -> slice:
        """Return Python slice for arcs originating at node u."""
        s = int(self.node_ptr[u])
        e = int(self.node_ptr[u + 1])
        return slice(s, e)

    def arcs_of(self, u: int) -> torch.Tensor:
        """Return view of arc indices originating at node u."""
        sl = self.arc_slice(u)
        return torch.arange(sl.start, sl.stop, device=self.src.device)

    def neighbors(self, u: int) -> torch.Tensor:
        """Return the ordered list of neighbors (with multiplicities) for node u."""
        sl = self.arc_slice(u)
        return self.dst[sl]

    # ------------------------------ Serialization --------------------------- #

    def as_dict(self) -> dict:
        """Lightweight dict for logging/serialization (tensor shapes & meta)."""
        return {
            "num_nodes": self.num_nodes,
            "num_arcs": self.num_arcs,
            "deg_sum": int(self.deg.sum().item()) if self.deg.numel() else 0,
            "device": str(self.src.device),
            "dtype": str(self.src.dtype),
        }


# ---------------------------- Invariant checking --------------------------- #


def _validate_portmap_invariants(
    n: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    node_ptr: torch.Tensor,
    port: torch.Tensor,
    rev: torch.Tensor,
    deg: torch.Tensor,
) -> None:
    device = src.device
    A = src.numel()

    if node_ptr.numel() != n + 1:
        raise AssertionError("node_ptr must have length N+1")
    if src.shape != dst.shape or src.shape != port.shape or src.shape != rev.shape:
        raise AssertionError("src, dst, port, rev must all have shape (A,)")
    if deg.numel() != n:
        raise AssertionError("deg must have shape (N,)")

    # CSR grouping correct
    counts = torch.bincount(src, minlength=n)
    if not torch.equal(counts, deg):
        raise AssertionError("deg must equal out-degree counts")
    if not torch.equal(node_ptr[0], torch.tensor(0, device=device)):
        raise AssertionError("node_ptr[0] must be 0")
    if not torch.equal(node_ptr[1:], torch.cumsum(deg, dim=0)):
        raise AssertionError("node_ptr must be cumulative sum of degrees")

    # Ports in range
    for u in range(n):
        s, e = int(node_ptr[u]), int(node_ptr[u + 1])
        if s == e:
            continue
        p = port[s:e]
        if p.min() < 0 or p.max() != (e - s - 1):
            raise AssertionError(f"port indices out of range for node {u}")

    # Reverse mapping is an involution: rev[rev[i]] == i
    if A > 0:
        if not torch.equal(rev[rev], torch.arange(A, device=device)):
            raise AssertionError("rev must be an involution (rev[rev[i]] == i)")

    # src/dst consistency with reverse
    if A > 0:
        if not torch.equal(src[rev], dst):
            raise AssertionError("src[rev[i]] must equal dst[i]")
        if not torch.equal(dst[rev], src):
            raise AssertionError("dst[rev[i]] must equal src[i]")

    # No out-of-range indices
    if src.min() < 0 or src.max() >= n or dst.min() < 0 or dst.max() >= n:
        raise AssertionError("src/dst out of node range")

    # Determinism: arcs per node are ordered by (dst then local rank)
    # We can spot-check monotonicity within each CSR block.
    for u in range(n):
        s, e = int(node_ptr[u]), int(node_ptr[u + 1])
        if e - s <= 1:
            continue
        d_us = dst[s:e]
        # Non-decreasing by dst
        if not torch.all(d_us[:-1] <= d_us[1:]):
            raise AssertionError("Within CSR slice, dst must be non-decreasing")

    # Everything passed


# ------------------------------- Public API -------------------------------- #


def build_portmap(
    edge_index: EdgeIndexLike,
    num_nodes: int | None = None,
    *,
    allow_self_loops: bool = False,
    coalesce: bool = False,
    device: torch.device | None = None,
) -> PortMap:
    """
    High-level constructor for a flip-flop PortMap.

    See `PortMap.from_undirected` for semantics. This function is the stable
    entrypoint used by the simulator and tests.
    """
    return PortMap.from_undirected(
        edge_index=edge_index,
        num_nodes=num_nodes,
        allow_self_loops=allow_self_loops,
        coalesce=coalesce,
        device=device,
    )

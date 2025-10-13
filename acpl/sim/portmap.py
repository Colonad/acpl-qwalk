# acpl/sim/portmap.py
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np

try:  # optional torch support (inputs may be torch tensors)
    import torch
except Exception:  # pragma: no cover - torch may not be installed in some environments
    torch = None  # type: ignore[assignment]


ArrayLike = Union[np.ndarray, "torch.Tensor"]


@dataclass(frozen=True)
class PortMap:
    """
    Flip-flop port map for a (simple, undirected) graph in *oriented-arc* form.

    Attributes
    ----------
    num_nodes : int
        Number of vertices |V|.
    num_edges : int
        Number of *undirected* edges |E|.
    num_arcs : int
        Number of oriented arcs = 2*|E|.
    tail : np.ndarray[int64]   (num_arcs,)
        For arc a, tail[a] = u (source vertex of oriented edge u->v).
    head : np.ndarray[int64]   (num_arcs,)
        For arc a, head[a] = v (destination vertex of oriented edge u->v).
    rev  : np.ndarray[int64]   (num_arcs,)
        Reverse-arc index: for a indexing u->v, rev[a] indexes v->u.
        This gives the *flip-flop* mapping used by the shift S.
    node_ptr : np.ndarray[int64]  (num_nodes + 1,)
        CSR-style pointer into `node_arcs`: arcs incident *leaving* vertex v are
        node_arcs[node_ptr[v] : node_ptr[v+1]] in stable (tail, head) order.
    node_arcs : np.ndarray[int64] (num_arcs,)
        Concatenation of per-vertex outgoing arc indices.
        For degree d_v, there are exactly d_v arcs for vertex v.
    """

    num_nodes: int
    num_edges: int
    num_arcs: int
    tail: np.ndarray
    head: np.ndarray
    rev: np.ndarray
    node_ptr: np.ndarray
    node_arcs: np.ndarray

    # Convenience views -----------------------------------------------------
    def arcs_of(self, v: int) -> np.ndarray:
        """Return a view of the outgoing arc indices for vertex v."""
        start, end = self.node_ptr[v], self.node_ptr[v + 1]
        return self.node_arcs[start:end]

    def degree(self, v: int) -> int:
        """Return degree of vertex v."""
        return int(self.node_ptr[v + 1] - self.node_ptr[v])


def _to_numpy_edges(
    edge_index: ArrayLike | Sequence[tuple[int, int]] | Sequence[Sequence[int]],
) -> np.ndarray:
    """
    Normalize various edge_index formats to a canonical numpy int64 array of shape (2, E).
    Accepts:
      - numpy array of shape (2, E) or (E, 2)
      - torch tensor of shape (2, E) or (E, 2)
      - list/tuple of pairs [(u, v), ...]
    """
    if isinstance(edge_index, np.ndarray):
        ei = edge_index
    elif torch is not None and isinstance(edge_index, torch.Tensor):
        ei = edge_index.detach().cpu().numpy()
    elif isinstance(edge_index, Iterable):
        # Try to interpret as list of pairs
        arr = np.asarray(list(edge_index), dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "edge_index as a sequence must be a list of (u, v) pairs with shape (E, 2)."
            )
        ei = arr.T  # to (2, E)
    else:
        raise TypeError("Unsupported edge_index type.")

    if ei.ndim != 2 or (ei.shape[0] != 2 and ei.shape[1] != 2):
        raise ValueError("edge_index must have shape (2, E) or (E, 2).")

    if ei.shape[0] == 2:
        out = ei.astype(np.int64, copy=False)
    else:  # (E, 2) -> transpose
        out = ei.T.astype(np.int64, copy=False)
    return out


def _coalesce_simple_undirected(ei: np.ndarray, num_nodes: int | None) -> tuple[np.ndarray, int]:
    """
    Make edges simple, undirected and sorted.
    - Removes self-loops and duplicate undirected edges.
    - Canonicalizes each edge as (min(u,v), max(u,v)).
    - Sorts edges lexicographically.

    Returns
    -------
    edges_uv : np.ndarray shape (2, E)
        Canonical undirected edges.
    num_nodes : int
        Inferred if not given.
    """
    assert ei.shape[0] == 2
    u, v = ei[0], ei[1]
    if num_nodes is None:
        n = int(max(int(u.max(initial=-1)), int(v.max(initial=-1))) + 1)
    else:
        n = int(num_nodes)

    # remove self-loops
    mask = u != v
    u, v = u[mask], v[mask]

    # canonicalize (min, max)
    uu = np.minimum(u, v)
    vv = np.maximum(u, v)

    # sort by (uu, vv)
    order = np.lexsort((vv, uu))
    uu, vv = uu[order], vv[order]

    # unique
    if uu.size > 0:
        keep = np.ones_like(uu, dtype=bool)
        keep[1:] = (uu[1:] != uu[:-1]) | (vv[1:] != vv[:-1])
        uu, vv = uu[keep], vv[keep]

    edges = np.stack([uu, vv], axis=0).astype(np.int64, copy=False)
    return edges, n


def make_flipflop_portmap(
    edge_index: ArrayLike | Sequence[tuple[int, int]] | Sequence[Sequence[int]],
    num_nodes: int | None = None,
) -> PortMap:
    """
    Build the flip-flop PortMap from an undirected simple graph.

    Parameters
    ----------
    edge_index : array-like
        Undirected edges. Shape (2, E) or (E, 2) or list of (u, v) pairs.
        If directed edges are supplied, they will be coalesced into undirected ones.
    num_nodes : int, optional
        Number of nodes. If None, inferred from max index + 1.

    Returns
    -------
    PortMap
        Data required to construct the shift permutation S and to index vertex-local arcs.

    Notes
    -----
    - The flip-flop shift maps each oriented arc (u->v) to its reverse (v->u).
    - We ensure deterministic arc ordering:
        1) coalesce edges to canonical (min, max) and sort lexicographically,
        2) emit arcs in the order: (u->v) then (v->u) for each undirected edge,
        3) build CSR node→arc index based on `tail` (stable).
    """
    ei = _to_numpy_edges(edge_index)
    undirected, n = _coalesce_simple_undirected(ei, num_nodes)
    uu, vv = undirected[0], undirected[1]
    num_edges_undirected = uu.size
    num_arcs_total = 2 * num_edges_undirected  # number of oriented arcs

    # Construct oriented arcs arrays: for each undirected edge i:
    # (uu[i] -> vv[i]) and (vv[i] -> uu[i])
    tail = np.empty(num_arcs_total, dtype=np.int64)
    head = np.empty(num_arcs_total, dtype=np.int64)
    # interleave (u->v) and (v->u)
    tail[0::2], head[0::2] = uu, vv
    tail[1::2], head[1::2] = vv, uu

    # Reverse arc map: pair (2*i) <-> (2*i+1)
    rev = np.empty(num_arcs_total, dtype=np.int64)
    idx = np.arange(num_arcs_total, dtype=np.int64)
    rev[0::2] = idx[1::2]
    rev[1::2] = idx[0::2]

    # Build CSR-style node->arcs index (group by tail, stable order).
    # Count degrees
    deg = np.bincount(tail, minlength=n).astype(np.int64, copy=False)
    node_ptr = np.empty(n + 1, dtype=np.int64)
    node_ptr[0] = 0
    np.cumsum(deg, out=node_ptr[1:])

    # Stable scatter of arc indices by their tail into node_arcs
    node_arcs = np.empty(num_arcs_total, dtype=np.int64)
    cursor = node_ptr[:-1].copy()
    for a in range(num_arcs_total):
        v = tail[a]
        pos = cursor[v]
        node_arcs[pos] = a
        cursor[v] += 1

    return PortMap(
        num_nodes=n,
        num_edges=num_edges_undirected,
        num_arcs=num_arcs_total,
        tail=tail,
        head=head,
        rev=rev,
        node_ptr=node_ptr,
        node_arcs=node_arcs,
    )


# ---------- Convenience helpers (optional public API) ----------


def degrees_from_portmap(pm: PortMap) -> np.ndarray:
    """Return degree array d_v from a PortMap."""
    return np.diff(pm.node_ptr)


def to_edge_index(pm: PortMap) -> np.ndarray:
    """
    Recover a canonical undirected edge_index (2, E) from the PortMap.
    Useful for debugging or round-trips.
    """
    # grab arcs with even index -> (u->v) corresponds to undirected edge (u, v)
    u = pm.tail[0::2]
    v = pm.head[0::2]
    return np.stack([u, v], axis=0)

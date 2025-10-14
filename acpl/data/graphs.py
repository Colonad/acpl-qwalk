# acpl/data/graphs.py
from __future__ import annotations

import numpy as np


def line_graph(
    n: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a path (line) graph on vertices {0, 1, ..., n-1}.

    Returns
    -------
    edge_index : np.ndarray, shape (2, E), dtype=int64
        Canonical undirected edges (u < v), lexicographically sorted.
        For a path, E = n - 1 and edges are (0,1), (1,2), ..., (n-2,n-1).
    degrees : np.ndarray, shape (n,), dtype=int64
        Degree of each vertex. For a path: [1, 2, 2, ..., 2, 1] (n>=2).
        For n == 1, degrees = [0].
    coords : np.ndarray, shape (n, 1), dtype=float32
        1-D coordinate along the line in [0, 1], i.e., i/(n-1) for node i.
        For n == 1, coords = [[0.0]].
    arc_slices : np.ndarray, shape (n, 2), dtype=int64
        For each vertex v, a half-open slice [start, end) into the **vertex-lumped**
        arc list (i.e., arcs grouped by their tail vertex) produced by the same
        deterministic convention used in `acpl.sim.portmap`:
          1) undirected edges coalesced and lexicographically sorted (u < v),
          2) for each undirected edge (u, v), emit arcs in order (u→v) then (v→u),
          3) concatenate, per vertex v, all arcs whose tail is v in encounter order.
        This makes arc_slices directly comparable with `node_ptr` if/when you build a
        `PortMap` later.

    Notes
    -----
    - `seed` is accepted for a uniform API but unused for the deterministic path.
    - This is intentionally minimal for Phase A; more graph families arrive in B2.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")

    # Edges: canonical and already sorted for a path.
    if n == 1:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        u = np.arange(n - 1, dtype=np.int64)
        v = u + 1
        edge_index = np.stack([u, v], axis=0)  # (2, E)

    # Degrees
    degrees = np.zeros(n, dtype=np.int64)
    if n >= 2:
        degrees[0] = 1
        degrees[-1] = 1
    if n >= 3:
        degrees[1:-1] = 2

    # 1-D coordinates in [0, 1]
    if n == 1:
        coords = np.array([[0.0]], dtype=np.float32)
    else:
        coords = (np.arange(n, dtype=np.float32) / float(n - 1)).reshape(n, 1)

    # Arc slices consistent with our PortMap convention.
    # For each undirected edge i with (u[i], v[i]), arcs are indexed:
    #   a_even = 2*i corresponds to u->v
    #   a_odd  = 2*i+1 corresponds to v->u
    # We gather per-vertex outgoing arcs in encounter order and compute prefix sums.
    if edge_index.shape[1] == 0:
        # No arcs when n == 1
        arc_slices = np.zeros((n, 2), dtype=np.int64)
    else:
        e = edge_index.shape[1]
        arc_lists: list[list[int]] = [[] for _ in range(n)]
        uu, vv = edge_index[0], edge_index[1]
        for i in range(e):
            a_even = 2 * i
            a_odd = a_even + 1
            u_i = int(uu[i])
            v_i = int(vv[i])
            arc_lists[u_i].append(a_even)  # u -> v
            arc_lists[v_i].append(a_odd)  # v -> u

        # Concatenate in vertex order to get the vertex-lumped arc order.
        # Then compute slices [start, end) via prefix sums of lengths.
        lengths = np.array([len(lst) for lst in arc_lists], dtype=np.int64)
        node_ptr = np.empty(n + 1, dtype=np.int64)
        node_ptr[0] = 0
        np.cumsum(lengths, out=node_ptr[1:])
        # Return only the slices (start, end), matching each vertex.
        arc_slices = np.stack([node_ptr[:-1], node_ptr[1:]], axis=1)

    return edge_index, degrees, coords, arc_slices

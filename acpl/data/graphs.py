# acpl/data/graphs.py
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _coalesce_undirected(n: int, edges: np.ndarray) -> np.ndarray:
    """
    Normalize to undirected canonical form:
      - ensure shape (2, E), dtype=int64
      - swap so each edge has u < v
      - drop self-loops and duplicates
      - sort lexicographically by (u, v)
    """
    if edges.size == 0:
        return np.empty((2, 0), dtype=np.int64)

    ei = np.asarray(edges, dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError("edges must have shape (2, E)")

    u, v = ei[0].copy(), ei[1].copy()
    # enforce u < v
    mask = u > v
    tmp = u[mask].copy()
    u[mask] = v[mask]
    v[mask] = tmp

    # remove self loops
    keep = u != v
    u, v = u[keep], v[keep]

    if u.size == 0:
        return np.empty((2, 0), dtype=np.int64)

    # lexicographic unique
    keys = u.astype(np.int64) * np.int64(n) + v.astype(np.int64)
    order = np.argsort(keys, kind="mergesort")
    u, v = u[order], v[order]
    keys = keys[order]
    uniq = np.ones_like(keys, dtype=bool)
    uniq[1:] = keys[1:] != keys[:-1]

    return np.stack([u[uniq], v[uniq]], axis=0)


def _degrees(n: int, edge_index: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=np.int64)
    if edge_index.size:
        u, v = edge_index
        np.add.at(deg, u, 1)
        np.add.at(deg, v, 1)
    return deg


def _arc_slices_from_edges(n: int, edge_index: np.ndarray) -> np.ndarray:
    """
    Build (start,end) slices for **vertex-lumped** arc order using the same
    arc indexing convention as PortMap:
      for edge i = (u[i], v[i]) in lexicographic order,
        arcs 2*i = u->v, 2*i+1 = v->u;
      then concatenate per-vertex outgoing arcs in vertex order.
    """
    if edge_index.shape[1] == 0:
        return np.zeros((n, 2), dtype=np.int64)

    e = edge_index.shape[1]
    lists: list[list[int]] = [[] for _ in range(n)]
    uu, vv = edge_index[0], edge_index[1]
    for i in range(e):
        a_even = 2 * i
        a_odd = a_even + 1
        u = int(uu[i])
        v = int(vv[i])
        lists[u].append(a_even)
        lists[v].append(a_odd)

    lengths = np.array([len(lst) for lst in lists], dtype=np.int64)
    node_ptr = np.empty(n + 1, dtype=np.int64)
    node_ptr[0] = 0
    np.cumsum(lengths, out=node_ptr[1:])
    return np.stack([node_ptr[:-1], node_ptr[1:]], axis=1)


# ---------------------------------------------------------------------
# Phase A: line graph (unchanged)
# ---------------------------------------------------------------------


def line_graph(
    n: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Path graph on {0,...,n-1}.
    coords: (n,1) in [0,1]
    """
    if n <= 0:
        raise ValueError("n must be >= 1")

    if n == 1:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        u = np.arange(n - 1, dtype=np.int64)
        v = u + 1
        edge_index = np.stack([u, v], axis=0)

    degrees = _degrees(n, edge_index)

    if n == 1:
        coords = np.array([[0.0]], dtype=np.float32)
    else:
        coords = (np.arange(n, dtype=np.float32) / float(n - 1)).reshape(n, 1)

    arc_slices = _arc_slices_from_edges(n, edge_index)
    return edge_index, degrees, coords, arc_slices


# ---------------------------------------------------------------------
# B2: grid LxL (4-neighbor, open boundary)
# ---------------------------------------------------------------------


def grid_graph(
    side: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    side x side grid (open boundary), 4-neighbor connectivity.
    coords: (N,2) normalized to [0,1]^2 with (i/(side-1), j/(side-1)).
    """
    if side <= 0:
        raise ValueError("side must be >= 1")
    n = side * side
    edges_u: list[int] = []
    edges_v: list[int] = []

    def nid(i: int, j: int) -> int:
        return i * side + j

    for i in range(side):
        for j in range(side):
            u = nid(i, j)
            if j + 1 < side:
                edges_u.append(u)
                edges_v.append(nid(i, j + 1))
            if i + 1 < side:
                edges_u.append(u)
                edges_v.append(nid(i + 1, j))

    if edges_u:
        ei = np.stack([np.array(edges_u), np.array(edges_v)], axis=0).astype(np.int64)
    else:
        ei = np.empty((2, 0), dtype=np.int64)

    edge_index = _coalesce_undirected(n, ei)
    degrees = _degrees(n, edge_index)

    if side == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float32)
    else:
        xs = np.linspace(0.0, 1.0, side, dtype=np.float32)
        xv, yv = np.meshgrid(xs, xs, indexing="ij")
        coords = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.float32)

    arc_slices = _arc_slices_from_edges(n, edge_index)
    return edge_index, degrees, coords, arc_slices


# ---------------------------------------------------------------------
# B2: hypercube (dimension = dim, nodes = 2**dim)
# ---------------------------------------------------------------------


def hypercube_graph(
    dim: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    d-dimensional hypercube (Q_dim). N = 2^dim.
    coords: (N, dim) binary {0,1} as float32.
    """
    if dim < 0:
        raise ValueError("dim must be >= 0")
    n = 1 << dim
    if n == 0:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.zeros(0, np.int64),
            np.zeros((0, dim), np.float32),
            np.zeros((0, 2), np.int64),
        )

    edges_u: list[int] = []
    edges_v: list[int] = []
    for u in range(n):
        for b in range(dim):
            v = u ^ (1 << b)
            if u < v:
                edges_u.append(u)
                edges_v.append(v)

    ei = (
        np.stack([np.array(edges_u), np.array(edges_v)], axis=0).astype(np.int64)
        if edges_u
        else np.empty((2, 0), dtype=np.int64)
    )
    edge_index = _coalesce_undirected(n, ei)
    degrees = _degrees(n, edge_index)

    # coords: binary representation
    coords = np.unpackbits(np.arange(n, dtype=np.uint32).reshape(-1, 1).view(np.uint8), axis=1)
    # take last 'dim' bits
    coords = coords[:, -dim:] if dim > 0 else np.zeros((n, 0), dtype=np.uint8)
    coords = coords.astype(np.float32)

    arc_slices = _arc_slices_from_edges(n, edge_index)
    return edge_index, degrees, coords, arc_slices


# ---------------------------------------------------------------------
# B2: random d-regular simple graph (pairing method with retries)
# ---------------------------------------------------------------------


def d_regular_graph(
    n: int,
    d: int,
    seed: int | None = None,
    max_tries: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Random simple d-regular graph (no self-loops or multi-edges) using stub pairing.
    Requires n*d even and 0 <= d < n.
    coords: (N,1) = linspace in [0,1] for convenience.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if d < 0 or d >= n:
        raise ValueError("d must satisfy 0 <= d < n")
    if (n * d) % 2 != 0:
        raise ValueError("n*d must be even for a d-regular graph")
    rng = np.random.default_rng(seed)

    stubs = np.repeat(np.arange(n, dtype=np.int64), d)
    for _ in range(max_tries):
        rng.shuffle(stubs)
        pairs = stubs.reshape(-1, 2)
        u = pairs[:, 0]
        v = pairs[:, 1]
        # remove self-loops
        ok = u != v
        u, v = u[ok], v[ok]
        # coalesce and test degrees
        ei = _coalesce_undirected(n, np.stack([u, v], axis=0))
        if ei.shape[1] == (n * d) // 2:
            deg = _degrees(n, ei)
            if np.all(deg == d):
                edge_index = ei
                break
    else:
        raise RuntimeError("failed to construct a simple d-regular graph")

    degrees = _degrees(n, edge_index)
    coords = (np.linspace(0.0, 1.0, n, dtype=np.float32)).reshape(n, 1)
    arc_slices = _arc_slices_from_edges(n, edge_index)
    return edge_index, degrees, coords, arc_slices


# ---------------------------------------------------------------------
# B2: Erdős–Rényi G(n, p)
# ---------------------------------------------------------------------


def er_graph(
    n: int,
    p: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Erdős–Rényi G(n, p) on the simple undirected graph (no self-loops, no multiedges).
    coords: (N,1) = linspace in [0,1].
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")

    rng = np.random.default_rng(seed)
    triu = np.triu(rng.random((n, n)) < p, k=1)
    uu, vv = np.nonzero(triu)
    ei = np.stack([uu.astype(np.int64), vv.astype(np.int64)], axis=0)
    edge_index = _coalesce_undirected(n, ei)

    degrees = _degrees(n, edge_index)
    coords = (np.linspace(0.0, 1.0, n, dtype=np.float32)).reshape(n, 1)
    arc_slices = _arc_slices_from_edges(n, edge_index)
    return edge_index, degrees, coords, arc_slices


# ---------------------------------------------------------------------
# B2: Watts–Strogatz small-world graph
# ---------------------------------------------------------------------


def watts_strogatz_graph(
    n: int,
    k: int,
    beta: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Watts–Strogatz model:
      - start from a ring lattice where each node connects to k/2 neighbors on
        each side (k must be even, 0 < k < n),
      - rewire each (i, i+t) edge with probability beta to (i, j) chosen
        uniformly from nodes excluding i and existing neighbors (no multiedges).
    coords: (N,1) = linspace in [0,1].
    """
    if n <= 2:
        raise ValueError("n must be > 2")
    if k <= 0 or k >= n or k % 2 != 0:
        raise ValueError("k must be even and satisfy 0 < k < n")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")

    rng = np.random.default_rng(seed)

    # ring lattice
    edges = set()
    half = k // 2
    for i in range(n):
        for t in range(1, half + 1):
            j = (i + t) % n
            u, v = (i, j) if i < j else (j, i)
            edges.add((u, v))

    # adjacency for fast “neighbor” checks
    neigh = [set() for _ in range(n)]
    for u, v in edges:
        neigh[u].add(v)
        neigh[v].add(u)

    # rewire each forward edge (i, i+t), i < j representation
    for i in range(n):
        for t in range(1, half + 1):
            j = (i + t) % n
            u, v = (i, j) if i < j else (j, i)
            if (u, v) not in edges:
                continue
            if rng.random() >= beta:
                continue  # keep as-is

            # remove (u, v)
            edges.remove((u, v))
            neigh[u].discard(v)
            neigh[v].discard(u)

            # pick new w for u uniformly, avoiding u and its current neighbors
            candidates = list(set(range(n)) - {u} - neigh[u])
            if not candidates:
                # rollback (very unlikely for sane k)
                edges.add((u, v))
                neigh[u].add(v)
                neigh[v].add(u)
                continue
            w = int(rng.choice(candidates))
            uu, vv = (u, w) if u < w else (w, u)
            # avoid duplicates (rare due to candidate filter)
            if (uu, vv) in edges:
                # rollback original edge
                edges.add((u, v))
                neigh[u].add(v)
                neigh[v].add(u)
                continue
            edges.add((uu, vv))
            neigh[uu].add(vv)
            neigh[vv].add(uu)

    if edges:
        uu = np.fromiter((u for (u, _) in edges), dtype=np.int64)
        vv = np.fromiter((v for (_, v) in edges), dtype=np.int64)
        ei = np.stack([uu, vv], axis=0)
    else:
        ei = np.empty((2, 0), dtype=np.int64)

    edge_index = _coalesce_undirected(n, ei)
    degrees = _degrees(n, edge_index)
    coords = (np.linspace(0.0, 1.0, n, dtype=np.float32)).reshape(n, 1)
    arc_slices = _arc_slices_from_edges(n, edge_index)
    return edge_index, degrees, coords, arc_slices

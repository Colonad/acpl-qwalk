# acpl/data/graphs.py
from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections import deque
from dataclasses import dataclass
import math


import numpy as np
import torch

__all__ = [
    # Core container & utilities
    "GraphData",
    "build_arc_index",
    "group_arcs_by_degree",
    "build_degree_buckets",
    # Canonical generators
    "line_graph",
    "cycle_graph",
    "grid_graph",
    "hypercube_graph",
    "erdos_renyi_graph",
    "watts_strogatz_graph",
    "d_regular_random_graph",
    # Small-world on grids + extras used in our experiments
    "watts_strogatz_grid_graph",
    "watts_strogatz_grid_graph_degree_preserving",
    "random_geometric_graph",
    # Aliases (Hydra family names)
    "ws_graph",
    "ws_grid_graph",

]

# ======================================================================================
# Data container
# ======================================================================================


@dataclass
class GraphData:
    """
    Container for ACPL graph inputs.

    Attributes
    ----------
    edge_index : LongTensor, shape (2, A)
        Oriented arcs in COO (src,dst) with both directions for each undirected edge.
        Sorted by (src, dst) for deterministic CSR grouping.
    degrees : LongTensor, shape (N,)
        Undirected degree per node.
    coords : FloatTensor, shape (N, C)
        Family-dependent coordinates / encodings (e.g., normalized grid coords, bitstrings).
    arc_slices : LongTensor, shape (N+1,)
        CSR node pointers into the arc list: arcs for node u are
        edge_index[:, arc_slices[u] : arc_slices[u+1]).
    """

    edge_index: torch.Tensor
    degrees: torch.Tensor
    coords: torch.Tensor
    arc_slices: torch.Tensor

    # ---- convenience -----------------------------------------------------------------------------

    def to(self, device: str | torch.device) -> GraphData:
        return GraphData(
            edge_index=self.edge_index.to(device),
            degrees=self.degrees.to(device),
            coords=self.coords.to(device),
            arc_slices=self.arc_slices.to(device),
        )

    def num_nodes(self) -> int:
        return int(self.degrees.numel())

    def num_arcs(self) -> int:
        return int(self.edge_index.size(1))

    def degree_values(self) -> torch.Tensor:
        """Sorted unique degree values present in the graph (LongTensor)."""
        return torch.unique(self.degrees, sorted=True)

    def degree_csr(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """
        Return per-degree CSR views for batching mixed-degree coins.

        Returns
        -------
        mapping : dict[int, (node_ids, arc_ptr)]
            For each degree d in the graph:
              - node_ids : LongTensor of node indices with degree d (sorted ascending)
              - arc_ptr  : LongTensor of shape (len(node_ids)+1,)
                           cumulative arc counts for those nodes, i.e.,
                           the arcs of node_ids[i] correspond to the slice
                           [ arc_ptr[i] : arc_ptr[i+1] ) inside the *concatenation*
                           of all degree-d node arc lists.
            This lets the simulator assemble block-diagonal coins and permute/gather
            the arc-state per degree in contiguous blocks.

        Notes
        -----
        The returned arc_ptr is relative to the degree-local concatenation order,
        not the global `edge_index` order. Use it together with `node_ids` to
        gather/scatter degree-specific arc chunks.
        """
        return group_arcs_by_degree(self.degrees, self.arc_slices)


# ======================================================================================
# Internal helpers (dtype, RNG, coordinates)
# ======================================================================================


def _ensure_torch_long(x: np.ndarray | list[int]) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.long)


def _ensure_torch_float(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def _coords_line(N: int) -> torch.Tensor:
    if N <= 1:
        return torch.zeros((N, 1), dtype=torch.float32)
    x = np.linspace(0.0, 1.0, N, dtype=np.float32)[:, None]
    return _ensure_torch_float(x)


def _coords_cycle(N: int) -> torch.Tensor:
    if N == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    theta = (2.0 * math.pi) * (np.arange(N, dtype=np.float32) / float(N))
    xy = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    return _ensure_torch_float(xy)


def _coords_grid(Lx: int, Ly: int) -> torch.Tensor:
    xs = np.arange(Lx, dtype=np.float32) / max(Lx - 1, 1)
    ys = np.arange(Ly, dtype=np.float32) / max(Ly - 1, 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return _ensure_torch_float(coords)


def _coords_hypercube(nbits: int) -> torch.Tensor:
    N = 1 << nbits
    if N == 0:
        return torch.zeros((0, max(1, nbits)), dtype=torch.float32)
    coords = np.zeros((N, nbits), dtype=np.float32)
    for i in range(N):
        for b in range(nbits):
            coords[i, b] = (i >> b) & 1
    return _ensure_torch_float(coords)


# ======================================================================================
# Edge assembly (undirected → oriented arcs) + degree-group CSR
# ======================================================================================


def _coalesce_undirected_edges(
    edges: Iterable[tuple[int, int]], num_nodes: int
) -> list[tuple[int, int]]:
    """
    Deduplicate undirected edges, drop self-loops, keep canonical (min,max) ordering.
    """
    seen = set()
    out: list[tuple[int, int]] = []
    for u, v in edges:
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"Edge ({u},{v}) out of bounds for N={num_nodes}.")
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in seen:
            seen.add((a, b))
            out.append((a, b))
    out.sort()
    return out


def _norm_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _is_connected(num_nodes: int, edges: Iterable[tuple[int, int]]) -> bool:
    """
    Connectivity check for an undirected graph.
    Returns True iff all nodes are reachable from node 0 (for N>0).
    """
    if num_nodes <= 1:
        return True
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    q: deque[int] = deque([0])
    seen = [False] * num_nodes
    seen[0] = True
    while q:
        u = q.popleft()
        for w in adj[u]:
            if not seen[w]:
                seen[w] = True
                q.append(w)
    return all(seen)



def build_arc_index(
    undirected_edges: Sequence[tuple[int, int]],
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    From a list of unique undirected edges (u<v), build the oriented arc list and CSR slices.

    Returns
    -------
    edge_index : (2, A) LongTensor
        Stacked [src; dst], sorted by (src, dst) for determinism.
    degrees : (N,) LongTensor
        Degree per node (undirected).
    arc_slices : (N+1,) LongTensor
        CSR pointers over arcs grouped by source node.

    Notes
    -----
    The orientation duplicates each undirected edge in both directions; A = 2|E|.
    """
    # Make oriented arcs & degrees
    src_list: list[int] = []
    dst_list: list[int] = []
    deg = np.zeros(num_nodes, dtype=np.int64)

    for u, v in undirected_edges:
        deg[u] += 1
        deg[v] += 1
        src_list.append(u)
        dst_list.append(v)
        src_list.append(v)
        dst_list.append(u)

    # Sort arcs by (src, dst) for deterministic CSR
    order = np.lexsort((np.asarray(dst_list), np.asarray(src_list)))
    src = np.asarray(src_list, dtype=np.int64)[order]
    dst = np.asarray(dst_list, dtype=np.int64)[order]

    # Build CSR slices
    node_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    counts = np.bincount(src, minlength=num_nodes)
    node_ptr[1:] = np.cumsum(counts, dtype=np.int64)

    edge_index = torch.stack((_ensure_torch_long(src), _ensure_torch_long(dst)), dim=0)
    degrees = _ensure_torch_long(deg)
    arc_slices = _ensure_torch_long(node_ptr)
    return edge_index, degrees, arc_slices



def _is_connected(num_nodes: int, undirected_edges: Sequence[tuple[int, int]]) -> bool:
    """
    Check connectivity of an undirected simple graph given edges (u<v).
    Treats isolated nodes as disconnected unless num_nodes <= 1.
    """
    if num_nodes <= 1:
        return True
    if len(undirected_edges) == 0:
        return False

    adj = [[] for _ in range(num_nodes)]
    for u, v in undirected_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Start BFS/DFS from node 0 (or first node with any neighbor)
    start = 0
    if not adj[start]:
        for i in range(num_nodes):
            if adj[i]:
                start = i
                break
        else:
            return False

    seen = [False] * num_nodes
    stack = [start]
    seen[start] = True
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if not seen[w]:
                seen[w] = True
                stack.append(w)

    return all(seen)



def build_degree_buckets(degrees: torch.Tensor) -> dict[int, torch.Tensor]:
    """
    Group node indices by (undirected) degree.

    Parameters
    ----------
    degrees : LongTensor (N,)

    Returns
    -------
    buckets : dict[int, LongTensor]
        For each degree d present, buckets[d] is a sorted LongTensor of node indices with degree d.
    """
    if degrees.dtype != torch.long:
        degrees = degrees.to(torch.long)
    vals = torch.unique(degrees, sorted=True)
    buckets: dict[int, torch.Tensor] = {}
    for d in vals.tolist():
        idx = torch.nonzero(degrees == d, as_tuple=False).flatten()
        buckets[int(d)] = torch.sort(idx).values
    return buckets


def group_arcs_by_degree(
    degrees: torch.Tensor, arc_slices: torch.Tensor
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """
    Build per-degree CSR views over arcs.

    For each degree d, we:
      (i) collect nodes with deg = d in ascending order,
      (ii) compute counts = (arc_slices[u+1]-arc_slices[u]) for those nodes (should all equal d),
      (iii) produce a degree-local pointer array arc_ptr where arc_ptr[i+1] = arc_ptr[i] + counts[i].

    Parameters
    ----------
    degrees   : LongTensor (N,)
    arc_slices: LongTensor (N+1,)

    Returns
    -------
    mapping : dict[int, (node_ids, arc_ptr)]
        node_ids : LongTensor (M_d,) nodes with degree d (sorted)
        arc_ptr  : LongTensor (M_d+1,) cumulative arc counts for those nodes
                   in the degree-local concatenation order.

    Notes
    -----
    - This does NOT reorder the global `edge_index`; it provides the metadata needed
      to gather/scatter arc chunks per degree into contiguous buffers for fast, batched
      block-diagonal coin application in the simulator.
    - Robust to irregular graphs: if a node has 0 degree, we still include d=0
      with arc_ptr=[0] and node_ids possibly nonempty.
    """
    if degrees.numel() + 1 != arc_slices.numel():
        raise ValueError("arc_slices must have length N+1 for N=len(degrees).")

    buckets = build_degree_buckets(degrees)
    out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for d, nodes in buckets.items():
        if nodes.numel() == 0:
            out[d] = (nodes, torch.zeros(1, dtype=torch.long))
            continue
        counts = (arc_slices[nodes + 1] - arc_slices[nodes]).to(torch.long)
        # We do not assert counts == d for robustness (directed/self-loop variants may differ),
        # but in simple undirected graphs it should hold.
        ptr = torch.zeros(nodes.numel() + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(counts, dim=0)
        out[d] = (nodes, ptr)

    return out


# ======================================================================================
# Graph builders — canonical families used throughout ACPL
# ======================================================================================


def line_graph(
    N: int, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if N < 0:
        raise ValueError("N must be nonnegative.")
    edges = [] if N <= 1 else [(i, i + 1) for i in range(N - 1)]
    edges = _coalesce_undirected_edges(edges, N)
    edge_index, degrees, arc_slices = build_arc_index(edges, N)
    coords = _coords_line(N)
    return edge_index, degrees, coords, arc_slices


def cycle_graph(
    N: int, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if N < 0:
        raise ValueError("N must be nonnegative.")
    if N < 2:
        edges: list[tuple[int, int]] = []
    elif N == 2:
        edges = [(0, 1)]
    else:
        edges = [(i, (i + 1) % N) for i in range(N)]
    edges = _coalesce_undirected_edges(edges, N)
    edge_index, degrees, arc_slices = build_arc_index(edges, N)
    coords = _coords_cycle(N)
    return edge_index, degrees, coords, arc_slices


def grid_graph(
    Lx: int, Ly: int | None = None, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    2D grid with 4-neighborhood (no wrap); coords are normalized (x/Lx, y/Ly).
    """
    if Ly is None:
        Ly = Lx
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx, Ly must be positive.")
    N = Lx * Ly

    def vid(x: int, y: int) -> int:
        return y * Lx + x

    edges: list[tuple[int, int]] = []
    for y in range(Ly):
        for x in range(Lx):
            u = vid(x, y)
            if x + 1 < Lx:
                edges.append((u, vid(x + 1, y)))
            if y + 1 < Ly:
                edges.append((u, vid(x, y + 1)))
    edges = _coalesce_undirected_edges(edges, N)
    edge_index, degrees, arc_slices = build_arc_index(edges, N)
    coords = _coords_grid(Lx, Ly)
    return edge_index, degrees, coords, arc_slices


def hypercube_graph(
    nbits: int, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    n-dimensional hypercube Q_n (N=2^n). Coordinates = bitstrings in {0,1}^n.
    """
    if nbits < 0 or nbits > 20:
        # >20 would create huge graphs; keep a sane guard for training.
        raise ValueError("nbits must be in [0, 20].")
    N = 1 << nbits
    edges: list[tuple[int, int]] = []
    for i in range(N):
        for b in range(nbits):
            j = i ^ (1 << b)
            if i < j:
                edges.append((i, j))
    edges = _coalesce_undirected_edges(edges, N)
    edge_index, degrees, arc_slices = build_arc_index(edges, N)
    coords = _coords_hypercube(nbits)
    return edge_index, degrees, coords, arc_slices


def erdos_renyi_graph(
    N: int,
    p: float,
    seed: int | None = None,
    ensure_simple: bool = True,
    *,
    keep_connected: bool = False,
    max_tries: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    G(N,p) with optional duplicate/self-loop culling (ensure_simple=True).
    Optionally resample until connected (keep_connected=True).
    Coordinates: normalized line positions (stable default for irregulars).
    """
    if N < 0:
        raise ValueError("N must be nonnegative.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    if max_tries <= 0:
        raise ValueError("max_tries must be positive.")

    rng = _rng(seed)

    # Trivial sizes are always connected.
    if N <= 1:
        edges: list[tuple[int, int]] = []
        edge_index, degrees, arc_slices = build_arc_index(edges, N)
        coords = _coords_line(N)
        return edge_index, degrees, coords, arc_slices

    for attempt in range(max_tries):
        edges: list[tuple[int, int]] = []
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < p:
                    edges.append((u, v))

        if ensure_simple:
            edges = _coalesce_undirected_edges(edges, N)

        if not keep_connected or _is_connected(N, edges):
            edge_index, degrees, arc_slices = build_arc_index(edges, N)
            coords = _coords_line(N)
            return edge_index, degrees, coords, arc_slices

    raise RuntimeError(
        f"ER keep_connected failed after {max_tries} attempts (N={N}, p={p})."
    )


def watts_strogatz_graph(
    N: int,
    k: int,
    beta: float,
    seed: int | None = None,
    *,
    keep_connected: bool = False,
    max_factor: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    1D Watts–Strogatz (degree-preserving variant).

    Start from a k-regular ring lattice, then perform degree-preserving double-edge swaps.
    This guarantees every node keeps degree exactly k (matching ACPL coin-dimension assumptions).

    target_swaps = round(beta * |E|)
    Deterministic given the same seed.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if k < 0 or k >= N or (k % 2) != 0:
        raise ValueError("k must be even and satisfy 0 <= k < N.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0,1].")
    if max_factor <= 0:
        raise ValueError("max_factor must be positive.")

    rng = _rng(seed)

    # Build k-regular ring lattice
    edges_set: set[tuple[int, int]] = set()
    nbr: list[set[int]] = [set() for _ in range(N)]
    for u in range(N):
        for j in range(1, k // 2 + 1):
            v = (u + j) % N
            a, b = _norm_edge(u, v)
            if (a, b) not in edges_set:
                edges_set.add((a, b))
                nbr[a].add(b)
                nbr[b].add(a)

    edges_list = sorted(edges_set)  # deterministic base ordering
    E = len(edges_list)
    target_swaps = int(round(beta * E))

    if target_swaps > 0 and E >= 2:
        attempts = 0
        max_attempts = max_factor * max(1, target_swaps)
        swaps_done = 0

        while swaps_done < target_swaps and attempts < max_attempts:
            attempts += 1
            i = int(rng.integers(0, E))
            j = int(rng.integers(0, E))
            if i == j:
                continue

            a, b = edges_list[i]
            c, d = edges_list[j]
            if len({a, b, c, d}) < 4:
                continue

            # Two possible swap patterns
            if rng.random() < 0.5:
                x1 = _norm_edge(a, d)
                x2 = _norm_edge(c, b)
            else:
                x1 = _norm_edge(a, c)
                x2 = _norm_edge(b, d)

            if x1[0] == x1[1] or x2[0] == x2[1]:
                continue
            if x1 in edges_set or x2 in edges_set:
                continue

            old1 = (a, b)
            old2 = (c, d)

            # Apply swap updates
            edges_set.remove(old1)
            edges_set.remove(old2)
            nbr[a].remove(b); nbr[b].remove(a)
            nbr[c].remove(d); nbr[d].remove(c)

            edges_set.add(x1); edges_set.add(x2)
            u1, v1 = x1
            u2, v2 = x2
            nbr[u1].add(v1); nbr[v1].add(u1)
            nbr[u2].add(v2); nbr[v2].add(u2)

            if keep_connected and not _is_connected(N, edges_set):
                # Revert
                edges_set.remove(x1); edges_set.remove(x2)
                nbr[u1].remove(v1); nbr[v1].remove(u1)
                nbr[u2].remove(v2); nbr[v2].remove(u2)

                edges_set.add(old1); edges_set.add(old2)
                nbr[a].add(b); nbr[b].add(a)
                nbr[c].add(d); nbr[d].add(c)
                continue

            edges_list[i] = x1
            edges_list[j] = x2
            swaps_done += 1

    edges = _coalesce_undirected_edges(list(edges_set), N)
    edge_index, degrees, arc_slices = build_arc_index(edges, N)
    coords = _coords_cycle(N)
    return edge_index, degrees, coords, arc_slices


# ---- d-regular (simple) ---------------------------------------------------------------------------


def _configuration_model_simple_d_regular(
    N: int, d: int, rng: np.random.Generator, max_tries: int = 5000
) -> list[tuple[int, int]]:
    """
    Sample a simple d-regular undirected graph using a configuration-model retry scheme.
    """
    if d < 0 or d >= N:
        raise ValueError("Need 0 <= d < N.")
    if (N * d) % 2 != 0:
        raise ValueError("N*d must be even for a d-regular simple graph.")

    for _ in range(max_tries):
        stubs = np.repeat(np.arange(N, dtype=np.int64), d)
        rng.shuffle(stubs)
        ok = True
        edges = []
        for i in range(0, len(stubs), 2):
            u = int(stubs[i])
            v = int(stubs[i + 1])
            if u == v:
                ok = False
                break
            a, b = (u, v) if u < v else (v, u)
            edges.append((a, b))
        if not ok:
            continue
        edges = _coalesce_undirected_edges(edges, N)
        deg = np.zeros(N, dtype=np.int64)
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        if np.all(deg == d):
            return edges
    raise RuntimeError("Failed to sample a simple d-regular graph within retries.")


def d_regular_random_graph(
    N: int, d: int, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simple d-regular graph (retry-based configuration model).
    """
    if N < 0:
        raise ValueError("N must be nonnegative.")
    rng = _rng(seed)
    if d == 0:
        edges: list[tuple[int, int]] = []
    else:
        edges = _configuration_model_simple_d_regular(N, d, rng)
    edge_index, degrees, arc_slices = build_arc_index(edges, N)
    coords = _coords_line(N)
    return edge_index, degrees, coords, arc_slices


# ======================================================================================
# Small-world on a 2D lattice (coords fixed) + degree-preserving swap variant
# ======================================================================================


def watts_strogatz_grid_graph(
    Lx: int,
    Ly: int | None = None,
    kx: int = 2,
    ky: int = 2,
    beta: float = 0.1,
    seed: int | None = None,
    *,
    torus: bool = False,
    degree_preserving: bool = True,
    keep_connected: bool = False,
    max_factor: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    2D small-world on an Lx×Ly lattice with fixed grid coordinates.

    If degree_preserving=True (default), we do degree-preserving swaps (recommended for ACPL).
    If degree_preserving=False, we do endpoint rewiring (degrees can vary), but deterministically
    iterate edges in sorted order and avoid duplicates/self-loops.

    torus=True enables periodic wrap on both axes.
    """
    if Ly is None:
        Ly = Lx
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx, Ly must be positive.")
    if kx < 0 or ky < 0:
        raise ValueError("kx, ky must be nonnegative.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0,1].")
    if max_factor <= 0:
        raise ValueError("max_factor must be positive.")

    if degree_preserving:
        return watts_strogatz_grid_graph_degree_preserving(
            Lx=Lx,
            Ly=Ly,
            kx=kx,
            ky=ky,
            beta=beta,
            torus=torus,
            seed=seed,
            max_factor=max_factor,
            keep_connected=keep_connected,
        )

    rng = _rng(seed)
    N = Lx * Ly

    def vid(x: int, y: int) -> int:
        return y * Lx + x

    edges_set: set[tuple[int, int]] = set()

    # base lattice
    for y in range(Ly):
        for x in range(Lx):
            u = vid(x, y)
            for dx in range(1, kx + 1):
                if torus:
                    v = vid((x + dx) % Lx, y)
                    edges_set.add(_norm_edge(u, v))
                else:
                    if x + dx < Lx:
                        v = vid(x + dx, y)
                        edges_set.add(_norm_edge(u, v))
            for dy in range(1, ky + 1):
                if torus:
                    v = vid(x, (y + dy) % Ly)
                    edges_set.add(_norm_edge(u, v))
                else:
                    if y + dy < Ly:
                        v = vid(x, y + dy)
                        edges_set.add(_norm_edge(u, v))

    # endpoint rewiring (degrees may vary)
    edges = sorted(edges_set)  # deterministic order
    for a, b in edges:
        if rng.random() < beta:
            u = min(a, b)
            for _ in range(256):
                w = int(rng.integers(0, N))
                if w == u:
                    continue
                nw = _norm_edge(u, w)
                if nw not in edges_set:
                    edges_set.remove(_norm_edge(a, b))
                    edges_set.add(nw)
                    break

    edges = _coalesce_undirected_edges(list(edges_set), N)

    if keep_connected and not _is_connected(N, edges):
        # If you asked for connectivity but used non-degree-preserving rewiring,
        # we resample by rerunning with a bumped seed-like stream (deterministic given seed).
        # Simpler: just error, since this path is not recommended for ACPL.
        raise RuntimeError("Non-degree-preserving WS-grid produced a disconnected graph.")

    edge_index, degrees, arc_slices = build_arc_index(edges, N)
    coords = _coords_grid(Lx, Ly)
    return edge_index, degrees, coords, arc_slices


def watts_strogatz_grid_graph_degree_preserving(
    Lx: int,
    Ly: int | None = None,
    kx: int = 1,
    ky: int = 1,
    beta: float = 0.1,
    *,
    torus: bool = False,
    seed: int | None = None,
    max_factor: int = 20,
    keep_connected: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Degree-preserving small-world on an Lx×Ly lattice using double-edge swaps.

    - Build a base lattice (with optional torus wrap).
    - Perform target_swaps = round(beta * |E|) successful swaps.
    - Degrees remain exactly preserved.
    - If keep_connected=True, reject swaps that disconnect the graph.
    """
    if Ly is None:
        Ly = Lx
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx, Ly must be positive.")
    if kx < 0 or ky < 0:
        raise ValueError("kx, ky must be nonnegative.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0,1].")
    if max_factor <= 0:
        raise ValueError("max_factor must be positive.")

    rng = _rng(seed)
    N = Lx * Ly

    def vid(x: int, y: int) -> int:
        return y * Lx + x

    edges_set: set[tuple[int, int]] = set()

    # Base lattice
    for y in range(Ly):
        for x in range(Lx):
            u = vid(x, y)
            for dx in range(1, kx + 1):
                if torus:
                    v = vid((x + dx) % Lx, y)
                    edges_set.add(_norm_edge(u, v))
                else:
                    if x + dx < Lx:
                        v = vid(x + dx, y)
                        edges_set.add(_norm_edge(u, v))
            for dy in range(1, ky + 1):
                if torus:
                    v = vid(x, (y + dy) % Ly)
                    edges_set.add(_norm_edge(u, v))
                else:
                    if y + dy < Ly:
                        v = vid(x, y + dy)
                        edges_set.add(_norm_edge(u, v))

    edges_list = sorted(edges_set)  # deterministic base
    E = len(edges_list)
    target_swaps = int(round(beta * E))

    if target_swaps == 0:
        edges = _coalesce_undirected_edges(list(edges_set), N)
        edge_index, degrees, arc_slices = build_arc_index(edges, N)
        coords = _coords_grid(Lx, Ly)
        return edge_index, degrees, coords, arc_slices

    attempts = 0
    max_attempts = max_factor * max(1, target_swaps)
    swaps_done = 0

    while swaps_done < target_swaps and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, E))
        j = int(rng.integers(0, E))
        if i == j:
            continue

        a, b = edges_list[i]
        c, d = edges_list[j]
        if len({a, b, c, d}) < 4:
            continue

        if rng.random() < 0.5:
            x1, x2 = _norm_edge(a, d), _norm_edge(c, b)
        else:
            x1, x2 = _norm_edge(a, c), _norm_edge(b, d)

        if x1[0] == x1[1] or x2[0] == x2[1]:
            continue
        if x1 in edges_set or x2 in edges_set:
            continue

        old1 = _norm_edge(a, b)
        old2 = _norm_edge(c, d)

        # apply
        edges_set.remove(old1)
        edges_set.remove(old2)
        edges_set.add(x1)
        edges_set.add(x2)

        if keep_connected and not _is_connected(N, edges_set):
            # revert
            edges_set.remove(x1)
            edges_set.remove(x2)
            edges_set.add(old1)
            edges_set.add(old2)
            continue

        edges_list[i] = x1
        edges_list[j] = x2
        swaps_done += 1

    edges = _coalesce_undirected_edges(list(edges_set), N)
    edge_index, degrees, arc_slices = build_arc_index(edges, N)
    coords = _coords_grid(Lx, Ly)
    return edge_index, degrees, coords, arc_slices









def ws_graph(
    N: int,
    k: int,
    beta: float,
    seed: int | None = None,
    *,
    sanitize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Alias for Hydra family name "ws"
    return watts_strogatz_graph(N=N, k=k, beta=beta, seed=seed, sanitize=sanitize)


def ws_grid_graph(
    Lx: int,
    Ly: int | None = None,
    kx: int = 1,
    ky: int = 1,
    beta: float = 0.1,
    seed: int | None = None,
    *,
    torus: bool = False,
    sanitize: bool = True,
    keep_connected: bool = False,
    max_tries: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Alias for Hydra family name "ws_grid"
    return watts_strogatz_grid_graph(
        Lx=Lx,
        Ly=Ly,
        kx=kx,
        ky=ky,
        beta=beta,
        seed=seed,
        torus=torus,
        sanitize=sanitize,
        keep_connected=keep_connected,
        max_tries=max_tries,
    )




# ======================================================================================
# Random geometric graph G(N,r) in [0,1]^d (optional torus metric)
# ======================================================================================


def random_geometric_graph(
    N: int,
    radius: float,
    dim: int = 2,
    seed: int | None = None,
    torus: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random geometric graph (RGG) in the unit d-cube.

    - Sample N points uniformly in [0,1]^d.
    - Connect (u,v) iff Euclidean distance <= radius (with periodic wrap if torus=True).
    """
    if N < 0:
        raise ValueError("N must be nonnegative.")
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if radius < 0:
        raise ValueError("radius must be nonnegative.")
    rng = _rng(seed)

    coords_np = rng.random((N, dim), dtype=np.float64).astype(np.float32)

    edges: list[tuple[int, int]] = []
    if N >= 2:
        for u in range(N):
            xu = coords_np[u]
            for v in range(u + 1, N):
                xv = coords_np[v]
                dvec = np.abs(xu - xv)
                if torus:
                    dvec = np.minimum(dvec, 1.0 - dvec)
                if float(np.dot(dvec, dvec)) <= radius * radius:
                    edges.append((u, v))

    edges = _coalesce_undirected_edges(edges, N)
    edge_index, degrees, arc_slices = build_arc_index(edges, N)
    coords = _ensure_torch_float(coords_np)
    return edge_index, degrees, coords, arc_slices

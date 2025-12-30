# acpl/eval/roles.py
from __future__ import annotations

"""
acpl.eval.roles
================

“Roles” are discrete labels (or feature bundles) assigned to nodes to support:
  - evaluation slices (e.g., performance stratified by node role),
  - analysis artifacts (role histograms, masks),
  - ACPL analogue of “ESOL target bins” but for graphs: bucketize nodes by their
    structural / task-relative position.

Core idea
---------
Given a graph G and (optionally) a source/target pair, we define role labels such as:
  - distance-to-source bins, distance-to-target bins
  - nodes lying on a shortest path(s) between source and target
  - k-hop neighborhoods around anchors (source/target/start/goal)
  - degree bins (quantile/log/uniform)

This module is designed to be:
  - dependency-light (NetworkX optional),
  - deterministic and reproducible,
  - efficient for repeated eval calls (small LRU adjacency cache),
  - flexible across the repo’s typical graph representations (edge_index/num_nodes).

Typical usage
-------------
    role_ids = bucketize_nodes(G, source=0, target=15, scheme="dist_bins")

    mask_sp = on_shortest_path_mask(G, source=0, target=15)
    mask_2hop = khop_mask(G, centers=[0, 15], k=2)

    deg_bin = bucketize_nodes(G, scheme="degree_bins", degree_num_bins=8)

If you want “feature bundles” (for slicing or export):
    feats = bucketize_nodes(G, source=0, target=15, scheme="dist_bins", return_features=True)
    feats["role_id"]   # LongTensor [N]
    feats["d_src"]     # LongTensor [N] (unreachable=-1)
    feats["d_tgt"]     # LongTensor [N]
    feats["on_sp"]     # BoolTensor [N]
    feats["deg"]       # LongTensor [N]
    feats["deg_bin"]   # LongTensor [N]
"""

from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from hashlib import blake2b
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union, Optional, Dict, List

import numpy as np
import torch

try:  # optional
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

__all__ = [
    # config / enums
    "RoleScheme",
    "RoleConfig",
    # graph coercion
    "coerce_edge_index_num_nodes",
    # distances / masks
    "shortest_path_distances",
    "on_shortest_path_mask",
    "khop_mask",
    # binning helpers
    "degree_bins",
    "distance_bins",
    "pack_roles",
    "role_vocab_size",
    # main API
    "bucketize_nodes",
]


# -----------------------------------------------------------------------------
# Enums / config
# -----------------------------------------------------------------------------

class RoleScheme(str, Enum):
    """
    Supported role schemes.

    - dist_bins:
        Combine distance-to-source and distance-to-target (binned/capped), optionally
        include on-shortest-path bit and/or degree bin.
    - dist_source:
        Distance-to-source only (binned/capped).
    - dist_target:
        Distance-to-target only (binned/capped).
    - khop:
        Hop-distance from a set of centers (binned/capped by k).
    - degree_bins:
        Degree-only bins.
    """
    DIST_BINS = "dist_bins"
    DIST_SOURCE = "dist_source"
    DIST_TARGET = "dist_target"
    KHOP = "khop"
    DEGREE_BINS = "degree_bins"


@dataclass(frozen=True)
class RoleConfig:
    """
    Configuration for bucketize_nodes().

    Parameters
    ----------
    scheme:
        One of RoleScheme.
    directed:
        Treat edge_index as directed (uses reverse BFS where relevant).
    dist_cap:
        Cap distance bins at this integer. Example: dist_cap=6 means bins 0..6 and 6 stands
        for ">=6"; unreachable is a separate bin (index=dist_cap+1).
    include_on_shortest_path:
        If True and source/target provided, include a bit indicating whether node lies on
        *some* shortest path from source to target.
    include_degree_bin:
        If True, include degree bin as another factor in the packed role id (for dist_bins).
    degree_num_bins:
        Number of bins for degree_bins (quantile/log/uniform).
    degree_bin_strategy:
        "quantile" (default), "log", or "uniform".
    khop_k:
        For KHOP scheme: maximum hop distance K. Distances are capped at K; unreachable is separate.
    """
    scheme: RoleScheme = RoleScheme.DIST_BINS
    directed: bool = False

    dist_cap: int = 6
    include_on_shortest_path: bool = True
    include_degree_bin: bool = False

    degree_num_bins: int = 6
    degree_bin_strategy: str = "quantile"

    khop_k: int = 2


# -----------------------------------------------------------------------------
# Graph coercion
# -----------------------------------------------------------------------------

GraphLike = Union[
    torch.Tensor,            # edge_index
    np.ndarray,              # edge_index
    "nx.Graph",              # networkx graph (optional)
    Mapping[str, Any],       # {"edge_index": ..., "num_nodes": ...}
    Any,                     # object with .edge_index and .num_nodes/.N
]


def _as_cpu_long_edge_index(edge_index: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Return edge_index as np.int64 array with shape (2, E)."""
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index.detach().to("cpu")
        if ei.ndim != 2:
            raise ValueError(f"edge_index must be 2D, got shape={tuple(ei.shape)}")
        if ei.shape[0] == 2:
            arr = ei.to(dtype=torch.long).numpy()
        elif ei.shape[1] == 2:
            arr = ei.t().to(dtype=torch.long).numpy()
        else:
            raise ValueError(f"edge_index must have shape (2,E) or (E,2), got {tuple(ei.shape)}")
        return arr.astype(np.int64, copy=False)

    arr = np.asarray(edge_index)
    if arr.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape={arr.shape}")
    if arr.shape[0] == 2:
        return arr.astype(np.int64, copy=False)
    if arr.shape[1] == 2:
        return arr.T.astype(np.int64, copy=False)
    raise ValueError(f"edge_index must have shape (2,E) or (E,2), got {arr.shape}")


def coerce_edge_index_num_nodes(
    G: GraphLike,
    num_nodes: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Coerce a graph-like object into (edge_index_np, num_nodes).

    Accepted formats:
      - torch.Tensor or np.ndarray edge_index (2,E) or (E,2)
      - networkx Graph / DiGraph (if networkx installed)
      - Mapping with keys: 'edge_index' and optional 'num_nodes'/'N'
      - object with attributes: .edge_index and .num_nodes or .N

    Returns
    -------
    edge_index: np.ndarray (2,E) int64
    num_nodes: int
    """
    # networkx graph
    if nx is not None and isinstance(G, (nx.Graph, nx.DiGraph)):  # type: ignore[arg-type]
        nodes = list(G.nodes())
        if not nodes:
            if num_nodes is None:
                raise ValueError("Empty graph: num_nodes must be provided.")
            return np.zeros((2, 0), dtype=np.int64), int(num_nodes)

        # Map arbitrary node labels -> 0..N-1 deterministically by sorted order
        # (stable across python versions if labels are comparable)
        try:
            nodes_sorted = sorted(nodes)
        except Exception:
            # fallback: keep insertion order
            nodes_sorted = nodes
        idx = {n: i for i, n in enumerate(nodes_sorted)}
        edges = list(G.edges())
        ei = np.zeros((2, len(edges)), dtype=np.int64)
        for j, (u, v) in enumerate(edges):
            ei[0, j] = idx[u]
            ei[1, j] = idx[v]
        N = len(nodes_sorted) if num_nodes is None else int(num_nodes)
        return ei, N

    # mapping
    if isinstance(G, Mapping):
        if "edge_index" not in G:
            raise KeyError("Mapping graph must contain key 'edge_index'.")
        ei = _as_cpu_long_edge_index(G["edge_index"])
        N0 = G.get("num_nodes", G.get("N", None))
        N = int(num_nodes if num_nodes is not None else N0) if (num_nodes is not None or N0 is not None) else None
        if N is None:
            raise ValueError("num_nodes could not be inferred; provide num_nodes explicitly.")
        return ei, int(N)

    # edge_index raw
    if isinstance(G, (torch.Tensor, np.ndarray)):
        ei = _as_cpu_long_edge_index(G)
        if num_nodes is None:
            if ei.size == 0:
                raise ValueError("num_nodes must be provided for empty edge_index.")
            mx = int(ei.max())
            num_nodes = mx + 1
        return ei, int(num_nodes)

    # object with attrs
    if hasattr(G, "edge_index"):
        ei = _as_cpu_long_edge_index(getattr(G, "edge_index"))
        if num_nodes is None:
            if hasattr(G, "num_nodes"):
                num_nodes = int(getattr(G, "num_nodes"))
            elif hasattr(G, "N"):
                num_nodes = int(getattr(G, "N"))
            else:
                if ei.size == 0:
                    raise ValueError("num_nodes must be provided for empty edge_index.")
                num_nodes = int(ei.max()) + 1
        return ei, int(num_nodes)

    raise TypeError(f"Unsupported graph type for roles: {type(G)}")


# -----------------------------------------------------------------------------
# Adjacency cache (LRU) for repeated BFS
# -----------------------------------------------------------------------------

class _AdjLRU:
    def __init__(self, maxsize: int = 32) -> None:
        self.maxsize = int(maxsize)
        self._od: "OrderedDict[Tuple[str, int, bool], Tuple[List[List[int]], List[List[int]]]]" = OrderedDict()

    def get(self, key: Tuple[str, int, bool]) -> Optional[Tuple[List[List[int]], List[List[int]]]]:
        if key in self._od:
            self._od.move_to_end(key)
            return self._od[key]
        return None

    def put(self, key: Tuple[str, int, bool], val: Tuple[List[List[int]], List[List[int]]]) -> None:
        self._od[key] = val
        self._od.move_to_end(key)
        while len(self._od) > self.maxsize:
            self._od.popitem(last=False)


_ADJ_CACHE = _AdjLRU(maxsize=64)


def _hash_edge_index(ei: np.ndarray, *, canonicalize: bool = True) -> str:
    """
    Hash edge_index for adjacency caching.

    canonicalize=True sorts edges lexicographically for stable hashes (better cache hits).
    For very large E, sorting can be expensive; we only canonicalize up to a threshold.
    """
    if ei.size == 0:
        return "empty"

    e = ei
    if canonicalize and e.shape[1] <= 200_000:
        # lexsort by (u, v)
        order = np.lexsort((e[1], e[0]))
        e = e[:, order]

    h = blake2b(digest_size=16)
    h.update(e.astype(np.int64, copy=False).tobytes())
    return h.hexdigest()


def _build_adj(
    ei: np.ndarray,
    num_nodes: int,
    *,
    directed: bool,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build adjacency lists:
      - adj[u] = out-neighbors (or neighbors if undirected)
      - radj[u] = reverse neighbors (for directed shortest-path-to-target)
    """
    N = int(num_nodes)
    adj: List[List[int]] = [[] for _ in range(N)]
    radj: List[List[int]] = [[] for _ in range(N)]
    if ei.size == 0:
        return adj, radj

    u = ei[0].astype(np.int64, copy=False)
    v = ei[1].astype(np.int64, copy=False)

    # filter invalid edges defensively
    mask = (u >= 0) & (u < N) & (v >= 0) & (v < N)
    u = u[mask]
    v = v[mask]

    for uu, vv in zip(u.tolist(), v.tolist(), strict=False):
        adj[uu].append(vv)
        radj[vv].append(uu)
        if not directed:
            adj[vv].append(uu)
            radj[uu].append(vv)

    return adj, radj


def _get_adj(
    ei: np.ndarray,
    num_nodes: int,
    *,
    directed: bool,
) -> Tuple[List[List[int]], List[List[int]]]:
    key = (_hash_edge_index(ei), int(num_nodes), bool(directed))
    got = _ADJ_CACHE.get(key)
    if got is not None:
        return got
    val = _build_adj(ei, num_nodes, directed=directed)
    _ADJ_CACHE.put(key, val)
    return val


# -----------------------------------------------------------------------------
# BFS distances / masks
# -----------------------------------------------------------------------------

def shortest_path_distances(
    G: GraphLike,
    *,
    num_nodes: int | None = None,
    sources: int | Sequence[int] = 0,
    directed: bool = False,
    max_dist: int | None = None,
    reverse: bool = False,
) -> np.ndarray:
    """
    Compute shortest-path distances from one or many sources using BFS (unweighted).

    Parameters
    ----------
    G:
        Graph-like (edge_index / nx.Graph / object).
    sources:
        Source node id or list of source ids.
    directed:
        Treat edges as directed. If reverse=True, BFS uses reverse adjacency.
    max_dist:
        If provided, stops expanding beyond this distance (faster masks).
    reverse:
        If True and directed=True, compute distances on reversed edges (useful for
        distances "to target" in directed graphs).

    Returns
    -------
    dist: np.ndarray[int64] shape (N,)
        dist[v] = shortest distance from any source to v; unreachable = -1.
    """
    ei, N = coerce_edge_index_num_nodes(G, num_nodes=num_nodes)
    adj, radj = _get_adj(ei, N, directed=directed)

    if isinstance(sources, (int, np.integer)):
        srcs = [int(sources)]
    else:
        srcs = [int(s) for s in sources]

    srcs = [s for s in srcs if 0 <= s < N]
    dist = np.full((N,), -1, dtype=np.int64)
    if not srcs:
        return dist

    q: deque[int] = deque()
    for s in srcs:
        dist[s] = 0
        q.append(s)

    use_adj = radj if (directed and reverse) else adj
    cap = None if max_dist is None else int(max_dist)

    while q:
        u = q.popleft()
        du = int(dist[u])
        if cap is not None and du >= cap:
            continue
        for w in use_adj[u]:
            if dist[w] != -1:
                continue
            dist[w] = du + 1
            q.append(w)

    return dist


def on_shortest_path_mask(
    G: GraphLike,
    *,
    source: int,
    target: int,
    num_nodes: int | None = None,
    directed: bool = False,
) -> np.ndarray:
    """
    Mask of nodes lying on *some* shortest path from source to target.

    Uses the characterization:
      v is on a shortest s->t path iff d_s[v] + d_to_t[v] == d_s[t]
    where d_to_t is computed via reverse BFS in directed graphs.

    Returns
    -------
    mask: np.ndarray[bool] shape (N,)
    """
    ei, N = coerce_edge_index_num_nodes(G, num_nodes=num_nodes)
    s = int(source)
    t = int(target)
    if not (0 <= s < N and 0 <= t < N):
        raise ValueError(f"source/target must be in [0,{N-1}], got source={s} target={t}")

    d_s = shortest_path_distances(ei, num_nodes=N, sources=s, directed=directed, reverse=False)
    d_to_t = shortest_path_distances(ei, num_nodes=N, sources=t, directed=directed, reverse=directed)

    d_st = int(d_s[t])
    if d_st < 0:
        # unreachable => no shortest path exists
        return np.zeros((N,), dtype=bool)

    ok = (d_s >= 0) & (d_to_t >= 0) & ((d_s + d_to_t) == d_st)
    return ok.astype(bool, copy=False)


def khop_mask(
    G: GraphLike,
    *,
    centers: Sequence[int],
    k: int,
    num_nodes: int | None = None,
    directed: bool = False,
    include_centers: bool = True,
) -> np.ndarray:
    """
    Boolean mask for nodes within k hops from any center.

    Parameters
    ----------
    centers:
        Anchor nodes.
    k:
        Hop radius.
    include_centers:
        If False, centers themselves are excluded.

    Returns
    -------
    mask: np.ndarray[bool] shape (N,)
    """
    ei, N = coerce_edge_index_num_nodes(G, num_nodes=num_nodes)
    adj, _radj = _get_adj(ei, N, directed=directed)

    K = int(k)
    if K < 0:
        raise ValueError("k must be >= 0")

    c = [int(x) for x in centers]
    c = [x for x in c if 0 <= x < N]
    if not c:
        return np.zeros((N,), dtype=bool)

    dist = np.full((N,), -1, dtype=np.int64)
    q: deque[int] = deque()
    for x in c:
        dist[x] = 0
        q.append(x)

    while q:
        u = q.popleft()
        du = int(dist[u])
        if du >= K:
            continue
        for w in adj[u]:
            if dist[w] != -1:
                continue
            dist[w] = du + 1
            q.append(w)

    m = (dist >= 0) & (dist <= K)
    if not include_centers:
        for x in c:
            m[x] = False
    return m.astype(bool, copy=False)


# -----------------------------------------------------------------------------
# Binning helpers
# -----------------------------------------------------------------------------

def distance_bins(dist: np.ndarray, *, cap: int) -> np.ndarray:
    """
    Convert integer distances into bins:
      - dist in [0..cap-1] => same
      - dist >= cap => cap  (meaning ">= cap")
      - dist == -1 (unreachable) => cap+1

    Returns
    -------
    b: np.ndarray[int64] shape (N,) with values in [0..cap+1]
    """
    c = int(cap)
    if c < 0:
        raise ValueError("cap must be >= 0")
    d = dist.astype(np.int64, copy=False)
    b = np.empty_like(d)
    unreachable = (d < 0)
    b[unreachable] = c + 1
    dd = d.copy()
    dd[unreachable] = 0
    b[~unreachable] = np.minimum(dd[~unreachable], c)
    return b


def degree_bins(
    deg: np.ndarray,
    *,
    num_bins: int = 6,
    strategy: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin node degrees into [0..num_bins-1], returning (bin_id, edges).

    strategy:
      - "quantile": equal-count bins based on empirical quantiles (default, robust)
      - "uniform":  uniform bins between [min, max]
      - "log":      log-spaced bins between [1, max+1] (good for heavy tails)

    Notes
    -----
    - Deterministic: uses numpy quantile with fixed method (where available).
    - If degrees are constant, all nodes go to bin 0 and edges are trivial.

    Returns
    -------
    bin_id: np.ndarray[int64] shape (N,)
    edges: np.ndarray[float64] shape (num_bins+1,)
        Monotone edges so that bin_id = digitize(deg, edges[1:-1], right=False).
    """
    x = deg.astype(np.float64, copy=False)
    B = int(num_bins)
    if B <= 0:
        raise ValueError("num_bins must be >= 1")

    if x.size == 0:
        return np.zeros((0,), dtype=np.int64), np.array([0.0, 1.0], dtype=np.float64)

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin + 1e-12:
        edges = np.linspace(xmin, xmin + 1.0, B + 1, dtype=np.float64)
        return np.zeros_like(x, dtype=np.int64), edges

    strat = str(strategy).lower().strip()
    if strat == "uniform":
        edges = np.linspace(xmin, xmax, B + 1, dtype=np.float64)
    elif strat == "log":
        # log-space on (deg+1) to include 0 degree cleanly
        xp = x + 1.0
        lo = float(np.min(xp))
        hi = float(np.max(xp))
        lo = max(lo, 1.0)
        edges_p = np.logspace(np.log10(lo), np.log10(hi), B + 1, dtype=np.float64)
        edges = edges_p - 1.0
    else:
        # quantile (default)
        qs = np.linspace(0.0, 1.0, B + 1, dtype=np.float64)
        try:
            edges = np.quantile(x, qs, method="linear")  # numpy>=1.22
        except TypeError:  # pragma: no cover
            edges = np.quantile(x, qs, interpolation="linear")  # older numpy
        edges[0] = xmin
        edges[-1] = xmax

    # ensure strict monotonicity (numerical ties can break digitize)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    # bin assignment (0..B-1)
    inner = edges[1:-1]
    bins = np.digitize(x, inner, right=False).astype(np.int64, copy=False)
    bins = np.clip(bins, 0, B - 1)
    return bins, edges


def pack_roles(components: Sequence[np.ndarray], radices: Sequence[int]) -> np.ndarray:
    """
    Pack multiple discrete components into a single integer id using mixed radix.

    Example: components=[a (0..A-1), b (0..B-1), c (0..C-1)]
             radices=[A, B, C]
             id = a*(B*C) + b*(C) + c

    Parameters
    ----------
    components:
        Sequence of arrays each shape (N,) int64-like.
    radices:
        Base sizes for each component.

    Returns
    -------
    role_id: np.ndarray[int64] shape (N,)
    """
    if len(components) != len(radices):
        raise ValueError("components and radices must have the same length")
    if not components:
        raise ValueError("components cannot be empty")

    N = int(np.asarray(components[0]).shape[0])
    for c in components:
        if int(np.asarray(c).shape[0]) != N:
            raise ValueError("all components must have the same length")

    rad = [int(r) for r in radices]
    if any(r <= 0 for r in rad):
        raise ValueError("all radices must be >= 1")

    out = np.zeros((N,), dtype=np.int64)
    mul = 1
    for comp, base in zip(reversed(components), reversed(rad), strict=False):
        cc = np.asarray(comp, dtype=np.int64)
        out += cc * mul
        mul *= base
    return out


def role_vocab_size(radices: Sequence[int]) -> int:
    """Total number of packed role ids for a given mixed-radix spec."""
    s = 1
    for r in radices:
        rr = int(r)
        if rr <= 0:
            raise ValueError("radices must be >= 1")
        s *= rr
    return int(s)


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------

def bucketize_nodes(
    G: GraphLike,
    *,
    source: int | None = None,
    target: int | None = None,
    num_nodes: int | None = None,
    scheme: str | RoleScheme = RoleScheme.DIST_BINS,
    directed: bool = False,
    dist_cap: int = 6,
    include_on_shortest_path: bool = True,
    include_degree_bin: bool = False,
    degree_num_bins: int = 6,
    degree_bin_strategy: str = "quantile",
    khop_k: int = 2,
    khop_centers: Sequence[int] | None = None,
    return_features: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    """
    Compute role labels (and optionally auxiliary role features) for all nodes.

    Parameters
    ----------
    G:
        Graph-like input.
    source, target:
        Anchor nodes for distance/path roles. Required for scheme="dist_bins" unless you
        intentionally set include_on_shortest_path=False and only want degree/bin roles.
    scheme:
        Role scheme (RoleScheme or string).
    directed:
        Whether to treat edges as directed (affects distances and shortest-path masks).
    dist_cap:
        Distance cap for distance bins (see distance_bins()).
    include_on_shortest_path:
        Only used for scheme="dist_bins". Adds an on-shortest-path bit if source/target provided.
    include_degree_bin:
        Only used for scheme="dist_bins". Adds degree bin as a factor.
    degree_*:
        Degree bin configuration (used for scheme="degree_bins" and optionally dist_bins).
    khop_k, khop_centers:
        Used for scheme="khop". If khop_centers is None, uses [source,target] if present else raises.
    return_features:
        If True, returns a dict of tensors instead of just role_id.
    device:
        Torch device for returned tensors. Default: cpu.

    Returns
    -------
    role_id: torch.LongTensor [N]
      or
    features: dict[str, torch.Tensor]
        Always includes:
          - "role_id": LongTensor [N]
        Additionally (when available):
          - "d_src", "d_tgt": LongTensor [N] (unreachable=-1)
          - "d_src_bin", "d_tgt_bin": LongTensor [N]
          - "on_sp": BoolTensor [N]
          - "deg": LongTensor [N]
          - "deg_bin": LongTensor [N]
          - "radices": LongTensor [K] describing role_id packing bases
          - "vocab_size": LongTensor scalar
    """
    ei, N = coerce_edge_index_num_nodes(G, num_nodes=num_nodes)
    dev = torch.device(device) if device is not None else torch.device("cpu")

    # degrees (undirected degree if directed=False; otherwise out-degree)
    deg = np.zeros((N,), dtype=np.int64)
    if ei.size > 0:
        u = ei[0]
        v = ei[1]
        mask = (u >= 0) & (u < N) & (v >= 0) & (v < N)
        u = u[mask]
        v = v[mask]
        np.add.at(deg, u, 1)
        if not directed:
            np.add.at(deg, v, 1)

    scheme0 = RoleScheme(str(scheme)) if not isinstance(scheme, RoleScheme) else scheme

    # -------------------------------------------------------------------------
    # DEGREE_BINS only
    # -------------------------------------------------------------------------
    if scheme0 == RoleScheme.DEGREE_BINS:
        dbin, edges = degree_bins(deg, num_bins=degree_num_bins, strategy=degree_bin_strategy)
        role_id = dbin.astype(np.int64, copy=False)
        out = {
            "role_id": torch.as_tensor(role_id, dtype=torch.long, device=dev),
            "deg": torch.as_tensor(deg, dtype=torch.long, device=dev),
            "deg_bin": torch.as_tensor(dbin, dtype=torch.long, device=dev),
            "degree_edges": torch.as_tensor(edges, dtype=torch.float64, device=dev),
            "radices": torch.as_tensor([int(degree_num_bins)], dtype=torch.long, device=dev),
            "vocab_size": torch.as_tensor(int(degree_num_bins), dtype=torch.long, device=dev),
        }
        return out if return_features else out["role_id"]

    # -------------------------------------------------------------------------
    # KHOP roles
    # -------------------------------------------------------------------------
    if scheme0 == RoleScheme.KHOP:
        K = int(khop_k)
        if khop_centers is None:
            cs: List[int] = []
            if source is not None:
                cs.append(int(source))
            if target is not None:
                cs.append(int(target))
            if not cs:
                raise ValueError("scheme='khop' requires khop_centers or (source/target).")
            khop_centers = cs

        # hop distances from centers (capped)
        dist = shortest_path_distances(
            ei, num_nodes=N, sources=list(khop_centers), directed=directed, max_dist=K, reverse=False
        )
        # convert to bins: 0..K, unreachable -> K+1
        d_bin = distance_bins(dist, cap=K)
        role_id = d_bin.astype(np.int64, copy=False)

        out = {
            "role_id": torch.as_tensor(role_id, dtype=torch.long, device=dev),
            "hop_dist": torch.as_tensor(dist, dtype=torch.long, device=dev),
            "hop_bin": torch.as_tensor(d_bin, dtype=torch.long, device=dev),
            "radices": torch.as_tensor([K + 2], dtype=torch.long, device=dev),  # includes unreachable
            "vocab_size": torch.as_tensor(int(K + 2), dtype=torch.long, device=dev),
        }
        return out if return_features else out["role_id"]

    # -------------------------------------------------------------------------
    # Distance-only roles
    # -------------------------------------------------------------------------
    if scheme0 in (RoleScheme.DIST_SOURCE, RoleScheme.DIST_TARGET):
        if scheme0 == RoleScheme.DIST_SOURCE:
            if source is None:
                raise ValueError("scheme='dist_source' requires source.")
            dist = shortest_path_distances(ei, num_nodes=N, sources=int(source), directed=directed, reverse=False)
        else:
            if target is None:
                raise ValueError("scheme='dist_target' requires target.")
            dist = shortest_path_distances(ei, num_nodes=N, sources=int(target), directed=directed, reverse=directed)

        d_bin = distance_bins(dist, cap=int(dist_cap))
        role_id = d_bin.astype(np.int64, copy=False)
        out = {
            "role_id": torch.as_tensor(role_id, dtype=torch.long, device=dev),
            "dist": torch.as_tensor(dist, dtype=torch.long, device=dev),
            "dist_bin": torch.as_tensor(d_bin, dtype=torch.long, device=dev),
            "radices": torch.as_tensor([int(dist_cap) + 2], dtype=torch.long, device=dev),
            "vocab_size": torch.as_tensor(int(dist_cap) + 2, dtype=torch.long, device=dev),
        }
        return out if return_features else out["role_id"]

    # -------------------------------------------------------------------------
    # DIST_BINS (combined roles) - default
    # -------------------------------------------------------------------------
    if scheme0 != RoleScheme.DIST_BINS:
        raise ValueError(f"Unknown role scheme: {scheme0}")

    if source is None or target is None:
        raise ValueError("scheme='dist_bins' requires both source and target.")

    s = int(source)
    t = int(target)
    if not (0 <= s < N and 0 <= t < N):
        raise ValueError(f"source/target must be in [0,{N-1}], got source={s} target={t}")

    d_src = shortest_path_distances(ei, num_nodes=N, sources=s, directed=directed, reverse=False)
    d_tgt = shortest_path_distances(ei, num_nodes=N, sources=t, directed=directed, reverse=directed)

    d_src_bin = distance_bins(d_src, cap=int(dist_cap))
    d_tgt_bin = distance_bins(d_tgt, cap=int(dist_cap))

    # Optional on-shortest-path bit (only meaningful if s->t is reachable)
    if include_on_shortest_path:
        on_sp = on_shortest_path_mask(ei, source=s, target=t, num_nodes=N, directed=directed)
        on_sp_i = on_sp.astype(np.int64, copy=False)  # 0/1
    else:
        on_sp = np.zeros((N,), dtype=bool)
        on_sp_i = np.zeros((N,), dtype=np.int64)

    # Optional degree bin
    if include_degree_bin:
        dbin, edges = degree_bins(deg, num_bins=degree_num_bins, strategy=degree_bin_strategy)
    else:
        dbin = np.zeros((N,), dtype=np.int64)
        edges = np.array([], dtype=np.float64)

    # Pack components into role_id with mixed-radix
    # Radices:
    #   dist bins have size (dist_cap + 2) because we include unreachable at cap+1
    #   on_sp bit has size 2
    #   degree bin has size degree_num_bins (if included)
    R_dist = int(dist_cap) + 2
    comps: List[np.ndarray] = [d_src_bin, d_tgt_bin]
    rads: List[int] = [R_dist, R_dist]

    if include_degree_bin:
        comps.append(dbin)
        rads.append(int(degree_num_bins))

    if include_on_shortest_path:
        comps.append(on_sp_i)
        rads.append(2)

    role_id = pack_roles(comps, rads)

    out: Dict[str, torch.Tensor] = {
        "role_id": torch.as_tensor(role_id, dtype=torch.long, device=dev),
        "d_src": torch.as_tensor(d_src, dtype=torch.long, device=dev),
        "d_tgt": torch.as_tensor(d_tgt, dtype=torch.long, device=dev),
        "d_src_bin": torch.as_tensor(d_src_bin, dtype=torch.long, device=dev),
        "d_tgt_bin": torch.as_tensor(d_tgt_bin, dtype=torch.long, device=dev),
        "on_sp": torch.as_tensor(on_sp, dtype=torch.bool, device=dev),
        "deg": torch.as_tensor(deg, dtype=torch.long, device=dev),
        "deg_bin": torch.as_tensor(dbin, dtype=torch.long, device=dev),
        "radices": torch.as_tensor(np.asarray(rads, dtype=np.int64), dtype=torch.long, device=dev),
        "vocab_size": torch.as_tensor(int(role_vocab_size(rads)), dtype=torch.long, device=dev),
    }
    if include_degree_bin and edges.size > 0:
        out["degree_edges"] = torch.as_tensor(edges, dtype=torch.float64, device=dev)

    return out if return_features else out["role_id"]

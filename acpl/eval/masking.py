# acpl/eval/masking.py
from __future__ import annotations

"""
acpl.eval.masking
=================

Centralized, reusable masking utilities for *evaluation-time* perturbations.

Why this module exists
----------------------
You want masking to mean one consistent thing across eval suites, robustness sweeps,
and ablations—without scattering ad-hoc “zero X here” logic across scripts.

In ACPL-qwalk, "masking" is defined with a *primary* semantic:

    Primary semantic (default):
        **Feature masking** — zero out node feature rows `X[i] = 0` for masked nodes i.

This is the most stable and least invasive interpretation:
- Keeps graph topology intact (important for fair comparisons),
- Preserves tensor shapes (no remapping surprises),
- Works for both training-time and eval-time pipelines without special casing.

Optional semantics (explicitly requested by the caller):
- Zero only a PE slice (if PE is concatenated into X).
- Drop edges incident to masked nodes (topology perturbation without remapping nodes).
- Drop nodes entirely (induced subgraph, with remapping + bookkeeping).

Mask generators provided
------------------------
- mask_shortest_path_nodes
- mask_khop_around_target(k)
- mask_topk_centrality(k)
- mask_random_nodes(p, seed)

All mask generators return a `NodeMask` object: a boolean mask over nodes + metadata.

Design notes
------------
- Does not require torch_geometric. If you pass a PyG Data object, helpers can read
  `data.x` and `data.edge_index`, but everything remains optional.
- Centrality uses networkx when available; otherwise falls back to degree-based scoring.
- Determinism: random masking uses a local torch.Generator with an explicit seed.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

import torch

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore


__all__ = [
    # Core types / semantics
    "MaskingMode",
    "NodeMask",
    "MaskApplyResult",
    "ensure_edge_index",
    "infer_num_nodes",
    "apply_mask_to_X",
    "apply_mask_to_graph",
    "apply_mask",
    # Mask generators
    "mask_shortest_path_nodes",
    "mask_khop_around_target",
    "mask_topk_centrality",
    "mask_random_nodes",
    # Combinators
    "union_masks",
    "intersect_masks",
    "invert_mask",
    "select_nodes_from_mask",
    # Convenience registry
    "build_mask",
]


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

class MaskingMode(str, Enum):
    """
    How a node-mask should be applied.

    Primary/default: ZERO_X
      - Zero out X[node_mask] (feature masking) while leaving topology unchanged.

    Optional modes:
      - ZERO_PE: zero out only a PE slice within X (caller provides pe_slice/pe_dim).
      - ZERO_X_AND_PE: zero X and also zero pe_slice (useful if you want to keep
        non-PE features but explicitly enforce PE removal too).
      - DROP_EDGES: remove edges incident to masked nodes (X unchanged by default).
      - DROP_NODES: induce subgraph on unmasked nodes, remap node ids (changes shapes).
    """
    ZERO_X = "zero_x"
    ZERO_PE = "zero_pe"
    ZERO_X_AND_PE = "zero_x_and_pe"
    DROP_EDGES = "drop_edges"
    DROP_NODES = "drop_nodes"


@dataclass(frozen=True)
class NodeMask:
    """
    A node mask + provenance metadata.
    """
    node_mask: torch.Tensor  # bool [N]
    name: str
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.node_mask, torch.Tensor):
            raise TypeError("NodeMask.node_mask must be a torch.Tensor.")
        if self.node_mask.dtype != torch.bool:
            raise TypeError("NodeMask.node_mask must be bool dtype.")
        if self.node_mask.ndim != 1:
            raise ValueError("NodeMask.node_mask must be 1D [N].")

    @property
    def num_nodes(self) -> int:
        return int(self.node_mask.numel())

    @property
    def num_masked(self) -> int:
        return int(self.node_mask.sum().item())


@dataclass(frozen=True)
class MaskApplyResult:
    """
    Output of applying a mask to (X, edge_index).

    Notes:
      - For ZERO_* modes, `node_perm` is None (shape preserved).
      - For DROP_NODES, `node_perm` maps new node indices -> old node indices.
    """
    X: torch.Tensor | None
    edge_index: torch.Tensor | None
    node_mask: torch.Tensor  # bool [N_old] (original node mask)
    edge_mask: torch.Tensor | None  # bool [E_old] when edges dropped
    node_perm: torch.Tensor | None  # long [N_new] when nodes dropped
    meta: Mapping[str, Any]


# -----------------------------------------------------------------------------
# Helpers: edge_index / graph utilities
# -----------------------------------------------------------------------------

def ensure_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Ensure edge_index is a LongTensor of shape [2, E].
    """
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index must be a torch.Tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.dtype != torch.long:
        edge_index = edge_index.to(dtype=torch.long)
    return edge_index


def infer_num_nodes(
    edge_index: torch.Tensor | None = None,
    X: torch.Tensor | None = None,
    num_nodes: int | None = None,
) -> int:
    """
    Infer number of nodes.

    Precedence:
      1) explicit num_nodes
      2) X.shape[0]
      3) max(edge_index) + 1
    """
    if num_nodes is not None:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        return int(num_nodes)
    if X is not None:
        if X.ndim < 1:
            raise ValueError("X must have at least 1 dimension (N, ...).")
        return int(X.shape[0])
    if edge_index is not None:
        edge_index = ensure_edge_index(edge_index)
        if edge_index.numel() == 0:
            raise ValueError("Cannot infer num_nodes from empty edge_index.")
        return int(edge_index.max().item()) + 1
    raise ValueError("infer_num_nodes needs at least one of: num_nodes, X, edge_index.")


def _coerce_cpu_long(t: torch.Tensor) -> torch.Tensor:
    if t.device.type != "cpu":
        t = t.detach().to("cpu")
    if t.dtype != torch.long:
        t = t.to(torch.long)
    return t


def _build_csr_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    undirected: bool = True,
    drop_self_loops: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build CSR adjacency (rowptr, col) for fast neighbor iteration.

    Returns:
      rowptr: long [N+1]
      col:    long [E’] (possibly doubled if undirected)
    """
    ei = ensure_edge_index(edge_index)
    src = _coerce_cpu_long(ei[0])
    dst = _coerce_cpu_long(ei[1])

    if undirected:
        src = torch.cat([src, dst], dim=0)
        dst = torch.cat([dst, src[: dst.numel()]], dim=0)  # careful: uses original src in first half
        # The above line is subtle; easiest is to recompute explicitly:
        # But keep it readable/robust instead of clever.
        # We'll redo properly:
        src0 = _coerce_cpu_long(ei[0])
        dst0 = _coerce_cpu_long(ei[1])
        src = torch.cat([src0, dst0], dim=0)
        dst = torch.cat([dst0, src0], dim=0)

    if drop_self_loops and src.numel() > 0:
        keep = src != dst
        src = src[keep]
        dst = dst[keep]

    if src.numel() == 0:
        rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        col = torch.empty((0,), dtype=torch.long)
        return rowptr, col

    order = torch.argsort(src)
    src = src[order]
    col = dst[order]

    counts = torch.bincount(src, minlength=num_nodes)
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.cumsum(counts, dim=0)
    return rowptr, col


def _bfs_parents(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    start: int,
    *,
    target: int | None = None,
    max_depth: int | None = None,
) -> tuple[list[int], list[int]]:
    """
    BFS returning parent pointers and depth (distance) arrays.

    parent[v] = u predecessor on BFS tree from start, or -1 if unreachable.
    depth[v] = distance from start, or -1 if unreachable.

    If target is provided, stops early when target is discovered.
    """
    N = int(rowptr.numel() - 1)
    if not (0 <= start < N):
        raise ValueError(f"start out of range [0, {N-1}]: {start}")
    if target is not None and not (0 <= target < N):
        raise ValueError(f"target out of range [0, {N-1}]: {target}")

    parent = [-1] * N
    depth = [-1] * N
    q: list[int] = [start]
    depth[start] = 0

    head = 0
    while head < len(q):
        u = q[head]
        head += 1

        du = depth[u]
        if max_depth is not None and du >= max_depth:
            continue

        s = int(rowptr[u].item())
        e = int(rowptr[u + 1].item())
        if s == e:
            continue

        for v in col[s:e].tolist():
            if depth[v] != -1:
                continue
            parent[v] = u
            depth[v] = du + 1
            if target is not None and v == target:
                return parent, depth
            q.append(v)

    return parent, depth


def select_nodes_from_mask(node_mask: torch.Tensor) -> torch.Tensor:
    """
    Return sorted node indices where node_mask is True.
    """
    if node_mask.dtype != torch.bool or node_mask.ndim != 1:
        raise ValueError("node_mask must be bool 1D.")
    return torch.nonzero(node_mask, as_tuple=False).flatten().to(torch.long)


# -----------------------------------------------------------------------------
# Mask combinators
# -----------------------------------------------------------------------------

def _ensure_same_N(a: NodeMask, b: NodeMask) -> None:
    if a.num_nodes != b.num_nodes:
        raise ValueError(f"Mask sizes differ: {a.num_nodes} vs {b.num_nodes}")


def union_masks(a: NodeMask, b: NodeMask, *, name: str = "union") -> NodeMask:
    _ensure_same_N(a, b)
    m = a.node_mask | b.node_mask
    params = {"a": a.name, "b": b.name, "a_params": dict(a.params), "b_params": dict(b.params)}
    return NodeMask(m, name=name, params=params)


def intersect_masks(a: NodeMask, b: NodeMask, *, name: str = "intersection") -> NodeMask:
    _ensure_same_N(a, b)
    m = a.node_mask & b.node_mask
    params = {"a": a.name, "b": b.name, "a_params": dict(a.params), "b_params": dict(b.params)}
    return NodeMask(m, name=name, params=params)


def invert_mask(mask: NodeMask, *, name: str = "invert") -> NodeMask:
    m = ~mask.node_mask
    params = {"base": mask.name, "base_params": dict(mask.params)}
    return NodeMask(m, name=name, params=params)


# -----------------------------------------------------------------------------
# Applying masks (central semantics)
# -----------------------------------------------------------------------------

def _resolve_pe_slice(
    *,
    X: torch.Tensor,
    pe_slice: slice | None = None,
    pe_dim: int | None = None,
) -> slice:
    """
    Resolve a PE slice into X's last dimension.

    - If pe_slice is given, use it.
    - Else if pe_dim is given, assume PE is in the *last* pe_dim feature channels.
    """
    if X.ndim < 2:
        raise ValueError("X must be at least 2D [N, F] to resolve PE slice.")
    F = int(X.shape[1])
    if pe_slice is not None:
        # Basic sanity check (slice may be open-ended, so we just ensure it’s slice)
        return pe_slice
    if pe_dim is None:
        raise ValueError("Must provide pe_slice or pe_dim for PE-only masking.")
    if pe_dim <= 0 or pe_dim > F:
        raise ValueError(f"Invalid pe_dim={pe_dim} for X with F={F}.")
    return slice(F - pe_dim, F)


def apply_mask_to_X(
    X: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    mode: MaskingMode = MaskingMode.ZERO_X,
    pe_slice: slice | None = None,
    pe_dim: int | None = None,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Apply a node mask to node features X.

    Primary:
      - ZERO_X: zero full rows X[i, :] for masked i

    Optional:
      - ZERO_PE: zero only PE channels (caller provides pe_slice or pe_dim)
      - ZERO_X_AND_PE: zero full rows + explicitly zero PE channels too (redundant if
        PE is part of X; useful if PE is computed on the fly and concatenated later)
    """
    if node_mask.dtype != torch.bool or node_mask.ndim != 1:
        raise ValueError("node_mask must be bool 1D [N].")
    if X.ndim < 2:
        raise ValueError("X must be at least [N, F].")
    if X.shape[0] != node_mask.numel():
        raise ValueError(f"X has N={X.shape[0]} but node_mask has N={node_mask.numel()}.")

    out = X if inplace else X.clone()

    if mode == MaskingMode.ZERO_X:
        out[node_mask] = 0
        return out

    if mode == MaskingMode.ZERO_PE:
        sl = _resolve_pe_slice(X=out, pe_slice=pe_slice, pe_dim=pe_dim)
        out[node_mask, sl] = 0
        return out

    if mode == MaskingMode.ZERO_X_AND_PE:
        out[node_mask] = 0
        sl = _resolve_pe_slice(X=out, pe_slice=pe_slice, pe_dim=pe_dim)
        out[node_mask, sl] = 0
        return out

    raise ValueError(f"apply_mask_to_X does not support mode={mode!r} (use apply_mask_to_graph).")


def _drop_edges_incident_to_mask(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Drop edges where either endpoint is masked. Returns (new_edge_index, edge_keep_mask).
    """
    ei = ensure_edge_index(edge_index)
    if node_mask.dtype != torch.bool or node_mask.ndim != 1:
        raise ValueError("node_mask must be bool 1D [N].")

    src = ei[0]
    dst = ei[1]
    if src.numel() == 0:
        keep = torch.ones((0,), dtype=torch.bool, device=src.device)
        return ei, keep

    keep = ~(node_mask[src] | node_mask[dst])
    return ei[:, keep], keep


def _induced_subgraph_keep_nodes(
    edge_index: torch.Tensor,
    keep_nodes: torch.Tensor,  # bool [N_old]
    *,
    relabel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Induce a subgraph on nodes where keep_nodes=True.

    Returns:
      edge_index_new: [2, E_new] (relabelled if relabel=True)
      edge_keep:      bool [E_old]
      node_perm:      long [N_new] mapping new -> old (i.e., old_index = node_perm[new])
    """
    ei = ensure_edge_index(edge_index)
    if keep_nodes.dtype != torch.bool or keep_nodes.ndim != 1:
        raise ValueError("keep_nodes must be bool 1D [N_old].")

    src = ei[0]
    dst = ei[1]
    if src.numel() == 0:
        edge_keep = torch.ones((0,), dtype=torch.bool, device=src.device)
        node_perm = torch.nonzero(keep_nodes, as_tuple=False).flatten().to(torch.long)
        if relabel:
            return ei, edge_keep, node_perm
        return ei, edge_keep, node_perm

    edge_keep = keep_nodes[src] & keep_nodes[dst]
    ei2 = ei[:, edge_keep]

    node_perm = torch.nonzero(keep_nodes, as_tuple=False).flatten().to(torch.long)
    if not relabel:
        return ei2, edge_keep, node_perm

    # relabel old -> new
    N_old = int(keep_nodes.numel())
    old2new = torch.full((N_old,), -1, dtype=torch.long, device=keep_nodes.device)
    old2new[node_perm] = torch.arange(node_perm.numel(), device=keep_nodes.device, dtype=torch.long)

    src2 = old2new[ei2[0]]
    dst2 = old2new[ei2[1]]
    if (src2 < 0).any() or (dst2 < 0).any():
        raise RuntimeError("Induced subgraph relabeling failed (unexpected -1 indices).")

    ei_new = torch.stack([src2, dst2], dim=0)
    return ei_new, edge_keep, node_perm


def apply_mask_to_graph(
    X: torch.Tensor | None,
    edge_index: torch.Tensor | None,
    node_mask: torch.Tensor,
    *,
    mode: MaskingMode = MaskingMode.ZERO_X,
    pe_slice: slice | None = None,
    pe_dim: int | None = None,
    inplace_X: bool = False,
) -> MaskApplyResult:
    """
    Apply node masking semantics to (X, edge_index).

    - If mode is ZERO_*: X is transformed; edge_index passed through unchanged.
    - If mode is DROP_EDGES: edge_index is filtered; X unchanged by default.
    - If mode is DROP_NODES: both X and edge_index are reduced; node_perm provided.
    """
    if node_mask.dtype != torch.bool or node_mask.ndim != 1:
        raise ValueError("node_mask must be bool 1D [N].")

    meta: dict[str, Any] = {"mode": str(mode), "num_masked": int(node_mask.sum().item())}

    if mode in (MaskingMode.ZERO_X, MaskingMode.ZERO_PE, MaskingMode.ZERO_X_AND_PE):
        if X is None:
            raise ValueError(f"mode={mode} requires X.")
        X2 = apply_mask_to_X(
            X,
            node_mask,
            mode=mode,
            pe_slice=pe_slice,
            pe_dim=pe_dim,
            inplace=inplace_X,
        )
        return MaskApplyResult(
            X=X2,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=None,
            node_perm=None,
            meta=meta,
        )

    if mode == MaskingMode.DROP_EDGES:
        if edge_index is None:
            raise ValueError("DROP_EDGES requires edge_index.")
        ei2, keep = _drop_edges_incident_to_mask(edge_index, node_mask)
        meta["num_edges_dropped"] = int((~keep).sum().item())
        return MaskApplyResult(
            X=X,
            edge_index=ei2,
            node_mask=node_mask,
            edge_mask=keep,
            node_perm=None,
            meta=meta,
        )

    if mode == MaskingMode.DROP_NODES:
        if X is None or edge_index is None:
            raise ValueError("DROP_NODES requires both X and edge_index.")
        keep_nodes = ~node_mask
        if int(keep_nodes.sum().item()) <= 0:
            raise ValueError("DROP_NODES would remove all nodes.")
        ei2, edge_keep, node_perm = _induced_subgraph_keep_nodes(edge_index, keep_nodes, relabel=True)
        X2 = X[node_perm].clone() if not inplace_X else X[node_perm]
        meta["num_nodes_new"] = int(node_perm.numel())
        meta["num_edges_dropped"] = int((~edge_keep).sum().item())
        return MaskApplyResult(
            X=X2,
            edge_index=ei2,
            node_mask=node_mask,
            edge_mask=edge_keep,
            node_perm=node_perm,
            meta=meta,
        )

    raise ValueError(f"Unknown masking mode: {mode!r}")


def apply_mask(
    obj: Any,
    mask: NodeMask,
    *,
    x_attr: str = "x",
    edge_attr: str = "edge_index",
    mode: MaskingMode = MaskingMode.ZERO_X,
    pe_slice: slice | None = None,
    pe_dim: int | None = None,
    inplace_X: bool = False,
) -> MaskApplyResult:
    """
    Convenience wrapper: apply a NodeMask to either:
      - raw tensors: pass obj as dict-like {"x": X, "edge_index": edge_index}
      - a PyG-like object with attributes `.x` and `.edge_index`

    Returns MaskApplyResult with updated tensors, leaving `obj` untouched.
    """
    X = None
    edge_index = None

    if isinstance(obj, Mapping):
        X = obj.get(x_attr, None)
        edge_index = obj.get(edge_attr, None)
    else:
        X = getattr(obj, x_attr, None)
        edge_index = getattr(obj, edge_attr, None)

    if X is not None and not isinstance(X, torch.Tensor):
        raise TypeError(f"{x_attr} must be a torch.Tensor if present.")
    if edge_index is not None and not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"{edge_attr} must be a torch.Tensor if present.")

    return apply_mask_to_graph(
        X,
        edge_index,
        mask.node_mask,
        mode=mode,
        pe_slice=pe_slice,
        pe_dim=pe_dim,
        inplace_X=inplace_X,
    )


# -----------------------------------------------------------------------------
# Mask generators
# -----------------------------------------------------------------------------

def mask_shortest_path_nodes(
    edge_index: torch.Tensor,
    source: int,
    target: int,
    *,
    num_nodes: int | None = None,
    undirected: bool = True,
    include_endpoints: bool = True,
    max_depth: int | None = None,
    fallback: Literal["target", "endpoints", "empty"] = "target",
) -> NodeMask:
    """
    Mask nodes on a shortest path from source to target (BFS on unweighted graph).

    If source->target is disconnected:
      - fallback="target": mask only target
      - fallback="endpoints": mask source and target
      - fallback="empty": mask nothing
    """
    ei = ensure_edge_index(edge_index)
    N = infer_num_nodes(edge_index=ei, num_nodes=num_nodes)

    rowptr, col = _build_csr_adjacency(ei, N, undirected=undirected, drop_self_loops=False)
    parent, depth = _bfs_parents(rowptr, col, source, target=target, max_depth=max_depth)

    on_path = torch.zeros((N,), dtype=torch.bool)
    if depth[target] == -1:
        if fallback == "empty":
            pass
        elif fallback == "endpoints":
            on_path[source] = True
            on_path[target] = True
        elif fallback == "target":
            on_path[target] = True
        else:
            raise ValueError(f"Unknown fallback: {fallback!r}")
        return NodeMask(
            on_path,
            name="shortest_path",
            params={
                "source": int(source),
                "target": int(target),
                "undirected": bool(undirected),
                "include_endpoints": bool(include_endpoints),
                "max_depth": None if max_depth is None else int(max_depth),
                "fallback": str(fallback),
                "found": False,
            },
        )

    # Reconstruct path from target back to source
    path: list[int] = [target]
    v = target
    while v != source:
        v = parent[v]
        if v == -1:
            # Should not happen if depth[target] != -1, but stay defensive.
            break
        path.append(v)

    if not include_endpoints:
        # remove source+target if present
        path = [u for u in path if u not in (source, target)]

    if len(path) > 0:
        on_path[torch.tensor(path, dtype=torch.long)] = True

    return NodeMask(
        on_path,
        name="shortest_path",
        params={
            "source": int(source),
            "target": int(target),
            "undirected": bool(undirected),
            "include_endpoints": bool(include_endpoints),
            "max_depth": None if max_depth is None else int(max_depth),
            "fallback": str(fallback),
            "found": True,
            "path_len_nodes": int(sum(1 for _ in path)),
        },
    )


def mask_khop_around_target(
    edge_index: torch.Tensor,
    target: int,
    k: int,
    *,
    num_nodes: int | None = None,
    undirected: bool = True,
    include_target: bool = True,
) -> NodeMask:
    """
    Mask nodes within k hops of target (including target by default).
    """
    if k < 0:
        raise ValueError("k must be >= 0.")
    ei = ensure_edge_index(edge_index)
    N = infer_num_nodes(edge_index=ei, num_nodes=num_nodes)

    rowptr, col = _build_csr_adjacency(ei, N, undirected=undirected, drop_self_loops=False)
    parent, depth = _bfs_parents(rowptr, col, target, target=None, max_depth=k)

    m = torch.zeros((N,), dtype=torch.bool)
    for v, dv in enumerate(depth):
        if dv != -1 and dv <= k:
            m[v] = True

    if not include_target:
        m[target] = False

    return NodeMask(
        m,
        name="khop_target",
        params={
            "target": int(target),
            "k": int(k),
            "undirected": bool(undirected),
            "include_target": bool(include_target),
        },
    )


def _degree_scores(edge_index: torch.Tensor, N: int, *, undirected: bool = True) -> torch.Tensor:
    ei = ensure_edge_index(edge_index)
    src = ei[0]
    dst = ei[1]
    deg = torch.zeros((N,), dtype=torch.float32, device=src.device)
    if src.numel() > 0:
        ones = torch.ones((src.numel(),), device=src.device, dtype=torch.float32)
        deg.index_add_(0, src.to(torch.long), ones)
        if undirected:
            ones2 = torch.ones((dst.numel(),), device=dst.device, dtype=torch.float32)
            deg.index_add_(0, dst.to(torch.long), ones2)
    return deg


def _topk_nodes(scores: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    N = int(scores.numel())
    k = min(k, N)
    # Stable tie-break: (-score, +index)
    idx = torch.arange(N, device=scores.device, dtype=torch.long)
    key = torch.stack([-scores, idx.to(scores.dtype)], dim=0)  # [2, N]
    # Lexsort emulation: sort by first, then second
    # We do a two-pass stable sort: second key then first key.
    order2 = torch.argsort(key[1], stable=True)
    order1 = torch.argsort(key[0][order2], stable=True)
    order = order2[order1]
    return order[:k]


def mask_topk_centrality(
    edge_index: torch.Tensor,
    k: int,
    *,
    num_nodes: int | None = None,
    undirected: bool = True,
    kind: Literal["degree", "pagerank", "betweenness", "closeness"] = "degree",
    approx_betweenness_k: int | None = None,
    seed: int = 0,
) -> NodeMask:
    """
    Mask top-k nodes by a centrality metric.

    - If networkx is available, supports: degree, pagerank, betweenness, closeness.
    - If networkx is unavailable, falls back to degree centrality.

    For large graphs, betweenness can be expensive; you can pass approx_betweenness_k
    to use networkx's sampling (if available).
    """
    if k < 0:
        raise ValueError("k must be >= 0.")
    ei = ensure_edge_index(edge_index)
    N = infer_num_nodes(edge_index=ei, num_nodes=num_nodes)

    scores: torch.Tensor

    if nx is None or kind == "degree":
        scores = _degree_scores(ei, N, undirected=undirected).to(torch.float32)
        used_kind = "degree" if kind != "degree" else kind
        if nx is None and kind != "degree":
            used_kind = "degree_fallback_no_networkx"
    else:
        # Build a networkx graph on CPU
        ei_cpu = _coerce_cpu_long(ei)
        edges = list(zip(ei_cpu[0].tolist(), ei_cpu[1].tolist()))
        G = nx.Graph() if undirected else nx.DiGraph()
        G.add_nodes_from(range(N))
        G.add_edges_from(edges)

        used_kind = kind
        if kind == "degree":
            c = dict(G.degree())
            scores = torch.tensor([float(c[i]) for i in range(N)], dtype=torch.float32)
        elif kind == "pagerank":
            c = nx.pagerank(G)
            scores = torch.tensor([float(c[i]) for i in range(N)], dtype=torch.float32)
        elif kind == "closeness":
            c = nx.closeness_centrality(G)
            scores = torch.tensor([float(c[i]) for i in range(N)], dtype=torch.float32)
        elif kind == "betweenness":
            if approx_betweenness_k is not None and approx_betweenness_k > 0:
                # networkx betweenness_centrality supports "k" samples + seed
                c = nx.betweenness_centrality(G, k=int(approx_betweenness_k), seed=int(seed))
                used_kind = f"betweenness_approx_k={int(approx_betweenness_k)}"
            else:
                c = nx.betweenness_centrality(G)
            scores = torch.tensor([float(c[i]) for i in range(N)], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown centrality kind: {kind!r}")

    top = _topk_nodes(scores, k)
    m = torch.zeros((N,), dtype=torch.bool)
    if top.numel() > 0:
        m[top.to("cpu")] = True  # mask is CPU bool; keep it simple/portable

    return NodeMask(
        m,
        name="topk_centrality",
        params={
            "k": int(k),
            "kind": str(used_kind),
            "requested_kind": str(kind),
            "undirected": bool(undirected),
            "approx_betweenness_k": None if approx_betweenness_k is None else int(approx_betweenness_k),
            "seed": int(seed),
        },
    )


def mask_random_nodes(
    num_nodes: int,
    p: float,
    *,
    seed: int = 0,
    min_masked: int = 0,
    max_masked: int | None = None,
) -> NodeMask:
    """
    Randomly mask nodes i.i.d. with probability p (deterministic with seed).

    Controls:
      - min_masked: enforce at least this many masked nodes (by flipping extra nodes on)
      - max_masked: cap to at most this many masked nodes (by flipping extra nodes off)

    Returns a CPU bool mask for maximum interoperability.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")
    if min_masked < 0:
        raise ValueError("min_masked must be >= 0.")
    if max_masked is not None and max_masked < 0:
        raise ValueError("max_masked must be >= 0 if provided.")
    if max_masked is not None and max_masked < min_masked:
        raise ValueError("max_masked must be >= min_masked.")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    u = torch.rand((num_nodes,), generator=g)
    m = (u < float(p))

    # Enforce min/max
    cnt = int(m.sum().item())
    if cnt < min_masked:
        # flip on additional nodes with largest u (closest to threshold) among currently unmasked
        need = min_masked - cnt
        idx = torch.nonzero(~m, as_tuple=False).flatten()
        # choose smallest u among unmasked to flip on (more “natural”)
        u_un = u[idx]
        order = torch.argsort(u_un, stable=True)
        chosen = idx[order[:need]]
        m[chosen] = True

    if max_masked is not None:
        cnt = int(m.sum().item())
        if cnt > max_masked:
            # flip off nodes with smallest u among masked
            extra = cnt - max_masked
            idx = torch.nonzero(m, as_tuple=False).flatten()
            u_m = u[idx]
            order = torch.argsort(u_m, stable=True)
            chosen = idx[order[:extra]]
            m[chosen] = False

    return NodeMask(
        m.to(torch.bool),
        name="random",
        params={
            "num_nodes": int(num_nodes),
            "p": float(p),
            "seed": int(seed),
            "min_masked": int(min_masked),
            "max_masked": None if max_masked is None else int(max_masked),
            "masked": int(m.sum().item()),
        },
    )


# -----------------------------------------------------------------------------
# Convenience registry / builder
# -----------------------------------------------------------------------------

def build_mask(
    kind: str,
    *,
    edge_index: torch.Tensor | None = None,
    num_nodes: int | None = None,
    X: torch.Tensor | None = None,
    **kwargs: Any,
) -> NodeMask:
    """
    Small registry for string-driven configs.

    Examples:
      build_mask("random", num_nodes=N, p=0.1, seed=0)
      build_mask("khop", edge_index=ei, target=t, k=2, num_nodes=N)
      build_mask("shortest_path", edge_index=ei, source=s, target=t, num_nodes=N)
      build_mask("topk_centrality", edge_index=ei, k=5, kind="pagerank", num_nodes=N)

    Notes:
      - For graph-dependent masks, pass edge_index and (optionally) num_nodes.
      - If num_nodes is not passed, we infer from X or edge_index.
    """
    kind_norm = kind.strip().lower()

    if kind_norm in ("random", "rand"):
        N = infer_num_nodes(edge_index=edge_index, X=X, num_nodes=num_nodes)
        return mask_random_nodes(N, **kwargs)

    if edge_index is None:
        raise ValueError(f"Mask kind {kind!r} requires edge_index (except random).")

    N = infer_num_nodes(edge_index=edge_index, X=X, num_nodes=num_nodes)

    if kind_norm in ("shortest_path", "sp"):
        return mask_shortest_path_nodes(edge_index, num_nodes=N, **kwargs)
    if kind_norm in ("khop", "k_hop", "khop_target"):
        return mask_khop_around_target(edge_index, num_nodes=N, **kwargs)
    if kind_norm in ("topk", "topk_centrality", "centrality"):
        return mask_topk_centrality(edge_index, num_nodes=N, **kwargs)

    raise ValueError(f"Unknown mask kind: {kind!r}")

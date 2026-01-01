# acpl/eval/graph_viz.py
from __future__ import annotations

"""
acpl/eval/graph_viz.py

Research-ready graph + representation visualization utilities for ACPL-qwalk.

Why this exists
---------------
Your eval plots (Pt curves, TV curves, etc.) are necessary but not sufficient to
communicate *how* the GNN policy is learning. In a thesis/paper, you typically
also want:
  - graph structure visualization (with start/target highlights)
  - node embedding visualization:
      * embeddings projected to 2D (PCA) in embedding space
      * embeddings painted onto the *graph layout* (color by PC1/PC2/norm)
      * optional embedding-vector field (quiver) on graph layout
  - temporal learning diagnostics:
      * representation drift across t: ||H_t - H_{t-1}||, var_t(H), etc.
      * coin/angle diagnostics: per-(t,v) θ heatmaps + time curves
  - dataset/graph structural stats that mirror “standard ML figures”:
      * degree histograms
      * Laplacian spectrum (small-k eigenvalues)
      * feature distribution snapshots

This module is written to be:
  - deterministic (seeded layouts, stable PCA)
  - robust (optional networkx/scipy; safe fallbacks)
  - safe on large graphs (caps + subgraph sampling helpers)
  - model-agnostic (works if policy has encode_nodes(); else uses policy.encoder)

Typical usage (from scripts/eval.py or artifacts pipeline)
---------------------------------------------------------
1) Capture internals (z, H, theta) via hooks:
    trace = trace_policy_forward(policy, X, edge_index, T=T)

2) Make figures:
    make_learning_figures(
        outdir=...,
        cond="ckpt_policy__abl_NoPE",
        edge_index=edge_index,
        X=X,
        z=trace.z,
        H=trace.H,
        theta=trace.theta,
        context=GraphContext(start=..., target=..., title=...),
    )

Notes
-----
- Uses matplotlib only (no seaborn).
- Optional networkx improves layout; otherwise a deterministic fallback layout is used.
- Optional scipy improves Laplacian spectrum; otherwise uses dense eig on small graphs only.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from acpl.data.features import FeatureSpec, build_node_features

import logging
import math

import numpy as np
import torch

# Headless-friendly: do not force global backend here (caller may set),
# but ensure we can safely import pyplot when generating figures.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

log = logging.getLogger(__name__)

__all__ = [
    # Context/config
    "GraphContext",
    "GraphVizConfig",
    "GraphVizStyle",
    "TraceResult",
    "PolicyTracer",
    # Core compute
    "ensure_numpy",
    "sanitize",
    "coerce_edge_index",
    "compute_degree",
    "pca_2d",
    "compute_layout_2d",
    "extract_node_embeddings",
    "trace_policy_forward",
    # Plot primitives
    "plot_graph",
    "plot_graph_colored",
    "plot_embedding_pca_scatter",
    "plot_embedding_on_graph",
    "plot_embedding_vector_field_on_graph",
    "plot_degree_hist",
    "plot_laplacian_spectrum",
    "plot_representation_drift",
    "plot_theta_heatmaps",
    "plot_theta_summary_curves",
    # High-level bundles
    "make_learning_figures",
]


# =============================================================================
# Small utilities
# =============================================================================


def ensure_numpy(x: Any) -> np.ndarray:
    """Convert torch / list-like to numpy array (CPU), without copying when possible."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sanitize(s: str) -> str:
    s = (s or "").strip()
    s = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")
    return s or "x"


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(fig: plt.Figure, path: Path, *, dpi: int = 200) -> None:
    _ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


# =============================================================================
# Data models
# =============================================================================


@dataclass(frozen=True)
class GraphContext:
    """Optional semantic annotations for a graph/episode."""
    title: str | None = None
    start: int | None = None
    target: int | None = None
    marked: Sequence[int] = field(default_factory=tuple)
    # If you already have node coordinates (N,2), pass them here
    pos2d: np.ndarray | None = None


@dataclass(frozen=True)
class GraphVizStyle:
    """Styling defaults (paper-friendly)."""
    figsize: tuple[float, float] = (6.4, 4.8)
    dpi: int = 200

    # Node styling
    node_size: float = 30.0
    node_alpha: float = 0.95
    node_edgecolor: str = "none"

    # Edge styling
    edge_alpha: float = 0.20
    edge_lw: float = 0.8

    # Highlight styling
    start_marker: str = "s"
    target_marker: str = "*"
    marked_marker: str = "o"
    highlight_size_mult: float = 2.8
    highlight_edgewidth: float = 1.4


@dataclass(frozen=True)
class GraphVizConfig:
    """
    Deterministic layout + safety knobs.

    layout:
      - "spring"   (networkx if available; else fallback)
      - "spectral" (small graphs; else fallback)
      - "circular"
      - "grid"
      - "random"   (deterministic)
    """
    layout: str = "spring"
    seed: int = 0

    # safety caps
    max_nodes: int = 1500
    max_edges: int = 20000

    # layout-specific caps
    max_nodes_spectral: int = 700
    max_nodes_networkx: int = 2000

    # If graph too large: sample a subgraph for visualization
    sample_if_too_large: bool = True
    sample_radius: int = 2
    sample_size: int = 400  # target nodes in sampled subgraph

    # PCA
    pca_center: bool = True

    # For embedding vector fields
    quiver_max_nodes: int = 600


# =============================================================================
# Graph helpers
# =============================================================================


def coerce_edge_index(edge_index: torch.Tensor | np.ndarray, *, num_nodes: int | None = None) -> np.ndarray:
    """
    Return edge_index as numpy int64 shape (2, E).
    Accepts torch (2,E) or (E,2) or numpy variants.
    """
    ei = ensure_numpy(edge_index)
    if ei.ndim != 2:
        raise ValueError(f"edge_index must be rank-2, got shape {ei.shape}")
    if ei.shape[0] == 2:
        out = ei
    elif ei.shape[1] == 2:
        out = ei.T
    else:
        raise ValueError(f"edge_index must be (2,E) or (E,2), got {ei.shape}")

    out = out.astype(np.int64, copy=False)
    if num_nodes is not None:
        if out.min() < 0 or out.max() >= int(num_nodes):
            raise ValueError("edge_index contains out-of-range node ids.")
    return out


def compute_degree(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Undirected degree (counts both endpoints)."""
    src = edge_index[0]
    dst = edge_index[1]
    deg = np.zeros((num_nodes,), dtype=np.int64)
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    return deg


def _to_undirected_edges(edge_index: np.ndarray) -> np.ndarray:
    """Return undirected edge list (duplicates possible)."""
    src, dst = edge_index
    rev = np.stack([dst, src], axis=0)
    return np.concatenate([edge_index, rev], axis=1)


def _build_adj_lists(edge_index: np.ndarray, num_nodes: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def _bfs_ball(adj: list[list[int]], start: int, radius: int, cap: int) -> list[int]:
    """Deterministic BFS ball (bounded)."""
    seen = set([start])
    frontier = [start]
    for _ in range(max(0, int(radius))):
        nxt: list[int] = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
                    if len(seen) >= cap:
                        return sorted(seen)
        frontier = nxt
        if not frontier:
            break
    return sorted(seen)


def _sample_subgraph_nodes(
    edge_index: np.ndarray,
    num_nodes: int,
    *,
    seed: int,
    context: GraphContext | None,
    radius: int,
    target_size: int,
) -> np.ndarray:
    """
    Choose a visualization subgraph node set.
    Preference order:
      start/target/marked -> BFS balls -> fill with degree-ranked nodes.
    """
    adj = _build_adj_lists(edge_index, num_nodes)

    seeds: list[int] = []
    if context is not None:
        if context.start is not None:
            seeds.append(int(context.start))
        if context.target is not None:
            seeds.append(int(context.target))
        for m in context.marked:
            seeds.append(int(m))
    if not seeds:
        seeds = [int(_rng(seed).integers(0, num_nodes))]

    chosen: set[int] = set()
    for s in seeds:
        if 0 <= s < num_nodes:
            ball = _bfs_ball(adj, s, radius=radius, cap=max(target_size, 10))
            chosen.update(ball)
            if len(chosen) >= target_size:
                break

    if len(chosen) < target_size:
        deg = compute_degree(edge_index, num_nodes)
        order = np.argsort(-deg)  # high degree first
        for u in order.tolist():
            chosen.add(int(u))
            if len(chosen) >= target_size:
                break

    return np.array(sorted(chosen), dtype=np.int64)


def _induce_subgraph(edge_index: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Induce subgraph edge_index over a chosen node set; returns (edge_index_sub, old->new map)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    inv = {int(u): i for i, u in enumerate(nodes.tolist())}
    src, dst = edge_index
    mask = np.isin(src, nodes) & np.isin(dst, nodes)
    src2 = src[mask]
    dst2 = dst[mask]
    # remap
    src3 = np.array([inv[int(u)] for u in src2], dtype=np.int64)
    dst3 = np.array([inv[int(v)] for v in dst2], dtype=np.int64)
    return np.stack([src3, dst3], axis=0), inv


# =============================================================================
# Layout computation
# =============================================================================


def compute_layout_2d(
    edge_index: np.ndarray,
    num_nodes: int,
    *,
    cfg: GraphVizConfig,
    context: GraphContext | None = None,
) -> np.ndarray:
    """
    Compute a deterministic (N,2) layout.

    If cfg.sample_if_too_large and graph is huge, returns a layout for the induced subgraph
    nodes and stores positions only for those nodes in returned array (full N) by filling
    others with NaN. (This makes downstream plotting easier to subselect.)
    """
    if context is not None and context.pos2d is not None:
        pos2d = ensure_numpy(context.pos2d)
        if pos2d.shape != (num_nodes, 2):
            raise ValueError(f"context.pos2d must be (N,2) with N={num_nodes}, got {pos2d.shape}")
        return pos2d.astype(np.float64, copy=False)

    E = int(edge_index.shape[1])
    too_big = (num_nodes > cfg.max_nodes) or (E > cfg.max_edges)
    if too_big and cfg.sample_if_too_large:
        nodes = _sample_subgraph_nodes(
            edge_index, num_nodes, seed=cfg.seed, context=context,
            radius=cfg.sample_radius, target_size=cfg.sample_size
        )
        # Induce subgraph and compute layout there
        ei_sub, inv = _induce_subgraph(edge_index, nodes)
        pos_sub = _compute_layout_core(ei_sub, len(nodes), cfg=cfg)
        # Lift to full-N with NaNs elsewhere
        pos_full = np.full((num_nodes, 2), np.nan, dtype=np.float64)
        for old_u, new_u in inv.items():
            pos_full[old_u] = pos_sub[new_u]
        return pos_full

    return _compute_layout_core(edge_index, num_nodes, cfg=cfg)


def _compute_layout_core(edge_index: np.ndarray, num_nodes: int, *, cfg: GraphVizConfig) -> np.ndarray:
    layout = (cfg.layout or "spring").lower()
    seed = int(cfg.seed)

    if layout == "circular":
        t = np.linspace(0.0, 2.0 * math.pi, num_nodes, endpoint=False)
        return np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float64)

    if layout == "grid":
        side = int(math.ceil(math.sqrt(num_nodes)))
        xs, ys = np.meshgrid(np.arange(side), np.arange(side))
        pts = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)[:num_nodes].astype(np.float64)
        pts -= pts.mean(axis=0, keepdims=True)
        denom = np.max(np.abs(pts)) if np.max(np.abs(pts)) > 0 else 1.0
        return pts / denom

    if layout == "random":
        r = _rng(seed)
        pts = r.standard_normal((num_nodes, 2))
        pts -= pts.mean(axis=0, keepdims=True)
        denom = np.max(np.abs(pts)) if np.max(np.abs(pts)) > 0 else 1.0
        return pts / denom

    # Try networkx spring layout (best visual)
    if layout == "spring":
        try:
            import networkx as nx  # type: ignore
            if num_nodes <= cfg.max_nodes_networkx:
                G = nx.Graph()
                G.add_nodes_from(range(num_nodes))
                src, dst = edge_index
                edges = list(zip(src.tolist(), dst.tolist()))
                G.add_edges_from(edges)
                pos = nx.spring_layout(G, seed=seed, dim=2)
                pts = np.array([pos[i] for i in range(num_nodes)], dtype=np.float64)
                pts -= pts.mean(axis=0, keepdims=True)
                denom = np.max(np.abs(pts)) if np.max(np.abs(pts)) > 0 else 1.0
                return pts / denom
        except Exception:
            pass
        # fallback
        return _compute_layout_core(edge_index, num_nodes, cfg=GraphVizConfig(**{**cfg.__dict__, "layout": "grid"}))

    # Spectral (nice for small graphs)
    if layout == "spectral":
        if num_nodes > cfg.max_nodes_spectral:
            return _compute_layout_core(edge_index, num_nodes, cfg=GraphVizConfig(**{**cfg.__dict__, "layout": "grid"}))
        try:
            # Prefer scipy sparse eigsh if available
            try:
                import scipy.sparse as sp  # type: ignore
                import scipy.sparse.linalg as spla  # type: ignore

                src, dst = edge_index
                w = np.ones((edge_index.shape[1],), dtype=np.float64)
                A = sp.coo_matrix((w, (src, dst)), shape=(num_nodes, num_nodes))
                A = A + A.T
                deg = np.asarray(A.sum(axis=1)).reshape(-1)
                D = sp.diags(deg)
                L = D - A
                # smallest eigenpairs; skip the trivial eigenvector
                k = min(3, num_nodes - 1)
                vals, vecs = spla.eigsh(L, k=k, which="SM")
                vecs = np.real(vecs)
                # choose 2 non-trivial components
                if vecs.shape[1] >= 3:
                    pts = vecs[:, 1:3]
                elif vecs.shape[1] == 2:
                    pts = vecs[:, :2]
                else:
                    pts = np.column_stack([vecs[:, 0], np.zeros((num_nodes,))])
                pts -= pts.mean(axis=0, keepdims=True)
                denom = np.max(np.abs(pts)) if np.max(np.abs(pts)) > 0 else 1.0
                return pts / denom
            except Exception:
                # Dense fallback (very small graphs only)
                src, dst = edge_index
                A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
                A[src, dst] = 1.0
                A[dst, src] = 1.0
                deg = A.sum(axis=1)
                L = np.diag(deg) - A
                vals, vecs = np.linalg.eigh(L)
                # pick 2 smallest non-trivial
                idx = np.argsort(vals)
                vecs = vecs[:, idx]
                if num_nodes >= 3:
                    pts = vecs[:, 1:3]
                else:
                    pts = np.column_stack([vecs[:, 0], np.zeros((num_nodes,))])
                pts -= pts.mean(axis=0, keepdims=True)
                denom = np.max(np.abs(pts)) if np.max(np.abs(pts)) > 0 else 1.0
                return pts / denom
        except Exception:
            return _compute_layout_core(edge_index, num_nodes, cfg=GraphVizConfig(**{**cfg.__dict__, "layout": "grid"}))

    # Unknown -> grid
    return _compute_layout_core(edge_index, num_nodes, cfg=GraphVizConfig(**{**cfg.__dict__, "layout": "grid"}))


# =============================================================================
# PCA + embedding utilities
# =============================================================================


def pca_2d(X: np.ndarray, *, center: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA to 2D via SVD. Returns (coords (N,2), explained_var_ratio (2,)).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"PCA expects (N,D), got {X.shape}")
    N, D = X.shape
    if D == 0:
        return np.zeros((N, 2), dtype=np.float64), np.zeros((2,), dtype=np.float64)
    Y = X - X.mean(axis=0, keepdims=True) if center else X
    # SVD
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    # coordinates in PC space
    coords = U[:, :2] * S[:2]
    # explained variance
    var = (S**2) / max(1, (N - 1))
    total = var.sum() if var.size > 0 else 1.0
    evr = (var[:2] / total) if total > 0 else np.zeros((2,), dtype=np.float64)
    return coords, evr


def extract_node_embeddings(
    policy: Any,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Best-effort: return node embeddings z=(N,D).

    Preference order:
      1) policy.encode_nodes(X, edge_index, edge_weight=...)
      2) policy.encoder(X, edge_index, edge_weight)
    """
    if hasattr(policy, "encode_nodes") and callable(getattr(policy, "encode_nodes")):
        try:
            return policy.encode_nodes(X, edge_index, edge_weight=edge_weight)
        except TypeError:
            return policy.encode_nodes(X, edge_index)
    if hasattr(policy, "encoder"):
        enc = getattr(policy, "encoder")
        if callable(enc):
            try:
                return enc(X, edge_index, edge_weight)
            except TypeError:
                return enc(X, edge_index)
    raise AttributeError("Policy does not expose encode_nodes(...) or encoder(...) to extract node embeddings.")


# =============================================================================
# Hook-based tracing (no model edits required)
# =============================================================================


@dataclass
class TraceResult:
    """
    Captured activations.

    z     : (N, Dz)      encoder output
    H     : (T, N, Dh)   controller hidden
    theta : (T, N, P)    head output (Euler angles etc.)
    y     : policy forward output (often theta)
    """
    z: torch.Tensor | None = None
    H: torch.Tensor | None = None
    theta: torch.Tensor | None = None
    y: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class PolicyTracer:
    """
    Context manager that attaches forward hooks to capture intermediate tensors.

    Defaults assume ACPLPolicy-style attributes: encoder, controller, head.
    You can override by passing module attribute names.
    """

    def __init__(
        self,
        policy: Any,
        *,
        encoder_attr: str = "encoder",
        controller_attr: str = "controller",
        head_attr: str = "head",
        capture_inputs: bool = False,
    ):
        self.policy = policy
        self.encoder_attr = encoder_attr
        self.controller_attr = controller_attr
        self.head_attr = head_attr
        self.capture_inputs = bool(capture_inputs)
        self._hooks: list[Any] = []
        self.trace = TraceResult()

    def __enter__(self) -> "PolicyTracer":
        self._attach()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._detach()

    def _attach(self) -> None:
        self._detach()
        self.trace = TraceResult()

        def _hook_encoder(module, inputs, output):
            self.trace.z = output.detach() if torch.is_tensor(output) else output
            if self.capture_inputs:
                self.trace.extra["encoder_inputs"] = inputs

        def _hook_controller(module, inputs, output):
            # output typically H (T,N,D)
            self.trace.H = output.detach() if torch.is_tensor(output) else output
            if self.capture_inputs:
                self.trace.extra["controller_inputs"] = inputs

        def _hook_head(module, inputs, output):
            self.trace.theta = output.detach() if torch.is_tensor(output) else output
            if self.capture_inputs:
                self.trace.extra["head_inputs"] = inputs

        # Best-effort registration
        for attr, hook in [
            (self.encoder_attr, _hook_encoder),
            (self.controller_attr, _hook_controller),
            (self.head_attr, _hook_head),
        ]:
            if hasattr(self.policy, attr):
                mod = getattr(self.policy, attr)
                if hasattr(mod, "register_forward_hook"):
                    try:
                        h = mod.register_forward_hook(hook)
                        self._hooks.append(h)
                    except Exception as e:
                        log.warning("Could not attach hook to %s: %s", attr, e)

        # Some controllers store attention maps; try to harvest later if present.

    def _detach(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def run(self, *args, **kwargs) -> TraceResult:
        y = self.policy(*args, **kwargs)
        self.trace.y = y
        # controller attention extras (best-effort)
        ctrl = getattr(self.policy, self.controller_attr, None)
        if ctrl is not None:
            for name in ["_last_attn", "_attn", "last_attn", "attn_weights"]:
                if hasattr(ctrl, name):
                    self.trace.extra["attn"] = getattr(ctrl, name)
                    break
        return self.trace


def trace_policy_forward(
    policy: Any,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    T: int | None = None,
    t_coords: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
    **forward_kwargs: Any,
) -> TraceResult:
    """
    Convenience wrapper: trace policy forward pass and capture z/H/theta if possible.
    """
    with PolicyTracer(policy) as tr:
        kwargs = dict(forward_kwargs)
        if T is not None:
            kwargs["T"] = int(T)
        if t_coords is not None:
            kwargs["t_coords"] = t_coords
        if edge_weight is not None:
            kwargs["edge_weight"] = edge_weight
        tr.run(X, edge_index, **kwargs)
        return tr.trace


# =============================================================================
# Plotting primitives
# =============================================================================


def _subselect_visible(
    pos2d: np.ndarray,
    edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If pos2d has NaNs for non-visualized nodes, subselect visible nodes and edges.
    Returns (pos_vis, edge_vis, idx_vis_old).
    """
    vis = np.isfinite(pos2d).all(axis=1)
    idx = np.where(vis)[0].astype(np.int64)
    inv = {int(u): i for i, u in enumerate(idx.tolist())}

    src, dst = edge_index
    mask = np.isin(src, idx) & np.isin(dst, idx)
    src2 = np.array([inv[int(u)] for u in src[mask]], dtype=np.int64)
    dst2 = np.array([inv[int(v)] for v in dst[mask]], dtype=np.int64)

    return pos2d[vis], np.stack([src2, dst2], axis=0), idx


def plot_graph(
    *,
    edge_index: np.ndarray,
    pos2d: np.ndarray,
    style: GraphVizStyle = GraphVizStyle(),
    context: GraphContext | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot plain graph structure (edges + nodes) with optional highlights.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    pos_vis, ei_vis, idx_old = _subselect_visible(pos2d, edge_index)

    # edges
    src, dst = ei_vis
    segs = np.stack([pos_vis[src], pos_vis[dst]], axis=1)
    lc = LineCollection(segs, linewidths=style.edge_lw, alpha=style.edge_alpha)
    ax.add_collection(lc)

    # nodes
    ax.scatter(
        pos_vis[:, 0],
        pos_vis[:, 1],
        s=style.node_size,
        alpha=style.node_alpha,
        edgecolors=style.node_edgecolor,
    )

    # highlights
    if context is not None:
        def _maybe_plot(u: int | None, marker: str, label: str):
            if u is None:
                return
            u = int(u)
            if u in set(idx_old.tolist()):
                i = int(np.where(idx_old == u)[0][0])
                ax.scatter(
                    [pos_vis[i, 0]],
                    [pos_vis[i, 1]],
                    s=style.node_size * style.highlight_size_mult,
                    marker=marker,
                    linewidths=style.highlight_edgewidth,
                    edgecolors="k",
                    facecolors="none",
                    label=label,
                    zorder=5,
                )

        _maybe_plot(context.start, style.start_marker, "start")
        _maybe_plot(context.target, style.target_marker, "target")

        for m in context.marked:
            _maybe_plot(int(m), style.marked_marker, "marked")

        if context.title:
            ax.set_title(context.title)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    # Only show legend if any labels were used
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", frameon=True)
    return fig, ax


def plot_graph_colored(
    *,
    edge_index: np.ndarray,
    pos2d: np.ndarray,
    node_values: np.ndarray,
    cmap: str = "viridis",
    style: GraphVizStyle = GraphVizStyle(),
    context: GraphContext | None = None,
    colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """
    Plot graph with nodes colored by scalar values.
    """
    node_values = np.asarray(node_values, dtype=np.float64)
    if node_values.ndim != 1:
        raise ValueError("node_values must be (N,)")

    pos_vis, ei_vis, idx_old = _subselect_visible(pos2d, edge_index)
    vals_vis = node_values[idx_old]

    fig, ax = plt.subplots(figsize=style.figsize)

    # edges
    src, dst = ei_vis
    segs = np.stack([pos_vis[src], pos_vis[dst]], axis=1)
    lc = LineCollection(segs, linewidths=style.edge_lw, alpha=style.edge_alpha)
    ax.add_collection(lc)

    sc = ax.scatter(
        pos_vis[:, 0],
        pos_vis[:, 1],
        s=style.node_size,
        c=vals_vis,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=style.node_alpha,
        edgecolors=style.node_edgecolor,
    )

    if colorbar:
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # highlights
    if context is not None:
        # reuse highlight helper
        _ = plot_graph(edge_index=edge_index, pos2d=pos2d, style=style, context=context, ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    return fig


def plot_embedding_pca_scatter(
    *,
    z: np.ndarray,
    style: GraphVizStyle = GraphVizStyle(),
    title: str | None = None,
    color: np.ndarray | None = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """
    PCA scatter in embedding space (no edges).
    """
    coords, evr = pca_2d(z, center=True)
    fig, ax = plt.subplots(figsize=style.figsize)
    if color is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=style.node_size, alpha=style.node_alpha)
    else:
        c = np.asarray(color, dtype=np.float64).reshape(-1)
        sc = ax.scatter(coords[:, 0], coords[:, 1], s=style.node_size, c=c, cmap=cmap, alpha=style.node_alpha)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ttl = title or f"PCA2 embeddings (EVR={evr[0]:.2f},{evr[1]:.2f})"
    ax.set_title(ttl)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    return fig


def plot_embedding_on_graph(
    *,
    edge_index: np.ndarray,
    pos2d: np.ndarray,
    z: np.ndarray,
    which: str = "pc1",
    style: GraphVizStyle = GraphVizStyle(),
    context: GraphContext | None = None,
) -> plt.Figure:
    """
    Color nodes on the graph by a simple embedding statistic:
      - pc1 / pc2 (from PCA)
      - norm (L2 norm of embedding)
    """
    z = np.asarray(z, dtype=np.float64)
    which = which.lower()

    if which in ("pc1", "pc2"):
        coords, _ = pca_2d(z, center=True)
        vals = coords[:, 0] if which == "pc1" else coords[:, 1]
        title = f"Graph colored by embedding {which.upper()}"
    elif which == "norm":
        vals = np.linalg.norm(z, axis=1)
        title = "Graph colored by embedding ||z||"
    else:
        raise ValueError("which must be one of: pc1, pc2, norm")

    ctx = context
    if ctx is not None and ctx.title is None:
        ctx = GraphContext(title=title, start=ctx.start, target=ctx.target, marked=ctx.marked, pos2d=ctx.pos2d)
    elif ctx is None:
        ctx = GraphContext(title=title)

    return plot_graph_colored(edge_index=edge_index, pos2d=pos2d, node_values=vals, style=style, context=ctx)


def plot_embedding_vector_field_on_graph(
    *,
    edge_index: np.ndarray,
    pos2d: np.ndarray,
    z: np.ndarray,
    cfg: GraphVizConfig = GraphVizConfig(),
    style: GraphVizStyle = GraphVizStyle(),
    context: GraphContext | None = None,
) -> plt.Figure:
    """
    Novelty figure: show embedding directions as a quiver field on the graph layout.
    We reduce z to 2D via PCA, then plot arrows at node positions.

    Safety: if too many nodes, subsamples deterministically.
    """
    z = np.asarray(z, dtype=np.float64)
    coords, evr = pca_2d(z, center=True)  # (N,2)

    # visible nodes only
    vis = np.isfinite(pos2d).all(axis=1)
    idx = np.where(vis)[0].astype(np.int64)

    if idx.size > cfg.quiver_max_nodes:
        r = _rng(cfg.seed)
        idx = r.choice(idx, size=cfg.quiver_max_nodes, replace=False)
        idx = np.sort(idx)

    pos = pos2d[idx]
    vec = coords[idx]

    fig, ax = plt.subplots(figsize=style.figsize)

    # base graph
    _ = plot_graph(edge_index=edge_index, pos2d=pos2d, style=style, context=context, ax=ax)

    # arrows (normalize for readability)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    u = vec / norms
    ax.quiver(
        pos[:, 0],
        pos[:, 1],
        u[:, 0],
        u[:, 1],
        angles="xy",
        scale_units="xy",
        scale=12.0,
        width=0.003,
        alpha=0.85,
    )
    ttl = context.title if (context and context.title) else f"Embedding vector field on graph (EVR={evr[0]:.2f},{evr[1]:.2f})"
    ax.set_title(ttl)
    return fig


def plot_degree_hist(
    *,
    edge_index: np.ndarray,
    num_nodes: int,
    style: GraphVizStyle = GraphVizStyle(),
    title: str | None = None,
) -> plt.Figure:
    deg = compute_degree(edge_index, num_nodes)
    fig, ax = plt.subplots(figsize=style.figsize)
    ax.hist(deg, bins=min(40, max(10, int(np.sqrt(num_nodes)))), alpha=0.9)
    ax.set_xlabel("degree")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.set_title(title or "Degree histogram")
    return fig


def plot_laplacian_spectrum(
    *,
    edge_index: np.ndarray,
    num_nodes: int,
    k: int = 25,
    style: GraphVizStyle = GraphVizStyle(),
    title: str | None = None,
) -> plt.Figure:
    """
    Plot smallest-k Laplacian eigenvalues (structural signature).
    Uses scipy sparse eigsh if available; otherwise dense on small graphs only.
    """
    k = int(k)
    k = max(2, min(k, num_nodes - 1)) if num_nodes > 2 else 2

    evals: np.ndarray
    try:
        import scipy.sparse as sp  # type: ignore
        import scipy.sparse.linalg as spla  # type: ignore

        src, dst = edge_index
        w = np.ones((edge_index.shape[1],), dtype=np.float64)
        A = sp.coo_matrix((w, (src, dst)), shape=(num_nodes, num_nodes))
        A = A + A.T
        deg = np.asarray(A.sum(axis=1)).reshape(-1)
        L = sp.diags(deg) - A
        evals = np.sort(np.real(spla.eigsh(L, k=k, which="SM", return_eigenvectors=False)))
    except Exception:
        # Dense fallback for small graphs only
        if num_nodes > 700:
            evals = np.full((k,), np.nan, dtype=np.float64)
        else:
            src, dst = edge_index
            A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
            A[src, dst] = 1.0
            A[dst, src] = 1.0
            deg = A.sum(axis=1)
            L = np.diag(deg) - A
            evals = np.sort(np.real(np.linalg.eigvalsh(L)))[:k]

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.plot(np.arange(len(evals)), evals, marker="o")
    ax.set_xlabel("index (smallest k)")
    ax.set_ylabel("Laplacian eigenvalue")
    ax.grid(True, alpha=0.25)
    ax.set_title(title or f"Laplacian spectrum (k={k})")
    return fig


def plot_representation_drift(
    *,
    H: np.ndarray,
    style: GraphVizStyle = GraphVizStyle(),
    title: str | None = None,
) -> plt.Figure:
    """
    Show how the controller representation changes over time.

    H: (T,N,D) controller hidden states
    Outputs:
      - mean ||H_t|| over nodes
      - var(H_t) (per-dim variance averaged)
      - drift ||H_t - H_{t-1}|| over nodes
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 3:
        raise ValueError(f"H must be (T,N,D), got {H.shape}")
    T = H.shape[0]

    # node-mean norms
    mean_norm = np.mean(np.linalg.norm(H, axis=2), axis=1)  # (T,)
    # per-dim variance over nodes, averaged over dims
    var_t = np.var(H, axis=1).mean(axis=1)  # (T,)
    # drift
    dH = H[1:] - H[:-1]
    drift = np.mean(np.linalg.norm(dH, axis=2), axis=1)  # (T-1,)

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.plot(np.arange(T), mean_norm, label="mean ||H_t||")
    ax.plot(np.arange(T), var_t, label="mean var(H_t)")
    ax.plot(np.arange(1, T), drift, label="mean ||H_t - H_{t-1}||")
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    ax.set_title(title or "Representation drift over time")
    return fig


def plot_theta_heatmaps(
    *,
    theta: np.ndarray,
    style: GraphVizStyle = GraphVizStyle(),
    title_prefix: str | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """
    Heatmaps of theta components over (t,v). For SU(2) Euler angles, theta is (T,N,3).
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 3:
        raise ValueError(f"theta must be (T,N,P), got {theta.shape}")
    T, N, P = theta.shape
    P_show = min(P, 3)

    fig, axes = plt.subplots(1, P_show, figsize=(style.figsize[0] * P_show, style.figsize[1]), constrained_layout=True)
    if P_show == 1:
        axes = [axes]

    names = ["theta0", "theta1", "theta2"] if P >= 3 else [f"theta{i}" for i in range(P_show)]
    for i in range(P_show):
        ax = axes[i]
        im = ax.imshow(theta[:, :, i], aspect="auto", interpolation="nearest", vmax=vmax)
        ax.set_title(f"{(title_prefix + ' — ') if title_prefix else ''}{names[i]} (T×N)")
        ax.set_xlabel("v")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig


def plot_theta_summary_curves(
    *,
    theta: np.ndarray,
    style: GraphVizStyle = GraphVizStyle(),
    title: str | None = None,
) -> plt.Figure:
    """
    Summaries: per-time mean absolute theta components, and dispersion over nodes.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 3:
        raise ValueError(f"theta must be (T,N,P), got {theta.shape}")
    T, N, P = theta.shape
    mean_abs = np.mean(np.abs(theta), axis=1)  # (T,P)
    std = np.std(theta, axis=1)               # (T,P)

    fig, ax = plt.subplots(figsize=style.figsize)
    for i in range(min(P, 6)):
        ax.plot(np.arange(T), mean_abs[:, i], label=f"mean |θ[{i}]|")
        ax.plot(np.arange(T), std[:, i], linestyle="--", label=f"std θ[{i}]")
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True, ncol=2)
    ax.set_title(title or "Coin/θ summary curves")
    return fig


# =============================================================================
# High-level: make the missing “how learning happens” figures
# =============================================================================


def make_learning_figures(
    *,
    outdir: str | Path,
    cond: str,
    edge_index: torch.Tensor | np.ndarray,
    X: torch.Tensor | np.ndarray | None = None,
    z: torch.Tensor | np.ndarray | None = None,
    H: torch.Tensor | np.ndarray | None = None,
    theta: torch.Tensor | np.ndarray | None = None,
    context: GraphContext | None = None,
    cfg: GraphVizConfig = GraphVizConfig(),
    style: GraphVizStyle = GraphVizStyle(),
) -> dict[str, str]:
    """
    Produce a research-ready set of figures that complement Pt/TV plots by showing:
      - graph structure
      - embeddings on graph (PC1/PC2/norm)
      - PCA scatter in embedding space
      - embedding vector field on graph
      - degree histogram + Laplacian spectrum
      - representation drift (if H)
      - theta heatmaps + theta curves (if theta)

    Returns: dict[name -> filepath]
    """
    outdir = _as_path(outdir)
    _ensure_dir(outdir)
    figdir = outdir / "figs"
    _ensure_dir(figdir)

    cond_safe = sanitize(cond)

    ei = coerce_edge_index(edge_index)
    # infer N
    if X is not None:
        Xn = ensure_numpy(X)
        if Xn.ndim != 2:
            raise ValueError(f"X must be (N,F), got {Xn.shape}")
        N = int(Xn.shape[0])
    elif z is not None:
        zn = ensure_numpy(z)
        if zn.ndim != 2:
            raise ValueError(f"z must be (N,D), got {zn.shape}")
        N = int(zn.shape[0])
    else:
        N = int(ei.max() + 1)

    # layout
    pos2d = compute_layout_2d(ei, N, cfg=cfg, context=context)

    paths: dict[str, str] = {}

    # --- Graph structure ---
    fig, _ = plot_graph(edge_index=ei, pos2d=pos2d, style=style, context=context)
    p = figdir / f"graph__{cond_safe}.png"
    _savefig(fig, p, dpi=style.dpi)
    paths["graph"] = str(p)

    # --- Structural stats ---
    fig = plot_degree_hist(edge_index=ei, num_nodes=N, style=style, title="Degree histogram")
    p = figdir / f"degree_hist__{cond_safe}.png"
    _savefig(fig, p, dpi=style.dpi)
    paths["degree_hist"] = str(p)

    fig = plot_laplacian_spectrum(edge_index=ei, num_nodes=N, k=25, style=style, title="Laplacian spectrum")
    p = figdir / f"lap_spectrum__{cond_safe}.png"
    _savefig(fig, p, dpi=style.dpi)
    paths["lap_spectrum"] = str(p)

    # --- Embedding figures (if z) ---
    if z is not None:
        zn = ensure_numpy(z)

        fig = plot_embedding_pca_scatter(z=zn, style=style, title=f"PCA2 embeddings — {cond_safe}")
        p = figdir / f"emb_pca2__{cond_safe}.png"
        _savefig(fig, p, dpi=style.dpi)
        paths["emb_pca2"] = str(p)

        fig = plot_embedding_on_graph(edge_index=ei, pos2d=pos2d, z=zn, which="pc1", style=style, context=context)
        p = figdir / f"emb_on_graph_pc1__{cond_safe}.png"
        _savefig(fig, p, dpi=style.dpi)
        paths["emb_on_graph_pc1"] = str(p)

        fig = plot_embedding_on_graph(edge_index=ei, pos2d=pos2d, z=zn, which="pc2", style=style, context=context)
        p = figdir / f"emb_on_graph_pc2__{cond_safe}.png"
        _savefig(fig, p, dpi=style.dpi)
        paths["emb_on_graph_pc2"] = str(p)

        fig = plot_embedding_on_graph(edge_index=ei, pos2d=pos2d, z=zn, which="norm", style=style, context=context)
        p = figdir / f"emb_on_graph_norm__{cond_safe}.png"
        _savefig(fig, p, dpi=style.dpi)
        paths["emb_on_graph_norm"] = str(p)

        fig = plot_embedding_vector_field_on_graph(edge_index=ei, pos2d=pos2d, z=zn, cfg=cfg, style=style, context=context)
        p = figdir / f"emb_vecfield__{cond_safe}.png"
        _savefig(fig, p, dpi=style.dpi)
        paths["emb_vecfield"] = str(p)

    # --- Temporal representation drift (if H) ---
    if H is not None:
        Hn = ensure_numpy(H)
        fig = plot_representation_drift(H=Hn, style=style, title=f"Representation drift — {cond_safe}")
        p = figdir / f"repr_drift__{cond_safe}.png"
        _savefig(fig, p, dpi=style.dpi)
        paths["repr_drift"] = str(p)

    # --- Theta diagnostics (if theta) ---
    if theta is not None:
        th = ensure_numpy(theta)
        fig = plot_theta_heatmaps(theta=th, style=style, title_prefix=f"{cond_safe}")
        p = figdir / f"theta_heatmaps__{cond_safe}.png"
        _savefig(fig, p, dpi=style.dpi)
        paths["theta_heatmaps"] = str(p)

        fig = plot_theta_summary_curves(theta=th, style=style, title=f"Theta summary — {cond_safe}")
        p = figdir / f"theta_curves__{cond_safe}.png"
        _savefig(fig, p, dpi=style.dpi)
        paths["theta_curves"] = str(p)

    return paths

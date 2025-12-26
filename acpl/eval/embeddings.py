# acpl/eval/embeddings.py
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence
from acpl.eval.stats import explained_variance
import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "EmbeddingEvalConfig",
    "EmbeddingArtifacts",
    "extract_node_embeddings",
    "aggregate_time_embeddings",
    "compute_embedding_statistics",
    "pca_2d",
    "plot_embeddings_2d",
    "save_embeddings_artifacts",
    "resolve_module_by_path",
]


log = logging.getLogger(__name__)


# =============================================================================
# Config + outputs
# =============================================================================


@dataclass(frozen=True)
class EmbeddingEvalConfig:
    """
    Configuration for extracting and saving node embeddings during evaluation.

    This module is designed to be robust to differences in model internals.
    Preferred behavior:
      * model.encode_nodes(X, edge_index, T=...) -> (N, D) or (T, N, D)
    Fallback behavior:
      * forward-hook a named submodule path and capture its output
        (e.g., "gnn", "encoder.gnn", "backbone", etc.)

    Notes
    -----
    - If embeddings are time-dependent (T, N, D), you can choose how to aggregate
      for plotting and summary stats via `time_aggregate`.
    - We save BOTH raw embeddings (torch .pt) and a 2D PCA projection (png + csv).
    """

    # Extraction
    method: Literal["auto", "encode_nodes", "hook"] = "auto"
    hook_module_path: str | None = None  # required if method="hook" (or auto falls back)
    hook_output_index: int | None = None  # if hooked module returns tuple/list, choose element

    # Time handling (if embeddings are (T, N, D))
    time_aggregate: Literal["last", "mean", "max", "stack"] = "last"
    time_index: int = -1  # used for time_aggregate="last" (or direct selection)

    # Saving
    prefix: str = "node_embeddings"
    save_pt: bool = True
    save_csv: bool = True
    save_npz: bool = False  # optional for numpy workflows
    save_metadata_json: bool = True

    # PCA / plots
    make_pca_plot: bool = True
    pca_center: bool = True
    pca_whiten: bool = False
    plot_annotate_nodes: bool = False
    plot_point_size: float = 18.0
    plot_alpha: float = 0.85

    # Coloring (optional)
    # If provided, must be length N and numeric; used to color points in scatter.
    color_values: Sequence[float] | None = None
    color_label: str | None = None

    # Numerical safety / dtype
    detach: bool = True
    force_cpu: bool = True  # recommended for saving/plotting deterministically
    allow_complex: bool = False  # if False and embeddings are complex -> take real

    # Limits (for huge graphs)
    max_nodes_for_plot: int | None = None  # if set, downsample nodes for plotting only
    downsample_seed: int = 0


@dataclass
class EmbeddingArtifacts:
    """
    Paths to produced artifacts (if created).
    """

    embeddings_pt: Path | None = None
    embeddings_csv: Path | None = None
    embeddings_npz: Path | None = None
    metadata_json: Path | None = None
    pca_csv: Path | None = None
    pca_png: Path | None = None


# =============================================================================
# Utilities
# =============================================================================


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_numeric_sequence(x: Sequence[Any]) -> bool:
    try:
        _ = float(x[0])  # type: ignore[index]
        return True
    except Exception:
        return False


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    if x.is_cuda:
        x = x.detach().cpu()
    else:
        x = x.detach()
    return x.numpy()


def _maybe_real(x: torch.Tensor, *, allow_complex: bool) -> torch.Tensor:
    if x.is_complex():
        if allow_complex:
            # If you truly want complex embeddings, you can decide externally how to handle them.
            # For now, we keep them as complex and let downstream fail loudly if it cannot.
            return x
        return x.real
    return x


def _maybe_squeeze_batch(x: torch.Tensor) -> torch.Tensor:
    # Common patterns: (1, N, D) or (1, T, N, D)
    if x.ndim >= 3 and x.size(0) == 1:
        return x.squeeze(0)
    return x


def _stable_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _downsample_indices(N: int, k: int, seed: int) -> np.ndarray:
    rng = _stable_rng(seed)
    if k >= N:
        return np.arange(N, dtype=np.int64)
    return rng.choice(N, size=k, replace=False).astype(np.int64)


# =============================================================================
# Hook resolution
# =============================================================================


def resolve_module_by_path(model: nn.Module, module_path: str) -> nn.Module:
    """
    Resolve a nested submodule from a dot-separated path.

    Supports:
      - attributes (model.gnn, model.encoder.gnn)
      - ModuleDict keys (model.blocks["0"] via path "blocks.0")
      - numeric indexing for ModuleList/Sequential ("layers.2")

    Examples
    --------
    resolve_module_by_path(model, "gnn")
    resolve_module_by_path(model, "encoder.gnn")
    resolve_module_by_path(model, "gnn.layers.2")
    resolve_module_by_path(model, "blocks.0")
    """
    cur: Any = model
    for part in module_path.split("."):
        if isinstance(cur, (nn.ModuleList, nn.Sequential)):
            # allow "2"
            try:
                idx = int(part)
            except Exception as e:
                raise KeyError(
                    f"Expected integer index into {type(cur).__name__}, got '{part}'."
                ) from e
            cur = cur[idx]
            continue

        if isinstance(cur, nn.ModuleDict):
            if part not in cur:
                raise KeyError(f"Key '{part}' not found in ModuleDict at '{module_path}'.")
            cur = cur[part]
            continue

        # nn.Module or arbitrary object with attributes
        if hasattr(cur, part):
            cur = getattr(cur, part)
            continue

        # as a fallback, try treating part as a key (rare)
        try:
            cur = cur[part]  # type: ignore[index]
            continue
        except Exception as e:
            raise KeyError(
                f"Could not resolve submodule path '{module_path}' at segment '{part}'."
            ) from e

    if not isinstance(cur, nn.Module):
        raise TypeError(
            f"Resolved object at '{module_path}' is not an nn.Module (got {type(cur)})."
        )
    return cur


class _HookCapture:
    def __init__(self, output_index: int | None = None):
        self.output_index = output_index
        self.value: torch.Tensor | None = None

    def __call__(self, _mod: nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
        # Many modules return Tensor; some return tuple/list
        if isinstance(out, torch.Tensor):
            self.value = out
            return
        if isinstance(out, (tuple, list)):
            if self.output_index is None:
                # Heuristic: first tensor-like
                for item in out:
                    if isinstance(item, torch.Tensor):
                        self.value = item
                        return
                self.value = None
                return
            idx = int(self.output_index)
            if idx < 0:
                idx = len(out) + idx
            if not (0 <= idx < len(out)):
                self.value = None
                return
            item = out[idx]
            self.value = item if isinstance(item, torch.Tensor) else None
            return
        # Unknown output type
        self.value = None


# =============================================================================
# Extraction
# =============================================================================


@torch.no_grad()
def extract_node_embeddings(
    model: nn.Module,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    T: int | None = None,
    cfg: EmbeddingEvalConfig | None = None,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """
    Extract per-node embeddings from a model.

    Parameters
    ----------
    model:
        The ACPL policy model (or compatible nn.Module).
    X:
        Node features, shape (N, F).
    edge_index:
        Graph edges, shape (2, E) (PyG convention).
    T:
        Optional time horizon. If your model encodes time, pass it through.
    cfg:
        Extraction configuration.
    forward_kwargs:
        Extra kwargs to pass into model forward / encode_nodes.

    Returns
    -------
    emb:
        Either (N, D) or (T, N, D) depending on the model.
    """
    if cfg is None:
        cfg = EmbeddingEvalConfig()

    fwkw = dict(forward_kwargs or {})

    # Ensure tensors on same device as model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = X.device

    X0 = X.to(device=device)

    ei0 = edge_index.to(device=device)
    if ei0.dtype != torch.long:
        ei0 = ei0.to(dtype=torch.long)

    method = cfg.method

    # ---------------------------------------------------------------------
    # Preferred: model.encode_nodes(...)
    # ---------------------------------------------------------------------
    if method in ("auto", "encode_nodes"):
        fn = getattr(model, "encode_nodes", None)
        if callable(fn):
            if T is not None:
                fwkw.setdefault("T", int(T))
            emb = fn(X0, ei0, **fwkw)  # type: ignore[misc]
            if not isinstance(emb, torch.Tensor):
                raise TypeError(
                    "model.encode_nodes(...) must return a torch.Tensor. "
                    f"Got {type(emb)}."
                )
            emb = _maybe_squeeze_batch(emb)
            emb = _maybe_real(emb, allow_complex=cfg.allow_complex)
            if cfg.detach:
                emb = emb.detach()
            if cfg.force_cpu:
                emb = emb.cpu()
            return emb

        if method == "encode_nodes":
            raise AttributeError(
                "cfg.method='encode_nodes' but model.encode_nodes is not defined. "
                "Implement encode_nodes(...) in acpl/policy/policy.py or use method='hook'."
            )

    # ---------------------------------------------------------------------
    # Fallback: hook a module path and run a forward
    # ---------------------------------------------------------------------
    if method in ("auto", "hook"):
        hook_path = cfg.hook_module_path
        if hook_path is None:
            # Try some common names as a best-effort fallback
            candidates = [
                "gnn",
                "encoder",
                "backbone",
                "model.gnn",
                "policy.gnn",
                "net.gnn",
            ]
            resolved: str | None = None
            for c in candidates:
                try:
                    _ = resolve_module_by_path(model, c)
                    resolved = c
                    break
                except Exception:
                    continue
            hook_path = resolved

        if hook_path is None:
            raise ValueError(
                "Could not infer a hook module path. "
                "Set cfg.hook_module_path (e.g., 'gnn' or 'encoder.gnn')."
            )

        submod = resolve_module_by_path(model, hook_path)
        cap = _HookCapture(output_index=cfg.hook_output_index)
        handle = submod.register_forward_hook(cap)

        try:
            # Run forward. We intentionally do not assume the model signature beyond (X, edge_index, T=?).
            if T is not None:
                fwkw.setdefault("T", int(T))
            _ = model(X0, ei0, **fwkw)  # type: ignore[misc]
        finally:
            handle.remove()

        if cap.value is None or not isinstance(cap.value, torch.Tensor):
            raise RuntimeError(
                f"Hook at '{hook_path}' did not capture a tensor output. "
                "If that module returns a tuple/list, set cfg.hook_output_index. "
                "Otherwise choose a different hook_module_path."
            )

        emb = _maybe_squeeze_batch(cap.value)
        emb = _maybe_real(emb, allow_complex=cfg.allow_complex)
        if cfg.detach:
            emb = emb.detach()
        if cfg.force_cpu:
            emb = emb.cpu()
        return emb

    raise ValueError(f"Unknown cfg.method='{cfg.method}'")


# =============================================================================
# Time aggregation
# =============================================================================


def aggregate_time_embeddings(
    emb: torch.Tensor,
    *,
    mode: Literal["last", "mean", "max", "stack"] = "last",
    time_index: int = -1,
) -> torch.Tensor:
    """
    Aggregate time-dependent embeddings.

    If emb is (T, N, D):
      - last: returns (N, D) at time_index
      - mean: returns (N, D)
      - max:  returns (N, D) (elementwise max over T)
      - stack: returns (T*N, D) (useful for global PCA across all time steps)

    If emb is already (N, D), it is returned unchanged.
    """
    if emb.ndim == 2:
        return emb
    if emb.ndim != 3:
        raise ValueError(f"Expected emb shape (N,D) or (T,N,D). Got {tuple(emb.shape)}")

    T, N, D = emb.shape
    if mode == "last":
        idx = int(time_index)
        if idx < 0:
            idx = T + idx
        idx = max(0, min(T - 1, idx))
        return emb[idx]
    if mode == "mean":
        return emb.mean(dim=0)
    if mode == "max":
        return emb.max(dim=0).values
    if mode == "stack":
        return emb.reshape(T * N, D)
    raise ValueError(f"Unknown aggregation mode '{mode}'")


# =============================================================================
# Statistics
# =============================================================================


def compute_embedding_statistics(emb: torch.Tensor) -> dict[str, Any]:
    """
    Compute “overall statistics” and per-dimension stats for embeddings.

    Works on (N, D) embeddings. If you have (T, N, D), aggregate first.

    Returns a JSON-serializable dict with:
      - shape
      - global mean/var/std, L2 norms
      - per-dimension mean/var/std/min/max
      - quantiles of norms
    """
    if emb.ndim != 2:
        raise ValueError(f"compute_embedding_statistics expects (N,D). Got {tuple(emb.shape)}")

    emb_f = emb
    if emb_f.is_complex():
        emb_f = emb_f.real

    emb_f = emb_f.to(dtype=torch.float32)
    N, D = emb_f.shape

    # per-node norms
    norms = torch.linalg.norm(emb_f, dim=1)  # (N,)
    norms_np = norms.detach().cpu().numpy()

    # per-dim summaries
    mean_d = emb_f.mean(dim=0)
    var_d = emb_f.var(dim=0, unbiased=False)
    std_d = torch.sqrt(var_d.clamp_min(0.0))
    min_d = emb_f.min(dim=0).values
    max_d = emb_f.max(dim=0).values

    # global
    mean_g = float(emb_f.mean().item())
    var_g = float(emb_f.var(unbiased=False).item())
    std_g = float(np.sqrt(max(var_g, 0.0)))

    # norm stats
    def _q(x: np.ndarray, q: float) -> float:
        return float(np.quantile(x, q))

    stats: dict[str, Any] = {
        "shape": [int(N), int(D)],
        "global": {
            "mean": mean_g,
            "var": var_g,
            "std": std_g,
            "abs_mean": float(emb_f.abs().mean().item()),
            "abs_max": float(emb_f.abs().max().item()),
        },
        "norms": {
            "mean": float(norms.mean().item()),
            "var": float(norms.var(unbiased=False).item()),
            "std": float(norms.std(unbiased=False).item()),
            "min": float(norms.min().item()),
            "max": float(norms.max().item()),
            "q05": _q(norms_np, 0.05),
            "q25": _q(norms_np, 0.25),
            "q50": _q(norms_np, 0.50),
            "q75": _q(norms_np, 0.75),
            "q95": _q(norms_np, 0.95),
        },
        "per_dim": {
            "mean": mean_d.detach().cpu().tolist(),
            "var": var_d.detach().cpu().tolist(),
            "std": std_d.detach().cpu().tolist(),
            "min": min_d.detach().cpu().tolist(),
            "max": max_d.detach().cpu().tolist(),
        },
    }
    stats["spectrum"] = explained_variance(emb_f, center=True, eps=1e-12)
    return stats




# =============================================================================
# PCA
# =============================================================================


def pca_2d(
    emb: torch.Tensor,
    *,
    center: bool = True,
    whiten: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Compute a 2D PCA projection.

    Parameters
    ----------
    emb:
        (N, D) float tensor (CPU preferred). If complex, real part is used.
    center:
        Whether to mean-center features.
    whiten:
        If True, scale components by inverse singular values (not always desirable).

    Returns
    -------
    coords:
        (N, 2) numpy array.
    info:
        Dict containing explained variance ratio, singular values, mean vector, etc.
    """
    if emb.ndim != 2:
        raise ValueError(f"pca_2d expects emb shape (N,D). Got {tuple(emb.shape)}")

    X = emb
    if X.is_complex():
        X = X.real
    X = X.to(dtype=torch.float64)  # stable SVD
    X = X.detach().cpu()

    N, D = X.shape
    mu = X.mean(dim=0) if center else torch.zeros(D, dtype=X.dtype)
    Xc = X - mu if center else X

    # SVD on (N,D): Xc = U S V^T
    # coords in PC space: U[:, :2] * S[:2]
    # explained variance: S^2 / (N-1)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

    # Take first 2 PCs
    U2 = U[:, :2]
    S2 = S[:2]

    Z = U2 * S2  # (N,2)

    if whiten:
        # Avoid divide by zero
        Z = Z / (S2.clamp_min(1e-12))

    # explained variance ratio
    # variance per component = S^2 / (N-1)
    denom = max(1, N - 1)
    var_components = (S**2) / float(denom)
    total_var = float(var_components.sum().item()) if var_components.numel() > 0 else 0.0
    evr = (
        (var_components[:2] / total_var).detach().cpu().numpy()
        if total_var > 0
        else np.array([0.0, 0.0], dtype=np.float64)
    )

    info = {
        "center": bool(center),
        "whiten": bool(whiten),
        "mean": mu.detach().cpu().tolist() if center else None,
        "singular_values": S.detach().cpu().tolist(),
        "explained_variance_ratio_2d": evr.tolist(),
        "N": int(N),
        "D": int(D),
    }
    return Z.detach().cpu().numpy(), info


# =============================================================================
# Plotting
# =============================================================================


def _import_matplotlib():
    import matplotlib

    # Headless-safe; don’t globally change if already set
    try:
        matplotlib.use("Agg", force=False)
    except Exception:
        pass

    import matplotlib.pyplot as plt

    return plt


def plot_embeddings_2d(
    coords_2d: np.ndarray,
    *,
    out_png: Path,
    title: str = "Node embeddings (PCA 2D)",
    annotate_nodes: bool = False,
    point_size: float = 18.0,
    alpha: float = 0.85,
    color_values: Sequence[float] | None = None,
    color_label: str | None = None,
) -> None:
    """
    Scatter plot of 2D embedding coordinates.

    If `color_values` is provided (length N), points are colored by it.
    """
    out_png = Path(out_png)
    _ensure_dir(out_png.parent)

    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError(f"coords_2d must be (N,2). Got {coords_2d.shape}")

    N = coords_2d.shape[0]
    if color_values is not None:
        if len(color_values) != N:
            raise ValueError(
                f"color_values length must match N={N}. Got len={len(color_values)}."
            )
        if not _is_numeric_sequence(color_values):
            raise TypeError("color_values must be numeric.")

    plt = _import_matplotlib()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    xs = coords_2d[:, 0]
    ys = coords_2d[:, 1]

    if color_values is None:
        sc = ax.scatter(xs, ys, s=point_size, alpha=alpha)
    else:
        sc = ax.scatter(xs, ys, s=point_size, alpha=alpha, c=np.asarray(color_values))
        cbar = fig.colorbar(sc, ax=ax)
        if color_label:
            cbar.set_label(str(color_label))

    if annotate_nodes and N <= 512:
        # Don’t annotate huge graphs by default
        for i in range(N):
            ax.annotate(str(i), (xs[i], ys[i]), fontsize=7, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# =============================================================================
# Saving helpers
# =============================================================================


def _save_csv_matrix(path: Path, header: list[str], rows: Iterable[Sequence[Any]]) -> None:
    path = Path(path)
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))


def _save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _save_embeddings_csv(path: Path, emb: torch.Tensor) -> None:
    if emb.ndim != 2:
        raise ValueError(f"_save_embeddings_csv expects (N,D). Got {tuple(emb.shape)}")
    N, D = emb.shape
    emb_np = _to_numpy(emb.to(torch.float32))
    header = ["node"] + [f"dim_{j}" for j in range(D)]
    rows = ((i, *emb_np[i].tolist()) for i in range(N))
    _save_csv_matrix(path, header, rows)


def _save_pca_csv(path: Path, coords_2d: np.ndarray) -> None:
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError(f"_save_pca_csv expects (N,2). Got {coords_2d.shape}")
    N = coords_2d.shape[0]
    header = ["node", "pc1", "pc2"]
    rows = ((i, float(coords_2d[i, 0]), float(coords_2d[i, 1])) for i in range(N))
    _save_csv_matrix(path, header, rows)


# =============================================================================
# High-level convenience: extract + save + plot
# =============================================================================


@torch.no_grad()
def save_embeddings_artifacts(
    *,
    model: nn.Module,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    run_dir: str | Path,
    T: int | None = None,
    cfg: EmbeddingEvalConfig | None = None,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], EmbeddingArtifacts]:
    """
    End-to-end helper:
      1) Extract embeddings
      2) Aggregate time (if needed) for stats/plotting
      3) Compute statistics (variance, norms, etc.)
      4) Save artifacts (pt/csv/npz/json + PCA plot/csv)

    Returns
    -------
    emb_raw:
        The raw extracted tensor (N,D) or (T,N,D) depending on the model.
    stats:
        Statistics dict (JSON-serializable) computed on aggregated embeddings (N,D).
    artifacts:
        Paths to written artifacts.
    """
    if cfg is None:
        cfg = EmbeddingEvalConfig()

    run_dir = Path(run_dir)
    _ensure_dir(run_dir)

    artifacts = EmbeddingArtifacts()

    emb_raw = extract_node_embeddings(
        model=model,
        X=X,
        edge_index=edge_index,
        T=T,
        cfg=cfg,
        forward_kwargs=forward_kwargs,
    )

    # Aggregate for stats/plotting
    emb_plot = aggregate_time_embeddings(
        emb_raw, mode=cfg.time_aggregate, time_index=cfg.time_index
    )

    if emb_plot.ndim != 2:
        raise RuntimeError(
            "After time aggregation, expected (N,D) embeddings for plotting/stats. "
            f"Got {tuple(emb_plot.shape)}. Check cfg.time_aggregate."
        )


    # Numerical safety: embeddings should be finite for stats/PCA
    if not torch.isfinite(emb_plot).all():
        bad = (~torch.isfinite(emb_plot)).sum().item()
        raise RuntimeError(f"Non-finite values in embeddings (count={bad}). Check model/hook output.")


    # Stats
    stats = compute_embedding_statistics(emb_plot)

    # Optional downsample for plot only
    emb_for_plot = emb_plot
    plot_color_values = cfg.color_values
    down_idx: np.ndarray | None = None

    if cfg.max_nodes_for_plot is not None:
        N = int(emb_plot.shape[0])
        k = int(cfg.max_nodes_for_plot)
        if k < N:
            down_idx = _downsample_indices(N, k, seed=cfg.downsample_seed)
            emb_for_plot = emb_plot[torch.from_numpy(down_idx).to(dtype=torch.long)]
            if plot_color_values is not None:
                plot_color_values = [plot_color_values[i] for i in down_idx.tolist()]

            stats["plot_downsample"] = {
                "enabled": True,
                "N_full": N,
                "N_plot": k,
                "seed": int(cfg.downsample_seed),
            }
        else:
            stats["plot_downsample"] = {"enabled": False, "reason": "k>=N"}
    else:
        stats["plot_downsample"] = {"enabled": False}

    # Save raw embeddings
    stem = cfg.prefix

    if cfg.save_pt:
        p = run_dir / f"{stem}.pt"
        torch.save(
            {
                "embeddings": emb_raw,
                "embeddings_agg": emb_plot,
                "time_aggregate": cfg.time_aggregate,
                "time_index": cfg.time_index,
            },
            p,
        )
        artifacts.embeddings_pt = p

    if cfg.save_csv:
        p = run_dir / f"{stem}.csv"
        _save_embeddings_csv(p, emb_plot)
        artifacts.embeddings_csv = p

    if cfg.save_npz:
        p = run_dir / f"{stem}.npz"
        np.savez_compressed(
            p,
            embeddings=_to_numpy(emb_plot.to(torch.float32)),
        )
        artifacts.embeddings_npz = p

    if cfg.save_metadata_json:
        meta = {
            "prefix": cfg.prefix,
            "extract": {
                "method": cfg.method,
                "hook_module_path": cfg.hook_module_path,
                "hook_output_index": cfg.hook_output_index,
            },
            "time": {
                "T": int(T) if T is not None else None,
                "aggregate": cfg.time_aggregate,
                "time_index": int(cfg.time_index),
            },
            "stats": stats,
            "plot": {
                "max_nodes_for_plot": cfg.max_nodes_for_plot,
                "annotate_nodes": cfg.plot_annotate_nodes,
                "point_size": float(cfg.plot_point_size),
                "alpha": float(cfg.plot_alpha),
                "color_label": cfg.color_label,
            },
        }
        if down_idx is not None:
            meta["plot"]["downsample_indices"] = down_idx.tolist()

        p = run_dir / f"{stem}.meta.json"
        _save_json(p, meta)
        artifacts.metadata_json = p

    # PCA + plot
    if cfg.make_pca_plot:
        coords, pca_info = pca_2d(
            emb_for_plot,
            center=cfg.pca_center,
            whiten=cfg.pca_whiten,
        )
        stats["pca"] = pca_info

        # Save coords CSV
        p_csv = run_dir / f"{stem}.pca2d.csv"
        _save_pca_csv(p_csv, coords)
        artifacts.pca_csv = p_csv

        # Plot
        p_png = run_dir / f"{stem}.pca2d.png"
        title = "Node embeddings (PCA 2D)"
        evr = pca_info.get("explained_variance_ratio_2d", None)
        if isinstance(evr, list) and len(evr) == 2:
            title += f" | EVR=[{evr[0]:.3f}, {evr[1]:.3f}]"

        plot_embeddings_2d(
            coords,
            out_png=p_png,
            title=title,
            annotate_nodes=cfg.plot_annotate_nodes,
            point_size=cfg.plot_point_size,
            alpha=cfg.plot_alpha,
            color_values=plot_color_values,
            color_label=cfg.color_label,
        )
        artifacts.pca_png = p_png

    log.info(
        "Saved embedding artifacts under %s (pt=%s csv=%s pca_png=%s)",
        str(run_dir),
        str(artifacts.embeddings_pt) if artifacts.embeddings_pt else None,
        str(artifacts.embeddings_csv) if artifacts.embeddings_csv else None,
        str(artifacts.pca_png) if artifacts.pca_png else None,
    )

    return emb_raw, stats, artifacts

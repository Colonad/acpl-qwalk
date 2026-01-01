# acpl/eval/embeddings.py
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, cast
from .stats import StatsEvalConfig, embedding_basic_stats
import numpy as np
import torch
import torch.nn as nn
from acpl.data.features import FeatureSpec, build_node_features

__all__ = [
    "EmbeddingEvalConfig",
    "EmbeddingArtifacts",
    "extract_node_embeddings",
    "aggregate_time_embeddings",
    "compute_embedding_statistics",
    "pca_2d",
    "tsne_2d"
    "plot_embeddings_2d",
    "save_embeddings_artifacts",
    "resolve_module_by_path",
    "write_embeddings",
    "save_embeddings",
    "export_embeddings",
    # graph/dataset-level embedding artifacts
    "pool_node_embeddings_to_graph",

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
    # Reproducibility / numerical safety
    # - PCA sign fixing makes 2D PCA stable across BLAS backends.
    pca_fix_sign: bool = True

    # If True, do NOT fail on non-finite embeddings; stats will report counts and PCA/plots
    # will operate on a finite-masked copy. Default False (fail-fast for defendability).
    allow_nonfinite: bool = False




    # -------------------------------------------------------------------------
    # Dataset / graph-level embedding artifacts (ESOL-like plots)
    #
    # Your node embedding writer produces (K,N,D) over K = seeds×episodes.
    # This section pools nodes -> per-sample graph embedding (K,D) and plots:
    #   - PCA2D scatter of graph embeddings
    #   - t-SNE2D scatter of graph embeddings (optional; can be slower)
    #
    # This is what you need for “dataset embedding vs baseline embedding” figures.
    # -------------------------------------------------------------------------

    # Enable pooled graph/dataset embedding artifacts.
    make_graph_embeddings: bool = True

    # Pooling from (K,N,D)->(K,D). All are deterministic.
    #  - mean: average over nodes
    #  - sum: sum over nodes
    #  - max: elementwise max over nodes
    #  - rms: sqrt(mean(x^2)) over nodes (preserves magnitude robustly)
    #  - mean_abs: mean(|x|) over nodes (magnitude-only)
    #  - pmean: signed power-mean: sign(mean(x))* (mean(|x|^p))^(1/p)
    graph_pool: Literal["mean", "sum", "max", "rms", "mean_abs", "pmean"] = "mean"
    graph_pmean_p: float = 2.0

    # Optional normalization for each graph embedding vector.
    graph_l2_normalize: bool = False

    # Save pooled embeddings
    graph_save_npy: bool = True
    graph_save_csv: bool = False  # CSV can be large; keep False by default
    graph_save_metadata_json: bool = True

    # Plots
    graph_make_pca_plot: bool = True
    graph_make_tsne_plot: bool = True

    # If K is huge, downsample points for plotting (still saves full .npy).
    graph_max_points_for_plot: int | None = 2500
    graph_downsample_seed: int = 0

    # Plot style (defaults match node style)
    graph_plot_point_size: float = 18.0
    graph_plot_alpha: float = 0.85
    graph_plot_annotate_points: bool = False

    # Optional numeric coloring for graph points (length K). If set, can bin like ESOL.
    graph_color_values: Sequence[float] | None = None
    graph_color_label: str | None = None
    graph_color_bins: int | None = None  # e.g. 5 like your ESOL “target bin”
    graph_color_bin_strategy: Literal["quantile", "uniform"] = "quantile"

    # t-SNE parameters (sklearn)
    tsne_seed: int = 0
    tsne_perplexity: float | None = None  # if None, chosen adaptively from K
    tsne_learning_rate: float | Literal["auto"] = "auto"
    tsne_max_iter: int = 1500
    tsne_init: Literal["pca", "random"] = "pca"
    tsne_metric: str = "euclidean"
    tsne_method: Literal["barnes_hut", "exact"] = "barnes_hut"
    tsne_angle: float = 0.5





    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "EmbeddingEvalConfig":
        """
        Robust parser for config manifests. Unknown keys are ignored so that
        experiment YAMLs can evolve without breaking evaluation code.

        Accepts either a flat dict of EmbeddingEvalConfig fields or a nested
        dict containing a likely "embeddings" block.
        """
        if not d:
            return cls()

        # Try common nesting patterns (cfg['eval']['embeddings'], cfg['embeddings'], etc.)
        cand = d
        for path in (("eval", "embeddings"), ("embeddings",), ("embedding",), ("artifacts", "embeddings")):
            cur = d
            ok = True
            for k in path:
                if not isinstance(cur, Mapping) or k not in cur:
                    ok = False
                    break
                cur = cur[k]  # type: ignore[index]
            if ok and isinstance(cur, Mapping):
                cand = cur
                break

        from dataclasses import fields

        valid = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}

        for k, v in cand.items():
            if k not in valid:
                continue
            # Basic coercions for YAML/JSON safety
            f = valid[k]
            if f.type in (int, "int"):
                try:
                    v = int(v)
                except Exception:
                    pass
            elif f.type in (float, "float"):
                try:
                    v = float(v)
                except Exception:
                    pass
            elif f.type in (bool, "bool"):
                v = bool(v)
            kwargs[k] = v

        return cls(**kwargs)


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


def compute_embedding_statistics(
    emb: torch.Tensor,
    *,
    cfg: StatsEvalConfig | None = None,
) -> dict[str, Any]:
    """
    Compute defendable "overall statistics" for embeddings using the canonical
    implementation in acpl.eval.stats.embedding_basic_stats.

    This wrapper exists for backward compatibility with older eval code that calls
    compute_embedding_statistics(...).
    """
    return embedding_basic_stats(emb, cfg=cfg)




# =============================================================================
# PCA
# =============================================================================


def pca_2d(
    emb: torch.Tensor,
    *,
    center: bool = True,
    whiten: bool = False,
    fix_sign: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Compute a 2D PCA projection with reproducibility fixes.

    `fix_sign=True` resolves the sign ambiguity of SVD-based PCA by forcing each
    component's largest-magnitude loading to be positive. This makes PCA plots
    stable across runs/platforms (BLAS differences can otherwise flip axes).

    Parameters
    ----------
    emb : (N,D) tensor
    center : bool
    whiten : bool
    fix_sign : bool

    Returns
    -------
    coords : (N,2) numpy
    info : dict
    """
    if emb.ndim != 2:
        raise ValueError(f"pca_2d expects emb shape (N,D). Got {tuple(emb.shape)}")

    X = emb
    if X.is_complex():
        X = X.real
    X = X.to(dtype=torch.float64).detach().cpu()

    N, D = X.shape
    mu = X.mean(dim=0) if center else torch.zeros(D, dtype=X.dtype)
    Xc = X - mu if center else X

    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

    U2 = U[:, :2]
    S2 = S[:2]
    Z = U2 * S2

    if fix_sign:
        signs = []
        for j in range(2):
            v = Vh[j]
            idx = int(torch.argmax(torch.abs(v)).item())
            sgn = 1.0 if float(v[idx]) >= 0.0 else -1.0
            Z[:, j] *= sgn
            signs.append(float(sgn))
    else:
        signs = None

    if whiten:
        Z = Z / (S2.clamp_min(1e-12))

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
        "fix_sign": bool(fix_sign),
        "signs": signs,
        "mean": mu.detach().cpu().tolist() if center else None,
        "singular_values": S.detach().cpu().tolist(),
        "explained_variance_ratio_2d": evr.tolist(),
        "N": int(N),
        "D": int(D),
    }
    return Z.detach().cpu().numpy(), info



# =============================================================================
# t-SNE (graph/dataset embedding view)
# =============================================================================


def tsne_2d(
    X: np.ndarray,
    *,
    seed: int = 0,
    perplexity: float | None = None,
    learning_rate: float | str = "auto",
    max_iter: int = 1500,
    init: Literal["pca", "random"] = "pca",
    metric: str = "euclidean",
    method: Literal["barnes_hut", "exact"] = "barnes_hut",
    angle: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Deterministic(ish) t-SNE wrapper for plotting graph embeddings.

    - Uses sklearn.manifold.TSNE if available.
    - Adapts perplexity to K when not provided (must be < K).
    - Records all parameters to an info dict for defendability.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"tsne_2d expects X shape (K,D). Got {X.shape}")
    K, D = map(int, X.shape)

    # Guardrails: t-SNE is ill-defined for tiny K
    if K < 3:
        coords = np.zeros((K, 2), dtype=np.float64)
        info = {
            "skipped": True,
            "reason": f"K={K} < 3",
            "K": K,
            "D": D,
        }
        return coords, info

    # Adaptive perplexity rule of thumb: < (K-1)/3
    if perplexity is None:
        # 30 is common, but bound it tightly for small K.
        p = min(30.0, max(2.0, float((K - 1) / 3.0)))
        perplexity = p
    else:
        perplexity = float(perplexity)

    # Enforce sklearn constraint: perplexity < K
    if perplexity >= K:
        perplexity = max(2.0, float(K - 1) * 0.333)
        perplexity = min(perplexity, float(K - 1) - 1e-6)

    # sklearn import locally (optional dependency)
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "t-SNE requested but scikit-learn is not available. "
            "Install it (pip install scikit-learn) or disable graph_make_tsne_plot."
        ) from e

    # t-SNE is sensitive to dtype; float64 is safer for stability.
    X64 = X.astype(np.float64, copy=False)

    # If metric != euclidean, barnes_hut is restricted; fall back to exact.
    method2: str = str(method)
    if metric != "euclidean" and method2 == "barnes_hut":
        method2 = "exact"

    # sklearn changed parameter name from n_iter -> max_iter; TSNE in modern sklearn uses max_iter.
    # We pass max_iter and keep compatibility by trying both in a guarded way.
    tsne_kwargs: dict[str, Any] = dict(
        n_components=2,
        perplexity=float(perplexity),
        learning_rate=learning_rate,
        init=str(init),
        metric=str(metric),
        method=str(method2),
        angle=float(angle),
        random_state=int(seed),
    )

    # Construct with max_iter if supported; else fall back.
    coords: np.ndarray
    model = None
    try:
        model = TSNE(**tsne_kwargs, max_iter=int(max_iter))
        coords = model.fit_transform(X64)
    except TypeError:
        model = TSNE(**tsne_kwargs, n_iter=int(max_iter))
        coords = model.fit_transform(X64)

    info = {
        "skipped": False,
        "K": K,
        "D": D,
        "seed": int(seed),
        "perplexity": float(perplexity),
        "learning_rate": learning_rate,
        "max_iter": int(max_iter),
        "init": str(init),
        "metric": str(metric),
        "method": str(method2),
        "angle": float(angle),
        "kl_divergence": float(getattr(model, "kl_divergence_", float("nan"))),
    }
    return coords.astype(np.float64, copy=False), info


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
    *,
    color_bins: int | None = None,
    color_bin_edges: Sequence[float] | None = None,
    color_bin_ticklabels: Sequence[str] | None = None,
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
        cv = np.asarray(list(color_values), dtype=np.float64)
        if color_bins is not None:
            # Discrete/binned coloring (ESOL-like “target bin”)
            bins = int(color_bins)
            if bins < 2:
                bins = 2
            if color_bin_edges is None:
                raise ValueError("color_bins provided but color_bin_edges is None.")
            edges = np.asarray(list(color_bin_edges), dtype=np.float64)
            if edges.ndim != 1 or edges.size != bins + 1:
                raise ValueError(
                    f"color_bin_edges must have length bins+1={bins+1}. Got {edges.size}."
                )
            # Compute bin ids [0..bins-1]
            # digitize returns 1..bins for right=False, so subtract 1 and clip.
            bin_ids = np.digitize(cv, edges[1:-1], right=False).astype(np.int64)
            bin_ids = np.clip(bin_ids, 0, bins - 1)

            from matplotlib.colors import BoundaryNorm  # type: ignore

            norm = BoundaryNorm(boundaries=np.arange(-0.5, bins + 0.5, 1.0), ncolors=bins)
            sc = ax.scatter(xs, ys, s=point_size, alpha=alpha, c=bin_ids, norm=norm)
            cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(0, bins, 1))
            if color_label:
                cbar.set_label(str(color_label))
            if color_bin_ticklabels is not None:
                cbar.set_ticklabels(list(color_bin_ticklabels))
        else:
            sc = ax.scatter(xs, ys, s=point_size, alpha=alpha, c=cv)
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



def _save_graph_embeddings_csv(path: Path, X: np.ndarray) -> None:
    """
    Save (K,D) graph embeddings to CSV. Off by default due to size.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"_save_graph_embeddings_csv expects (K,D). Got {X.shape}")
    K, D = map(int, X.shape)
    header = ["sample"] + [f"dim_{j}" for j in range(D)]
    rows = ((i, *[float(v) for v in X[i].tolist()]) for i in range(K))
    _save_csv_matrix(path, header, rows)


def _save_tsne_csv(path: Path, coords_2d: np.ndarray) -> None:
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError(f"_save_tsne_csv expects (K,2). Got {coords_2d.shape}")
    K = int(coords_2d.shape[0])
    header = ["sample", "tsne1", "tsne2"]
    rows = ((i, float(coords_2d[i, 0]), float(coords_2d[i, 1])) for i in range(K))
    _save_csv_matrix(path, header, rows)


# =============================================================================
# Graph/dataset-level embeddings
# =============================================================================


def pool_node_embeddings_to_graph(
    stack: np.ndarray,
    *,
    pool: Literal["mean", "sum", "max", "rms", "mean_abs", "pmean"] = "mean",
    pmean_p: float = 2.0,
    l2_normalize: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Pool node embeddings from (K,N,D) to (K,D) graph/sample embeddings.

    This is the key operation needed to produce ESOL-like dataset embedding plots.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected stack shape (K,N,D). Got {stack.shape}")
    K, N, D = map(int, stack.shape)
    X = stack.astype(np.float64, copy=False)

    if pool == "mean":
        G = X.mean(axis=1)
    elif pool == "sum":
        G = X.sum(axis=1)
    elif pool == "max":
        G = X.max(axis=1)
    elif pool == "rms":
        G = np.sqrt(np.mean(X * X, axis=1))
    elif pool == "mean_abs":
        G = np.mean(np.abs(X), axis=1)
    elif pool == "pmean":
        p = float(pmean_p)
        if not np.isfinite(p) or p <= 0:
            p = 2.0
        # signed power-mean: preserve sign of mean, magnitude via power mean of |x|
        mu = X.mean(axis=1)
        mag = np.power(np.mean(np.power(np.abs(X), p), axis=1), 1.0 / p)
        G = np.sign(mu) * mag
    else:
        raise ValueError(f"Unknown pool='{pool}'")

    if l2_normalize:
        norms = np.linalg.norm(G, axis=1, keepdims=True) + eps
        G = G / norms

    return G.astype(np.float64, copy=False)


def _extract_numeric_vector_from_meta(
    meta: Mapping[str, Any] | None,
    *,
    K: int,
    keys: Sequence[str],
) -> Sequence[float] | None:
    if not meta:
        return None
    for k in keys:
        if k not in meta:
            continue
        v = meta.get(k)
        if v is None:
            continue
        try:
            arr = np.asarray(v, dtype=np.float64)
        except Exception:
            continue
        if arr.ndim != 1 or int(arr.size) != int(K):
            continue
        if not np.isfinite(arr).all():
            # allow; caller can decide to bin or mask
            pass
        return [float(x) for x in arr.tolist()]
    return None


def _bin_numeric_values(
    values: Sequence[float],
    *,
    bins: int,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Produce ESOL-like bins for a numeric scalar.
    Returns:
      ids   : (K,) int bin ids
      edges : (bins+1,) edges
      labels: (bins,) human-readable “[lo, hi]”
    """
    x = np.asarray(list(values), dtype=np.float64)
    bins = int(bins)
    if bins < 2:
        bins = 2

    finite = np.isfinite(x)
    xf = x[finite]
    if xf.size == 0:
        edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
        ids = np.zeros_like(x, dtype=np.int64)
        labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f}]" for i in range(bins)]
        return ids, edges, labels

    if strategy == "uniform":
        lo = float(np.min(xf))
        hi = float(np.max(xf))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
        else:
            edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
    else:
        # quantile (default) — robust to outliers
        qs = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(xf, qs).astype(np.float64)
        # ensure strictly non-decreasing edges; fix ties by tiny jitter
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12

    ids = np.digitize(x, edges[1:-1], right=False).astype(np.int64)
    ids = np.clip(ids, 0, bins - 1)
    labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f}]" for i in range(bins)]
    return ids, edges, labels


def _graph_embedding_stability_metrics(G: np.ndarray, mean: np.ndarray, *, eps: float = 1e-12) -> dict[str, Any]:
    """
    Stability metrics across K samples (dataset-level):
      - cosine similarity to mean embedding (per-sample), summarized
      - relative L2 deviation to mean (per-sample), summarized
    """
    G = np.asarray(G, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"G must be (K,D). Got {G.shape}")
    K, D = map(int, G.shape)
    mean_norm = float(np.linalg.norm(mean)) + eps
    g_norm = np.linalg.norm(G, axis=1) + eps
    cos = (G @ mean) / (g_norm * mean_norm)
    rel = np.linalg.norm(G - mean[None, :], axis=1) / mean_norm

    def q(a: np.ndarray, p: float) -> float:
        return float(np.quantile(a, p)) if a.size else float("nan")

    return {
        "K": int(K),
        "D": int(D),
        "cosine_to_mean": {
            "mean": float(np.mean(cos)) if K else float("nan"),
            "std": float(np.std(cos, ddof=0)) if K else float("nan"),
            "q05": q(cos, 0.05),
            "q50": q(cos, 0.50),
            "q95": q(cos, 0.95),
        },
        "rel_l2_deviation": {
            "mean": float(np.mean(rel)) if K else float("nan"),
            "std": float(np.std(rel, ddof=0)) if K else float("nan"),
            "q05": q(rel, 0.05),
            "q50": q(rel, 0.50),
            "q95": q(rel, 0.95),
        },
    }



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
      3) Compute statistics (finite-safe; includes spectrum/effective-rank)
      4) Save artifacts (pt/csv/npz/json + PCA plot/csv)

    Returns
    -------
    emb_raw : raw extracted tensor (N,D) or (T,N,D)
    stats   : JSON-ready stats dict computed on aggregated embeddings (N,D)
    artifacts : paths to written artifacts
    """
    if cfg is None:
        cfg = EmbeddingEvalConfig()

    run_dir = Path(run_dir)
    _ensure_dir(run_dir)
    artifacts = EmbeddingArtifacts()

    # ---- extract ----
    emb_raw = extract_node_embeddings(
        model=model,
        X=X,
        edge_index=edge_index,
        T=T,
        cfg=cfg,
        forward_kwargs=forward_kwargs,
    )

    # ---- aggregate to (N,D) for stats/plotting ----
    emb_plot = aggregate_time_embeddings(
        emb_raw, mode=cfg.time_aggregate, time_index=cfg.time_index
    )
    if emb_plot.ndim != 2:
        raise RuntimeError(
            "After time aggregation, expected (N,D) embeddings for plotting/stats. "
            f"Got {tuple(emb_plot.shape)}. Check cfg.time_aggregate."
        )

    finite_mask = torch.isfinite(emb_plot)
    if not bool(finite_mask.all().item()):
        bad = int((~finite_mask).sum().item())
        if not cfg.allow_nonfinite:
            raise RuntimeError(
                f"Non-finite values in embeddings (count={bad}). "
                "Fix model/hook output or set cfg.allow_nonfinite=True."
            )
        log.warning(
            "Non-finite values in embeddings (count=%d). Proceeding with finite-masked copy.",
            bad,
        )

    # ---- stats (canonical, finite-safe) ----
    st_cfg = StatsEvalConfig()
    stats = compute_embedding_statistics(emb_plot, cfg=st_cfg)

    # ---- optional downsample for plotting only ----
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

    # ---- write raw embeddings ----
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
        np.savez_compressed(p, embeddings=_to_numpy(emb_plot.to(torch.float32)))
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

    # ---- PCA + plot ----
    if cfg.make_pca_plot:
        emb_for_plot_safe = torch.where(
            torch.isfinite(emb_for_plot),
            emb_for_plot,
            torch.zeros_like(emb_for_plot),
        )
        coords, pca_info = pca_2d(
            emb_for_plot_safe,
            center=cfg.pca_center,
            whiten=cfg.pca_whiten,
            fix_sign=cfg.pca_fix_sign,
        )
        stats["pca"] = pca_info

        p_csv = run_dir / f"{stem}.pca2d.csv"
        _save_pca_csv(p_csv, coords)
        artifacts.pca_csv = p_csv

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



# =============================================================================
# First-class embedding artifact writer (used by scripts/eval.py)
# =============================================================================

def _extract_embeddings_cfg_dict(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not cfg:
        return {}
    if "embeddings" in cfg and isinstance(cfg["embeddings"], Mapping):
        return cfg["embeddings"]  # type: ignore[return-value]
    if "eval" in cfg and isinstance(cfg["eval"], Mapping):
        ev = cfg["eval"]
        if "embeddings" in ev and isinstance(ev["embeddings"], Mapping):
            return ev["embeddings"]  # type: ignore[return-value]
    return {}


def _coerce_embeddings_stack(embeddings: Any) -> np.ndarray:
    """Coerce `embeddings` into a numpy array shaped (K,N,D)."""
    if isinstance(embeddings, torch.Tensor):
        t = embeddings
        if t.is_complex():
            t = t.real
        t = t.detach().cpu()
        arr = t.numpy()
    else:
        arr = np.asarray(embeddings)

    if arr.ndim == 2:
        arr = arr[None, :, :]  # (1,N,D)
    if arr.ndim != 3:
        raise ValueError(f"Expected embeddings with shape (K,N,D) or (N,D). Got {arr.shape}")
    return arr


def _embedding_stability_metrics(stack: np.ndarray, mean: np.ndarray, *, eps: float = 1e-12) -> dict[str, Any]:
    """
    Novel research-grade stability diagnostics across K samples (seeds×episodes):

      - avg node-wise cosine similarity to mean embedding
      - relative Frobenius deviation to mean embedding

    Low stability indicates the embedding geometry is not consistent and PCA plots
    should be interpreted carefully.
    """
    K, N, D = stack.shape
    mean_norm = np.linalg.norm(mean, axis=1) + eps
    fro_mean = float(np.linalg.norm(mean)) + eps

    cos_per_sample = np.zeros(K, dtype=np.float64)
    rel_fro = np.zeros(K, dtype=np.float64)

    for k in range(K):
        Ek = stack[k]
        Ek_norm = np.linalg.norm(Ek, axis=1) + eps
        cos = (Ek * mean).sum(axis=1) / (Ek_norm * mean_norm)
        cos_per_sample[k] = float(np.mean(cos))
        rel_fro[k] = float(np.linalg.norm(Ek - mean) / fro_mean)

    def q(x: np.ndarray, p: float) -> float:
        return float(np.quantile(x, p)) if x.size else float("nan")

    return {
        "K": int(K),
        "cosine_to_mean": {
            "mean": float(cos_per_sample.mean()) if K else float("nan"),
            "std": float(cos_per_sample.std(ddof=0)) if K else float("nan"),
            "q05": q(cos_per_sample, 0.05),
            "q50": q(cos_per_sample, 0.50),
            "q95": q(cos_per_sample, 0.95),
        },
        "rel_frobenius_deviation": {
            "mean": float(rel_fro.mean()) if K else float("nan"),
            "std": float(rel_fro.std(ddof=0)) if K else float("nan"),
            "q05": q(rel_fro, 0.05),
            "q50": q(rel_fro, 0.50),
            "q95": q(rel_fro, 0.95),
        },
    }


def write_embeddings(
    *,
    outdir: str | Path,
    cond: str,
    embeddings: Any,
    meta: Mapping[str, Any] | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    First-class embedding artifact writer used by scripts/eval.py.

    Writes:
      - embeddings.npy (K,N,D)
      - embeddings_mean.npy (N,D)
      - embeddings_stats.json (mean+stack stats + stability)
      - embeddings.pca2d.csv/png
      - embeddings.meta.json (provenance)
    """
    outdir = Path(outdir)
    _ensure_dir(outdir)

    cfg_dict = _extract_embeddings_cfg_dict(cfg)
    ecfg = EmbeddingEvalConfig.from_dict(cfg_dict)

    stack = _coerce_embeddings_stack(embeddings).astype(np.float64, copy=False)
    K, N, D = map(int, stack.shape)

    finite = np.isfinite(stack)
    nonfinite_n = int((~finite).sum())
    if nonfinite_n > 0 and not ecfg.allow_nonfinite:
        raise RuntimeError(
            f"Non-finite values in embeddings stack (count={nonfinite_n}). "
            "Fix the model/hook output or set allow_nonfinite=True in embeddings cfg."
        )


    if nonfinite_n > 0:
        count = finite.sum(axis=0).astype(np.float64)
        s = np.where(finite, stack, 0.0).sum(axis=0)
        mean = s / np.maximum(count, 1.0)
        mean[count == 0] = np.nan  # explicit: no samples contributed to this entry
    else:
        mean = stack.mean(axis=0)

    mean_safe = np.where(np.isfinite(mean), mean, 0.0)

    # If non-finite values exist and we allow them, stability metrics must be computed
    # on a finite-masked copy; otherwise norms/dot-products become NaN.
    stack_safe = np.where(finite, stack, 0.0) if nonfinite_n > 0 else stack


    files: dict[str, str] = {}
    raw_path = outdir / "embeddings.npy"
    np.save(raw_path, stack.astype(np.float32, copy=False))
    files["embeddings_npy"] = str(raw_path)

    mean_path = outdir / "embeddings_mean.npy"
    np.save(mean_path, mean.astype(np.float32, copy=False))
    files["embeddings_mean_npy"] = str(mean_path)

    st_cfg = StatsEvalConfig()
    mean_stats = compute_embedding_statistics(torch.from_numpy(mean).to(torch.float32), cfg=st_cfg)
    flat = stack.reshape(K * N, D)
    flat_stats = compute_embedding_statistics(torch.from_numpy(flat).to(torch.float32), cfg=st_cfg)

    stability = _embedding_stability_metrics(stack_safe, mean_safe)

    stats_payload: dict[str, Any] = {
        "cond": str(cond),
        "shape": {"K": K, "N": N, "D": D},
        "nonfinite_n": nonfinite_n,
        "stats_mean": mean_stats,
        "stats_stack": flat_stats,
        "stability": stability,
    }

    stats_path = outdir / "embeddings_stats.json"
    _save_json(stats_path, stats_payload)
    files["embeddings_stats_json"] = str(stats_path)

    mean_t = torch.from_numpy(mean).to(torch.float32)
    mean_t = torch.where(torch.isfinite(mean_t), mean_t, torch.zeros_like(mean_t))

    coords, pca_info = pca_2d(
        mean_t,
        center=ecfg.pca_center,
        whiten=ecfg.pca_whiten,
        fix_sign=ecfg.pca_fix_sign,
    )

    pca_csv = outdir / "embeddings.pca2d.csv"
    _save_pca_csv(pca_csv, coords)
    files["pca_csv"] = str(pca_csv)

    pca_png = outdir / "embeddings.pca2d.png"
    title = f"{cond} | embeddings PCA2D"
    evr = pca_info.get("explained_variance_ratio_2d")
    if isinstance(evr, list) and len(evr) == 2:
        title += f" | EVR=[{evr[0]:.3f},{evr[1]:.3f}]"

    plot_embeddings_2d(
        coords,
        out_png=pca_png,
        title=title,
        annotate_nodes=ecfg.plot_annotate_nodes,
        point_size=ecfg.plot_point_size,
        alpha=ecfg.plot_alpha,
        color_values=ecfg.color_values,
        color_label=ecfg.color_label,
    )
    files["pca_png"] = str(pca_png)

    from dataclasses import asdict

    meta_payload = {
        "cond": str(cond),
        "writer": "acpl.eval.embeddings.write_embeddings",
        "writer_cfg": asdict(ecfg),
        "meta": dict(meta or {}),
        "cfg_excerpt": dict(cfg_dict),
        "files": files,
        "pca": pca_info,
        "stability": stability,
    }
    meta_path = outdir / "embeddings.meta.json"
    _save_json(meta_path, meta_payload)
    files["meta_json"] = str(meta_path)




    # -------------------------------------------------------------------------
    # NEW: Graph/dataset-level embedding artifacts (ESOL-like)
    # -------------------------------------------------------------------------
    graph_summary: dict[str, Any] = {}
    if ecfg.make_graph_embeddings:
        # Pool nodes -> graph embeddings
        G = pool_node_embeddings_to_graph(
            stack_safe,
            pool=ecfg.graph_pool,
            pmean_p=float(ecfg.graph_pmean_p),
            l2_normalize=bool(ecfg.graph_l2_normalize),
        )  # (K,D)
        # Mean graph embedding (D,)
        g_mean = np.mean(G, axis=0) if G.size else np.zeros((D,), dtype=np.float64)
        g_mean_safe = np.where(np.isfinite(g_mean), g_mean, 0.0)

        # Save full pooled embeddings
        if ecfg.graph_save_npy:
            gp = outdir / "graph_embeddings.npy"
            np.save(gp, G.astype(np.float32, copy=False))
            files["graph_embeddings_npy"] = str(gp)

            gm = outdir / "graph_embeddings_mean.npy"
            np.save(gm, g_mean.astype(np.float32, copy=False))
            files["graph_embeddings_mean_npy"] = str(gm)

        if ecfg.graph_save_csv:
            gcsv = outdir / "graph_embeddings.csv"
            _save_graph_embeddings_csv(gcsv, G)
            files["graph_embeddings_csv"] = str(gcsv)

        # Stats across dataset points (K,D)
        st_cfg2 = StatsEvalConfig()
        graph_stats = compute_embedding_statistics(torch.from_numpy(G).to(torch.float32), cfg=st_cfg2)
        graph_stability = _graph_embedding_stability_metrics(G, g_mean_safe)

        graph_stats_payload: dict[str, Any] = {
            "cond": str(cond),
            "shape": {"K": int(K), "D": int(D)},
            "pool": str(ecfg.graph_pool),
            "pmean_p": float(ecfg.graph_pmean_p),
            "l2_normalize": bool(ecfg.graph_l2_normalize),
            "nonfinite_n": int(np.sum(~np.isfinite(G))),
            "stats": graph_stats,
            "stability": graph_stability,
        }

        gstats_path = outdir / "graph_embeddings_stats.json"
        _save_json(gstats_path, graph_stats_payload)
        files["graph_embeddings_stats_json"] = str(gstats_path)

        # Plot subset selection (plotting only)
        plot_idx: np.ndarray | None = None
        G_plot = G
        if ecfg.graph_max_points_for_plot is not None:
            kk = int(ecfg.graph_max_points_for_plot)
            if kk < K:
                plot_idx = _downsample_indices(K, kk, seed=int(ecfg.graph_downsample_seed))
                G_plot = G_plot[plot_idx]

        # Determine coloring vector for graph points (length K or length K_plot)
        # Priority:
        #   1) ecfg.graph_color_values (explicit)
        #   2) meta["graph_color_values"] / meta["color_values"] / meta["episode_color_values"] etc.
        graph_color_values: Sequence[float] | None = None
        if ecfg.graph_color_values is not None and len(ecfg.graph_color_values) == K:
            graph_color_values = list(ecfg.graph_color_values)
        else:
            graph_color_values = _extract_numeric_vector_from_meta(
                meta,
                K=K,
                keys=[
                    "graph_color_values",
                    "color_values",
                    "episode_color_values",
                    "episode_metric",
                    "episode_metrics",
                    "y",
                    "targets",
                ],
            )

        graph_color_values_plot: Sequence[float] | None = graph_color_values
        if graph_color_values_plot is not None and plot_idx is not None:
            idx_list = plot_idx.tolist()
            graph_color_values_plot = [graph_color_values_plot[i] for i in idx_list]

        # Optional binning for ESOL-like discrete colorbar
        color_bins = ecfg.graph_color_bins
        color_bin_edges: np.ndarray | None = None
        color_bin_labels: list[str] | None = None
        color_values_for_plot: Sequence[float] | None = graph_color_values_plot
        if graph_color_values_plot is not None and color_bins is not None:
            ids, edges, labels = _bin_numeric_values(
                graph_color_values_plot,
                bins=int(color_bins),
                strategy=str(ecfg.graph_color_bin_strategy),
            )
            color_values_for_plot = [float(x) for x in ids.tolist()]
            color_bin_edges = edges
            color_bin_labels = labels

        # PCA2D on dataset-level points (K_plot,D) -> (K_plot,2)
        if ecfg.graph_make_pca_plot:
            # finite mask (defendable)
            G_plot_safe = np.where(np.isfinite(G_plot), G_plot, 0.0)
            coords_pca, pca_info2 = pca_2d(
                torch.from_numpy(G_plot_safe).to(torch.float32),
                center=bool(ecfg.pca_center),
                whiten=bool(ecfg.pca_whiten),
                fix_sign=bool(ecfg.pca_fix_sign),
            )
            pca_csv2 = outdir / "graph_embeddings.pca2d.csv"
            _save_pca_csv(pca_csv2, coords_pca)
            files["graph_pca_csv"] = str(pca_csv2)

            pca_png2 = outdir / "graph_embeddings.pca2d.png"
            title2 = f"{cond} | graph embeddings PCA2D"
            evr2 = pca_info2.get("explained_variance_ratio_2d")
            if isinstance(evr2, list) and len(evr2) == 2:
                title2 += f" | EVR=[{evr2[0]:.3f},{evr2[1]:.3f}]"
            plot_embeddings_2d(
                coords_pca,
                out_png=pca_png2,
                title=title2,
                annotate_nodes=bool(ecfg.graph_plot_annotate_points),
                point_size=float(ecfg.graph_plot_point_size),
                alpha=float(ecfg.graph_plot_alpha),
                color_values=color_values_for_plot,
                color_label=ecfg.graph_color_label,
                color_bins=int(color_bins) if color_bins is not None else None,
                color_bin_edges=color_bin_edges.tolist() if color_bin_edges is not None else None,
                color_bin_ticklabels=color_bin_labels,
            )
            files["graph_pca_png"] = str(pca_png2)

            graph_summary["pca"] = pca_info2
            if plot_idx is not None:
                graph_summary["plot_downsample_indices"] = [int(i) for i in plot_idx.tolist()]
        # t-SNE2D on dataset-level points (K_plot,D)
        if ecfg.graph_make_tsne_plot:
            G_plot_safe = np.where(np.isfinite(G_plot), G_plot, 0.0)
            try:
                coords_tsne, tsne_info = tsne_2d(
                    G_plot_safe,
                    seed=int(ecfg.tsne_seed),
                    perplexity=ecfg.tsne_perplexity,
                    learning_rate=ecfg.tsne_learning_rate,
                    max_iter=int(ecfg.tsne_max_iter),
                    init=str(ecfg.tsne_init),
                    metric=str(ecfg.tsne_metric),
                    method=str(ecfg.tsne_method),
                    angle=float(ecfg.tsne_angle),
                )
                tsne_csv = outdir / "graph_embeddings.tsne2d.csv"
                _save_tsne_csv(tsne_csv, coords_tsne)
                files["graph_tsne_csv"] = str(tsne_csv)

                tsne_png = outdir / "graph_embeddings.tsne2d.png"
                title3 = f"{cond} | graph embeddings t-SNE2D"
                plot_embeddings_2d(
                    coords_tsne,
                    out_png=tsne_png,
                    title=title3,
                    annotate_nodes=False,
                    point_size=float(ecfg.graph_plot_point_size),
                    alpha=float(ecfg.graph_plot_alpha),
                    color_values=color_values_for_plot,
                    color_label=ecfg.graph_color_label,
                    color_bins=int(color_bins) if color_bins is not None else None,
                    color_bin_edges=color_bin_edges.tolist() if color_bin_edges is not None else None,
                    color_bin_ticklabels=color_bin_labels,
                )
                files["graph_tsne_png"] = str(tsne_png)
                graph_summary["tsne"] = tsne_info
            except Exception as e:
                # Do not fail the whole eval artifact pipeline due to optional t-SNE,
                # but record the failure for defendability.
                graph_summary["tsne"] = {
                    "skipped": True,
                    "error": f"{type(e).__name__}: {e}",
                }
                log.warning("[embeddings] t-SNE graph plot failed for cond=%s: %s", str(cond), str(e))
        if ecfg.graph_save_metadata_json:
            gmeta = {
                "cond": str(cond),
                "writer": "acpl.eval.embeddings.write_embeddings (graph embeddings block)",
                "shape": {"K": int(K), "N": int(N), "D": int(D)},
                "pool": str(ecfg.graph_pool),
                "pmean_p": float(ecfg.graph_pmean_p),
                "l2_normalize": bool(ecfg.graph_l2_normalize),
                "plot": {
                    "max_points_for_plot": ecfg.graph_max_points_for_plot,
                    "downsample_seed": int(ecfg.graph_downsample_seed),
                    "point_size": float(ecfg.graph_plot_point_size),
                    "alpha": float(ecfg.graph_plot_alpha),
                    "annotate_points": bool(ecfg.graph_plot_annotate_points),
                    "color_label": ecfg.graph_color_label,
                    "color_bins": int(ecfg.graph_color_bins) if ecfg.graph_color_bins is not None else None,
                    "color_bin_strategy": str(ecfg.graph_color_bin_strategy),
                },
                "tsne_params": {
                    "seed": int(ecfg.tsne_seed),
                    "perplexity": ecfg.tsne_perplexity,
                    "learning_rate": ecfg.tsne_learning_rate,
                    "max_iter": int(ecfg.tsne_max_iter),
                    "init": str(ecfg.tsne_init),
                    "metric": str(ecfg.tsne_metric),
                    "method": str(ecfg.tsne_method),
                    "angle": float(ecfg.tsne_angle),
                },
                "files": {k: v for k, v in files.items() if k.startswith("graph_")},
                "summary": graph_summary,
            }
            gmeta_path = outdir / "graph_embeddings.meta.json"
            _save_json(gmeta_path, gmeta)
            files["graph_meta_json"] = str(gmeta_path)
        meta_payload.setdefault("graph_embeddings", {})
        meta_payload["graph_embeddings"].update(
            {
                "enabled": True,
                "pool": str(ecfg.graph_pool),
                "l2_normalize": bool(ecfg.graph_l2_normalize),
                "files": {k: v for k, v in files.items() if k.startswith("graph_")},
                "stability": graph_stability,
                "summary": graph_summary,
            }
        )

        # Update main meta json with graph block (rewrite)
        _save_json(meta_path, meta_payload)







    spectrum = mean_stats.get("spectrum", {}) if isinstance(mean_stats, dict) else {}
    return {
        "writer": "acpl.eval.embeddings.write_embeddings",
        "files": files,
        "shape": {"K": K, "N": N, "D": D},
        "nonfinite_n": nonfinite_n,
        "effective_rank": spectrum.get("effective_rank"),
        "evr_top2": (
            spectrum.get("explained_variance_ratio", [])[:2]
            if isinstance(spectrum.get("explained_variance_ratio"), list)
            else None
        ),
        "stability": stability,

        "graph_embeddings": {
            "enabled": bool(ecfg.make_graph_embeddings),
            "pool": str(ecfg.graph_pool),
            "files": {k: v for k, v in files.items() if k.startswith("graph_")},
        },



    }


# Alternate entrypoints discovered by scripts/eval.py
save_embeddings = write_embeddings
export_embeddings = write_embeddings

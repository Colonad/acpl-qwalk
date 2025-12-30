# acpl/eval/stats.py
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence



import numpy as np
import torch

__all__ = [
    # dataclasses
    "MetricSummary",
    "StatsEvalConfig",
    "ProbStatsConfig",
    "ProbStatsResult",
    # core summaries
    "summarize_samples",
    "summarize_tensor",
    "summarize_named_samples",
    "format_summary_table",
    "save_summary_json",
    # probability helpers
    "normalize_prob",
    "check_probability_simplex",
    "entropy",
    "gini_impurity",
    "total_variation",
    "kl_divergence",
    "js_divergence",
    "hellinger_distance",
    "l2_distance",
    "prob_mass_on_targets",
    "compute_prob_stats",
    "aggregate_prob_stats_over_time",
    # embedding helpers (for variance / overall stats)
    "covariance_matrix",
    "explained_variance",
    "embedding_basic_stats",
    "postprocess_stats",
    "finalize_stats",
    "augment_stats",


    # attribution / masking helpers (deltas + bootstrap CI)
    "BootstrapCI",
    "DeltaSummary",
    "bootstrap_ci",
    "paired_deltas",
    "summarize_delta",
    "summarize_named_deltas",
    "format_delta_table",
    "cohens_d",
    "hedges_g",


]


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class MetricSummary:
    """
    Summary statistics for a 1D sample of values.

    All fields are JSON-serializable.
    """

    n: int
    mean: float
    var: float
    std: float
    stderr: float
    min: float
    max: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StatsEvalConfig:
    """
    High-level evaluation stats configuration.

    This config is intentionally generic: it can summarize
    - scalar metric streams (loss, success, tv, etc.)
    - probability tensors (P over nodes)
    - embeddings (N,D) tensors
    """

    # Numerical
    eps: float = 1e-12
    float_dtype: torch.dtype = torch.float32

    # Quantiles
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

    # Formatting
    table_float_fmt: str = ".6f"
    table_max_rows: int = 50  # safety when printing huge metric dictionaries


@dataclass(frozen=True)
class ProbStatsConfig:
    """
    Configuration for probability-distribution metrics.

    Assumes probabilities live on last dim (N).
    Accepted shapes:
      - (N,)
      - (B,N)
      - (T,N)
      - (B,T,N)

    'targets' can be None, int, list[int], or LongTensor.
    """

    eps: float = 1e-12
    renormalize: bool = True



    # How to interpret a 2D tensor P with shape (A, N):
    # - "BN": treat as (B, N) batch of distributions (default for evaluation episodes)
    # - "TN": treat as (T, N) time-series for a single episode (will become (1, T, N))
    # - "auto": heuristic (safe default; ambiguous cases should be overridden explicitly)
    interpret_2d_as: Literal["BN", "TN", "auto"] = "auto"






    # Compare to uniform baseline U over nodes
    compare_to_uniform: bool = True

    # Metrics toggles
    compute_entropy: bool = True
    compute_gini: bool = True
    compute_maxp: bool = True

    compute_tv_vsU: bool = True
    compute_js_vsU: bool = True
    compute_hell_vsU: bool = True
    compute_l2_vsU: bool = True
    compute_kl_pu: bool = True  # KL(P || U)

    # If targets are provided
    compute_target_mass: bool = True


@dataclass
class ProbStatsResult:
    """
    Holds per-sample tensors and optionally aggregated summaries.

    raw: dict[str, torch.Tensor] where each tensor shape is (B,) or (B,T)
    summaries: dict[str, MetricSummary] computed over the flattened sample axis
    """

    raw: dict[str, torch.Tensor]
    summaries: dict[str, MetricSummary] | None = None


# =============================================================================
# Core helpers: summarization
# =============================================================================


def _to_1d_numpy(x: Any) -> np.ndarray:
    """
    Convert x to a 1D numpy array of float64.

    Accepts:
      - list/tuple of numbers
      - numpy arrays
      - torch tensors (any shape -> flattened)
    """
    if isinstance(x, torch.Tensor):
        if x.is_complex():
            x = x.real
        x = x.detach()
        if x.is_cuda:
            x = x.cpu()
        arr = x.to(torch.float64).reshape(-1).numpy()
        return arr

    if isinstance(x, np.ndarray):
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return arr

    # sequences
    arr = np.asarray(list(x), dtype=np.float64).reshape(-1)
    return arr


def summarize_samples(
    samples: Any,
    *,
    quantiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> MetricSummary:
    """
    Summarize a sample of scalar values into mean/var/std/stderr/min/max/quantiles.

    samples can be a list/np.ndarray/torch.Tensor (any shape; will be flattened).
    """
    x = _to_1d_numpy(samples)
    n = int(x.size)
    if n <= 0:
        # Avoid NaNs in downstream formatting; keep explicit emptiness.
        return MetricSummary(
            n=0,
            mean=float("nan"),
            var=float("nan"),
            std=float("nan"),
            stderr=float("nan"),
            min=float("nan"),
            max=float("nan"),
            q05=float("nan"),
            q25=float("nan"),
            q50=float("nan"),
            q75=float("nan"),
            q95=float("nan"),
        )

    mean = float(x.mean())
    var = float(x.var(ddof=0))
    std = float(math.sqrt(max(var, 0.0)))
    stderr = float(std / math.sqrt(n)) if n > 0 else float("nan")

    qs = np.quantile(x, list(quantiles))
    qmap = {float(q): float(v) for q, v in zip(quantiles, qs)}

    def _q(q: float) -> float:
        return qmap.get(q, float("nan"))

    return MetricSummary(
        n=n,
        mean=mean,
        var=var,
        std=std,
        stderr=stderr,
        min=float(x.min()),
        max=float(x.max()),
        q05=_q(0.05),
        q25=_q(0.25),
        q50=_q(0.50),
        q75=_q(0.75),
        q95=_q(0.95),
    )


def summarize_tensor(
    x: torch.Tensor,
    *,
    dim: int | None = None,
    keepdim: bool = False,
    cfg: StatsEvalConfig | None = None,
) -> dict[str, Any]:
    """
    Summarize a tensor either globally or along a dimension.

    Returns a JSON-serializable dict with:
      - shape
      - mean/var/std/min/max
      - (optional) per-dim arrays if dim is provided
    """
    if cfg is None:
        cfg = StatsEvalConfig()

    t = x
    if t.is_complex():
        t = t.real
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    t = t.to(cfg.float_dtype)

    out: dict[str, Any] = {"shape": list(t.shape)}

    if dim is None:
        out.update(
            {
                "mean": float(t.mean().item()),
                "var": float(t.var(unbiased=False).item()),
                "std": float(t.std(unbiased=False).item()),
                "min": float(t.min().item()),
                "max": float(t.max().item()),
            }
        )
        return out

    out.update(
        {
            "mean": t.mean(dim=dim, keepdim=keepdim).tolist(),
            "var": t.var(dim=dim, unbiased=False, keepdim=keepdim).tolist(),
            "std": t.std(dim=dim, unbiased=False, keepdim=keepdim).tolist(),
            "min": t.amin(dim=dim, keepdim=keepdim).tolist(),
            "max": t.amax(dim=dim, keepdim=keepdim).tolist(),
        }
    )
    return out


def summarize_named_samples(
    named_samples: Mapping[str, Any],
    *,
    cfg: StatsEvalConfig | None = None,
) -> dict[str, MetricSummary]:
    """
    Summarize many named metric samples into MetricSummary objects.
    """
    if cfg is None:
        cfg = StatsEvalConfig()

    out: dict[str, MetricSummary] = {}
    for k, v in named_samples.items():
        out[k] = summarize_samples(v, quantiles=cfg.quantiles)
    return out


def format_summary_table(
    summaries: Mapping[str, MetricSummary],
    *,
    cfg: StatsEvalConfig | None = None,
    sort_by: Literal["name", "mean", "std", "var", "n"] = "name",
    descending: bool = False,
) -> str:
    """
    Pretty-print a compact table for terminal logs.

    This intentionally avoids external deps (tabulate, pandas).
    """
    if cfg is None:
        cfg = StatsEvalConfig()

    items = list(summaries.items())

    key_fn = {
        "name": lambda kv: kv[0],
        "mean": lambda kv: kv[1].mean,
        "std": lambda kv: kv[1].std,
        "var": lambda kv: kv[1].var,
        "n": lambda kv: kv[1].n,
    }[sort_by]
    items.sort(key=key_fn, reverse=descending)

    # Safety limit for printing
    if len(items) > cfg.table_max_rows:
        items = items[: cfg.table_max_rows]

    fmt = "{:" + cfg.table_float_fmt + "}"

    header = (
        f"{'metric':<32}  {'n':>6}  {'mean':>12}  {'std':>12}  {'stderr':>12}  "
        f"{'min':>12}  {'q50':>12}  {'max':>12}"
    )
    lines = [header, "-" * len(header)]
    for name, s in items:
        lines.append(
            f"{name:<32}  {s.n:>6d}  "
            f"{fmt.format(s.mean):>12}  {fmt.format(s.std):>12}  {fmt.format(s.stderr):>12}  "
            f"{fmt.format(s.min):>12}  {fmt.format(s.q50):>12}  {fmt.format(s.max):>12}"
        )
    return "\n".join(lines)


def save_summary_json(path: str | Path, summaries: Mapping[str, MetricSummary]) -> None:
    """
    Save summaries as JSON.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.to_dict() for k, v in summaries.items()}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =============================================================================
# Probability helpers (robust, self-contained)
# =============================================================================


def normalize_prob(P: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize a probability-like tensor along the last dimension.

    P: (..., N)
    """
    if P.is_complex():
        P = P.real
    denom = P.sum(dim=-1, keepdim=True).clamp_min(eps)
    return P / denom


def check_probability_simplex(P: torch.Tensor, *, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    """
    Check if P is approximately a valid distribution along the last dimension:
      - nonnegative (within tolerance)
      - sums to 1 (within tolerance)

    Returns bool (does not throw).
    """
    if P.is_complex():
        P = P.real
    P = P.detach()
    # allow slight negatives from numerical noise
    min_ok = bool((P >= -atol).all().item())
    s = P.sum(dim=-1)
    sum_ok = bool(torch.allclose(s, torch.ones_like(s), atol=atol, rtol=rtol))
    return bool(min_ok and sum_ok)


def entropy(P: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Shannon entropy of distributions P along last dim.
    Returns shape (...,).
    """
    Pn = normalize_prob(P, eps=eps)
    return -(Pn * (Pn.clamp_min(eps).log())).sum(dim=-1)


def gini_impurity(P: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Gini impurity: 1 - sum_i p_i^2 along last dim.
    Returns shape (...,).
    """
    Pn = normalize_prob(P, eps=eps)
    return 1.0 - (Pn * Pn).sum(dim=-1)


def total_variation(P: torch.Tensor, Q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Total variation distance: 0.5 * ||P - Q||_1 along last dim.
    """
    Pn = normalize_prob(P, eps=eps)
    Qn = normalize_prob(Q, eps=eps)
    return 0.5 * (Pn - Qn).abs().sum(dim=-1)


def kl_divergence(P: torch.Tensor, Q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    KL(P || Q) along last dim.
    """
    Pn = normalize_prob(P, eps=eps)
    Qn = normalize_prob(Q, eps=eps)
    return (Pn * (Pn.clamp_min(eps).log() - Qn.clamp_min(eps).log())).sum(dim=-1)


def js_divergence(P: torch.Tensor, Q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Jensen–Shannon divergence JS(P,Q) = 0.5 KL(P||M) + 0.5 KL(Q||M), M=(P+Q)/2.
    """
    Pn = normalize_prob(P, eps=eps)
    Qn = normalize_prob(Q, eps=eps)
    M = 0.5 * (Pn + Qn)
    return 0.5 * kl_divergence(Pn, M, eps=eps) + 0.5 * kl_divergence(Qn, M, eps=eps)


def hellinger_distance(P: torch.Tensor, Q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Hellinger distance: sqrt(0.5 * sum_i (sqrt(p_i) - sqrt(q_i))^2).
    """
    Pn = normalize_prob(P, eps=eps).clamp_min(eps)
    Qn = normalize_prob(Q, eps=eps).clamp_min(eps)
    return torch.sqrt(0.5 * (torch.sqrt(Pn) - torch.sqrt(Qn)).pow(2).sum(dim=-1))


def l2_distance(P: torch.Tensor, Q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    L2 distance: sqrt(sum_i (p_i - q_i)^2).
    """
    Pn = normalize_prob(P, eps=eps)
    Qn = normalize_prob(Q, eps=eps)
    return torch.sqrt((Pn - Qn).pow(2).sum(dim=-1))


def _targets_to_index(
    targets: int | Sequence[int] | torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if targets is None:
        return None
    if isinstance(targets, int):
        return torch.tensor([targets], device=device, dtype=torch.long)
    if isinstance(targets, (list, tuple)):
        return torch.tensor(list(targets), device=device, dtype=torch.long)
    if isinstance(targets, torch.Tensor):
        if targets.ndim == 0:
            targets = targets.view(1)
        return targets.to(device=device, dtype=torch.long)
    raise TypeError(f"Unsupported targets type: {type(targets)}")


def prob_mass_on_targets(
    P: torch.Tensor,
    targets: int | Sequence[int] | torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Probability mass on a target set.
    Returns shape P.shape[:-1].
    """
    if P.is_complex():
        P = P.real
    Pn = normalize_prob(P, eps=eps)
    idx = _targets_to_index(targets, device=Pn.device)
    if idx is None:
        raise ValueError("targets is required.")
    return Pn.index_select(dim=-1, index=idx).sum(dim=-1)


# =============================================================================
# Probability stats: compute + time aggregation
# =============================================================================


def _as_prob_tensor(P: torch.Tensor, *, cfg: ProbStatsConfig) -> torch.Tensor:
    if P.is_complex():
        P = P.real
    P = P.to(torch.float32)
    if cfg.renormalize:
        P = normalize_prob(P, eps=cfg.eps)
    return P


def _uniform_like(P: torch.Tensor) -> torch.Tensor:
    # build U with same shape as P, constant along last dim
    N = int(P.shape[-1])
    return torch.full_like(P, fill_value=(1.0 / max(1, N)))


def compute_prob_stats(
    P: torch.Tensor,
    *,
    targets: int | Sequence[int] | torch.Tensor | None = None,
    cfg: ProbStatsConfig | None = None,
) -> ProbStatsResult:
    """
    Compute a suite of stats on probability tensors P over nodes.

    Supported shapes:
      - (N,) -> treated as (1,N)
      - (B,N)
      - (T,N) -> treated as (1,T,N)
      - (B,T,N)

    Returns:
      ProbStatsResult.raw: tensors shaped (B,) or (B,T)
    """
    if cfg is None:
        cfg = ProbStatsConfig()

    P0 = P
    if P0.ndim == 1:
        P0 = P0.unsqueeze(0)  # (1,N)
    elif P0.ndim == 2:
        # could be (B,N) or (T,N); make it explicit / robust.
        mode = cfg.interpret_2d_as
        if mode == "auto":
            # Heuristic:
            # - In suite eval, (B,N) is common with B=episodes (often >> 64).
            # - Time-series (T,N) tends to have small T (<=128) and N usually >= T.
            A, N = int(P0.shape[0]), int(P0.shape[1])
            if (A <= 128) and (N >= A) and (A != N):
                mode = "TN"
            else:
                mode = "BN"
        if mode == "TN":
            P0 = P0.unsqueeze(0)  # (1,T,N)
    
    
    
    
    
    
    
    
    elif P0.ndim == 3:
        # (B,T,N) expected
        pass
    else:
        raise ValueError(f"P must have shape (N), (B,N), (T,N), or (B,T,N). Got {tuple(P.shape)}")

    # Normalize to either (B,N) or (B,T,N)
    if P0.ndim == 2:
        Pn = _as_prob_tensor(P0, cfg=cfg)  # (B,N)
        raw_shape = "BT" if False else "B"
    else:
        # (B,T,N)
        Pn = _as_prob_tensor(P0, cfg=cfg)
        raw_shape = "BT"

    raw: dict[str, torch.Tensor] = {}

    # Baselines
    U = _uniform_like(Pn) if cfg.compare_to_uniform else None

    # Core metrics
    if cfg.compute_maxp:
        raw["maxp"] = Pn.max(dim=-1).values

    if cfg.compute_entropy:
        raw["H"] = entropy(Pn, eps=cfg.eps)

    if cfg.compute_gini:
        raw["gini"] = gini_impurity(Pn, eps=cfg.eps)

    # Uniform comparisons
    if cfg.compare_to_uniform and U is not None:
        if cfg.compute_tv_vsU:
            raw["tv_vsU"] = total_variation(Pn, U, eps=cfg.eps)
        if cfg.compute_js_vsU:
            raw["js_vsU"] = js_divergence(Pn, U, eps=cfg.eps)
        if cfg.compute_hell_vsU:
            raw["hell_vsU"] = hellinger_distance(Pn, U, eps=cfg.eps)
        if cfg.compute_l2_vsU:
            raw["l2_vsU"] = l2_distance(Pn, U, eps=cfg.eps)
        if cfg.compute_kl_pu:
            raw["KLpU"] = kl_divergence(Pn, U, eps=cfg.eps)

    # Target mass
    if cfg.compute_target_mass and targets is not None:
        raw["target_mass"] = prob_mass_on_targets(Pn, targets, eps=cfg.eps)

    # Enforce float32/float64 stability
    for k in list(raw.keys()):
        v = raw[k]
        if v.is_complex():
            v = v.real
        raw[k] = v.to(torch.float32)

    return ProbStatsResult(raw=raw, summaries=None)


def aggregate_prob_stats_over_time(
    stats: ProbStatsResult,
    *,
    mode: Literal["last", "mean", "max", "min"] = "last",
    time_index: int = -1,
) -> dict[str, torch.Tensor]:
    """
    If stats.raw contains tensors of shape (B,T), aggregate over T to produce (B,).

    If stats are already (B,), returns them unchanged.

    Returns dict[str, torch.Tensor] with (B,) tensors.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in stats.raw.items():
        if v.ndim == 1:
            out[k] = v
            continue
        if v.ndim != 2:
            raise ValueError(f"Expected stat tensor shape (B,) or (B,T). Got {k}: {tuple(v.shape)}")

        B, T = v.shape
        if mode == "last":
            idx = int(time_index)
            if idx < 0:
                idx = T + idx
            idx = max(0, min(T - 1, idx))
            out[k] = v[:, idx]
        elif mode == "mean":
            out[k] = v.mean(dim=1)
        elif mode == "max":
            out[k] = v.max(dim=1).values
        elif mode == "min":
            out[k] = v.min(dim=1).values
        else:
            raise ValueError(f"Unknown aggregation mode '{mode}'")
    return out


# =============================================================================
# Embedding stats helpers (variance + overall stats for node embeddings)
# =============================================================================


def covariance_matrix(
    emb: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute covariance matrix of embeddings.

    emb: (N,D)
    returns: (D,D)

    Notes:
      - Uses population normalization by N (not N-1) for stability and consistency.
    """
    if emb.ndim != 2:
        raise ValueError(f"covariance_matrix expects (N,D). Got {tuple(emb.shape)}")
    X = emb
    if X.is_complex():
        X = X.real
    X = X.to(torch.float64)
    if center:
        X = X - X.mean(dim=0, keepdim=True)
    N = int(X.shape[0])
    denom = float(max(1, N))
    C = (X.transpose(0, 1) @ X) / denom
    # numerical symmetry
    C = 0.5 * (C + C.transpose(0, 1))
    # tiny ridge to avoid singular issues in downstream eig
    C = C + torch.eye(C.shape[0], dtype=C.dtype) * float(eps)
    return C


def explained_variance(
    emb: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Compute eigen-spectrum summary of embedding covariance: eigenvalues, EVR, effective rank.

    Returns JSON-serializable dict.
    """
    C = covariance_matrix(emb, center=center, eps=eps)
    evals = torch.linalg.eigvalsh(C).clamp_min(0.0)  # (D,)
    # sort descending
    evals = torch.sort(evals, descending=True).values
    total = float(evals.sum().item())
    if total <= 0:
        evr = torch.zeros_like(evals)
    else:
        evr = evals / total

    # effective rank: exp(H(p)) where p = evals / sum(evals)
    p = evr.clamp_min(1e-18)
    H = float(-(p * p.log()).sum().item())
    eff_rank = float(math.exp(H))

    return {
        "D": int(evals.numel()),
        "total_variance": total,
        "eigenvalues": evals.detach().cpu().tolist(),
        "explained_variance_ratio": evr.detach().cpu().tolist(),
        "effective_rank": eff_rank,
    }


def embedding_basic_stats(emb: torch.Tensor, *, cfg: StatsEvalConfig | None = None) -> dict[str, Any]:
    """
    "Overall statistics" for node embeddings, research-grade + finite-safe.

    Returns:
      - shape
      - global scalar mean/var/std/min/max over all entries (finite-masked)
      - per-dimension mean/var/std/min/max (finite-masked, per-dim denominators)
      - per-node norm summary (computed on finite-masked embeddings)
      - covariance spectrum summary (variance/eigenvalues) on finite-masked embeddings

    Notes
    -----
    - If embeddings contain NaN/Inf, we DO NOT silently pretend everything is fine:
        * we report nonfinite counts + fractions
        * we compute stats on the finite subset only
        * we use a zero-filled masked copy for norms/spectrum so those remain defined
    - This makes evaluation artifacts defendable and reproducible.
    """
    if cfg is None:
        cfg = StatsEvalConfig()

    if emb.ndim != 2:
        raise ValueError(f"embedding_basic_stats expects (N,D). Got {tuple(emb.shape)}")

    X = emb
    if X.is_complex():
        X = X.real
    X = X.detach()
    if X.is_cuda:
        X = X.cpu()

    # Use float64 internally for stable masked moments, but keep cfg.float_dtype for outputs if desired.
    # (The returned numbers are Python floats/lists anyway.)
    X = X.to(torch.float64)

    N, D = X.shape

    finite = torch.isfinite(X)  # (N,D) bool
    finite_n = int(finite.sum().item())
    total_n = int(X.numel())
    nonfinite_n = int(total_n - finite_n)
    finite_fraction = float(finite_n / max(1, total_n))

    # Zero-filled masked copy (safe for norms/spectrum; moments computed with explicit denominators below).
    Xf = torch.where(finite, X, torch.zeros_like(X))

    # ----------------------------
    # Global moments on finite set
    # ----------------------------
    denom_g = float(max(1, finite_n))
    sum_g = float(Xf.sum().item())
    sumsq_g = float((Xf * Xf).sum().item())

    mean_g = sum_g / denom_g if finite_n > 0 else float("nan")
    ex2_g = sumsq_g / denom_g if finite_n > 0 else float("nan")
    var_g = max(0.0, ex2_g - mean_g * mean_g) if finite_n > 0 else float("nan")
    std_g = math.sqrt(var_g) if finite_n > 0 else float("nan")

    # Finite-only min/max
    if finite_n > 0:
        x_min = float(torch.where(finite, X, torch.full_like(X, float("inf"))).min().item())
        x_max = float(torch.where(finite, X, torch.full_like(X, float("-inf"))).max().item())
        abs_mean = float(Xf.abs().sum().item() / denom_g)
        abs_max = float(Xf.abs().max().item())
    else:
        x_min = float("nan")
        x_max = float("nan")
        abs_mean = float("nan")
        abs_max = float("nan")

    # -----------------------------------
    # Per-dimension finite-masked moments
    # -----------------------------------
    finite_d = finite.sum(dim=0)  # (D,)
    denom_d = finite_d.to(torch.float64).clamp_min(1.0)  # avoid divide-by-zero
    sum_d = Xf.sum(dim=0)  # (D,)
    sumsq_d = (Xf * Xf).sum(dim=0)  # (D,)

    mean_d = sum_d / denom_d
    ex2_d = sumsq_d / denom_d
    var_d = (ex2_d - mean_d * mean_d).clamp_min(0.0)
    std_d = torch.sqrt(var_d)

    # Mark dims with zero finite entries as NaN
    zero_d = (finite_d == 0)
    if bool(zero_d.any().item()):
        nanv = torch.full_like(mean_d, float("nan"))
        mean_d = torch.where(zero_d, nanv, mean_d)
        var_d = torch.where(zero_d, nanv, var_d)
        std_d = torch.where(zero_d, nanv, std_d)

    # Per-dimension min/max on finite set
    if finite_n > 0:
        min_d = torch.where(finite, X, torch.full_like(X, float("inf"))).amin(dim=0)
        max_d = torch.where(finite, X, torch.full_like(X, float("-inf"))).amax(dim=0)
        # Dims with no finite entries -> NaN
        if bool(zero_d.any().item()):
            min_d = torch.where(zero_d, torch.full_like(min_d, float("nan")), min_d)
            max_d = torch.where(zero_d, torch.full_like(max_d, float("nan")), max_d)
    else:
        min_d = torch.full((D,), float("nan"), dtype=torch.float64)
        max_d = torch.full((D,), float("nan"), dtype=torch.float64)

    # -----------------------------------
    # Node-level diagnostics + norms
    # -----------------------------------
    row_all_finite = finite.all(dim=1)  # (N,)
    finite_rows_n = int(row_all_finite.sum().item())
    nonfinite_rows_n = int(N - finite_rows_n)
    finite_rows_fraction = float(finite_rows_n / max(1, N))

    # Norms computed on masked embeddings; report how many rows were non-finite.
    norms = torch.linalg.norm(Xf, dim=1)  # (N,)

    # Cast back to cfg.float_dtype for spectrum/norm summary stability if you prefer
    Xf_out = Xf.to(cfg.float_dtype)

    stats: dict[str, Any] = {
        "shape": [int(N), int(D)],
        "global": {
            # original keys (but now finite-masked)
            "mean": float(mean_g),
            "var": float(var_g),
            "std": float(std_g),
            "min": float(x_min),
            "max": float(x_max),
            # research-grade additions
            "abs_mean": float(abs_mean),
            "abs_max": float(abs_max),
            "finite_fraction": float(finite_fraction),
            "nonfinite_n": int(nonfinite_n),
            "finite_n": int(finite_n),
            "total_n": int(total_n),
        },
        "per_dim": {
            # original keys
            "mean": mean_d.to(torch.float32).tolist(),
            "var": var_d.to(torch.float32).tolist(),
            "std": std_d.to(torch.float32).tolist(),
            "min": min_d.to(torch.float32).tolist(),
            "max": max_d.to(torch.float32).tolist(),
            # additions (defendability)
            "finite_n": finite_d.to(torch.int64).tolist(),
            "nonfinite_n": (N - finite_d).to(torch.int64).tolist(),
            "finite_fraction": (finite_d.to(torch.float64) / float(max(1, N))).tolist(),
        },
        "nodes": {
            "finite_rows_n": int(finite_rows_n),
            "nonfinite_rows_n": int(nonfinite_rows_n),
            "finite_rows_fraction": float(finite_rows_fraction),
        },
        "norms": {
            **summarize_samples(norms, quantiles=cfg.quantiles).to_dict(),
            # additions (so norms aren’t misread if many rows had NaN/Inf originally)
            "nonfinite_rows_n": int(nonfinite_rows_n),
            "finite_rows_n": int(finite_rows_n),
        },
        # Spectrum computed on the finite-masked copy (zero-filled for non-finite)
        "spectrum": explained_variance(Xf_out, center=True, eps=cfg.eps),
    }
    return stats




# =============================================================================
# Stats post-processing + merging helpers (first-class artifacts)
# =============================================================================

def postprocess_stats(
    obj: Any,
    *,
    nonfinite: Literal["keep", "null", "string", "raise"] = "keep",
    _depth: int = 0,
    _max_depth: int = 64,
) -> Any:
    """
    Recursively convert stats objects into JSON-friendly Python types.

    Handles:
      - torch.Tensor -> list / scalar
      - numpy arrays/scalars -> list / scalar
      - Path -> str
      - dataclasses -> dict
      - NaN/Inf policy via `nonfinite`

    nonfinite policies:
      - "keep":   keep float('nan')/inf as-is (Python json allows by default, but non-standard JSON)
      - "null":   convert NaN/Inf to None
      - "string": convert NaN/Inf to "nan"/"inf"/"-inf"
      - "raise":  raise ValueError if any NaN/Inf encountered
    """
    if _depth > _max_depth:
        raise RecursionError("postprocess_stats exceeded maximum recursion depth.")

    # dataclass -> dict
    try:
        # don't import dataclasses at top; keep local to avoid overhead
        import dataclasses
        if dataclasses.is_dataclass(obj):
            obj = dataclasses.asdict(obj)
    except Exception:
        pass

    # Path -> str
    if isinstance(obj, Path):
        return str(obj)

    # torch.Tensor -> python
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        if t.is_complex():
            t = t.real
        if t.is_cuda:
            t = t.cpu()
        if t.numel() == 1:
            return postprocess_stats(t.item(), nonfinite=nonfinite, _depth=_depth + 1)
        return postprocess_stats(t.tolist(), nonfinite=nonfinite, _depth=_depth + 1)

    # numpy -> python
    if isinstance(obj, np.ndarray):
        return postprocess_stats(obj.tolist(), nonfinite=nonfinite, _depth=_depth + 1)
    if isinstance(obj, np.generic):
        return postprocess_stats(obj.item(), nonfinite=nonfinite, _depth=_depth + 1)

    # basic scalars
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj

    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        if nonfinite == "keep":
            return obj
        if nonfinite == "null":
            return None
        if nonfinite == "string":
            if math.isnan(obj):
                return "nan"
            return "inf" if obj > 0 else "-inf"
        raise ValueError(f"Non-finite float encountered: {obj!r}")

    # mappings
    from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
    if isinstance(obj, ABCMapping):
        return {str(k): postprocess_stats(v, nonfinite=nonfinite, _depth=_depth + 1) for k, v in obj.items()}

    # sequences (but not strings)
    if isinstance(obj, ABCSequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [postprocess_stats(v, nonfinite=nonfinite, _depth=_depth + 1) for v in obj]

    # fallback: try to stringify (last resort)
    return str(obj)


def augment_stats(
    base: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """
    Deep-merge `extra` into `base` (both treated as dict-like).

    - If both values are dicts, merges recursively.
    - Otherwise overwrites if `overwrite=True`; else keeps base.
    """
    out: dict[str, Any] = dict(base or {})
    if not extra:
        return out

    for k, v in extra.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = augment_stats(out[k], v, overwrite=overwrite)
        else:
            if overwrite or (k not in out):
                out[k] = v
    return out


def finalize_stats(
    stats: Mapping[str, Any],
    *,
    nonfinite: Literal["keep", "null", "string", "raise"] = "keep",
    add_meta: bool = True,
) -> dict[str, Any]:
    """
    Finalize a stats payload for artifact writing:
      - deep converts to JSON-safe types (postprocess_stats)
      - optionally adds a small meta block with counts/notes

    This is intended to be called right before writing JSON artifacts.
    """
    processed = postprocess_stats(stats, nonfinite=nonfinite)

    if not isinstance(processed, dict):
        processed = {"value": processed}

    if add_meta:
        # Lightweight meta: helps provenance without forcing callers to remember it.
        meta = {
            "writer": "acpl.eval.stats.finalize_stats",
            "nonfinite_policy": str(nonfinite),
        }
        processed = augment_stats(processed, {"_finalize_meta": meta}, overwrite=False)

    return processed

# =============================================================================
# Attribution / masking helpers: deltas + bootstrap CI
# =============================================================================

@dataclass(frozen=True)
class BootstrapCI:
    """
    Bootstrap confidence interval for a scalar statistic.

    All fields are JSON-serializable.

    Notes:
      - `stat` is the statistic on the *original* (finite-filtered) sample.
      - CI bounds are computed from bootstrap replicates.
      - If there are insufficient finite samples, fields may be NaN.
    """

    n: int
    stat: float
    low: float
    high: float
    alpha: float
    method: Literal["percentile", "bca"]
    n_boot: int
    seed: int
    finite_n: int
    dropped_nonfinite_n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeltaSummary:
    """
    Paired/unpaired delta summary between a baseline sample and a perturbed/masked sample.

    This is the "substructure masking analogue" helper:
      - you run eval normally -> baseline metrics
      - you run eval under a mask -> masked metrics
      - this object summarizes the delta + CI for defendable reporting.

    Fields:
      base: MetricSummary over baseline sample
      pert: MetricSummary over perturbed sample
      delta: MetricSummary over (pert - base) if paired, else difference of means (as a 1-sample)
      ci_mean_delta: BootstrapCI on mean(delta) (paired) or mean(pert) - mean(base) (unpaired via
                    bootstrap on each sample + differencing)  [implemented as paired delta when possible]
      rel_change_mean: (mean(pert) - mean(base)) / max(|mean(base)|, eps)
    """

    paired: bool
    base: MetricSummary
    pert: MetricSummary
    delta: MetricSummary
    ci_mean_delta: BootstrapCI | None
    rel_change_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "paired": bool(self.paired),
            "base": self.base.to_dict(),
            "pert": self.pert.to_dict(),
            "delta": self.delta.to_dict(),
            "ci_mean_delta": None if self.ci_mean_delta is None else self.ci_mean_delta.to_dict(),
            "rel_change_mean": float(self.rel_change_mean),
        }


# ------------------------------
# Internal numeric utilities
# ------------------------------

def _finite_1d(x: Any) -> tuple[np.ndarray, int]:
    """
    Convert to 1D float64 numpy array and filter non-finite values.

    Returns: (finite_values, dropped_nonfinite_count)
    """
    arr = _to_1d_numpy(x)
    finite_mask = np.isfinite(arr)
    dropped = int((~finite_mask).sum())
    return arr[finite_mask], dropped


def _mean_np(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else float("nan")


def _norm_cdf(z: float) -> float:
    # Standard normal CDF via erf
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (probit), Acklam-style rational approximation.

    Valid for p in (0,1). Raises for invalid p.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")

    # Coefficients from Peter J. Acklam's approximation
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den

    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def _bootstrap_reps(
    x: np.ndarray,
    *,
    stat_fn: callable,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(x.size)
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    idx = rng.integers(0, n, size=(int(n_boot), n), endpoint=False)
    xb = x[idx]  # (B, n)
    # apply statistic along axis 1
    return np.asarray([stat_fn(row) for row in xb], dtype=np.float64)


def _jackknife_stats(x: np.ndarray, *, stat_fn: callable) -> np.ndarray:
    n = int(x.size)
    if n <= 1:
        return np.asarray([], dtype=np.float64)
    out = np.empty((n,), dtype=np.float64)
    for i in range(n):
        out[i] = float(stat_fn(np.delete(x, i)))
    return out


def bootstrap_ci(
    samples: Any,
    *,
    stat: Literal["mean", "median"] | callable = "mean",
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    method: Literal["percentile", "bca"] = "bca",
) -> BootstrapCI:
    """
    Bootstrap CI for a scalar statistic on a 1D sample.

    - Filters non-finite values first (NaN/Inf).
    - Supports percentile and BCa intervals.
    - Defaults to BCa (more defendable under skew / bias).

    Returns BootstrapCI; if insufficient finite samples, CI bounds are NaN.
    """
    x, dropped = _finite_1d(samples)
    n = int(_to_1d_numpy(samples).size)
    finite_n = int(x.size)

    if isinstance(stat, str):
        if stat == "mean":
            stat_fn = lambda a: float(np.mean(a)) if a.size > 0 else float("nan")
        elif stat == "median":
            stat_fn = lambda a: float(np.median(a)) if a.size > 0 else float("nan")
        else:
            raise ValueError(f"Unknown stat '{stat}'. Use 'mean', 'median', or a callable.")
        stat_name = stat
    else:
        stat_fn = stat
        stat_name = getattr(stat, "__name__", "stat")

    if finite_n <= 0:
        return BootstrapCI(
            n=n,
            stat=float("nan"),
            low=float("nan"),
            high=float("nan"),
            alpha=float(alpha),
            method=method,
            n_boot=int(n_boot),
            seed=int(seed),
            finite_n=int(finite_n),
            dropped_nonfinite_n=int(dropped),
        )

    theta_hat = float(stat_fn(x))

    # If finite_n == 1, bootstrap is degenerate; CI is undefined -> NaN bounds
    if finite_n <= 1 or n_boot <= 0:
        return BootstrapCI(
            n=n,
            stat=float(theta_hat),
            low=float("nan"),
            high=float("nan"),
            alpha=float(alpha),
            method=method,
            n_boot=int(n_boot),
            seed=int(seed),
            finite_n=int(finite_n),
            dropped_nonfinite_n=int(dropped),
        )

    rng = np.random.default_rng(int(seed))
    reps = _bootstrap_reps(x, stat_fn=stat_fn, n_boot=int(n_boot), rng=rng)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return BootstrapCI(
            n=n,
            stat=float(theta_hat),
            low=float("nan"),
            high=float("nan"),
            alpha=float(alpha),
            method=method,
            n_boot=int(n_boot),
            seed=int(seed),
            finite_n=int(finite_n),
            dropped_nonfinite_n=int(dropped),
        )

    reps.sort()
    lo_p = float(alpha / 2.0)
    hi_p = float(1.0 - alpha / 2.0)

    def _pct(p: float) -> float:
        # robust percentile with interpolation
        return float(np.quantile(reps, p, method="linear"))

    if method == "percentile":
        low = _pct(lo_p)
        high = _pct(hi_p)
        return BootstrapCI(
            n=n,
            stat=float(theta_hat),
            low=float(low),
            high=float(high),
            alpha=float(alpha),
            method="percentile",
            n_boot=int(n_boot),
            seed=int(seed),
            finite_n=int(finite_n),
            dropped_nonfinite_n=int(dropped),
        )

    if method != "bca":
        raise ValueError(f"Unknown bootstrap method '{method}' (use 'percentile' or 'bca').")

    # BCa interval
    # Bias-correction z0 from bootstrap distribution
    prop_less = float(np.mean(reps < theta_hat))
    # Guard against 0/1 which would send ppf to inf
    prop_less = min(max(prop_less, 1e-12), 1.0 - 1e-12)
    z0 = _norm_ppf(prop_less)

    # Acceleration a from jackknife
    jack = _jackknife_stats(x, stat_fn=stat_fn)
    if jack.size == 0 or not np.isfinite(jack).all():
        a = 0.0
    else:
        jack_mean = float(np.mean(jack))
        num = float(np.sum((jack_mean - jack) ** 3))
        den = float(6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5))
        a = (num / den) if den > 0 else 0.0

    def _bca_adj(p: float) -> float:
        z = _norm_ppf(p)
        adj = _norm_cdf(z0 + (z0 + z) / max(1e-12, (1.0 - a * (z0 + z))))
        # clamp
        return float(min(max(adj, 1e-12), 1.0 - 1e-12))

    lo_adj = _bca_adj(lo_p)
    hi_adj = _bca_adj(hi_p)

    low = _pct(lo_adj)
    high = _pct(hi_adj)

    return BootstrapCI(
        n=n,
        stat=float(theta_hat),
        low=float(low),
        high=float(high),
        alpha=float(alpha),
        method="bca",
        n_boot=int(n_boot),
        seed=int(seed),
        finite_n=int(finite_n),
        dropped_nonfinite_n=int(dropped),
    )


def paired_deltas(base: Any, pert: Any) -> np.ndarray:
    """
    Compute paired deltas (pert - base) after converting to 1D arrays.

    Rules:
      - Both are flattened to 1D
      - Non-finite entries are dropped *pairwise* (same indices removed from both)
      - Lengths must match before filtering
    """
    a = _to_1d_numpy(base)
    b = _to_1d_numpy(pert)
    if a.size != b.size:
        raise ValueError(f"Paired deltas require equal sizes. Got {a.size} vs {b.size}.")
    finite = np.isfinite(a) & np.isfinite(b)
    return (b[finite] - a[finite]).astype(np.float64, copy=False)


def cohens_d(delta: Any, *, ddof: int = 1) -> float:
    """
    Cohen's d for paired deltas: mean(delta) / std(delta).

    Returns NaN if insufficient samples or zero variance.
    """
    x, _ = _finite_1d(delta)
    n = int(x.size)
    if n <= 1:
        return float("nan")
    sd = float(np.std(x, ddof=ddof))
    if sd <= 0:
        return float("nan")
    return float(np.mean(x) / sd)


def hedges_g(delta: Any, *, ddof: int = 1) -> float:
    """
    Hedges' g for paired deltas (small-sample correction of Cohen's d).

    g = J(n) * d, with J(n) = 1 - 3/(4n - 9)
    """
    x, _ = _finite_1d(delta)
    n = int(x.size)
    d = cohens_d(x, ddof=ddof)
    if not math.isfinite(d) or n <= 1:
        return float("nan")
    J = 1.0 - (3.0 / max(1.0, (4.0 * n - 9.0)))
    return float(J * d)


def summarize_delta(
    base: Any,
    pert: Any,
    *,
    paired: bool = True,
    eps: float = 1e-12,
    ci: bool = True,
    ci_alpha: float = 0.05,
    ci_n_boot: int = 2000,
    ci_seed: int = 0,
    ci_method: Literal["percentile", "bca"] = "bca",
    cfg: StatsEvalConfig | None = None,
) -> DeltaSummary:
    """
    Summarize baseline vs perturbed samples.

    - If paired=True, computes deltas elementwise (pert - base) with pairwise finite filtering.
    - Computes MetricSummary for base, pert, delta.
    - Optionally computes BootstrapCI for mean(delta).

    This is the helper you want to call from evaluation when you compare:
      ckpt_policy vs masked_policy
      baseline vs ablation
      clean vs disorder
    """
    if cfg is None:
        cfg = StatsEvalConfig()

    base_sum = summarize_samples(base, quantiles=cfg.quantiles)
    pert_sum = summarize_samples(pert, quantiles=cfg.quantiles)

    if paired:
        d = paired_deltas(base, pert)
        delta_sum = summarize_samples(d, quantiles=cfg.quantiles)
        ci_obj = (
            bootstrap_ci(
                d,
                stat="mean",
                alpha=ci_alpha,
                n_boot=ci_n_boot,
                seed=ci_seed,
                method=ci_method,
            )
            if ci
            else None
        )
    else:
        # Unpaired: still report a "delta sample" as difference of means (single value),
        # but for CI we bootstrap the two means and subtract them.
        a, da = _finite_1d(base)
        b, db = _finite_1d(pert)

        delta_mean = _mean_np(b) - _mean_np(a)
        delta_sum = summarize_samples([delta_mean], quantiles=cfg.quantiles)

        ci_obj = None
        if ci:
            rng = np.random.default_rng(int(ci_seed))
            # bootstrap mean(a), mean(b) independently then subtract
            rep_a = _bootstrap_reps(a, stat_fn=_mean_np, n_boot=int(ci_n_boot), rng=rng)
            rep_b = _bootstrap_reps(b, stat_fn=_mean_np, n_boot=int(ci_n_boot), rng=rng)
            reps = (rep_b - rep_a)
            reps = reps[np.isfinite(reps)]
            if reps.size > 0:
                reps.sort()
                lo = float(np.quantile(reps, ci_alpha / 2.0, method="linear"))
                hi = float(np.quantile(reps, 1.0 - ci_alpha / 2.0, method="linear"))
                ci_obj = BootstrapCI(
                    n=int(_to_1d_numpy(base).size),
                    stat=float(delta_mean),
                    low=float(lo),
                    high=float(hi),
                    alpha=float(ci_alpha),
                    method="percentile",
                    n_boot=int(ci_n_boot),
                    seed=int(ci_seed),
                    finite_n=int(min(a.size, b.size)),
                    dropped_nonfinite_n=int(da + db),
                )

    rel = float((pert_sum.mean - base_sum.mean) / max(abs(base_sum.mean), float(eps))) if base_sum.n > 0 else float("nan")

    return DeltaSummary(
        paired=bool(paired),
        base=base_sum,
        pert=pert_sum,
        delta=delta_sum,
        ci_mean_delta=ci_obj,
        rel_change_mean=rel,
    )


def summarize_named_deltas(
    base_metrics: Mapping[str, Any],
    pert_metrics: Mapping[str, Any],
    *,
    paired: bool = True,
    eps: float = 1e-12,
    ci: bool = True,
    ci_alpha: float = 0.05,
    ci_n_boot: int = 2000,
    ci_seed: int = 0,
    ci_method: Literal["percentile", "bca"] = "bca",
    cfg: StatsEvalConfig | None = None,
    only_common_keys: bool = True,
) -> dict[str, DeltaSummary]:
    """
    Compute DeltaSummary for many metrics in parallel.

    If only_common_keys=True, only keys present in both mappings are used.
    Otherwise, missing keys raise KeyError.
    """
    if cfg is None:
        cfg = StatsEvalConfig()

    keys = set(base_metrics.keys())
    if only_common_keys:
        keys = keys.intersection(set(pert_metrics.keys()))
    else:
        missing = keys.difference(set(pert_metrics.keys()))
        if missing:
            raise KeyError(f"pert_metrics missing keys: {sorted(missing)}")

    out: dict[str, DeltaSummary] = {}
    for k in sorted(keys):
        out[k] = summarize_delta(
            base_metrics[k],
            pert_metrics[k],
            paired=paired,
            eps=eps,
            ci=ci,
            ci_alpha=ci_alpha,
            ci_n_boot=ci_n_boot,
            ci_seed=ci_seed,
            ci_method=ci_method,
            cfg=cfg,
        )
    return out


def format_delta_table(
    deltas: Mapping[str, DeltaSummary],
    *,
    cfg: StatsEvalConfig | None = None,
    sort_by: Literal["name", "delta_mean", "abs_delta_mean", "rel_change"] = "name",
    descending: bool = False,
    show_ci: bool = True,
) -> str:
    """
    Compact terminal table for delta summaries (good for CI logs and artifact previews).
    """
    if cfg is None:
        cfg = StatsEvalConfig()

    items = list(deltas.items())

    def key_fn(kv: tuple[str, DeltaSummary]) -> Any:
        name, ds = kv
        if sort_by == "name":
            return name
        if sort_by == "delta_mean":
            return ds.delta.mean
        if sort_by == "abs_delta_mean":
            return abs(ds.delta.mean)
        if sort_by == "rel_change":
            return ds.rel_change_mean
        raise ValueError(f"Unknown sort_by={sort_by!r}")

    items.sort(key=key_fn, reverse=descending)
    if len(items) > cfg.table_max_rows:
        items = items[: cfg.table_max_rows]

    fmt = "{:" + cfg.table_float_fmt + "}"

    if show_ci:
        header = (
            f"{'metric':<28}  {'base_mean':>12}  {'pert_mean':>12}  "
            f"{'d_mean':>12}  {'d_CI_low':>12}  {'d_CI_high':>12}  {'rel%':>10}"
        )
    else:
        header = (
            f"{'metric':<28}  {'base_mean':>12}  {'pert_mean':>12}  "
            f"{'d_mean':>12}  {'rel%':>10}"
        )

    lines = [header, "-" * len(header)]
    for name, ds in items:
        rel_pct = 100.0 * ds.rel_change_mean if math.isfinite(ds.rel_change_mean) else float("nan")
        if show_ci and ds.ci_mean_delta is not None:
            lines.append(
                f"{name:<28}  "
                f"{fmt.format(ds.base.mean):>12}  {fmt.format(ds.pert.mean):>12}  "
                f"{fmt.format(ds.delta.mean):>12}  "
                f"{fmt.format(ds.ci_mean_delta.low):>12}  {fmt.format(ds.ci_mean_delta.high):>12}  "
                f"{fmt.format(rel_pct):>10}"
            )
        else:
            lines.append(
                f"{name:<28}  "
                f"{fmt.format(ds.base.mean):>12}  {fmt.format(ds.pert.mean):>12}  "
                f"{fmt.format(ds.delta.mean):>12}  "
                f"{fmt.format(rel_pct):>10}"
            )

    return "\n".join(lines)


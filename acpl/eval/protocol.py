# acpl/eval/protocol.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import math
import random

import torch
import torch.nn as nn

from acpl.train.loops import LoopConfig  # only for device & UI knobs
from acpl.train.loops import RolloutFn  # Callable[[nn.Module, dict], tuple[torch.Tensor, dict]]
from acpl.train.loops import build_metric_pack
from acpl.utils.logging import MetricLogger

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x


__all__ = [
    "EvalConfig",
    "CISummary",
    "compute_ci",
    "run_ci_eval",
    "summarize_results",
]


# --------------------------------------------------------------------------------------
#                                      Config
# --------------------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """
    CI-style evaluation across seeds.

    Attributes
    ----------
    seeds:
        Explicit RNG seeds to evaluate. If empty and n_seeds>0, we synthesize
        deterministic seeds [0..n_seeds-1].
    n_seeds:
        Used only when `seeds` is empty.
    device:
        Torch device string for model & tensors.
    progress_bar:
        Whether to show tqdm bars.
    ci_method:
        "student" or "bootstrap".
    ci_alpha:
        1 - confidence level (e.g., 0.05 -> 95% CI).
    bootstrap_samples:
        Number of bootstrap resamples if ci_method="bootstrap".
    keep_per_seed_means:
        Include per-seed means in the returned dict (for tables).
    """

    seeds: list[int] = field(default_factory=list)
    n_seeds: int = 10
    device: str = "cuda"
    progress_bar: bool = True

    ci_method: str = "bootstrap"  # "student" or "bootstrap"
    ci_alpha: float = 0.05
    bootstrap_samples: int = 2000

    keep_per_seed_means: bool = True


# --------------------------------------------------------------------------------------
#                           CI computation (Student / Bootstrap)
# --------------------------------------------------------------------------------------


@dataclass
class CISummary:
    mean: float
    lo: float
    hi: float
    stderr: float
    n: int


def _student_interval(
    samples: torch.Tensor, alpha: float
) -> tuple[float, float, float, float, int]:
    """
    Student-t (or normal for large n) CI on the mean.

    Returns
    -------
    (mean, lo, hi, stderr, n)
    """
    if samples.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    x = samples.to(torch.float64)
    n = int(x.numel())
    mean = float(x.mean().item())

    # Unbiased sample std; handle n==1
    s = float(x.std(unbiased=True).item()) if n > 1 else 0.0
    stderr = s / math.sqrt(max(n, 1))

    # For small n, use Student-t; for large n, z is fine.
    # We avoid scipy; use Normal icdf as an approximation to t-quantile when n>30.
    # For n<=30, inflate by sqrt((n-1)/(n-3)) heuristically if n>3 (conservative).
    z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2)).item())
    if 3 < n <= 30:
        z *= math.sqrt((n - 1) / (n - 3))

    half_width = z * stderr
    return mean, mean - half_width, mean + half_width, stderr, n


def _bootstrap_interval(
    samples: torch.Tensor,
    alpha: float,
    B: int,
) -> tuple[float, float, float, float, int]:
    """
    Percentile bootstrap CI on the mean, with B resamples.

    Returns
    -------
    (mean, lo, hi, stderr, n)
    """
    if samples.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    x = samples.to(torch.float64)
    n = int(x.numel())
    mean = float(x.mean().item())

    if n == 1:
        return mean, mean, mean, 0.0, 1

    # Bootstrap resamples of the mean
    idx = torch.randint(0, n, size=(B, n), device=x.device)
    boot_means = x[idx].mean(dim=1)
    lo = float(torch.quantile(boot_means, q=torch.tensor(alpha / 2, dtype=torch.float64)).item())
    hi = float(
        torch.quantile(boot_means, q=torch.tensor(1 - alpha / 2, dtype=torch.float64)).item()
    )

    # Bootstrap estimate of stderr (std of the bootstrap distribution)
    stderr = float(boot_means.std(unbiased=True).item())
    return mean, lo, hi, stderr, n


def compute_ci(
    samples: torch.Tensor,
    *,
    method: str = "bootstrap",
    alpha: float = 0.05,
    bootstrap_samples: int = 2000,
) -> CISummary:
    """
    Compute mean ± CI for a 1-D tensor of samples.
    """
    method = method.lower()
    if method == "student":
        m, lo, hi, se, n = _student_interval(samples, alpha)
    elif method == "bootstrap":
        m, lo, hi, se, n = _bootstrap_interval(samples, alpha, bootstrap_samples)
    else:
        raise ValueError(f"Unknown ci_method: {method}")
    return CISummary(mean=m, lo=lo, hi=hi, stderr=se, n=n)


# --------------------------------------------------------------------------------------
#                          CI-style evaluation over seeds
# --------------------------------------------------------------------------------------

# Episode metric pack callable type: f(P, **aux) -> Dict[str, float]
MetricPack = dict[str, Callable[[torch.Tensor], dict[str, float]]]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)  # no-op on CPU-only
    except Exception:
        pass


def run_ci_eval(
    *,
    model: nn.Module,
    dataloader_factory: Callable[[int], Iterable[dict]],
    rollout_fn: RolloutFn,
    loop_cfg: LoopConfig | None = None,
    eval_cfg: EvalConfig | None = None,
    logger: MetricLogger | None = None,
    step: int | None = None,
) -> dict[str, dict[str, CISummary]]:
    """
    Evaluate a *fixed model* across multiple dataset seeds, aggregate per-episode metrics,
    and return mean ± CI for each scalar metric.

    Parameters
    ----------
    model:
        The trained model (already on the right device).
    dataloader_factory:
        Callable that receives a seed and returns an iterable of batches (dicts).
        Each batch is fed to `rollout_fn(model, batch)`.
    rollout_fn:
        Callable that returns (P, aux) for a batch (same contract as in train/loops.py).
    loop_cfg:
        Only used for device and whether to display progress bars.
    eval_cfg:
        Evaluation settings, seeds and CI options.
    logger:
        Optional MetricLogger to persist the aggregate summaries.
    step:
        Optional global step to tag logs.

    Returns
    -------
    results:
        Nested dict: {metric_key: {"all": CISummary, **(per_seed: CISummary if enabled)}}.
        The "all" entry is the CI over all per-episode samples pooled across seeds.
    """
    loop_cfg = loop_cfg or LoopConfig()
    eval_cfg = eval_cfg or EvalConfig()

    # Resolve seeds
    seeds = list(eval_cfg.seeds)
    if not seeds:
        seeds = list(range(eval_cfg.n_seeds))

    device = torch.device(loop_cfg.device)
    model.eval()
    model.to(device)

    # Accumulate raw episode-level scalars across *all* seeds
    pooled: dict[str, list[float]] = {}
    # Optionally, per-seed means (for reporting)
    per_seed: dict[int, dict[str, list[float]]] = {}

    # Main loop over seeds
    outer_iter = seeds
    if eval_cfg.progress_bar:
        outer_iter = tqdm(seeds, desc="eval-seeds", leave=False)

    for sd in outer_iter:
        _seed_everything(int(sd))
        dl = dataloader_factory(int(sd))

        # Lazily create a metric pack based on the first batch's aux
        metric_pack: MetricPack | None = None

        # Book-keeping for this seed
        if eval_cfg.keep_per_seed_means:
            per_seed[sd] = {}

        inner = dl
        if eval_cfg.progress_bar:
            # Try to display progress, but don't assume __len__
            total = len(dl) if hasattr(dl, "__len__") else None
            inner = tqdm(dl, total=total, desc=f"seed={sd}", leave=False)

        for batch in inner:
            # Move tensors to device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
            }

            with torch.no_grad():
                P, aux = rollout_fn(model, batch)  # P: (B,N) or (N,)
                if isinstance(P, torch.Tensor) and P.ndim == 1:
                    P = P.unsqueeze(0)

                # Lazily make metric pack based on whether targets exist
                if metric_pack is None:
                    with_targets = ("targets" in aux) and (aux["targets"] is not None)
                    metric_pack = build_metric_pack(with_targets=with_targets, cvar_alpha=0.1)

                # Compute all metric groups and flatten dicts into one level
                scalars: dict[str, float] = {}
                for name, fn in metric_pack.items():
                    vals = fn(P, **aux)
                    for k, v in vals.items():
                        scalars[f"{name}.{k}"] = float(v)

            # Append to pooled buffers
            for k, v in scalars.items():
                pooled.setdefault(k, []).append(v)
                if eval_cfg.keep_per_seed_means:
                    per_seed[sd].setdefault(k, []).append(v)

    # Compute CI summaries
    results: dict[str, dict[str, CISummary]] = {}

    # All-seed pooled CI
    for k, arr in pooled.items():
        t = torch.tensor(arr, dtype=torch.float64, device=device)
        results.setdefault(k, {})
        results[k]["all"] = compute_ci(
            t,
            method=eval_cfg.ci_method,
            alpha=eval_cfg.ci_alpha,
            bootstrap_samples=eval_cfg.bootstrap_samples,
        )

    # Per-seed means (then CI over *within-seed* episode samples)
    if eval_cfg.keep_per_seed_means:
        for sd, d in per_seed.items():
            for k, arr in d.items():
                t = torch.tensor(arr, dtype=torch.float64, device=device)
                results.setdefault(k, {})
                results[k][f"seed={sd}"] = compute_ci(
                    t,
                    method=eval_cfg.ci_method,
                    alpha=eval_cfg.ci_alpha,
                    bootstrap_samples=eval_cfg.bootstrap_samples,
                )

    # Optional structured logging → also persists to JSONL/TB/W&B via MetricLogger
    if logger is not None and step is not None:
        to_log = {}
        for k, d in results.items():
            ci = d["all"]
            to_log[f"{k}/mean"] = ci.mean
            to_log[f"{k}/lo"] = ci.lo
            to_log[f"{k}/hi"] = ci.hi
            to_log[f"{k}/stderr"] = ci.stderr
            to_log[f"{k}/n"] = float(ci.n)
        logger.log_dict("eval_CI/", to_log, step=step)

    return results


# --------------------------------------------------------------------------------------
#                             Formatting / pretty printing
# --------------------------------------------------------------------------------------


def _fmt_ci(ci: CISummary, conf: float) -> str:
    return f"{ci.mean:.4f} [{ci.lo:.4f}, {ci.hi:.4f}] @ {int(conf*100)}% (n={ci.n}, se={ci.stderr:.4g})"


def summarize_results(
    results: dict[str, dict[str, CISummary]],
    *,
    title: str | None = None,
    show_per_seed: bool = False,
    ci_alpha: float = 0.05,
) -> str:
    """
    Produce a human-readable multi-line summary string for console / logs.
    """
    conf = 1.0 - ci_alpha
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))

    # Stable ordering: alphabetical by metric key
    for k in sorted(results.keys()):
        all_ci = results[k].get("all")
        if all_ci is not None:
            lines.append(f"{k}: {_fmt_ci(all_ci, conf)}")
        if show_per_seed:
            # Show per-seed summaries in numeric seed order if present
            seeds = [x for x in results[k].keys() if x.startswith("seed=")]
            seeds.sort(key=lambda s: int(s.split("=")[-1]))
            for tag in seeds:
                lines.append(f"  {tag}: {_fmt_ci(results[k][tag], conf)}")

    return "\n".join(lines)

# acpl/eval/protocol.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import math
import random
import zlib

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
    # Optional override. If empty, we use loop_cfg.device (so CLI/device flags keep working).
    device: str = ""

    progress_bar: bool = True

    ci_method: str = "bootstrap"  # "student" or "bootstrap"
    ci_alpha: float = 0.05
    bootstrap_samples: int = 2000
    # Make CI computation deterministic / comparable across runs (esp. bootstrap).
    # If None -> use global RNG state (not recommended for “defendable” CI).
    bootstrap_seed: int | None = 0

    # Compute CI on CPU even if evaluation runs on GPU (more stable + avoids GPU RNG nondeterminism).
    ci_on_cpu: bool = True

    # In addition to pooled CI over all episodes, also compute CI over per-seed means (“across seeds”).
    include_seed_mean_ci: bool = True

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
    Student-t CI on the mean (exact t-quantile via torch.distributions.StudentT).

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

    # Unbiased sample std
    s = float(x.std(unbiased=True).item())
    stderr = s / math.sqrt(n)

    # Exact Student-t quantile with df=n-1
    tdist = torch.distributions.StudentT(df=n - 1)
    tcrit = float(tdist.icdf(torch.tensor(1 - alpha / 2, dtype=torch.float64)).item())
    half_width = tcrit * stderr

    return mean, mean - half_width, mean + half_width, stderr, n



def _bootstrap_interval(
    samples: torch.Tensor,
    alpha: float,
    B: int,
    *,
    generator: torch.Generator | None = None,
    chunk_size: int = 256,
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

    if generator is None:
        generator = torch.Generator(device=x.device)

    # Chunked bootstrap to reduce peak memory: build boot_means in pieces
    means_chunks: list[torch.Tensor] = []
    for start in range(0, B, chunk_size):
        b = min(chunk_size, B - start)
        idx = torch.randint(0, n, size=(b, n), device=x.device, generator=generator)
        means_chunks.append(x[idx].mean(dim=1))

    boot_means = torch.cat(means_chunks, dim=0)

    q_lo = float(alpha / 2)
    q_hi = float(1 - alpha / 2)
    lo = float(torch.quantile(boot_means, q=q_lo).item())
    hi = float(torch.quantile(boot_means, q=q_hi).item())

    stderr = float(boot_means.std(unbiased=True).item())
    return mean, lo, hi, stderr, n




def compute_ci(
    samples,
    *,
    method: str = "bootstrap",
    alpha: float = 0.05,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int | None = None,
    ci_on_cpu: bool = True,
) -> CISummary:

    """
    Compute mean ± CI for a 1-D tensor of samples.
    """
    # Normalize CI method names (plots/configs may use aliases like "student_t")
    method = str(method).strip().lower()
    method = method.replace("-", "_")

    if method in {"student_t", "studentt", "t", "t_ci", "t_interval", "student"}:
        method = "student"



    # Accept numpy arrays / lists (plots often operate in numpy) → normalize to torch.Tensor
    if not isinstance(samples, torch.Tensor):
        samples = torch.as_tensor(samples)

    # Ensure 1-D (CI expects a vector of samples)
    if samples.ndim > 1:
        samples = samples.reshape(-1)

    samples = samples.to(torch.float64)

    if ci_on_cpu:
        samples = samples.detach().to("cpu")
    else:
        samples = samples.detach()


    



    if method == "student":
        m, lo, hi, se, n = _student_interval(samples, alpha)
    elif method == "bootstrap":
        gen = None
        if bootstrap_seed is not None:
            gen = torch.Generator(device=samples.device)
            gen.manual_seed(int(bootstrap_seed))
        m, lo, hi, se, n = _bootstrap_interval(
            samples, alpha, bootstrap_samples, generator=gen
        )

    else:
        raise ValueError(f"Unknown ci_method: {method}")
    return CISummary(mean=m, lo=lo, hi=hi, stderr=se, n=n)

def _mix_bootstrap_seed(base: int | None, metric_key: str) -> int | None:
    """
    Deterministically mix a user-provided base seed with a metric key.
    Avoids “same bootstrap indices for every metric”.
    """
    if base is None:
        return None
    h = zlib.crc32(metric_key.encode("utf-8")) & 0xFFFFFFFF
    return int((int(base) + h) & 0x7FFFFFFF)

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

    # Determinism knobs (safe for eval; won’t crash if cudnn not present)
    try:  # pragma: no cover
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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

    device_str = (eval_cfg.device or "").strip()
    if not device_str:
        device_str = (loop_cfg.device or "").strip()


    if not device_str:
        device_str = "cpu"
    device = torch.device(device_str)
    model.eval()
    model.to(device)

    # Accumulate raw episode-level scalars across *all* seeds
    pooled: dict[str, list[float]] = {}
    # Optionally, per-seed means (for reporting)
    per_seed: dict[int, dict[str, list[float]]] = {}


    # For “across-seed” CI: track per-seed means without needing to retain everything
    seed_sum: dict[int, dict[str, float]] = {}
    seed_cnt: dict[int, dict[str, int]] = {}


    # Main loop over seeds
    outer_iter = seeds
    if eval_cfg.progress_bar:
        outer_iter = tqdm(seeds, desc="eval-seeds", leave=False)

    for sd in outer_iter:
        _seed_everything(int(sd))
        dl = dataloader_factory(int(sd))


        seed_sum[sd] = {}
        seed_cnt[sd] = {}


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

                # seed-mean book-keeping
                seed_sum[sd][k] = seed_sum[sd].get(k, 0.0) + float(v)
                seed_cnt[sd][k] = seed_cnt[sd].get(k, 0) + 1

                if eval_cfg.keep_per_seed_means:
                    per_seed[sd].setdefault(k, []).append(v)


    # Compute CI summaries
    results: dict[str, dict[str, CISummary]] = {}

    # All-seed pooled CI
    for k, arr in pooled.items():
        t = torch.tensor(arr, dtype=torch.float64)
        results.setdefault(k, {})
        bs = _mix_bootstrap_seed(eval_cfg.bootstrap_seed, k)
        results[k]["all"] = compute_ci(
            t,
            method=eval_cfg.ci_method,
            alpha=eval_cfg.ci_alpha,
            bootstrap_samples=eval_cfg.bootstrap_samples,
            bootstrap_seed=bs,
            ci_on_cpu=eval_cfg.ci_on_cpu,
        )


    # CI over per-seed means (the usual “multi-seed” aggregate people expect)
    if getattr(eval_cfg, "include_seed_mean_ci", True):
        for k in pooled.keys():
            means: list[float] = []
            for sd in seeds:
                if k in seed_sum.get(sd, {}) and seed_cnt[sd].get(k, 0) > 0:
                    means.append(seed_sum[sd][k] / seed_cnt[sd][k])


            if len(means) == 0:
                continue


            t = torch.tensor(means, dtype=torch.float64)
            bs = _mix_bootstrap_seed(eval_cfg.bootstrap_seed, f"seed_mean::{k}")
            results.setdefault(k, {})
            results[k]["seed_mean"] = compute_ci(
                t,
                method=eval_cfg.ci_method,
                alpha=eval_cfg.ci_alpha,
                bootstrap_samples=eval_cfg.bootstrap_samples,
                bootstrap_seed=bs,
                ci_on_cpu=eval_cfg.ci_on_cpu,
            )


    # Per-seed means (then CI over *within-seed* episode samples)
    if eval_cfg.keep_per_seed_means:
        for sd, d in per_seed.items():
            for k, arr in d.items():
                t = torch.tensor(arr, dtype=torch.float64)
                results.setdefault(k, {})
                bs = _mix_bootstrap_seed(eval_cfg.bootstrap_seed, f"seed={sd}::{k}")
                results[k][f"seed={sd}"] = compute_ci(
                    t,
                    method=eval_cfg.ci_method,
                    alpha=eval_cfg.ci_alpha,
                    bootstrap_samples=eval_cfg.bootstrap_samples,
                    bootstrap_seed=bs,
                    ci_on_cpu=eval_cfg.ci_on_cpu,
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
        
        
        seed_ci = results[k].get("seed_mean")
        if seed_ci is not None:
            lines.append(f"  seed_mean: {_fmt_ci(seed_ci, conf)}")
        
        
        
        if show_per_seed:
            # Show per-seed summaries in numeric seed order if present
            seeds = [x for x in results[k].keys() if x.startswith("seed=")]
            seeds.sort(key=lambda s: int(s.split("=")[-1]))
            for tag in seeds:
                lines.append(f"  {tag}: {_fmt_ci(results[k][tag], conf)}")

    return "\n".join(lines)

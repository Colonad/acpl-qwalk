# acpl/objectives/metrics.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Literal

import torch

__all__ = [
    # Simplex & normalization
    "normalize_prob",
    "validate_simplex",
    "is_probability_vector",
    "simplex_projection",
    # Basic distributions
    "uniform_prob",
    "delta_prob",
    # Core divergences/distances
    "total_variation",
    "l1_distance",
    "l2_distance",
    "kl_divergence",
    "reverse_kl_divergence",
    "js_divergence",
    "hellinger_distance",
    "renyi_divergence",
    # Entropy family
    "entropy",
    "cross_entropy",
    # Success & concentration
    "success_on_targets",
    "argmax_prob",
    "gini_coefficient",
    "max_prob",
    # Reduction helper
    "reduce_metric",
    # From complex state (position ⊗ coin → position marginals)
    "position_marginal_from_state",
    # Risk-sensitive aggregations
    "cvar",
    # Summaries used in training/eval
    "mixing_summary",
    "targeting_summary",
    "MetricSummary",
    # ---- Phase B4 additions: Hitting-time & Confidence Intervals ----
    # Hitting-time from success curves
    "success_curve",
    "first_passage_from_success",
    "hitting_time_stats",
    "hitting_time_summary",
    # Confidence intervals
    "proportion_ci",
    "mean_ci",
    "bootstrap_ci",
]

# ------------------------------- Utilities -------------------------------- #


def _as_tensor(x: torch.Tensor | float, *, device=None, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype) if (device or dtype) else x
    return torch.tensor(x, device=device, dtype=dtype)


def _safe_eps(dtype: torch.dtype) -> float:
    # A conservative epsilon that plays well with fp32 and fp64
    if dtype in (torch.float32, torch.complex64):
        return 1e-7
    if dtype in (torch.float64, torch.complex128):
        return 1e-12
    return 1e-7


# ---------------------- Simplex checks and normalization ------------------- #


def normalize_prob(
    p: torch.Tensor,
    dim: int = -1,
    eps: float | None = None,
) -> torch.Tensor:
    """
    Normalize non-negative scores to a probability vector along `dim`.

    • Stable for zero or near-zero sums (falls back to uniform on the subspace).
    • Preserves gradients almost everywhere.
    """
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    p_clamped = p.clamp_min(0)  # keep gradients for positives; zeros are fine
    s = p_clamped.sum(dim=dim, keepdim=True)
    zero_mask = s <= eps
    if zero_mask.any():
        n = p_clamped.size(dim)
        u = torch.full_like(p_clamped, 1.0 / max(1, n))
        q = p_clamped / s.clamp_min(eps)
        return torch.where(zero_mask, u, q)
    else:
        return p_clamped / s


def validate_simplex(
    p: torch.Tensor,
    dim: int = -1,
    atol: float = 1e-5,
    rtol: float = 1e-6,
) -> dict:
    """
    Validate that `p` lies approximately on the probability simplex along `dim`.
    """
    p_min, _ = p.min(dim=dim, keepdim=True)
    p_max, _ = p.max(dim=dim, keepdim=True)
    p_sum = p.sum(dim=dim, keepdim=True)

    one = _as_tensor(1.0, device=p.device, dtype=p.dtype)
    nonneg_mask = p_min >= -abs(atol)
    sum_ok_mask = torch.isclose(p_sum, one, rtol=rtol, atol=atol)

    slice_num = p_sum.numel()
    violations_nonneg = int((~nonneg_mask).sum().item())
    violations_sum = int((~sum_ok_mask).sum().item())

    return {
        "sum": p_sum.squeeze(dim),
        "min": p_min.squeeze(dim),
        "max": p_max.squeeze(dim),
        "nonneg_ok": violations_nonneg == 0,
        "sum_ok": violations_sum == 0,
        "violations_nonneg": violations_nonneg,
        "violations_sum": violations_sum,
        "slices": slice_num,
    }


def is_probability_vector(
    p: torch.Tensor,
    dim: int = -1,
    atol: float = 1e-5,
    rtol: float = 1e-6,
) -> bool:
    diag = validate_simplex(p, dim=dim, atol=atol, rtol=rtol)
    return bool(diag["nonneg_ok"] and diag["sum_ok"])


def simplex_projection(
    p: torch.Tensor,
    dim: int = -1,
    eps: float | None = None,
) -> torch.Tensor:
    """
    Project onto the probability simplex along `dim` (Michelot / WCP algorithm).
    """
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    x = p.clone().movedim(dim, -1)
    shape = x.shape
    n = shape[-1]
    flat = x.reshape(-1, n)

    u, _ = torch.sort(flat, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1)
    j = torch.arange(1, n + 1, device=p.device, dtype=p.dtype).unsqueeze(0)
    cond = u - (cssv - 1.0) / j > 0
    rho = cond.sum(dim=1).clamp(min=1)
    idx = rho - 1
    theta = (cssv.gather(1, idx.unsqueeze(1)) - 1.0) / rho.unsqueeze(1)
    proj = (flat - theta).clamp_min(0.0)
    proj = proj.reshape(shape).movedim(-1, dim)
    return normalize_prob(proj, dim=dim, eps=eps)


# ---------------------------- Basic distributions -------------------------- #


def uniform_prob(
    n: int,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if n <= 0:
        raise ValueError("n must be positive.")
    return torch.full((n,), 1.0 / n, device=device, dtype=dtype)


def delta_prob(
    n: int,
    index: int,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not (0 <= index < n):
        raise ValueError("index out of range for delta distribution.")
    v = torch.zeros(n, device=device, dtype=dtype)
    v[index] = 1.0
    return v


# ---------------------- Distances & divergences (prob) --------------------- #


def reduce_metric(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if reduction == "none":
        return x
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")


def total_variation(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    d = (p - q).abs().sum(dim=dim) * 0.5
    return reduce_metric(d, reduction)


def l1_distance(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    return reduce_metric((p - q).abs().sum(dim=dim), reduction)


def l2_distance(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    return reduce_metric(((p - q) ** 2).sum(dim=dim).sqrt(), reduction)


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
    eps: float | None = None,
) -> torch.Tensor:
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    p = normalize_prob(p, dim=dim, eps=eps).clamp_min(eps)
    q = normalize_prob(q, dim=dim, eps=eps).clamp_min(eps)
    kl = (p * (p.log() - q.log())).sum(dim=dim)
    return reduce_metric(kl, reduction)


def reverse_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
    eps: float | None = None,
) -> torch.Tensor:
    return kl_divergence(q, p, dim=dim, reduction=reduction, eps=eps)


def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
    eps: float | None = None,
) -> torch.Tensor:
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    p = normalize_prob(p, dim=dim, eps=eps).clamp_min(eps)
    q = normalize_prob(q, dim=dim, eps=eps).clamp_min(eps)
    m = 0.5 * (p + q)
    js = 0.5 * (
        kl_divergence(p, m, dim=dim, reduction="none", eps=eps)
        + kl_divergence(q, m, dim=dim, reduction="none", eps=eps)
    )
    return reduce_metric(js, reduction)


def hellinger_distance(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
    eps: float | None = None,
) -> torch.Tensor:
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    p = normalize_prob(p, dim=dim, eps=eps).clamp_min(0.0)
    q = normalize_prob(q, dim=dim, eps=eps).clamp_min(0.0)
    hp = p.clamp_min(0.0).sqrt()
    hq = q.clamp_min(0.0).sqrt()
    h = (0.5 * ((hp - hq) ** 2).sum(dim=dim)).sqrt()
    return reduce_metric(h, reduction)


def renyi_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    alpha: float,
    dim: int = -1,
    reduction: str = "mean",
    eps: float | None = None,
) -> torch.Tensor:
    if alpha <= 0.0 or abs(alpha - 1.0) < 1e-12:
        raise ValueError("alpha must be > 0 and != 1 for Rényi divergence.")
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    p = normalize_prob(p, dim=dim, eps=eps).clamp_min(eps)
    q = normalize_prob(q, dim=dim, eps=eps).clamp_min(eps)
    t = (p**alpha) * (q ** (1.0 - alpha))
    s = t.sum(dim=dim).clamp_min(eps)
    d = (s.log()) / (alpha - 1.0)
    return reduce_metric(d, reduction)


# ----------------------------- Entropy family ------------------------------ #


def entropy(
    p: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
    eps: float | None = None,
) -> torch.Tensor:
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    p = normalize_prob(p, dim=dim, eps=eps).clamp_min(eps)
    h = -(p * p.log()).sum(dim=dim)
    return reduce_metric(h, reduction)


def cross_entropy(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
    eps: float | None = None,
) -> torch.Tensor:
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    p = normalize_prob(p, dim=dim, eps=eps)
    q = normalize_prob(q, dim=dim, eps=eps).clamp_min(eps)
    ce = -(p * q.log()).sum(dim=dim)
    return reduce_metric(ce, reduction)


# ---------------------- Success & concentration metrics -------------------- #


def success_on_targets(
    p: torch.Tensor,
    targets: Sequence[int] | torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Sum of probabilities over a target set Ω ⊆ {0..N-1}.
    """
    if isinstance(targets, torch.Tensor):
        idx = targets.to(device=p.device, dtype=torch.long)
    else:
        idx = torch.tensor(list(targets), device=p.device, dtype=torch.long)
    if idx.numel() == 0:
        s = torch.zeros(p.sum(dim=dim).shape, device=p.device, dtype=p.dtype)
        return reduce_metric(s, reduction)
    x = p.movedim(dim, -1)
    s = x.index_select(-1, idx).sum(dim=-1)
    return reduce_metric(s, reduction)


def argmax_prob(
    p: torch.Tensor,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.max(p, dim=dim)


def max_prob(
    p: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    m, _ = torch.max(p, dim=dim)
    return reduce_metric(m, reduction)


def gini_coefficient(
    p: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
    eps: float | None = None,
) -> torch.Tensor:
    """
    Gini coefficient for a probability vector:
      G = (1/(2n)) * sum_{i,j} |p_i - p_j|
    """
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    p = normalize_prob(p, dim=dim, eps=eps)
    x = p.movedim(dim, -1)
    n = x.shape[-1]
    x_sorted, _ = torch.sort(x, dim=-1)
    idx = torch.arange(n, device=p.device, dtype=p.dtype)
    left = (idx * x_sorted).sum(dim=-1)
    right = ((n - 1 - idx) * x_sorted).sum(dim=-1)
    pair_sum = left - right
    total_abs = 2.0 * pair_sum
    g = total_abs / (2.0 * n)
    return reduce_metric(g, reduction)


# ---------------- From complex state → position marginals ------------------ #


def position_marginal_from_state(
    psi: torch.Tensor,
    coin_dim: int | None = None,
    pos_dim: int = -2,
    coin_last: bool = True,
) -> torch.Tensor:
    """
    Compute position marginals from a pure state amplitude tensor.
    """
    if psi.is_complex():
        x2 = (psi.conj() * psi).real
    else:
        x2 = psi**2

    if coin_last:
        if pos_dim != -2:
            raise ValueError("When coin_last=True, pos_dim must index the position axis as -2.")
        P = x2.sum(dim=-1)
        return P
    else:
        P = x2.sum(dim=-2)
        return P


# ----------------------------- Risk aggregations --------------------------- #


def cvar(
    values: torch.Tensor,
    alpha: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Lower-tail CVaR over a flat vector of values.
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    flat = values.reshape(-1)
    k = max(1, int(math.ceil(alpha * flat.numel())))
    v, _ = torch.topk(flat, k, largest=False)
    return reduce_metric(v.mean(), reduction)


# ------------------------------ Summaries --------------------------------- #


@dataclass
class MetricSummary:
    tv: float
    js: float
    hellinger: float
    l2: float
    entropy_p: float
    kl_pu: float
    # Optional task-anchored scalars
    success: float | None = None
    maxp: float | None = None
    gini: float | None = None


@torch.no_grad()
def mixing_summary(
    p: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
    uniform: torch.Tensor | None = None,
) -> MetricSummary:
    """
    Summarize mixing quality vs. uniform:
      TV, JS, Hellinger, L2, Entropy(p), KL(p || U).
    """
    if uniform is None:
        n = p.size(dim)
        uniform = uniform_prob(n, device=p.device, dtype=p.dtype)
    tv = float(total_variation(p, uniform, dim=dim, reduction=reduction).item())
    js = float(js_divergence(p, uniform, dim=dim, reduction=reduction).item())
    he = float(hellinger_distance(p, uniform, dim=dim, reduction=reduction).item())
    l2 = float(l2_distance(p, uniform, dim=dim, reduction=reduction).item())
    ent = float(entropy(p, dim=dim, reduction=reduction).item())
    klu = float(kl_divergence(p, uniform, dim=dim, reduction=reduction).item())
    return MetricSummary(tv=tv, js=js, hellinger=he, l2=l2, entropy_p=ent, kl_pu=klu)


@torch.no_grad()
def targeting_summary(
    p: torch.Tensor,
    targets: Sequence[int] | torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
) -> MetricSummary:
    """
    Summarize task success on a target set along with concentration diagnostics.
    """
    n = p.size(dim)
    U = uniform_prob(n, device=p.device, dtype=p.dtype)
    tv = float(total_variation(p, U, dim=dim, reduction=reduction).item())
    js = float(js_divergence(p, U, dim=dim, reduction=reduction).item())
    he = float(hellinger_distance(p, U, dim=dim, reduction=reduction).item())
    l2 = float(l2_distance(p, U, dim=dim, reduction=reduction).item())
    ent = float(entropy(p, dim=dim, reduction=reduction).item())
    klu = float(kl_divergence(p, U, dim=dim, reduction=reduction).item())
    succ = float(success_on_targets(p, targets, dim=dim, reduction=reduction).item())
    maxp = float(max_prob(p, dim=dim, reduction=reduction).item())
    gini = float(gini_coefficient(p, dim=dim, reduction=reduction).item())
    return MetricSummary(
        tv=tv,
        js=js,
        hellinger=he,
        l2=l2,
        entropy_p=ent,
        kl_pu=klu,
        success=succ,
        maxp=maxp,
        gini=gini,
    )


# =========================================================================== #
#                         PHASE B4: HITTING-TIME METRICS                      #
# =========================================================================== #


def success_curve(
    P_tn: torch.Tensor,
    targets: Sequence[int] | torch.Tensor,
    *,
    time_dim: int = -2,
    prob_dim: int = -1,
    reduction: Literal["none", "mean", "sum"] = "none",
) -> torch.Tensor:
    """
    Build the *success* curve s_t = Pr(target at time t) from per-time distributions.

    Args
    ----
    P_tn : tensor of probabilities with a time and a position axis.
           Common shapes: [T, N], [B, T, N], [..., T, N]
    targets : set of target vertex indices
    time_dim : which dimension indexes time
    prob_dim : which dimension indexes positions
    reduction : if 'none' returns s with same leading dims except `prob_dim` removed;
                'mean'/'sum' reduce across all non-time dims.

    Returns
    -------
    s : tensor of shape [..., T] when reduction='none', or [T] for mean/sum.
    """
    if isinstance(targets, torch.Tensor):
        idx = targets.to(device=P_tn.device, dtype=torch.long)
    else:
        idx = torch.tensor(list(targets), device=P_tn.device, dtype=torch.long)

    x = P_tn.movedim(time_dim, -2).movedim(prob_dim, -1)  # [..., T, N]
    s = x.index_select(-1, idx).sum(dim=-1)  # [..., T]
    if reduction == "none":
        return s
    if reduction == "mean":
        # average across all leading batch dims
        if s.ndim > 1:
            lead_dims = list(range(0, s.ndim - 1))  # except last (time)
            if lead_dims:
                s = s.mean(dim=lead_dims)
        return s
    if reduction == "sum":
        if s.ndim > 1:
            lead_dims = list(range(0, s.ndim - 1))
            if lead_dims:
                s = s.sum(dim=lead_dims)
        return s
    raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")


def first_passage_from_success(
    s_t: torch.Tensor,
    *,
    time_dim: int = -1,
    clamp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a *success* curve s_t ≈ Pr(hit at t | not hit before) into:
      f_t (first-passage PMF), F_t (CDF), S_t (survival).

    Assumption (standard measurement-at-each-step model):
      f_t = s_t * ∏_{k < t} (1 - s_k)
      S_t = ∏_{k <= t} (1 - s_k)
      F_t = 1 - S_t

    Args
    ----
    s_t : tensor [..., T]
    time_dim : dimension indexing time
    clamp : if True, clamps s_t to [0,1] for numeric safety

    Returns
    -------
    f_t, F_t, S_t : tensors of same shape as s_t
    """
    x = s_t.movedim(time_dim, -1)  # [..., T]
    if clamp:
        x = x.clamp(0.0, 1.0)

    one = _as_tensor(1.0, device=x.device, dtype=x.dtype)
    # Survival up to t-1
    # prefix_prod[t] = ∏_{k < t} (1 - s_k)
    one_minus = one - x
    # cumprod for survival including time t:
    S_inclusive = torch.cumprod(one_minus, dim=-1)  # [..., T] = ∏_{k<=t} (1-s_k)
    # Shifted survival (pre-hit) for pmf:
    # pref[t] = 1 for t==0 else S_inclusive[..., t-1]
    pad = torch.ones_like(S_inclusive[..., :1])
    pref = torch.cat([pad, S_inclusive[..., :-1]], dim=-1)

    f = x * pref  # pmf
    S = S_inclusive  # survival (has t included)
    F = 1.0 - S  # cdf
    # Move back to original time axis
    return f.movedim(-1, time_dim), F.movedim(-1, time_dim), S.movedim(-1, time_dim)


def hitting_time_stats(
    f_t: torch.Tensor,
    *,
    time_dim: int = -1,
    right_censor: Literal["none", "mass_to_T", "kaplan_meier"] = "mass_to_T",
) -> dict:
    """
    Compute expected and median hitting time from a (possibly truncated) first-passage PMF.

    Args
    ----
    f_t : first-passage probabilities [..., T] with ∑_t f_t ≤ 1 due to truncation
    time_dim : time dimension index
    right_censor :
        "none"          : assume ∑ f_t == 1 (no truncation)
        "mass_to_T"     : treat leftover mass as hitting at time T (conservative upper bound)
        "kaplan_meier"  : KM plug-in for the mean using survival tail at the boundary

    Returns
    -------
    {
      "mean":  E[T_hit] (same leading dims as f_t w/o time),
      "median": median hitting time (linear interpolation between grid points),
      "mass":  total observed hit probability up to T,
      "unhit": leftover 1 - mass,
    }
    """
    x = f_t.movedim(time_dim, -1)  # [..., T]
    T = x.shape[-1]
    device, dtype = x.device, x.dtype
    one = _as_tensor(1.0, device=device, dtype=dtype)

    mass = x.sum(dim=-1)  # [...,]
    unhit = (one - mass).clamp_min(0.0)

    # grid of times starting at 1 ... T (standard discrete hitting time)
    tgrid = torch.arange(1, T + 1, device=device, dtype=dtype)
    mean_uncensored = (x * tgrid).sum(dim=-1)  # [...,]

    if right_censor == "none":
        mean = mean_uncensored
    elif right_censor == "mass_to_T":
        mean = mean_uncensored + unhit * tgrid[-1]
    elif right_censor == "kaplan_meier":
        # Kaplan–Meier discrete plug-in: E[T] = Σ_t S(t-1)
        # We reconstruct S from f: S(t) = 1 - F(t) with F cumulative of f.
        F = torch.cumsum(x, dim=-1)
        S = (one - F).clamp_min(0.0)
        # S(t-1) sequence length T equals [1, S1, S2, ..., S_{T-1}]
        pad = torch.ones_like(S[..., :1])
        S_prev = torch.cat([pad, S[..., :-1]], dim=-1)
        mean = S_prev.sum(dim=-1)
    else:
        raise ValueError("right_censor must be one of {'none','mass_to_T','kaplan_meier'}")

    # Median: smallest t with CDF >= 0.5 (handle truncation by allowing T)
    F = torch.cumsum(x, dim=-1)  # [..., T]
    med_mask = F >= 0.5
    # default T if never reaches 0.5
    default_med = torch.full_like(mean, fill_value=float(T))
    if med_mask.any():
        # first index where CDF >= 0.5
        idx = torch.argmax(med_mask.to(torch.long), dim=-1)
        # Convert 0-based to 1..T
        median = idx.to(dtype) + 1.0
        # If no reach, fall back to T
        never = ~med_mask.any(dim=-1)
        median = torch.where(never, default_med, median)
    else:
        median = default_med

    return {
        "mean": mean,  # [...]
        "median": median,  # [...]
        "mass": mass,  # [...]
        "unhit": unhit,  # [...]
    }


@torch.no_grad()
def hitting_time_summary(
    P_tn: torch.Tensor,
    targets: Sequence[int] | torch.Tensor,
    *,
    time_dim: int = -2,
    prob_dim: int = -1,
    reduction: Literal["none", "mean"] = "mean",
    censor: Literal["mass_to_T", "kaplan_meier", "none"] = "mass_to_T",
) -> dict:
    """
    End-to-end summary from per-time position distributions to hitting-time stats.

    Pipeline:
      P_tn -> s_t (success) -> (f_t, F_t, S_t) -> {mean, median, mass, unhit}

    Supports batching across any number of leading dims.
    """
    s = success_curve(
        P_tn, targets, time_dim=time_dim, prob_dim=prob_dim, reduction="none"
    )  # [..., T]
    f, F, S = first_passage_from_success(s, time_dim=-1)
    stats = hitting_time_stats(f, time_dim=-1, right_censor=censor)

    if reduction == "none":
        return {
            "success": s,
            "first_passage": f,
            "cdf": F,
            "survival": S,
            **stats,
        }

    # mean across all leading dims (if any)
    def _mean_last(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x
        dims = list(range(0, x.ndim - 1))
        return x.mean(dim=dims) if dims else x

    return {
        "success": _mean_last(s),
        "first_passage": _mean_last(f),
        "cdf": _mean_last(F),
        "survival": _mean_last(S),
        "mean": stats["mean"].mean() if stats["mean"].ndim > 0 else stats["mean"],
        "median": stats["median"].mean() if stats["median"].ndim > 0 else stats["median"],
        "mass": stats["mass"].mean() if stats["mass"].ndim > 0 else stats["mass"],
        "unhit": stats["unhit"].mean() if stats["unhit"].ndim > 0 else stats["unhit"],
    }


# =========================================================================== #
#                        PHASE B4: CONFIDENCE INTERVALS                       #
# =========================================================================== #


def proportion_ci(
    k: torch.Tensor | int,
    n: torch.Tensor | int,
    *,
    conf: float = 0.95,
    method: Literal["wilson", "clopper_pearson", "normal"] = "wilson",
    add_half: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Binomial proportion confidence interval for p = k/n.

    • 'wilson'           : Wilson score with/without continuity correction (add_half)
    • 'clopper_pearson'  : exact interval via Beta quantiles (through torch distributions)
    • 'normal'           : p̂ ± z * sqrt(p̂(1-p̂)/n)

    Returns (low, high) tensors broadcasted over inputs.
    """
    k = _as_tensor(k, device=device, dtype=dtype)
    n = _as_tensor(n, device=device, dtype=dtype)
    n = n.clamp_min(1.0)
    phat = (k / n).clamp(0.0, 1.0)

    alpha = 1.0 - conf
    z = torch.distributions.Normal(0, 1).icdf(_as_tensor(1 - alpha / 2, device=device, dtype=dtype))

    if method == "normal":
        half = z * torch.sqrt((phat * (1 - phat)) / n)
        lo = (phat - half).clamp_min(0.0)
        hi = (phat + half).clamp_max(1.0)
        return lo, hi

    if method == "wilson":
        # Wilson score interval
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (phat + z2 / (2 * n)) / denom
        half = z * torch.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n) / denom
        lo = (center - half).clamp_min(0.0)
        hi = (center + half).clamp_max(1.0)
        if add_half:
            # Agresti–Coull correction: add 1/2 success and 1/2 failure
            n_adj = n + 1.0
            p_adj = (k + 0.5) / n_adj
            center = (p_adj + z2 / (2 * n_adj)) / (1.0 + z2 / n_adj)
            half = (
                z
                * torch.sqrt((p_adj * (1 - p_adj) + z2 / (4 * n_adj)) / n_adj)
                / (1.0 + z2 / n_adj)
            )
            lo = torch.minimum(lo, (center - half).clamp_min(0.0))
            hi = torch.maximum(hi, (center + half).clamp_max(1.0))
        return lo, hi

    if method == "clopper_pearson":
        # Exact via Beta quantiles: [Beta(alpha/2; k, n-k+1), Beta(1-alpha/2; k+1, n-k)]
        # Handle edge cases k=0 or k=n
        a = k
        b = n - k
        # For Beta quantiles, use Gamma composition; rely on PyTorch Beta distribution
        Beta = torch.distributions.Beta
        # Avoid invalid params (0) by correcting to small eps inside distribution
        eps = _safe_eps(dtype)
        lo = torch.where(
            a > 0,
            Beta(a, b + 1.0).icdf(_as_tensor(alpha / 2, device=device, dtype=dtype)),
            _as_tensor(0.0, device=device, dtype=dtype),
        )
        hi = torch.where(
            b > 0,
            Beta(a + 1.0, b).icdf(_as_tensor(1 - alpha / 2, device=device, dtype=dtype)),
            _as_tensor(1.0, device=device, dtype=dtype),
        )
        lo = lo.clamp(0.0, 1.0)
        hi = hi.clamp(0.0, 1.0)
        return lo, hi

    raise ValueError("method must be one of {'wilson','clopper_pearson','normal'}")


def mean_ci(
    x: torch.Tensor,
    *,
    dim: int | None = 0,
    conf: float = 0.95,
    use_student_t: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Confidence interval for the *mean* along dimension `dim`.

    Returns (mean, lo, hi).
    """
    if dim is None:
        x = x.reshape(-1)
        dim = 0

    n = x.size(dim)
    if n < 1:
        raise ValueError("mean_ci: sample size must be >= 1")

    mu = x.mean(dim=dim)
    var = x.var(dim=dim, unbiased=True) if n > 1 else torch.zeros_like(mu)
    se = torch.sqrt(var / max(n, 1))

    alpha = 1.0 - conf
    if use_student_t and n > 1:
        tcrit = (
            torch.distributions.StudentT(df=_as_tensor(n - 1, device=x.device, dtype=torch.float32))
            .icdf(_as_tensor(1 - alpha / 2, device=x.device, dtype=torch.float32))
            .to(dtype=x.dtype, device=x.device)
        )
    else:
        tcrit = torch.distributions.Normal(0, 1).icdf(
            _as_tensor(1 - alpha / 2, device=x.device, dtype=x.dtype)
        )

    half = tcrit * se
    lo = mu - half
    hi = mu + half
    return mu, lo, hi


def bootstrap_ci(
    samples: torch.Tensor,
    stat_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_boot: int = 1000,
    conf: float = 0.95,
    dim: int = 0,
    method: Literal["percentile", "basic"] = "percentile",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generic bootstrap CI for a statistic of i.i.d. samples along `dim`.

    Args
    ----
    samples : tensor with sample axis `dim`
    stat_fn : function mapping a resampled tensor (same shape as samples along `dim`)
              to a statistic tensor (broadcast-friendly)
    n_boot  : number of bootstrap replicates
    conf    : confidence level
    method  : 'percentile' (default) or 'basic' (reverse-percentile)
    seed    : RNG seed for reproducibility

    Returns
    -------
    (stat, lo, hi)
    """
    x = samples.movedim(dim, 0)  # [N, ...]
    N = x.size(0)
    if N < 1:
        raise ValueError("bootstrap_ci: N must be >= 1")

    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)

    stat = stat_fn(x)  # original statistic (on full data)

    # Collect bootstrap stats
    boots: list[torch.Tensor] = []
    for _ in range(int(n_boot)):
        idx = torch.randint(0, N, (N,), generator=g)
        xb = x.index_select(0, idx)
        boots.append(stat_fn(xb))
    B = torch.stack(boots, dim=0)  # [B, *stat_shape]

    alpha = 1.0 - conf
    q_lo = alpha / 2
    q_hi = 1 - alpha / 2

    # compute empirical quantiles by sorting (differentiability not required here)
    B_sorted, _ = torch.sort(B, dim=0)
    k_lo = max(0, min(B_sorted.size(0) - 1, int(math.floor(q_lo * (B_sorted.size(0) - 1)))))
    k_hi = max(0, min(B_sorted.size(0) - 1, int(math.floor(q_hi * (B_sorted.size(0) - 1)))))

    if method == "percentile":
        lo = B_sorted[k_lo]
        hi = B_sorted[k_hi]
    elif method == "basic":
        # basic = 2*stat - percentile
        lo = 2 * stat - B_sorted[k_hi]
        hi = 2 * stat - B_sorted[k_lo]
    else:
        raise ValueError("method must be one of {'percentile','basic'}")

    return stat, lo, hi

# acpl/objectives/metrics.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

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

    Args:
        p: Tensor of scores ≥ 0 (not strictly required; negatives get clamped).
        dim: Dimension to normalize over.
        eps: Small value to avoid division by zero. If None, chosen by dtype.

    Returns:
        Probabilities with sum 1 along `dim`.
    """
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    p_clamped = p.clamp_min(0)  # keep gradients for positives; zeros are fine
    s = p_clamped.sum(dim=dim, keepdim=True)
    # When sum==0, fall back to uniform (still differentiable wrt p via clamp)
    zero_mask = s <= eps
    if zero_mask.any():
        # Shape of the prob-axis
        n = p_clamped.size(dim)
        # Build uniform along dim
        u = torch.full_like(p_clamped, 1.0 / max(1, n))
        # Normal path where sum > eps
        q = p_clamped / s.clamp_min(eps)
        # Select per-slice
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

    Checks:
      1) non-negativity (up to atol),
      2) sums close to 1 (rtol/atol).

    Returns diagnostic dict:
      {
        "sum": tensor[..., 1],
        "min": tensor[..., 1],
        "max": tensor[..., 1],
        "nonneg_ok": bool,
        "sum_ok": bool,
        "violations_nonneg": int,
        "violations_sum": int,
      }
    """
    # Basic reductions
    p_min, _ = p.min(dim=dim, keepdim=True)
    p_max, _ = p.max(dim=dim, keepdim=True)
    p_sum = p.sum(dim=dim, keepdim=True)

    # Nonneg check (allow small negative up to atol)
    nonneg_mask = p_min >= -abs(atol)
    # Sum ≈ 1 check
    one = _as_tensor(1.0, device=p.device, dtype=p.dtype)
    sum_ok_mask = torch.isclose(p_sum, one, rtol=rtol, atol=atol)

    # Count violations across slices
    # Collapse all other dims to count slices
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
    """
    Return True if `p` is (approximately) a probability vector along `dim`.
    """
    diag = validate_simplex(p, dim=dim, atol=atol, rtol=rtol)
    return bool(diag["nonneg_ok"] and diag["sum_ok"])


def simplex_projection(
    p: torch.Tensor,
    dim: int = -1,
    eps: float | None = None,
) -> torch.Tensor:
    """
    Project onto the probability simplex along `dim` using a fast algorithm (Michelot).

    This is useful for post-processing (e.g., logits → clamp → projection) when you
    need *exact* simplex membership. Differentiable almost everywhere.

    Complexity: O(n log n) along the simplex axis due to sorting.

    Reference:
      - Projection onto the probability simplex: Wang & Carreira-Perpinan (2013).
    """
    # Implementation adapted to be torch-batch friendly.
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    x = p.clone()
    # Move target dim to last
    x = x.movedim(dim, -1)
    shape = x.shape
    n = shape[-1]
    flat = x.reshape(-1, n)

    # Sort descending
    u, _ = torch.sort(flat, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1)
    # Find rho = max { j | u_j - (cssv_j - 1)/j > 0 }
    j = torch.arange(1, n + 1, device=p.device, dtype=p.dtype).unsqueeze(0)
    cond = u - (cssv - 1.0) / j > 0
    rho = cond.sum(dim=1).clamp(min=1)  # at least 1
    # Compute theta
    idx = rho - 1  # zero-based
    theta = (cssv.gather(1, idx.unsqueeze(1)) - 1.0) / rho.unsqueeze(1)
    # Project
    proj = (flat - theta).clamp_min(0.0)
    proj = proj.reshape(shape).movedim(-1, dim)

    # Numerical safety: renormalize tiny deviations
    return normalize_prob(proj, dim=dim, eps=eps)


# ---------------------------- Basic distributions -------------------------- #


def uniform_prob(
    n: int,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Uniform probability vector of length n.
    """
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
    """
    Dirac (delta) probability vector: 1 at `index`, 0 elsewhere (length n).
    """
    if not (0 <= index < n):
        raise ValueError("index out of range for delta distribution.")
    v = torch.zeros(n, device=device, dtype=dtype)
    v[index] = 1.0
    return v


# ---------------------- Distances & divergences (prob) --------------------- #


def reduce_metric(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Reduction helper: 'none' | 'mean' | 'sum'.
    """
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
    """
    Total variation distance TV(p, q) = 0.5 * ||p - q||_1 along `dim`.

    Assumes p, q are probabilities (but does not strictly require renormalization).
    """
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
    """
    KL(p || q) = sum_i p_i * log(p_i / q_i)

    • Stable: clamps with eps and renormalizes inputs first (soft).
    • Returns 0 when p == q pointwise (up to eps).
    """
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
    """
    KL(q || p) for mode-seeking behavior comparisons.
    """
    return kl_divergence(q, p, dim=dim, reduction=reduction, eps=eps)


def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
    eps: float | None = None,
) -> torch.Tensor:
    """
    Jensen–Shannon divergence: JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),
    with m = 0.5*(p+q). Symmetric and bounded in [0, log 2].
    """
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
    """
    Hellinger distance H(p, q) = (1/sqrt(2)) * || sqrt(p) - sqrt(q) ||_2
    """
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
    """
    Rényi divergence D_α(p || q) = (1/(α-1)) * log sum_i p_i^α q_i^(1-α), α>0, α≠1
    """
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
    """
    Shannon entropy H(p) = -sum p log p
    """
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
    """
    Cross-entropy H(p, q) = -sum p log q
    """
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

    Args:
        p: (..., N) probability vectors.
        targets: 1-D tensor/sequence of target indices.
        dim: prob dimension (default last).
        reduction: 'none' | 'mean' (default) | 'sum' across batch-like axes.

    Returns:
        Tensor with success probabilities (...,) reduced per sample, then reduced
        according to `reduction`.
    """
    if isinstance(targets, torch.Tensor):
        idx = targets.to(device=p.device)
    else:
        idx = torch.tensor(list(targets), device=p.device, dtype=torch.long)
    if idx.numel() == 0:
        # Empty target set ⇒ success = 0
        s = torch.zeros(p.sum(dim=dim).shape, device=p.device, dtype=p.dtype)
        return reduce_metric(s, reduction)

    # Gather along the prob dimension. Move dim to last for simple indexing.
    x = p.movedim(dim, -1)
    s = x.index_select(-1, idx).sum(dim=-1)
    return reduce_metric(s, reduction)


def argmax_prob(
    p: torch.Tensor,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (max_value, argmax_index) along `dim`.
    """
    return torch.max(p, dim=dim)


def max_prob(
    p: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Max probability mass per vector (e.g., for concentration curves).
    """
    m, _ = torch.max(p, dim=dim)
    return reduce_metric(m, reduction)


def gini_coefficient(
    p: torch.Tensor,
    dim: int = -1,
    reduction: str = "mean",
    eps: float | None = None,
) -> torch.Tensor:
    """
    Gini coefficient for a probability vector (concentration/inequality measure).

    For probability vectors (sum=1, p_i >= 0), a convenient formula is:
      G = 1 - sum_i (2i - n - 1)/[n-1] * p_(i)   (with p_ sorted ascending)
    but we implement an equivalent normalized L1-based measure:

      G = (1/(2n)) * sum_{i,j} |p_i - p_j|

    We compute this efficiently without forming the full (i,j) matrix.

    Batches supported. Returns scalar if reduction != 'none'.
    """
    eps = _safe_eps(p.dtype) if eps is None else float(eps)
    p = normalize_prob(p, dim=dim, eps=eps)
    # Move axis to last for convenience
    x = p.movedim(dim, -1)
    n = x.shape[-1]

    # Sort ascending along last axis
    x_sorted, _ = torch.sort(x, dim=-1)
    # Cumulative sums trick for pairwise L1 sum:
    # sum_{i<j} (x_j - x_i) = sum_j j*x_j - sum_i (n-1-i)*x_i
    idx = torch.arange(n, device=p.device, dtype=p.dtype)
    left = (idx * x_sorted).sum(dim=-1)
    right = ((n - 1 - idx) * x_sorted).sum(dim=-1)
    pair_sum = left - right  # equals sum_{i<j} (x_j - x_i)
    total_abs = 2.0 * pair_sum  # includes both i<j and i>j
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

    Expected shapes:
      • If coin_last=True:  psi[..., N, d]   (pos then coin axis)
      • If coin_last=False: psi[..., d, N]   (coin then pos axis)
    where d is the local coin dimension per position (assumed constant here).

    Returns:
      P: (..., N) with P[v] = sum_{a=1..d} |psi[..., v, a]|^2   (or adapted axes)

    Notes:
      • For irregular graphs with varying local dv, use a mask per-position (not
        handled here). This utility covers the common constant-d case.
      • Gradient-safe: uses squared magnitudes; no detach.
    """
    if psi.is_complex():
        x2 = (psi.conj() * psi).real
    else:
        x2 = psi**2

    if coin_last:
        # prob along coin axis = last; positions at pos_dim (default -2)
        P = x2.sum(dim=-1)  # sum over coin
        # Ensure position axis is last in returned vector shape
        if pos_dim != -2:
            # Move the pos axis to the last and squeeze coin dimension out
            raise ValueError("When coin_last=True, pos_dim must index the position axis as -2.")
        return P
    else:
        # coin first, then pos; coin_dim arg unused in this branch
        P = x2.sum(dim=-2)  # sum over coin axis
        return P


# ----------------------------- Risk aggregations --------------------------- #


def cvar(
    values: torch.Tensor,
    alpha: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Conditional Value at Risk (CVaR_α) over the *lower* tail for minimization.

    For rewards (to maximize) over randomness ξ, you may want upper-tail CVaR.
    Here we implement the lower-tail CVaR by default (worst α-quantile).

    Args:
        values: (...,) tensor
        alpha:  in (0, 1], fraction of worst outcomes to average
        reduction: 'none' | 'mean' | 'sum' across all elements

    Returns:
        CVaR_α(values)
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    flat = values.reshape(-1)
    k = max(1, int(math.ceil(alpha * flat.numel())))
    # Smallest k elements
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

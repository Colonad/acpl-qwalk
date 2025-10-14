# acpl/objectives/metrics.py
from __future__ import annotations

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("acpl.objectives.metrics requires PyTorch to be installed.") from exc


TensorLike = torch.Tensor | np.ndarray


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _to_tensor(x: TensorLike, *, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device) if device is not None else x
    t = torch.from_numpy(np.asarray(x))
    return t.to(device=device) if device is not None else t


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Probability checks / utilities
# -----------------------------------------------------------------------------


def prob_sum_error(p: TensorLike, dim: int = -1) -> TensorLike:
    """
    Return |sum(p) - 1| reduced along `dim`.
    Preserves type: returns torch.Tensor for torch inputs, numpy.ndarray otherwise.
    """
    t = _to_tensor(p)
    err = (t.sum(dim=dim) - 1.0).abs()
    return err if isinstance(p, torch.Tensor) else _to_numpy(err)


def is_prob_vector(
    p: TensorLike,
    dim: int = -1,
    *,
    atol_sum: float = 1e-6,
    atol_neg: float = 1e-12,
) -> TensorLike:
    """
    Check if `p` is a valid probability vector along `dim`:
      - entries >= -atol_neg
      - sum close to 1 within atol_sum
    Returns a boolean tensor/array with the reduced shape (i.e., `p` with `dim` removed).
    """
    t = _to_tensor(p)
    # Nonnegativity (allow tiny negatives within atol_neg)
    nonneg = (t >= -float(atol_neg)).all(dim=dim)
    # Sums to ~1
    s_ok = (t.sum(dim=dim) - 1.0).abs() <= float(atol_sum)
    ok = nonneg & s_ok
    return ok if isinstance(p, torch.Tensor) else _to_numpy(ok)


def assert_is_prob_vector(
    p: TensorLike,
    dim: int = -1,
    *,
    atol_sum: float = 1e-6,
    atol_neg: float = 1e-12,
    name: str = "p",
) -> None:
    """
    Raise ValueError if `p` is not a valid probability vector along `dim`.
    """
    ok = is_prob_vector(p, dim=dim, atol_sum=atol_sum, atol_neg=atol_neg)
    if isinstance(ok, np.ndarray):
        valid = bool(ok.all())
    else:
        valid = bool(ok.all().item())
    if not valid:
        raise ValueError(
            f"{name} is not a valid probability vector (dim={dim}): nonneg + sums≈1 violated."
        )


def renorm_simplex(
    p: TensorLike,
    dim: int = -1,
    *,
    clamp_min: float = 0.0,
    eps: float = 1e-12,
) -> TensorLike:
    """
    Clamp to [clamp_min, +inf) and renormalize to sum=1 along `dim`.
    If all mass is clamped to zero, falls back to a uniform distribution.
    """
    t = _to_tensor(p).to(dtype=torch.float32)
    t = torch.clamp(t, min=float(clamp_min))
    s = t.sum(dim=dim, keepdim=True)
    # If sum ~ 0, use uniform
    mask_zero = s.le(float(eps))
    if mask_zero.any():
        shape = list(t.shape)
        n = shape[dim]
        uniform = torch.full_like(t, 1.0 / float(n))
        t = torch.where(mask_zero.expand_as(t), uniform, t)
        s = t.sum(dim=dim, keepdim=True)
    out = t / s
    return out if isinstance(p, torch.Tensor) else _to_numpy(out)


# -----------------------------------------------------------------------------
# Distances / divergences
# -----------------------------------------------------------------------------


def tv_distance(
    p: TensorLike,
    q: TensorLike,
    dim: int = -1,
    *,
    keepdim: bool = False,
) -> TensorLike:
    """
    Total-variation distance TV(p,q) = 0.5 * ||p - q||_1 along `dim`.
    No internal renormalization; call `renorm_simplex` first if needed.
    """
    pt = _to_tensor(p)
    qt = _to_tensor(q, device=pt.device).to(dtype=pt.dtype)
    d = 0.5 * (pt - qt).abs().sum(dim=dim, keepdim=keepdim)
    return d if isinstance(p, torch.Tensor) or isinstance(q, torch.Tensor) else _to_numpy(d)


def l1_distance(
    p: TensorLike,
    q: TensorLike,
    dim: int = -1,
    *,
    keepdim: bool = False,
) -> TensorLike:
    """
    Plain L1 distance ||p - q||_1 along `dim`. (Useful alongside TV.)
    """
    pt = _to_tensor(p)
    qt = _to_tensor(q, device=pt.device).to(dtype=pt.dtype)
    d = (pt - qt).abs().sum(dim=dim, keepdim=keepdim)
    return d if isinstance(p, torch.Tensor) or isinstance(q, torch.Tensor) else _to_numpy(d)


def mse_distance(
    p: TensorLike,
    q: TensorLike,
    dim: int = -1,
    *,
    keepdim: bool = False,
) -> TensorLike:
    """
    Mean-squared error along `dim` (not a probability metric, but handy for debugging).
    """
    pt = _to_tensor(p)
    qt = _to_tensor(q, device=pt.device).to(dtype=pt.dtype)
    d = ((pt - qt) ** 2).mean(dim=dim, keepdim=keepdim)
    return d if isinstance(p, torch.Tensor) or isinstance(q, torch.Tensor) else _to_numpy(d)

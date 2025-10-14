# acpl/objectives/transfer.py
from __future__ import annotations

import torch

from .metrics import assert_is_prob_vector, renorm_simplex


def _gather_target_prob(
    p: torch.Tensor,
    j_star: int | torch.Tensor,
) -> torch.Tensor:
    """
    Gather P[j_star] from p shaped (V,) or (V,B).

    Returns a 0-D tensor (scalar) for (V,)
    or a (B,) tensor for (V,B).
    """
    if p.ndim == 1:
        if isinstance(j_star, torch.Tensor):
            if j_star.numel() != 1:
                raise ValueError("j_star must be scalar for 1-D p.")
            j = int(j_star.item())
        else:
            j = int(j_star)
        return p[j]

    if p.ndim == 2:
        v, b = p.shape
        if isinstance(j_star, int):
            return p[j_star, :]  # (B,)
        if not isinstance(j_star, torch.Tensor):
            raise TypeError("j_star must be int or a 1-D torch.Tensor of indices.")
        if j_star.ndim != 1 or j_star.shape[0] != b:
            raise ValueError("j_star must have shape (B,) for p with shape (V,B).")
        cols = torch.arange(b, device=p.device, dtype=torch.long)
        return p[j_star.long(), cols]  # (B,)

    raise ValueError(f"p must be 1-D or 2-D (got ndim={p.ndim}).")


def success_prob(
    p: torch.Tensor,
    j_star: int | torch.Tensor,
    *,
    check_prob: bool = False,
    renorm: bool = False,
) -> torch.Tensor:
    """
    Success probability: P(j_star).

    Parameters
    ----------
    p : torch.Tensor
        Partial-trace position distribution, shape (V,) or (V,B).
    j_star : int or torch.Tensor
        Target vertex index. If p is (V,B), may be a length-B LongTensor
        of per-batch targets.
    check_prob : bool
        If True, assert p is a valid probability vector along dim=0.
    renorm : bool
        If True, renormalize p to the simplex along dim=0 (safe clamp & renorm).

    Returns
    -------
    torch.Tensor
        Scalar (0-D) if p is (V,), or (B,) if p is (V,B).
    """
    if p.ndim not in (1, 2):
        raise ValueError(f"p must be 1-D or 2-D (got ndim={p.ndim}).")

    if renorm:
        p = renorm_simplex(p, dim=0)  # type: ignore[assignment]
    elif check_prob:
        assert_is_prob_vector(p, dim=0, name="p")

    return _gather_target_prob(p, j_star)


def loss_state_transfer(
    p: torch.Tensor,
    j_star: int | torch.Tensor,
    *,
    reduction: str = "mean",
    check_prob: bool = False,
    renorm: bool = False,
) -> torch.Tensor:
    """
    State-transfer loss using TV distance to the delta target e_{j*}.

    For valid probability vectors p, TV(p, e_{j*}) = 1 - p[j*].
    We return:
        - scalar if p is (V,)
        - reduced scalar (mean/sum) or vector (none) if p is (V,B)

    Parameters
    ----------
    p : torch.Tensor
        Position distribution from partial trace, shape (V,) or (V,B).
    j_star : int or torch.Tensor
        Target vertex. If p is (V,B), may be a length-B LongTensor
        of per-batch targets.
    reduction : {"mean", "sum", "none"}
        Reduction across batch when p is (V,B). Ignored for 1-D p.
    check_prob : bool
        If True, assert p is a proper probability vector along dim=0.
    renorm : bool
        If True, clamp and renormalize p to the simplex along dim=0.

    Returns
    -------
    torch.Tensor
        Loss: scalar for (V,), or scalar/vector depending on `reduction` for (V,B).
    """
    p_js = success_prob(p, j_star, check_prob=check_prob, renorm=renorm)

    # Loss = 1 - success probability
    loss = 1.0 - p_js

    if p.ndim == 1 or reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be 'mean', 'sum', or 'none'.")

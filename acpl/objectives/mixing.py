# acpl/objectives/mixing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import torch
import torch.nn as nn

__all__ = [
    "tv_distance_to_uniform",
    "tv_curve_from_prob",
    "tv_curve_from_state",
    "aggregate_curve",
    "MixingLossConfig",
    "MixingObjective",
]


# ----------------------------- #
#    Utilities & core metrics   #
# ----------------------------- #


def _ensure_bool_mask(
    mask: torch.Tensor | None, *, ref: torch.Tensor, node_dim: int = -1
) -> torch.Tensor | None:
    """
    Normalize a node mask to boolean dtype and broadcast it to `ref` except along the specified node dimension.

    Parameters
    ----------
    mask : (N,) or (..., N) or None
        True at valid nodes; False for padded/invalid nodes. Will be broadcasted to match `ref` shape on non-node axes.
    ref : Tensor
        Reference tensor for broadcasting, e.g., probabilities of shape (B?, T?, N).
    node_dim : int
        Axis along which nodes lie inside `ref`.

    Returns
    -------
    mask_b : Tensor[bool] | None
        Boolean mask broadcastable to `ref` (same shape as `ref` except possibly singleton on non-node axes), or None.
    """
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        mask = mask != 0
    # Move node axis index into positive form
    node_axis = node_dim if node_dim >= 0 else (ref.dim() + node_dim)
    # Build target shape with 1s except node axis, which must equal N
    target_shape = [1] * ref.dim()
    target_shape[node_axis] = ref.shape[node_axis]

    # If mask is 1D (N,), reshape; else try to broadcast to target non-node dims = 1
    if mask.dim() == 1:
        mask = mask.view(target_shape)
    else:
        # Ensure last dim (or corresponding node axis) matches N; other dims must be 1 or broadcastable
        if mask.shape[-1] == ref.shape[node_axis] and node_axis == ref.dim() - 1:
            # Already aligned at last axis; we will add singleton dims to the left if needed
            while mask.dim() < ref.dim():
                mask = mask.unsqueeze(0)
        else:
            # Try to reshape to target mask (same N on node axis, 1 elsewhere)
            mask = mask
            # Collapse non-node dims (if any) by checking broadcast consistency
            # We accept shapes already broadcastable; F.broadcast_shapes is only in newer PyTorch,
            # so we rely on actual arithmetic broadcast in ops downstream. Here we just place a view when possible.
            pass

        # Finally, expand with ones to target shape on non-node dims
        # We don't expand to full ref yet; returning a shape that's broadcastable is enough.
        desired = torch.ones(target_shape, dtype=mask.dtype, device=mask.device)
        mask = mask & desired.bool() | mask  # no-op but keeps dtype/broadcast-friendly

    return mask


def _masked_sum(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return x.sum(dim=dim)
    # Broadcast mask
    m = mask
    # Convert mask to same dtype for multiplication
    return (x * m.to(dtype=x.dtype)).sum(dim=dim)


def _masked_mean(
    x: torch.Tensor, mask: torch.Tensor | None, dim: int, eps: float = 1e-12
) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    m = mask.to(dtype=x.dtype)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(eps)
    return num / den


def _safe_normalize_prob(
    P: torch.Tensor,
    mask: torch.Tensor | None = None,
    node_dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Ensure that probabilities sum to 1 over valid nodes (per batch/time item) with optional masking.
    """
    if mask is None:
        Z = P.sum(dim=node_dim, keepdim=True).clamp_min(eps)
        return P / Z
    m = mask.to(dtype=P.dtype)
    Z = (P * m).sum(dim=node_dim, keepdim=True).clamp_min(eps)
    return P / Z


def tv_distance_to_uniform(
    P: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    node_dim: int = -1,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""
    Total-variation distance between a (possibly masked) distribution P and the uniform distribution over valid nodes.

    TV(P, U) = 1/2 * sum_{v in valid} |P(v) - U(v)|,

    where U is uniform on the set of valid nodes (size K) and zero elsewhere. If `mask is None`, K = N.

    Parameters
    ----------
    P : Tensor
        Probability tensor whose last (or node_dim) axis is nodes.
        Allowed shapes: (N,), (T, N), (B, T, N), (B, N) etc.
    mask : Tensor[bool] or None
        Node mask (True=valid) of shape (N,) or broadcastable to P except along the node axis.
    node_dim : int
        Dimension index of nodes in P.
    normalize : bool
        If True, (re)normalize P over valid nodes so that probabilities sum to 1 on the masked set.
        This both stabilizes and ensures physical correctness when P is obtained via partial sums.
    eps : float
        Numerical floor for denominators.

    Returns
    -------
    tv : Tensor
        TV distance with shape P.shape without the node dimension (i.e., reduced over `node_dim`).
        Example: if P is (B, T, N), return (B, T).
    """
    if P.numel() == 0:
        raise ValueError("P has no elements.")

    # Align mask to P
    mask_b = _ensure_bool_mask(mask, ref=P, node_dim=node_dim)

    # Optionally (re)normalize over valid nodes
    Q = _safe_normalize_prob(P, mask=mask_b if normalize else None, node_dim=node_dim, eps=eps)

    # Compute K = number of valid nodes
    if mask_b is None:
        K = P.shape[node_dim]
        U = 1.0 / float(K)
        diff = (Q - U).abs()
        tv = 0.5 * diff.sum(dim=node_dim)
        return tv

    # With mask: U(v) = 1/K if mask[v]=True else 0
    m = mask_b.to(dtype=P.dtype)
    K = m.sum(dim=node_dim, keepdim=True).clamp_min(
        1.0
    )  # avoid divide-by-zero; if zero valid nodes, treat as 1
    U = m / K  # broadcasted uniform over valid nodes
    diff = (Q - U).abs() * m  # Ignore invalid nodes contributions
    tv = 0.5 * diff.sum(dim=node_dim)
    return tv


# ------------------------------------- #
#   Curves from probabilities / states  #
# ------------------------------------- #


def tv_curve_from_prob(
    P: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    node_dim: int = -1,
    time_dim: int | None = -2,
    batch_dim: int | None = 0,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""
    Compute per-time-step TV distance curve from probabilities.

    Parameters
    ----------
    P : Tensor
        Probabilities over nodes. Typical shapes:
        - (T, N)  : single episode/time series
        - (B, T, N): batch of episodes
        Node dimension must be `node_dim`. If `time_dim` is None, treat P as a single distribution (no curve).
    mask : Tensor[bool] or None
        Node mask (True=valid nodes), shape (N,) or broadcastable to P.
    node_dim : int
        Axis of nodes.
    time_dim : int or None
        Axis of time. If None, returns a scalar TV (no curve).
    batch_dim : int or None
        Which axis is batch (if any); not strictly needed, only helpful for clarity in mask broadcasting.
    normalize : bool
        Re-normalize P across valid nodes before computing TV at each t.
    eps : float
        Numerical floor.

    Returns
    -------
    tv_t : Tensor
        TV curve over time with shape:
        - If P is (T, N): (T,)
        - If P is (B, T, N): (B, T)
        - If time_dim is None (e.g., P is (B, N)): (B,)
    """
    if time_dim is None:
        # Single distribution; reduce over nodes only
        return tv_distance_to_uniform(P, mask=mask, node_dim=node_dim, normalize=normalize, eps=eps)

    # Ensure contiguous semantics for dims
    # We'll move node_dim to -1 and time_dim to -2 for simpler logic
    perm = list(range(P.dim()))
    node_axis = node_dim if node_dim >= 0 else (P.dim() + node_dim)
    time_axis = time_dim if time_dim >= 0 else (P.dim() + time_dim)

    # Bring to (..., T, N)
    # If node/time are already last two axes in that order, this is a no-op
    # Otherwise, permute
    target_order = [i for i in range(P.dim()) if i not in (time_axis, node_axis)] + [
        time_axis,
        node_axis,
    ]
    P2 = P.permute(*target_order)  # (..., T, N)
    tv = tv_distance_to_uniform(
        P2, mask=mask, node_dim=-1, normalize=normalize, eps=eps
    )  # (..., T)
    return tv


def tv_curve_from_state(
    psi: torch.Tensor,
    *,
    node_dim: int = -2,
    coin_dim: int = -1,
    time_dim: int | None = -3,
    mask: torch.Tensor | None = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""
    Compute the per-time-step TV curve directly from the state amplitudes by marginalizing coin DOFs.

    We expect `psi` holds complex amplitudes in an arc/port basis that factors into position (nodes) × coin.

    PT(v) = sum_a |psi(v, a)|^2.
    TV_t = 1/2 * sum_v | PT_t(v) - U(v) | with U uniform on valid nodes.

    Parameters
    ----------
    psi : Tensor[complex]
        State amplitudes. Typical shapes:
        - (T, N, d_c)    (single episode)
        - (B, T, N, d_c) (batch)
        - without time: (N, d_c) or (B, N, d_c)
    node_dim : int
        Axis for nodes/positions inside `psi`.
    coin_dim : int
        Axis for coin. Will be reduced by sum of squared magnitudes.
    time_dim : int or None
        Axis for time; if None, returns a single TV value (no curve).
    mask : Tensor[bool] or None
        Node mask (True=valid nodes). If provided, uniform is over masked nodes only.
    normalize : bool
        Whether to (re)normalize the marginal position probabilities at each time over valid nodes.
    eps : float
        Numerical floor.

    Returns
    -------
    tv_t : Tensor
        TV curve over time (or scalar if `time_dim` is None). Same leading axes as `psi` without the coin axis.
    """
    # Probability over nodes = sum over coin ports of |psi|^2
    # Make sure we sum over coin_dim
    prob = (psi.abs() ** 2).sum(dim=coin_dim)
    # Now prob has shape like psi without coin axis; e.g., (T, N) or (B, T, N)
    # Delegate to probability-based curve
    # Find node/time dims in the new tensor (coin_dim removed)
    coin_axis = coin_dim if coin_dim >= 0 else (psi.dim() + coin_dim)

    # After removing coin axis, adjust indices if needed
    def _adjust(idx: int | None) -> int | None:
        if idx is None:
            return None
        idx2 = idx if idx >= 0 else (psi.dim() + idx)
        return idx2 - (1 if coin_axis < idx2 else 0)

    node_dim2 = _adjust(node_dim)
    time_dim2 = _adjust(time_dim)

    return tv_curve_from_prob(
        prob,
        mask=mask,
        node_dim=node_dim2 if node_dim2 is not None else -1,
        time_dim=time_dim2,
        normalize=normalize,
        eps=eps,
    )


# ------------------------------------- #
#      Curve aggregations for loss      #
# ------------------------------------- #


def aggregate_curve(
    tv_curve: torch.Tensor,
    *,
    mode: Literal["final", "min", "mean", "auc"] = "final",
    discount: float | None = None,
    time_dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""
    Aggregate a TV curve along time.

    Parameters
    ----------
    tv_curve : Tensor
        TV per time step; time is `time_dim`.
    mode : {"final", "min", "mean", "auc"}
        - "final": value at the last time step
        - "min"  : minimum TV over time (best mixing achieved)
        - "mean" : average TV across time steps
        - "auc"  : area under the curve (plain sum) or discounted sum if `discount` given
    discount : float or None
        If provided in "auc" mode, compute discounted sum: sum_t discount^t * tv_t.
        Use discount in (0,1] for emphasis on early steps; discount=1 -> plain sum.
    time_dim : int
        Time axis.
    eps : float
        Numerical floor for small denominators in "mean".

    Returns
    -------
    agg : Tensor
        Aggregated scalar per batch item (i.e., tv_curve with time axis removed).
    """
    if mode == "final":
        return tv_curve.select(dim=time_dim, index=tv_curve.size(time_dim) - 1)
    elif mode == "min":
        return tv_curve.min(dim=time_dim).values
    elif mode == "mean":
        return tv_curve.mean(dim=time_dim)
    elif mode == "auc":
        if discount is None or discount == 1.0:
            return tv_curve.sum(dim=time_dim)
        # Discounted sum
        T = tv_curve.size(time_dim)
        # Build weights [1, d, d^2, ...]
        device = tv_curve.device
        w = discount ** torch.arange(T, device=device, dtype=tv_curve.dtype)
        # Align w to broadcast over all leading dims
        # Move time_dim to last temporarily
        perm = list(range(tv_curve.dim()))
        t_axis = time_dim if time_dim >= 0 else (tv_curve.dim() + time_dim)
        target_order = [i for i in range(tv_curve.dim()) if i != t_axis] + [t_axis]
        x = tv_curve.permute(*target_order)  # (..., T)
        # Weighted sum over last dim
        agg = (x * w).sum(dim=-1)
        # Restore original leading-ordering (time removed)
        return agg
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


# ------------------------------------- #
#           Training objective          #
# ------------------------------------- #


@dataclass
class MixingLossConfig:
    """
    Configuration for the mixing-to-uniform objective.

    Fields
    ------
    normalize_prob : whether to (re)normalize P_t over valid nodes before TV
    curve_weight   : weight for curve aggregation term (e.g., AUC/mean/min)
    final_weight   : weight for final-time TV
    curve_mode     : aggregation over time: {"final","min","mean","auc"}; ignored if curve_weight=0
    discount       : optional discount factor in (0,1] for "auc" (if None or 1.0 -> plain sum)
    average_batch  : if True, average loss over batch; else return per-item
    reduce_time    : if True and input lacks time dim (single distribution), returns scalar (still respects average_batch)

    Notes
    -----
    - The objective is a minimization of TV distance(s): lower is better (closer to uniform).
    - Robust to masked/padded graphs: pass mask (True=valid nodes). Uniform is defined over the mask.
    """

    normalize_prob: bool = True
    curve_weight: float = 0.0
    final_weight: float = 1.0
    curve_mode: Literal["final", "min", "mean", "auc"] = "final"
    discount: float | None = None
    average_batch: bool = True
    reduce_time: bool = True


class MixingObjective(nn.Module):
    """
    Differentiable objective for fast mixing to the uniform distribution (total-variation distance).

    Supports both:
      1) passing probabilities P over nodes, or
      2) passing complex-state amplitudes psi and letting the module marginalize coin DOFs.
    """

    def __init__(self, cfg: MixingLossConfig):
        super().__init__()
        self.cfg = cfg

    @overload
    def forward(
        self,
        *,
        P: torch.Tensor,
        mask: torch.Tensor | None = None,
        node_dim: int = -1,
        time_dim: int | None = -2,
    ) -> dict[str, torch.Tensor]: ...

    @overload
    def forward(
        self,
        *,
        psi: torch.Tensor,
        mask: torch.Tensor | None = None,
        node_dim: int = -2,
        coin_dim: int = -1,
        time_dim: int | None = -3,
    ) -> dict[str, torch.Tensor]: ...

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        out : dict
            {
              "tv_curve": Tensor[..., T]  # if time_dim present, else scalar per item
              "tv_final": Tensor[...]     # per item, final TV (if curve is present) else same as tv_curve
              "loss":     Tensor[...] or scalar depending on average_batch
            }
        """
        cfg = self.cfg

        if "P" in kwargs:
            P = kwargs["P"]
            mask = kwargs.get("mask", None)
            node_dim = kwargs.get("node_dim", -1)
            time_dim = kwargs.get("time_dim", -2)
            tv_curve = tv_curve_from_prob(
                P,
                mask=mask,
                node_dim=node_dim,
                time_dim=time_dim,
                normalize=cfg.normalize_prob,
            )
        elif "psi" in kwargs:
            psi = kwargs["psi"]
            mask = kwargs.get("mask", None)
            node_dim = kwargs.get("node_dim", -2)
            coin_dim = kwargs.get("coin_dim", -1)
            time_dim = kwargs.get("time_dim", -3)
            tv_curve = tv_curve_from_state(
                psi,
                mask=mask,
                node_dim=node_dim,
                coin_dim=coin_dim,
                time_dim=time_dim,
                normalize=cfg.normalize_prob,
            )
        else:
            raise ValueError("MixingObjective.forward expects either P=... or psi=...")

        # If no time dimension was provided -> tv_curve is scalar per item
        has_time = (
            tv_curve.dim() > 0 and tv_curve.shape[-1] > 1
            if isinstance(tv_curve, torch.Tensor)
            else False
        )

        if has_time:
            # Final value at last time step
            tv_final = aggregate_curve(tv_curve, mode="final", time_dim=-1)
        else:
            tv_final = tv_curve  # scalar per item

        # Optional curve aggregation
        curve_term = None
        if cfg.curve_weight != 0.0 and has_time:
            curve_term = aggregate_curve(
                tv_curve, mode=cfg.curve_mode, discount=cfg.discount, time_dim=-1
            )

        # Compose loss (minimize TVs)
        if curve_term is not None:
            loss_items = cfg.final_weight * tv_final + cfg.curve_weight * curve_term
        else:
            loss_items = cfg.final_weight * tv_final

        if cfg.average_batch and loss_items.dim() > 0:
            loss = loss_items.mean()
        else:
            loss = loss_items

        out = {
            "tv_curve": tv_curve,
            "tv_final": tv_final,
            "loss": loss,
        }
        if curve_term is not None:
            out["tv_agg"] = curve_term
        return out

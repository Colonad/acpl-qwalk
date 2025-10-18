# acpl/objectives/transfer.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from acpl.sim.portmap import PortMap

# NEW: import our partial-trace helper
from acpl.sim.utils import check_probability_simplex, position_probabilities

from .metrics import (
    cvar,
    delta_prob,
    entropy,
    normalize_prob,
    success_on_targets,
    total_variation,
)

__all__ = [
    "TransferLossConfig",
    "aggregate_over_time",
    "prob_on_targets",
    "loss_state_transfer",
    # NEW
    "position_marginal_from_state",
    "loss_state_transfer_from_state",
]


# --------------------------- config & helpers --------------------------- #


@dataclass
class TransferLossConfig:
    """
    Configuration for **state transfer** objectives on position marginals P.

    Shapes expected by `loss_state_transfer`:
      * P: (N,) | (T, N) | (B, N) | (B, T, N)
      * targets: int | Sequence[int] | 1-D LongTensor

    Loss families (set `loss`):
      • "nll"        : minimize −log p(Ω)   (optionally with label smoothing)
      • "cvar_nll"   : CVaR_α of −log p(Ω) over (batch×time)
      • "neg_prob"   : minimize −p(Ω)  (i.e., maximize success mass)
      • "hinge"      : margin hinge on p(Ω) − max_{¬Ω} p, with `margin`

    Time aggregation (set `time_agg`):
      • "last"       : take last time step
      • "mean"       : arithmetic mean over T
      • "max"        : max over T (subgradient at ties)
      • "softmax"    : softmax-τ weighted mean over T, with temperature `tau`

    Other:
      • label_smoothing: ε in [0,1). With "nll", builds a smoothed target
        distribution y = (1−ε)·U(Ω) + ε·U(V), and computes CE(y, P).
      • reduce: "mean" | "sum"  (reduction across batch if present)
      • eps: numeric floor for logs & normalization
    """

    loss: Literal["nll", "cvar_nll", "neg_prob", "hinge"] = "nll"
    time_agg: Literal["last", "mean", "max", "softmax"] = "last"
    tau: float = 0.2  # temperature for softmax time agg
    label_smoothing: float = 0.0  # only used for "nll"
    cvar_alpha: float = 0.1  # only used for "cvar_nll"
    margin: float = 0.0  # only used for "hinge"
    reduce: Literal["mean", "sum"] = "mean"
    eps: float = 1e-8


def _ensure_targets_tensor(
    targets: int | Sequence[int] | torch.Tensor,
    *,
    device,
) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        t = targets.to(device=device, dtype=torch.long).view(-1)
    elif isinstance(targets, int):
        t = torch.tensor([targets], device=device, dtype=torch.long)
    else:
        t = torch.tensor(list(targets), device=device, dtype=torch.long)
    if t.numel() == 0:
        raise ValueError("targets must be non-empty.")
    return t


def _to_btn(P: torch.Tensor) -> tuple[torch.Tensor, bool, bool]:
    """
    Normalize shape to (B, T, N). Return (P_btn, had_batch, had_time).
    """
    if P.ndim == 1:  # (N,)
        return P.unsqueeze(0).unsqueeze(0), False, False
    if P.ndim == 2:  # (T, N)  OR  (B, N)
        return P.unsqueeze(0), False, True
    if P.ndim == 3:  # (B, T, N)
        return P, True, True
    raise ValueError("P must be (N,), (T,N), (B,N) or (B,T,N).")


# ------------------------------ time aggregation ----------------------------- #


def aggregate_over_time(
    P_btn: torch.Tensor,  # (B, T, N)
    mode: Literal["last", "mean", "max", "softmax"],
    *,
    tau: float = 0.2,
) -> torch.Tensor:
    """
    Aggregate probabilities over time into a single (B, N) distribution per batch.

    Returns:
        P_bN: (B, N)
    """
    B, T, N = P_btn.shape
    if mode == "last":
        return P_btn[:, -1, :]
    if mode == "mean":
        return P_btn.mean(dim=1)
    if mode == "max":
        return P_btn.max(dim=1).values
    if mode == "softmax":
        if tau <= 0:
            raise ValueError("tau must be positive for softmax aggregation.")
        w = torch.softmax(P_btn / float(tau), dim=1)
        return (w * P_btn).sum(dim=1)
    raise ValueError("Unknown time aggregation mode.")


# ---------------------------- target probability ---------------------------- #


def prob_on_targets(
    P: torch.Tensor,  # (N,) | (T,N) | (B,N) | (B,T,N)
    targets: int | Sequence[int] | torch.Tensor,
    *,
    time_agg: Literal["last", "mean", "max", "softmax"] = "last",
    tau: float = 0.2,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute success probability mass on Ω (targets).

    Returns:
        (p_omega, P_bN)
        p_omega: (B,) success mass per batch element after time aggregation.
        P_bN:    (B, N) aggregated position probability per batch.
    """
    P_btn, had_batch, _ = _to_btn(P)  # (B,T,N)
    device = P_btn.device
    P_btn = normalize_prob(P_btn, dim=-1, eps=eps)
    P_bN = aggregate_over_time(P_btn, mode=time_agg, tau=tau)  # (B,N)
    t_idx = _ensure_targets_tensor(targets, device=device)
    p_omega = success_on_targets(P_bN, t_idx, dim=-1, reduction="none")  # (B,)
    return p_omega, P_bN


# ------------------------------- main objective ------------------------------ #


def _reduce(v: torch.Tensor, how: Literal["mean", "sum"]) -> torch.Tensor:
    return v.mean() if how == "mean" else v.sum()


def _nll_smoothed(
    P_bN: torch.Tensor,  # (B,N)
    targets: torch.Tensor,  # (K,)
    eps: float,
    smooth_eps: float,
) -> torch.Tensor:
    """
    Cross-entropy with a smoothed target distribution:
      y = (1−ε)·U(Ω) + ε·U(V)
    Returns per-batch loss vector (B,).
    """
    B, N = P_bN.shape
    K = max(1, int(targets.numel()))
    device = P_bN.device
    dtype = P_bN.dtype

    y = torch.full((N,), smooth_eps / N, device=device, dtype=dtype)
    if K > 0:
        y_omega = (1.0 - smooth_eps) / K
        y.index_fill_(0, targets, y_omega + y[targets] - (smooth_eps / N))
    y = normalize_prob(y, dim=0, eps=eps)
    P_bN = P_bN.clamp_min(eps)
    ce = -(y.unsqueeze(0) * P_bN.log()).sum(dim=-1)  # (B,)
    return ce


def loss_state_transfer(
    P: torch.Tensor,  # (N,) | (T,N) | (B,N) | (B,T,N)
    targets: int | Sequence[int] | torch.Tensor,
    cfg: TransferLossConfig | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Differentiable **state transfer** objective operating on position marginals.

    Args:
        P: position probabilities (or unnormalized non-negative scores).
        targets: the destination vertex/vertices Ω.
        cfg: TransferLossConfig (see class docstring). Defaults to NLL + last.

    Returns:
        (loss, info)
        loss: scalar tensor
        info: dict with diagnostics (p_omega, entropy, tv_to_delta*, etc.)
    """
    if cfg is None:
        cfg = TransferLossConfig()

    P_btn, _, _ = _to_btn(P)  # (B,T,N)
    device = P_btn.device
    eps = float(cfg.eps)

    t_idx = _ensure_targets_tensor(targets, device=device)

    # Aggregate over time, and get success per batch
    p_omega, P_bN = prob_on_targets(
        P_btn, t_idx, time_agg=cfg.time_agg, tau=cfg.tau, eps=eps
    )  # (B,), (B,N)

    # --- build scalar loss per sample ---
    if cfg.loss == "neg_prob":
        per = -p_omega
    elif cfg.loss == "nll":
        if cfg.label_smoothing and cfg.label_smoothing > 0.0:
            per = _nll_smoothed(P_bN, t_idx, eps=eps, smooth_eps=float(cfg.label_smoothing))
        else:
            per = -(p_omega.clamp_min(eps).log())
    elif cfg.loss == "cvar_nll":
        per_nll = -(p_omega.clamp_min(eps).log())
        loss = cvar(per_nll, alpha=float(cfg.cvar_alpha), reduction="mean")
        with torch.no_grad():
            info = {
                "p_omega_mean": float(p_omega.mean().item()),
                "p_omega_min": float(p_omega.min().item()),
                "p_omega_max": float(p_omega.max().item()),
                "entropy_mean": float(entropy(P_bN, dim=-1, reduction="mean").item()),
            }
        return loss, info
    elif cfg.loss == "hinge":
        B, N = P_bN.shape
        mask = torch.ones(N, device=device, dtype=torch.bool)
        mask[t_idx] = False
        p_non = P_bN[:, mask].max(dim=-1).values if mask.any() else torch.zeros_like(p_omega)
        per = torch.relu(float(cfg.margin) - (p_omega - p_non))
    else:
        raise ValueError(f"Unknown loss type: {cfg.loss}")

    loss = _reduce(per, cfg.reduce)

    with torch.no_grad():
        B, N = P_bN.shape
        if t_idx.numel() == 1:
            delta = delta_prob(N, int(t_idx.item()), device=device, dtype=P_bN.dtype)
            tv = total_variation(P_bN, delta, dim=-1, reduction="mean").item()
        else:
            vals = P_bN.index_select(-1, t_idx)  # (B,K)
            best = t_idx[vals.argmax(dim=-1)]  # (B,)
            deltas = torch.stack(
                [delta_prob(N, int(b.item()), device=device, dtype=P_bN.dtype) for b in best], dim=0
            )
            tv = total_variation(P_bN, deltas, dim=-1, reduction="mean").item()

        info = {
            "p_omega_mean": float(p_omega.mean().item()),
            "p_omega_min": float(p_omega.min().item()),
            "p_omega_max": float(p_omega.max().item()),
            "entropy_mean": float(entropy(P_bN, dim=-1, reduction="mean").item()),
            "tv_to_delta": float(tv),
        }

    return loss, info


# --------------------------- NEW: simulator adapters --------------------------- #


def position_marginal_from_state(
    psi: torch.Tensor,  # (A,) | (T,A) | (B,A) | (B,T,A) complex
    pm: PortMap,
    *,
    normalize: bool = True,
    check_simplex: bool = False,
) -> torch.Tensor:
    """
    Convert **arc-basis** complex state(s) ψ into **position marginals** P.

    Shapes:
      - Input  : (A,), (T,A), (B,A), (B,T,A)
      - Output : (N,), (T,N), (B,N), (B,T,N)

    Notes:
      - Uses the batched path in `position_probabilities`, so everything is
        differentiable w.r.t. ψ (since |ψ|^2 is used).
      - `normalize=True` ensures each slice over the last axis sums to 1.
    """
    if psi.ndim == 1:
        P = position_probabilities(psi, pm, normalize=normalize)  # (N,)
        if check_simplex:
            check_probability_simplex(P)
        return P

    A = pm.src.numel()
    if psi.shape[-1] != A:
        raise ValueError(f"psi last dim must be A={A}, got {psi.shape[-1]}.")

    if psi.ndim == 2:  # (T,A) or (B,A) — treat first dim as batch
        P = position_probabilities(psi, pm, normalize=normalize)  # (T,N) or (B,N)
        if check_simplex:
            # Validate each row
            if P.ndim == 2:
                for r in range(P.shape[0]):
                    check_probability_simplex(P[r])
            else:
                check_probability_simplex(P)
        return P

    if psi.ndim == 3:  # (B,T,A)
        B, T, A_ = psi.shape
        flat = psi.reshape(B * T, A_)
        P_flat = position_probabilities(flat, pm, normalize=normalize)  # (B*T, N)
        N = P_flat.shape[-1]
        P = P_flat.view(B, T, N).contiguous()
        if check_simplex:
            for b in range(B):
                for t in range(T):
                    check_probability_simplex(P[b, t])
        return P

    raise ValueError("psi must be (A,), (T,A), (B,A), or (B,T,A).")


def loss_state_transfer_from_state(
    psi: torch.Tensor,  # (A,) | (T,A) | (B,A) | (B,T,A)
    pm: PortMap,
    targets: int | Sequence[int] | torch.Tensor,
    cfg: TransferLossConfig | None = None,
    *,
    normalize: bool = True,
    check_simplex: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Convenience wrapper: ψ (arc-basis) → P (position) → transfer loss.

    Keeps autograd intact:
      ψ  --(partial trace)-->  P  --(objective)-->  loss

    Args:
        psi: complex state(s) over arcs.
        pm: PortMap for the graph (CSR grouping).
        targets: vertex id(s) for successful transfer set Ω.
        cfg: TransferLossConfig (loss/time-agg/etc).
        normalize: ensure each slice of P sums to 1 (recommended).
        check_simplex: run numeric simplex assertions (debug).

    Returns:
        (loss, info) as in `loss_state_transfer`.
    """
    P = position_marginal_from_state(psi, pm, normalize=normalize, check_simplex=check_simplex)
    return loss_state_transfer(P, targets=targets, cfg=cfg)

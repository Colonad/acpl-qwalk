# acpl/objectives/search.py
# Copyright (c) 2025
# ACPL: Oracle-marked search objectives (differentiable & RL variants)

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = [
    "OracleSearchDiffConfig",
    "OracleSearchDiffObjective",
    "OracleSearchRLConfig",
    "OracleSearchRLEnvReward",
    "aggregate_mark_success",
    "prepare_batch_mask",
]


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def _validate_probs(p: torch.Tensor) -> None:
    if not torch.is_floating_point(p):
        raise TypeError("Position probabilities tensor must be floating point.")
    if p.dim() not in (1, 2):
        raise ValueError(f"Expected P to have shape (N,) or (B,N); got {tuple(p.shape)}.")
    if torch.any(p < -1e-6) or torch.any(p > 1 + 1e-6):
        # We allow tiny numeric drift
        raise ValueError("Probabilities appear out of [0,1] range.")


def prepare_batch_mask(
    mark_mask: torch.Tensor,
    *,
    batch: torch.Tensor | None = None,
    num_graphs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """
    Normalize/validate mark mask & batch vector.

    Parameters
    ----------
    mark_mask : (N,) or (B,N) bool/float tensor
        Node-level indicator(s) for marked set Ω. Nonzero = marked.
    batch : (N,) LongTensor or None
        Graph assignment per node (0..G-1). Required if you use ragged graphs packed
        as a single (N,) vector AND want per-graph aggregation.
    num_graphs : int or None
        Number of graphs G when using `batch`. If None, inferred from batch.max()+1.

    Returns
    -------
    mask_flat : (N,) float32 in {0,1}
    batch : (N,) or None
    G : int  (>=1)
    """
    if mark_mask.dim() == 2:
        # (B, N) → flatten into (B*N,) and synthesize batch
        B, N = mark_mask.shape
        mask_flat = (mark_mask != 0).to(torch.float32).reshape(B * N)
        if batch is None:
            device = mark_mask.device
            batch = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N)
        G = B if num_graphs is None else num_graphs
        return mask_flat, batch, int(G)

    if mark_mask.dim() != 1:
        raise ValueError("mark_mask must be 1D (N,) or 2D (B,N).")
    mask_flat = (mark_mask != 0).to(torch.float32)

    if batch is None:
        # Single graph implied
        return mask_flat, None, 1

    if batch.dim() != 1 or batch.numel() != mark_mask.numel():
        raise ValueError("batch must be 1D with same length as mark_mask when provided.")
    G = int(batch.max().item() + 1) if num_graphs is None else int(num_graphs)
    return mask_flat, batch, G


def aggregate_mark_success(
    P: torch.Tensor,
    mark_mask: torch.Tensor,
    *,
    batch: torch.Tensor | None = None,
    reduce: str = "mean",
) -> torch.Tensor:
    r"""
    Compute success probability on a marked set Ω:
        s_g = ∑_{v in Ω_g} P_g(v)

    Supports:
      • Single graph: P ∈ (N,), mark_mask ∈ (N,) → scalar
      • Mini-batch (stacked): P ∈ (B,N), mark_mask ∈ (B,N) → (B,)
      • Packed ragged graphs: P ∈ (N,), mark_mask ∈ (N,), batch ∈ (N,) → (G,)

    Parameters
    ----------
    P : tensor
        Position probabilities at terminal (or current) time. Shape (N,) or (B,N).
    mark_mask : tensor
        Indicator(s) for marked nodes; nonzero = marked. Matches shape of P
        except the ragged case where you also pass `batch`.
    batch : tensor, optional
        Graph assignment per node for ragged packing, shape (N,). Ignored if P is (B,N).
    reduce : {"none","mean","sum"}
        If "none": return per-graph vector; else reduce over graphs.

    Returns
    -------
    Tensor
        Scalar if single-graph and reduction != "none", else per-graph vector.
    """
    _validate_probs(P)
    if P.dim() == 2:
        if mark_mask.dim() != 2 or mark_mask.shape != P.shape:
            raise ValueError("For P with shape (B,N), mark_mask must have shape (B,N).")
        success = (P * (mark_mask != 0).to(P)).sum(dim=1)  # (B,)
        if reduce == "none":
            return success
        if reduce == "sum":
            return success.sum()
        return success.mean()

    # 1D P
    mask_flat, batch_vec, G = prepare_batch_mask(mark_mask, batch=batch)

    if batch_vec is None:
        success = (P * mask_flat.to(P)).sum()
        return success if reduce != "none" else success.unsqueeze(0)

    # Ragged: segment sum by batch
    if batch_vec.dtype != torch.long:
        batch_vec = batch_vec.long()
    if batch_vec.numel() != P.numel():
        raise ValueError("batch length must match P length in ragged mode.")
    success = torch.zeros(G, device=P.device, dtype=P.dtype)
    success.index_add_(0, batch_vec.to(torch.long), P * mask_flat.to(P))
    if reduce == "none":
        return success
    if reduce == "sum":
        return success.sum()
    return success.mean()


# --------------------------------------------------------------------------------------
# Differentiable (supervised) objective
# --------------------------------------------------------------------------------------


@dataclass
class OracleSearchDiffConfig:
    """
    Differentiable oracle-marked search loss L = -f, where
        f = E_batch[ success(P_T, Ω) ] with optional risk shaping.

    Fields
    ------
    use_log : bool
        Use -log(ε + success) instead of -success for steeper gradients at low success.
    eps : float
        Numerical ε for log barrier.
    time_average : bool
        If True, also reward mid-episode accumulation on Ω via a convex combination
        over provided P_t's (see `time_weights`).
    time_weights : Optional[Sequence[float]]
        Nonnegative weights for a list/tuple of P_t provided to .loss(...).
        If None and time_average=True, uses uniform weights over timesteps.
    cvar_alpha : float in (0,1]
        CVaR tail mass; α=1 means standard mean (risk-neutral).
        We *maximize* expected success, so CVaR is computed on the *loss* side.
    detach_probs : bool
        If True, treat input probabilities as constants (for ablations/diagnostics).
    label_smoothing : float in [0,1)
        Softens the "target=1 on Ω" to 1 - smoothing (helps stability on tiny Ω).
    clamp_prob : Optional[Tuple[float,float]]
        If provided, clamp P into [lo, hi] before computing success to avoid NaNs.
    """

    use_log: bool = True
    eps: float = 1e-6
    time_average: bool = False
    time_weights: Sequence[float] | None = None
    cvar_alpha: float = 1.0
    detach_probs: bool = False
    label_smoothing: float = 0.0
    clamp_prob: tuple[float, float] | None = (0.0, 1.0)


class OracleSearchDiffObjective(nn.Module):
    """
    Differentiable oracle-marked search objective.

    Usage
    -----
    obj = OracleSearchDiffObjective(cfg)
    loss = obj.loss(P_T, mark_mask, batch=batch)                       # terminal-only
    loss = obj.loss([P_t0, P_t1, ... , P_T], mark_mask, batch=batch)   # time-averaged
    """

    def __init__(self, cfg: OracleSearchDiffConfig):
        super().__init__()
        self.cfg = cfg

    def _score(
        self,
        P: torch.Tensor | Sequence[torch.Tensor],
        mark_mask: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute per-graph success score (no reduction).

        P : (N,), (B,N) or a sequence of such tensors (time series).
        Returns vector of shape (G,) where G is #graphs in batch.
        """
        cfg = self.cfg

        if isinstance(P, (list, tuple)):
            if not cfg.time_average:
                # If time series was passed but averaging disabled, use the last.
                P = P[-1]
                return self._score(P, mark_mask, batch=batch)

            # Weighted average over time
            T = len(P)
            if T == 0:
                raise ValueError("Empty list of probabilities.")
            weights = cfg.time_weights
            if weights is None:
                w = torch.full((T,), 1.0 / T, dtype=torch.float32, device=P[-1].device)
            else:
                if len(weights) != T:
                    raise ValueError("time_weights must match the number of P_t provided.")
                w = torch.tensor(weights, dtype=torch.float32, device=P[-1].device)
                s = w.clamp_min(0).sum()
                if s <= 0:
                    raise ValueError("time_weights must have positive sum.")
                w = w / s

            # accumulate per-graph success over time with weights
            per_graph = None
            for t, Pt in enumerate(P):
                Pt = Pt.detach() if cfg.detach_probs else Pt
                if cfg.clamp_prob is not None:
                    lo, hi = cfg.clamp_prob
                    Pt = Pt.clamp(min=lo, max=hi)
                st = aggregate_mark_success(Pt, mark_mask, batch=batch, reduce="none")  # (G,)
                per_graph = (w[t] * st) if per_graph is None else (per_graph + w[t] * st)
            return per_graph

        # Single snapshot
        P = P.detach() if cfg.detach_probs else P
        if cfg.clamp_prob is not None:
            lo, hi = cfg.clamp_prob
            P = P.clamp(min=lo, max=hi)
        per_graph = aggregate_mark_success(P, mark_mask, batch=batch, reduce="none")  # (G,)
        return per_graph

    def _to_loss(self, success: torch.Tensor) -> torch.Tensor:
        """
        Convert per-graph success scores to scalar loss using optional log barrier and CVaR.
        """
        cfg = self.cfg
        # Optional label smoothing: target=1-s on marked set raises effective ceiling slightly below 1
        # We implement it by shrinking success toward 0 (i.e., "target" < 1); equivalently, reduce reward a bit.
        if cfg.label_smoothing > 0:
            s = float(cfg.label_smoothing)
            # success' = (1 - s) * success
            success = (1.0 - s) * success

        # Base negative objective
        if cfg.use_log:
            # Maximize success → minimize -log(ε + success)
            loss_vec = -torch.log(success.clamp_min(cfg.eps))
        else:
            # Simpler: maximize success → minimize -success
            loss_vec = -success

        # CVaR over the *loss* (risk-averse): tail-mean of worst α-fraction
        alpha = float(cfg.cvar_alpha)
        if not (0.0 < alpha <= 1.0):
            raise ValueError("cvar_alpha must be in (0,1].")
        if alpha < 1.0 and loss_vec.numel() > 1:
            k = max(1, int(alpha * loss_vec.numel()))
            # top-k largest losses (worst cases)
            topk, _ = torch.topk(loss_vec, k=k, largest=True, sorted=False)
            return topk.mean()

        return loss_vec.mean()

    @torch.no_grad()
    def success(
        self,
        P: torch.Tensor | Sequence[torch.Tensor],
        mark_mask: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
        reduce: str = "mean",
    ) -> torch.Tensor:
        """
        Convenience metric: return success on Ω.

        If `P` is a list of P_t and time_average=True in config, returns the weighted
        time-averaged success; otherwise returns terminal success.
        """
        if isinstance(P, (list, tuple)) and self.cfg.time_average:
            per_graph = self._score(P, mark_mask, batch=batch)
            if reduce == "none":
                return per_graph
            return per_graph.mean() if reduce == "mean" else per_graph.sum()

        # Terminal-only
        if isinstance(P, (list, tuple)):
            P = P[-1]
        return aggregate_mark_success(P, mark_mask, batch=batch, reduce=reduce)

    def loss(
        self,
        P: torch.Tensor | Sequence[torch.Tensor],
        mark_mask: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute scalar loss to minimize.

        Parameters
        ----------
        P : (N,), (B,N), or Sequence of such
            Position probabilities (terminal or per-time). If a sequence is provided
            and `cfg.time_average=True`, a convex combination is used internally.
        mark_mask : (N,), (B,N), or ragged (N,) with `batch`
            Oracle marks (nonzero = marked).
        batch : (N,) or None
            Graph assignment per node for ragged packing.

        Returns
        -------
        loss : scalar tensor
        """
        per_graph_success = self._score(P, mark_mask, batch=batch)  # (G,)
        return self._to_loss(per_graph_success)


# --------------------------------------------------------------------------------------
# RL reward (environment-facing)
# --------------------------------------------------------------------------------------


@dataclass
class OracleSearchRLConfig:
    """
    RL reward shaping for oracle-marked search.

    Fields
    ------
    terminal_only : bool
        If True, r_T = success(P_T, Ω), r_t=0 for t<T (sparse terminal reward).
        If False, use potential-shaping dense reward on every step.
    shaping_mode : {"delta","absolute"}
        Dense shaping mode if terminal_only=False:
          - "delta": r_t = κ * (success(P_t) - success(P_{t-1}))
          - "absolute": r_t = κ * success(P_t)
    kappa : float
        Scaling for dense shaping.
    gamma : float
        Discount factor used by the agent (for info); we do not apply discount inside
        this object (the learner/algorithm will).
    clamp_prob : Optional[Tuple[float,float]]
        Clamp probabilities before scoring (safety).
    """

    terminal_only: bool = True
    shaping_mode: str = "delta"
    kappa: float = 1.0
    gamma: float = 0.99
    clamp_prob: tuple[float, float] | None = (0.0, 1.0)


class OracleSearchRLEnvReward:
    """
    Environment-side reward computation for marked search.

    This helper is *stateless* across episodes unless you call .reset(); it tracks
    the previous-step success s_{t-1} only when dense 'delta' shaping is enabled.

    Typical loop
    ------------
        rew = OracleSearchRLEnvReward(cfg)
        rew.reset()
        for t in range(T):
            # ... step env to get P_t (position marginals)
            r_t = rew.step(P_t, mark_mask, batch=batch, done=(t==T-1))

    Notes
    -----
    • Provide P_t as (N,) or (B,N); for ragged batches use (N,) with `batch`.
    • 'done' indicates terminal step; a terminal bonus equals success(P_T) is added
      *only* when terminal_only=True or when shaping_mode="delta" (to ensure that
      the sum of deltas telescopes to the terminal success).
    """

    def __init__(self, cfg: OracleSearchRLConfig):
        self.cfg = cfg
        self._prev_success: torch.Tensor | None = None  # per-graph vector

    def reset(self) -> None:
        """Reset previous success tracker between episodes (or vectorized envs)."""
        self._prev_success = None

    def _success_vec(
        self,
        P: torch.Tensor,
        mark_mask: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        if cfg.clamp_prob is not None:
            lo, hi = cfg.clamp_prob
            P = P.clamp(min=lo, max=hi)
        return aggregate_mark_success(P, mark_mask, batch=batch, reduce="none")  # (G,)

    def step(
        self,
        P_t: torch.Tensor,
        mark_mask: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
        done: bool = False,
    ) -> torch.Tensor:
        """
        Compute reward r_t given current probabilities.

        Returns
        -------
        r_t : scalar if single graph, else (G,) per-graph rewards (no reduction).
        """
        cfg = self.cfg
        s_t = self._success_vec(P_t, mark_mask, batch=batch)  # (G,)

        # Terminal-only sparse reward
        if cfg.terminal_only:
            r = torch.zeros_like(s_t)
            if done:
                r = s_t
            return r if s_t.numel() > 1 else r.squeeze(0)

        # Dense shaping
        mode = cfg.shaping_mode.lower()
        if mode not in ("delta", "absolute"):
            raise ValueError("shaping_mode must be 'delta' or 'absolute'.")

        if mode == "absolute":
            r = cfg.kappa * s_t
            return r if s_t.numel() > 1 else r.squeeze(0)

        # delta mode
        if self._prev_success is None:
            # First step: reward is kappa * s_0 (potential-based shaping can start at 0)
            r = cfg.kappa * s_t
        else:
            r = cfg.kappa * (s_t - self._prev_success)

        # Terminal bonus to ensure telescoping sums to final success (optional but standard)
        if done:
            # No extra bonus needed: sum_{t=0..T-1} (s_t - s_{t-1}) = s_{T-1} - s_{-1};
            # with our initialization r_0 = s_0 we already telescope to s_{T-1}.
            # Some environments define terminal P_T (post-step); if your P_t is pre-shift,
            # you can add a final bonus = success(P_T) - success(P_{T-1}) here explicitly.
            pass

        # Update state
        self._prev_success = s_t.detach()

        return r if r.numel() > 1 else r.squeeze(0)


# --------------------------------------------------------------------------------------
# Optional: convenience module for plug-and-play training loops
# --------------------------------------------------------------------------------------


class _SearchObjectiveAdapter(nn.Module):
    """
    Lightweight adapter that can be slotted into generic training loops.

    Example
    -------
        obj = _SearchObjectiveAdapter(
            OracleSearchDiffObjective(OracleSearchDiffConfig()),
            reduction="mean"
        )
        loss = obj(P_T, mark_mask, batch=batch)
    """

    def __init__(self, diff_obj: OracleSearchDiffObjective, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        self.diff_obj = diff_obj
        self.reduction = reduction

    def forward(
        self,
        P: torch.Tensor | Sequence[torch.Tensor],
        mark_mask: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.diff_obj.loss(P, mark_mask, batch=batch)


# --------------------------------------------------------------------------------------
# Sanity tests (quick, no heavy imports) – can be run under pytest -q
# --------------------------------------------------------------------------------------


def _quick_self_test(device: torch.device | None = None) -> None:
    """
    Minimal checks for core paths. Not exhaustive; meant to catch shape/grad pitfalls.
    """
    dev = device or torch.device("cpu")

    # Single graph, terminal only
    P = torch.tensor([0.1, 0.2, 0.3, 0.4], device=dev)
    mask = torch.tensor([0, 1, 0, 1], device=dev)
    cfg = OracleSearchDiffConfig(use_log=True, cvar_alpha=1.0)
    obj = OracleSearchDiffObjective(cfg)
    loss = obj.loss(P, mask)
    assert torch.isfinite(loss), "Loss should be finite."

    # Batched (B,N)
    P2 = torch.stack([P, 1 - P], dim=0)  # (2,4)
    M2 = mask.repeat(2, 1)
    s = obj.success(P2, M2, reduce="none")
    assert s.shape == (2,), "Per-batch success shape mismatch."

    # Ragged with batch vector
    batch = torch.tensor([0, 0, 1, 1], device=dev)
    s_ragged = aggregate_mark_success(P, mask, batch=batch, reduce="none")
    assert s_ragged.shape == (2,), "Per-graph success shape mismatch (ragged)."

    # RL dense shaping
    rlcfg = OracleSearchRLConfig(terminal_only=False, shaping_mode="delta", kappa=2.0)
    rew = OracleSearchRLEnvReward(rlcfg)
    rew.reset()
    r0 = rew.step(P, mask, batch=batch, done=False)
    r1 = rew.step(P * 1.1, mask, batch=batch, done=True)
    assert torch.isfinite(r0).all() and torch.isfinite(r1).all(), "RL rewards must be finite."


if __name__ == "__main__":
    _quick_self_test()


# EOF

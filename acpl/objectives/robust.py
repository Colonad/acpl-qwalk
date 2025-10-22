# acpl/objectives/robust.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal, Protocol

import torch
import torch.nn as nn

__all__ = [
    # Core config / API
    "RobustConfig",
    "RobustObjective",
    "robust_objective",
    "aggregate_risk",
    # Risk helpers
    "cvar",
    "quantile",
    # Sampler protocol + ready-to-use samplers
    "PerturbationSampler",
    "NoPerturbation",
    "StaticEdgePhaseJitter",
    "DynamicCoinDephasing",
    "CompositeSampler",
    # RNG helpers
    "seeded_generator",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RiskName = Literal["expectation", "cvar", "worst_case", "mean_var"]
TailName = Literal["lower", "upper"]
ReduceName = Literal["mean", "sum", "none"]


@dataclass
class RobustConfig:
    """
    Configuration for robust objectives under stochastic perturbations.

    Fields
    ------
    n_samples : number of Monte Carlo samples from the perturbation law.
    risk      : risk aggregator ("expectation", "cvar", "worst_case", "mean_var").
    alpha     : tail mass for CVaR (e.g., 0.1 = 10% worst outcomes for 'lower' tail).
    tail      : which tail for CVaR / worst_case ('lower' = risk-averse for returns).
    lambda_var: variance weight for the mean-variance risk (J = mean - λ·var).
    reduce    : reduction across batch if the wrapped simulator returns a batch of episodes.
    microbatch: if not None, split samples into chunks of this size to cap memory.
    seed      : optional base RNG seed for reproducible perturbations.
    """

    n_samples: int = 8
    risk: RiskName = "expectation"
    alpha: float = 0.1
    tail: TailName = "lower"
    lambda_var: float = 0.0
    reduce: ReduceName = "mean"
    microbatch: int | None = None
    seed: int | None = None


# ---------------------------------------------------------------------------
# Sampler protocol & built-ins
# ---------------------------------------------------------------------------


class PerturbationSampler(Protocol):
    """
    Protocol for noise/disorder samplers.

    The sampler receives:
      - idx:      integer sample index (0..n_samples-1)
      - context:  arbitrary dict/object from the caller (graph tensors, sizes, etc.)
      - rng:      torch.Generator for reproducible sampling (already seeded)
      - device:   preferred torch device for any tensors produced
      - dtype:    preferred dtype for any tensors produced

    It must return a dict-like payload that the user's `run(noise)` understands.
    """

    def __call__(
        self,
        idx: int,
        *,
        context: Any | None,
        rng: torch.Generator | None,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> dict[str, Any]: ...


class NoPerturbation:
    """Returns an empty dict – useful as a placeholder or to disable noise."""

    def __call__(
        self,
        idx: int,
        *,
        context: Any | None,
        rng: torch.Generator | None,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> dict[str, Any]:
        return {}


class StaticEdgePhaseJitter:
    """
    Static edge-phase disorder ϕ_e ~ N(0, σ^2) (or Uniform) sampled once per episode.
    Intended for flip-flop shift models where phases appear on edges/ports.

    The sampler expects `context` to contain either:
      - 'num_edges': int (number of undirected edges; phases broadcast to both arcs), OR
      - 'num_arcs' : int (already oriented arc count; use per-arc phases)

    Returned payload:
      {
        "edge_phases": Tensor shape [num_edges] or [num_arcs], dtype=float32,
        "phase_space": "edges" | "arcs"
      }
    """

    def __init__(
        self,
        sigma: float = 0.2,
        distribution: Literal["normal", "uniform"] = "normal",
        arcs: bool = False,
        clip: float | None = math.pi,  # clip to [-pi, pi] by default
    ):
        self.sigma = float(sigma)
        self.distribution = distribution
        self.arcs = bool(arcs)
        self.clip = clip

    def __call__(
        self,
        idx: int,
        *,
        context: Any | None,
        rng: torch.Generator | None,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> dict[str, Any]:
        if context is None:
            raise ValueError(
                "StaticEdgePhaseJitter requires a context with 'num_edges' or 'num_arcs'."
            )

        key = "num_arcs" if self.arcs else "num_edges"
        if key not in context:
            # Try to infer if only the other key exists.
            if "num_arcs" in context:
                key = "num_arcs"
            elif "num_edges" in context:
                key = "num_edges"
            else:
                raise KeyError("Context must include 'num_edges' or 'num_arcs'.")

        n = int(context[key])
        if n <= 0:
            raise ValueError(f"{key} must be positive; got {n}.")

        device = device or torch.device("cpu")
        dtype = dtype or torch.float32

        if self.distribution == "normal":
            phases = torch.randn(n, generator=rng, device=device, dtype=dtype) * self.sigma
        elif self.distribution == "uniform":
            phases = (
                torch.rand(n, generator=rng, device=device, dtype=dtype) * 2.0 - 1.0
            ) * self.sigma
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        if self.clip is not None:
            phases = torch.clamp(phases, min=-abs(self.clip), max=abs(self.clip))

        return {
            "edge_phases": phases,
            "phase_space": "arcs" if key == "num_arcs" else "edges",
        }


class DynamicCoinDephasing:
    """
    Per-time, per-node coin dephasing rate p ~ Beta(a,b) or fixed p.
    The simulator can interpret this as a position-unital channel applied after Ct (or fold into Ct).

    Expected context keys:
      - 'num_nodes' : int
      - optional 'T' : int (if you want per-time draws; otherwise returns single vector per node)

    Returned payload (one of):
      {"coin_dephase_p": [N]}                if no T in context
      {"coin_dephase_p": [T, N]}             if T provided (per time)
    """

    def __init__(
        self,
        p: float | None = None,
        beta_ab: tuple[float, float] | None = (2.0, 5.0),
        per_time: bool = True,
    ):
        if p is None and beta_ab is None:
            raise ValueError("Provide either a fixed p or Beta(a,b) parameters.")
        self.p = p
        self.beta_ab = beta_ab
        self.per_time = per_time

    def __call__(
        self,
        idx: int,
        *,
        context: Any | None,
        rng: torch.Generator | None,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> dict[str, Any]:
        if context is None or "num_nodes" not in context:
            raise ValueError("DynamicCoinDephasing requires 'num_nodes' in context.")

        N = int(context["num_nodes"])
        T = (
            int(context["T"])
            if (self.per_time and context is not None and "T" in context)
            else None
        )

        device = device or torch.device("cpu")
        dtype = dtype or torch.float32

        def _beta(shape: Sequence[int]) -> torch.Tensor:
            a, b = self.beta_ab
            x1 = (
                torch.distributions.Gamma(a, 1.0)
                .sample(shape, generator=rng)
                .to(device=device, dtype=dtype)
            )
            x2 = (
                torch.distributions.Gamma(b, 1.0)
                .sample(shape, generator=rng)
                .to(device=device, dtype=dtype)
            )
            return x1 / (x1 + x2 + 1e-9)

        if self.p is not None:
            if T is None:
                p = torch.full((N,), float(self.p), device=device, dtype=dtype)
            else:
                p = torch.full((T, N), float(self.p), device=device, dtype=dtype)
        else:
            if T is None:
                p = _beta((N,))
            else:
                p = _beta((T, N))

        return {"coin_dephase_p": p}


class CompositeSampler:
    """
    Compose multiple samplers; their outputs are merged into a single dict.
    Later keys overwrite earlier keys on collision.
    """

    def __init__(self, *samplers: PerturbationSampler):
        self.samplers = list(samplers)

    def __call__(
        self,
        idx: int,
        *,
        context: Any | None,
        rng: torch.Generator | None,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for s in self.samplers:
            out = s(idx, context=context, rng=rng, device=device, dtype=dtype)
            if not isinstance(out, dict):
                raise TypeError(f"Sampler {s} returned a non-dict payload.")
            merged.update(out)
        return merged


# ---------------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------------


def seeded_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    g = torch.Generator(device="cpu")
    # allow full uint64 range for determinism across workers
    g.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return g


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------


def quantile(x: torch.Tensor, q: float, dim: int = 0) -> torch.Tensor:
    """
    Differentiable quantile via sort and selection.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1].")
    sorted_x, _ = torch.sort(x, dim=dim)
    # index using floor((n-1)*q), supports broadcast if x has batch dims
    n = x.size(dim)
    idx = max(0, min(n - 1, int(math.floor((n - 1) * q))))
    return torch.index_select(sorted_x, dim, torch.tensor([idx], device=x.device)).squeeze(dim)


def cvar(
    x: torch.Tensor,
    alpha: float = 0.1,
    *,
    dim: int = 0,
    tail: TailName = "lower",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Conditional Value-at-Risk (CVaR) along dimension `dim`.

    Returns (cvar_value, threshold) with:
      - for tail='lower': average of the worst alpha-fraction (smallest values)
      - for tail='upper': average of the best  alpha-fraction (largest values)

    Notes
    -----
    * Uses a sort-based implementation (subgradient-friendly).
    * If alpha * n < 1, we still include at least one element in the tail set.
    """
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in (0, 1].")
    values, _ = torch.sort(x, dim=dim)
    n = x.size(dim)
    k = max(1, int(math.floor(alpha * n)))
    if tail == "lower":
        slice_vals = values.narrow(dim, 0, k)
        thr = torch.index_select(values, dim, torch.tensor([k - 1], device=x.device)).squeeze(dim)
    elif tail == "upper":
        slice_vals = values.narrow(dim, n - k, k)
        thr = torch.index_select(values, dim, torch.tensor([n - k], device=x.device)).squeeze(dim)
    else:
        raise ValueError("tail must be 'lower' or 'upper'.")

    # Mean across the selected tail; keep remaining dims as-is
    cvar_val = slice_vals.mean(dim=dim)
    return cvar_val, thr


def aggregate_risk(
    samples: torch.Tensor,
    *,
    risk: RiskName = "expectation",
    alpha: float = 0.1,
    tail: TailName = "lower",
    lambda_var: float = 0.0,
    dim: int = 0,
) -> torch.Tensor:
    """
    Aggregate sample values according to a risk measure.

    samples : Tensor with sample dimension `dim` (e.g., [S, ...]).
              These are assumed to be RETURNS/REWARDS (higher is better).
    """
    if risk == "expectation":
        return samples.mean(dim=dim)

    if risk == "worst_case":
        if tail == "lower":
            return samples.min(dim=dim).values
        else:
            return samples.max(dim=dim).values

    if risk == "cvar":
        c, _ = cvar(samples, alpha=alpha, tail=tail, dim=dim)
        return c

    if risk == "mean_var":
        mu = samples.mean(dim=dim)
        var = samples.var(dim=dim, unbiased=False)
        return mu - float(lambda_var) * var

    raise ValueError(f"Unknown risk type: {risk}")


# ---------------------------------------------------------------------------
# Core objective wrapper
# ---------------------------------------------------------------------------


class RobustObjective(nn.Module):
    """
    Wrap a differentiable (or RL) simulator/policy with Monte-Carlo robustness.

    Parameters
    ----------
    run : Callable[[dict], torch.Tensor]
        A callable that accepts a single `noise` payload (dict) and returns
        a scalar or a batch tensor of returns/rewards (higher is better).
        Shapes can be:
            - scalar:    torch.Size([])
            - per-ep:    torch.Size([B])
            - per-ep+... arbitrary extra episode axes are supported; reduction controls batch handling.

        `run` should be differentiable if used in supervised/diff mode.

    sampler : PerturbationSampler
        Stochastic perturbation law; receives (idx, context, rng, device, dtype) and returns a dict.

    cfg : RobustConfig
        Monte Carlo & risk configuration.

    device/dtype/context :
        Optional defaults forwarded to the sampler; context can hold graph sizes, T, etc.

    Usage
    -----
    >>> # Define your simulator wrapper:
    >>> def run_once(noise):
    ...     # compute return given sampled noise
    ...     return score  # tensor scalar or [B]
    >>>
    >>> obj = RobustObjective(run_once, StaticEdgePhaseJitter(0.2), RobustConfig(n_samples=16, risk="cvar", alpha=0.1))
    >>> value = obj()  # differentiable robust value (to maximize); use -value as loss
    """

    def __init__(
        self,
        run: Callable[[dict[str, Any]], torch.Tensor],
        sampler: PerturbationSampler,
        cfg: RobustConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        context: Any | None = None,
    ):
        super().__init__()
        self.run = run
        self.sampler = sampler
        self.cfg = cfg
        self._device = device
        self._dtype = dtype
        self.context = context

        # store alpha etc. as buffers for TorchScript friendliness (no gradients)
        self.register_buffer("_alpha_buf", torch.tensor(float(cfg.alpha)), persistent=False)
        self.register_buffer(
            "_lambda_var_buf", torch.tensor(float(cfg.lambda_var)), persistent=False
        )

    @property
    def device(self) -> torch.device | None:
        return self._device

    @property
    def dtype(self) -> torch.dtype | None:
        return self._dtype

    def forward(self) -> torch.Tensor:
        """
        Compute the robust value = risk-aggregated expected return across perturbations.
        This returns a *value to maximize*. If you want a loss, take the negative.
        """
        cfg = self.cfg
        n = int(cfg.n_samples)
        if n <= 0:
            raise ValueError("n_samples must be positive.")

        gen = seeded_generator(cfg.seed)
        # Collect per-sample returns; we allow per-episode batches.
        # Shape after stack: [S, *ep_shape]
        # Memory-friendly microbatching:
        values = []
        mb = cfg.microbatch or n
        i = 0
        while i < n:
            j = min(n, i + mb)
            chunk = []
            for k in range(i, j):
                noise = self.sampler(
                    k,
                    context=self.context,
                    rng=gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                val = self.run(noise)
                if not isinstance(val, torch.Tensor):
                    raise TypeError("run(noise) must return a torch.Tensor.")
                chunk.append(val)
            # stack chunk on new leading dim
            values.append(torch.stack(chunk, dim=0))
            i = j
        samples = torch.cat(values, dim=0)

        # Aggregate across sample dimension (dim=0)
        agg = aggregate_risk(
            samples,
            risk=cfg.risk,
            alpha=float(self._alpha_buf.item()),
            tail=cfg.tail,
            lambda_var=float(self._lambda_var_buf.item()),
            dim=0,
        )

        # Optional reduction across episode batch (if present)
        if cfg.reduce == "mean":
            return agg.mean()
        elif cfg.reduce == "sum":
            return agg.sum()
        elif cfg.reduce == "none":
            return agg
        else:
            raise ValueError(f"Unknown reduce: {cfg.reduce}")


# Convenience functional entrypoint
def robust_objective(
    run: Callable[[dict[str, Any]], torch.Tensor],
    sampler: PerturbationSampler,
    cfg: RobustConfig,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    context: Any | None = None,
) -> torch.Tensor:
    """
    Functional helper equivalent to RobustObjective(...).forward().
    """
    return RobustObjective(run, sampler, cfg, device=device, dtype=dtype, context=context)()


# ---------------------------------------------------------------------------
# Notes on integration with ACPL (Phase B4)
# ---------------------------------------------------------------------------
#
# • Differentiable training:
#     Use the robust *value* as your objective to maximize; define loss = -value.
#     Gradients flow through the simulator and the coin-parameter map. Sorting in CVaR
#     is subgradient-friendly in PyTorch; it’s widely used for risk objectives.
#
# • RL training:
#     If `run(noise)` executes an episode and returns a terminal reward estimate that
#     is used by your policy-gradient algorithm, you can still wrap it here to report
#     robust metrics (for logging/selection) or place risk in the policy loss if the
#     environment is differentiable-through (e.g., reparameterized noise).
#
# • Static vs Dynamic perturbations:
#     - StaticEdgePhaseJitter   → context should provide num_edges or num_arcs.
#     - DynamicCoinDephasing    → context should provide num_nodes (and optionally T).
#     Combine them via CompositeSampler(...) to apply both.
#
# • CVaR tail:
#     For success probabilities (higher is better), risk-averse training typically uses
#     tail='lower' and small alpha (e.g., 0.05–0.2). For cost/minimization targets,
#     either negate returns before calling this wrapper or use tail='upper' accordingly.
#
# • Shapes:
#     If run() returns per-episode values [B], RobustObjective.reduce governs the final
#     scalar (mean/sum) or returns the vector (none). The sample dimension is always the
#     leading axis during risk aggregation.
#
# • Reproducibility:
#     Provide RobustConfig.seed to seed the sampler RNG. For multi-worker setups,
#     consider adding a worker_id offset to the seed.
#
# • Numerical stability:
#     cvar() includes at least one element in the tail set even when alpha·S < 1.
#     quantile() uses sort-based selection to keep gradients defined almost everywhere.
#
# ---------------------------------------------------------------------------

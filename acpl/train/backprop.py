# acpl/train/backprop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Tuple

import torch
from torch import Tensor
import torch.nn as nn

# Policy imports: we use the forward (angles) and the inline SU(2) lift
from acpl.policy.policy import ACPLPolicy

__all__ = [
    "StepFn",
    "ProbsFn",
    "LossFn",
    "RolloutConfig",
    "RegularizerConfig",
    "RolloutOutputs",
    "rollout_and_loss",
    "clip_gradients",
    "backward_with_optional_clip",
]


# ------------------------------ Interfaces & Configs ------------------------------ #

class StepFn(Protocol):
    """
    Signature for a differentiable one-step DTQW update.
    Must apply the *local* coin and then the flip-flop shift (or whichever variant you use).

    Args:
        state: (..., D) or (..., N, d_c) complex tensor — your simulator decides the layout.
        coin_t: (N, 2, 2) complex (for dv=2). If you support dv>2, accept a list or block-diagonal encoding.
        ctx:   arbitrary dict-like context (eg. port map, degree metadata, cached permutations).

    Returns:
        new_state: same shape as `state`, complex and connected to autograd.
    """
    def __call__(self, state: Tensor, coin_t: Tensor, ctx: Dict[str, Any]) -> Tensor: ...


class ProbsFn(Protocol):
    """
    Signature to produce position marginals from the full state (used by most objectives).

    Args:
        state: simulator state at terminal time.
        ctx:   context if needed.

    Returns:
        P: (N,) real tensor with probabilities per vertex, sum ~ 1 (up to num. error).
    """
    def __call__(self, state: Tensor, ctx: Dict[str, Any]) -> Tensor: ...


class LossFn(Protocol):
    """
    Signature for task loss (negative of the return).

    You can consume either the terminal position marginals `P_T`, the full terminal state,
    or both. Keep it flexible: if you only need P_T, ignore `state_T`.

    Must return a scalar loss and an aux log dict.
    """
    def __call__(
        self,
        P_T: Optional[Tensor],
        state_T: Optional[Tensor],
        ctx: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...


@dataclass
class RegularizerConfig:
    """
    Policy-output regularizers (θ are the per-node Euler angles for SU(2) coins).

    We expose several knobs discussed in the report:
      • Temporal smoothness (first- and second-order) and a frequency-domain (FFT) penalty.
      • Spatial (graph) smoothness via Dirichlet/Laplacian energy over edges.
      • Magnitude/L2 on angles.
      • Gradient clipping settings (applied *after* backprop if you use the helper below).

    All λ hyperparameters are nonnegative scalars (float). Set to 0.0 to disable.
    """
    # --- Temporal regularizers on θ(t, v, :) ---
    smooth_theta: float = 0.0        # λ * Σ_{t>0,v} ||θ_t(v) - θ_{t-1}(v)||^2
    curvature_theta: float = 0.0     # λ * Σ_{t=1..T-2,v} ||θ_{t+1}(v) - 2θ_t(v) + θ_{t-1}(v)||^2  (high-freq suppressor)
    spectral_time: float = 0.0       # λ * Σ_{k} (ω_k^p * |Θ̂_k|^2)  (FFT penalty)
    spectral_time_power: int = 2     # p in the weight ω_k^p (≥1). Only used if spectral_time>0

    # --- Spatial (graph) regularizer over edges (Dirichlet/Laplacian energy) ---
    spatial_graph_smooth: float = 0.0  # λ * Σ_{t} Σ_{(u,v)∈E} ||θ_t(u) - θ_t(v)||^2

    # --- Magnitude control ---
    l2_theta: float = 0.0            # λ * Σ_{t,v} ||θ_t(v)||^2

    # --- Gradient clipping settings (optional utility) ---
    grad_clip_norm: Optional[float] = None   # if set, clip global L2 norm of all params to this value
    grad_clip_value: Optional[float] = None  # if set, clamp each grad component to [-value, +value]

    # Optional: log power spectrum instead of penalizing (for debugging). Keep False in training.
    log_temporal_spectrum: bool = False


@dataclass
class RolloutConfig:
    """
    Controls the differentiable rollout.

    Attributes
    ----------
    T: horizon (steps)
    dtype: complex dtype for the simulator state and coins
    keep_trajectory: if True, returns the full state trajectory (for analysis / curriculum)
    """
    T: int
    dtype: torch.dtype = torch.complex64
    keep_trajectory: bool = False


@dataclass
class RolloutOutputs:
    """
    Return container for rollout_and_loss.
    """
    loss: Tensor
    logs: Dict[str, Tensor]
    state_T: Optional[Tensor] = None
    P_T: Optional[Tensor] = None
    trajectory: Optional[Tuple[Tensor, ...]] = None  # tuple of states, if keep_trajectory=True
    theta: Optional[Tensor] = None                   # (T, N, 3) angles when SU(2) policy is used
    coins: Optional[Tensor] = None                   # (T, N, 2, 2) complex coins (SU(2))


# ------------------------------ Regularizer primitives ------------------------------ #

def _temporal_first_diff(theta: Tensor) -> Tensor:
    """
    θ: (T, N, P)  ->  Δθ_t = θ_t - θ_{t-1}  for t=1..T-1
    Returns (T-1, N, P)
    """
    if theta.size(0) < 2:
        return theta.new_zeros((0,) + theta.shape[1:])
    return theta[1:] - theta[:-1]


def _temporal_second_diff(theta: Tensor) -> Tensor:
    """
    Discrete curvature along time: θ_{t+1} - 2 θ_t + θ_{t-1}  for t=1..T-2
    """
    if theta.size(0) < 3:
        return theta.new_zeros((0,) + theta.shape[1:])
    return theta[2:] - 2.0 * theta[1:-1] + theta[:-2]


def _temporal_smoothness_penalty(theta: Tensor) -> Tensor:
    diffs = _temporal_first_diff(theta)
    return (diffs**2).sum()


def _temporal_curvature_penalty(theta: Tensor) -> Tensor:
    curv = _temporal_second_diff(theta)
    return (curv**2).sum()


def _temporal_spectral_penalty(theta: Tensor, power: int = 2) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Frequency-domain (real-FFT) penalty along time.
      penalty = Σ_k (ω_k^power * ||F[k]||^2)

    θ: (T, N, P) real-valued angles
    Returns: (penalty, optional power spectrum for logging (T_rfft,))
    """
    T = theta.size(0)
    if T <= 1:
        return theta.new_zeros(()), None

    # rFFT over time axis; result shape: (T//2 + 1, N, P)
    F = torch.fft.rfft(theta, dim=0)  # complex
    # Frequency bins in [0, 0.5, ...] (assuming unit sampling)
    try:
        freqs = torch.fft.rfftfreq(T, d=1.0).to(theta.device)
    except AttributeError:
        # PyTorch < 1.8 fallback
        freqs = torch.arange(F.size(0), device=theta.device, dtype=theta.dtype) / float(T)

    # Weight emphasizing higher temporal frequencies; avoid zero at k=0
    w = (freqs + 1e-8) ** float(power)  # (T_rfft,)
    # Squared magnitude over (N,P), then weight and sum over k
    mag2 = (F.real**2 + F.imag**2).sum(dim=(1, 2))  # (T_rfft,)
    penalty = (w.to(mag2.dtype) * mag2).sum()

    return penalty, mag2.detach()  # mag2 is the raw periodogram (unweighted) for optional logging


def _l2_penalty(theta: Tensor) -> Tensor:
    return (theta**2).sum()


def _graph_dirichlet_penalty(theta: Tensor, edge_index: Tensor) -> Tensor:
    """
    Spatial (graph) smoothness on angles via Dirichlet energy:
      Σ_t Σ_(u,v) ||θ_t[u] - θ_t[v]||^2
    where edge_index is (2, E) with both directions allowed (it doesn't matter if undirected is doubled).
    θ: (T, N, P) real
    """
    if edge_index.numel() == 0:
        return theta.new_zeros(())
    src, dst = edge_index.long()
    # Gather per edge, broadcast over T and P
    # Shapes: (T, E, P)
    diff = theta[:, src, :] - theta[:, dst, :]
    return (diff**2).sum()


# ------------------------------ Gradient clipping utilities ------------------------------ #

def clip_gradients(
    parameters: Iterable[nn.Parameter],
    *,
    max_norm: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Dict[str, Tensor]:
    """
    Apply gradient clipping to `parameters`. Both modes are independent; you can use either or both.

    Args:
        parameters: iterable of nn.Parameters with .grad populated
        max_norm: if set, clip global L2 norm to this value (torch.nn.utils.clip_grad_norm_)
        max_value: if set, clamp each grad component to [-max_value, max_value] (clip_grad_value_)

    Returns:
        dict with diagnostics: {"pre_norm": ..., "post_norm": ..., "max_abs_grad": ...}
    """
    # Materialize a list and filter out parameters without grads
    params = [p for p in parameters if (p is not None and p.grad is not None)]
    logs: Dict[str, Tensor] = {}
    if not params:
        return logs

    # Pre-clipping diagnostics
    with torch.no_grad():
        total_sq = torch.zeros((), device=params[0].grad.device)
        max_abs = torch.zeros((), device=params[0].grad.device)
        for p in params:
            g = p.grad
            total_sq = total_sq + (g.detach() ** 2).sum()
            max_abs = torch.maximum(max_abs, g.detach().abs().max())
        pre_norm = total_sq.sqrt()
        logs["grad_pre_norm"] = pre_norm
        logs["grad_pre_max_abs"] = max_abs

    if max_value is not None and max_value > 0.0:
        torch.nn.utils.clip_grad_value_(params, max_value)

    if max_norm is not None and max_norm > 0.0:
        # clip_grad_norm_ returns the (pre-clipping) total norm as a Python float; we re-measure for logs.
        torch.nn.utils.clip_grad_norm_(params, max_norm)

    # Post-clipping diagnostics
    with torch.no_grad():
        total_sq = torch.zeros((), device=params[0].grad.device)
        max_abs = torch.zeros((), device=params[0].grad.device)
        for p in params:
            g = p.grad
            total_sq = total_sq + (g.detach() ** 2).sum()
            max_abs = torch.maximum(max_abs, g.detach().abs().max())
        logs["grad_post_norm"] = total_sq.sqrt()
        logs["grad_post_max_abs"] = max_abs

    return logs


def backward_with_optional_clip(
    loss: Tensor,
    parameters: Iterable[nn.Parameter],
    *,
    reg_cfg: Optional[RegularizerConfig] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Dict[str, Tensor]:
    """
    Convenience helper: backprop the loss, then apply gradient clipping if requested.

    Returns a small log dict with grad diagnostics if clipping was applied.
    """
    reg_cfg = reg_cfg or RegularizerConfig()
    loss.backward(retain_graph=retain_graph, create_graph=create_graph)

    if reg_cfg.grad_clip_norm is not None or reg_cfg.grad_clip_value is not None:
        return clip_gradients(parameters, max_norm=reg_cfg.grad_clip_norm, max_value=reg_cfg.grad_clip_value)
    return {}


# ------------------------------ Core Rollout ------------------------------ #

def rollout_and_loss(
    *,
    policy: ACPLPolicy,
    X: Tensor,                       # (N, Fin) float
    edge_index: Tensor,              # (2, E) long
    psi0: Tensor,                    # simulator initial state (complex)
    step_fn: StepFn,                 # callable(state, coins_t, ctx) -> state
    probs_fn: Optional[ProbsFn],     # callable(state, ctx) -> (N,) probs; may be None if your loss uses state
    loss_fn: LossFn,                 # callable(P_T, state_T, ctx) -> (loss, logs)
    cfg: RolloutConfig,
    *,
    edge_weight: Optional[Tensor] = None,  # (E,) if used by policy encoder
    reg: Optional[RegularizerConfig] = None,
    sim_ctx: Optional[Dict[str, Any]] = None,
    coins_from_policy: str = "su2",  # currently "su2" (supports dv=2 graphs). Extend as needed.
) -> RolloutOutputs:
    """
    Differentiable rollout through T steps with autograd intact.

    Pipeline
    --------
    1) θ = policy(X, edge_index, T)              # (T, N, P)   (policy emits angles; SU(2): P=3)
    2) C_t = Φ(θ_t)                              # (N, 2, 2)   (unitary lift; SU(2) inline & exact)
    3) ψ_{t+1} = S C_t ψ_t                       # simulator step function (user-supplied)
    4) P_T = Tr_C(|ψ_T><ψ_T|) diag               # position marginals (user-supplied probs_fn)
    5) L = task loss(P_T, ψ_T) + regularizers    # autograd to policy parameters

    Notes
    -----
    - We build coins via the policy's exact differentiable SU(2) lift to keep the path short & stable.
    - Regularizers operate on θ (angles), which is the most "physical" locus for smooth/time/space penalties.
    - This function is agnostic to your state/vectorization layout; `step_fn` and `probs_fn` define it.

    Returns
    -------
    RolloutOutputs with scalar loss, logs, terminal artifacts, and (optional) trajectory.
    """
    if cfg.T <= 0:
        raise ValueError("RolloutConfig.T must be positive.")

    sim_ctx = {} if sim_ctx is None else sim_ctx
    reg = reg or RegularizerConfig()

    # 1) Emit policy angles θ: (T, N, 3) for SU(2) ZYZ (alpha, beta, gamma).
    theta = policy(X, edge_index, T=cfg.T, edge_weight=edge_weight)  # (T, N, 3)

    # 2) Map θ → C_t via exact ZYZ → SU(2) (keeps autograd). Shape: (T, N, 2, 2)
    if coins_from_policy.lower() != "su2":
        raise NotImplementedError(
            "Only SU(2) coins are currently supported in this trainer. "
            "Extend by adding your U(d) lifting here when dv>2."
        )
    T_, N_, P = theta.shape
    if P != 3:
        raise RuntimeError(f"Expected 3 Euler angles per node/time for SU(2), got P={P}.")

    flat = theta.reshape(T_ * N_, 3)
    coins_flat = policy._su2_from_euler_batch(flat, dtype=cfg.dtype)  # (T*N, 2, 2) complex
    coins = coins_flat.view(T_, N_, 2, 2).contiguous()

    # 3) Rollout ψ_t with provided step function.
    state = psi0
    traj: Optional[list] = [] if cfg.keep_trajectory else None
    for t in range(cfg.T):
        if traj is not None:
            traj.append(state)
        state = step_fn(state, coins[t], sim_ctx)

    state_T = state
    if traj is not None:
        traj.append(state_T)

    # 4) Compute position marginals if a probs_fn is provided (common case).
    P_T: Optional[Tensor] = None
    if probs_fn is not None:
        P_T = probs_fn(state_T, sim_ctx)

    # 5) Task loss
    loss_main, logs = loss_fn(P_T, state_T, sim_ctx)
    if loss_main.ndim != 0:
        raise RuntimeError("loss_fn must return a scalar loss tensor.")

    # ------------------------------ Regularizers ------------------------------ #
    # All penalties are computed on θ (angles). These are differentiable wrt policy parameters.

    reg_loss = theta.new_zeros(())

    # (a) Temporal smoothness (first-order)
    if reg.smooth_theta and reg.smooth_theta > 0.0:
        val = _temporal_smoothness_penalty(theta)
        reg_loss = reg_loss + reg.smooth_theta * val
        logs["reg_smooth_theta"] = val.detach()

    # (b) Temporal curvature (second-order differences)
    if reg.curvature_theta and reg.curvature_theta > 0.0:
        val = _temporal_curvature_penalty(theta)
        reg_loss = reg_loss + reg.curvature_theta * val
        logs["reg_curvature_theta"] = val.detach()

    # (c) Temporal spectral penalty (FFT)
    if reg.spectral_time and reg.spectral_time > 0.0:
        spec_val, periodogram = _temporal_spectral_penalty(theta, power=int(reg.spectral_time_power))
        reg_loss = reg_loss + reg.spectral_time * spec_val
        logs["reg_spectral_time"] = spec_val.detach()
        if reg.log_temporal_spectrum and periodogram is not None:
            # Log small stats instead of the full vector to keep logs compact
            logs["spec_energy_total"] = periodogram.sum()
            logs["spec_energy_dc"] = periodogram[0]
            if periodogram.numel() > 1:
                logs["spec_energy_nyq"] = periodogram[-1]

    # (d) Spatial (graph) Dirichlet penalty across edges for each time slice
    if reg.spatial_graph_smooth and reg.spatial_graph_smooth > 0.0:
        val = _graph_dirichlet_penalty(theta, edge_index)
        reg_loss = reg_loss + reg.spatial_graph_smooth * val
        logs["reg_spatial_graph"] = val.detach()

    # (e) Magnitude control (L2)
    if reg.l2_theta and reg.l2_theta > 0.0:
        val = _l2_penalty(theta)
        reg_loss = reg_loss + reg.l2_theta * val
        logs["reg_l2_theta"] = val.detach()

    loss_total = loss_main + reg_loss

    # Diagnostics
    with torch.no_grad():
        logs["loss_main"] = loss_main.detach()
        logs["loss_total"] = loss_total.detach()
        if P_T is not None:
            logs["P_sum"] = P_T.sum()
            logs["P_max"] = P_T.max()
            logs["P_min"] = P_T.min()
        # Angle stats (helpful to spot saturation)
        logs["theta_abs_mean"] = theta.abs().mean()
        logs["theta_abs_max"] = theta.abs().max()

    return RolloutOutputs(
        loss=loss_total,
        logs=logs,
        state_T=state_T,
        P_T=P_T,
        trajectory=tuple(traj) if traj is not None else None,
        theta=theta,
        coins=coins,
    )

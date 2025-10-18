# acpl/train/backprop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import torch
from torch import Tensor
import torch.nn as nn

# Policy imports: we use the forward (angles) and the inline SU(2) lift
from acpl.policy.policy import ACPLPolicy

__all__ = [
    "RolloutConfig",
    "RegularizerConfig",
    "RolloutOutputs",
    "rollout_and_loss",
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
    Optional regularizers on the *policy outputs* (Euler angles for SU(2) head)
    to encourage physically smooth, stable coin schedules.

    Notes
    -----
    - smooth_theta penalizes temporal differences of angles (t-1 -> t) across nodes.
    - l2_theta penalizes magnitude of angles.
    - When coins are parameterized differently (e.g., U(d) via generators),
      you can add a parallel branch to regularize that tensor in your own code path.
    """
    smooth_theta: float = 0.0    # λ_t * sum_{t>0,v} ||theta[t,v] - theta[t-1,v]||^2
    l2_theta: float = 0.0        # λ_m * sum_{t,v}   ||theta[t,v]||^2
    # Optional gradient clipping for simulator stability (applied outside here typically):
    grad_clip_norm: Optional[float] = None


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


# ------------------------------ Core Rollout ------------------------------ #

def _temporal_smoothness_penalty(theta: Tensor) -> Tensor:
    """
    Sum of squared temporal differences: sum_{t=1..T-1, v} ||theta[t,v] - theta[t-1,v]||^2
    theta: (T, N, P)

    Returns: scalar tensor
    """
    if theta.size(0) < 2:
        return theta.new_zeros(())
    diffs = theta[1:] - theta[:-1]  # (T-1, N, P)
    return (diffs**2).sum()


def _l2_penalty(theta: Tensor) -> Tensor:
    """
    Sum of squared magnitudes: sum_{t,v} ||theta[t,v]||^2
    """
    return (theta**2).sum()


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
    - Regularizers operate on θ (angles), which is the most "physical" locus for smooth/time penalties.
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
    #    We call the policy *once* for efficiency/memory and reuse θ for coin construction & regularizers.
    theta = policy(X, edge_index, T=cfg.T, edge_weight=edge_weight)  # (T, N, 3)

    # 2) Map θ → C_t via exact ZYZ → SU(2) (keeps autograd). Shape: (T, N, 2, 2)
    if coins_from_policy.lower() != "su2":
        raise NotImplementedError(
            "Only SU(2) coins are currently supported in this trainer. "
            "Extend by adding your U(d) lifting here when dv>2."
        )
    T_, N_, P = theta.shape
    if P != 3:
        raise RuntimeError(
            f"Expected 3 Euler angles per node/time for SU(2), got P={P}."
        )
    flat = theta.reshape(T_ * N_, 3)                             # (T*N, 3)
    coins_flat = policy._su2_from_euler_batch(flat, dtype=cfg.dtype)  # (T*N, 2, 2) complex
    coins = coins_flat.view(T_, N_, 2, 2).contiguous()           # (T, N, 2, 2)

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

    # 5) Task loss + regularizers
    loss_main, logs = loss_fn(P_T, state_T, sim_ctx)
    assert loss_main.ndim == 0, "loss_fn must return a scalar loss."

    # Regularizers on angles (physics-informed smoothness & magnitude).
    reg_loss = theta.new_zeros(())
    if reg.smooth_theta and reg.smooth_theta > 0.0:
        reg_val = _temporal_smoothness_penalty(theta)
        reg_loss = reg_loss + reg.smooth_theta * reg_val
        logs["reg_smooth_theta"] = reg_val.detach()

    if reg.l2_theta and reg.l2_theta > 0.0:
        reg_val = _l2_penalty(theta)
        reg_loss = reg_loss + reg.l2_theta * reg_val
        logs["reg_l2_theta"] = reg_val.detach()

    loss_total = loss_main + reg_loss

    # Numerics/diagnostics
    with torch.no_grad():
        logs["loss_main"] = loss_main.detach()
        logs["loss_total"] = loss_total.detach()
        if P_T is not None:
            logs["P_sum"] = P_T.sum()        # helpful sanity for probability mass
            logs["P_max"] = P_T.max()
            logs["P_min"] = P_T.min()

    return RolloutOutputs(
        loss=loss_total,
        logs=logs,
        state_T=state_T,
        P_T=P_T,
        trajectory=tuple(traj) if traj is not None else None,
        theta=theta,
        coins=coins,
    )

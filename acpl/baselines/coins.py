# acpl/baselines/coins.py
from __future__ import annotations

import hashlib
import logging
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Union
from acpl.data.features import FeatureSpec, build_node_features

import numpy as np
import torch

__all__ = [
    # configs
    "FixedCoinKind",
    "SU2Convention",
    "FixedCoinConfig",
    "RandomCoinConfig",




    # schedule builders (used by baselines/policies.py)
    "FixedSU2ThetaSchedule",
    "RandomSU2ThetaSchedule",
    "GlobalScheduleSU2ThetaSchedule",
    "make_coin_schedule",
    "build",





    "GlobalScheduleBasis",
    "GlobalScheduleConfig",
    "BaselineCoinConfig",
    # core generators (high-level)
    "make_su2_theta_baseline",
    "make_unitary_coin_baseline_by_degree",
    # SU(2) primitives (unitary <-> ZYZ angles)
    "su2_from_zyz",
    "zyz_from_su2",
    "wrap_angle_2pi",
    "wrap_angle_pi",

    "wrap_angle_4pi",

    "project_to_su",
    # canonical fixed coins
    "hadamard_su2",
    "grover_unitary",
    "identity_unitary",
    "fixed_coin_su2_theta",
    # random & schedules
    "random_su2_theta",
    "global_schedule_su2_theta",
    # optional: tuning helper (research-grade baseline)
    "random_search_global_schedule",
]

log = logging.getLogger(__name__)

# =============================================================================
# Types / Config
# =============================================================================

FixedCoinKind = Literal["hadamard", "grover", "identity"]
GlobalScheduleBasis = Literal["fourier", "poly", "piecewise_constant"]
SU2Convention = Literal["zyz"]


@dataclass(frozen=True)
class FixedCoinConfig:
    """
    Fixed-coin baseline.

    kind:
      - hadamard:   (d=2) i*H (SU(2) by global phase)
      - grover:     Grover diffusion coin G_d = 2/d J - I (unitary/orthogonal)
                   For d=2 this is Pauli-X up to phase.
      - identity:   I_d
    """

    kind: FixedCoinKind = "hadamard"
    su2_convention: SU2Convention = "zyz"


@dataclass(frozen=True)
class RandomCoinConfig:
    """
    Seeded random coin baselines.

    For SU(2):
      - haar=True  -> sample from Haar measure on SU(2) (recommended).
      - haar=False -> sample angles independently uniform (quick sanity baseline).

    For unitary d>2:
      - We provide Grover/Identity as primary baselines; fully Haar U(d) sampling is
        intentionally *not* included here to avoid heavy QR/complex-Gaussian costs.
        If you need Haar U(d) for research, add a separate module.

    Notes
    -----
    - This baseline is meant to set *sanity bounds* and to show non-triviality of learning.
    """

    seed: int = 0
    haar: bool = True
    per_node: bool = True          # if False, one random coin shared by all nodes per time
    per_time: bool = True          # if False, one random coin shared across all times
    su2_convention: SU2Convention = "zyz"


@dataclass(frozen=True)
class GlobalScheduleConfig:
    """
    Global (time-dependent, node-independent) SU(2) schedule baseline.

    You can pick a basis:
      - fourier: low-frequency Fourier series over normalized time τ=t/(T-1)
      - poly: polynomial in τ
      - piecewise_constant: K steps

    This is a *novelty-grade baseline* because you can:
      - fix it deterministically (seeded) and report results, OR
      - tune it (random search / grid search) without learning a GNN,
        then compare your learned policy against a tuned global schedule.

    Parameters
    ----------
    basis:
        Which schedule family to use.
    K:
        Complexity / number of terms / pieces.
    seed:
        Determines coefficients (deterministic).
    amplitude:
        Scales the schedule variation around an optional offset.
    offset:
        Base angles (alpha,beta,gamma) around which the schedule varies.
        If None, uses identity (0,0,0).
    clamp_beta:
        Keep beta within [0, pi] by clamping after wrapping.
    """

    basis: GlobalScheduleBasis = "fourier"
    K: int = 3
    seed: int = 0
    amplitude: float = 0.75
    offset: Optional[Tuple[float, float, float]] = None
    clamp_beta: bool = True
    su2_convention: SU2Convention = "zyz"


@dataclass(frozen=True)
class BaselineCoinConfig:
    """
    Master baseline config for producing SU(2) parameters (theta) and/or unitary coins.

    Exactly one of:
      - fixed
      - random
      - global_schedule

    should be provided.

    family:
      - "su2_theta": produce theta with shape (T,N,3) for SU(2) ZYZ lift
      - "unitary_by_degree": produce dict[deg] -> (deg,deg) unitary (useful if your sim
                           consumes degree-specific coins directly).
    """

    family: Literal["su2_theta", "unitary_by_degree"] = "su2_theta"
    fixed: Optional[FixedCoinConfig] = None
    random: Optional[RandomCoinConfig] = None
    global_schedule: Optional[GlobalScheduleConfig] = None

    # If your graph has varying degrees and you want a coherent degree-aware baseline:
    # - for deg==2, fixed.kind applies (hadamard/grover/identity)
    # - for deg!=2, we default to grover or identity depending on fixed.kind
    degree_fallback: Literal["grover", "identity"] = "grover"


# =============================================================================
# Deterministic seeding helpers (research/repro)
# =============================================================================

def _blake2b_u64(*parts: Union[str, int, bytes]) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        elif isinstance(p, int):
            h.update(str(p).encode("utf-8"))
        else:
            h.update(p.encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def _np_rng(seed: int, *, salt: str = "") -> np.random.Generator:
    s = _blake2b_u64("acpl.baselines.coins", seed, salt)
    return np.random.default_rng(s)


# =============================================================================
# Angle utilities
# =============================================================================

TAU = 2.0 * math.pi


def wrap_angle_2pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles into [0, 2π)."""
    return torch.remainder(x, TAU)


def wrap_angle_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles into (-π, π]."""
    y = torch.remainder(x + math.pi, TAU) - math.pi
    # map -pi to +pi for consistency
    y = torch.where(torch.isclose(y, x.new_tensor(-math.pi)), y + TAU, y)
    return y





FOUR_PI = 2.0 * TAU

def wrap_angle_4pi(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap angles into (-2π, 2π]. (4π-periodic)

    Important for SU(2) Euler lifts where α,γ appear as half-angles: shifting by 2π
    multiplies U by -I, so only 4π-periodic wrapping preserves the SU(2) element.
    """
    y = torch.remainder(x + TAU, FOUR_PI) - TAU  # [-2π, 2π)
    y = torch.where(torch.isclose(y, x.new_tensor(-TAU)), y + FOUR_PI, y)  # (-2π, 2π]
    return y







# =============================================================================
# SU(n) projection (unitary determinant fix)
# =============================================================================

def project_to_su(U: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Project a unitary U ∈ U(n) to SU(n) by removing the global phase:
      U_su = U / det(U)^(1/n)

    Works for complex tensors with shape (..., n, n).
    """
    if U.ndim < 2 or U.shape[-1] != U.shape[-2]:
        raise ValueError(f"project_to_su expects (...,n,n). Got {tuple(U.shape)}")
    n = U.shape[-1]
    det = torch.linalg.det(U)
    # avoid zero
    det = torch.where(det.abs() < eps, det + (eps + 0j), det)
    phase = det ** (1.0 / float(n))
    return U / phase.unsqueeze(-1).unsqueeze(-1)


# =============================================================================
# SU(2) unitary <-> ZYZ angles (convention must match your lift)
# =============================================================================

def su2_from_zyz(theta: torch.Tensor) -> torch.Tensor:
    r"""
    Build SU(2) matrix from ZYZ Euler angles.

    We use the common convention:
      U(α,β,γ) = Rz(α) Ry(β) Rz(γ)

    which yields:
      a = e^{-i(α+γ)/2} cos(β/2)
      b = -e^{-i(α-γ)/2} sin(β/2)
      U = [[a, b],
           [-b*, a*]]

    theta: (...,3) with (α,β,γ)
    returns: (...,2,2) complex64/complex128
    """
    if theta.shape[-1] != 3:
        raise ValueError(f"su2_from_zyz expects (...,3). Got {tuple(theta.shape)}")

    alpha = theta[..., 0]
    beta = theta[..., 1]
    gamma = theta[..., 2]

    # promote to complex
    dtype = (
        torch.complex64 if theta.dtype in (torch.float16, torch.float32, torch.bfloat16) else torch.complex128
    )
    
    
    
    device = theta.device

    half = 0.5
    ca = torch.cos(beta * half)
    sa = torch.sin(beta * half)

    phase_a = torch.exp((-0.5j) * (alpha + gamma).to(dtype=dtype))
    phase_b = torch.exp((-0.5j) * (alpha - gamma).to(dtype=dtype))

    a = phase_a * ca.to(dtype=dtype)
    b = -phase_b * sa.to(dtype=dtype)

    U = torch.empty(theta.shape[:-1] + (2, 2), device=device, dtype=dtype)
    U[..., 0, 0] = a
    U[..., 0, 1] = b
    U[..., 1, 0] = -torch.conj(b)
    U[..., 1, 1] = torch.conj(a)
    # should already be SU(2); projection is harmless but a bit costly. Keep off by default.
    return U


def zyz_from_su2(U: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    r"""
    Recover ZYZ angles (α,β,γ) from an SU(2) matrix U using the same convention as su2_from_zyz.

    Input:
      U: (...,2,2) complex tensor
    Output:
      theta: (...,3) float tensor in ranges:
        α,γ in (-π,π], β in [0,π]

    Notes:
      - Angles are not unique at β≈0 or β≈π. We choose a stable gauge in those cases.
      - This is intended for baseline construction (Hadamard/Grover), not for gradient-based learning.
    """
    if U.ndim < 2 or U.shape[-2:] != (2, 2):
        raise ValueError(f"zyz_from_su2 expects (...,2,2). Got {tuple(U.shape)}")
    if not torch.is_complex(U):
        raise TypeError("zyz_from_su2 expects complex input U.")

    # enforce SU(2) up to numerical noise
    U2 = project_to_su(U)

    a = U2[..., 0, 0]
    b = U2[..., 0, 1]

    # beta from magnitudes: |a| = cos(beta/2), |b| = sin(beta/2)
    ca = a.abs().clamp(0.0, 1.0)
    sa = b.abs().clamp(0.0, 1.0)
    beta = 2.0 * torch.atan2(sa, ca)  # in [0,pi]

    # phases
    # a = e^{-i(α+γ)/2} cos(beta/2)
    # b = -e^{-i(α-γ)/2} sin(beta/2)
    # define:
    #   phi_a = arg(a) = -(α+γ)/2
    #   phi_b = arg(-b) = -(α-γ)/2
    phi_a = torch.angle(a)
    phi_b = torch.angle(-b)

    # singular cases
    sin_small = sa < eps
    cos_small = ca < eps

    # default solve
    alpha = -(phi_a + phi_b)
    gamma = -phi_a + phi_b

    # if beta ~ 0 -> b ~ 0, phi_b unstable. choose gamma = -2*phi_a, alpha=0.
    alpha = torch.where(sin_small, torch.zeros_like(alpha), alpha)
    gamma = torch.where(sin_small, -2.0 * phi_a, gamma)

    # if beta ~ pi -> a ~ 0, phi_a unstable. choose alpha = -2*phi_b, gamma=0.
    alpha = torch.where(cos_small, -2.0 * phi_b, alpha)
    gamma = torch.where(cos_small, torch.zeros_like(gamma), gamma)

    # wrap
    alpha = wrap_angle_pi(alpha)
    gamma = wrap_angle_pi(gamma)
    # beta already in [0,pi], but clamp for safety
    beta = beta.clamp(0.0, math.pi)

    theta = torch.stack([alpha, beta, gamma], dim=-1).to(dtype=torch.float32)
    return theta


# =============================================================================
# Canonical fixed coins
# =============================================================================

def identity_unitary(d: int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    if d <= 0:
        raise ValueError("d must be positive.")
    return torch.eye(d, device=device, dtype=dtype)


def grover_unitary(d: int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    Grover diffusion coin:
      G_d = (2/d) * 11^T - I

    This is real orthogonal (hence unitary). Determinant is (-1)^(d-1).
    If you need SU(d), apply project_to_su().
    """
    if d <= 0:
        raise ValueError("d must be positive.")
    one = torch.ones((d, d), device=device, dtype=dtype)
    I = torch.eye(d, device=device, dtype=dtype)
    G = (2.0 / float(d)) * one - I
    # ensure complex dtype (in case dtype was complex)
    return G.to(dtype=dtype)


def hadamard_su2(*, device: Optional[torch.device] = None, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    SU(2)-compatible Hadamard coin for d=2.

    Standard Hadamard H has det(H)=-1. Multiply by global phase i so det(iH)=1:
      U_H = i * (1/sqrt(2)) [[1,1],[1,-1]]  ∈ SU(2)
    """
    s = 1.0 / math.sqrt(2.0)
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=torch.float32) * s
    U = (1j * H.to(dtype=torch.complex64)).to(dtype=dtype)
    U = project_to_su(U)
    return U


def fixed_coin_su2_theta(kind: FixedCoinKind, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Return a single SU(2) theta=(α,β,γ) for a canonical fixed coin (d=2 only).
    """
    if kind == "identity":
        # identity is θ=(0,0,0)
        return torch.zeros((3,), device=device, dtype=torch.float32)

    if kind == "hadamard":
        # Canonical (defendable) representative for U_H = iH in our ZYZ convention.
        # Using wrapped angle recovery can land on the -U representative, which is not
        # always a *global* phase when degrees differ across nodes (e.g., line endpoints).
        #
        # This theta satisfies su2_from_zyz(theta) == hadamard_su2() (up to fp error),
        # without relying on inversion heuristics:
        #   alpha = 2π, beta = π/2, gamma = π
        return torch.tensor([TAU, 0.5 * math.pi, math.pi], device=device, dtype=torch.float32)


    if kind == "grover":
        # Canonical representative for iX in our ZYZ convention:
        #   alpha = 2π, beta = π, gamma = π
        return torch.tensor([TAU, math.pi, math.pi], device=device, dtype=torch.float32)

    raise ValueError(f"Unknown fixed coin kind: {kind}")


# =============================================================================
# SU(2) baseline generators (theta with shape (T,N,3))
# =============================================================================

def _broadcast_theta(theta_single: torch.Tensor, *, T: int, N: int) -> torch.Tensor:
    if theta_single.shape != (3,):
        raise ValueError(f"Expected theta_single shape (3,), got {tuple(theta_single.shape)}")
    return theta_single.view(1, 1, 3).expand(T, N, 3).contiguous()


def make_su2_theta_baseline(
    *,
    kind: Literal["fixed", "random", "global_schedule"],
    T: int,
    N: int,
    device: Union[str, torch.device] = "cpu",
    fixed: Optional[FixedCoinConfig] = None,
    random: Optional[RandomCoinConfig] = None,
    global_schedule: Optional[GlobalScheduleConfig] = None,
) -> torch.Tensor:
    """
    Unified entrypoint: produce SU(2) theta with shape (T,N,3).
    """
    device = torch.device(device)
    if T <= 0 or N <= 0:
        raise ValueError("T and N must be positive.")

    if kind == "fixed":
        if fixed is None:
            fixed = FixedCoinConfig(kind="hadamard")
        theta0 = fixed_coin_su2_theta(fixed.kind, device=device)
        return _broadcast_theta(theta0, T=T, N=N)

    if kind == "random":
        if random is None:
            random = RandomCoinConfig(seed=0, haar=True)
        return random_su2_theta(T=T, N=N, cfg=random, device=device)

    if kind == "global_schedule":
        if global_schedule is None:
            global_schedule = GlobalScheduleConfig(basis="fourier", K=3, seed=0)
        return global_schedule_su2_theta(T=T, N=N, cfg=global_schedule, device=device)

    raise ValueError(f"Unknown kind: {kind}")


# =============================================================================
# Random SU(2) theta
# =============================================================================

def random_su2_theta(
    *,
    T: int,
    N: int,
    cfg: RandomCoinConfig,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Random SU(2) theta schedule with shape (T,N,3).

    If cfg.haar=True:
      - α,γ ~ Uniform(0,2π)
      - u ~ Uniform(0,1), set cos(β/2)=sqrt(u), sin(β/2)=sqrt(1-u)
        => β = 2 arccos(sqrt(u)), which matches Haar on SU(2) under ZYZ.

    If cfg.haar=False:
      - α,γ ~ Uniform(-π,π), β ~ Uniform(0,π)
    """
    if T <= 0 or N <= 0:
        raise ValueError("T and N must be positive.")
    device = torch.device(device)

    rng = _np_rng(cfg.seed, salt="random_su2_theta")

    # determine sampling granularity
    # base samples shape depends on per_node/per_time flags
    t_dim = T if cfg.per_time else 1
    n_dim = N if cfg.per_node else 1

    if cfg.haar:
        a = rng.uniform(0.0, TAU, size=(t_dim, n_dim)).astype(np.float64)
        g = rng.uniform(0.0, TAU, size=(t_dim, n_dim)).astype(np.float64)
        u = rng.uniform(0.0, 1.0, size=(t_dim, n_dim)).astype(np.float64)
        # Haar for SU(2): beta = 2 arccos(sqrt(u)) ∈ [0,pi]
        b = 2.0 * np.arccos(np.sqrt(np.clip(u, 0.0, 1.0)))
    else:
        a = rng.uniform(-math.pi, math.pi, size=(t_dim, n_dim)).astype(np.float64)
        g = rng.uniform(-math.pi, math.pi, size=(t_dim, n_dim)).astype(np.float64)
        b = rng.uniform(0.0, math.pi, size=(t_dim, n_dim)).astype(np.float64)

    theta = np.stack([a, b, g], axis=-1)  # (t_dim, n_dim, 3)
    theta_t = torch.from_numpy(theta).to(device=device, dtype=torch.float32)

    # broadcast to (T,N,3)
    if not cfg.per_time:
        theta_t = theta_t.expand(T, theta_t.shape[1], 3)
    if not cfg.per_node:
        theta_t = theta_t.expand(theta_t.shape[0], N, 3)

    # final wrap
    theta_t[..., 0] = wrap_angle_4pi(theta_t[..., 0])
    theta_t[..., 2] = wrap_angle_4pi(theta_t[..., 2])
    theta_t[..., 1] = theta_t[..., 1].clamp(0.0, math.pi)
    return theta_t.contiguous()


# =============================================================================
# Global schedules (research-grade baseline)
# =============================================================================

def _time_grid(T: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if T == 1:
        return torch.zeros((1,), device=device, dtype=dtype)
    return torch.linspace(0.0, 1.0, steps=T, device=device, dtype=dtype)


def global_schedule_su2_theta(
    *,
    T: int,
    N: int,
    cfg: GlobalScheduleConfig,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Produce a time-varying, node-independent SU(2) schedule θ(t) and broadcast to (T,N,3).
    """
    if T <= 0 or N <= 0:
        raise ValueError("T and N must be positive.")
    if cfg.K <= 0:
        raise ValueError("cfg.K must be positive.")
    device = torch.device(device)

    offset = cfg.offset if cfg.offset is not None else (0.0, 0.0, 0.0)
    theta0 = torch.tensor(offset, device=device, dtype=torch.float32)  # (3,)

    tau = _time_grid(T, device=device)  # (T,)

    rng = _np_rng(cfg.seed, salt=f"global_schedule:{cfg.basis}:{cfg.K}")
    # coefficients for each angle channel
    # shape (3, K) for basis terms
    coef = rng.normal(loc=0.0, scale=1.0, size=(3, cfg.K)).astype(np.float64)
    coef_t = torch.from_numpy(coef).to(device=device, dtype=torch.float32)

    if cfg.basis == "fourier":
        # θ(t) = θ0 + A * Σ_k c_k sin(2π k τ) + d_k cos(2π k τ)
        # For simplicity, use a single set of coefficients and phase-shift via cos/sin mix.
        # (still deterministic and tunable via seed)
        coef2 = rng.normal(loc=0.0, scale=1.0, size=(3, cfg.K)).astype(np.float64)
        coef2_t = torch.from_numpy(coef2).to(device=device, dtype=torch.float32)

        ks = torch.arange(1, cfg.K + 1, device=device, dtype=torch.float32)  # (K,)
        arg = (TAU * tau.view(T, 1)) * ks.view(1, cfg.K)  # (T,K)
        s = torch.sin(arg)
        c = torch.cos(arg)

        delta = (s @ coef_t.transpose(0, 1)) + (c @ coef2_t.transpose(0, 1))  # (T,3)

    elif cfg.basis == "poly":
        # θ(t) = θ0 + A * Σ_k c_k τ^k
        powers = torch.stack([tau ** (k + 1) for k in range(cfg.K)], dim=1)  # (T,K)
        delta = powers @ coef_t.transpose(0, 1)  # (T,3)

    elif cfg.basis == "piecewise_constant":
        # K segments: choose K values and repeat
        vals = coef_t.transpose(0, 1)  # (K,3)
        # map time indices to segments
        seg = torch.clamp((tau * cfg.K).to(torch.int64), max=cfg.K - 1)  # (T,)
        delta = vals.index_select(dim=0, index=seg)  # (T,3)

    else:
        raise ValueError(f"Unknown schedule basis: {cfg.basis}")

    theta_t = theta0.view(1, 3) + float(cfg.amplitude) * delta  # (T,3)

    # wrap / clamp
    theta_t[:, 0] = wrap_angle_pi(theta_t[:, 0])
    theta_t[:, 2] = wrap_angle_pi(theta_t[:, 2])
    if cfg.clamp_beta:
        theta_t[:, 1] = theta_t[:, 1].clamp(0.0, math.pi)
    else:
        # still keep it in a reasonable range
        theta_t[:, 1] = wrap_angle_pi(theta_t[:, 1]).abs().clamp(0.0, math.pi)

    # broadcast to (T,N,3)
    out = theta_t.view(T, 1, 3).expand(T, N, 3).contiguous()
    return out


# =============================================================================
# Degree-aware unitary baseline (for sims that want explicit coins per degree)
# =============================================================================

def make_unitary_coin_baseline_by_degree(
    degrees: Iterable[int],
    *,
    cfg: BaselineCoinConfig,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> Dict[int, torch.Tensor]:
    """
    Produce a dict mapping degree -> (deg,deg) unitary coin matrix.

    Intended use:
      - If your simulator groups arcs by out-degree and expects one coin per degree bucket.

    Policy:
      - deg == 1: identity
      - deg == 2: use fixed.kind (hadamard/grover/identity)
      - deg > 2:
          * if fixed.kind == "identity" -> identity
          * else use cfg.degree_fallback ("grover" or "identity")
    """
    device = torch.device(device)
    degs = sorted({int(d) for d in degrees})
    out: Dict[int, torch.Tensor] = {}

    fixed_kind: FixedCoinKind = "hadamard"
    if cfg.fixed is not None:
        fixed_kind = cfg.fixed.kind
    elif cfg.random is not None or cfg.global_schedule is not None:
        # unitary_by_degree is primarily for fixed / grover style baselines;
        # for random/schedule, keep a deterministic grover/identity fallback.
        fixed_kind = "grover"

    for d in degs:
        if d <= 0:
            raise ValueError(f"Invalid degree: {d}")
        if d == 1:
            out[d] = identity_unitary(1, device=device, dtype=dtype)
            continue
        if d == 2:
            if fixed_kind == "identity":
                U = identity_unitary(2, device=device, dtype=dtype)
            elif fixed_kind == "grover":
                # Use canonical iX (SU(2)) rather than projecting X, which can choose -iX.
                X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device, dtype=torch.float32)
                U = (1j * X.to(torch.complex64)).to(dtype=dtype)
                U = project_to_su(U)  # safe
            
            
            
            else:
                U = hadamard_su2(device=device, dtype=dtype)
            out[d] = U
            continue

        # d > 2
        if fixed_kind == "identity" or cfg.degree_fallback == "identity":
            out[d] = identity_unitary(d, device=device, dtype=dtype)
        else:
            out[d] = grover_unitary(d, device=device, dtype=dtype)
    return out


# =============================================================================
# Master config entrypoint (baseline config -> theta or unitary map)
# =============================================================================

def _validate_baseline_cfg(cfg: BaselineCoinConfig) -> None:
    n = int(cfg.fixed is not None) + int(cfg.random is not None) + int(cfg.global_schedule is not None)
    if n != 1:
        raise ValueError(
            "BaselineCoinConfig must set exactly one of: fixed, random, global_schedule."
        )


def make_baseline_from_config(
    *,
    cfg: BaselineCoinConfig,
    T: int,
    N: int,
    degrees: Optional[Iterable[int]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Convenience wrapper:
      - if cfg.family == "su2_theta": returns theta (T,N,3)
      - if cfg.family == "unitary_by_degree": returns dict[deg] -> (deg,deg) unitary

    degrees is required for unitary_by_degree.
    """
    _validate_baseline_cfg(cfg)
    if cfg.family == "su2_theta":
        if cfg.fixed is not None:
            return make_su2_theta_baseline(kind="fixed", T=T, N=N, device=device, fixed=cfg.fixed)
        if cfg.random is not None:
            return make_su2_theta_baseline(kind="random", T=T, N=N, device=device, random=cfg.random)
        assert cfg.global_schedule is not None
        return make_su2_theta_baseline(kind="global_schedule", T=T, N=N, device=device, global_schedule=cfg.global_schedule)

    if cfg.family == "unitary_by_degree":
        if degrees is None:
            raise ValueError("degrees is required when cfg.family == 'unitary_by_degree'.")
        return make_unitary_coin_baseline_by_degree(degrees, cfg=cfg, device=device)

    raise ValueError(f"Unknown cfg.family: {cfg.family}")







# =============================================================================
# Schedule objects + builders (for BaselinePolicy auto-construction)
# =============================================================================

@dataclass(frozen=True)
class FixedSU2ThetaSchedule:
    cfg: FixedCoinConfig = FixedCoinConfig(kind="hadamard")

    def __call__(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        T: int,
        edge_weight: torch.Tensor | None = None,
        outdeg: torch.Tensor | None = None,
        indeg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = int(X.shape[0])
        return make_su2_theta_baseline(kind="fixed", T=int(T), N=N, device=X.device, fixed=self.cfg)


@dataclass(frozen=True)
class RandomSU2ThetaSchedule:
    cfg: RandomCoinConfig = RandomCoinConfig(seed=0, haar=True, per_node=True, per_time=True)

    def __call__(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        T: int,
        edge_weight: torch.Tensor | None = None,
        outdeg: torch.Tensor | None = None,
        indeg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = int(X.shape[0])
        return make_su2_theta_baseline(kind="random", T=int(T), N=N, device=X.device, random=self.cfg)


@dataclass(frozen=True)
class GlobalScheduleSU2ThetaSchedule:
    cfg: GlobalScheduleConfig = GlobalScheduleConfig(basis="fourier", K=3, seed=0)

    def __call__(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        T: int,
        edge_weight: torch.Tensor | None = None,
        outdeg: torch.Tensor | None = None,
        indeg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = int(X.shape[0])
        return make_su2_theta_baseline(kind="global_schedule", T=int(T), N=N, device=X.device, global_schedule=self.cfg)


def make_coin_schedule(kind: str, **kwargs: Any):
    """
    Baseline schedule factory used by acpl/baselines/policies.py.

    Supported kinds (aliases included):
      - "hadamard" / "grover" / "identity"  -> fixed SU(2) theta schedule
      - "fixed"                              -> FixedSU2ThetaSchedule (requires fixed=... or kind=...)
      - "random"                             -> RandomSU2ThetaSchedule
      - "global_schedule" / "schedule"       -> GlobalScheduleSU2ThetaSchedule
    """
    k = str(kind).lower().strip()

    if k in {"hadamard", "grover", "identity"}:
        return FixedSU2ThetaSchedule(FixedCoinConfig(kind=k))  # type: ignore[arg-type]

    if k == "fixed":
        fixed_cfg = kwargs.get("fixed", None)
        if isinstance(fixed_cfg, FixedCoinConfig):
            return FixedSU2ThetaSchedule(fixed_cfg)
        # allow passing fixed.kind directly
        fk = kwargs.get("kind", kwargs.get("fixed_kind", "hadamard"))
        return FixedSU2ThetaSchedule(FixedCoinConfig(kind=str(fk)))  # type: ignore[arg-type]

    if k == "random":
        rcfg = kwargs.get("random", None)
        if isinstance(rcfg, RandomCoinConfig):
            return RandomSU2ThetaSchedule(rcfg)
        return RandomSU2ThetaSchedule(RandomCoinConfig(**kwargs))

    if k in {"global_schedule", "schedule"}:
        gcfg = kwargs.get("global_schedule", None)
        if isinstance(gcfg, GlobalScheduleConfig):
            return GlobalScheduleSU2ThetaSchedule(gcfg)
        return GlobalScheduleSU2ThetaSchedule(GlobalScheduleConfig(**kwargs))

    raise ValueError(
        f"Unknown baseline schedule kind='{kind}'. "
        "Use hadamard/grover/identity/fixed/random/global_schedule."
    )


def build(*, kind: str, **kwargs: Any):
    """Alias for make_coin_schedule(kind, **kwargs) (supported by policies.py)."""
    return make_coin_schedule(kind, **kwargs)




# =============================================================================
# Research-grade helper: tune global schedule without learning (random search)
# =============================================================================

@dataclass(frozen=True)
class RandomSearchResult:
    best_score: float
    best_cfg: GlobalScheduleConfig
    tried: int


def random_search_global_schedule(
    *,
    objective: Callable[[GlobalScheduleConfig], float],
    base_cfg: GlobalScheduleConfig,
    trials: int = 64,
    seed: int = 0,
) -> RandomSearchResult:
    """
    Random-search tuner for global schedule baselines.

    This is intentionally simple and dependency-free:
      - We vary only the schedule seed and (optionally) amplitude slightly.
      - You can extend this in your thesis experiments (grid over K/basis/amplitude).

    Why this matters for novelty:
      Your ACPL policy should beat:
        (i) fixed coins,
        (ii) random coins,
        (iii) *tuned* global schedules (time-only control),
      showing the value of *node-adaptive* control learned by the GNN.

    Parameters
    ----------
    objective:
        Function that takes a GlobalScheduleConfig and returns a scalar score.
        Higher is better (e.g., success probability at target, negative loss, etc.).
    base_cfg:
        Starting config (basis/K/amplitude/offset/clamp_beta fixed).
    trials:
        Number of random seeds to try.
    seed:
        Master seed controlling which schedule seeds are tried.

    Returns
    -------
    RandomSearchResult(best_score, best_cfg, tried)
    """
    if trials <= 0:
        raise ValueError("trials must be positive.")

    rng = _np_rng(seed, salt="random_search_global_schedule")
    best_score = -float("inf")
    best_cfg = base_cfg

    for i in range(int(trials)):
        # deterministically propose a new schedule seed + mild amplitude jitter
        sched_seed = int(rng.integers(0, 2**31 - 1))
        amp_jit = float(base_cfg.amplitude) * float(rng.uniform(0.75, 1.25))
        cand = GlobalScheduleConfig(
            basis=base_cfg.basis,
            K=base_cfg.K,
            seed=sched_seed,
            amplitude=amp_jit,
            offset=base_cfg.offset,
            clamp_beta=base_cfg.clamp_beta,
            su2_convention=base_cfg.su2_convention,
        )
        score = float(objective(cand))
        if score > best_score:
            best_score = score
            best_cfg = cand

    return RandomSearchResult(best_score=best_score, best_cfg=best_cfg, tried=int(trials))

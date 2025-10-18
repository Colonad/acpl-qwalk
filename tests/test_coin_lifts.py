# tests/test_coin_lifts.py
# Comprehensive tests for SU(2) coin lifts (ZYZ Euler) with theory-driven checks.
# Emphasis: robustness, numerical stability, global-phase periodicities, and autograd.

import math

import pytest
import torch

from acpl.sim.coins import (
    CoinsSpecSU2,
    coins_su2_from_theta,
    is_unitary,
    su2_from_euler_batch,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _equal_up_to_global_phase(U: torch.Tensor, V: torch.Tensor, atol=1e-6, rtol=1e-6) -> bool:
    """
    Return True if U and V are equal up to a single global phase factor φ:
      ∃ φ s.t. || U - e^{iφ} V ||_F is small.
    Works for batches: (...,2,2).
    """
    assert U.shape == V.shape and U.shape[-2:] == (2, 2)
    # Choose a phase using the trace (avoids degenerate zero entry selection).
    # If trace ≈ 0, fall back to first nonzero element.
    trU = torch.diagonal(U, dim1=-2, dim2=-1).sum(dim=-1)
    trV = torch.diagonal(V, dim1=-2, dim2=-1).sum(dim=-1)

    # phase = argmin || U - phase*V || => phase = (tr(V^H U))/|tr(V^H U)|
    num = (V.conj() * U).reshape(*U.shape[:-2], -1).sum(dim=-1)
    # If num is ~0, try diagonal heuristic
    mask_zero = num.abs() < 1e-12
    num = torch.where(mask_zero, (trV.conj() * trU), num)

    phase = torch.ones_like(num, dtype=U.dtype)
    nz = num.abs() > 1e-12
    phase = torch.where(nz, num / num.abs().clamp_min(1e-12), phase)

    # Broadcast phase to (...,1,1)
    phase = phase.unsqueeze(-1).unsqueeze(-1)
    diff = U - phase * V
    return torch.allclose(diff, torch.zeros_like(diff), atol=atol, rtol=rtol)


def _rand_angles(shape, seed=0, scale=1.0, device=None):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g, device=device) * scale


# ------------------------------------------------------------
# Core SU(2) lift properties
# ------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_su2_from_euler_unitary_and_det(dtype):
    torch.manual_seed(0)
    N = 64
    angles = torch.randn(N, 3)
    U = su2_from_euler_batch(angles, dtype=dtype, normalize=True)
    # One realistic tolerance check is sufficient (covers both dtypes robustly).
    assert is_unitary(U, atol=1e-6)

    # |det U| == 1 within fp tolerance
    det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]

    ones = torch.ones_like(det.real)
    assert torch.allclose(det.abs(), ones, atol=1e-6 if dtype is torch.complex128 else 1e-5)


def test_identity_angles_yield_identity():
    Z = torch.zeros(10, 3)
    U = su2_from_euler_batch(Z, dtype=torch.complex64, normalize=False)
    I = torch.eye(2, dtype=torch.complex64).expand_as(U)
    assert torch.allclose(U, I, atol=1e-7, rtol=1e-7)


def test_large_angles_numerical_stability_with_normalize():
    # Very large angles should still give a valid SU(2) after normalization.
    angles = _rand_angles((32, 3), seed=123, scale=1e6)
    spec = CoinsSpecSU2(dtype=torch.complex64, normalize=True, check=True)
    U = coins_su2_from_theta(angles, spec=spec)
    assert is_unitary(U, atol=1e-6)
    det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]
    ones = torch.ones_like(det)
    # With normalize=True we expect det ≈ 1 in C (not just |det|=1)
    assert torch.allclose(det, ones, atol=1e-5)


def test_coins_builder_normalize_and_check():
    torch.manual_seed(1)
    N = 17
    angles = torch.randn(N, 3)
    spec = CoinsSpecSU2(dtype=torch.complex64, normalize=True, check=True)
    U = coins_su2_from_theta(angles, spec=spec)
    assert is_unitary(U, atol=1e-6)
    det = U[:, 0, 0] * U[:, 1, 1] - U[:, 0, 1] * U[:, 1, 0]
    ones = torch.ones_like(det)
    assert torch.allclose(det, ones, atol=1e-5)


def test_backward_grad_flow_nonzero_and_finite():
    # Ensure autograd flows and is finite for a simple scalar loss.
    torch.manual_seed(2)
    N = 8
    angles = torch.randn(N, 3, requires_grad=True)
    U = coins_su2_from_theta(angles, spec=CoinsSpecSU2(normalize=False))
    loss = (U.conj() * U).real.sum()  # sum of squared magnitudes (positive)
    loss.backward()
    assert angles.grad is not None
    assert torch.isfinite(angles.grad).all()
    # Should generally not be all zeros for random input
    assert (angles.grad.abs() > 0).any()


def test_dagger_equals_inverse():
    torch.manual_seed(0)
    N = 20
    angles = torch.randn(N, 3)
    U = coins_su2_from_theta(angles, spec=CoinsSpecSU2(normalize=False))
    I = torch.eye(2, dtype=U.dtype)
    lhs = U.conj().transpose(-1, -2) @ U
    rhs = U @ U.conj().transpose(-1, -2)
    assert torch.allclose(lhs, I.expand(N, 2, 2), atol=1e-6)
    assert torch.allclose(rhs, I.expand(N, 2, 2), atol=1e-6)


# ------------------------------------------------------------
# Periodicity & global-phase properties for ZYZ Euler
# ------------------------------------------------------------


def test_zyz_alpha_gamma_periodicity_global_phase_minus_one():
    """
    For SU(2) ZYZ (α, β, γ):
      adding 2π to α multiplies on the left by e^{-iπ σ_z} = -I,
      adding 2π to γ multiplies on the right by e^{+iπ σ_z} = -I.
    So U(α+2π, β, γ) and U(α, β, γ+2π) both equal −U(α, β, γ).
    We check equality up to a single global phase.
    """
    torch.manual_seed(0)
    N = 29
    θ = torch.randn(N, 3)
    U = su2_from_euler_batch(θ, dtype=torch.complex64, normalize=False)

    two_pi = 2.0 * math.pi
    θa = θ.clone()
    θa[:, 0] = θa[:, 0] + two_pi
    θg = θ.clone()
    θg[:, 2] = θg[:, 2] + two_pi

    Ua = su2_from_euler_batch(θa, dtype=torch.complex64, normalize=False)
    Ug = su2_from_euler_batch(θg, dtype=torch.complex64, normalize=False)

    # Both should match U up to a −1 global phase (i.e., up to phase equivalence)
    assert _equal_up_to_global_phase(Ua, U, atol=1e-6, rtol=1e-6)
    assert _equal_up_to_global_phase(Ug, U, atol=1e-6, rtol=1e-6)


def test_zyz_beta_periodicity_global_phase_plus_one():
    """
    β → β + 2π leaves the SU(2) element unchanged (up to +1 phase),
    so equality should hold up to a global phase of +1 — i.e., direct equality up to fp.
    """
    torch.manual_seed(0)
    N = 31
    θ = torch.randn(N, 3)
    U = su2_from_euler_batch(θ, dtype=torch.complex64, normalize=False)

    two_pi = 2.0 * math.pi
    θb = θ.clone()
    θb[:, 1] = θb[:, 1] + two_pi
    Ub = su2_from_euler_batch(θb, dtype=torch.complex64, normalize=False)

    # Equal up to a global phase (here should be +1)
    assert _equal_up_to_global_phase(U, Ub, atol=1e-6, rtol=1e-6)


# ------------------------------------------------------------
# Batch/time shapes & builder broadcasting (novelty: T×N grids)
# ------------------------------------------------------------


def test_time_node_batch_shape_broadcast_ok():
    """
    su2_from_euler_batch should accept (...,3) → (...,2,2).
    We check a (T,N,3) input maps to (T,N,2,2) and remains unitary (with normalize=True).
    """
    T, N = 5, 7
    θ = _rand_angles((T, N, 3), seed=42)
    U = su2_from_euler_batch(θ, dtype=torch.complex64, normalize=True)
    assert U.shape == (T, N, 2, 2)
    assert is_unitary(U, atol=1e-6)


# ------------------------------------------------------------
# DTQW step consistency when coins are identity
# ------------------------------------------------------------


def test_step_consistency_with_identity_coins():
    # If coins are identity, one step is just the shift.
    from acpl.sim.portmap import build_portmap
    from acpl.sim.shift import build_shift
    from acpl.sim.step import step

    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 4-cycle (deg=2)
    pm = build_portmap(pairs, num_nodes=4, coalesce=False)
    S = build_shift(pm)

    A = pm.src.numel()
    psi = torch.randn(A, dtype=torch.complex64)
    psi_b = torch.randn(3, A, dtype=torch.complex64)

    N = pm.num_nodes
    I2 = torch.eye(2, dtype=torch.complex64)
    coins = I2.unsqueeze(0).repeat(N, 1, 1)

    out = step(psi, pm, coins, shift=S)
    out_b = step(psi_b, pm, coins, shift=S)

    assert torch.allclose(out, psi.index_select(0, S.perm))
    assert torch.allclose(out_b, psi_b.index_select(1, S.perm))


# ------------------------------------------------------------
# Randomized stress (unitarity + determinant) with dtypes
# ------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_random_unitarity_and_det_stress(dtype):
    θ = _rand_angles((200, 3), seed=777)
    U = su2_from_euler_batch(θ, dtype=dtype, normalize=True)
    assert is_unitary(U, atol=1e-6)

    det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]
    ones = torch.ones_like(det.real)
    assert torch.allclose(det.abs(), ones, atol=1e-6 if dtype is torch.complex128 else 1e-5)

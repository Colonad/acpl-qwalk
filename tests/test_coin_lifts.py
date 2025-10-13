# tests/test_coin_lifts.py
import math

import pytest
import torch

from acpl.sim.coins import (
    coins_su2_from_theta,
    is_unitary,
    ry,
    rz,
    su2_from_euler,
)


def test_rz_identity_and_pi():
    theta0 = torch.tensor(0.0, dtype=torch.float32)
    z0 = rz(theta0)
    eye = torch.eye(2, dtype=torch.complex64)
    assert torch.allclose(z0.squeeze(0), eye, atol=1e-7, rtol=0)

    thetap = torch.tensor(math.pi, dtype=torch.float32)
    zp = rz(thetap).squeeze(0)
    # Rz(pi) = diag(e^{-i pi/2}, e^{+i pi/2}) = diag(-i, i)
    exp_neg = torch.exp(torch.tensor(-1j * math.pi / 2, dtype=torch.complex64))
    exp_pos = torch.exp(torch.tensor(+1j * math.pi / 2, dtype=torch.complex64))
    expect = torch.diag(torch.stack([exp_neg, exp_pos]))
    assert torch.allclose(zp, expect, atol=1e-6, rtol=0)
    assert is_unitary(zp)


def test_ry_identity_and_pi():
    theta0 = torch.tensor(0.0, dtype=torch.float32)
    y0 = ry(theta0).squeeze(0).to(dtype=torch.complex64)
    eye = torch.eye(2, dtype=torch.complex64)
    assert torch.allclose(y0, eye, atol=1e-7, rtol=0)

    thetap = torch.tensor(math.pi, dtype=torch.float32)
    yp = ry(thetap).squeeze(0).to(dtype=torch.complex64)
    # Ry(pi) = [[0, -1], [1, 0]] (purely real), still unitary
    expect = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.complex64)
    assert torch.allclose(yp, expect, atol=1e-6, rtol=0)
    assert is_unitary(yp)


def test_su2_from_euler_batched_unitary_and_broadcast():
    torch.manual_seed(0)
    alpha = torch.randn(5)
    beta = torch.randn(1)  # broadcast to 5
    gamma = torch.randn(1, 1)  # broadcast to 5
    c = su2_from_euler(alpha, beta, gamma)  # shape (5, 2, 2)
    assert c.shape == (5, 2, 2)
    assert is_unitary(c)


@pytest.mark.parametrize(
    "dtype_real, dtype_complex",
    [
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
    ],
)
def test_dtype_mapping(dtype_real, dtype_complex):
    theta = torch.zeros(3, dtype=dtype_real)
    r = rz(theta)  # (..., 2, 2)
    y = ry(theta)
    assert r.dtype == dtype_complex
    assert y.dtype == dtype_complex


def test_coins_su2_from_theta_shape_unitary_and_grad():
    torch.manual_seed(123)
    v = 7
    # Random Euler angles; require grad to test backprop
    theta = torch.randn(v, 3, dtype=torch.float32, requires_grad=True)
    c = coins_su2_from_theta(theta)  # (v, 2, 2) complex
    assert c.shape == (v, 2, 2)
    assert c.is_complex()
    assert is_unitary(c)

    # Simple scalar loss to drive gradients (sum of real parts)
    loss = c.real.sum()
    loss.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()


def test_invalid_theta_raises():
    bad = torch.zeros(4, 2)  # wrong last dim
    with pytest.raises(ValueError):
        _ = coins_su2_from_theta(bad)

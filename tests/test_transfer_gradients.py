import math
from pathlib import Path

import torch


def test_no_clamp_min_in_transfer_loss():
    """
    Prevent reintroducing the dead-gradient bug:
    clamp_min(eps) before log() makes gradients zero when p < eps.
    """
    p = Path("acpl/objectives/transfer.py").read_text()
    assert ".clamp_min(" not in p or "transfer" not in p.lower()


def test_log_eps_has_gradient_at_zero():
    """
    The core issue: if p=0, loss = -log(p+eps) must still have nonzero grad wrt p.
    """
    eps = 1e-8
    p = torch.tensor([0.0], requires_grad=True)
    loss = -(p + eps).log().mean()
    loss.backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert (p.grad.abs() > 0).all()


def test_clamp_min_would_have_zero_gradient():
    """
    This test documents the failure mode so it’s obvious to future-you.
    """
    eps = 1e-8
    p = torch.tensor([0.0], requires_grad=True)
    loss = -(p.clamp_min(eps)).log().mean()
    loss.backward()
    assert p.grad is not None
    assert (p.grad.abs() == 0).all()

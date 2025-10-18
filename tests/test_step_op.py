# tests/test_step_op.py
import pytest
import torch

from acpl.sim.portmap import build_portmap
from acpl.sim.shift import build_shift
from acpl.sim.step import step
from acpl.sim.coins import coins_su2_from_theta, CoinsSpecSU2
from acpl.sim.utils import position_probabilities, renorm_state_


def _cycle_pairs(n: int):
    """Undirected n-cycle as list of (u,v) pairs."""
    return [(i, (i + 1) % n) for i in range(n)]


def _identity_su2_stack(num_nodes: int, dtype=torch.complex64) -> torch.Tensor:
    I2 = torch.eye(2, dtype=dtype)
    return I2.unsqueeze(0).repeat(num_nodes, 1, 1)


def test_step_local_norm_check_passes_on_unitary_coins():
    # 4-cycle: deg==2 everywhere, so SU(2) applies at all nodes.
    pairs = _cycle_pairs(4)
    pm = build_portmap(pairs, num_nodes=4, coalesce=False)
    S = build_shift(pm)

    A = pm.src.numel()
    torch.manual_seed(0)
    psi = (torch.randn(A) + 1j * torch.randn(A)).to(torch.complex64)
    renorm_state_(psi)

    angles = torch.randn(pm.num_nodes, 3)
    coins = coins_su2_from_theta(angles, spec=CoinsSpecSU2(normalize=True, check=True))

    # Should not raise when check_local_norm=True
    out = step(psi, pm, coins, shift=S, check_local_norm=True)
    assert out.shape == psi.shape


def test_step_local_norm_check_detects_nonunitary():
    # Break unitarity slightly at a single node and ensure the checker trips.
    pairs = _cycle_pairs(4)
    pm = build_portmap(pairs, num_nodes=4, coalesce=False)
    S = build_shift(pm)

    A = pm.src.numel()
    torch.manual_seed(1)
    psi = (torch.randn(A) + 1j * torch.randn(A)).to(torch.complex64)
    renorm_state_(psi)

    angles = torch.randn(pm.num_nodes, 3)
    coins = coins_su2_from_theta(angles, spec=CoinsSpecSU2(normalize=True, check=True)).clone()

    # Deliberately make one block slightly non-unitary
    coins[0] = 1.02 * coins[0]

    with pytest.raises(AssertionError):
        step(psi, pm, coins, shift=S, check_local_norm=True)


def test_step_batched_matches_per_sample():
    # 5-cycle to vary sizes
    pairs = _cycle_pairs(5)
    pm = build_portmap(pairs, num_nodes=5, coalesce=False)
    S = build_shift(pm)

    A = pm.src.numel()
    torch.manual_seed(2)
    B = 3
    psi_b = (torch.randn(B, A) + 1j * torch.randn(B, A)).to(torch.complex64)
    renorm_state_(psi_b)

    angles = torch.randn(pm.num_nodes, 3)
    coins = coins_su2_from_theta(angles, spec=CoinsSpecSU2(normalize=True, check=True))

    # Batched step
    out_b = step(psi_b, pm, coins, shift=S, check_local_norm=True)

    # Per-sample step, then stack
    outs = []
    for b in range(B):
        outs.append(step(psi_b[b], pm, coins, shift=S, check_local_norm=True))
    out_stack = torch.stack(outs, dim=0)

    assert torch.allclose(out_b, out_stack, atol=1e-6)


def test_identity_coins_reduce_to_pure_shift():
    # 6-cycle; identity coins → step equals index_select by perm
    pairs = _cycle_pairs(6)
    pm = build_portmap(pairs, num_nodes=6, coalesce=False)
    S = build_shift(pm)

    A = pm.src.numel()
    torch.manual_seed(3)
    psi = (torch.randn(A) + 1j * torch.randn(A)).to(torch.complex64)
    renorm_state_(psi)

    coins = _identity_su2_stack(pm.num_nodes, dtype=torch.complex64)

    out = step(psi, pm, coins, shift=S, check_local_norm=True)
    assert torch.allclose(out, psi.index_select(0, S.perm), atol=1e-7)


def test_gradient_flows_through_coins_with_position_loss():
    # 4-cycle; define a target node and drive its probability with a simple loss.
    pairs = _cycle_pairs(4)
    pm = build_portmap(pairs, num_nodes=4, coalesce=False)
    S = build_shift(pm)

    A = pm.src.numel()
    torch.manual_seed(4)
    psi = (torch.randn(A) + 1j * torch.randn(A)).to(torch.complex64)
    renorm_state_(psi)

    # Angles are learnable
    angles = torch.randn(pm.num_nodes, 3, requires_grad=True)
    coins = coins_su2_from_theta(angles, spec=CoinsSpecSU2(normalize=False, check=False))

    # One step, compute position probabilities, take negative prob at node j*
    j_star = 2
    psi_next = step(psi, pm, coins, shift=S, check_local_norm=False)
    P = position_probabilities(psi_next, pm)  # (N,)
    loss = -P[j_star]
    loss.backward()

    assert angles.grad is not None
    assert torch.isfinite(angles.grad).all()
    # Some gradient signal should reach the controller parameters (angles)
    assert angles.grad.abs().sum().item() > 0.0

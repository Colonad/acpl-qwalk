# tests/test_step_unitarity.py
import torch
import pytest

from acpl.sim.portmap import build_portmap
from acpl.sim.shift import build_shift
from acpl.sim.step import step
from acpl.sim.coins import coins_su2_from_theta, CoinsSpecSU2
from acpl.sim.utils import state_norm2, renorm_state_

@pytest.mark.parametrize("n_cycle", [3, 4, 5, 8])
def test_step_preserves_global_norm(n_cycle):
    # n-cycle → deg==2 everywhere (Phase A coins apply at all nodes)
    pairs = [(i, (i + 1) % n_cycle) for i in range(n_cycle)]
    pm = build_portmap(pairs, num_nodes=n_cycle, coalesce=False)
    S = build_shift(pm)

    A = pm.src.numel()
    torch.manual_seed(123 + n_cycle)
    psi = (torch.randn(A) + 1j * torch.randn(A)).to(torch.complex64)
    renorm_state_(psi)

    # Random SU(2) coins (unitary by construction)
    angles = torch.randn(pm.num_nodes, 3)
    coins = coins_su2_from_theta(angles, spec=CoinsSpecSU2(normalize=True, check=True))

    n0 = state_norm2(psi)
    psi_next = step(psi, pm, coins, shift=S, check_local_norm=True)
    n1 = state_norm2(psi_next)

    assert torch.allclose(n0, n1, atol=1e-6)

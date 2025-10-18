# tests/test_step_gradcheck.py
import pytest
import torch

from acpl.sim.portmap import build_portmap
from acpl.sim.shift import build_shift
from acpl.sim.step import step
from acpl.sim.coins import coins_su2_from_theta, CoinsSpecSU2
from acpl.sim.utils import position_probabilities

@pytest.mark.slow
def test_gradcheck_step_angles():
    # Small 3-cycle to keep gradcheck cheap
    pairs = [(0,1), (1,2), (2,0)]
    pm = build_portmap(pairs, num_nodes=3, coalesce=False)
    S = build_shift(pm)

    A = pm.src.numel()

    # Use double precision for gradcheck
    torch.manual_seed(77)
    psi = (torch.randn(A, dtype=torch.float64) + 1j * torch.randn(A, dtype=torch.float64)).to(torch.complex128)
    psi = (psi / torch.linalg.norm(psi))  # normalize to avoid scaling issues

    # Angles are the parameters; use double precision
    angles = torch.randn(pm.num_nodes, 3, dtype=torch.float64, requires_grad=True)

    def fn(angles_):
        coins = coins_su2_from_theta(angles_.to(torch.float64), spec=CoinsSpecSU2(dtype=torch.complex128, normalize=False, check=False))
        psi_next = step(psi, pm, coins, shift=S, check_local_norm=False)
        # Smooth scalar loss (sum of squared position probabilities)
        P = position_probabilities(psi_next, pm).to(torch.float64)  # (N,) real
        return (P * P).sum()

    # Wrap for gradcheck: returns tensor; inputs tuple
    def wrapped(angles_):
        return fn(angles_)

    assert torch.autograd.gradcheck(wrapped, (angles,), eps=1e-6, atol=1e-4, rtol=1e-3)

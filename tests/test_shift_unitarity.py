import numpy as np
import pytest
import torch

from acpl.sim.portmap import make_flipflop_portmap
from acpl.sim.shift import (
    apply_shift_torch,
    build_shift_indices,
    build_shift_matrix_torch,
    verify_shift_unitarity_torch,
)


@pytest.mark.parametrize(
    "mode,edges,num_nodes",
    [
        # Undirected simple line: 0-1-2 (E=2 → A=4)
        ("undirected", np.array([[0, 1], [1, 2]]).T, 3),
        # Undirected with a self-loop at 0 and parallel 1-2 twice
        # (keep_multi=True will be tested separately)
        ("undirected", np.array([[0, 0], [0, 0], [1, 2], [2, 1]]).T, 3),
        # Directed with reverse edges present (pairs neatly)
        ("directed", np.array([[0, 1], [1, 0], [1, 2], [2, 1]]).T, 3),
        # Directed with an unmatched arc (1->2 without 2->1): should self-pair
        ("directed", np.array([[0, 1], [1, 0], [1, 2]]).T, 3),
    ],
)
def test_shift_is_involution_and_unitary(mode, edges, num_nodes):
    pm = make_flipflop_portmap(
        edges, num_nodes=num_nodes, mode=mode, allow_self_loops=True, keep_multi=True
    )
    ok, msg = verify_shift_unitarity_torch(pm)
    assert ok, msg

    # Check involution property directly
    rev = torch.as_tensor(pm.rev, dtype=torch.long)
    assert torch.equal(rev[rev], torch.arange(pm.num_arcs, dtype=torch.long))

    # Index vs sparse application agree
    # NOTE: stepper will be added when step.py is implemented
    shift_idx = build_shift_indices(pm, backend="torch", device=torch.device("cpu"))

    A = pm.num_arcs
    if A == 0:
        return

    psi = torch.randn(A, dtype=torch.complex64)
    psi = psi + 1j * torch.randn_like(psi)

    psi_idx = apply_shift_torch(psi, shift_idx)
    S = build_shift_matrix_torch(
        pm, device=torch.device("cpu"), dtype=torch.complex64, layout="coo"
    )
    psi_sp = torch.sparse.mm(S, psi.view(-1, 1)).view(-1)
    assert torch.allclose(psi_idx, psi_sp, atol=1e-6, rtol=1e-6)


def test_keep_multi_undirected_parallel_edges():
    # Parallel edges between (0,1) kept distinct:
    # arcs should be 4 per undirected pair (2 pairs -> 4 arcs)
    edges = np.array([[0, 1], [1, 0], [0, 1], [1, 0]]).T
    pm = make_flipflop_portmap(edges, num_nodes=2, mode="undirected", keep_multi=True)
    assert pm.num_edges == 4  # four undirected edges kept
    assert pm.num_arcs == 8
    ok, msg = verify_shift_unitarity_torch(pm)
    assert ok, msg

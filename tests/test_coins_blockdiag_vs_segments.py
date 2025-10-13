import numpy as np
import pytest
import torch

from acpl.sim.coins import (
    coins_from_theta,
    group_vertices_by_degree,
)
from acpl.sim.portmap import make_flipflop_portmap
from acpl.sim.step import build_stepper


def _randn(*shape, dtype=torch.float32, device="cpu"):
    return torch.randn(*shape, dtype=dtype, device=device)


def make_mixed_graph():
    """
    Build a small mixed-degree undirected graph:
      - path 0-1-2-3 (deg2 for nodes 1 and 2)
      - star centered at 4 with leaves {5,6,7,8} (center deg4, leaves deg1)
      - self-loop at 9 (deg2: two arcs (9->9) paired)
    """
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),  # path
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),  # star
        (9, 9),  # loop
    ]
    return np.array(edges, dtype=np.int64)


@pytest.mark.parametrize("lift", ["exp", "cayley"])
def test_sparse_blockdiag_matches_segment_application(lift):
    device = torch.device("cpu")
    edges = make_mixed_graph().T
    pm = make_flipflop_portmap(
        edges, num_nodes=10, mode="undirected", allow_self_loops=True, keep_multi=False
    )
    stepper = build_stepper(pm, device=device)

    layout = stepper.layout
    groups = group_vertices_by_degree(layout)
    A = layout.num_arcs

    # Degrees present: 1,2,4,2(loop node 9 contributes degree 2)
    # Provide SU(2) for all V (only used for deg==2)
    V = layout.num_nodes
    theta = _randn(V, 3, device=device)

    # Provide params for d>2
    params_by_deg = {}
    for d, idx in groups.items():
        d = int(d)
        if d <= 2:
            continue
        Nd = idx.numel()
        params_by_deg[d] = _randn(Nd, 2, d, d, device=device)

    # Build coin stacks
    U_by_deg = coins_from_theta(
        layout, theta_su2=theta, lift=lift, params_by_deg=params_by_deg, dtype=torch.complex64
    )

    # Build sparse block-diagonal C (debug) and compare with segmented application via Stepper.apply_coins
    C = stepper.build_C_sparse(U_by_deg, layout_kind="coo", dtype=torch.complex64).coalesce()
    # random psi
    psi = (
        _randn(A, dtype=torch.float32, device=device)
        + 1j * _randn(A, dtype=torch.float32, device=device)
    ).to(torch.complex64)

    psi_seg = stepper.apply_coins(psi, U_by_deg)
    psi_sp = torch.sparse.mm(C, psi.view(-1, 1)).view(-1)
    assert torch.allclose(psi_seg, psi_sp, atol=1e-6, rtol=1e-6)

    # Now full step with shift: compare sparse S * C vs. stepper.step
    S = stepper.build_S_sparse(layout="coo", dtype=torch.complex64).coalesce()
    psi_step = stepper.step(psi, U_by_deg)
    psi_mat = torch.sparse.mm(S, psi_sp.view(-1, 1)).view(-1)
    assert torch.allclose(psi_step, psi_mat, atol=1e-6, rtol=1e-6)


def test_degree1_identity_default():
    # Star graph center 0 connected to (1..4); deg(0)=4, leaves deg(1)=1
    edges = np.array([(0, 1), (0, 2), (0, 3), (0, 4)], dtype=np.int64).T
    pm = make_flipflop_portmap(edges, num_nodes=5, mode="undirected")
    stepper = build_stepper(pm, device=torch.device("cpu"))

    A = pm.num_arcs
    psi = torch.randn(A, dtype=torch.complex64)
    # Provide only d=4 coins, omit d=1; expect leaves pass-through
    theta_dummy = torch.zeros(pm.num_nodes, 3)  # unused
    layout = stepper.layout
    groups = {
        int(d): idx
        for d, idx in [(int(d), layout.indices_degree(int(d))) for d in layout.unique_degrees()]
    }
    params_by_deg = {}
    if 4 in groups:
        Nd = groups[4].numel()
        params_by_deg[4] = torch.randn(Nd, 2, 4, 4)

    U_by_deg = coins_from_theta(
        layout,
        theta_su2=theta_dummy,
        lift="exp",
        params_by_deg=params_by_deg,
        dtype=torch.complex64,
    )
    psi_out = stepper.apply_coins(
        psi, U_by_deg, default_identity_for_deg1=True, default_identity_for_missing=True
    )
    # Just ensure it runs and preserves norm
    assert torch.allclose(psi_out.abs().norm(), psi.abs().norm(), atol=1e-6)

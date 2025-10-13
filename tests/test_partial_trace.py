import numpy as np
import pytest
import torch

from acpl.sim.coins import build_coin_layout
from acpl.sim.portmap import make_flipflop_portmap
from acpl.sim.utils import (
    density_from_state,
    is_normalized,
    normalize_state,
    partial_trace_coin_from_density,
    vertex_marginals_from_state,
)


def _randn_complex(shape, device="cpu", dtype=torch.complex64):
    x = torch.randn(*shape, dtype=torch.float32, device=device)
    y = torch.randn(*shape, dtype=torch.float32, device=device)
    return (x + 1j * y).to(dtype)


def _manual_vertex_marginals(layout, psi: torch.Tensor) -> torch.Tensor:
    """Reference implementation: sum |psi|^2 over each vertex arc slice."""
    prob = psi.abs().pow(2)
    V = layout.num_nodes
    out = torch.zeros(*psi.shape[:-1], V, dtype=prob.dtype, device=prob.device)
    for v in range(V):
        s = int(layout.arc_start[v].item())
        e = int(layout.arc_end[v].item())
        if e > s:
            out[..., v] = prob[..., s:e].sum(dim=-1)
    return out


@pytest.mark.parametrize(
    "edges,num_nodes",
    [
        # simple undirected line 0-1-2 (E=2 → A=4, degrees: [1,2,1])
        (np.array([(0, 1), (1, 2)], dtype=np.int64), 3),
        # undirected with a loop at 0 (two arcs on node 0), so degrees: [3,1] on a 2-node graph
        (np.array([(0, 1), (0, 0)], dtype=np.int64), 2),
    ],
)
def test_vertex_marginals_matches_manual(edges, num_nodes):
    pm = make_flipflop_portmap(
        edges.T, num_nodes=num_nodes, mode="undirected", allow_self_loops=True
    )
    layout = build_coin_layout(pm, device=torch.device("cpu"))

    A = pm.num_arcs
    psi = _randn_complex((A,))
    psi = normalize_state(psi)
    assert is_normalized(psi)

    P_func = vertex_marginals_from_state(layout, psi)
    P_ref = _manual_vertex_marginals(layout, psi)

    assert torch.allclose(P_func, P_ref, atol=1e-7, rtol=1e-6)
    # distribution sums to 1 (up to fp error)
    assert torch.allclose(P_func.sum(), torch.tensor(1.0, dtype=P_func.dtype), atol=1e-6)


def test_vertex_marginals_batched():
    # star graph: center 0 connected to 1..4 → degrees: [4,1,1,1,1], A=8
    edges = np.array([(0, 1), (0, 2), (0, 3), (0, 4)], dtype=np.int64)
    pm = make_flipflop_portmap(edges.T, num_nodes=5, mode="undirected")
    layout = build_coin_layout(pm, device=torch.device("cpu"))

    B = 3
    A = pm.num_arcs
    psi = _randn_complex((B, A))
    psi = normalize_state(psi)
    assert is_normalized(psi)

    P_func = vertex_marginals_from_state(layout, psi)  # (B, V)
    P_ref = _manual_vertex_marginals(layout, psi)

    assert P_func.shape == P_ref.shape == (B, layout.num_nodes)
    assert torch.allclose(P_func, P_ref, atol=1e-7, rtol=1e-6)
    # each batch sums to 1
    ones = torch.ones(B, dtype=P_func.dtype)
    assert torch.allclose(P_func.sum(dim=-1), ones, atol=1e-6)


def test_partial_trace_coin_matches_marginals_diagonal():
    # small mixed graph: path 0-1-2 and self-loop at 2
    edges = np.array([(0, 1), (1, 2), (2, 2)], dtype=np.int64)
    pm = make_flipflop_portmap(edges.T, num_nodes=3, mode="undirected", allow_self_loops=True)
    layout = build_coin_layout(pm, device=torch.device("cpu"))

    A = pm.num_arcs
    psi = _randn_complex((A,))
    psi = normalize_state(psi)

    # vertex marginals from state
    P = vertex_marginals_from_state(layout, psi)  # (V,)

    # build density and partial trace over coin
    rho = density_from_state(psi)  # (A, A)
    sigma_v = partial_trace_coin_from_density(layout, rho)  # (V, V) diagonal

    # diagonal of sigma_v should equal P (as real values), off-diagonals ~ 0
    diag = sigma_v.diagonal(dim1=-2, dim2=-1).real
    offdiag_max = (sigma_v - torch.diag_embed(diag)).abs().max()

    assert torch.allclose(diag, P, atol=1e-6)
    assert offdiag_max.item() < 1e-7

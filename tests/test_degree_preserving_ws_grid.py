# tests/test_degree_preserving_ws_grid.py
import torch
import pytest

from acpl.data.graphs import (
    watts_strogatz_grid_graph_degree_preserving,
    watts_strogatz_grid_graph,
    build_arc_index,
)

def _baseline_degrees(Lx, Ly, kx, ky):
    # Compute baseline lattice degrees per node (clipped by borders)
    deg = torch.zeros(Lx * Ly, dtype=torch.long)
    idx = 0
    for y in range(Ly):
        for x in range(Lx):
            left  = min(kx, x)
            right = min(kx, Lx - 1 - x)
            down  = min(ky, y)
            up    = min(ky, Ly - 1 - y)
            deg[idx] = left + right + down + up
            idx += 1
    return deg

@pytest.mark.parametrize("Lx,Ly,kx,ky,beta", [
    (6, 5, 1, 1, 0.0),
    (6, 5, 1, 1, 0.3),
    (8, 8, 2, 1, 0.5),
])
def test_degree_preserving_ws_grid_preserves_degrees(Lx, Ly, kx, ky, beta):
    e, d, c, s = watts_strogatz_grid_graph_degree_preserving(
        Lx=Lx, Ly=Ly, kx=kx, ky=ky, beta=beta, seed=123
    )
    # Same average degree as baseline AND per-node degrees match baseline exactly
    base_d = _baseline_degrees(Lx, Ly, kx, ky)
    assert torch.equal(d, base_d)
    assert torch.allclose(d.float().mean(), base_d.float().mean(), atol=1e-6)

    # CSR sanity
    A = e.shape[1]
    assert s.shape == (Lx * Ly + 1,)
    assert e.shape[0] == 2 and e.shape[1] == A
    assert s[-1] == A
    # consistency: slice lengths = degrees
    slice_lens = s[1:] - s[:-1]
    assert torch.equal(slice_lens, d)

def test_degree_preserving_ws_grid_is_simple_and_deterministic():
    Lx, Ly, kx, ky, beta = 7, 6, 1, 1, 0.4
    e1, d1, c1, s1 = watts_strogatz_grid_graph_degree_preserving(
        Lx=Lx, Ly=Ly, kx=kx, ky=ky, beta=beta, seed=777
    )
    e2, d2, c2, s2 = watts_strogatz_grid_graph_degree_preserving(
        Lx=Lx, Ly=Ly, kx=kx, ky=ky, beta=beta, seed=777
    )
    assert torch.equal(e1, e2)
    assert torch.equal(d1, d2)
    assert torch.allclose(c1, c2)
    assert torch.equal(s1, s2)

    # Check there are no self-loops in oriented arcs, and no duplicates in undirected backbone
    src, dst = e1[0], e1[1]
    assert (src != dst).all()
    # Rebuild undirected set and ensure uniqueness
    undirected = set()
    for u, v in zip(src.tolist(), dst.tolist()):
        a, b = (u, v) if u < v else (v, u)
        undirected.add((a, b))
    assert len(undirected) * 2 == e1.shape[1]  # arcs are doubled

def test_degree_preserving_ws_grid_matches_lattice_when_beta_zero():
    Lx, Ly, kx, ky = 6, 6, 1, 2
    e_dp, d_dp, c_dp, s_dp = watts_strogatz_grid_graph_degree_preserving(
        Lx=Lx, Ly=Ly, kx=kx, ky=ky, beta=0.0, seed=1
    )
    # Compare to base lattice made by the non-degree-preserving builder with beta=0.0
    from acpl.data.graphs import watts_strogatz_grid_graph
    e0, d0, c0, s0 = watts_strogatz_grid_graph(Lx=Lx, Ly=Ly, kx=kx, ky=ky, beta=0.0, seed=999)

    assert torch.equal(d_dp, d0)
    assert torch.allclose(c_dp, c0)
    # Edges may be in different (src,dst) arc order due to deterministic sort, but should match exactly
    assert torch.equal(e_dp, e0)
    assert torch.equal(s_dp, s0)

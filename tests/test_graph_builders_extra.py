# tests/test_graph_builders_extra.py
import pytest
import torch

from acpl.data.graphs import (
    random_geometric_graph,
    watts_strogatz_grid_graph,
)


def _check_arc_csr(edge_index: torch.Tensor, arc_slices: torch.Tensor, num_nodes: int):
    # edge_index: (2, A)
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    A = edge_index.shape[1]
    src, dst = edge_index[0], edge_index[1]
    assert src.shape == (A,) and dst.shape == (A,)
    assert src.dtype == torch.long and dst.dtype == torch.long
    assert (src >= 0).all() and (src < num_nodes).all()
    assert (dst >= 0).all() and (dst < num_nodes).all()

    # CSR pointers
    assert arc_slices.shape == (num_nodes + 1,)
    assert arc_slices.dtype == torch.long
    assert arc_slices[0] == 0
    assert arc_slices[-1] == A
    assert (arc_slices[1:] >= arc_slices[:-1]).all()

    # Check arcs grouped by source (slice-by-slice)
    for u in range(num_nodes):
        s, e = int(arc_slices[u]), int(arc_slices[u + 1])
        if s == e:
            continue
        # all src equal u, dst non-decreasing (determinism)
        assert (src[s:e] == u).all()
        if e - s > 1:
            assert torch.all(dst[s : e - 1] <= dst[s + 1 : e])


def _check_degree_consistency(
    edge_index: torch.Tensor, degrees: torch.Tensor, arc_slices: torch.Tensor
):
    # degrees[u] must equal number of outgoing arcs from u in undirected-to-arcs convention
    # because for every undirected edge (u,v) we added arcs u->v and v->u.
    src = edge_index[0]
    num_nodes = degrees.numel()
    # out-degree in the oriented-arc list equals degrees[u]
    out_counts = torch.bincount(src, minlength=num_nodes)
    assert torch.equal(out_counts, degrees)

    # Also, CSR slice lengths should match degrees
    slice_lens = arc_slices[1:] - arc_slices[:-1]
    assert torch.equal(slice_lens, degrees)


# ------------------------------------------------------------------------------------
# Watts–Strogatz on a 2D lattice with fixed coordinates
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Lx,Ly,kx,ky,beta",
    [
        (6, 5, 1, 1, 0.0),  # pure lattice
        (6, 5, 1, 1, 0.3),  # some rewiring
        (8, 8, 2, 1, 0.5),  # asymmetric neighborhoods
    ],
)
def test_ws_grid_shapes_and_invariants(Lx, Ly, kx, ky, beta):
    edge_index, degrees, coords, arc_slices = watts_strogatz_grid_graph(
        Lx=Lx, Ly=Ly, kx=kx, ky=ky, beta=beta, seed=123
    )
    N = Lx * Ly
    # shapes
    assert degrees.shape == (N,)
    assert coords.shape == (N, 2)
    assert arc_slices.shape == (N + 1,)
    assert edge_index.shape[0] == 2
    # coords are normalized grid in [0,1]
    assert (coords >= 0).all() and (coords <= 1).all()

    _check_arc_csr(edge_index, arc_slices, N)
    _check_degree_consistency(edge_index, degrees, arc_slices)

    # --- Lattice baseline (no rewiring) degrees per node ---
    def base_deg_xy(x: int, y: int) -> int:
        # neighbors up to kx on each x side and ky on each y side, clipped by borders
        left = min(kx, x)
        right = min(kx, Lx - 1 - x)
        down = min(ky, y)
        up = min(ky, Ly - 1 - y)
        return left + right + down + up

    base_degs = torch.zeros(N, dtype=torch.long)
    idx = 0
    for y in range(Ly):
        for x in range(Lx):
            base_degs[idx] = base_deg_xy(x, y)
            idx += 1

    base_mean = base_degs.float().mean()
    deg_mean = degrees.float().mean()

    # In WS rewiring, total edges are preserved → average degree preserved
    assert torch.allclose(deg_mean, base_mean, atol=1e-6)

    # Degrees can vary per node after rewiring; allow slack tied to neighborhood size.
    # Soft bounds: [base_min - m*nb, base_max + m*nb] with m=3 and nb=max(kx,ky)
    m = 3
    nb = max(kx, ky)
    base_min = int(base_degs.min())
    base_max = int(base_degs.max())
    lo = max(0, base_min - m * nb)
    hi = base_max + m * nb

    assert (degrees >= lo).all()
    assert (degrees <= hi).all()


def test_ws_grid_deterministic_with_seed():
    args = dict(Lx=5, Ly=4, kx=1, ky=1, beta=0.35)
    e1, d1, c1, s1 = watts_strogatz_grid_graph(**args, seed=999)
    e2, d2, c2, s2 = watts_strogatz_grid_graph(**args, seed=999)
    assert torch.equal(e1, e2)
    assert torch.equal(d1, d2)
    assert torch.allclose(c1, c2)
    assert torch.equal(s1, s2)


# ------------------------------------------------------------------------------------
# Random geometric graphs (RGG)
# ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,dim,torus",
    [
        (1, 2, False),
        (10, 2, False),
        (10, 2, True),
        (12, 3, False),
    ],
)
def test_rgg_shapes_and_basic_invariants(N, dim, torus):
    edge_index, degrees, coords, arc_slices = random_geometric_graph(
        N=N, radius=0.5, dim=dim, torus=torus, seed=777
    )
    # shapes
    assert degrees.shape == (N,)
    assert coords.shape == (N, dim)
    assert arc_slices.shape == (N + 1,)
    assert edge_index.shape[0] == 2

    _check_arc_csr(edge_index, arc_slices, N)
    _check_degree_consistency(edge_index, degrees, arc_slices)

    # coords in [0,1]
    if N > 0:
        assert (coords >= 0).all() and (coords <= 1).all()


def test_rgg_deterministic_with_seed_and_monotone_radius():
    N, dim = 20, 2
    # larger radius should produce a supergraph (at least not fewer edges)
    e_small, d_small, c_small, s_small = random_geometric_graph(N, radius=0.12, dim=dim, seed=2024)
    e_big, d_big, c_big, s_big = random_geometric_graph(N, radius=0.25, dim=dim, seed=2024)

    # Deterministic coords under seed
    assert torch.allclose(c_small, c_big)

    # Edge count non-decreasing in radius
    A_small = e_small.shape[1]
    A_big = e_big.shape[1]
    assert A_big >= A_small

    # Basic invariants still hold
    _check_arc_csr(e_small, s_small, N)
    _check_arc_csr(e_big, s_big, N)
    _check_degree_consistency(e_small, d_small, s_small)
    _check_degree_consistency(e_big, d_big, s_big)

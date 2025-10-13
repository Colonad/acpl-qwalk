# tests/test_portmap.py
import numpy as np

from acpl.sim.portmap import (
    degrees_from_portmap,
    make_flipflop_portmap,
    to_edge_index,
)

try:
    import torch  # optional
except Exception:  # pragma: no cover
    torch = None


def path_edges(n: int):
    """Return undirected path edges for nodes 0..n-1 as a (2, E) numpy array."""
    u = np.arange(n - 1, dtype=np.int64)
    v = u + 1
    return np.stack([u, v], axis=0)


def test_path_three_nodes_basic():
    # Path 0-1-2
    ei = path_edges(3)  # shape (2, 2): edges (0,1) and (1,2)
    pm = make_flipflop_portmap(ei)

    # Sizes
    assert pm.num_nodes == 3
    assert pm.num_edges == 2
    assert pm.num_arcs == 4

    # Oriented arcs: interleaved (u->v), (v->u) per undirected edge, lexicographic by (u, v)
    # Expect: 0: 0->1, 1: 1->0, 2: 1->2, 3: 2->1
    assert np.array_equal(pm.tail, np.array([0, 1, 1, 2]))
    assert np.array_equal(pm.head, np.array([1, 0, 2, 1]))

    # Reverse pairing is an involution and flips endpoints
    idx = np.arange(pm.num_arcs, dtype=np.int64)
    assert np.array_equal(pm.rev[pm.rev], idx)
    assert np.array_equal(pm.tail[pm.rev], pm.head)
    assert np.array_equal(pm.head[pm.rev], pm.tail)

    # CSR node -> arcs (stable)
    # degrees: [1, 2, 1]; pointers: [0, 1, 3, 4]
    assert np.array_equal(degrees_from_portmap(pm), np.array([1, 2, 1]))
    assert np.array_equal(pm.node_ptr, np.array([0, 1, 3, 4]))
    # arcs of node 1 should be the two arcs with tail==1, in encounter order
    assert np.array_equal(pm.arcs_of(1), np.array([1, 2]))

    # Round-trip to canonical edge_index (even arcs correspond to canonical undirected edges)
    assert np.array_equal(to_edge_index(pm), ei)


def test_coalesces_duplicates_and_self_loops_and_sorts():
    # Intentionally messy input:
    #   duplicates (0,1) & (1,0), repeated (1,2), and self-loops (0,0), (2,2)
    messy = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [1, 2],
            [0, 0],
            [2, 2],
        ],
        dtype=np.int64,
    )
    pm = make_flipflop_portmap(messy)  # accepts (E,2) or (2,E)

    # Only two undirected edges remain: (0,1) and (1,2)
    assert pm.num_edges == 2
    # Sorted lexicographically, so arcs match the same layout as the clean path case
    assert np.array_equal(pm.tail, np.array([0, 1, 1, 2]))
    assert np.array_equal(pm.head, np.array([1, 0, 2, 1]))


def test_accepts_various_input_formats_and_is_deterministic():
    # Reference from (2, E)
    ref_ei = path_edges(4)  # edges: (0,1),(1,2),(2,3)
    pm_ref = make_flipflop_portmap(ref_ei)

    # (E, 2) format
    pm_e2 = make_flipflop_portmap(ref_ei.T)
    assert np.array_equal(pm_ref.tail, pm_e2.tail)
    assert np.array_equal(pm_ref.head, pm_e2.head)
    assert np.array_equal(pm_ref.rev, pm_e2.rev)
    assert np.array_equal(pm_ref.node_ptr, pm_e2.node_ptr)
    assert np.array_equal(pm_ref.node_arcs, pm_e2.node_arcs)

    # List of pairs (shuffled order, with duplicates)
    pairs = [(2, 3), (1, 2), (0, 1), (1, 2)]
    pm_list = make_flipflop_portmap(pairs)
    assert np.array_equal(pm_ref.tail, pm_list.tail)
    assert np.array_equal(pm_ref.head, pm_list.head)

    # Torch tensor (if available)
    if torch is not None:
        t = torch.tensor(ref_ei, dtype=torch.long)
        pm_t = make_flipflop_portmap(t)
        assert np.array_equal(pm_ref.tail, pm_t.tail)
        assert np.array_equal(pm_ref.head, pm_t.head)


def test_csr_consistency_and_counts_small_graph():
    # Small undirected graph: triangle + leaf
    # Edges: (0,1), (1,2), (0,2), (2,3)
    ei = np.array([[0, 1, 1, 0, 2, 2, 2, 3]]).reshape(2, 4)  # malformed on purpose
    # Fix to proper (2, E)
    ei = np.array([[0, 1, 0, 2], [1, 2, 2, 3]], dtype=np.int64)

    pm = make_flipflop_portmap(ei)

    # Degrees: d0=2 (to 1,2), d1=2 (to 0,2), d2=3 (to 0,1,3), d3=1 (to 2)
    deg = degrees_from_portmap(pm)
    assert np.array_equal(deg, np.array([2, 2, 3, 1]))

    # node_ptr must be a valid prefix-sum over degrees and end at num_arcs
    assert pm.node_ptr[0] == 0
    assert pm.node_ptr[-1] == pm.num_arcs
    assert np.array_equal(np.diff(pm.node_ptr), deg)

    # node_arcs must list exactly each arc once
    listed = np.sort(pm.node_arcs.copy())
    assert np.array_equal(listed, np.arange(pm.num_arcs, dtype=np.int64))

    # For every arc a, reverse arc swaps endpoints
    a = np.arange(pm.num_arcs, dtype=np.int64)
    assert np.array_equal(pm.tail[pm.rev[a]], pm.head[a])
    assert np.array_equal(pm.head[pm.rev[a]], pm.tail[a])

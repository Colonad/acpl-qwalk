# tests/test_shift_unitarity.py
import random

import pytest
import torch

from acpl.sim.portmap import build_portmap
from acpl.sim.shift import build_shift, check_shift_unitarity

# ----------------------
# Helpers
# ----------------------


def _edge_index_from_pairs(pairs: list[tuple[int, int]]):
    # We pass undirected (or directed) arc pairs directly to build_portmap.
    # The portmap builder understands multigraph edges and will expand to arcs.
    return pairs


def _random_complex_state(A: int, dtype=torch.complex64, device=None):
    re = torch.randn(A, dtype=torch.float32, device=device)
    im = torch.randn(A, dtype=torch.float32, device=device)
    psi = torch.complex(re, im).to(dtype)
    psi = psi / psi.norm().clamp_min(1e-12)
    return psi


def _unitary_from_qr(n: int, dtype=torch.complex64, device=None):
    # Random Haar-ish unitary via complex QR (sufficient for tests)
    X = torch.randn(n, n, dtype=torch.float32, device=device)
    Y = torch.randn(n, n, dtype=torch.float32, device=device)
    Z = torch.complex(X, Y).to(dtype)
    # torch.linalg.qr returns Q,R with R upper-triangular; fix signs on diag(R)
    Q, R = torch.linalg.qr(Z)
    ph = torch.diagonal(R) / torch.abs(torch.diagonal(R)).clamp_min(1e-12)
    Q = Q * ph.conj()
    return Q


def _permute_nodes(pairs: list[tuple[int, int]], pi: list[int]) -> list[tuple[int, int]]:
    # Relabel (u,v) -> (pi[u], pi[v])
    return [(pi[u], pi[v]) for (u, v) in pairs]


def _arc_perm_from_relabel(pm, pm_pi, pi: list[int]) -> torch.Tensor:
    """
    Build the arc permutation r : indices(pm) -> indices(pm_pi) induced by relabeling
    nodes by π and then rebuilding the portmap. For each arc i in pm with (v -> w),
    we map it to the next unused arc j in pm_pi with (π(v) -> π(w)).

    This matches multiplicities: the k-th occurrence of (v,w) in pm maps to the k-th
    occurrence of (π(v), π(w)) in pm_pi, respecting each CSR ordering.
    """
    src, dst = pm.src.tolist(), pm.dst.tolist()
    src2, dst2 = pm_pi.src.tolist(), pm_pi.dst.tolist()
    A = pm.node_ptr[-1].item()
    assert A == pm_pi.node_ptr[-1].item(), "arc counts must match after relabel"

    # Build lookup of arcs by directed pair (u,v) with occurrence order
    buckets = {}
    for j, (u, v) in enumerate(zip(src2, dst2, strict=False)):
        buckets.setdefault((u, v), []).append(j)

    # Track how many times we've used each (u,v)
    used = {k: 0 for k in buckets.keys()}

    r = torch.empty(A, dtype=torch.long)
    for i, (u, v) in enumerate(zip(src, dst, strict=False)):
        u2, v2 = pi[u], pi[v]
        key = (u2, v2)
        # If graph is undirected but encoded as bidirected arcs, the exact (u2,v2) must exist
        assert key in buckets, f"Relabeled arc ({u2}->{v2}) not present in pm_pi"
        k = used[key]
        assert k < len(buckets[key]), "Multiplicity mismatch while matching arcs"
        r[i] = buckets[key][k]
        used[key] += 1

    return r


def _inverse_perm(p: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), device=p.device, dtype=p.dtype)
    return inv


# ----------------------
# Parameterized core tests
# ----------------------


@pytest.mark.parametrize(
    "pairs,num_nodes,coalesce",
    [
        # Line graph 0-1-2-3 (endpoints degree 1)
        ([(0, 1), (1, 2), (2, 3)], 4, False),
        # 4-cycle (all degree 2)
        ([(0, 1), (1, 2), (2, 3), (3, 0)], 4, False),
        # Same 4-cycle but coalesced (duplicates would be merged if present)
        ([(0, 1), (1, 2), (2, 3), (3, 0)], 4, True),
        # Multigraph with repeated arcs
        ([(0, 1), (1, 0), (0, 1), (1, 2)], 3, False),
        # Include isolated node 3 (no edges touching 3)
        ([(0, 1), (1, 2)], 4, False),
    ],
)
def test_shift_unitarity_and_involution_dense_sparse(pairs, num_nodes, coalesce):
    pm = build_portmap(_edge_index_from_pairs(pairs), num_nodes=num_nodes, coalesce=coalesce)
    S = build_shift(pm)
    # Invariants: internal self-check + library-level check
    S.check(strict=True)
    check_shift_unitarity(S)

    # Dense/Sparse equivalence: S^H S == I
    A = S.A
    if A <= 2048:  # keep dense checks bounded
        S_sp = S.to_sparse()
        Id = torch.eye(A, dtype=torch.complex64)
        SdS = (S_sp.transpose(0, 1).conj() @ S_sp).to_dense()
        SSd = (S_sp @ S_sp.transpose(0, 1).conj()).to_dense()
        assert torch.allclose(SdS, Id)
        assert torch.allclose(SSd, Id)

    # Involution: S @ S == I  (flip-flop)
    perm = S.perm  # long tensor
    assert torch.equal(perm[perm], torch.arange(perm.numel(), device=perm.device))

    # Self-adjoint: permutation involution implies Hermitian (S == S^H)
    # Check via action on random complex vector
    psi = _random_complex_state(S.A)
    v1 = S.apply(psi)  # S psi
    v2 = S.apply(v1)  # S (S psi)
    assert torch.allclose(v2, psi, atol=1e-6, rtol=1e-6)

    # Norm preservation (unitarity): ||S psi|| = ||psi||
    assert torch.allclose(v1.norm(), psi.norm(), atol=1e-6, rtol=1e-6)


def test_csr_ordering_and_degree_tallies():
    # 4-cycle: verify CSR slices are monotone and ports are 0..deg(u)-1
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    pm = build_portmap(pairs, num_nodes=4, coalesce=False)
    src, dst, node_ptr, port, deg = pm.src, pm.dst, pm.node_ptr, pm.port, pm.deg

    N = pm.num_nodes
    for u in range(N):
        s, e = int(node_ptr[u]), int(node_ptr[u + 1])
        if e == s:
            continue
        # Monotone non-decreasing by destination within slice
        if (e - s) > 1:
            assert torch.all(dst[s : e - 1] <= dst[s + 1 : e])
        # Ports are exactly 0..deg(u)-1 and degree matches slice length
        assert torch.equal(port[s:e], torch.arange(e - s, device=port.device))
        assert (e - s) == int(deg[u])


def test_multigraph_pairing_is_deterministic():
    # Ensure involution holds deterministically with duplicated arcs
    pairs = [(0, 1), (1, 0), (0, 1), (0, 1), (1, 2), (2, 1)]
    pm = build_portmap(pairs, num_nodes=3, coalesce=False)
    perm = pm.to_shift_permutation()
    A = perm.numel()
    assert torch.equal(perm[perm], torch.arange(A, device=perm.device))


# ----------------------
# Covariance under node relabeling (novelty: theorem-driven test)
# ----------------------


@pytest.mark.parametrize("num_nodes,seed", [(6, 0), (8, 1), (12, 2)])
def test_shift_conjugates_under_node_permutation_er(num_nodes, seed):
    """
    Theoretical guarantee (report §12 & §15):
        If we relabel nodes by π and rebuild the shift S', then on the arc basis
        S' = R S R^T where R only reindexes node labels, preserving local port order.
    We verify this equivalence at the level of permutation vectors:
        perm(S') == r^{-1} ∘ perm(S) ∘ r
    where r is the arc permutation induced by π.
    """
    # Build a small ER-like random graph with possible multi-edges
    rng = random.Random(seed)
    p = 0.3
    pairs = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if rng.random() < p:
                pairs.append((u, v))
    # Guarantee connectivity-ish: fall back to a path if empty
    if not pairs:
        pairs = [(i, i + 1) for i in range(num_nodes - 1)]

    pm = build_portmap(pairs, num_nodes=num_nodes, coalesce=False)
    S = build_shift(pm)

    # Random permutation of nodes
    pi = list(range(num_nodes))
    rng.shuffle(pi)

    # Relabel edges and rebuild
    pairs_pi = _permute_nodes(pairs, pi)
    pm_pi = build_portmap(pairs_pi, num_nodes=num_nodes, coalesce=False)
    S_pi = build_shift(pm_pi)

    # Arc-basis permutation r induced by node permutation (ports preserved by index)
    # Arc-basis permutation r: indices(pm, old) -> indices(pm_pi, new) under relabeling
    r = _arc_perm_from_relabel(pm, pm_pi, pi)

    r_inv = _inverse_perm(r)

    s = S.perm  # perm vector in original arc basis
    s_pi = S_pi.perm  # perm vector in relabeled arc basis

    # Conjugation on permutation vectors (with r: old→new): s' = r ∘ s ∘ r^{-1}
    lhs = s_pi
    rhs = r[s[r_inv]]
    assert torch.equal(lhs, rhs)


# ----------------------
# Physics-facing checks (norms & step composition)
# ----------------------


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_shift_preserves_norm_for_random_state(dtype):
    pairs = [(0, 1), (1, 2), (2, 0), (2, 3)]  # small irregular
    pm = build_portmap(pairs, num_nodes=4, coalesce=False)
    S = build_shift(pm)

    psi = _random_complex_state(S.A, dtype=dtype)
    out = S.apply(psi)
    assert torch.allclose(out.norm(), psi.norm(), atol=1e-7, rtol=1e-7)


def test_coin_then_shift_step_preserves_norm_any_degree():
    """
    Assemble a block-diagonal 'coin' from random unitaries per vertex and
    check that one DTQW step (psi <- S @ blkdiag(C_v) @ psi) preserves norm.
    This covers the irregular-degree port model emphasized in the report.
    """
    # Irregular example with isolated vertex (degree 0 included)
    pairs = [(0, 1), (1, 2), (1, 3), (3, 4)]
    num_nodes = 6  # node 5 is isolated (deg=0)
    pm = build_portmap(pairs, num_nodes=num_nodes, coalesce=False)
    S = build_shift(pm)

    node_ptr = pm.node_ptr
    A = S.A
    psi = _random_complex_state(A, dtype=torch.complex64)

    # Build block-diagonal coin (dense) by scattering per-node unitaries
    C = torch.zeros((A, A), dtype=torch.complex64)
    for v in range(num_nodes):
        s, e = int(node_ptr[v]), int(node_ptr[v + 1])
        d = e - s
        if d == 0:
            continue  # no arcs at this vertex
        Uv = _unitary_from_qr(d, dtype=torch.complex64)
        C[s:e, s:e] = Uv

    # One step: psi <- S @ C @ psi
    S_sp = S.to_sparse()
    mid = C @ psi
    psi_next = S_sp @ mid

    # Norm preserved (unitary coin + permutation shift)
    assert torch.allclose(psi_next.norm(), psi.norm(), atol=1e-6, rtol=1e-6)


# ----------------------
# Stress / randomized shapes
# ----------------------


@pytest.mark.slow
@pytest.mark.parametrize("N,p,seed", [(20, 0.15, 123), (30, 0.10, 456), (40, 0.08, 789)])
def test_random_er_graphs_unitarity_and_involution(N, p, seed):
    rng = random.Random(seed)
    pairs = []
    for u in range(N):
        for v in range(u + 1, N):
            if rng.random() < p:
                # add single edge; with small prob add a duplicate to stress multigraph
                pairs.append((u, v))
                if rng.random() < 0.05:
                    pairs.append((u, v))
    if not pairs:
        pairs = [(i, i + 1) for i in range(N - 1)]

    for coalesce in (False, True):
        pm = build_portmap(pairs, num_nodes=N, coalesce=coalesce)
        S = build_shift(pm)
        S.check(strict=True)
        check_shift_unitarity(S)
        # Large A only: verify involution via permutation vector
        perm = S.perm
        assert torch.equal(perm[perm], torch.arange(perm.numel(), device=perm.device))


# ----------------------
# Device sanity (CPU-centric by default; CUDA if available)
# ----------------------


def test_shift_on_available_device():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    pm = build_portmap(pairs, num_nodes=4, coalesce=False)
    S = build_shift(pm)
    psi = _random_complex_state(S.A, device=device)
    out = S.apply(psi)
    assert out.device == device
    assert torch.allclose(out.norm(), psi.norm(), atol=1e-6, rtol=1e-6)

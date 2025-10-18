# tests/test_message_passing_equiv.py
# Permutation–equivariance for message passing and the full ACPL policy.
# Theory: For any node permutation π, if we permute X and edge_index accordingly
# and re-run the same model weights, the outputs reindex back by π^{-1}.
# We test this for the policy forward (θ: T×N×3) and the lifted coins (T×N×2×2),
# across several irregular graphs (incl. multigraph-like edge sets).

from collections.abc import Iterable
import random

import pytest
import torch

from acpl.policy.policy import ACPLPolicy, ACPLPolicyConfig

# ----------------------------
# Helpers
# ----------------------------


def _edge_index_from_pairs(pairs: Iterable[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    """
    Build a directed edge_index from a list of undirected (u,v) pairs by inserting both
    (u,v) and (v,u). Pairs may include directed entries already; we just add reverse arcs
    for (u,v) with u < v to emulate an undirected multigraph (common in our sims).
    """
    es: list[tuple[int, int]] = []
    for u, v in pairs:
        es.append((u, v))
        es.append((v, u))
    ei = torch.tensor(es, dtype=torch.long)  # (E, 2)

    # Sanity
    assert ei.numel() > 0
    assert int(ei[:, 0].max()) < num_nodes and int(ei[:, 1].max()) < num_nodes
    assert int(ei[:, 0].min()) >= 0 and int(ei[:, 1].min()) >= 0

    # Canonicalize: sort columns lexicographically by (src, dst)
    key = ei[:, 0] * (num_nodes + 1) + ei[:, 1]
    order = torch.argsort(key)
    ei = ei.index_select(0, order)

    return ei.t().contiguous()  # (2, E)


def _perm_nodes(pi: list[int], X: torch.Tensor, edge_index: torch.Tensor):
    """
    Apply node permutation π to (X, edge_index).
    - X': X'[i] = X[π^{-1}(i)]
    - edge_index': relabel u->π(u), v->π(v)
    Returns (X_pi, edge_index_pi, pi_t, inv) where:
      * pi_t is π as a LongTensor on the correct device
      * inv satisfies inv[pi_t] = arange(N)
    """
    device = X.device
    N = X.size(0)
    pi_t = torch.tensor(pi, dtype=torch.long, device=device)
    inv = torch.empty_like(pi_t)
    inv[pi_t] = torch.arange(N, device=device)

    # Relabel X by inverse to keep "semantics" at the same logical node
    X_pi = X.index_select(0, inv)

    # Relabel edges forward: (u,v) -> (π(u), π(v))
    u0, v0 = edge_index[0], edge_index[1]
    u = pi_t.index_select(0, u0)
    v = pi_t.index_select(0, v0)

    # Canonicalize permuted edges by (src, dst)
    key = u * (N + 1) + v
    order = torch.argsort(key)
    u = u.index_select(0, order)
    v = v.index_select(0, order)
    edge_index_pi = torch.stack([u, v], dim=0)

    return X_pi, edge_index_pi, pi_t, inv


def _rand_pairs_er(num_nodes: int, p: float, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if rng.random() < p:
                pairs.append((u, v))
                # small chance of duplicating to stress multigraph behavior
                if rng.random() < 0.1:
                    pairs.append((u, v))
    if not pairs:
        pairs = [(i, i + 1) for i in range(num_nodes - 1)]  # path fallback
    return pairs


def _make_policy(in_dim: int = 3) -> ACPLPolicy:
    cfg = ACPLPolicyConfig(
        in_dim=in_dim,
        gnn_hidden=32,
        gnn_out=32,
        gnn_activation="gelu",
        gnn_dropout=0.0,
        gnn_layernorm=True,
        gnn_residual=True,
        gnn_dropedge=0.0,
        controller="gru",
        ctrl_hidden=32,
        ctrl_layers=1,
        ctrl_dropout=0.0,
        ctrl_layernorm=True,
        ctrl_bidirectional=False,
        time_pe_dim=16,
        time_pe_learned_scale=True,
        head_hidden=0,
        head_out_scale=1.0,
        head_layernorm=True,
        head_dropout=0.0,
    )
    return ACPLPolicy(cfg)


def _rand_features(N: int, F: int, seed: int = 0, dtype=torch.float32, device=None):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    X = torch.randn(N, F, generator=g, device=device, dtype=dtype)
    # add a simple coordinate channel to break symmetries, as in training
    coord = torch.arange(N, device=device, dtype=dtype).unsqueeze(1) / max(N - 1, 1)
    if F >= 2:
        X[:, 0:1] = X[:, 0:1].abs() + 1.0  # nonnegative degree-like feature
        X[:, 1:2] = coord
    return X


# ----------------------------
# Core equivariance tests
# ----------------------------


@pytest.mark.parametrize(
    "pairs,num_nodes",
    [
        # Path / line with isolated at end (N=5; node 4 isolated)
        ([(0, 1), (1, 2), (2, 3)], 5),
        # 6-cycle
        ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], 6),
        # ER-like irregular (possible duplicates)
        (_rand_pairs_er(8, p=0.35, seed=7), 8),
        (_rand_pairs_er(10, p=0.25, seed=11), 10),
    ],
)
def test_policy_forward_equivariant_under_node_permutation(pairs, num_nodes):
    torch.manual_seed(123)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Graph + features
    edge_index = _edge_index_from_pairs(pairs, num_nodes).to(device)
    X = _rand_features(num_nodes, F=3, seed=1234, device=device)

    # Model
    model = _make_policy(in_dim=X.size(1)).to(device)
    model.eval()  # determinism (no dropout)

    # Time horizon
    T = 4

    # Original run
    with torch.no_grad():
        theta = model(X, edge_index, T=T)  # (T, N, 3)

    # Random permutation π
    pi = list(range(num_nodes))
    random.Random(2024).shuffle(pi)
    X_pi, edge_index_pi, pi_t, inv = _perm_nodes(pi, X, edge_index)
    # Permuted run
    with torch.no_grad():
        theta_pi = model(X_pi, edge_index_pi, T=T)

    # Unpermute back by π^{-1} on the node axis and compare
    theta_back = theta_pi.index_select(1, pi_t)  # gather θ_π[π(u)] → original order u

    assert torch.allclose(theta_back, theta, atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("num_nodes,seed", [(7, 0), (9, 1), (12, 2)])
def test_coins_lift_equivariant_under_node_permutation(num_nodes, seed):
    torch.manual_seed(321)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pairs = _rand_pairs_er(num_nodes, p=0.3, seed=seed)
    edge_index = _edge_index_from_pairs(pairs, num_nodes).to(device)
    X = _rand_features(num_nodes, F=2, seed=99, device=device)

    model = _make_policy(in_dim=X.size(1)).to(device)
    model.eval()
    T = 3

    with torch.no_grad():
        coins = model.coins_su2(X, edge_index, T=T)  # (T, N, 2, 2), complex

    # Permute nodes and edges
    pi = list(range(num_nodes))
    random.Random(17 + seed).shuffle(pi)
    X_pi, edge_index_pi, pi_t, inv = _perm_nodes(pi, X, edge_index)
    with torch.no_grad():
        coins_pi = model.coins_su2(X_pi, edge_index_pi, T=T)

    coins_back = coins_pi.index_select(1, pi_t)  # gather coins_π[π(u)] → original order u
    assert torch.allclose(coins_back, coins, atol=1e-6, rtol=1e-6)


# ----------------------------
# Gradient equivariance (∇ commutes with permutation)
# ----------------------------


@pytest.mark.parametrize("num_nodes,seed", [(6, 0), (8, 2)])
def test_gradient_equivariance_wrt_features(num_nodes, seed):
    """
    If f(X, E) is permutation–equivariant in X, then ∇_X L(f(X,E)) should permute the same way:
      let π be a permutation; define X' = Π^{-1}X and E' = ΠE;
      then grad' pulled back by Π equals grad on the original.
    We test with a simple scalar loss L = ||θ||^2 + ||coins||^2 aggregated over time.
    """
    torch.manual_seed(202)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pairs = _rand_pairs_er(num_nodes, p=0.35, seed=seed)
    edge_index = _edge_index_from_pairs(pairs, num_nodes).to(device)

    F = 3
    X = _rand_features(num_nodes, F=F, seed=7 + seed, device=device).requires_grad_(True)

    model = _make_policy(in_dim=F).to(device)
    model.train()  # enable potential layernorm affine grads etc.
    T = 3

    # Original loss
    theta = model(X, edge_index, T=T)  # (T, N, 3)
    coins = model.coins_su2(X, edge_index, T=T)  # (T, N, 2, 2)
    loss = (theta**2).sum() + (coins.abs() ** 2).sum()
    loss.backward()
    grad = X.grad.detach().clone()

    # Permute inputs
    pi = list(range(num_nodes))
    random.Random(12345 + seed).shuffle(pi)
    X.grad.zero_()  # clear
    X_pi, edge_index_pi, pi_t, inv = _perm_nodes(pi, X.detach(), edge_index)
    X_pi.requires_grad_(True)

    theta_pi = model(X_pi, edge_index_pi, T=T)
    coins_pi = model.coins_su2(X_pi, edge_index_pi, T=T)
    loss_pi = (theta_pi**2).sum() + (coins_pi.abs() ** 2).sum()
    loss_pi.backward()
    grad_pi = X_pi.grad.detach()

    # Pull back grad' by π^{-1} and compare
    grad_back = grad_pi.index_select(0, pi_t)  # gather grad_π[π(u)] → original order u

    assert torch.allclose(grad_back, grad, atol=5e-6, rtol=5e-6)


# ----------------------------
# Sanity: output shapes under broadcasted time coords
# ----------------------------


def test_forward_shapes_with_time_length():
    torch.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pairs = [(0, 1), (1, 2), (2, 0), (2, 3)]
    N = 4
    edge_index = _edge_index_from_pairs(pairs, N).to(device)
    X = _rand_features(N, F=2, seed=77, device=device)

    model = _make_policy(in_dim=X.size(1)).to(device)
    model.eval()

    T = 5
    θ = model(X, edge_index, T=T)  # (T, N, 3)
    U = model.coins_su2(X, edge_index, T=T)  # (T, N, 2, 2)

    assert θ.shape == (T, N, 3)
    assert U.shape == (T, N, 2, 2)

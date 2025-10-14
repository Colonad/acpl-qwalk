# tests/test_message_passing_equiv.py
from __future__ import annotations

import numpy as np
import torch

from acpl.data.graphs import line_graph
from acpl.policy.policy import GNNTemporalPolicy


def _inverse_perm(p: np.ndarray) -> np.ndarray:
    inv = np.empty_like(p)
    inv[p] = np.arange(p.size, dtype=p.dtype)
    return inv


def _permute_edge_index(edge_index: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Apply the SAME reordering used by x_perm = x[perm].
    Since perm maps new_idx -> old_idx, we need inv to map old_idx -> new_idx
    when rewriting edge endpoints.
    """
    inv = _inverse_perm(perm)
    ei = edge_index.copy()
    ei[0, :] = inv[ei[0, :]]
    ei[1, :] = inv[ei[1, :]]
    return ei


def test_policy_forward_is_node_permutation_equivariant():
    torch.manual_seed(0)
    np.random.seed(0)

    # Small line graph: permutation-equivalence easiest to see
    n = 8
    edge_index_np, degrees_np, coords_np, _ = line_graph(n, seed=123)
    edge_index = torch.from_numpy(edge_index_np).to(dtype=torch.long)

    # Random node features (float), no positional encodings
    f = 3
    x = torch.randn(n, f, dtype=torch.float32)

    # Build policy with zero dropout and deterministic settings
    policy = GNNTemporalPolicy(
        in_dim=f,
        gnn_hidden=16,
        gnn_dropout=0.0,
        controller_hidden=16,
        controller_layers=1,
        controller_dropout=0.0,
        controller_bidirectional=False,
        head_angle_range="tanh_pi",
        head_dropout=0.0,
        use_time_embed=False,  # time scalar won't affect equivariance, but keep off
    )

    # Forward on the original ordering
    theta, h_next = policy(
        g=edge_index,
        x=x,
        pos_enc=None,
        t_over_t=0.25,
        h_prev=None,
    )  # theta: (N,3), h_next: (N,Hc)

    # Create a random permutation of nodes and apply to graph + features
    perm = np.random.permutation(n).astype(np.int64)
    inv = _inverse_perm(perm)

    edge_index_perm_np = _permute_edge_index(edge_index_np, perm)
    edge_index_perm = torch.from_numpy(edge_index_perm_np).to(dtype=torch.long)
    x_perm = x[torch.from_numpy(perm), :]

    # Forward on permuted inputs
    theta_perm, h_next_perm = policy(
        g=edge_index_perm,
        x=x_perm,
        pos_enc=None,
        t_over_t=0.25,
        h_prev=None,
    )

    # Equivariance: outputs must permute the same way
    #  theta_perm (in permuted order) == theta[perm]
    assert torch.allclose(theta_perm, theta[torch.from_numpy(perm)], atol=1e-6, rtol=0.0)
    assert torch.allclose(h_next_perm, h_next[torch.from_numpy(perm)], atol=1e-6, rtol=0.0)

    # Also check the inverse indexing aligns both ways (more explicit sanity)
    #  Bring permuted outputs back to the original order: out_perm[inv] == out
    assert torch.allclose(
        theta_perm[torch.from_numpy(inv)],
        theta,
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        h_next_perm[torch.from_numpy(inv)],
        h_next,
        atol=1e-6,
        rtol=0.0,
    )

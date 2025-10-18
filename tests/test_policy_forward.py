# tests/test_policy_forward.py
import torch
import pytest

from acpl.policy.policy import ACPLPolicy, ACPLPolicyConfig
from acpl.data.graphs import grid_graph
from acpl.data.features import FeatureSpec, build_node_features


def _make_graph_and_features(Lx=4, Ly=4, seed=0, use_lap_pe=True):
    edge_index, degrees, coords, _ = grid_graph(Lx, Ly, seed=seed)
    spec = FeatureSpec(
        use_degree=True, degree_norm="inv_sqrt",
        use_coords=True,
        use_lap_pe=use_lap_pe, lap_pe_k=min(4, degrees.numel() - 1), lap_pe_norm="sym",
        use_rwse=False,
    )
    X, _ = build_node_features(edge_index, degrees, coords, spec=spec)
    return edge_index, X


@pytest.mark.parametrize("controller", ["gru", "transformer"])
def test_policy_forward_shapes(controller):
    torch.manual_seed(123)
    edge_index, X = _make_graph_and_features(4, 4, seed=0, use_lap_pe=True)
    N = X.size(0)
    T = 5

    cfg = ACPLPolicyConfig(
        in_dim=X.size(1),
        gnn_hidden=64, gnn_out=64,
        controller=controller, ctrl_hidden=64, ctrl_layers=1, ctrl_dropout=0.0,
        time_pe_dim=32,
        head_hidden=0,
    )
    policy = ACPLPolicy(cfg)

    theta = policy(X, edge_index, T=T)         # (T, N, 3)
    coins = policy.coins_su2(X, edge_index, T=T)

    assert theta.shape == (T, N, 3)
    assert coins.shape == (T, N, 2, 2)
    assert torch.isfinite(theta).all()
    assert torch.isfinite(coins.real).all() and torch.isfinite(coins.imag).all()


def _blocks_unitary_and_det1(U: torch.Tensor, atol=1e-5):
    """
    U: (..., 2, 2) complex
    Check U^\dagger U ≈ I and |det(U)| ≈ 1 blockwise.
    """
    assert U.shape[-2:] == (2, 2)
    UhU = torch.matmul(U.conj().transpose(-1, -2), U)
    I = torch.eye(2, dtype=U.dtype, device=U.device).expand_as(UhU)
    ok_unitary = torch.allclose(UhU, I, atol=atol, rtol=0)

    det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]
    ones = torch.ones_like(det.abs())
    ok_det = torch.allclose(det.abs(), ones, atol=1e-5, rtol=0)

    return ok_unitary and ok_det


def test_policy_coins_are_unitary_su2():
    torch.manual_seed(7)
    edge_index, X = _make_graph_and_features(5, 5, seed=1, use_lap_pe=True)
    T = 4
    cfg = ACPLPolicyConfig(
        in_dim=X.size(1),
        gnn_hidden=48, gnn_out=48,
        controller="gru", ctrl_hidden=48, ctrl_layers=1,
        time_pe_dim=16,
        head_hidden=0,
    )
    policy = ACPLPolicy(cfg)
    coins = policy.coins_su2(X, edge_index, T=T)  # (T, N, 2, 2) complex

    assert _blocks_unitary_and_det1(coins, atol=1e-5)


def test_policy_determinism_same_seed():
    # Same seed → same initialized parameters → same outputs
    torch.manual_seed(2025)
    edge_index, X = _make_graph_and_features(4, 3, seed=2, use_lap_pe=False)
    T = 3

    torch.manual_seed(999)
    pol1 = ACPLPolicy(ACPLPolicyConfig(
        in_dim=X.size(1),
        gnn_hidden=32, gnn_out=32,
        controller="gru", ctrl_hidden=32, ctrl_layers=1,
        time_pe_dim=16,
        head_hidden=0,
    ))
    out1 = pol1(X, edge_index, T=T)
    coins1 = pol1.coins_su2(X, edge_index, T=T)

    torch.manual_seed(999)
    pol2 = ACPLPolicy(ACPLPolicyConfig(
        in_dim=X.size(1),
        gnn_hidden=32, gnn_out=32,
        controller="gru", ctrl_hidden=32, ctrl_layers=1,
        time_pe_dim=16,
        head_hidden=0,
    ))
    out2 = pol2(X, edge_index, T=T)
    coins2 = pol2.coins_su2(X, edge_index, T=T)

    assert torch.allclose(out1, out2, atol=0.0)
    assert torch.allclose(coins1, coins2, atol=0.0)


def test_policy_backward_path_exists():
    # Ensure gradients propagate to parameters through coins_su2
    torch.manual_seed(31415)
    edge_index, X = _make_graph_and_features(3, 3, seed=0, use_lap_pe=True)
    T = 2

    cfg = ACPLPolicyConfig(
        in_dim=X.size(1),
        gnn_hidden=32, gnn_out=32,
        controller="transformer", ctrl_hidden=32, ctrl_layers=1,
        time_pe_dim=16,
        head_hidden=0,
    )
    policy = ACPLPolicy(cfg)

    # Simple scalar loss: sum of squared magnitudes of coin entries at all (t,n)
    coins = policy.coins_su2(X, edge_index, T=T)  # complex
    loss = (coins.real ** 2 + coins.imag ** 2).sum()
    loss.backward()

    # Some parameter must have non-zero grad (e.g., head or encoder)
    has_grad = False
    for p in policy.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "Expected at least one parameter to receive gradient."

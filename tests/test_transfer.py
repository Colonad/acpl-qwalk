# tests/test_transfer.py
import pytest
import torch

from acpl.objectives.transfer import loss_state_transfer, success_prob


def test_success_prob_single_vector():
    p = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
    s2 = success_prob(p, 2, check_prob=True)
    assert torch.isclose(s2, torch.tensor(0.7))
    s0 = success_prob(p, 0, check_prob=True)
    assert torch.isclose(s0, torch.tensor(0.1))


def test_loss_single_vector_equals_one_minus_success():
    p = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float32)
    s = success_prob(p, 2, check_prob=True)
    loss = loss_state_transfer(p, 2, check_prob=True)
    assert torch.isclose(loss, 1.0 - s)


def test_success_prob_batched_shared_target():
    # p shape (V, B) with V=3, B=4
    p = torch.tensor(
        [
            [0.7, 0.1, 0.2, 0.6],
            [0.2, 0.7, 0.1, 0.2],
            [0.1, 0.2, 0.7, 0.2],
        ],
        dtype=torch.float32,
    )
    # Same target for all batch items
    s = success_prob(p, 2, check_prob=True)
    assert s.shape == (4,)
    exp = torch.tensor([0.1, 0.2, 0.7, 0.2], dtype=torch.float32)
    assert torch.allclose(s, exp)


def test_success_prob_batched_per_example_targets():
    p = torch.tensor(
        [
            [0.7, 0.1, 0.2],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=torch.float32,
    )  # (3, 3)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    s = success_prob(p, targets, check_prob=True)
    assert s.shape == (3,)
    assert torch.allclose(s, torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32))


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_loss_batched_reductions(reduction: str):
    p = torch.tensor(
        [
            [0.7, 0.1, 0.2],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=torch.float32,
    )  # (3,3)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    loss = loss_state_transfer(p, targets, reduction=reduction, check_prob=True)
    if reduction == "none":
        assert loss.shape == (3,)
        assert torch.allclose(loss, torch.tensor([0.3, 0.3, 0.3], dtype=torch.float32))
    elif reduction == "mean":
        assert torch.isclose(loss, torch.tensor(0.3))
    else:
        assert torch.isclose(loss, torch.tensor(0.9))


def test_check_prob_raises_on_invalid_vector():
    # Negative mass and sum != 1 should raise when check_prob=True
    p = torch.tensor([0.6, -0.1, 0.6], dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = success_prob(p, 0, check_prob=True)


def test_renorm_repairs_and_loss_matches_1_minus_success():
    # Not a valid prob vector; renorm should fix it
    p = torch.tensor([-0.1, 0.2, 0.1], dtype=torch.float32)
    s = success_prob(p, 1, renorm=True)
    # After renorm, p becomes [0.0, 2/3, 1/3]
    assert torch.isclose(s, torch.tensor(2.0 / 3.0))
    loss = loss_state_transfer(p, 1, renorm=True)
    assert torch.isclose(loss, 1.0 - s)

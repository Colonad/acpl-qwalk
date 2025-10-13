import pytest
import torch

from acpl.sim.coins import (
    coins_cayley_from_params,
    coins_exp_from_params,
    coins_su2_from_theta,
    skew_hermitian_from_params,
    verify_unitary_blocks,
)


def _randn(*shape, dtype=torch.float32, device="cpu"):
    return torch.randn(*shape, dtype=dtype, device=device)


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_su2_unitary_batched(batch_shape):
    N = 7
    theta = _randn(*batch_shape, N, 3)
    U = coins_su2_from_theta(theta, dtype=torch.complex64)
    ok, err = verify_unitary_blocks(U)
    assert ok, f"SU(2) not unitary; max err={float(err.max()):.3e}"


@pytest.mark.parametrize("d", [3, 4, 5])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 2)])
def test_exp_and_cayley_unitary(d, batch_shape):
    N = 5
    params = _randn(*batch_shape, N, 2, d, d)  # unconstrained real
    # sanity: K is skew-Hermitian
    K = skew_hermitian_from_params(params, d)
    Kdag = K.conj().transpose(-1, -2)
    assert torch.allclose(Kdag, -K, atol=1e-6)

    U_exp = coins_exp_from_params({d: params})[d]
    U_cay = coins_cayley_from_params({d: params})[d]

    ok_e, err_e = verify_unitary_blocks(U_exp)
    ok_c, err_c = verify_unitary_blocks(U_cay)
    assert ok_e and ok_c, (
        f"exp or cayley not unitary; max={max(float(err_e.max()), float(err_c.max())):.3e}"
    )

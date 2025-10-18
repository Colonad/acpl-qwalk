# tests/test_partial_trace.py
# Theory-driven tests for position marginals (partial trace over coin space)
# covering: normalization, batching, dtype robustness, isolated vertices,
# and invariance under two shifts with identity coins (S^2 = I / flip-flop).

import pytest
import torch

from acpl.sim.portmap import build_portmap
from acpl.sim.shift import build_shift
from acpl.sim.step import step
from acpl.sim.utils import (
    as_complex,
    check_probability_simplex,
    partial_trace_coin,
    position_probabilities,
    renorm_state_,
    state_norm2,
)


def _identity_su2_stack(num_nodes: int, dtype=torch.complex64) -> torch.Tensor:
    I2 = torch.eye(2, dtype=dtype)
    return I2.unsqueeze(0).repeat(num_nodes, 1, 1)


def _rand_complex_vec(shape, dtype=torch.complex64, seed=0, device=None):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    re = torch.randn(*shape, generator=g, device=device)
    im = torch.randn(*shape, generator=g, device=device)
    return (re + 1j * im).to(dtype)


# --------------------------------------------------------------------------------------
# Core: partial trace equals position probabilities and lies on probability simplex
# --------------------------------------------------------------------------------------


def test_partial_trace_simplex_line_graph():
    pairs = [(0, 1), (1, 2)]
    pm = build_portmap(pairs, num_nodes=3, coalesce=False)
    A = pm.src.numel()

    torch.manual_seed(0)
    psi = torch.randn(A) + 1j * torch.randn(A)
    psi = as_complex(psi)
    renorm_state_(psi)

    P = position_probabilities(psi, pm)  # normalized by default
    check_probability_simplex(P)

    P_alias = partial_trace_coin(psi, pm)
    assert torch.allclose(P, P_alias, atol=1e-7)


def test_batched_partial_trace_and_simplex():
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    pm = build_portmap(pairs, num_nodes=5, coalesce=False)
    A = pm.src.numel()

    torch.manual_seed(2)
    B = 4
    psi_b = (torch.randn(B, A) + 1j * torch.randn(B, A)).to(torch.complex64)
    renorm_state_(psi_b)

    P_b = position_probabilities(psi_b, pm)
    check_probability_simplex(P_b)
    ones = torch.ones(B, dtype=P_b.dtype, device=P_b.device)
    assert torch.allclose(P_b.sum(dim=-1), ones, atol=1e-6)


# --------------------------------------------------------------------------------------
# Normalization consistency: with normalize=False sum equals ||psi||^2
# --------------------------------------------------------------------------------------


def test_normalize_false_matches_state_norm2():
    pairs = [(0, 1), (1, 2)]
    pm = build_portmap(pairs, num_nodes=3, coalesce=False)
    A = pm.src.numel()

    torch.manual_seed(3)
    psi = (torch.randn(A) + 1j * torch.randn(A)).to(torch.complex64)  # not normalized
    n2 = state_norm2(psi)

    P_raw = position_probabilities(psi, pm, normalize=False)
    assert torch.allclose(P_raw.sum(), n2, atol=1e-6)

    renorm_state_(psi)
    P_norm = position_probabilities(psi, pm, normalize=True)
    check_probability_simplex(P_norm)


def test_batched_normalize_false_matches_state_norm2():
    pairs = [(0, 1), (1, 2), (2, 0)]
    pm = build_portmap(pairs, num_nodes=3, coalesce=False)
    A = pm.src.numel()

    psi_b = _rand_complex_vec((5, A), dtype=torch.complex64, seed=44)
    n2_b = (psi_b.conj() * psi_b).real.sum(dim=-1)

    P_raw = position_probabilities(psi_b, pm, normalize=False)
    assert torch.allclose(P_raw.sum(dim=-1), n2_b, atol=1e-6)

    renorm_state_(psi_b)
    P_norm = position_probabilities(psi_b, pm, normalize=True)
    check_probability_simplex(P_norm)


# --------------------------------------------------------------------------------------
# Manual grouping check (∑|ψ_a|^2 over arcs a with src(a)=v)
# --------------------------------------------------------------------------------------


def test_against_manual_grouping_small_graph():
    pairs = [(0, 1), (1, 0), (0, 1), (1, 2)]
    pm = build_portmap(pairs, num_nodes=3, coalesce=False)
    A = pm.src.numel()

    torch.manual_seed(4)
    psi = (torch.randn(A) + 1j * torch.randn(A)).to(torch.complex64)
    renorm_state_(psi)

    P = position_probabilities(psi, pm)

    src = pm.src
    abs2 = (psi.conj() * psi).real
    P_manual = torch.zeros(pm.num_nodes, dtype=abs2.dtype, device=abs2.device)
    for a in range(A):
        P_manual[int(src[a])] += abs2[a]

    P_manual = P_manual / P_manual.sum()
    assert torch.allclose(P, P_manual, atol=1e-7)


# --------------------------------------------------------------------------------------
# Isolated vertices: probability mass excludes deg=0 nodes but P still sums to 1
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_isolated_vertices_simplex_and_zero_mass(dtype):
    # Graph with nodes {0,1,2,3,4} where node 4 is isolated.
    pairs = [(0, 1), (1, 2), (2, 3)]
    pm = build_portmap(pairs, num_nodes=5, coalesce=False)
    A = pm.src.numel()

    psi = _rand_complex_vec((A,), dtype=dtype, seed=55)
    renorm_state_(psi)

    P = position_probabilities(psi, pm)  # normalized → simplex
    check_probability_simplex(P)

    # Node 4 has deg=0 → no arcs contribute to it, so probability should be 0
    assert P.shape[-1] == 5
    assert torch.allclose(P[-1], torch.zeros((), dtype=P.dtype, device=P.device), atol=1e-8)


# --------------------------------------------------------------------------------------
# Device/dtype coverage
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_dtype_coverage_and_alias(dtype):
    pairs = [(0, 1), (1, 2), (2, 0)]
    pm = build_portmap(pairs, num_nodes=3, coalesce=False)
    A = pm.src.numel()

    psi = _rand_complex_vec((A,), dtype=dtype, seed=99)
    renorm_state_(psi)

    P1 = position_probabilities(psi, pm)
    P2 = partial_trace_coin(psi, pm)
    assert torch.allclose(P1, P2, atol=1e-7 if dtype is torch.complex128 else 1e-6)
    check_probability_simplex(P1)


def test_shift_on_available_device_and_trace_simplex():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    pm = build_portmap(pairs, num_nodes=4, coalesce=False)
    S = build_shift(pm).to_sparse()

    A = pm.src.numel()
    psi = _rand_complex_vec((A,), seed=77, device=device)
    renorm_state_(psi)

    P0 = position_probabilities(psi, pm)
    check_probability_simplex(P0)

    # Pure shift preserves norm; tracing coin afterward should still yield a simplex
    psi1 = S @ psi
    P1 = position_probabilities(psi1, pm)
    check_probability_simplex(P1)


# --------------------------------------------------------------------------------------
# Physics-facing: two shifts + identity coins ⇒ same position distribution
# --------------------------------------------------------------------------------------


def test_invariance_under_two_shifts_with_identity_coins():
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    pm = build_portmap(pairs, num_nodes=4, coalesce=False)
    S = build_shift(pm)
    A = pm.src.numel()

    torch.manual_seed(1)
    psi = (torch.randn(A) + 1j * torch.randn(A)).to(torch.complex64)
    renorm_state_(psi)

    coins = _identity_su2_stack(pm.num_nodes, dtype=torch.complex64)

    P0 = position_probabilities(psi, pm)
    psi_1 = step(psi, pm, coins, shift=S)  # one shift
    psi_2 = step(psi_1, pm, coins, shift=S)  # two shifts (flip-flop ⇒ back)
    P2 = position_probabilities(psi_2, pm)

    check_probability_simplex(P0)
    check_probability_simplex(P2)
    assert torch.allclose(P0, P2, atol=1e-6)

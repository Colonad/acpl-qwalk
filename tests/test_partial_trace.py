# tests/test_partial_trace.py
from __future__ import annotations

import numpy as np
import torch

from acpl.sim.portmap import make_flipflop_portmap
from acpl.sim.utils import partial_trace_position, portmap_tensors


def _path_edges(n: int) -> np.ndarray:
    """Undirected path 0-1-...-(n-1) as (2, E) array."""
    u = np.arange(n - 1, dtype=np.int64)
    v = u + 1
    return np.stack([u, v], axis=0)


def _random_complex(shape, device, dtype=torch.complex64) -> torch.Tensor:
    real = torch.randn(*shape, device=device, dtype=torch.float32)
    imag = torch.randn(*shape, device=device, dtype=torch.float32)
    return (real + 1j * imag).to(dtype=dtype)


def test_partial_trace_single_state_sums_to_one_and_nonnegative():
    ei = _path_edges(6)  # V=6, E=5, A=10
    pm = make_flipflop_portmap(ei)
    a = pm.num_arcs
    device = torch.device("cpu")

    # Random complex psi with ||psi||_2 = 1
    psi = _random_complex((a,), device=device)
    psi = psi / psi.norm(p=2)

    p = partial_trace_position(psi, pm)  # (V,)
    assert p.shape == (pm.num_nodes,)
    # Sum ≈ 1
    assert torch.isclose(p.sum(), torch.tensor(1.0, device=device), atol=1e-6, rtol=0)
    # Nonnegative (allow tiny numerical negatives)
    assert torch.all(p >= -1e-12)


def test_partial_trace_batched_states_sum_to_one_per_column():
    ei = _path_edges(5)  # V=5, E=4, A=8
    pm = make_flipflop_portmap(ei)
    a = pm.num_arcs
    b = 3
    device = torch.device("cpu")

    # Random complex psi (A,B), each column normalized to unit norm
    psi = _random_complex((a, b), device=device)
    norms = torch.linalg.vector_norm(psi, ord=2, dim=0, keepdim=True).clamp_min(1e-12)
    psi = psi / norms

    p = partial_trace_position(psi, pm)  # (V,B)
    assert p.shape == (pm.num_nodes, b)

    # Column sums ≈ 1
    col_sums = p.sum(dim=0)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-6, rtol=0)

    # Nonnegative
    assert torch.all(p >= -1e-12)


def test_partial_trace_concentrated_on_vertex_is_one_hot():
    ei = _path_edges(7)  # V=7
    pm = make_flipflop_portmap(ei)
    a = pm.num_arcs
    v = pm.num_nodes
    device = torch.device("cpu")

    # Put amplitude uniformly on the outgoing ports of a chosen vertex v0.
    v0 = 3
    pt = portmap_tensors(pm, device=device)
    start = int(pt.node_ptr[v0].item())
    end = int(pt.node_ptr[v0 + 1].item())
    deg_v0 = max(1, end - start)

    psi = torch.zeros(a, device=device, dtype=torch.complex64)
    amp = 1.0 / (deg_v0**0.5)
    psi[start:end] = torch.tensor(amp, device=device, dtype=torch.complex64)

    p = partial_trace_position(psi, pm)  # (V,)
    # One-hot at v0 (within tolerance)
    expect = torch.zeros(v, device=device, dtype=p.dtype)
    expect[v0] = 1.0
    assert torch.allclose(p, expect, atol=1e-7, rtol=0)

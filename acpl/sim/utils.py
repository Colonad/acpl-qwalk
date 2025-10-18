# acpl/sim/utils.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import torch

from .portmap import PortMap

__all__ = [
    # dtype / device helpers
    "require_complex",
    "as_complex",
    "state_norm2",
    "renorm_state_",
    # CSR / arc helpers
    "node_arc_range",
    "arange_arcs_of",
    # segment reductions
    "segment_sum",
    "segment_max",
    "segment_softmax",
    # block-diagonal utilities (arbitrary degrees)
    "degrees_from_portmap",
    "arc_slices_from_portmap",
    "check_unitary_blocks",
    "ensure_coin_blocks",
    "blockdiag_mv",
    # partial trace
    "partial_trace_coin",
    "position_probabilities",
    "check_probability_simplex",
]

# --------------------------------------------------------------------------- #
#                           dtype / device helpers                            #
# --------------------------------------------------------------------------- #


def require_complex(x: torch.Tensor) -> None:
    """Raise if `x` is not complex."""
    if not x.is_complex():
        raise TypeError("Expected complex64 or complex128 tensor.")


def as_complex(
    x: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Cast `x` to complex dtype (default complex64). If `x` is already complex,
    enforce target dtype when provided.

    Notes
    -----
    • This never copies if the dtype already matches.
    • Real tensors are promoted to complex with zero imaginary part.
    """
    if dtype is None:
        dtype = torch.complex64
    if x.is_complex():
        return x if x.dtype == dtype else x.to(dtype)
    base = torch.float32 if dtype == torch.complex64 else torch.float64
    return x.to(base).to(dtype)


def _add_axis_for_div(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Utility: expand x with trailing singleton axes until ranks match `like`."""
    while x.ndim < like.ndim:
        x = x.unsqueeze(-1)
    return x


def state_norm2(psi: torch.Tensor) -> torch.Tensor:
    """
    2-norm squared of state over last axis.

    psi: (..., A) complex
    returns: (...) real >= 0
    """
    require_complex(psi)
    return (psi.conj() * psi).real.sum(dim=-1)


@torch.no_grad()
def renorm_state_(
    psi: torch.Tensor,
    *,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    In-place L2 normalization of `psi` over the last axis.

    psi: (..., A) complex
    returns: psi (normalized) for convenience
    """
    require_complex(psi)
    n2 = state_norm2(psi).clamp_min(eps).sqrt()  # (...,)
    psi /= _add_axis_for_div(n2, psi)
    return psi


# --------------------------------------------------------------------------- #
#                        CSR / arc indexing helpers                           #
# --------------------------------------------------------------------------- #


def node_arc_range(pm: PortMap, u: int) -> tuple[int, int]:
    """
    Return (start, end) arc indices for the CSR slice of node u (outgoing arcs).
    """
    s = int(pm.node_ptr[u])
    e = int(pm.node_ptr[u + 1])
    return s, e


def arange_arcs_of(pm: PortMap, u: int) -> torch.Tensor:
    """
    Return a 1-D LongTensor of arc indices originating at node u.
    """
    s, e = node_arc_range(pm, u)
    if e <= s:
        return torch.empty(0, dtype=torch.long, device=pm.src.device)
    return torch.arange(s, e, dtype=torch.long, device=pm.src.device)


def degrees_from_portmap(pm: PortMap) -> torch.Tensor:
    """
    Degrees dv as a 1-D LongTensor of shape (N,), where N=pm.num_nodes,
    computed from CSR pointer differences.
    """
    ptr = pm.node_ptr  # (N+1,)
    return (ptr[1:] - ptr[:-1]).to(torch.long)


def arc_slices_from_portmap(pm: PortMap) -> list[slice]:
    """
    List of Python slices [ slice(s_u, e_u) ] per node u following CSR layout.
    """
    ptr = pm.node_ptr
    return [slice(int(ptr[u]), int(ptr[u + 1])) for u in range(pm.num_nodes)]


# --------------------------------------------------------------------------- #
#                          Segment reductions (CPU/GPU)                       #
# --------------------------------------------------------------------------- #


def _check_segment_inputs(values: torch.Tensor, index: torch.Tensor, num_segments: int) -> None:
    if index.dtype != torch.long:
        raise TypeError("index must be a LongTensor")
    if values.size(-1) != index.numel():
        raise ValueError("values last-dimension must match index length")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")


def segment_sum(
    values: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Sum `values` into `num_segments` bins using `index` (scatter-add).

    values: (A,)      OR (B, A) real/complex
    index:  (A,) long (bin id per element in A)
    returns:
      if values.ndim == 1 → (N,) where N=num_segments
      if values.ndim == 2 → (B, N)

    Notes
    -----
    • Implemented with index_add_; fast and differentiable.
    • Works for real and complex tensors.
    """
    _check_segment_inputs(values, index, num_segments)

    if values.ndim == 1:
        out = torch.zeros(num_segments, dtype=values.dtype, device=values.device)
        out.index_add_(0, index, values)
        return out

    if values.ndim == 2:
        B, A = values.shape
        out = torch.zeros(B, num_segments, dtype=values.dtype, device=values.device)
        # Loop over batch dimension (A dominates, N moderate); keeps memory small.
        for b in range(B):
            out[b].index_add_(0, index, values[b])
        return out

    raise ValueError("values must be rank 1 or 2 (A) or (B, A).")


def segment_max(
    values: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """
    Per-segment maximum (stable fallback without torch.scatter_reduce('amax')).

    values: (A,) or (B, A) real
    returns: same shape as segment_sum output: (N,) or (B, N)

    Notes
    -----
    • Uses a simple O(A) loop per batch to avoid extra deps; A dominates and N is moderate
      in our settings (arc count vs. node count).
    • Suitable for softmax stabilization.
    """
    _check_segment_inputs(values, index, num_segments)
    if values.is_complex():
        raise TypeError("segment_max expects real values.")

    def _single(v_1d: torch.Tensor) -> torch.Tensor:
        buf = torch.full((num_segments,), -torch.inf, dtype=v_1d.dtype, device=v_1d.device)
        # Manual reduce-maximum
        # (Vectorized alternatives require scatter_reduce_ which we avoid for portability.)
        for a in range(v_1d.numel()):
            i = int(index[a])
            val = v_1d[a]
            if val > buf[i]:
                buf[i] = val
        return buf

    if values.ndim == 1:
        return _single(values)

    if values.ndim == 2:
        B, A = values.shape
        out = torch.empty(B, num_segments, dtype=values.dtype, device=values.device)
        for b in range(B):
            out[b] = _single(values[b])
        return out

    raise ValueError("values must be rank 1 or 2 (A) or (B, A).")


def segment_softmax(
    scores: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
    *,
    temperature: float = 1.0,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Softmax within segments indicated by `index`.

    scores: (A,) or (B, A) real
    returns: same shape as scores, with softmax per segment.

    Notes
    -----
    • Numerically stable: subtracts segment-wise max.
    • Batched variant loops across batch but stays memory-friendly.
    • `temperature` > 0 rescales logits; lower is sharper.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if scores.is_complex():
        raise TypeError("segment_softmax expects real scores.")
    _check_segment_inputs(scores, index, num_segments)

    if scores.ndim == 1:
        seg_max = segment_max(scores, index, num_segments)  # (N,)
        shifted = scores - seg_max.index_select(0, index)  # (A,)
        exps = torch.exp(shifted / temperature)  # (A,)
        denom = segment_sum(exps, index, num_segments).index_select(0, index).clamp_min(eps)
        return exps / denom

    if scores.ndim == 2:
        B, A = scores.shape
        out = torch.empty_like(scores)
        for b in range(B):
            out[b] = segment_softmax(
                scores[b], index, num_segments, temperature=temperature, eps=eps
            )
        return out

    raise ValueError("scores must be rank 1 or 2.")


# --------------------------------------------------------------------------- #
#         Block-diagonal (direct-sum) builders and matvec for coins           #
#                 (arbitrary degrees, mixed across vertices)                  #
# --------------------------------------------------------------------------- #

_BlockList = Sequence[torch.Tensor]
_PackedBlocks = torch.Tensor  # shape (N, dmax, dmax), padded; use per-node degrees to slice
_CoinBlocks = Union[_BlockList, _PackedBlocks]


def _is_square(mat: torch.Tensor) -> bool:
    return mat.ndim == 2 and mat.shape[0] == mat.shape[1]


@torch.no_grad()
def check_unitary_blocks(
    blocks: _CoinBlocks,
    degrees: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> None:
    """
    Debug/CI helper: assert each local block is unitary (C† C ≈ I_{dv}).

    Parameters
    ----------
    blocks : list[Tensor(dv,dv)] or Tensor(N,dmax,dmax)
        Local coin blocks per vertex.
    degrees : LongTensor(N,)
        Per-vertex degrees dv.
    """
    if isinstance(blocks, torch.Tensor):
        assert blocks.ndim == 3, "packed blocks must be (N,dmax,dmax)"
        N, dmax, _ = blocks.shape
        for u in range(N):
            dv = int(degrees[u])
            if dv == 0:
                continue
            Cu = blocks[u, :dv, :dv]
            if not _is_square(Cu):
                raise AssertionError(f"Block {u} not square.")
            I = torch.eye(dv, dtype=Cu.dtype, device=Cu.device)
            err = (Cu.conj().T @ Cu - I).abs().max().item()
            if err > atol:
                raise AssertionError(f"Block {u} not unitary within atol={atol}: max|C†C-I|={err}")
        return

    # list-like
    for u, Cu in enumerate(blocks):
        if Cu.numel() == 0:
            continue
        if not _is_square(Cu):
            raise AssertionError(f"Block {u} not square.")
        dv = Cu.shape[0]
        I = torch.eye(dv, dtype=Cu.dtype, device=Cu.device)
        err = (Cu.conj().T @ Cu - I).abs().max().item()
        if err > atol:
            raise AssertionError(f"Block {u} not unitary within atol={atol}: max|C†C-I|={err}")


def ensure_coin_blocks(
    blocks: _CoinBlocks,
    degrees: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[_CoinBlocks, torch.dtype, torch.device]:
    """
    Validate/adjust local coin blocks:
      • ensure complex dtype (default complex64)
      • ensure device match
      • ensure per-node shapes match degrees

    Returns
    -------
    (blocks_adj, dtype, device)

    Notes
    -----
    • Accepts a list of (dv,dv) tensors or a packed (N,dmax,dmax) tensor.
    • Does not check unitarity by default; use `check_unitary_blocks` in tests.
    """
    if dtype is None:
        dtype = torch.complex64

    if isinstance(blocks, torch.Tensor):
        if blocks.ndim != 3:
            raise ValueError("Packed blocks must be rank-3: (N,dmax,dmax).")
        if device is None:
            device = blocks.device
        blk = as_complex(blocks.to(device), dtype=dtype)
        # shape checks deferred to matvec where dv slicing occurs
        return blk, dtype, device

    # list-like
    if not isinstance(blocks, (list, tuple)):
        raise TypeError("blocks must be list/tuple of per-node mats or a packed 3D tensor.")

    _device = device
    out_list: list[torch.Tensor] = []
    for u, Cu in enumerate(blocks):
        # Allow empty blocks for dv=0 (rare but not illegal)
        if Cu.numel() == 0:
            out_list.append(Cu)
            continue
        CuC = as_complex(Cu, dtype=dtype)
        if _device is None:
            _device = CuC.device
        else:
            CuC = CuC.to(_device)
        dv = int(degrees[u])
        if dv != 0 and CuC.shape != (dv, dv):
            raise ValueError(f"Block {u} shape {tuple(CuC.shape)} != (dv,dv) with dv={dv}.")
        out_list.append(CuC)
    if _device is None:
        _device = torch.device("cpu")
    return out_list, dtype, _device


def blockdiag_mv(
    blocks: _CoinBlocks,
    x: torch.Tensor,
    pm: PortMap,
    *,
    degrees: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    y = ( ⊕_v C_v ) @ x

    Mixed-degree block-diagonal matvec applied to a state x laid out in the
    **arc (port) basis** grouped by CSR slices per source node.

    Parameters
    ----------
    blocks : list[Tensor(dv,dv)] or Tensor(N, dmax, dmax)
        Local coin matrices C_v (complex). May be list-like (ragged) or packed.
    x : (A,) or (B, A) complex
        State vector(s) in the arc basis, with arcs grouped by source node CSR.
    pm : PortMap
        Provides CSR grouping via node_ptr; A = total #arcs.
    degrees : optional LongTensor(N,)
        Per-node degrees; if None, deduced from pm.
    out : optional Tensor
        Optional output buffer (same shape/dtype/device as x).

    Returns
    -------
    y : Tensor with same shape as x

    Notes
    -----
    • Handles batched and unbatched x.
    • For packed blocks (N,dmax,dmax), we slice [:dv,:dv] for each node.
    • Complexity O(∑_v d_v^2) per vector (or per batch).
    """
    require_complex(x)
    N = pm.num_nodes
    ptr = pm.node_ptr
    A = int(ptr[-1].item())
    if x.size(-1) != A:
        raise ValueError(f"x last dim {x.size(-1)} != #arcs {A}.")

    deg = degrees if degrees is not None else degrees_from_portmap(pm)

    # Normalize blocks: complex dtype & device; shape compliance for list form
    blocks, _, dev = ensure_coin_blocks(blocks, deg, dtype=x.dtype, device=x.device)

    # Prepare output buffer
    y = out
    if y is None:
        y = torch.empty_like(x)

    # Fast paths for ranks
    batched = x.ndim == 2  # (B,A)
    if not batched and x.ndim != 1:
        raise ValueError("x must be shape (A,) or (B,A).")

    # Iterate nodes; tiny dv×dv dense multiplies over CSR slices
    if isinstance(blocks, torch.Tensor):
        # packed (N, dmax, dmax)
        for u in range(N):
            s = int(ptr[u])
            e = int(ptr[u + 1])
            dv = e - s
            if dv == 0:
                continue
            Cu = blocks[u, :dv, :dv]  # (dv,dv)
            if not Cu.is_complex():
                Cu = as_complex(Cu, dtype=x.dtype)
            if batched:
                # (B,dv) ← (B,dv) @ (dv,dv)^T? No, we want (B,dv_out) = (B,dv_in) @ Cu^T if x is row-major.
                # But x[..., s:e] is (B,dv) representing column vector per batch; we want y = Cu @ x_vec.
                # Use bmm by reshaping to (B,dv,1).
                x_slice = x[:, s:e]  # (B,dv)
                y[:, s:e] = (Cu @ x_slice.transpose(0, 1)).transpose(0, 1)
            else:
                x_slice = x[s:e]  # (dv,)
                y[s:e] = Cu @ x_slice
        return y

    # list-like blocks
    for u in range(N):
        s = int(ptr[u])
        e = int(ptr[u + 1])
        dv = e - s
        if dv == 0:
            continue
        Cu = blocks[u]
        if Cu.numel() == 0:
            # treat as identity on empty? nothing to do
            continue
        if batched:
            x_slice = x[:, s:e]  # (B,dv)
            y[:, s:e] = (Cu @ x_slice.transpose(0, 1)).transpose(0, 1)
        else:
            x_slice = x[s:e]  # (dv,)
            y[s:e] = Cu @ x_slice

    return y


# --------------------------------------------------------------------------- #
#                 Partial trace over coin → position marginal                 #
# --------------------------------------------------------------------------- #


def position_probabilities(
    psi: torch.Tensor,
    pm: PortMap,
    *,
    normalize: bool = True,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Position probabilities P(v) from an arc-basis state ψ.

    Theory
    ------
    Our state ψ is indexed by **oriented arcs** grouped in CSR by source node u.
    The coin space at vertex u is the set of its outgoing ports; therefore the
    position marginal is obtained by summing |ψ|^2 over the CSR slice of u.

    Parameters
    ----------
    psi : (A,) or (B, A) complex
        Statevector(s) in the arc basis.
    pm : PortMap
        Supplies `src` (bin ids) and `num_nodes`.
    normalize : bool
        If True, renormalize P so sum_v P(v) = 1 per batch (guards fp drift).
    eps : float
        Floor for the sum when normalizing.

    Returns
    -------
    P : (N,) or (B, N) real
        Position probabilities for each vertex.
    """
    require_complex(psi)
    src = pm.src  # (A,)
    N = pm.num_nodes

    if psi.ndim == 1:
        prob_arcs = (psi.conj() * psi).real  # (A,)
        P = segment_sum(prob_arcs, src, N)  # (N,)
        if normalize:
            s = P.sum().clamp_min(eps)
            P = P / s
        return P

    if psi.ndim == 2:
        prob_arcs = (psi.conj() * psi).real  # (B, A)
        P = segment_sum(prob_arcs, src, N)  # (B, N)
        if normalize:
            s = P.sum(dim=-1, keepdim=True).clamp_min(eps)
            P = P / s
        return P

    raise ValueError("psi must be rank 1 or 2 over the arc axis.")


def partial_trace_coin(
    psi: torch.Tensor,
    pm: PortMap,
    *,
    normalize: bool = True,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Alias for `position_probabilities`: Tr_coin{|ψ⟩⟨ψ|} projected on the position basis.

    Returns
    -------
    P : (N,) or (B, N) real
        Position marginal probabilities.
    """
    return position_probabilities(psi, pm, normalize=normalize, eps=eps)


# --------------------------------------------------------------------------- #
#                         Probability sanity check                            #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def check_probability_simplex(
    P: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> None:
    """
    Assert that P lies in a (batched) simplex: nonnegative, sums ~ 1.

    P: (N,) or (B, N) real
    """
    if P.ndim == 1:
        if (P < -1e-12).any():
            raise AssertionError("P has negative entries.")
        s = float(P.sum().item())
        if abs(s - 1.0) > atol:
            raise AssertionError(f"sum(P)={s} differs from 1 by > {atol}.")
        return

    if P.ndim == 2:
        if (P < -1e-12).any():
            raise AssertionError("P has negative entries.")
        s = P.sum(dim=-1)
        if not torch.allclose(s, torch.ones_like(s), atol=atol):
            raise AssertionError("Each batch row must sum to 1 within tolerance.")
        return

    raise ValueError("P must be rank 1 or 2.")


# --------------------------------------------------------------------------- #
#                                   Self-test                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    # Mixed-degree smoke tests on a tiny path graph: 1 -- 2 -- 3 with degrees (1,2,1)
    # Requires build_portmap / build_shift / step from simulator.
    import random

    torch.manual_seed(1234)
    random.seed(1234)

    from .portmap import build_portmap
    from .shift import build_shift
    from .step import step

    # Undirected path 1-2-3 → oriented arcs D=4
    pairs = [(0, 1), (1, 2)]
    pm = build_portmap(pairs, num_nodes=3, coalesce=False)
    N = pm.num_nodes
    A = pm.src.numel()

    # ---- Degrees & slices
    dv = degrees_from_portmap(pm)  # [1,2,1]
    slices = arc_slices_from_portmap(pm)

    # ---- Random complex state; renormalize
    psi = torch.randn(A, dtype=torch.complex64)
    psi = as_complex(psi)
    renorm_state_(psi)

    # ---- Position probabilities (partial trace)
    P0 = position_probabilities(psi, pm)
    check_probability_simplex(P0)

    # ---- Build identity coin per node (list form), then step → preserves P
    C_list = []
    for u in range(N):
        du = int(dv[u])
        C_list.append(torch.eye(du, dtype=torch.complex64))
    check_unitary_blocks(C_list, dv)
    S = build_shift(pm)
    psi1 = step(psi, pm, C_list, shift=S)
    P1 = position_probabilities(psi1, pm)
    check_probability_simplex(P1)
    assert torch.allclose(P0, P1, atol=1e-6), "Identity coins must preserve P across a shift."

    # ---- Use packed blocks (N,dmax,dmax): test blockdiag_mv consistency
    dmax = int(dv.max().item())
    C_packed = torch.zeros(N, dmax, dmax, dtype=torch.complex64)
    for u in range(N):
        du = int(dv[u])
        C_packed[u, :du, :du] = torch.eye(du, dtype=torch.complex64)
    y_list = blockdiag_mv(C_list, psi, pm)
    y_pack = blockdiag_mv(C_packed, psi, pm)
    assert torch.allclose(y_list, y_pack, atol=1e-7)

    # ---- Segment softmax sanity (batched & 1D)
    idx = pm.src  # (A,)
    s = torch.randn(A) * 3.0
    w = segment_softmax(s, idx, pm.num_nodes)
    for u in range(N):
        sl = slices[u]
        if sl.stop - sl.start > 0:
            su = w[sl].sum().item()
            assert abs(su - 1.0) < 1e-6

    sb = torch.randn(5, A) * 3.0
    wb = segment_softmax(sb, idx, pm.num_nodes)
    for b in range(5):
        for u in range(N):
            sl = slices[u]
            if sl.stop - sl.start > 0:
                su = wb[b, sl].sum().item()
                assert abs(su - 1.0) < 1e-6

    print("utils.py mixed-degree self-test passed.")

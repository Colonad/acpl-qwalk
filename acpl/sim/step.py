# acpl/sim/step.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import torch

from .portmap import PortMap
from .shift import ShiftOp, build_shift

__all__ = [
    "apply_coin_blockdiag",  # Phase A (deg==2) path (kept for BC)
    "apply_coin_blocks_general",  # Phase B1: ragged/mixed-degree coins
    "apply_then_shift",
    "step",
]

# --------------------------------------------------------------------------- #
#                                 Utilities                                   #
# --------------------------------------------------------------------------- #


def _ensure_complex(psi: torch.Tensor) -> None:
    """Raise if psi is not complex."""
    if not psi.is_complex():
        raise TypeError("psi must be complex64/complex128.")


def _same_lastdim(psi: torch.Tensor, A: int) -> None:
    if psi.shape[-1] != A:
        raise ValueError(f"psi last-dim must be {A} (got {psi.shape[-1]}).")


def _is_su2_stack(x: torch.Tensor) -> bool:
    return x.ndim == 3 and x.shape[-2:] == (2, 2)


def _is_blockdiag_matrix(x: torch.Tensor, A: int) -> bool:
    return x.is_complex() and x.ndim in (2, 3) and x.shape[-2:] == (A, A)


def _precompute_local_norms_1D(
    psi: torch.Tensor, pm: PortMap, nodes: Sequence[int]
) -> torch.Tensor:
    """Return per-node norms for a 1D state (A,)."""
    start = pm.node_ptr[:-1]
    end = pm.node_ptr[1:]
    norms = torch.zeros(len(nodes), dtype=psi.real.dtype, device=psi.device)
    for idx, u in enumerate(nodes):
        s, e = int(start[u]), int(end[u])
        if s == e:
            norms[idx] = 0.0
        else:
            seg = psi[s:e]
            norms[idx] = (seg.conj() * seg).real.sum()
    return norms


def _precompute_local_norms_2D(
    psi: torch.Tensor, pm: PortMap, nodes: Sequence[int]
) -> torch.Tensor:
    """Return per-node norms for a batched state (B, A)."""
    B = psi.shape[0]
    start = pm.node_ptr[:-1]
    end = pm.node_ptr[1:]
    norms = torch.zeros((B, len(nodes)), dtype=psi.real.dtype, device=psi.device)
    for j, u in enumerate(nodes):
        s, e = int(start[u]), int(end[u])
        if s == e:
            norms[:, j] = 0.0
        else:
            seg = psi[:, s:e]  # (B, deg)
            norms[:, j] = (seg.conj() * seg).real.sum(dim=-1)
    return norms


# --------------------------------------------------------------------------- #
#                      Phase A: deg-2 SU(2) stack application                 #
# --------------------------------------------------------------------------- #


def _validate_shapes_for_coins_su2(coins: torch.Tensor) -> None:
    """
    Phase A: coins are (N, 2, 2) complex blocks.
    We apply only where deg==2; other degrees receive identity.
    """
    if coins.ndim != 3 or coins.shape[-2:] != (2, 2):
        raise ValueError("coins must have shape (N, 2, 2) for Phase A.")
    if not coins.is_complex():
        raise TypeError("coins must be complex64/complex128.")


def apply_coin_blockdiag(
    psi: torch.Tensor,
    coins: torch.Tensor,
    pm: PortMap,
    *,
    out: torch.Tensor | None = None,
    check_local_norm: bool = False,
    atol: float = 1e-6,
) -> torch.Tensor:
    r"""
    Apply the block-diagonal **coin operator** \(\bigoplus_v C_v\) in the arc basis
    for the Phase-A case (per-vertex SU(2) and only for deg==2 slices).

    Contract
    --------
    - psi: (A,) or (B, A) complex — amplitudes on oriented arcs
    - coins: (N, 2, 2) complex — per-vertex SU(2) blocks (deg==2 only)
    - pm: supplies CSR layout (node_ptr) and degrees
    - returns: same shape as `psi`
    """
    _ensure_complex(psi)
    _validate_shapes_for_coins_su2(coins)

    A = pm.src.numel()
    _same_lastdim(psi, A)

    batched = psi.ndim == 2
    device = psi.device

    if out is None:
        out = torch.empty_like(psi)

    if A == 0:
        return psi if out is None else out.copy_(psi)

    start = pm.node_ptr[:-1]
    end = pm.node_ptr[1:]
    deg = pm.deg
    N = pm.num_nodes

    if not batched:
        out.copy_(psi)  # default identity
        if check_local_norm:
            pre_norm = _precompute_local_norms_1D(psi, pm, range(N))

        for u in range(N):
            if deg[u].item() != 2:
                continue
            s, e = int(start[u]), int(end[u])
            if (e - s) != 2:
                continue
            seg = psi[s:e]  # (2,)
            C = coins[u]  # (2,2)
            out[s:e] = C @ seg  # (2,)

        if check_local_norm:
            post_norm = _precompute_local_norms_1D(out, pm, range(N))
            # Only check deg==2 nodes
            mask = (deg == 2).cpu().numpy()
            pre = pre_norm[mask]
            post = post_norm[mask]
            if pre.numel() > 0 and not torch.allclose(post, pre, atol=atol, rtol=0):
                raise AssertionError("Local norm not preserved on some degree-2 nodes.")
        return out

    # Batched
    out.copy_(psi)
    if check_local_norm:
        pre_norm = _precompute_local_norms_2D(psi, pm, range(N))

    for u in range(N):
        if deg[u].item() != 2:
            continue
        s, e = int(start[u]), int(end[u])
        if (e - s) != 2:
            continue
        seg = psi[:, s:e]  # (B, 2)
        C = coins[u]  # (2, 2)
        out[:, s:e] = seg @ C.transpose(0, 1)

    if check_local_norm:
        post_norm = _precompute_local_norms_2D(out, pm, range(N))
        mask = (deg == 2).cpu().numpy()
        pre = pre_norm[:, mask]
        post = post_norm[:, mask]
        if pre.numel() > 0 and not torch.allclose(post, pre, atol=atol, rtol=0):
            raise AssertionError("Local norm not preserved at some degree-2 nodes (batched).")

    return out


# --------------------------------------------------------------------------- #
#                  Phase B1: Mixed-degree / general coin apply                #
# --------------------------------------------------------------------------- #

CoinsLike = Union[
    torch.Tensor,  # (N,2,2) or blockdiag (A,A) or batched (B,A,A)
    Sequence[torch.Tensor],  # list of length N with shapes (..., d_v, d_v)
]


def _apply_coin_blocks_list(
    psi: torch.Tensor,
    coins_list: Sequence[torch.Tensor],
    pm: PortMap,
    *,
    out: torch.Tensor | None = None,
    check_local_norm: bool = False,
    atol: float = 1e-6,
) -> torch.Tensor:
    """
    Apply per-node coins given as a ragged list of size N with shapes
    (..., d_v, d_v). Missing/mismatched shapes are treated as identity
    on that node slice (robust-by-default).
    """
    _ensure_complex(psi)

    A = pm.src.numel()
    _same_lastdim(psi, A)
    batched = psi.ndim == 2

    if out is None:
        out = torch.empty_like(psi)
    out.copy_(psi)

    start = pm.node_ptr[:-1]
    end = pm.node_ptr[1:]
    deg = pm.deg
    N = pm.num_nodes

    # Optional pre norms
    if check_local_norm:
        if batched:
            pre_norm = _precompute_local_norms_2D(psi, pm, range(N))
        else:
            pre_norm = _precompute_local_norms_1D(psi, pm, range(N))

    for u in range(N):
        s, e = int(start[u]), int(end[u])
        dv = e - s
        if dv <= 0:
            continue

        # Robust behavior: if we don't have a coin for this node, or its trailing
        # dims don't match dv, we skip (i.e., identity).
        if u >= len(coins_list):
            continue
        C = coins_list[u]
        if C.shape[-2:] != (dv, dv):
            continue

        if not batched:
            seg = psi[s:e]
            out[s:e] = C @ seg
        else:
            seg = psi[:, s:e]  # (B, dv)
            out[:, s:e] = seg @ C.transpose(-1, -2)  # (B, dv)

    # Optional post norms
    if check_local_norm:
        if batched:
            post_norm = _precompute_local_norms_2D(out, pm, range(N))
            if not torch.allclose(post_norm, pre_norm, atol=atol, rtol=0):
                raise AssertionError("Local norms changed for some nodes (batched).")
        else:
            post_norm = _precompute_local_norms_1D(out, pm, range(N))
            if not torch.allclose(post_norm, pre_norm, atol=atol, rtol=0):
                raise AssertionError("Local norms changed for some nodes.")

    return out


def _apply_coin_blockdiag_matrix(
    psi: torch.Tensor,
    Ct: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply a pre-built block-diagonal coin matrix Ct.

    Supports:
      • Ct: (A, A)   with psi: (A,) or (B, A)
      • Ct: (B, A, A) with psi: (B, A)

    We intentionally allow non-block-diagonal Ct as well; this is a general
    unitary on the arc space (e.g., for debugging/ablation).
    """
    _ensure_complex(psi)
    A = psi.shape[-1]

    if not _is_blockdiag_matrix(Ct, A):
        raise ValueError(f"Ct must be (A,A) or (B,A,A) with A={A}.")

    # (A,A) × (A,) -> (A,)
    if psi.ndim == 1 and Ct.ndim == 2:
        res = Ct @ psi
        if out is not None:
            out.copy_(res)
            return out
        return res

    # (B,A) with Ct either (A,A) (broadcast) or (B,A,A)
    if psi.ndim == 2:
        if Ct.ndim == 2:
            res = psi @ Ct.transpose(-1, -2)
        else:
            # Batched matmul
            res = torch.matmul(psi.unsqueeze(1), Ct.transpose(-1, -2)).squeeze(1)
        if out is not None:
            out.copy_(res)
            return out
        return res

    raise ValueError("psi must be rank-1 or rank-2 over the arc axis.")


def apply_coin_blocks_general(
    psi: torch.Tensor,
    coins: CoinsLike,
    pm: PortMap,
    *,
    out: torch.Tensor | None = None,
    check_local_norm: bool = False,
    atol: float = 1e-6,
) -> torch.Tensor:
    """
    Phase B1 coin application supporting several inputs:

    coins can be:
      1) Tensor with shape (N, 2, 2): Phase-A behavior (only deg==2 slices used).
      2) Sequence (list/tuple) of length N with per-node coins of shape
         (..., d_v, d_v). Ragged is supported. Identity used on mismatch.
      3) Pre-built (A, A) or (B, A, A) complex matrix `Ct` (block-diagonal or not).

    Returns a tensor with the same shape as psi.
    """
    _ensure_complex(psi)
    A = pm.src.numel()
    _same_lastdim(psi, A)

    # Case 3: global Ct (A,A) or (B,A,A)
    if isinstance(coins, torch.Tensor) and _is_blockdiag_matrix(coins, A):
        return _apply_coin_blockdiag_matrix(psi, coins, out=out)

    # Case 1: classic SU(2) stack
    if isinstance(coins, torch.Tensor) and _is_su2_stack(coins):
        return apply_coin_blockdiag(
            psi,
            coins,
            pm,
            out=out,
            check_local_norm=check_local_norm,
            atol=atol,
        )

    # Case 2: ragged list
    if isinstance(coins, (list, tuple)):
        return _apply_coin_blocks_list(
            psi,
            coins,
            pm,
            out=out,
            check_local_norm=check_local_norm,
            atol=atol,
        )

    raise TypeError(
        "Unsupported 'coins' type. Expected (N,2,2) tensor, list of per-node (d_v,d_v), "
        "or Ct matrix of shape (A,A)/(B,A,A)."
    )


# --------------------------------------------------------------------------- #
#                               Step operator                                 #
# --------------------------------------------------------------------------- #


def apply_then_shift(
    psi: torch.Tensor,
    pm: PortMap,
    coins: CoinsLike,
    *,
    shift: ShiftOp | None = None,
    out: torch.Tensor | None = None,
    check_local_norm: bool = False,
    atol: float = 1e-6,
) -> torch.Tensor:
    r"""
    Convenience façade:
        \(\psi_{t+1} \leftarrow S \cdot \Big(\bigoplus_v C_v(t)\Big)\, \psi_t\).

    Supports all Phase B1 coin formats (see `apply_coin_blocks_general`).
    """
    if shift is None:
        shift = build_shift(pm)

    tmp = apply_coin_blocks_general(
        psi,
        coins,
        pm,
        out=None,  # allocate fresh tmp (lets us share `out` for the final result)
        check_local_norm=check_local_norm,
        atol=atol,
    )

    # Flip–flop shift is an index permutation over the arc axis
    if tmp.ndim == 1:
        res = tmp.index_select(0, shift.perm)
    elif tmp.ndim == 2:
        res = tmp.index_select(1, shift.perm)
    else:
        raise ValueError("psi must be rank-1 or rank-2 over the arc axis.")

    if out is not None:
        out.copy_(res)
        return out
    return res


def step(
    psi: torch.Tensor,
    pm: PortMap,
    coins: CoinsLike,
    *,
    shift: ShiftOp | None = None,
    out: torch.Tensor | None = None,
    check_local_norm: bool = False,
    atol: float = 1e-6,
) -> torch.Tensor:
    r"""
    One DTQW step implementing the unitary:
        \( U_t = S \, C_t, \quad C_t = \bigoplus_v C_v(t) \).

    Args
    ----
    psi : (A,) or (B, A) complex
        Statevector(s) in the **arc basis**.
    pm : PortMap
        Flip–flop port mapping with CSR layout per vertex.
    coins : CoinsLike
        • (N, 2, 2) complex (Phase-A): apply only where deg==2.
        • sequence of per-node blocks (..., d_v, d_v) (ragged allowed).
        • Ct matrix (A, A) or (B, A, A) (block-diagonal or general).
    shift : ShiftOp, optional
        Reuse an existing built shift; otherwise constructed from `pm`.
    out : Tensor, optional
        Optional output buffer (same shape and dtype as `psi`).
    check_local_norm : bool
        If True, assert that the 2-norm of each CSR slice is preserved
        by its coin (debug aid). For Ct matrices, this is skipped (global check
        would be too costly here and is the caller's responsibility).
    atol : float
        Tolerance for local norm-preservation checks.

    Returns
    -------
    psi_next : same shape as `psi`
    """
    _ensure_complex(psi)
    return apply_then_shift(
        psi,
        pm,
        coins,
        shift=shift,
        out=out,
        check_local_norm=check_local_norm,
        atol=atol,
    )


# ------------------------------ Self-test ---------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    # Quick smoke: 4-cycle, deg==2 everywhere
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    from .coins import CoinsSpecSU2, coins_su2_from_theta
    from .portmap import build_portmap

    pm = build_portmap(pairs, num_nodes=4, coalesce=False)
    A = pm.src.numel()

    # Single state
    psi = torch.zeros(A, dtype=torch.complex64)
    psi[0] = 1  # unit amplitude on first arc
    theta = torch.randn(pm.num_nodes, 3)
    coins_su2 = coins_su2_from_theta(theta, spec=CoinsSpecSU2(normalize=True))

    # Phase-A compatibility
    S = build_shift(pm)
    psi_next = step(psi, pm, coins_su2, shift=S, check_local_norm=True)
    psi_next2 = apply_then_shift(psi, pm, coins_su2, shift=S, check_local_norm=True)
    assert torch.allclose(psi_next, psi_next2)

    # Batched
    psi_b = torch.randn(3, A, dtype=torch.complex64)
    psi_b_next = step(psi_b, pm, coins_su2, shift=S, check_local_norm=True)
    assert psi_b_next.shape == psi_b.shape

    # Phase B1: ragged coins (deg list may vary with graph)
    deg = (pm.node_ptr[1:] - pm.node_ptr[:-1]).tolist()
    # Make a simple identity + small phase per node
    coins_list: list[torch.Tensor] = []
    for d in deg:
        if d <= 0:
            coins_list.append(torch.zeros((0, 0), dtype=torch.complex64))
            continue
        eye = torch.eye(d, dtype=torch.complex64)
        # tiny skew-Hermitian diagonal -> unitary ~ exp(i*eps)
        eps = 1e-3
        phase = torch.exp(1j * torch.linspace(0, eps, d, dtype=torch.complex64))
        coins_list.append(eye * phase)

    psi_b2 = step(psi_b, pm, coins_list, shift=S)
    assert psi_b2.shape == psi_b.shape

    # Phase B1: provide a global Ct (A,A)
    Ct = torch.eye(A, dtype=torch.complex64)  # identity coin
    psi_b3 = step(psi_b, pm, Ct, shift=S)
    assert torch.allclose(psi_b3, psi_b.index_select(1, S.perm))

    print("step.py (Phase B1) self-tests passed.")

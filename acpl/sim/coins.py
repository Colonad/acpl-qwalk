# acpl/sim/coins.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("acpl.sim.coins requires PyTorch.") from e

from .portmap import PortMap

DeviceLike = torch.device | None
DTypeLike = torch.dtype | None
LiftKind = Literal["su2", "exp", "cayley"]
TorchSparseLayout = Literal["coo", "csr"]


# ---------------------------------------------------------------------------
# Graph-dependent layout: build once per graph, reuse every step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoinLayout:
    """
    Per-graph layout for applying vertex-local coin blocks over arc segments.

    For each vertex v:
      - outgoing arcs occupy a contiguous slice [arc_start[v], arc_end[v]) in the
        arc axis.
      - degree[v] = arc_end[v] - arc_start[v].

    Assumes PortMap groups arcs by tail (make_flipflop_portmap does this).
    """

    num_nodes: int
    num_arcs: int
    degree: torch.Tensor  # (V,) int64
    arc_start: torch.Tensor  # (V,) int64
    arc_end: torch.Tensor  # (V,) int64

    @staticmethod
    def from_portmap(pm: PortMap, *, device: DeviceLike = None) -> CoinLayout:
        v_count = pm.num_nodes
        a_count = pm.num_arcs
        node_ptr = torch.as_tensor(pm.node_ptr, dtype=torch.long, device=device)
        deg = (node_ptr[1:] - node_ptr[:-1]).to(torch.long)
        arc_start = node_ptr[:-1].clone()
        arc_end = node_ptr[1:].clone()
        return CoinLayout(
            num_nodes=v_count,
            num_arcs=a_count,
            degree=deg,
            arc_start=arc_start,
            arc_end=arc_end,
        )

    def unique_degrees(self) -> torch.Tensor:
        """Sorted unique degrees present in the graph (torch 1-D int64)."""
        return torch.unique(self.degree).cpu().to(torch.long)

    def mask_degree(self, d: int) -> torch.Tensor:
        """Boolean mask for vertices with degree == d (on the same device)."""
        return self.degree == int(d)

    def indices_degree(self, d: int) -> torch.Tensor:
        """Indices of vertices with degree == d."""
        return torch.nonzero(self.mask_degree(d), as_tuple=False).flatten()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _ensure_complex_dtype(dtype: DTypeLike) -> torch.dtype:
    if dtype is None:
        return torch.complex64
    if dtype not in (torch.complex64, torch.complex128):
        return torch.complex64
    return dtype


# ---------------------------------------------------------------------------
# SU(2) coins (degree == 2): Euler ZYZ parameterization (batched)
# ---------------------------------------------------------------------------


def rz(theta: torch.Tensor, *, dtype: DTypeLike = None) -> torch.Tensor:
    """
    Rz(theta) = [[e^{-i theta/2}, 0],
                 [0, e^{+i theta/2}]].

    Broadcasts over leading dims of theta. Returns (..., 2, 2).
    """
    ctype = _ensure_complex_dtype(dtype)
    theta = theta.to(torch.float32)
    half = 0.5 * theta
    im = torch.complex(torch.zeros_like(half), torch.ones_like(half))
    d0 = torch.exp(-im * half)
    d1 = torch.exp(+im * half)
    z = torch.zeros_like(d0, dtype=ctype)
    return torch.stack(
        [torch.stack([d0, z], dim=-1), torch.stack([z, d1], dim=-1)],
        dim=-2,
    ).to(ctype)


def ry(theta: torch.Tensor, *, dtype: DTypeLike = None) -> torch.Tensor:
    """
    Ry(theta) = [[ cos(theta/2), -sin(theta/2)],
                 [ sin(theta/2),  cos(theta/2)]].

    Broadcasts over leading dims. Returns (..., 2, 2) complex.
    """
    ctype = _ensure_complex_dtype(dtype)
    theta = theta.to(torch.float32)
    half = 0.5 * theta
    c = torch.cos(half)
    s = torch.sin(half)
    return torch.stack(
        [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
        dim=-2,
    ).to(ctype)


def su2_from_euler(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    dtype: DTypeLike = None,
) -> torch.Tensor:
    """
    U = Rz(alpha) @ Ry(beta) @ Rz(gamma).

    Broadcasts over leading dims; returns (..., 2, 2) complex.
    """
    ctype = _ensure_complex_dtype(dtype)
    rz_a = rz(alpha, dtype=ctype)
    ry_b = ry(beta, dtype=ctype)
    rz_c = rz(gamma, dtype=ctype)
    return rz_a @ ry_b @ rz_c


def coins_su2_from_theta(theta: torch.Tensor, *, dtype: DTypeLike = None) -> torch.Tensor:
    """
    Map Euler angles to SU(2) with batching.

    Parameters
    ----------
    theta : tensor of shape (..., N, 3) or (N, 3)
        Leading dims (optional) are treated as batch/time; N is number of
        selected vertices.

    Returns
    -------
    U : tensor of shape (..., N, 2, 2), complex
    """
    if theta.dim() < 2 or theta.size(-1) != 3:
        raise ValueError("theta must have shape (..., N, 3)")
    alpha, beta, gamma = theta.unbind(dim=-1)
    return su2_from_euler(alpha, beta, gamma, dtype=dtype)


# ---------------------------------------------------------------------------
# General d>2 coins: unitary lifts from unconstrained real parameters (batched)
# ---------------------------------------------------------------------------


def _symmetrize(m: torch.Tensor) -> torch.Tensor:
    return 0.5 * (m + m.transpose(-1, -2))


def _skew(m: torch.Tensor) -> torch.Tensor:
    return 0.5 * (m - m.transpose(-1, -2))


def skew_hermitian_from_params(params: torch.Tensor, d: int) -> torch.Tensor:
    r"""
    Build skew-Hermitian K from unconstrained real params.

    params : (..., 2, d, d) real
      params[..., 0, :, :] -> antisym (real part)
      params[..., 1, :, :] -> sym    (imag part)
    Returns K (..., d, d) complex with K^\dagger = -K.
    """
    if params.shape[-3:] != (2, d, d):
        raise ValueError(f"expected (..., 2, {d}, {d}), got {tuple(params.shape)}")
    r = params[..., 0, :, :]
    s = params[..., 1, :, :]
    re = _skew(r)
    im = _symmetrize(s)
    return torch.complex(re, im)


def coins_exp_from_params(
    params_by_deg: dict[int, torch.Tensor],
    *,
    dtype: DTypeLike = None,
) -> dict[int, torch.Tensor]:
    """
    params_by_deg[d]: (..., N_d, 2, d, d) real  →  U_by_deg[d]: (..., N_d, d, d) complex.

    Supports arbitrary leading batch dims.
    """
    ctype = _ensure_complex_dtype(dtype)
    out: dict[int, torch.Tensor] = {}
    for d, params in params_by_deg.items():
        if params.dim() < 4:
            raise ValueError("params must have shape (..., N_d, 2, d, d)")
        k = skew_hermitian_from_params(params, d)
        u = torch.linalg.matrix_exp(k.to(dtype=ctype))
        out[d] = u
    return out


def coins_cayley_from_params(
    params_by_deg: dict[int, torch.Tensor],
    *,
    dtype: DTypeLike = None,
    eps: float = 1e-6,
) -> dict[int, torch.Tensor]:
    r"""
    params_by_deg[d]: (..., N_d, 2, d, d) real  →  U_by_deg[d]: (..., N_d, d, d) complex.

    Uses batched linear solves for stability:
      U = (I - K)^{-1} (I + K), with K^\dagger = -K.
    """
    ctype = _ensure_complex_dtype(dtype)
    out: dict[int, torch.Tensor] = {}
    for d, params in params_by_deg.items():
        if params.dim() < 4:
            raise ValueError("params must have shape (..., N_d, 2, d, d)")
        k = skew_hermitian_from_params(params, d).to(dtype=ctype)
        eye = torch.eye(d, dtype=ctype, device=k.device)
        bmat = eye + k
        amat = eye - k
        u = torch.linalg.solve(amat, bmat)
        if not torch.isfinite(u).all():
            u = torch.linalg.solve(amat + eps * eye, bmat)
        out[d] = u
    return out


# ---------------------------------------------------------------------------
# High-level builders (build once per graph → reuse per step)
# ---------------------------------------------------------------------------


def build_coin_layout(pm: PortMap, *, device: DeviceLike = None) -> CoinLayout:
    return CoinLayout.from_portmap(pm, device=device)


def coins_from_theta(
    layout: CoinLayout,
    theta_su2: torch.Tensor | None = None,  # (..., V, 3)
    lift: LiftKind = "su2",
    params_by_deg: dict[int, torch.Tensor] | None = None,  # d>2: (..., N_d, 2, d, d)
    *,
    dtype: DTypeLike = None,
) -> dict[int, torch.Tensor]:
    """
    Produce per-degree stacks of unitary coin blocks aligned with vertices of
    that degree.

    - Phase A (only degree-2): pass theta_su2 (..., V, 3), lift='su2'
      → {2: (..., N2, 2, 2)}
    - Mixed degrees (Phase B): degree==2 via theta_su2; d>2 via lift in
      {'exp','cayley'} and params_by_deg.

    Returns dict d -> tensor of shape (..., N_d, d, d) complex.
    """
    ctype = _ensure_complex_dtype(dtype)
    out: dict[int, torch.Tensor] = {}

    # degree 2 via SU(2): select vertices of degree==2 in order
    if theta_su2 is not None:
        if theta_su2.dim() < 2 or theta_su2.size(-2) != layout.num_nodes or theta_su2.size(-1) != 3:
            raise ValueError(f"theta_su2 must have shape (..., V, 3); got {tuple(theta_su2.shape)}")
        d = 2
        idx = layout.indices_degree(d)
        if idx.numel() > 0:
            # Gather along V axis (penultimate dim)
            leading = theta_su2.shape[:-2]
            n2 = idx.numel()
            idx_expand = idx.view(*((1,) * len(leading)), n2, 1).expand(*leading, n2, 3)
            theta_sel = torch.take_along_dim(theta_su2, idx_expand, dim=-2)  # (..., n2, 3)
            u2 = coins_su2_from_theta(theta_sel, dtype=ctype)  # (..., n2, 2, 2)
            out[d] = u2

    # d>2 via lifts
    if params_by_deg:
        if lift == "exp":
            lift_out = coins_exp_from_params(params_by_deg, dtype=ctype)
        elif lift == "cayley":
            lift_out = coins_cayley_from_params(params_by_deg, dtype=ctype)
        elif lift == "su2":
            raise ValueError("lift='su2' only valid for d==2. Use 'exp' or 'cayley' for d>2.")
        else:
            raise ValueError("lift must be one of {'su2','exp','cayley'}")
        out.update(lift_out)

    return out


# ---------------------------------------------------------------------------
# Convenience helpers for ragged parameter provisioning (Phase B ergonomics)
# ---------------------------------------------------------------------------


def group_vertices_by_degree(layout: CoinLayout) -> dict[int, torch.Tensor]:
    """
    Return indices per degree: {d: idx_tensor}.

    Useful to pre-extract per-degree features or to align parameter batches.
    """
    groups: dict[int, torch.Tensor] = {}
    for d in layout.unique_degrees().tolist():
        groups[int(d)] = layout.indices_degree(int(d))
    return groups


def make_empty_params_by_deg(
    layout: CoinLayout,
    *,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> dict[int, torch.Tensor]:
    """
    Allocate an empty (zero) param dict for all degrees > 2 present in layout.

    Each tensor has shape (N_d, 2, d, d) real, ready to be filled by your
    policy head.
    """
    out: dict[int, torch.Tensor] = {}
    for d_t in layout.unique_degrees().tolist():
        d = int(d_t)
        if d <= 2:
            continue
        idx = layout.indices_degree(d)
        out[d] = torch.zeros((idx.numel(), 2, d, d), dtype=dtype, device=device)
    return out


# ---------------------------------------------------------------------------
# Safety checks (unitarity & shape assertions)
# ---------------------------------------------------------------------------


def unitary_error(u: torch.Tensor) -> torch.Tensor:
    """
    Return per-block max deviation ||U^†U - I||_max across all trailing matrix
    dims. If U has leading batch dims (..., N, d, d), returns tensor with shape
    (..., N).
    """
    if u.dim() < 2:
        raise ValueError("U must have at least 2 dims (..., d, d)")
    d = u.size(-1)
    eye = torch.eye(d, dtype=u.dtype, device=u.device)
    prod = u.conj().transpose(-1, -2) @ u
    err = (prod - eye).abs().amax(dim=(-1, -2))
    return err


def verify_unitary_blocks(u: torch.Tensor, atol: float = 1e-5) -> tuple[bool, torch.Tensor]:
    """
    Returns (ok, err) where err is per-block max deviation. ok=True iff all <= atol.
    """
    err = unitary_error(u)
    ok = bool(torch.all(err <= atol))
    return ok, err


def verify_coins_dict(
    layout: CoinLayout,
    u_by_deg: dict[int, torch.Tensor],
    atol: float = 1e-5,
) -> tuple[bool, str]:
    """
    Verify:
      - each degree d has shape (..., N_d, d, d),
      - blocks are unitary up to atol.
    """
    for d, u in u_by_deg.items():
        if u.size(-1) != d or u.size(-2) != d:
            return False, f"Degree {d}: expected (..., N_d, {d}, {d}), got {tuple(u.shape)}"
        # Infer N_d from layout
        n_d = int((layout.degree == d).sum().item())
        if u.size(-3) != n_d:
            return False, f"Degree {d}: expected N_d={n_d} blocks, got {u.size(-3)}"
        ok, err = verify_unitary_blocks(u, atol=atol)
        if not ok:
            worst = float(err.max().item())
            return False, f"Degree {d}: non-unitary coins, max ||U†U-I||={worst:.3e}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Debug/experiments: materialize block-diagonal C as a sparse matrix
# ---------------------------------------------------------------------------


def build_blockdiag_sparse_torch(
    layout: CoinLayout,
    u_by_deg: dict[int, torch.Tensor],
    *,
    device: DeviceLike = None,
    dtype: DTypeLike = None,
    layout_kind: TorchSparseLayout = "coo",
    default_identity_for_missing: bool = True,
    default_identity_for_deg1: bool = True,
) -> torch.Tensor:
    """
    Materialize the block-diagonal coin operator C over arc space as a sparse
    matrix (A x A).

    This is **for debugging/experiments only**. Prefer in-place block
    application in step.py for speed. Supports only a single snapshot (no
    leading batch dims).

    Missing-degree behavior mirrors `Stepper.apply_coins`:
      - If a degree d is not present in `u_by_deg` and
        `default_identity_for_missing=True`, an identity block of size d is
        inserted for each vertex of degree d.
      - If d==1 and `default_identity_for_deg1=True`, identity 1×1 blocks are
        inserted unless {1: (N1,1,1)} is explicitly provided.

    Returns
    -------
    torch.sparse.Tensor (COO or CSR) with complex entries forming block-diagonal
    structure.
    """
    ctype = _ensure_complex_dtype(dtype)
    a_count = int(layout.num_arcs)
    device = device or (next(iter(u_by_deg.values())).device if u_by_deg else torch.device("cpu"))

    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []

    # Iterate over *all* degrees present in the layout to mirror Stepper behavior
    for d_t in layout.unique_degrees().tolist():
        d = int(d_t)
        if d <= 0:
            continue

        idx = layout.indices_degree(d)
        n_d = idx.numel()
        if n_d == 0:
            continue

        # Decide which blocks to place for this degree
        if d in u_by_deg:
            u = u_by_deg[d]
            if u.dim() != 3 or u.size(0) != n_d or u.size(1) != d or u.size(2) != d:
                raise ValueError(
                    f"Degree {d}: expected U shape (N_d={n_d}, {d}, {d}); got {tuple(u.shape)}"
                )
            u_place = u.to(device=device, dtype=ctype)
        else:
            # Identity fallback logic
            if d == 1 and default_identity_for_deg1:
                eye = torch.eye(1, dtype=ctype, device=device).expand(n_d, 1, 1).clone()
                u_place = eye
            elif default_identity_for_missing:
                eye = torch.eye(d, dtype=ctype, device=device).expand(n_d, d, d).clone()
                u_place = eye
            else:
                # No blocks -> skip entirely (same as leaving zeros for those slices)
                continue

        # Place n_d blocks for all vertices of degree d
        for k, v in enumerate(idx.tolist()):
            start = int(layout.arc_start[v].item())
            end = int(layout.arc_end[v].item())
            if end - start != d:
                raise ValueError(f"Vertex {v}: degree mismatch; slice len {end - start} != d={d}")

            r_base = torch.arange(start, end, device=device)
            c_base = torch.arange(start, end, device=device)
            r = r_base[:, None].expand(d, d).reshape(-1)
            c = c_base[None, :].expand(d, d).reshape(-1)
            rows.append(r)
            cols.append(c)
            vals.append(u_place[k].reshape(-1))

    if not rows:
        if layout_kind == "coo":
            return torch.sparse_coo_tensor(size=(a_count, a_count), dtype=ctype, device=device)
        crow = torch.zeros(1, dtype=torch.int64, device=device)
        return torch.sparse_csr_tensor(crow, crow[:0], crow[:0].to(ctype), size=(a_count, a_count))

    rows_t = torch.cat(rows)
    cols_t = torch.cat(cols)
    vals_t = torch.cat(vals)

    if layout_kind == "coo":
        indices = torch.vstack((rows_t, cols_t))
        return torch.sparse_coo_tensor(
            indices, vals_t, size=(a_count, a_count), dtype=ctype, device=device
        ).coalesce()
    if layout_kind == "csr":
        coo = torch.sparse_coo_tensor(
            torch.vstack((rows_t, cols_t)),
            vals_t,
            size=(a_count, a_count),
            dtype=ctype,
            device=device,
        ).coalesce()
        return coo.to_sparse_csr()
    raise ValueError("layout_kind must be 'coo' or 'csr'")

# acpl/sim/coins.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch

__all__ = [
    # Phase A (kept for backward compatibility)
    "rz",
    "ry",
    "su2_from_euler",
    "su2_from_euler_batch",
    "coins_su2_from_theta",
    "is_unitary",
    "project_to_su2",
    "CoinsSpecSU2",
    # Phase B1 additions
    "gell_mann_basis",
    "skew_from_coeffs",
    "unitary_from_skew",
    "CoinsSpec",
    "coins_general_from_params",
    "coins_blockdiag_from_params",
]

# --------------------------------------------------------------------------- #
#                              dtype / device utils                           #
# --------------------------------------------------------------------------- #


def _to_complex_dtype(dtype: torch.dtype | None) -> torch.dtype:
    if dtype is None:
        return torch.complex64
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("coins require complex64 or complex128 dtype.")
    return dtype


def _infer_device(*tensors: torch.Tensor) -> torch.device:
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def _as_real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise TypeError("expected complex dtype")


# --------------------------------------------------------------------------- #
#                        SU(2) parameterization utilities                     #
# --------------------------------------------------------------------------- #
# Convention (ZYZ): U(α,β,γ) = Rz(α) · Ry(β) · Rz(γ)
# with
#   Rz(θ) = diag(e^{-iθ/2}, e^{+iθ/2})
#   Ry(θ) = [[ cos(θ/2), -sin(θ/2)],
#            [ sin(θ/2),  cos(θ/2)]]
#
# These choices guarantee det(U)=1 exactly in exact arithmetic and are
# differentiable everywhere. We also expose optional post-fixes to remove
# tiny numerical drift (norm and determinant projection).
# --------------------------------------------------------------------------- #


def rz(theta: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Z-axis SU(2) rotation Rz(theta) in 2×2 form (batched over theta).
    theta: (...,) real
    returns: (..., 2, 2) complex
    """
    dtype = _to_complex_dtype(dtype)
    device = theta.device
    half = 0.5 * theta
    # e^{-iθ/2}, e^{iθ/2}
    im = torch.tensor(1j, dtype=dtype, device=device)
    e_neg = torch.exp(-im * half)
    e_pos = torch.exp(+im * half)
    # Build 2×2 diagonal
    shape = theta.shape
    out = torch.zeros((*shape, 2, 2), dtype=dtype, device=device)
    out[..., 0, 0] = e_neg
    out[..., 1, 1] = e_pos
    return out


def ry(theta: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Y-axis SU(2) rotation Ry(theta) in 2×2 form (batched over theta).
    theta: (...,) real
    returns: (..., 2, 2) complex
    """
    dtype = _to_complex_dtype(dtype)
    device = theta.device
    half = 0.5 * theta
    c = torch.cos(half)
    s = torch.sin(half)
    shape = theta.shape
    out = torch.zeros((*shape, 2, 2), dtype=dtype, device=device)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out.to(dtype=dtype)


def su2_from_euler(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Build a single/batched SU(2) from ZYZ Euler angles.
    alpha,beta,gamma: broadcastable real tensors
    returns: (..., 2, 2) complex (unitary with det=1 up to fp error)
    """
    # Broadcast to common shape
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    dtype = _to_complex_dtype(dtype)
    # Rz(α) @ Ry(β) @ Rz(γ)
    U = rz(alpha, dtype=dtype) @ ry(beta, dtype=dtype) @ rz(gamma, dtype=dtype)
    if normalize:
        U = project_to_su2(U)
    return U


def su2_from_euler_batch(
    angles: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """
    ZYZ Euler → SU(2) lift (batched):

        angles: (..., 3) real  [alpha, beta, gamma]
        returns: (..., 2, 2) complex

    Notes
    -----
    • Supports arbitrary leading batch dims (T, N, ...) as long as the last dim is 3.
    • With normalize=True, we enforce det≈+1 via a global phase correction.
    """
    if angles.ndim < 2 or angles.size(-1) != 3:
        raise ValueError("angles must have shape (..., 3) -> [alpha, beta, gamma].")

    if dtype is None:
        dtype = torch.complex64

    # Split angles and broadcast
    alpha, beta, gamma = angles.unbind(dim=-1)  # each (...,)

    half = 0.5
    cb = torch.cos(half * beta)
    sb = torch.sin(half * beta)

    # Complex exponential helper: e^{i x} = cos x + i sin x
    def cexp(x: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.cos(x), torch.sin(x)).to(dtype)

    p = cexp(-half * (alpha + gamma))
    q = cexp(-half * (alpha - gamma))
    r = cexp(+half * (alpha - gamma))
    s = cexp(+half * (alpha + gamma))

    U00 = p * cb
    U01 = -q * sb
    U10 = r * sb
    U11 = s * cb

    # Stack into (..., 2, 2)
    U = torch.stack(
        [
            torch.stack([U00, U01], dim=-1),
            torch.stack([U10, U11], dim=-1),
        ],
        dim=-2,
    ).to(dtype)

    if normalize:
        # Phase-correct to SU(2): set det(U) ≈ 1 by applying a global phase factor
        det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]  # (...,)
        det_real = det.real
        det_imag = det.imag
        phase = torch.atan2(det_imag, det_real)  # angle in (-pi, pi]
        s_phase = torch.complex(torch.cos(-0.5 * phase), torch.sin(-0.5 * phase)).to(
            dtype
        )  # (...,)
        U = U * s_phase.unsqueeze(-1).unsqueeze(-1)

    return U


# --------------------------------------------------------------------------- #
#                           Robustness / validation                           #
# --------------------------------------------------------------------------- #


def is_unitary(U: torch.Tensor, *, atol: float = 1e-6) -> bool:
    """
    Check unitarity on the trailing 2×2 (or d×d) blocks. Supports shapes (..., d, d).
    """
    if U.ndim < 2 or U.shape[-1] != U.shape[-2]:
        raise ValueError("U must end with a square matrix (..., d, d).")
    d = U.shape[-1]
    eye = torch.eye(d, dtype=U.dtype, device=U.device)
    UhU = U.conj().transpose(-1, -2) @ U
    diff = (UhU - eye).abs().amax()
    return bool(diff <= atol)


def _det2x2(U: torch.Tensor) -> torch.Tensor:
    # det for trailing 2×2
    return U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]


def project_to_su2(U: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Project a near-SU(2) matrix onto SU(2):
      1) Unitarize via (right) polar factor: U ← U (U†U)^{-1/2}
      2) Normalize determinant to 1: U ← U / sqrt(det(U))
    """
    if U.ndim < 2 or U.shape[-2:] != (2, 2):
        raise ValueError("U must end with shape (2, 2).")
    # Right polar factor: U = QH, with Q unitary. Compute H = (U†U)^{1/2}.
    G = U.conj().transpose(-1, -2) @ U  # Hermitian pos.def (near)
    evals, evecs = torch.linalg.eigh(G)
    evals = torch.clamp(evals, min=eps)
    diag_rs = torch.diag_embed(evals.rsqrt()).to(evecs.dtype)
    Hm12 = evecs @ diag_rs @ evecs.conj().transpose(-1, -2)

    Q = U @ Hm12
    # Fix determinant phase to exactly 1
    detQ = _det2x2(Q)
    det_sqrt = torch.sqrt(detQ)  # complex principal sqrt
    Q = Q / det_sqrt.unsqueeze(-1).unsqueeze(-1)
    return Q


# --------------------------------------------------------------------------- #
#                              Coins (degree-2)                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CoinsSpecSU2:
    """
    Spec for building vertex-local SU(2) coins (degree-2 vertices).
    """

    dtype: torch.dtype = torch.complex64
    normalize: bool = False  # project to SU(2) to remove tiny drift
    check: bool = False  # assert unitarity in debug


def coins_su2_from_theta(
    theta_vt: torch.Tensor,
    *,
    spec: CoinsSpecSU2 | None = None,
) -> torch.Tensor:
    """
    Build per-vertex 2×2 coin blocks from Euler angles.

    Parameters
    ----------
    theta_vt : (N, 3) real
        Per-vertex angles [alpha, beta, gamma] in ZYZ convention.
    spec : CoinsSpecSU2
        dtype: complex64/complex128
        normalize: if True, project to exact SU(2) (polar + det fix)
        check: if True, assert unitarity for debugging

    Returns
    -------
    coins : (N, 2, 2) complex
        Vertex-local unitaries suitable for block-diagonal application.
    """
    if spec is None:
        spec = CoinsSpecSU2()
    if theta_vt.ndim != 2 or theta_vt.size(-1) != 3:
        raise ValueError("theta_vt must have shape (N, 3) = [alpha, beta, gamma].")
    if not theta_vt.is_floating_point():
        theta_vt = theta_vt.float()

    U = su2_from_euler_batch(theta_vt, dtype=spec.dtype, normalize=spec.normalize)

    if spec.check:
        if not is_unitary(U, atol=1e-6):
            raise AssertionError("coins_su2_from_theta produced a non-unitary block.")

        # Determinant check (tolerant)
        detU = _det2x2(U)
        # |det - 1| may have tiny float error; check magnitude ≈ 1 (SU(2) has det exactly 1)
        if (detU.abs() - 1).abs().amax().item() > 1e-5:
            raise AssertionError("coins_su2_from_theta produced det far from 1.")

    return U


# --------------------------------------------------------------------------- #
#                     Phase B1: General U(d) / SU(d) coins                     #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=64)
def gell_mann_basis(
    d: int, *, include_identity: bool = True, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Return an orthonormal (w.r.t. Tr(A B)) Hermitian basis of R^{dxd}:

    • Off-diagonal symmetric    S_{ij} = (E_{ij} + E_{ji})/sqrt(2)
    • Off-diagonal antisymm.    A_{ij} = -i(E_{ij} - E_{ji})/sqrt(2)  (but *Hermitian*: multiply by i later)
    • Diagonal (traceless)      D_k   = diag(1,...,1,-k,0,...,0)/sqrt(k(k+1)) for k=1..d-1
    • Optional identity         I / sqrt(d)

    Basis is stacked as a tensor of shape (q, d, d) with q = d^2 (if include_identity) else d^2 - 1.
    All tensors are REAL Hermitian (imag=0); skew-Hermitian construction multiplies by i later.
    """
    if d < 1:
        raise ValueError("d must be >= 1")
    device = torch.device("cpu")
    # Work in real dtype; callers can .to(complex) later
    real = dtype
    mats: list[torch.Tensor] = []

    # Off-diagonal
    for i in range(d):
        for j in range(i + 1, d):
            S = torch.zeros((d, d), dtype=real, device=device)
            S[i, j] = 1.0 / (2.0**0.5)
            S[j, i] = 1.0 / (2.0**0.5)
            mats.append(S)

            A = torch.zeros((d, d), dtype=real, device=device)
            A[i, j] = -1.0 / (2.0**0.5)
            A[j, i] = +1.0 / (2.0**0.5)
            # Note: A is real anti-symmetric; when multiplied by i it becomes Hermitian.
            mats.append(A)

    # Diagonal (traceless)
    for k in range(1, d):
        D = torch.zeros((d, d), dtype=real, device=device)
        D[:k, :k] = D[:k, :k] + torch.eye(k, dtype=real)
        D[k, k] = -float(k)
        D /= (k * (k + 1)) ** 0.5
        mats.append(D)

    # Identity
    if include_identity:
        I = torch.eye(d, dtype=real, device=device) / (d**0.5)
        mats.append(I)

    return torch.stack(mats, dim=0)  # (q, d, d)


def _basis_size_for_group(d: int, group: str, *, use_identity: bool) -> int:
    if group.upper() == "SU":
        return d * d - 1
    if group.upper() == "U":
        return d * d if use_identity else d * d - 1
    raise ValueError("group must be 'U' or 'SU'")


def skew_from_coeffs(
    theta: torch.Tensor,
    *,
    d: int,
    group: str = "U",
    basis: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Build a skew-Hermitian K from real coefficients and a Hermitian basis:
        K = i * sum_k theta_k G_k

    Args
    ----
    theta : (..., q) real
        Coefficients; q must match the chosen basis size for d and group.
    d : int
        Matrix size.
    group : 'U' or 'SU'
        If 'SU', the (traceless) basis is used (no identity direction).
    basis : (q, d, d) real Hermitian basis
        If None, use Gell–Mann (+identity if group='U').
    dtype : complex dtype for the output (default complex64).

    Returns
    -------
    K : (..., d, d) complex skew-Hermitian
    """
    if dtype is None:
        dtype = torch.complex64
    real_dtype = _as_real_dtype(dtype)

    use_identity = group.upper() == "U"
    if basis is None:
        basis = gell_mann_basis(d, include_identity=use_identity, dtype=real_dtype)
    q = basis.size(0)

    if theta.size(-1) != q:
        raise ValueError(f"theta last dim must be {q} for d={d}, group={group}")

    # Contract: sum_k theta_k G_k
    # theta: (..., q), basis: (q, d, d) -> (..., d, d)
    H = torch.tensordot(theta.to(real_dtype), basis.to(real_dtype), dims=([-1], [0]))
    # Make Hermitian exactly (protect tiny numerical drift)
    H = 0.5 * (H + H.transpose(-1, -2))

    # Skew-Hermitian: K = i * H
    i = torch.tensor(1j, dtype=dtype, device=H.device)
    return (i * H.to(dtype)).to(dtype)


def _cayley_unitary_from_skew(K: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    """
    Cayley retraction:
        U = (I - K) (I + K)^{-1}
    For skew-Hermitian K, U is unitary whenever I + K is invertible.
    """
    d = K.shape[-1]
    I = torch.eye(d, dtype=K.dtype, device=K.device)
    A = I - K
    B = I + K
    # Stabilize inversion slightly
    if eps is not None and eps > 0:
        B = B + eps * I
    U = torch.linalg.solve(B, A)  # (I+K)^{-1}(I-K)
    return U


def unitary_from_skew(
    K: torch.Tensor,
    *,
    method: str = "expm",
) -> torch.Tensor:
    """
    Map a skew-Hermitian K to a unitary:
        method='expm'  -> U = expm(K)
        method='cayley'-> U = (I-K)(I+K)^{-1}
    """
    method = method.lower()
    if method == "expm":
        return torch.linalg.matrix_exp(K)
    if method == "cayley":
        return _cayley_unitary_from_skew(K)
    raise ValueError("method must be 'expm' or 'cayley'")


# --------------------------------------------------------------------------- #
#                       Public spec (general-degree coins)                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CoinsSpec:
    """
    General coin-construction spec.

    • Supports U(d) or SU(d) via a Hermitian generator basis and an exponential
      or Cayley map.
    • Works per node with arbitrary degrees; returns per-node unitary blocks
      or a global block-diagonal.

    Fields
    ------
    group : 'U' or 'SU'
        Which unitary group to realize (trace direction kept for 'U').
    method : 'expm' or 'cayley'
        Skew-Hermitian → unitary map (matrix exponential vs Cayley retraction).
    dtype : torch.complex64 or complex128
        Complex working dtype for unitary matrices.
    check : bool
        If True, assert unitarity (debugging).
    """

    group: str = "U"
    method: str = "expm"
    dtype: torch.dtype = torch.complex64
    check: bool = False


def _ensure_list_params(
    params: Sequence[torch.Tensor] | torch.Tensor, N: int | None = None
) -> list[torch.Tensor]:
    """
    Accept a list of per-node parameter tensors or a single (N, q) tensor.
    Returns a list of length N with shape (..., q_v) per node.

    (This intentionally supports ragged q_v via a list; a single tensor implies
    the same q for all nodes.)
    """
    if isinstance(params, torch.Tensor):
        if params.ndim < 2:
            raise ValueError("Tensor params must have shape (N, q)")
        if N is None:
            N = params.shape[-2]
        out = [params[..., i, :] for i in range(N)]
        return out
    else:
        return list(params)


def _coins_from_params_single(
    theta_v: torch.Tensor,
    d_v: int,
    *,
    spec: CoinsSpec,
    basis: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Map a single node's parameter vector theta_v (..., q_v) to U(d_v).
    """
    # theta_v may have batch dims; skew_from_coeffs broadcasts
    K = skew_from_coeffs(theta_v, d=d_v, group=spec.group, basis=basis, dtype=spec.dtype)
    U = unitary_from_skew(K, method=spec.method)
    if spec.check:
        if not is_unitary(U):
            raise AssertionError("Constructed coin is not unitary.")
        if spec.group.upper() == "SU":
            # For SU(d), det should be ~1 (up to phase drift from numerics with Cayley).
            det = torch.linalg.det(U)
            if (det.abs() - 1).abs().amax().item() > 1e-4:
                # Numerically loud tolerances—Cayley may pick a branch; acceptable in practice.
                raise AssertionError("SU(d) determinant deviates significantly from 1.")
    return U


def coins_general_from_params(
    params_per_node: Sequence[torch.Tensor] | torch.Tensor,
    degrees: Sequence[int] | torch.Tensor,
    *,
    spec: CoinsSpec | None = None,
    share_basis_across_nodes: bool = True,
) -> list[torch.Tensor]:
    """
    Build per-node U(d_v)/SU(d_v) coins for mixed degrees.

    Parameters
    ----------
    params_per_node : list[Tensor] or (N, q) Tensor
        Either a list of length N with shapes (..., q_v) (ragged friendly),
        or a single dense tensor (..., N, q) meaning all nodes use the same q.
        Coefficients are interpreted against a Hermitian basis for each node.
    degrees : list[int] or (N,) Tensor
        Per-node coin dimension d_v (i.e., vertex degree).
    spec : CoinsSpec
        Group (U/SU), mapping method (expm/cayley), dtype, and integrity checks.
    share_basis_across_nodes : bool
        If True (default), cache/reuse the same basis per degree value to avoid
        re-allocations; safe because the basis is canonical.

    Returns
    -------
    coins : list[Tensor]
        List of length N of arrays with shape (..., d_v, d_v), complex dtype.
    """
    if spec is None:
        spec = CoinsSpec()
    if isinstance(degrees, torch.Tensor):
        deg_list = [int(x) for x in degrees.tolist()]
    else:
        deg_list = list(map(int, degrees))

    # Normalize params structure to a list
    theta_list = _ensure_list_params(params_per_node, N=len(deg_list))
    if len(theta_list) != len(deg_list):
        raise ValueError("params_per_node and degrees must have the same length N.")

    coins: list[torch.Tensor] = []
    # Prepare per-degree basis if sharing
    cached_basis: dict[int, torch.Tensor] = {}
    for theta_v, d_v in zip(theta_list, deg_list, strict=False):
        if d_v <= 0:
            raise ValueError("degrees must be positive")

        basis = None
        if share_basis_across_nodes:
            if d_v not in cached_basis:
                include_id = spec.group.upper() == "U"
                real_dtype = _as_real_dtype(spec.dtype)
                cached_basis[d_v] = gell_mann_basis(
                    d_v, include_identity=include_id, dtype=real_dtype
                )
            basis = cached_basis[d_v]

        U = _coins_from_params_single(theta_v, d_v, spec=spec, basis=basis)
        coins.append(U)

    return coins


def _block_diag_stack(blocks: list[torch.Tensor]) -> torch.Tensor:
    """
    Assemble a block-diagonal matrix from a list of square matrices with shared
    leading batch shape.

    Input: blocks = [B0, B1, ..., B_{N-1}], each with shape (..., d_i, d_i)
    Output: (..., D, D) where D = sum_i d_i

    This is batch-friendly (but loops over batch dims to keep memory modest).
    """
    if len(blocks) == 0:
        raise ValueError("blocks must be non-empty")
    # Infer leading batch shape and device/dtype
    lead_shape = torch.broadcast_shapes(*[b.shape[:-2] for b in blocks])
    device = blocks[0].device
    dtype = blocks[0].dtype
    d_list = [b.shape[-1] for b in blocks]
    D = sum(d_list)

    # Flatten batch dims to iterate; re-stack at the end
    B = int(torch.tensor(lead_shape).prod().item()) if len(lead_shape) > 0 else 1

    # Prepare expanded views
    expanded = [b.expand(lead_shape + b.shape[-2:]) for b in blocks]
    out_list: list[torch.Tensor] = []

    for flat_i in range(B):
        # Index into each expanded block
        idx = []
        if len(lead_shape) > 0:
            # Convert flat_i to multi-index
            rem = flat_i
            for s in reversed(lead_shape):
                idx.append(rem % s)
                rem //= s
            idx = list(reversed(idx))
            index = tuple(idx)
        else:
            index = ()

        rows = []
        for b in expanded:
            rows.append(b[index])  # (d_i, d_i)
        out_list.append(torch.block_diag(*rows))  # (D, D)

    out = torch.stack(out_list, dim=0).to(dtype=dtype, device=device)  # (B, D, D)
    if len(lead_shape) == 0:
        return out[0]
    return out.view(lead_shape + (D, D))


def coins_blockdiag_from_params(
    params_per_node: Sequence[torch.Tensor] | torch.Tensor,
    degrees: Sequence[int] | torch.Tensor,
    *,
    spec: CoinsSpec | None = None,
    share_basis_across_nodes: bool = True,
) -> torch.Tensor:
    """
    Convenience wrapper that returns a single block-diagonal Ct suitable for
    one QW step on the global arc space.

    Returns
    -------
    Ct : (..., D, D) complex
        Block-diagonal concatenation of the N local coins, with
        D = sum_v d_v.
    """
    coins = coins_general_from_params(
        params_per_node,
        degrees,
        spec=spec,
        share_basis_across_nodes=share_basis_across_nodes,
    )
    return _block_diag_stack(coins)


# --------------------------------------------------------------------------- #
#                                   Self-test                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    torch.manual_seed(0)

    # --- SU(2) quick test (Phase A compatibility) ---
    N = 5
    th = torch.randn(N, 3)
    U = coins_su2_from_theta(
        th, spec=CoinsSpecSU2(dtype=torch.complex64, normalize=True, check=True)
    )
    assert U.shape == (N, 2, 2)
    eye2 = torch.eye(2, dtype=U.dtype)
    err = (U.conj().transpose(-1, -2) @ U - eye2).abs().amax().item()
    print("[SU2] max unitary error:", err)

    # --- General coins: assorted degrees with Cayley map ---
    degs = [2, 3, 4, 1]
    spec = CoinsSpec(group="U", method="cayley", dtype=torch.complex64, check=True)

    # Build per-node parameter lists with correct q_v each
    params: list[torch.Tensor] = []
    for d in degs:
        include_id = spec.group.upper() == "U"
        q = _basis_size_for_group(d, spec.group, use_identity=include_id)
        params.append(torch.randn(q))  # no batch dims; per-node vector

    coins = coins_general_from_params(params, degs, spec=spec)
    for Uv, d in zip(coins, degs, strict=False):
        assert Uv.shape == (d, d)
        assert is_unitary(Uv)

    Ct = coins_blockdiag_from_params(params, degs, spec=spec)
    D = sum(degs)
    assert Ct.shape == (D, D)
    eyeD = torch.eye(D, dtype=Ct.dtype)
    print("[Ud] blockdiag max error:", (Ct.conj().T @ Ct - eyeD).abs().amax().item())

    print("ok")

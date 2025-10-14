# acpl/sim/coins.py
from __future__ import annotations

try:  # Torch is required for simulator/backprop, but keep import guarded.
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("acpl.sim.coins requires PyTorch to be installed.") from exc


# ---------------------------------------------------------------------------
# DType helpers
# ---------------------------------------------------------------------------


def _complex_for(dtype: torch.dtype) -> torch.dtype:
    """
    Map a real dtype to its complex counterpart.
    float32 -> complex64, float64 -> complex128.
    """
    if dtype in (torch.float32, torch.complex64):
        return torch.complex64
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    # Fallback: prefer complex64
    return torch.complex64


def _canonical_device_dtype(
    *tensors: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.device, torch.dtype]:
    """
    Infer (device, dtype) from provided tensors, optionally overridden by args.
    If no tensors are given, fall back to (cpu, float32).
    """
    dev = device
    dt = dtype
    for t in tensors:
        if dev is None:
            dev = t.device
        if dt is None:
            dt = t.dtype
    if dev is None:
        dev = torch.device("cpu")
    if dt is None:
        dt = torch.float32
    return dev, dt


# ---------------------------------------------------------------------------
# Elementary rotations
# ---------------------------------------------------------------------------


def rz(
    theta: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""
    Z-axis rotation:
        Rz(θ) = [[e^{-i θ/2}, 0],
                 [0, e^{+i θ/2}]]
    Supports broadcasting over θ; returns tensor of shape (..., 2, 2).
    """
    dev, real_dt = _canonical_device_dtype(theta, device=device, dtype=dtype)
    cdt = _complex_for(real_dt)

    half = 0.5 * theta.to(device=dev, dtype=real_dt)
    # complex exponentials (broadcasted)
    exp_pos = torch.exp(1j * half.to(dtype=cdt))
    exp_neg = torch.exp(-1j * half.to(dtype=cdt))

    m = torch.zeros(*half.shape, 2, 2, device=dev, dtype=cdt)
    m[..., 0, 0] = exp_neg
    m[..., 1, 1] = exp_pos
    return m


def ry(
    theta: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""
    Y-axis rotation:
        Ry(θ) = [[ cos(θ/2), -sin(θ/2)],
                 [ sin(θ/2),  cos(θ/2)]]
    Supports broadcasting over θ; returns tensor of shape (..., 2, 2).
    """
    dev, real_dt = _canonical_device_dtype(theta, device=device, dtype=dtype)
    half = 0.5 * theta.to(device=dev, dtype=real_dt)
    c = torch.cos(half)
    s = torch.sin(half)

    # Build in real, upcast to matching complex type for consistency with rz/su2
    m_real = torch.zeros(*half.shape, 2, 2, device=dev, dtype=real_dt)
    m_real[..., 0, 0] = c
    m_real[..., 0, 1] = -s
    m_real[..., 1, 0] = s
    m_real[..., 1, 1] = c
    return m_real.to(dtype=_complex_for(real_dt))


# ---------------------------------------------------------------------------
# ZYZ Euler map into SU(2)
# ---------------------------------------------------------------------------


def su2_from_euler(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""
    Batched SU(2) from ZYZ Euler angles:

        C(α,β,γ) = Rz(α) @ Ry(β) @ Rz(γ)

    All inputs broadcast to a common shape (...), result has shape (..., 2, 2).
    """
    dev, real_dt = _canonical_device_dtype(alpha, beta, gamma, device=device, dtype=dtype)

    # Squeeze *only* redundant singleton dims to avoid spurious leading 1s,
    # e.g., (5,), (1,), (1,1) -> (5,) instead of (1,5).
    alpha = alpha.to(device=dev, dtype=real_dt).squeeze()
    beta = beta.to(device=dev, dtype=real_dt).squeeze()
    gamma = gamma.to(device=dev, dtype=real_dt).squeeze()

    rz_a = rz(alpha, device=dev, dtype=real_dt)  # (..., 2, 2)
    ry_b = ry(beta, device=dev, dtype=real_dt)  # (..., 2, 2)
    rz_g = rz(gamma, device=dev, dtype=real_dt)  # (..., 2, 2)

    # Batched matmul with broadcasting over the (possibly) common shape.
    cg = torch.matmul(ry_b, rz_g)
    c = torch.matmul(rz_a, cg)
    return c


# ---------------------------------------------------------------------------
# Phase-A API: per-node SU(2) blocks from theta_vt
# ---------------------------------------------------------------------------


def coins_su2_from_theta(
    theta_vt: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""
    Map per-node Euler triples θ = (α, β, γ) to SU(2) coin blocks.

    Parameters
    ----------
    theta_vt : torch.Tensor, shape (V, 3)
        Each row is (alpha, beta, gamma) for a vertex.
    device, dtype : optional
        Overrides for device / real part dtype (float32/float64). The output
        complex dtype is chosen accordingly (complex64/complex128).

    Returns
    -------
    C : torch.Tensor, shape (V, 2, 2), complex
        Per-node unitary blocks suitable for a block-diagonal coin operator.

    Notes
    -----
    - This function is autograd-friendly; gradients flow through Euler angles.
    - Use with `step.py` to assemble blkdiag(C) per vertex.
    """
    if theta_vt.ndim != 2 or theta_vt.size(-1) != 3:
        raise ValueError("theta_vt must have shape (V, 3) with Euler angles (alpha, beta, gamma).")

    dev, real_dt = _canonical_device_dtype(theta_vt, device=device, dtype=dtype)
    theta_vt = theta_vt.to(device=dev, dtype=real_dt)

    alpha = theta_vt[:, 0]
    beta = theta_vt[:, 1]
    gamma = theta_vt[:, 2]
    c = su2_from_euler(alpha, beta, gamma, device=dev, dtype=real_dt)  # (V, 2, 2)
    return c


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def is_unitary(u: torch.Tensor, atol: float = 1e-6) -> bool:
    """
    Check unitarity: U^† U ≈ I for batched or single 2x2 matrices.

    Parameters
    ----------
    u : torch.Tensor
        Shape (..., 2, 2), complex dtype preferred.
    atol : float
        Absolute tolerance for closeness.

    Returns
    -------
    bool
        True if all slices are approximately unitary.
    """
    if u.ndim < 2 or u.shape[-2:] != (2, 2):
        return False
    cdt = u.dtype if u.is_complex() else _complex_for(u.dtype)
    u = u.to(dtype=cdt)
    eye = torch.eye(2, dtype=cdt, device=u.device)
    # (..., 2, 2)
    prod = u.conj().transpose(-2, -1) @ u
    diff = prod - eye
    # Reduce over matrix dims
    err = diff.abs().amax(dim=(-2, -1))
    return bool((err <= atol).all())


# ---------------------------------------------------------------------------
# Mixed-degree coins for d > 2 via skew-Hermitian lifts
#   - We use real skew-symmetric K (K^T = -K). This is skew-Hermitian as well,
#     so exp(K) is unitary (real orthogonal), and Cayley(K) is unitary too.
#   - Per-vertex degrees can differ; we slice the appropriate number of params
#     per vertex: P(d) = d * (d - 1) // 2, filling the strict upper triangle.
# ---------------------------------------------------------------------------


def _num_params_skew(d: int) -> int:
    """Number of free params to define a dxd real skew-symmetric matrix (strict upper triangle)."""
    return (d * (d - 1)) // 2


def _skew_from_params(params: torch.Tensor, d: int) -> torch.Tensor:
    """
    Build a real skew-symmetric matrix K (d x d) from a flat parameter tensor
    of length P(d) = d*(d-1)//2 that fills the strict upper triangle, and
    mirrors to the lower triangle with a negative sign.
    """
    if params.numel() != _num_params_skew(d):
        raise ValueError(
            f"Expected {_num_params_skew(d)} params for d={d}, got {int(params.numel())}."
        )
    k = torch.zeros(d, d, dtype=params.dtype, device=params.device)
    # Fill strict upper triangle in row-major order
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            k[i, j] = params[idx]
            k[j, i] = -params[idx]
            idx += 1
    return k


def _per_vertex_params(
    theta: torch.Tensor,
    degrees: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Slice per-vertex parameter vectors (length P(d_v)) out of a (V, P_max) tensor `theta`,
    where P_max >= max_v P(d_v). Returns a list of 1-D tensors, one per vertex.
    """
    if theta.ndim != 2:
        raise ValueError("theta must have shape (V, P_max).")
    if degrees.ndim != 1 or degrees.size(0) != theta.size(0):
        raise ValueError("degrees must be a 1-D tensor with length V == theta.size(0).")
    v, pmax = theta.shape
    out: list[torch.Tensor] = []
    for i in range(v):
        d = int(degrees[i].item())
        need = _num_params_skew(d)
        if need > pmax:
            raise ValueError(
                f"theta has only {pmax} columns but vertex {i} with degree d={d} "
                f"requires {need} parameters."
            )
        out.append(theta[i, :need])
    return out


def coins_exp_from_skewk(
    theta: torch.Tensor,
    degrees: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> list[torch.Tensor]:
    r"""
    Mixed-degree coin blocks via matrix exponential of a skew-symmetric generator.

    Parameters
    ----------
    theta : torch.Tensor, shape (V, P_max)
        Row i contains parameters for vertex i. For degree d_v, we use the first
        P(d_v) = d_v(d_v-1)/2 entries to fill the strict upper triangle of K_v.
    degrees : torch.Tensor, shape (V,)
        Degree d_v for each vertex (integer dtype).
    device, dtype : optional
        Real dtype and device to place `theta`. Complex output dtype follows _complex_for().

    Returns
    -------
    list[torch.Tensor]
        A list of length V, where item i is a complex tensor of shape (d_v, d_v)
        representing unitary coin block for vertex i.

    Notes
    -----
    - Using real skew-symmetric K yields an orthogonal (thus unitary) matrix via exp(K).
    - This parameterization is stable for gradient-based training.
    """
    dev, real_dt = _canonical_device_dtype(theta, degrees, device=device, dtype=dtype)
    theta = theta.to(device=dev, dtype=real_dt)
    degrees = degrees.to(device=dev)

    per_row = _per_vertex_params(theta, degrees)
    cdt = _complex_for(real_dt)

    out: list[torch.Tensor] = []
    for i, pr in enumerate(per_row):
        d_i = int(degrees[i].item())
        if d_i <= 0:
            raise ValueError(f"degrees[{i}] must be >= 1 (got {d_i}).")
        if d_i == 1:
            # 1x1 unitary is [1]
            u = torch.ones(1, 1, device=dev, dtype=cdt)
            out.append(u)
            continue
        if d_i == 2:
            # For d=2, we could use Euler SU(2), but for exp(K) a 2x2 skew gives a planar rotation.
            # Build K and exponentiate.
            k = _skew_from_params(pr, d_i)  # (2,2), real
            u_real = torch.matrix_exp(k)  # (2,2), real orthogonal
            out.append(u_real.to(dtype=cdt))
            continue

        # General d > 2
        k = _skew_from_params(pr, d_i)  # (d_i, d_i)
        u_real = torch.matrix_exp(k)  # real orthogonal
        out.append(u_real.to(dtype=cdt))
    return out


def coins_cayley_from_skewk(
    theta: torch.Tensor,
    degrees: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    eps: float = 1e-6,
) -> list[torch.Tensor]:
    r"""
    Mixed-degree coin blocks via the Cayley transform of a skew-symmetric generator:

        U = (I - K)^{-1} (I + K)

    Parameters
    ----------
    theta : torch.Tensor, shape (V, P_max)
        Row i contains parameters for vertex i. For degree d_v, we use the first
        P(d_v) = d_v(d_v-1)/2 entries to fill K_v (strict upper triangle).
    degrees : torch.Tensor, shape (V,)
        Per-vertex degree (integer dtype).
    device, dtype : optional
        Target real dtype / device. Output is complex (complex64/128).
    eps : float
        Small Tikhonov regularization added to the system matrix for extra robustness
        when (I - K) is ill-conditioned.

    Returns
    -------
    list[torch.Tensor]
        A list of length V with (d_v, d_v) complex unitary blocks.

    Notes
    -----
    - We solve (I - K) X = (I + K) instead of explicitly inverting (I - K).
    - For d=1, returns [1]. For d=2, produces a 2D orthogonal rotation from skew K.
    """
    dev, real_dt = _canonical_device_dtype(theta, degrees, device=device, dtype=dtype)
    theta = theta.to(device=dev, dtype=real_dt)
    degrees = degrees.to(device=dev)

    per_row = _per_vertex_params(theta, degrees)
    cdt = _complex_for(real_dt)

    out: list[torch.Tensor] = []
    for i, pr in enumerate(per_row):
        d_i = int(degrees[i].item())
        if d_i <= 0:
            raise ValueError(f"degrees[{i}] must be >= 1 (got {d_i}).")
        if d_i == 1:
            out.append(torch.ones(1, 1, device=dev, dtype=cdt))
            continue

        k = _skew_from_params(pr, d_i)  # real (d_i,d_i)
        i_eye = torch.eye(d_i, dtype=real_dt, device=dev)
        a = i_eye - k  # (I - K)
        b = i_eye + k  # (I + K)

        # Optional diagonal Tikhonov to reduce risk of singularity
        if eps > 0.0:
            a = a + eps * i_eye

        # Solve A X = B
        # torch.linalg.solve supports batched right-hand sides; here (d,d)
        x = torch.linalg.solve(a, b)
        out.append(x.to(dtype=cdt))
    return out


# ---------------------------------------------------------------------------
# Convenience dispatcher (optional)
# ---------------------------------------------------------------------------


def coins_from_params_mixed(
    family: str,
    theta: torch.Tensor,
    degrees: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    cayley_eps: float = 1e-6,
) -> list[torch.Tensor]:
    fam = family.lower()
    if fam == "su2":
        if not torch.all(degrees == 2):
            raise ValueError("family='su2' requires all degrees == 2.")
        c = coins_su2_from_theta(theta, device=device, dtype=dtype)  # (V,2,2)
        return [c[i] for i in range(c.size(0))]
    if fam == "exp":
        return coins_exp_from_skewk(theta, degrees, device=device, dtype=dtype)
    if fam == "cayley":
        return coins_cayley_from_skewk(theta, degrees, device=device, dtype=dtype, eps=cayley_eps)
    raise ValueError(f"Unknown coin family: {family!r}. Expected 'su2' | 'exp' | 'cayley'.")

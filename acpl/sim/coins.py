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

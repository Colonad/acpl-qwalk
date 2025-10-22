# acpl/utils/tensor_shapes.py
# Phase B8 — Utils & typing hardening: shape & dtype guards
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import os
from typing import overload

import torch
from torch import Tensor

__all__ = [
    "TensorShapeError",
    "is_real_dtype",
    "is_complex_dtype",
    "expect_device",
    "expect_dtype",
    "expect_shape",
    "assert_edge_index",
    "assert_arc_slices",
    "assert_node_features",
    "assert_theta",
    "assert_state",
    "assert_coins_su2",
    "assert_blockdiag_compat",
]


# --------------------------- Exceptions & helpers --------------------------- #


@dataclass
class TensorShapeError(AssertionError):
    message: str
    got_shape: tuple[int, ...] | None = None
    expected: str | None = None
    got_dtype: torch.dtype | None = None
    expected_dtype: str | None = None

    def __str__(self) -> str:  # pragma: no cover - formatting only
        parts = [self.message]
        if self.got_shape is not None or self.expected is not None:
            parts.append(f"shape={self.got_shape} expected={self.expected}")
        if self.got_dtype is not None or self.expected_dtype is not None:
            parts.append(f"dtype={self.got_dtype} expected_dtype={self.expected_dtype}")
        return " | ".join(parts)


def _strict_mode() -> bool:
    # Opt-in hard checks. Defaults to on in CI; can be disabled by env.
    val = os.getenv("ACPL_STRICT_SHAPES", "1").strip().lower()
    return val not in ("0", "false", "no")


def is_real_dtype(dt: torch.dtype) -> bool:
    return dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def is_complex_dtype(dt: torch.dtype) -> bool:
    return dt in (torch.complex64, torch.complex128)


def _shape_str(x: Tensor) -> str:
    return "(" + ",".join(str(int(s)) for s in x.shape) + ")"


def expect_device(x: Tensor, device: torch.device | str, *, name: str = "tensor") -> Tensor:
    if str(x.device) != str(device):
        raise TensorShapeError(f"{name} must be on device {device}", got_shape=tuple(x.shape))
    return x


def expect_dtype(
    x: Tensor,
    *,
    real: bool | None = None,
    complex: bool | None = None,
    one_of: Iterable[torch.dtype] | None = None,
    name: str = "tensor",
) -> Tensor:
    dt = x.dtype
    if one_of is not None and dt not in set(one_of):
        raise TensorShapeError(
            f"{name} has bad dtype", got_dtype=dt, expected_dtype=str(list(one_of))
        )
    if real is True and not is_real_dtype(dt):
        raise TensorShapeError(f"{name} must be real dtype", got_dtype=dt)
    if complex is True and not is_complex_dtype(dt):
        raise TensorShapeError(f"{name} must be complex dtype", got_dtype=dt)
    if real is False and is_real_dtype(dt):
        raise TensorShapeError(f"{name} must NOT be real dtype", got_dtype=dt)
    if complex is False and is_complex_dtype(dt):
        raise TensorShapeError(f"{name} must NOT be complex dtype", got_dtype=dt)
    return x


# Named-shape checker with wildcards. Example:
#   expect_shape(x, ("T","N",3), named={"T":T, "N":N})
@overload
def expect_shape(
    x: Tensor, shape: Sequence[int | str], *, named: Mapping[str, int] | None = ..., name: str = ...
) -> Tensor: ...


def expect_shape(
    x: Tensor,
    shape: Sequence[int | str],
    *,
    named: Mapping[str, int] | None = None,
    name: str = "tensor",
) -> Tensor:
    if not isinstance(x, torch.Tensor):
        raise TensorShapeError(f"{name} must be a Tensor")
    if len(x.shape) != len(shape):
        raise TensorShapeError(
            f"{name} has wrong rank", got_shape=tuple(x.shape), expected=str(tuple(shape))
        )
    for i, (s_got, s_exp) in enumerate(zip(x.shape, shape, strict=True)):
        if isinstance(s_exp, int):
            if int(s_got) != int(s_exp) and _strict_mode():
                raise TensorShapeError(
                    f"{name} dim[{i}] expected {s_exp} got {int(s_got)}",
                    got_shape=tuple(x.shape),
                    expected=str(tuple(shape)),
                )
        elif isinstance(s_exp, str):
            if s_exp == "*" or s_exp == "?":
                continue  # wildcard any
            if named is None or s_exp not in named:
                # Best-effort: treat as wildcard if unknown name
                continue
            if int(s_got) != int(named[s_exp]) and _strict_mode():
                raise TensorShapeError(
                    f"{name} dim[{i}] '{s_exp}' expected {named[s_exp]} got {int(s_got)}",
                    got_shape=tuple(x.shape),
                    expected=str(tuple(shape)),
                )
        else:  # pragma: no cover - defensive
            raise TensorShapeError(f"Invalid spec for {name}: {s_exp}")
    return x


# ------------------------------- Core asserts ------------------------------ #


def assert_edge_index(
    edge_index: Tensor,
    *,
    num_nodes: int | None = None,
    allow_empty: bool = True,
    device: torch.device | str | None = None,
) -> Tensor:
    """(2, E) long tensor with nodes in range [0, num_nodes-1].
    We do not force coalescing/deduplication here.
    """
    name = "edge_index"
    expect_dtype(edge_index, one_of=[torch.long, torch.int64], name=name)
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise TensorShapeError("edge_index must be (2, E)", got_shape=tuple(edge_index.shape))
    if device is not None:
        expect_device(edge_index, device, name=name)
    E = int(edge_index.size(1))
    if E == 0 and not allow_empty and _strict_mode():
        raise TensorShapeError("edge_index is empty but allow_empty=False")
    if num_nodes is not None and E > 0 and _strict_mode():
        lo = int(edge_index.min().item())
        hi = int(edge_index.max().item())
        if lo < 0 or hi >= num_nodes:
            raise TensorShapeError(
                f"edge_index out of bounds [0,{num_nodes-1}]",
                got_shape=tuple(edge_index.shape),
            )
    return edge_index


def assert_arc_slices(
    arc_slices: Tensor,
    *,
    num_nodes: int | None = None,
    A: int | None = None,
    name: str = "arc_slices",
) -> Tensor:
    """arc_slices (N+1,) non-decreasing with arc_slices[0]=0 and arc_slices[-1]=A.
    Differences give node degrees.
    """
    if arc_slices.ndim != 1:
        raise TensorShapeError(f"{name} must be 1-D", got_shape=tuple(arc_slices.shape))
    if arc_slices.dtype not in (torch.long, torch.int64):
        raise TensorShapeError(f"{name} must be long dtype", got_dtype=arc_slices.dtype)
    if arc_slices.numel() == 0 and _strict_mode():
        raise TensorShapeError(f"{name} must not be empty")
    if num_nodes is not None and arc_slices.numel() != num_nodes + 1 and _strict_mode():
        raise TensorShapeError(
            f"{name} expected length N+1={num_nodes+1}", got_shape=(arc_slices.numel(),)
        )
    # monotonic
    if not torch.all(arc_slices[1:] >= arc_slices[:-1]):
        raise TensorShapeError(f"{name} must be non-decreasing")
    if arc_slices[0].item() != 0 and _strict_mode():
        raise TensorShapeError(f"{name}[0] must be 0")
    if A is not None and int(arc_slices[-1]) != int(A) and _strict_mode():
        raise TensorShapeError(f"{name}[-1] must equal A={A}")
    return arc_slices


def assert_node_features(
    X: Tensor,
    *,
    N: int | None = None,
    F: int | None = None,
    name: str = "X",
    allow_extra: bool = True,
) -> Tensor:
    if X.ndim != 2:
        raise TensorShapeError(f"{name} must be (N, F)", got_shape=tuple(X.shape))
    expect_dtype(X, real=True, name=name)
    if N is not None and int(X.size(0)) != int(N) and _strict_mode():
        raise TensorShapeError(
            f"{name} N mismatch", got_shape=tuple(X.shape), expected=f"({N}, {F or 'F'})"
        )
    if F is not None and int(X.size(1)) != int(F) and _strict_mode() and not allow_extra:
        raise TensorShapeError(f"{name} F mismatch", got_shape=tuple(X.shape), expected=f"(N, {F})")
    return X


def assert_theta(
    theta: Tensor,
    *,
    T: int | None = None,
    N: int | None = None,
    P: int | None = None,
    name: str = "theta",
) -> Tensor:
    if theta.ndim != 3:
        raise TensorShapeError(f"{name} must be (T, N, P)", got_shape=tuple(theta.shape))
    expect_dtype(theta, real=True, name=name)
    if T is not None and int(theta.size(0)) != int(T) and _strict_mode():
        raise TensorShapeError(
            f"{name} T mismatch", got_shape=tuple(theta.shape), expected=f"({T}, N, P)"
        )
    if N is not None and int(theta.size(1)) != int(N) and _strict_mode():
        raise TensorShapeError(
            f"{name} N mismatch", got_shape=tuple(theta.shape), expected=f"(T, {N}, P)"
        )
    if P is not None and int(theta.size(2)) != int(P) and _strict_mode():
        raise TensorShapeError(
            f"{name} P mismatch", got_shape=tuple(theta.shape), expected=f"(T, N, {P})"
        )
    return theta


def assert_state(
    psi: Tensor, *, A: int | None = None, unit_norm: bool | float = False, name: str = "psi"
) -> Tensor:
    """Check walker's full arc-amplitude vector psi ∈ C^A.
    unit_norm: if True or float, check ||psi||≈1 within tolerance (default 1e-5 or given).
    """
    if psi.ndim != 1:
        raise TensorShapeError(f"{name} must be (A,)", got_shape=tuple(psi.shape))
    expect_dtype(psi, complex=True, name=name)
    if A is not None and int(psi.numel()) != int(A) and _strict_mode():
        raise TensorShapeError(f"{name} A mismatch", got_shape=tuple(psi.shape), expected=f"({A},)")
    if unit_norm is not False:
        tol = 1e-5 if unit_norm is True else float(unit_norm)
        n = torch.linalg.vector_norm(psi)
        if not torch.isfinite(n):
            raise TensorShapeError(f"{name} norm is not finite")
        if abs(float(n.item()) - 1.0) > tol and _strict_mode():
            raise TensorShapeError(f"{name} not unit-normalized (||psi||={n.item():.3e})")
    return psi


def _check_unitary(U: Tensor, *, atol: float = 1e-4) -> bool:
    I = torch.eye(U.size(-1), dtype=U.dtype, device=U.device)
    prod = U.mH @ U
    err = torch.linalg.norm(prod - I).item()
    return err <= atol


def assert_coins_su2(
    C: Tensor,
    *,
    T: int | None = None,
    N: int | None = None,
    d: int = 2,
    check_unitary: bool | float = False,
    name: str = "coins",
) -> Tensor:
    """Coins shape (T, N, d, d) complex. Optional unitary check with tolerance.
    When check_unitary is float, it is used as atol; when True, default atol=1e-4.
    """
    if C.ndim != 4:
        raise TensorShapeError(f"{name} must be (T, N, d, d)", got_shape=tuple(C.shape))
    expect_dtype(C, complex=True, name=name)
    if T is not None and int(C.size(0)) != int(T) and _strict_mode():
        raise TensorShapeError(f"{name} T mismatch")
    if N is not None and int(C.size(1)) != int(N) and _strict_mode():
        raise TensorShapeError(f"{name} N mismatch")
    if int(C.size(2)) != int(d) or int(C.size(3)) != int(d):
        raise TensorShapeError(f"{name} d mismatch (expected {d}×{d})", got_shape=tuple(C.shape))
    if check_unitary is not False:
        atol = 1e-4 if check_unitary is True else float(check_unitary)
        # Cheap aggregated check: sample first time and a few nodes to limit cost.
        t0 = 0
        idx = (
            torch.linspace(0, C.size(1) - 1, steps=min(8, C.size(1)), device=C.device)
            .round()
            .long()
        )
        for v in idx.tolist():
            if not _check_unitary(C[t0, v], atol=atol) and _strict_mode():
                raise TensorShapeError(
                    f"{name}[t0={t0}, v={v}] not approximately unitary (atol={atol})"
                )
    return C


def assert_blockdiag_compat(
    arc_slices: Tensor, *, coins_T: int | None = None, msg_prefix: str = "blockdiag"
) -> Tensor:
    """Validate that arc_slices can index per-node coins during stepping.
    This function mostly sanity-checks deg=v_{i+1}-v_i ≥ 0 and global A.
    """
    assert_arc_slices(arc_slices)
    degs = arc_slices[1:] - arc_slices[:-1]
    if torch.any(degs < 0):
        raise TensorShapeError(f"{msg_prefix}: negative degree encountered")
    if coins_T is not None and coins_T <= 0:
        raise TensorShapeError(f"{msg_prefix}: coins_T must be positive")
    return arc_slices

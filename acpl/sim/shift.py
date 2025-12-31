# acpl/sim/shift.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    # If available (PyTorch >= 2.5), use real alias for complex dtypes
    ComplexDType = torch.complex64.__class__
except Exception:  # pragma: no cover
    ComplexDType = object  # type: ignore

from .portmap import PortMap

__all__ = [
    "ShiftOp",
    "build_shift",
    "perm_from_portmap",
    "sparse_from_portmap",
    "apply_shift",
    "check_shift_unitarity",
]






# --------------------------------------------------------------------------- #
#                               Core definition                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ShiftOp:
    """
    Flip–flop shift operator S on the arc basis.

    Storage:
      - perm: (A,) long tensor giving the reindexing: psi_out[i] = psi_in[perm[i]].
      - A:    number of arcs.
    """

    perm: torch.Tensor  # (A,) long
    A: int

    @property
    def device(self) -> torch.device:
        return self.perm.device

    # ------------------------------- Constructors --------------------------- #

    @staticmethod
    def from_portmap(pm: PortMap) -> "ShiftOp":
        perm = pm.to_shift_permutation()
        if perm.dtype != torch.long:
            perm = perm.to(dtype=torch.long)
        return ShiftOp(perm=perm, A=int(perm.numel()))

    # --------------------------------- APIs -------------------------------- #

    def apply(
        self,
        psi: torch.Tensor,
        *,
        phase: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply S to ψ laid out in the arc basis.

        Accepted shapes:
          - (A,) complex
          - (B, A) complex
        """
        if psi.numel() == 0:
            return psi
        if psi.shape[-1] != self.A:
            raise ValueError(f"psi last-dim must be {self.A} (got {psi.shape[-1]})")
        if not psi.is_complex():
            raise TypeError("psi must be a complex tensor (complex64/complex128).")

        # index_select requires perm on same device
        perm = self.perm
        if perm.device != psi.device:
            perm = perm.to(device=psi.device, dtype=torch.long)

        result = psi.index_select(-1, perm)

        if phase is not None:
            if phase.ndim != 1 or int(phase.numel()) != self.A:
                raise ValueError(f"phase must be (A,) with A={self.A}")
            if not phase.is_complex():
                raise TypeError("phase must be complex.")
            result = result * phase.to(device=result.device, dtype=result.dtype)

        if mask is not None:
            if mask.ndim != 1 or int(mask.numel()) != self.A:
                raise ValueError(f"mask must be (A,) with A={self.A}")
            m = mask.to(device=result.device)
            if m.dtype == torch.bool or not m.is_floating_point():
                m = m.to(dtype=result.real.dtype)
            else:
                m = m.to(dtype=result.real.dtype)
            result = result * m

        if out is not None:
            out.copy_(result)
            return out
        return result

    def to_sparse(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.complex64,
    ) -> torch.Tensor:
        """
        Return sparse COO permutation matrix S of shape (A, A) with ones at (i, perm[i]).
        Default device = self.perm.device (CPU in most builds).
        """
        dev = self.perm.device if device is None else torch.device(device)

        A = int(self.A)
        if A == 0:
            idx = torch.empty((2, 0), dtype=torch.long, device=dev)
            val = torch.empty((0,), dtype=dtype, device=dev)
            return torch.sparse_coo_tensor(idx, val, (0, 0), dtype=dtype, device=dev).coalesce()

        rows = torch.arange(A, device=dev, dtype=torch.long)
        cols = self.perm.to(device=dev, dtype=torch.long)
        idx = torch.stack([rows, cols], dim=0)
        val = torch.ones((A,), device=dev, dtype=dtype)
        return torch.sparse_coo_tensor(idx, val, (A, A), dtype=dtype, device=dev).coalesce()

    # ------------------------------- Invariants ----------------------------- #

    def check(self, strict: bool = True) -> None:
        perm = self.perm
        A = self.A
        if perm.dtype != torch.long:
            raise AssertionError("perm must be torch.long.")
        if int(perm.numel()) != A:
            raise AssertionError("perm length != A.")
        if A == 0:
            return
        if torch.unique(perm).numel() != A or perm.min() < 0 or perm.max() >= A:
            raise AssertionError("perm is not a valid permutation.")
        if strict:
            if not torch.equal(perm[perm], torch.arange(A, device=perm.device)):
                raise AssertionError("S must be an involution: perm[perm] != arange(A).")



# --------------------------------------------------------------------------- #
#                         Functional convenience layer                        #
# --------------------------------------------------------------------------- #


def perm_from_portmap(pm: PortMap) -> torch.Tensor:
    """
    Convenience: return the shift permutation (A,) from a PortMap.
    """
    return pm.to_shift_permutation()


def sparse_from_portmap(pm: PortMap, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    Convenience: return the sparse COO matrix for S built from a PortMap.
    """
    return ShiftOp.from_portmap(pm).to_sparse(dtype=dtype)


def build_shift(pm: PortMap) -> ShiftOp:
    """
    High-level constructor used by the simulator.
    """
    op = ShiftOp.from_portmap(pm)
    op.check(strict=True)
    return op


# --------------------------------------------------------------------------- #
#                             Application utilities                           #
# --------------------------------------------------------------------------- #


def _ensure_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("Expected complex64 or complex128 dtype for quantum state tensors.")
    return dtype


def apply_shift(
    psi: torch.Tensor,
    shift: ShiftOp | torch.Tensor,
    *,


    phase: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,


    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply the flip–flop shift to `psi`.

    Args
    ----
    psi : Tensor
        Statevector in the arc basis. Shape (A,) or (B, A). Complex dtype required.
    shift : ShiftOp | Tensor
        Either a `ShiftOp` or a permutation tensor `perm` of shape (A,) (torch.long).
    out : Optional[Tensor]
        Optional pre-allocated output tensor (same shape & dtype as `psi`).

    Returns
    -------
    Tensor
        S @ psi, same shape as input.

    Notes
    -----
    - This is purely an index permutation; it preserves norms exactly.
    - Gradients flow through `psi` only (no learnable parameters here).
    """
    _ensure_complex_dtype(psi.dtype)

    if isinstance(shift, ShiftOp):
        return shift.apply(psi, phase=phase, mask=mask, out=out)

    # Assume `shift` is a (A,) long permutation
    perm = shift
    if perm.dtype != torch.long:
        raise TypeError("`shift` as Tensor must be torch.long permutation of shape (A,).")


    if perm.device != psi.device:
        perm = perm.to(device=psi.device)

    A = perm.numel()
    if psi.shape[-1] != A:
        raise ValueError(f"psi last-dim must be {A} (got {psi.shape[-1]}).")






    # Validate / align optional modifiers
    if phase is not None:
        if (phase.ndim != 1) or (phase.numel() != A):
            raise ValueError(f"phase must be shape (A,) with A={A}, got {tuple(phase.shape)}.")
        if not phase.is_complex():
            raise TypeError("phase must be complex (unit-magnitude complex phases).")
        if phase.device != psi.device:
            phase = phase.to(device=psi.device)
        if phase.dtype != psi.dtype:
            phase = phase.to(dtype=psi.dtype)

    if mask is not None:
        if (mask.ndim != 1) or (mask.numel() != A):
            raise ValueError(f"mask must be shape (A,) with A={A}, got {tuple(mask.shape)}.")
        if mask.device != psi.device:
            mask = mask.to(device=psi.device)
        # allow bool or float-ish, but apply as real dtype
        if mask.dtype == torch.bool or not mask.is_floating_point():
            mask = mask.to(dtype=psi.real.dtype)
        else:
            mask = mask.to(dtype=psi.real.dtype)







    if psi.ndim == 1:
        result = psi.index_select(0, perm)
        
        
        if phase is not None: result = result * phase.to(device=psi.device, dtype=psi.dtype)
        if mask is not None:  result = result * mask.to(device=psi.device, dtype=psi.real.dtype)
        
        
        
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif psi.ndim == 2:
        result = psi.index_select(1, perm)
        
        if phase is not None:
            result = result * phase
        if mask is not None:
            result = result * mask
            
            
        
        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError("psi must be rank-1 or rank-2 (batched) over the arc axis.")


# --------------------------------------------------------------------------- #
#                                Diagnostics                                  #
# --------------------------------------------------------------------------- #


def check_shift_unitarity(
    shift: ShiftOp | torch.Tensor,
    *,
    atol: float = 0.0,
) -> None:
    """
    Assert that S is a valid permutation unitary.

    Conditions:
      1) perm is a permutation of [0..A-1]
      2) S^2 = I (involution)  <=>  perm[perm] == arange(A)

    For sparse matrices, you’d use algebraic checks; here we operate on perms.
    """
    if isinstance(shift, ShiftOp):
        shift.check(strict=True)
        return

    perm = shift
    if perm.dtype != torch.long:
        raise AssertionError("Permutation must be torch.long.")
    A = int(perm.numel())
    if A == 0:
        return
    if torch.unique(perm).numel() != A or perm.min() < 0 or perm.max() >= A:
        raise AssertionError("Not a permutation over [0..A-1].")
    if not torch.equal(perm[perm], torch.arange(A, device=perm.device)):
        # If `atol` > 0 were meaningful we’d allow “nearly equal”, but indices are exact.
        raise AssertionError("Involution failed: perm[perm] != arange(A).")


# --------------------------------------------------------------------------- #
#                               Module self-test                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    # Tiny sanity check on a 3-node path 1-2-3:
    # arcs: [1->2, 2->1, 2->3, 3->2]
    perm = torch.tensor([1, 0, 3, 2], dtype=torch.long)
    S = ShiftOp(perm=perm, A=int(perm.numel()))
    S.check()

    psi = torch.tensor([1, 2, 3, 4], dtype=torch.complex64)
    out = S.apply(psi)
    assert torch.equal(out, torch.tensor([2, 1, 4, 3], dtype=torch.complex64))

    check_shift_unitarity(S)
    print("shift.py self-test passed.")

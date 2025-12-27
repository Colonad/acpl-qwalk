# acpl/sim/shift.py
from __future__ import annotations

from dataclasses import dataclass

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

    The shift S is a permutation unitary determined entirely by the PortMap:
      • Involutive: S^2 = I (equivalently, perm[perm] = arange(A)).
      • Hermitian and unitary: S† = S = S^{-1}.
      • Acts by reindexing amplitudes between reverse arcs.

    Storage:
      - `perm`: (A,) long tensor giving the reindexing: psi_out[i] = psi_in[perm[i]].
      - `A`:    number of arcs (total coin dimension).
      - `device`: CUDA/CPU device of the permutation.
    """

    perm: torch.Tensor  # (A,) long
    A: int
    device: torch.device

    # ------------------------------- Constructors --------------------------- #

    @staticmethod
    def from_portmap(pm: PortMap) -> ShiftOp:
        """
        Build S from a PortMap. This is just the `rev` involution.
        """
        perm = pm.to_shift_permutation()  # (A,)
        return ShiftOp(perm=perm, A=int(perm.numel()), device=perm.device)

    # --------------------------------- APIs -------------------------------- #

    def to_sparse(self, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
        """
        Return a sparse COO matrix S of shape (A, A) with ones at (i, perm[i]).
        """
        A = self.A
        if A == 0:
            idx = torch.empty((2, 0), dtype=torch.long, device=self.device)
            val = torch.empty((0,), dtype=dtype, device=self.device)
            return torch.sparse_coo_tensor(
                idx, val, (0, 0), dtype=dtype, device=self.device
            ).coalesce()

        rows = torch.arange(A, device=self.device, dtype=torch.long)
        cols = self.perm
        idx = torch.stack([rows, cols], dim=0)
        val = torch.ones(A, dtype=dtype, device=self.device)
        return torch.sparse_coo_tensor(idx, val, (A, A), dtype=dtype, device=self.device).coalesce()

    def apply(
        self,
        psi: torch.Tensor,
        *,
        phase: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply S to a statevector ψ laid out in the arc basis.

        Accepted shapes:
          - (A,) complex
          - (B, A) complex (batched first)

        Returns:
          same shape as `psi`.
        """
        if psi.numel() == 0:
            return psi

        if psi.shape[-1] != self.A:
            raise ValueError(f"psi last-dim must be {self.A} (got {psi.shape[-1]})")

        if not psi.is_complex():
            raise TypeError("psi must be a complex tensor (complex64/complex128).")

        # Ensure perm is long and on the same device as psi (index_select requirement).
        perm = self.perm
        if perm.dtype != torch.long:
            perm = perm.to(dtype=torch.long)
        if perm.device != psi.device:
            perm = perm.to(device=psi.device)
            # Optional cache-back (dataclass is frozen, so bypass safely)
            try:
                object.__setattr__(self, "perm", perm)
                object.__setattr__(self, "device", psi.device)
            except Exception:
                pass

        def _apply_modifiers(x: torch.Tensor) -> torch.Tensor:
            if phase is not None:
                if phase.ndim != 1 or int(phase.numel()) != self.A:
                    raise ValueError(f"phase must be (A,) with A={self.A}")
                x = x * phase.to(device=x.device, dtype=x.dtype)
            if mask is not None:
                if mask.ndim != 1 or int(mask.numel()) != self.A:
                    raise ValueError(f"mask must be (A,) with A={self.A}")
                x = x * mask.to(device=x.device, dtype=x.real.dtype)
            return x
        
        
        
        
        
        if psi.ndim == 1:


            result = psi.index_select(0, perm)
            result = _apply_modifiers(result)            
            
            
            if out is not None:
                out.copy_(result)
                return out
            return result
        elif psi.ndim == 2:
            # (B, A) -> gather columns per row using the same perm
            
            
            result = psi.index_select(1, perm)
            result = _apply_modifiers(result)            
            
            if out is not None:
                out.copy_(result)
                return out
            return result
        else:
            raise ValueError("psi must be rank-1 or rank-2 (batched) over the arc axis.")

    # ------------------------------- Properties ----------------------------- #

    @property
    def sparse(self) -> torch.Tensor:
        """Alias for `to_sparse()` with default dtype=complex64."""
        return self.to_sparse(dtype=torch.complex64)

    # ------------------------------- Invariants ----------------------------- #

    def check(self, strict: bool = True) -> None:
        """
        Validate S invariants: permutation, involution, and device consistency.
        """
        perm = self.perm
        A = self.A
        if perm.dtype != torch.long:
            raise AssertionError("perm must be torch.long.")
        if perm.device != self.device:
            raise AssertionError("perm/device mismatch.")
        if int(perm.numel()) != A:
            raise AssertionError("perm length != A.")
        # Check it's a true permutation of [0..A-1]
        if A > 0:
            if torch.unique(perm).numel() != A or perm.min() < 0 or perm.max() >= A:
                raise AssertionError("perm is not a valid permutation.")
            # Flip–flop involution: S^2 = I  <=>  perm[perm] == arange(A)
            if strict:
                if not torch.equal(perm[perm], torch.arange(A, device=self.device)):
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
        
        if phase is not None: result = result * phase.to(device=psi.device, dtype=psi.dtype)
        if mask is not None:  result = result * mask.to(device=psi.device, dtype=psi.real.dtype)
        
        
        
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
    S = ShiftOp(perm=perm, A=perm.numel(), device=perm.device)
    S.check()

    psi = torch.tensor([1, 2, 3, 4], dtype=torch.complex64)
    out = S.apply(psi)
    assert torch.equal(out, torch.tensor([2, 1, 4, 3], dtype=torch.complex64))

    check_shift_unitarity(S)
    print("shift.py self-test passed.")

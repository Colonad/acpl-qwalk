# acpl/policy/pe.py
from __future__ import annotations

import torch
from torch import nn


class PositionalEncodingStub(nn.Module):
    """
    Minimal PE wrapper for Phase A.

    Behavior
    --------
    - If neither pos nor t_over_t is provided: returns x unchanged.
    - If pos is provided (shape (N, P)): concatenates as extra features.
    - If t_over_t is provided: appends a (N, 1) column with that scalar.

    Parameters
    ----------
    use_time_scalar : bool
        If True, include the time scalar when t_over_t is passed to forward.
    """

    def __init__(self, use_time_scalar: bool = True) -> None:
        super().__init__()
        self.use_time_scalar = use_time_scalar

    @staticmethod
    def _as_scalar_tensor(
        val: float | torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return torch.tensor(float(val), device=device, dtype=dtype)
        return torch.as_tensor(val, device=device, dtype=dtype).reshape(())

    def forward(
        self,
        x: torch.Tensor,  # (N, F)
        *,
        pos: torch.Tensor | None = None,  # (N, P) optional
        t_over_t: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Concatenate optional positional encodings and/or a temporal scalar.

        Returns
        -------
        torch.Tensor
            Feature tensor with any requested encodings appended along dim=1.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (N, F), got {tuple(x.shape)}")

        feats = x
        if pos is not None:
            if pos.ndim != 2 or pos.size(0) != x.size(0):
                raise ValueError("pos must have shape (N, P) with same N as x")
            feats = torch.cat([feats, pos.to(device=x.device, dtype=x.dtype)], dim=1)

        if self.use_time_scalar and t_over_t is not None:
            scalar = self._as_scalar_tensor(t_over_t, device=x.device, dtype=x.dtype)
            # Broadcast to a (N, 1) column and append
            tcol = scalar.expand(x.size(0), 1)  # type: ignore[arg-type]
            feats = torch.cat([feats, tcol], dim=1)

        return feats

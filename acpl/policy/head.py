# acpl/policy/head.py
from __future__ import annotations

import math

import torch
from torch import nn


class CoinEulerHead(nn.Module):
    """
    Linear head mapping hidden states -> Euler angles (alpha, beta, gamma).

    Accepts either a single snapshot (N, H) or a time sequence in either layout:
        - time-first:  (T, N, H)  with time_first=True  (default)
        - batch-first: (N, T, H)  with time_first=False

    Output shape mirrors the input leading dims and appends 3 for angles:
        (N, 3) or (T, N, 3) / (N, T, 3)

    Parameters
    ----------
    hidden_dim : int
        Size H of the hidden state per node.
    angle_range : {"unbounded", "tanh_pi"}
        If "tanh_pi", the head outputs π * tanh(raw) to bound angles to [-π, π].
        If "unbounded", angles are left unconstrained (ℝ).
    dropout : float
        Optional dropout applied on the hidden states before the linear layer.
    bias : bool
        Whether to include bias in the linear projection.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        angle_range: str = "unbounded",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if angle_range not in {"unbounded", "tanh_pi"}:
            raise ValueError('angle_range must be "unbounded" or "tanh_pi"')
        self.hidden_dim = hidden_dim
        self.angle_range = angle_range
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.proj = nn.Linear(hidden_dim, 3, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Small init to keep early angles near zero; stable for SU(2) mapping.
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        if self.angle_range == "tanh_pi":
            return math.pi * torch.tanh(raw)
        return raw

    def forward(
        self,
        h: torch.Tensor,
        *,
        time_first: bool = True,
    ) -> torch.Tensor:
        """
        Map hidden states to Euler angles.

        Parameters
        ----------
        h : torch.Tensor
            Hidden states with shape:
              - (N, H)               single snapshot
              - (T, N, H)            time-first when time_first=True  (default)
              - (N, T, H)            batch-first when time_first=False
        time_first : bool
            When h is 3-D, indicates whether layout is (T, N, H) (True) or (N, T, H) (False).

        Returns
        -------
        torch.Tensor
            Angles with shape:
              - (N, 3) or (T, N, 3) / (N, T, 3) corresponding to input layout.
        """
        if h.ndim == 2:
            n, f = h.shape
            if f != self.hidden_dim:
                raise ValueError(f"hidden dim mismatch: got {f}, expected {self.hidden_dim}")
            out = self.drop(h)
            raw = self.proj(out)  # (N, 3)
            return self._activate(raw)

        if h.ndim != 3:
            raise ValueError("h must be 2-D (N,H) or 3-D ((T,N,H)/(N,T,H))")

        if time_first:
            t, n, f = h.shape
            if f != self.hidden_dim:
                raise ValueError(f"hidden dim mismatch: got {f}, expected {self.hidden_dim}")
            flat = self.drop(h).reshape(t * n, f)
            raw = self.proj(flat).reshape(t, n, 3)
            return self._activate(raw)

        # batch-first (N, T, H)
        n, t, f = h.shape
        if f != self.hidden_dim:
            raise ValueError(f"hidden dim mismatch: got {f}, expected {self.hidden_dim}")
        flat = self.drop(h).reshape(n * t, f)
        raw = self.proj(flat).reshape(n, t, 3)
        return self._activate(raw)

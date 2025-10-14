# acpl/policy/controller/gru.py
from __future__ import annotations

import torch
from torch import nn


class NodeGRUController(nn.Module):
    """
    Shared GRU over time for each node's embedding.

    The GRU parameters are shared across all nodes. We treat nodes as the
    "batch" dimension and unroll for `steps` time steps.

    Args
    ----
    in_dim : int
        Size of input node embeddings F.
    hidden_dim : int
        GRU hidden size H (also the output feature size).
    num_layers : int
        Number of stacked GRU layers (default 1).
    dropout : float
        Dropout between GRU layers (effective only if num_layers > 1).
    bidirectional : bool
        If True, use a bidirectional GRU (output size becomes 2*hidden_dim).

    Notes
    -----
    - Phase-A usage: a static per-node embedding z (N, F) repeated for `steps`.
      For time-varying inputs, use `forward_sequence`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        do = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=do,
            bidirectional=bidirectional,
            batch_first=True,  # input (N, T, F)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        steps: int,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evolve per-node states for `steps` using a repeated input z.

        Parameters
        ----------
        z : torch.Tensor, shape (N, F)
            Node embeddings at t=0 (shared over time for Phase A).
        steps : int
            Number of time steps to unroll.
        h0 : torch.Tensor, optional
            Initial hidden state of shape (L * D, N, H), where
              L = num_layers, D = 2 if bidirectional else 1.

        Returns
        -------
        h_seq : torch.Tensor, shape (T, N, H_out)
            Hidden sequence (time-major). If bidirectional, H_out = 2*hidden_dim.
        h_last : torch.Tensor, shape (N, H_out)
            Final hidden state at time T for the top layer.
        """
        if z.ndim != 2:
            raise ValueError(f"z must have shape (N, F), got {tuple(z.shape)}")
        if steps <= 0:
            raise ValueError("steps must be >= 1")

        n, f = z.shape
        if f != self.in_dim:
            raise ValueError(f"z has F={f}, expected in_dim={self.in_dim}")

        # Repeat the same per-node input for T steps: (N, T, F)
        x = z.unsqueeze(1).expand(n, steps, f)

        # Run GRU (batch_first=True): out (N, T, H_out), h_n (L*D, N, H)
        out, _h_n = self.gru(x, h0)

        # Convert to time-major for downstream convenience: (T, N, H_out)
        h_seq = out.transpose(0, 1).contiguous()

        # Final hidden for the top layer.
        h_last = out[:, -1, :]  # (N, H_out)
        return h_seq, h_last

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
        *,
        time_first: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        General-time-input variant.

        Parameters
        ----------
        x_seq : torch.Tensor, shape (T, N, F) or (N, T, F)
            Per-time per-node inputs. If time-first, it will be transposed.
        h0 : torch.Tensor, optional
            Initial hidden state of shape (L * D, N, H).
        time_first : bool, optional
            If None, infer by comparing dims: if x_seq.shape[0] != x_seq.shape[1]
            and the first dim is smaller, treat as (T, N, F). Otherwise assume
            (N, T, F). Pass explicitly to avoid ambiguity.

        Returns
        -------
        h_seq : torch.Tensor, shape (T, N, H_out)
        h_last : torch.Tensor, shape (N, H_out)
        """
        if x_seq.ndim != 3:
            raise ValueError("x_seq must be 3-D (T,N,F) or (N,T,F)")

        if time_first is None:
            time_first = x_seq.size(0) < x_seq.size(1)

        x_bt = x_seq.transpose(0, 1).contiguous() if time_first else x_seq
        out, _h = self.gru(x_bt, h0)
        h_seq = out.transpose(0, 1).contiguous()
        h_last = out[:, -1, :]
        return h_seq, h_last

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def init_hidden(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Create a zeros initial state h0 with correct shape (L*D, N, H).
        """
        layers = self.num_layers * (2 if self.bidirectional else 1)
        return torch.zeros(layers, num_nodes, self.hidden_dim, device=device)

# acpl/policy/policy.py
from __future__ import annotations

import torch
from torch import nn

from acpl.policy.controller.gru import NodeGRUController
from acpl.policy.gnn.gcn import TwoLayerGCN
from acpl.policy.head import CoinEulerHead


class GNNTemporalPolicy(nn.Module):
    """
    Compose {GCN, GRU, Head} to produce per-node Euler angles θ_{v,t}.

    forward(g, x, pos_enc, t_over_t, h_prev) -> (theta_vt, h_next)

    - g : edge_index (2, E) torch.long
    - x : node features (N, F_x)
    - pos_enc : optional positional encodings (N, F_p) to concat with x
    - t_over_t : optional scalar in [0,1] (float or 0-D/1-D tensor); adds a small
                 time embedding to the GRU input
    - h_prev : optional previous hidden (L*D, N, Hc). If None, GRU starts from 0.

    Returns:
      theta_vt : (N, 3)  — Euler angles for coins at this time step
      h_next   : (N, H_out) — final hidden for the top GRU layer
                 (H_out = Hc for unidirectional; 2*Hc if bidirectional)
    """

    def __init__(
        self,
        in_dim: int,
        gnn_hidden: int,
        *,
        controller_hidden: int | None = None,
        gnn_dropout: float = 0.0,
        controller_layers: int = 1,
        controller_dropout: float = 0.0,
        controller_bidirectional: bool = False,
        head_angle_range: str = "unbounded",
        head_dropout: float = 0.0,
        use_time_embed: bool = True,
    ) -> None:
        super().__init__()
        hc = controller_hidden if controller_hidden is not None else gnn_hidden

        # GCN encodes per-node context (sum aggregation with GCN normalization)
        self.gnn = TwoLayerGCN(in_dim=in_dim, hidden_dim=gnn_hidden, dropout=gnn_dropout)

        # GRU evolves per-node state over time (shared across nodes)
        self.controller = NodeGRUController(
            in_dim=gnn_hidden,
            hidden_dim=hc,
            num_layers=controller_layers,
            dropout=controller_dropout,
            bidirectional=controller_bidirectional,
        )

        # Head maps hidden -> Euler angles (alpha, beta, gamma)
        self.head = CoinEulerHead(
            hidden_dim=(2 * hc if controller_bidirectional else hc),
            angle_range=head_angle_range,
            dropout=head_dropout,
        )

        # Optional tiny time embedding: scalar -> (N, gnn_hidden), then add to GCN output
        self.use_time_embed = use_time_embed
        if use_time_embed:
            self.time_proj = nn.Linear(1, gnn_hidden, bias=True)
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)

    @staticmethod
    def _as_scalar_tensor(
        x: torch.Tensor | float | None,
        device: torch.device,
    ) -> torch.Tensor:
        if x is None:
            return torch.zeros(1, device=device, dtype=torch.float32)
        if isinstance(x, (int, float)):
            return torch.tensor([float(x)], device=device, dtype=torch.float32)
        t = torch.as_tensor(x, dtype=torch.float32, device=device)
        return t.view(1)

    def forward(
        self,
        g: torch.Tensor,
        x: torch.Tensor,
        pos_enc: torch.Tensor | None = None,
        t_over_t: torch.Tensor | float | None = None,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compose GCN -> GRU(one step) -> Head.

        Parameters
        ----------
        g : torch.Tensor, shape (2, E), long
            Edge index.
        x : torch.Tensor, shape (N, F_x)
            Node features.
        pos_enc : torch.Tensor, optional, shape (N, F_p)
            Positional encodings to concatenate with x.
        t_over_t : float or tensor
            Scalar in [0,1]; if provided and time embedding is enabled, it nudges
            the GRU input via a tiny linear projection.
        h_prev : torch.Tensor, optional, shape (L*D, N, Hc)
            Previous GRU hidden.

        Returns
        -------
        theta_vt : torch.Tensor, shape (N, 3)
        h_next   : torch.Tensor, shape (N, H_out)
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D (N, F), got {tuple(x.shape)}")
        if g.ndim != 2 or g.size(0) != 2:
            raise ValueError("g must be edge_index of shape (2, E)")

        dev = x.device
        feats = x if pos_enc is None else torch.cat([x, pos_enc], dim=1)

        # 1) Per-node embedding via GCN
        z = self.gnn(feats, g)  # (N, gnn_hidden)

        # 2) Optional time conditioning (tiny projection of scalar -> add to z)
        if self.use_time_embed:
            t_scalar = self._as_scalar_tensor(t_over_t, device=dev)  # (1,)
            bias = self.time_proj(t_scalar.view(1, 1)).view(-1)  # (gnn_hidden,)
            z = z + bias.view(1, -1).expand_as(z)

        # 3) One GRU step (treat as sequence length 1)
        x_bt = z.unsqueeze(1)  # (N, 1, gnn_hidden)
        out_bt, _h = self.controller.gru(x_bt, h_prev)  # out: (N, 1, H_out)
        h_next = out_bt[:, -1, :]  # (N, H_out)

        # 4) Head to Euler angles (N, 3)
        theta_vt = self.head(h_next)  # (N, 3)
        return theta_vt, h_next

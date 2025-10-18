# acpl/policy/policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

# For dtype preference in coins_su2; we do NOT call the external builder to keep the
# autograd path intact. We just read the dtype from this spec if provided.
from acpl.sim.coins import CoinsSpecSU2

from .controller.gru import GRUController, GRUControllerConfig
from .controller.tiny_transformer import TinyTimeTransformer, TinyTimeTransformerConfig
from .gnn.gcn import GCN
from .head import CoinHeadConfig, CoinHeadSU2
from .pe import TimeFourierPE, TimeFourierPEConfig

__all__ = ["ACPLPolicyConfig", "ACPLPolicy"]


@dataclass
class ACPLPolicyConfig:
    # Encoder (GNN)
    in_dim: int
    gnn_hidden: int = 128
    gnn_out: int = 128
    gnn_activation: str = "gelu"
    gnn_dropout: float = 0.1
    gnn_layernorm: bool = True
    gnn_residual: bool = True
    gnn_dropedge: float = 0.0

    # Controller
    controller: Literal["gru", "transformer"] = "gru"
    ctrl_hidden: int = 128
    ctrl_layers: int = 1
    ctrl_dropout: float = 0.0
    ctrl_layernorm: bool = True
    ctrl_bidirectional: bool = False

    # Time positional encoding
    time_pe_dim: int = 32
    time_pe_learned_scale: bool = True

    # Head → Euler angles
    head_hidden: int = 0  # 0 = linear head
    head_out_scale: float = 1.0
    head_layernorm: bool = True
    head_dropout: float = 0.0


class ACPLPolicy(nn.Module):
    r"""
    π_ω = HEAD ∘ CONTROLLER ∘ GNN

    Primary contracts
    -----------------
    * Minimal (backward-compatible):
        forward(X, edge_index, T=...) -> theta: (T, N, 3)

    * Extended (for richer control, without breaking tests):
        forward(X, edge_index, *, T=None, t_coords=None, edge_weight=None,
                h_prev=None, mask=None, return_state=False)
            → theta or (theta, h_next)

        - Provide **either** T (int) **or** t_coords (Tensor of shape (T,) or (T,1)).
          If both are given, T is ignored and len(t_coords) is used.
        - h_prev (optional) is passed to the controller (GRU); if the controller
          exposes the terminal hidden state, we return it when return_state=True.
          If not available, h_next=None is returned when return_state=True.

    Convenience
    -----------
    * coins_su2(..., T=...) : returns (T, N, 2, 2) complex (inline differentiable SU(2) lift).
    """

    def __init__(self, cfg: ACPLPolicyConfig):
        super().__init__()
        self.cfg = cfg

        # --- GNN encoder ---
        self.encoder = GCN(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.gnn_hidden,
            out_dim=cfg.gnn_out,
            activation=cfg.gnn_activation,
            dropout=cfg.gnn_dropout,
            layernorm=cfg.gnn_layernorm,
            residual=cfg.gnn_residual,
            dropedge=cfg.gnn_dropedge,
        )

        # --- Time P.E. ---
        self.time_pe = TimeFourierPE(
            TimeFourierPEConfig(dim=cfg.time_pe_dim, learned_scale=cfg.time_pe_learned_scale)
        )

        # --- Temporal controller ---
        ctrl_in = cfg.gnn_out + cfg.time_pe_dim
        if cfg.controller == "gru":
            self.controller = GRUController(
                GRUControllerConfig(
                    in_dim=ctrl_in,
                    hidden_dim=cfg.ctrl_hidden,
                    num_layers=cfg.ctrl_layers,
                    dropout=cfg.ctrl_dropout,
                    layernorm=cfg.ctrl_layernorm,
                    bidirectional=cfg.ctrl_bidirectional,
                )
            )
            ctrl_out_dim = cfg.ctrl_hidden * (2 if cfg.ctrl_bidirectional else 1)

        elif cfg.controller == "transformer":
            self.controller = TinyTimeTransformer(
                TinyTimeTransformerConfig(
                    in_dim=ctrl_in,
                    model_dim=cfg.ctrl_hidden,
                    num_layers=cfg.ctrl_layers,
                    num_heads=max(1, min(8, cfg.ctrl_hidden // 32)),
                    mlp_ratio=2.0,
                    dropout=cfg.ctrl_dropout,
                    layernorm=cfg.ctrl_layernorm,
                    residual=False,
                )
            )
            ctrl_out_dim = cfg.ctrl_hidden
        else:
            raise ValueError(f"Unknown controller: {cfg.controller}")

        # --- Coin head (Euler ZYZ angles) ---
        self.head = CoinHeadSU2(
            CoinHeadConfig(
                in_dim=ctrl_out_dim,
                hidden_dim=(cfg.head_hidden if cfg.head_hidden > 0 else None),
                out_scale=cfg.head_out_scale,
                layernorm=cfg.head_layernorm,
                activation="gelu",
                dropout=cfg.head_dropout,
            )
        )

    @torch.no_grad()
    def _check_minimal(self, X: torch.Tensor, edge_index: torch.Tensor):
        if X.ndim != 2:
            raise ValueError("X must be (N, Fin).")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be (2, E).")

    def _make_time_pe(
        self,
        *,
        T: int | None,
        t_coords: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build time positional encodings. If t_coords is provided, we use its length
        to define T. Current TimeFourierPE uses only T; we keep t_coords for future
        curriculum/schedules (e.g., nonuniform steps).
        """
        if t_coords is not None:
            if t_coords.ndim == 1:
                T_eff = int(t_coords.shape[0])
            elif t_coords.ndim == 2 and t_coords.shape[-1] == 1:
                T_eff = int(t_coords.shape[0])
            else:
                raise ValueError("t_coords must be (T,) or (T,1) if provided.")
            return self.time_pe(T_eff, device=device)  # (T_eff, Pt)
        else:
            if T is None or T <= 0:
                raise ValueError("Provide a positive T when t_coords is not given.")
            return self.time_pe(int(T), device=device)

    def forward(
        self,
        X: torch.Tensor,  # (N, Fin)
        edge_index: torch.Tensor,  # (2, E)
        *,
        T: int | None = 1,
        t_coords: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        h_prev: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns
        -------
        theta : (T, N, 3)    Euler angles per (time,node)
        (theta, h_next)      if return_state=True (h_next may be None if controller
                             does not expose it).

        Notes
        -----
        * Backward compatibility: callers that pass only (X, edge_index, T=...)
          get the same behavior as before.
        * If both T and t_coords are provided, t_coords length is used.
        """
        self._check_minimal(X, edge_index)
        device = X.device

        # 1) Node encoding
        z = self.encoder(X, edge_index, edge_weight)  # (N, Dz)

        # 2) Time embeddings
        tpe = self._make_time_pe(T=T, t_coords=t_coords, device=device)  # (T, Pt)

        # 3) Temporal controller
        # Our controllers currently return only the time trajectory H. If/when they
        # support returning h_next, we will detect it. For GRUController we pass h_prev
        # through (it already accepts it).
        h_next: torch.Tensor | None = None
        try:
            # Try calling with all bells & whistles; controllers ignore extras they don't support.
            H = self.controller(z, tpe, h0=h_prev, mask=mask)  # (T, N, Dh)
        except TypeError:
            # Fall back to the minimal signature
            H = self.controller(z, tpe)  # (T, N, Dh)

        # If the controller exposes a terminal state via attribute (future-proof),
        # capture it. (Current GRUController does not, so this remains None.)
        if hasattr(self.controller, "_last_hN"):
            h_next = self.controller._last_hN

        # 4) Coin head → Euler ZYZ angles
        theta = self.head(H)  # (T, N, 3)

        if return_state:
            return theta, h_next
        return theta

    # ---------------------- differentiable inline SU(2) lift ---------------------- #
    @staticmethod
    def _su2_from_euler_batch(theta: torch.Tensor, *, dtype=torch.complex64) -> torch.Tensor:
        """
        Differentiable ZYZ Euler → SU(2) map.

        theta: (..., 3) real [alpha, beta, gamma]
        returns U: (..., 2, 2) complex (unitary, det=1 exactly)
        """
        if theta.size(-1) != 3:
            raise ValueError("theta[..., 3] expected")
        alpha, beta, gamma = theta.unbind(dim=-1)

        half = 0.5
        cb = torch.cos(half * beta)
        sb = torch.sin(half * beta)

        def cexp(x: torch.Tensor) -> torch.Tensor:
            # e^{i x} = cos x + i sin x
            return torch.complex(torch.cos(x), torch.sin(x)).to(dtype)

        p = cexp(-half * (alpha + gamma))
        q = cexp(-half * (alpha - gamma))
        r = cexp(+half * (alpha - gamma))
        s = cexp(+half * (alpha + gamma))

        U00 = p * cb
        U01 = -q * sb
        U10 = r * sb
        U11 = s * cb

        U = torch.stack(
            [torch.stack([U00, U01], dim=-1), torch.stack([U10, U11], dim=-1)],
            dim=-2,
        )
        return U.to(dtype)

    def coins_su2(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        T: int,
        spec: CoinsSpecSU2 | None = None,
        edge_weight: torch.Tensor | None = None,
        t_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convenience: directly get a **stack** of SU(2) coins from the policy.

        Returns:
            coins: (T, N, 2, 2) complex

        Notes:
        * Uses our differentiable inline lift to preserve gradients. We only read
          `spec.dtype` to choose complex dtype; normalize/check are handled upstream.
        """
        theta = self.forward(
            X, edge_index, T=T, t_coords=t_coords, edge_weight=edge_weight
        )  # (T, N, 3)

        cdtype = torch.complex64 if spec is None else spec.dtype
        if theta.ndim == 3:
            T_, N_, _ = theta.shape
            flat = theta.reshape(T_ * N_, 3)
            U = self._su2_from_euler_batch(flat, dtype=cdtype)  # (T*N, 2, 2)
            return U.view(T_, N_, 2, 2).contiguous()
        elif theta.ndim == 2:
            U = self._su2_from_euler_batch(theta, dtype=cdtype)  # (N, 2, 2)
            return U.unsqueeze(0)  # (1, N, 2, 2)
        else:
            raise ValueError("theta must be (T,N,3) or (N,3).")

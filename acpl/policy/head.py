# acpl/policy/head.py
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

__all__ = ["CoinHeadSU2", "CoinHeadConfig", "wrap_to_pi"]


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap angles to (-pi, pi]. Uses modulo; not gradient-friendly by itself.
    We apply a straight-through estimator (STE) around this in the head.
    """
    return ((x + math.pi) % (2 * math.pi)) - math.pi


def _activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    if n == "tanh":
        return nn.Tanh()
    if n == "silu" or n == "swish":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


@dataclass
class CoinHeadConfig:
    # Input dim (per node/time)
    in_dim: int

    # MLP head
    hidden_dim: int | None = None  # if None or 0 -> linear head
    num_layers: int = 1  # used only if hidden_dim is set
    activation: str = "gelu"
    dropout: float = 0.0
    layernorm: bool = True  # LayerNorm on the 3-angle output

    # Output scaling & stabilization
    out_scale: float = 1.0  # multiply logits before wrap
    weightnorm: bool = False  # apply weight norm on final projection
    init_small: bool = False  # smaller output init for stable early steps

    # Angle coupling (mix [α,β,γ] via learnable lower-triangular)
    angle_coupling: bool = False

    # Temperature (per-angle or scalar). If learnable, initialized to 1.0.
    learnable_temp: bool = False
    fixed_temp: float = 1.0

    # Exploration noise (applied to pre-wrap outputs during training)
    noise_std: float = 0.0  # e.g., 0.01 for mild exploration


class _LowerTriangularMixer(nn.Module):
    """
    Learnable lower-triangular 3x3 mixer with ones on diagonal at init.
    y = L @ x, L lower-triangular.
    """

    def __init__(self):
        super().__init__()
        # Parametrize full matrix, mask as lower-triangular in forward.
        L = torch.eye(3, dtype=torch.float32)
        self.param = nn.Parameter(L)  # we'll mask on-the-fly

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3)
        L = torch.tril(self.param)
        return x @ L.T


class CoinHeadSU2(nn.Module):
    """
    Map per-(node,time) features to **Euler angles** (α, β, γ) for SU(2).

    Default behavior (backward-compatible with your tests):
      - single linear layer to 3 angles
      - optional LayerNorm over angle triplet
      - scale by `out_scale`
      - **STE-wrapped** to (-π, π]

    Optional upgrades (disabled by default):
      - deeper MLP with `hidden_dim` and `num_layers`
      - weight norm on the final linear layer
      - learnable temperature (per-angle) or fixed scalar temperature
      - angle coupling via learnable lower-triangular mixer
      - training-time Gaussian noise on pre-wrap logits
    """

    def __init__(self, cfg: CoinHeadConfig):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        in_dim = cfg.in_dim

        if cfg.hidden_dim and cfg.hidden_dim > 0 and cfg.num_layers > 0:
            for li in range(cfg.num_layers):
                layers.append(nn.Linear(in_dim, cfg.hidden_dim, bias=True))
                layers.append(_activation(cfg.activation))
                if cfg.dropout and cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
                in_dim = cfg.hidden_dim
            self.mlp = nn.Sequential(*layers)
            out_in = cfg.hidden_dim
        else:
            self.mlp = nn.Identity()
            out_in = cfg.in_dim

        # Final linear to 3 angles
        out = nn.Linear(out_in, 3, bias=True)
        if cfg.weightnorm:
            out = nn_utils.weight_norm(out)
        self.out = out

        # Output normalization over the 3 angles (pre-wrap)
        self.out_norm = nn.LayerNorm(3) if cfg.layernorm else nn.Identity()

        # Optional angle coupling
        self.mixer = _LowerTriangularMixer() if cfg.angle_coupling else nn.Identity()

        # Temperature
        if cfg.learnable_temp:
            self.temp = nn.Parameter(torch.ones(3, dtype=torch.float32))
        else:
            self.register_buffer(
                "temp", torch.tensor(float(cfg.fixed_temp), dtype=torch.float32), persistent=False
            )

        # Inits
        if isinstance(self.out, nn.Linear):
            if cfg.init_small:
                nn.init.xavier_uniform_(self.out.weight, gain=0.5)
            else:
                nn.init.xavier_uniform_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            # If weight_norm wraps it, .weight is a Parameter on the wrapper.
            # Use default init; it's fine in practice.
            pass

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (..., N, F)
        returns theta: (..., N, 3) in (-π, π] (wrapped with STE)
        """
        x = self.mlp(h)  # (..., N, H)
        theta = self.out(x)  # (..., N, 3)

        # Temperature (supports scalar or per-angle)
        if isinstance(self.temp, torch.Tensor) and self.temp.ndim == 0:
            theta = theta * (float(self.cfg.out_scale) * float(self.temp.item()))
        else:
            # Per-angle temperature broadcast
            theta = theta * float(self.cfg.out_scale)
            theta = theta * self.temp.view(*([1] * (theta.ndim - 1)), 3)

        theta = self.out_norm(theta)  # stabilize channels

        # Optional angle coupling (learnable lower-triangular mixing)
        theta = self.mixer(theta)  # (..., N, 3)

        # Exploration noise (train only), applied pre-wrap
        if self.training and self.cfg.noise_std and self.cfg.noise_std > 0.0:
            noise = torch.randn_like(theta) * float(self.cfg.noise_std)
            theta = theta + noise

        # --- Differentiable wrapping via straight-through estimator (STE) ---
        wrapped = wrap_to_pi(theta)
        theta = theta + (wrapped - theta).detach()

        return theta

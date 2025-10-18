# acpl/policy/pe.py
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

__all__ = ["TimeFourierPE", "TimeFourierPEConfig"]


@dataclass
class TimeFourierPEConfig:
    """
    Fourier-style time positional encoding configuration.

    Fields
    ------
    dim : even number of channels = 2 * (#frequencies)
    base : geometric span for frequencies; larger → wider spectrum
    learned_scale : learn a global scalar s so that t_eff = s * t

    Optional (stable defaults preserve old behavior):
    normalize : apply LayerNorm over the PE channels
    dropout   : dropout on PE (useful regularizer at the controller input)
    learned_phase : learn per-frequency phase φ_k (shifts sin/cos arguments)
    learned_freqs : learn per-frequency multipliers ω_k (initialized from base)
    """

    dim: int = 32
    base: float = 10000.0
    learned_scale: bool = True

    # extras (off by default; keep backward-compat)
    normalize: bool = False
    dropout: float = 0.0
    learned_phase: bool = False
    learned_freqs: bool = False


class TimeFourierPE(nn.Module):
    r"""
    Fourier time embeddings:
        PE_k(t) = [sin(ω_k t_eff + φ_k), cos(ω_k t_eff + φ_k)],
        with t_eff = s * t, where s is learned if `learned_scale=True`.

    Notes
    -----
    * This module is intentionally **stateless wrt T**; you can call it for any T.
    * Defaults reproduce the previous implementation exactly (learned global scale,
      fixed log-spaced ω_k, zero phase, no norm/dropout).
    * We keep the simple call contract used by policy:
          pe = TimeFourierPE(...)(T, device=X.device)  # (T, dim)
      You may also pass `dtype` explicitly if needed.
    """

    def __init__(self, cfg: TimeFourierPEConfig):
        super().__init__()
        if cfg.dim % 2 != 0:
            raise ValueError("TimeFourierPE dim must be even.")
        self.cfg = cfg

        # scale s (global)
        if cfg.learned_scale:
            self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)

        d = cfg.dim // 2

        # base frequencies ω_k (log-spaced), in float32 (moved to device on forward)
        # ω_k = base^{-k/(d-1)}  for k=0..d-1  (robust when d==1)
        k = torch.arange(d, dtype=torch.float32)
        denom = max(d - 1, 1)
        w0 = torch.exp(-k * math.log(cfg.base) / denom)  # shape: (d,)

        if cfg.learned_freqs:
            self.freqs = nn.Parameter(w0)  # learnable ω_k
        else:
            self.register_buffer("freqs", w0, persistent=False)

        if cfg.learned_phase:
            self.phase = nn.Parameter(torch.zeros(d, dtype=torch.float32))  # φ_k
        else:
            self.register_buffer("phase", torch.zeros(d, dtype=torch.float32), persistent=False)

        self.norm = nn.LayerNorm(cfg.dim) if cfg.normalize else nn.Identity()
        self.do = nn.Dropout(cfg.dropout) if cfg.dropout and cfg.dropout > 0 else nn.Identity()

    def forward(
        self,
        T: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Build PE for time indices t = 0,1,...,T-1

        Returns
        -------
        pe : (T, dim) tensor on the requested device/dtype
        """
        if T <= 0:
            raise ValueError("T must be positive.")

        d = self.cfg.dim // 2

        # Ensure parameters/buffers are on the target device/dtype
        scale = self.scale.to(device=device, dtype=dtype)
        freqs = self.freqs.to(device=device, dtype=dtype)
        phase = self.phase.to(device=device, dtype=dtype)

        t = torch.arange(T, dtype=dtype, device=device) * scale  # (T,)
        # arg = t[:,None] * ω[None,:] + φ[None,:]
        arg = t.unsqueeze(1) * freqs.unsqueeze(0) + phase.unsqueeze(0)  # (T, d)

        pe = torch.cat([torch.sin(arg), torch.cos(arg)], dim=1)  # (T, 2d)
        pe = self.do(pe)
        pe = self.norm(pe)
        return pe

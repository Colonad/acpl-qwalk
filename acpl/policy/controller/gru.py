# acpl/policy/controller/gru.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

__all__ = ["GRUController", "GRUControllerConfig"]


@dataclass
class GRUControllerConfig:
    # Input dims
    in_dim: int  # expected feature dim into the GRU (Dz+Pt if concat)
    hidden_dim: int  # GRU hidden size (per direction)
    num_layers: int = 1
    bidirectional: bool = False

    # Regularization / stabilization
    dropout: float = 0.0  # inter-layer dropout in GRU (PyTorch semantics)
    layernorm: bool = True  # LN on outputs per time step
    zoneout: float = 0.0  # probability to keep previous h_t (0 disables)
    clamp_hidden_norm: float = 0.0  # >0 to clamp hidden vector norms

    # Input preprocessing
    conditioning: Literal["concat", "film"] = "concat"
    in_proj_dim: int = 0  # 0 to disable; else project to this dim + LN + GELU
    input_dropout: float = 0.0  # dropout on the per-node/per-time input before GRU

    # Initialization
    learnable_h0: bool = False  # learn an h0 per layer (and direction)
    init_orthogonal: bool = True  # orthogonal recurrent weights; xavier input weights
    bias_carry_init: float = 1.0  # positive bias on update gate to promote carry-through


class _InputBlock(nn.Module):
    """
    Build (N, T, Fin) from node embeddings z (N, Dz) and time encodings tpe (T, Pt).

    Modes:
      - 'concat' : [z, tpe]            -> shape (N,T,Dz+Pt)  (default)
      - 'film'   : FiLM(z; tpe) = γ(tpe) ⊙ z + β(tpe), with global γ, β per time step
                    (broadcast across nodes). Produces (N,T,Dz).
    Optional projection to `in_proj_dim` + GELU + LayerNorm for scale control.
    """

    def __init__(
        self,
        in_dim_concat: int,
        dz: int,
        conditioning: str,
        in_proj_dim: int,
        input_dropout: float,
        use_layernorm: bool,
    ):
        super().__init__()
        self.conditioning = conditioning
        self.dz = dz
        self.use_proj = in_proj_dim > 0
        self.do = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()

        if conditioning == "film":
            # γ, β: linear maps from Pt -> Dz (applied per time step, broadcast over nodes)
            self.gamma = nn.Linear(in_dim_concat - dz, dz, bias=True)
            self.beta = nn.Linear(in_dim_concat - dz, dz, bias=True)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.bias)
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.beta.weight)
            fin = dz
        elif conditioning == "concat":
            fin = in_dim_concat
        else:
            raise ValueError("conditioning must be 'concat' or 'film'")

        if self.use_proj:
            self.proj = nn.Linear(fin, in_proj_dim, bias=True)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.act = nn.GELU()
            self.norm = nn.LayerNorm(in_proj_dim) if use_layernorm else nn.Identity()
            self.out_dim = in_proj_dim
        else:
            self.proj = None
            self.norm = nn.Identity()
            self.act = nn.Identity()
            self.out_dim = fin

    def forward(self, z: torch.Tensor, tpe: torch.Tensor) -> torch.Tensor:
        """
        z:   (N, Dz)
        tpe: (T, Pt)
        return x: (N, T, F_in)
        """
        N, Dz = z.shape
        T, Pt = tpe.shape

        if self.conditioning == "concat":
            z_rep = z.unsqueeze(1).expand(N, T, Dz)  # (N,T,Dz)
            t_rep = tpe.unsqueeze(0).expand(N, T, Pt)  # (N,T,Pt)
            x = torch.cat([z_rep, t_rep], dim=-1)  # (N,T,Dz+Pt)
        else:  # FiLM
            gamma = self.gamma(tpe)  # (T, Dz)
            beta = self.beta(tpe)  # (T, Dz)
            x = (gamma.unsqueeze(0) * z.unsqueeze(1)) + beta.unsqueeze(0)  # (N,T,Dz)

        if self.proj is not None:
            x = self.proj(x)
            x = self.act(x)
            x = self.norm(x)

        x = self.do(x)
        return x  # (N,T,Fin)


def _zoneout(h_new: torch.Tensor, h_prev: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0 or not h_new.requires_grad:
        return h_new
    if not h_new.is_cuda and h_new.dtype == torch.float16:
        # Rare corner, but keep sane: fallback to float32 mask
        mask = (torch.rand(h_new.shape, dtype=torch.float32, device=h_new.device) > p).to(
            h_new.dtype
        )
    else:
        mask = (torch.rand_like(h_new) > p).to(h_new.dtype)
    # y = m ⊙ h_new + (1-m) ⊙ h_prev
    return mask * h_new + (1.0 - mask) * h_prev


def _clamp_hidden(h: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return h
    # Clamp per-vector norms across (layers*dirs, N, hidden)
    flat = h.view(-1, h.size(-1))
    norms = flat.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(max_norm / norms, max=1.0)
    flat = flat * scale
    return flat.view_as(h)


class GRUController(nn.Module):
    """
    Node-wise GRU controller over **time**, with stability + conditioning extras.

    Inputs
    ------
    - z:   (N, Dz)  node embeddings (constant across time within an episode)
    - tpe: (T, Pt)  time positional encodings

    Output
    ------
    - H:   (T, N, Dh_eff)   Dh_eff = hidden_dim * (2 if bidirectional else 1)
    """

    def __init__(self, cfg: GRUControllerConfig):
        super().__init__()
        self.cfg = cfg

        # Input builder
        dz_plus_pt = cfg.in_dim  # caller passes Dz+Pt for concat; Dz for film path after FiLM
        # We need Dz to build FiLM; derive Dz conservatively:
        # If using FiLM, require that cfg.in_dim >= 2 and treat z_dim from runtime.
        # At build time, keep placeholder; we won't create FiLM layers until we see input.
        # To keep things simple and static, we expect: if conditioning='film',
        # 'in_dim' must equal Dz + Pt, and Dz will be inferred on first call.
        self._input_block: _InputBlock | None = None
        self._input_block_built = False

        # Recurrent core
        self.gru = nn.GRU(
            input_size=cfg.in_dim if cfg.in_proj_dim == 0 else cfg.in_proj_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,  # (N,T,F)
            dropout=(cfg.dropout if cfg.num_layers > 1 else 0.0),
            bidirectional=cfg.bidirectional,
        )
        self.out_norm = (
            nn.LayerNorm(cfg.hidden_dim * (2 if cfg.bidirectional else 1))
            if cfg.layernorm
            else nn.Identity()
        )

        # Learnable initial hidden state
        if cfg.learnable_h0:
            num_dirs = 2 if cfg.bidirectional else 1
            self.h0 = nn.Parameter(torch.zeros(cfg.num_layers * num_dirs, 1, cfg.hidden_dim))
        else:
            self.register_parameter("h0", None)

        # Initialize weights/biases for stability
        self._init_parameters()

    def _build_input_block(self, z: torch.Tensor, tpe: torch.Tensor):
        if self._input_block_built:
            return
        Dz = z.size(-1)
        Pt = tpe.size(-1)
        if self.cfg.conditioning == "film":
            in_dim_concat = Dz + Pt
            self._input_block = _InputBlock(
                in_dim_concat=in_dim_concat,
                dz=Dz,
                conditioning="film",
                in_proj_dim=self.cfg.in_proj_dim,
                input_dropout=self.cfg.input_dropout,
                use_layernorm=self.cfg.layernorm,
            )
        else:
            # concat
            in_dim_concat = Dz + Pt
            self._input_block = _InputBlock(
                in_dim_concat=in_dim_concat,
                dz=Dz,
                conditioning="concat",
                in_proj_dim=self.cfg.in_proj_dim,
                input_dropout=self.cfg.input_dropout,
                use_layernorm=self.cfg.layernorm,
            )

        # If a projection is used, the GRU input_size must match it.
        if self.cfg.in_proj_dim > 0 and self.gru.input_size != self.cfg.in_proj_dim:
            # Rebuild GRU with correct input size (rare if user changed cfg)
            new_gru = nn.GRU(
                input_size=self.cfg.in_proj_dim,
                hidden_size=self.cfg.hidden_dim,
                num_layers=self.cfg.num_layers,
                batch_first=True,
                dropout=(self.cfg.dropout if self.cfg.num_layers > 1 else 0.0),
                bidirectional=self.cfg.bidirectional,
            )
            # reinit
            self.gru = new_gru
            self._init_parameters()

        self._input_block_built = True

    def _init_parameters(self):
        cfg = self.cfg
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                if cfg.init_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # GRU gate order in PyTorch: r, z, n
                # Encourage carry-through at init by making z-gate bias positive.
                # Split into three segments and bump the z segment.
                hidden = cfg.hidden_dim
                # bias shape = 3*hidden
                param.data[hidden : 2 * hidden].add_(cfg.bias_carry_init)

        # h0 already zero-initialized if present.

    def _maybe_make_h0(self, batch_N: int, device, dtype):
        if self.h0 is None:
            return None
        # Tile learnable h0 across N
        return self.h0.to(device=device, dtype=dtype).expand(-1, batch_N, -1).contiguous()

    def forward(
        self,
        z: torch.Tensor,  # (N, Dz)
        tpe: torch.Tensor,  # (T, Pt)
        h0: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,  # (T,) or (N,T) boolean; True=keep/valid
    ) -> torch.Tensor:
        """
        Returns H: (T, N, Dh_eff)
        """
        # Build input block on first call (requires Dz/Pt)
        self._build_input_block(z, tpe)
        x_ntf = self._input_block(z, tpe)  # (N,T,Fin)

        # Prepare initial hidden
        H0 = h0 if h0 is not None else self._maybe_make_h0(x_ntf.size(0), x_ntf.device, x_ntf.dtype)

        # Run GRU
        out, hN = self.gru(x_ntf, H0)  # out: (N,T,Dh_eff)

        # Optional zoneout on the output trajectory (w.r.t. previous time)
        if self.training and self.cfg.zoneout > 0.0 and out.size(1) > 1:
            # apply between consecutive steps
            h_prev = out[:, :-1, :].detach()  # do not backprop through carried value
            h_new = out[:, 1:, :]
            mixed = _zoneout(h_new, h_prev, p=self.cfg.zoneout)
            out = torch.cat([out[:, :1, :], mixed], dim=1)

        # Optional per-step mask (if ragged timestamps)
        if mask is not None:
            if mask.dim() == 1:  # (T,)
                m = mask.view(1, -1, 1).to(out.dtype)
            elif mask.dim() == 2:  # (N,T)
                m = mask.transpose(0, 1).unsqueeze(-1).to(out.dtype)  # -> (T,N,1) then back
                m = m.transpose(0, 1)  # (N,T,1)
            else:
                raise ValueError("mask must be (T,) or (N,T)")
            out = out * m

        # Clamp hidden norms (stability under long horizons)
        if self.cfg.clamp_hidden_norm and self.cfg.clamp_hidden_norm > 0:
            out = _clamp_hidden(out, self.cfg.clamp_hidden_norm)

        out = self.out_norm(out)  # (N,T,Dh_eff)
        return out.transpose(0, 1).contiguous()  # (T,N,Dh_eff)

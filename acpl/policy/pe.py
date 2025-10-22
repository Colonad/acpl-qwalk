# acpl/policy/pe.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn

__all__ = [
    # Time PE (kept backward-compatible)
    "TimeFourierPE",
    "TimeFourierPEConfig",
    # Node PE wrapper (Phase B3)
    "NodePEConfig",
    "NodePE",
]


# ---------------------------------------------------------------------------
# Time positional encodings (kept exactly backward-compatible with Phase A)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Node positional encoding wrapper (Phase B3)
#   Switch over {none, coords, LapPE, roles} with ablation toggles
# ---------------------------------------------------------------------------


def _mlp(
    sizes: list[int], act: str = "gelu", dropout: float = 0.0, layernorm: bool = False
) -> nn.Sequential:
    assert len(sizes) >= 2
    acts = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "identity": nn.Identity,
    }
    A = acts.get(act, nn.GELU)
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
        if layernorm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(A())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
    return nn.Sequential(*layers)


def _build_sym_norm_laplacian(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.

    Assumes edge_index is (2, E) with integer indices in [0, N).
    Automatically symmetrizes A.
    """
    N = int(num_nodes)
    if edge_index.numel() == 0:
        I = torch.eye(N, dtype=dtype, device=device)
        return I  # isolated graph

    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    if edge_weight is None:
        w = torch.ones_like(src, dtype=dtype)
    else:
        w = edge_weight.to(dtype=dtype)

    # Symmetrize (if duplicates exist, they are accumulated)
    idx_i = torch.cat([src, dst], dim=0)
    idx_j = torch.cat([dst, src], dim=0)
    vals = torch.cat([w, w], dim=0)

    A = torch.zeros((N, N), dtype=dtype, device=device)
    A.index_put_((idx_i, idx_j), vals, accumulate=True)

    # Degree and normalization
    deg = A.sum(dim=1)  # (N,)
    # avoid divide by zero
    d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    I = torch.eye(N, dtype=dtype, device=device)
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    # Numerical symmetrization (guards small asymmetry)
    L = 0.5 * (L + L.transpose(0, 1))
    return L


def _eig_lap_pe(
    L: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute k smallest eigenpairs of symmetric normalized Laplacian L.
    Returns (eigvecs [N,k], eigvals [k]) sorted ascending.

    Notes
    -----
    * We use dense eigh (sufficient for small/medium tests).
    * If graph is small or k > N, we clamp safely.
    * Sign of eigenvectors is stabilized by flipping each column so that
      the entry with largest magnitude has non-negative sign.
    """
    N = L.size(0)
    if k <= 0 or N == 0:
        return L.new_zeros((N, 0)), L.new_zeros((0,))

    k_eff = min(k, N)
    # Full eigh is fine here; it is differentiable in PyTorch
    evals, evecs = torch.linalg.eigh(L)  # ascending

    # take k smallest
    evals_k = evals[:k_eff].contiguous()
    evecs_k = evecs[:, :k_eff].contiguous()

    # Stabilize signs (per column)
    if k_eff > 0:
        idx_max = torch.argmax(evecs_k.abs(), dim=0)  # [k]
        signs = torch.sign(evecs_k.gather(0, idx_max.unsqueeze(0)).squeeze(0))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)  # avoid zero sign
        evecs_k = evecs_k * signs.unsqueeze(0)

    return evecs_k, evals_k


def _degree_features(
    num_nodes: int,
    edge_index: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Simple role/structural statistics:
      - deg
      - sqrt(deg)
      - log1p(deg)
      - normed degree (deg / max(deg,1))
      - stationary RW approx (deg / sum(deg))
    """
    N = int(num_nodes)
    if edge_index.numel() == 0:
        return torch.zeros((N, 5), device=device, dtype=dtype)

    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)

    deg = torch.zeros(N, device=device, dtype=dtype)
    deg.index_add_(0, src, torch.ones_like(src, dtype=dtype))
    # undirected assumption (common in our tests) is fine — edge_index usually has both dirs

    deg_clamped = torch.clamp(deg, min=0)
    sqrt_deg = torch.sqrt(deg_clamped)
    log_deg = torch.log1p(deg_clamped)
    max_deg = torch.clamp(deg_clamped.max(), min=torch.tensor(1.0, device=device, dtype=dtype))
    norm_deg = deg_clamped / max_deg
    sum_deg = torch.clamp(deg_clamped.sum(), min=torch.tensor(1.0, device=device, dtype=dtype))
    rw_stat = deg_clamped / sum_deg

    feats = torch.stack([deg_clamped, sqrt_deg, log_deg, norm_deg, rw_stat], dim=1)  # (N, 5)
    return feats


@dataclass
class NodePEConfig:
    """
    Full node positional encoding wrapper (Phase B3).

    mode:
        - "none"    : no PE (returns zeros of shape [N, dim] unless passthrough set)
        - "coords"  : use provided coordinates `coords` in forward(); projects to dim
        - "lappe"   : Laplacian eigenvectors (k = lap_k), then project to dim
        - "roles"   : simple role/structural features (degree stats), then project to dim

    Common:
        dim            : output PE dimension
        dropout        : dropout applied after projection (ablation)
        layernorm      : LayerNorm on the output (ablation)
        act            : activation inside the projection MLP
        proj_layers    : number of layers in the projection MLP (>=1). 1 means linear.

    LapPE-specific:
        lap_k          : number of eigenvectors to use
        include_evals  : if True, append eigenvalues broadcasted as features
        sincos         : if True, apply sin/cos to eigenvectors (doubles channels)
        safe_fallback  : if eig fails (rare), return zeros instead of raising

    Coords-specific:
        coords_in_dim  : expected coordinate dimension; if None, inferred at first call

    Extras:
        gate_x         : if True, also return a learned gate α in (0,1) to blend with X upstream
    """

    mode: Literal["none", "coords", "lappe", "roles"] = "none"
    dim: int = 32

    # generic projection
    dropout: float = 0.0
    layernorm: bool = False
    act: str = "gelu"
    proj_layers: int = 2

    # LapPE
    lap_k: int = 8
    include_evals: bool = False
    sincos: bool = False
    safe_fallback: bool = True

    # Coords
    coords_in_dim: int | None = None

    # extras
    gate_x: bool = False


class NodePE(nn.Module):
    """
    Compute node positional encodings with a switchable backend:
      {none, coords, LapPE, roles}. The output has fixed dimension `cfg.dim`.

    Contract
    --------
    forward(X, edge_index, *, coords=None, edge_weight=None) -> PE (N, dim)  [and optional gate]

    - X is unused here (left for future tasks like feature-dependent roles),
      kept for ergonomic parity with encoders that call into PE wrapper.
    - coords: required if mode == "coords".
    - For LapPE, we compute on the **symmetric normalized** Laplacian.

    If cfg.gate_x is True, forward returns a tuple (pe, alpha) where
      alpha ∈ (0,1)^{N×1} is a learned gate that can be used upstream to blend:
        X_aug = (1 - α) * X  +  α * f(pe)
    """

    def __init__(self, cfg: NodePEConfig):
        super().__init__()
        self.cfg = cfg

        # We'll build a small projection MLP lazily once we know the input feature width
        self._proj_in_dim: int | None = None
        self.proj: nn.Module = nn.Identity()

        if cfg.gate_x:
            # gate from projected PE (scalar per node)
            self.gate = nn.Sequential(
                nn.Linear(cfg.dim, 1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.gate = nn.Identity()

        self.out_ln = nn.LayerNorm(cfg.dim) if cfg.layernorm else nn.Identity()
        self.out_drop = (
            nn.Dropout(cfg.dropout) if cfg.dropout and cfg.dropout > 0 else nn.Identity()
        )

    # ---------------- internal helpers ---------------- #

    def _maybe_build_proj(self, in_dim: int):
        if self._proj_in_dim == in_dim:
            return
        self._proj_in_dim = in_dim

        if self.cfg.proj_layers <= 1:
            self.proj = nn.Linear(in_dim, self.cfg.dim, bias=True)
        else:
            # simple width schedule: keep hidden width near output dim
            hidden = max(self.cfg.dim, in_dim)
            sizes = [in_dim] + [hidden] * (self.cfg.proj_layers - 2) + [self.cfg.dim]
            self.proj = _mlp(sizes, act=self.cfg.act, dropout=self.cfg.dropout, layernorm=False)

    def _coords_pe(
        self,
        coords: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if coords is None:
            raise ValueError("coords must be provided when NodePE.mode='coords'.")
        if coords.ndim != 2:
            raise ValueError("coords must be (N, C).")
        if self.cfg.coords_in_dim is not None and coords.size(1) != self.cfg.coords_in_dim:
            raise ValueError(
                f"coords has dim {coords.size(1)} but cfg.coords_in_dim={self.cfg.coords_in_dim}."
            )

        feats = coords.to(device=device, dtype=dtype)  # (N, C)
        return feats

    def _lap_pe(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        L = _build_sym_norm_laplacian(
            num_nodes, edge_index, edge_weight, device=device, dtype=dtype
        )
        k = int(self.cfg.lap_k)
        if k <= 0:
            return torch.zeros((num_nodes, 0), device=device, dtype=dtype)

        try:
            evecs, evals = _eig_lap_pe(L, k)
        except Exception:
            if self.cfg.safe_fallback:
                return torch.zeros((num_nodes, k), device=device, dtype=dtype)
            raise

        feats = evecs  # (N, k)
        if self.cfg.sincos:
            feats = torch.cat([torch.sin(feats), torch.cos(feats)], dim=1)  # (N, 2k)
        if self.cfg.include_evals:
            # broadcast eigenvalues as features (N,k or N,2k depending on sincos)
            if self.cfg.sincos:
                k_eff = evecs.size(1) * 2
                evals_rep = torch.repeat_interleave(evals, repeats=2)  # match channels
            else:
                k_eff = evecs.size(1)
                evals_rep = evals
            feats = torch.cat([feats, evals_rep.unsqueeze(0).expand(num_nodes, k_eff)], dim=1)
        return feats.to(device=device, dtype=dtype)

    def _roles_pe(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        feats = _degree_features(num_nodes, edge_index, device=device, dtype=dtype)  # (N, 5)
        return feats

    # ---------------- public API ---------------- #

    def forward(
        self,
        X: torch.Tensor,  # (N, Fin) — not used, reserved
        edge_index: torch.Tensor,  # (2, E)
        *,
        coords: torch.Tensor | None = None,  # (N, C), if mode="coords"
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Return node PE of shape (N, dim). If cfg.gate_x=True, also returns α ∈ (0,1)^{N×1}.
        """
        device = X.device
        dtype = X.dtype
        N = X.size(0)

        mode = self.cfg.mode.lower()
        if mode not in {"none", "coords", "lappe", "roles"}:
            raise ValueError(f"Unknown NodePE.mode={self.cfg.mode!r}")

        # Collect raw features according to the selected mode
        if mode == "none":
            raw = torch.zeros((N, 1), device=device, dtype=dtype)  # minimal constant channel
        elif mode == "coords":
            raw = self._coords_pe(coords, device=device, dtype=dtype)
        elif mode == "lappe":
            raw = self._lap_pe(N, edge_index, edge_weight, device=device, dtype=dtype)
        else:  # roles
            raw = self._roles_pe(N, edge_index, device=device, dtype=dtype)

        # Project to requested output dimension
        self._maybe_build_proj(in_dim=int(raw.size(1)))
        pe = self.proj(raw)

        pe = self.out_drop(pe)
        pe = self.out_ln(pe)

        if isinstance(self.gate, nn.Identity):
            return pe
        else:
            alpha = self.gate(pe)  # (N, 1) in (0,1)
            return pe, alpha


# EOF

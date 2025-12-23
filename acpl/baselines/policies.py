# acpl/baselines/policies.py
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, MutableMapping, Optional, Protocol, Sequence

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)

__all__ = [
    # Core policy-like wrapper
    "BaselinePolicyConfig",
    "BaselinePolicy",
    "CoinScheduleLike",
    "build_baseline_policy",
    # Embeddings adapter (for eval/plots)
    "NodeEmbeddingConfig",
    "NodeEmbeddingAdapter",
    # Small graph helpers
    "infer_out_degree",
    "infer_in_degree",
]


# =============================================================================
# Interfaces
# =============================================================================


class CoinScheduleLike(Protocol):
    """
    A coin baseline/schedule that can be *called* to produce per-node, per-time
    coin parameters (theta) in the same format as the learned policy emits.

    The project’s simulator typically expects:
        theta: (T, N, P) float tensor

    BUT we keep this wrapper robust: we will try multiple call signatures
    (with kwargs, with positional args, with/out degrees).

    Recommended schedule signature:
        schedule(X, edge_index, *, T: int, edge_weight=None, outdeg=None, indeg=None) -> theta
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...


# =============================================================================
# Degree helpers (needed by many baselines, and also useful for logging)
# =============================================================================


def infer_out_degree(edge_index: Tensor, *, num_nodes: int) -> Tensor:
    """
    Compute out-degree from a PyG-style edge_index (2, E).
    If your graph is undirected and stored as *both* directions, this equals degree.

    Returns:
        outdeg: (N,) long tensor
    """
    if not isinstance(edge_index, Tensor):
        raise TypeError(f"edge_index must be a torch.Tensor, got {type(edge_index)}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be shape (2,E), got {tuple(edge_index.shape)}")
    src = edge_index[0].to(dtype=torch.long)
    outdeg = torch.bincount(src, minlength=int(num_nodes))
    return outdeg


def infer_in_degree(edge_index: Tensor, *, num_nodes: int) -> Tensor:
    """
    Compute in-degree from edge_index (2, E).

    Returns:
        indeg: (N,) long tensor
    """
    if not isinstance(edge_index, Tensor):
        raise TypeError(f"edge_index must be a torch.Tensor, got {type(edge_index)}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be shape (2,E), got {tuple(edge_index.shape)}")
    dst = edge_index[1].to(dtype=torch.long)
    indeg = torch.bincount(dst, minlength=int(num_nodes))
    return indeg


# =============================================================================
# Embeddings adapter (so eval harness can always call model.encode_nodes)
# =============================================================================


@dataclass(frozen=True)
class NodeEmbeddingConfig:
    """
    Controls what BaselinePolicy.encode_nodes returns.

    Why this exists:
      - Your eval harness extracts embeddings for plots/stats.
      - Baselines often don’t have an internal GNN module to hook.
      - So we provide a deterministic, explicit encode_nodes implementation.

    Modes
    -----
    - "identity": returns X (optionally cast / normalized)
    - "linear_fixed": fixed random linear projection (deterministic by seed)
    - "mlp_fixed": fixed random 2-layer MLP (deterministic by seed)

    Notes:
      - These are NOT trained (requires_grad=False).
      - They exist for analysis/visualization consistency across runs/seeds.
    """

    mode: Literal["identity", "linear_fixed", "mlp_fixed"] = "identity"
    out_dim: int | None = None  # if None, uses X.shape[1] for identity, else required for projections
    seed: int = 0
    normalize: bool = False  # L2 normalize embeddings per node
    dtype: torch.dtype = torch.float32


class NodeEmbeddingAdapter(nn.Module):
    """
    Deterministic node embedding adaptor for baseline policies.

    This module is *frozen* by construction (no gradients), so it won’t interfere
    with “baseline-ness” while still enabling embeddings plots/statistics.
    """

    def __init__(self, in_dim: int, cfg: NodeEmbeddingConfig):
        super().__init__()
        self.cfg = cfg
        self.in_dim = int(in_dim)

        mode = cfg.mode
        if mode == "identity":
            self.out_dim = self.in_dim if cfg.out_dim is None else int(cfg.out_dim)
            if self.out_dim != self.in_dim:
                # allow identity-with-pad via fixed projection (still deterministic)
                self._proj = self._make_fixed_linear(self.in_dim, self.out_dim, seed=cfg.seed)
            else:
                self._proj = None
            return

        if cfg.out_dim is None:
            raise ValueError(f"NodeEmbeddingConfig.out_dim is required for mode='{mode}'")

        self.out_dim = int(cfg.out_dim)

        if mode == "linear_fixed":
            self.lin = self._make_fixed_linear(self.in_dim, self.out_dim, seed=cfg.seed)
            return

        if mode == "mlp_fixed":
            hidden = max(8, min(4 * self.out_dim, 512))
            self.lin1 = self._make_fixed_linear(self.in_dim, hidden, seed=cfg.seed + 1)
            self.lin2 = self._make_fixed_linear(hidden, self.out_dim, seed=cfg.seed + 2)
            return

        raise ValueError(f"Unknown embedding mode: {mode}")

    @staticmethod
    def _make_fixed_linear(in_dim: int, out_dim: int, *, seed: int) -> nn.Linear:
        g = torch.Generator()
        g.manual_seed(int(seed))
        lin = nn.Linear(in_dim, out_dim, bias=True)
        with torch.no_grad():
            # Xavier-like scaling, deterministic
            w = torch.randn((out_dim, in_dim), generator=g) / max(1.0, in_dim**0.5)
            b = torch.zeros((out_dim,), dtype=w.dtype)
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        for p in lin.parameters():
            p.requires_grad_(False)
        return lin

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            raise ValueError(f"Expected X (N,F), got {tuple(X.shape)}")

        X = X.to(dtype=self.cfg.dtype)

        if self.cfg.mode == "identity":
            if getattr(self, "_proj", None) is not None:
                Z = self._proj(X)
            else:
                Z = X
        elif self.cfg.mode == "linear_fixed":
            Z = self.lin(X)
        elif self.cfg.mode == "mlp_fixed":
            Z = torch.tanh(self.lin1(X))
            Z = self.lin2(Z)
        else:
            raise RuntimeError("Unreachable: unknown embedding mode.")

        if self.cfg.normalize:
            Z = torch.nn.functional.normalize(Z, p=2.0, dim=1, eps=1e-12)
        return Z


# =============================================================================
# BaselinePolicy wrapper
# =============================================================================


@dataclass(frozen=True)
class BaselinePolicyConfig:
    """
    High-level wrapper config. The actual *coin math* lives in the schedule object.

    - policy returns theta of shape (T,N,P), like ACPLPolicy does.
    - encode_nodes returns embeddings for eval harness (plots/stats).

    Important:
      - This wrapper does NOT assume a particular theta parameterization
        (SU(2) vs exp vs Cayley). That is the schedule’s responsibility.
      - This keeps baselines future-proof if you change coin lifts.
    """

    name: str = "baseline"
    # If True, we validate theta shape at runtime.
    strict_theta_shape: bool = True

    # Optional scaling/noise to mimic “policy output post-processing”.
    theta_scale: float = 1.0
    theta_noise_std: float = 0.0
    theta_clip: float | None = None

    # Output dtype
    theta_dtype: torch.dtype = torch.float32

    # Embeddings
    embedding: NodeEmbeddingConfig = NodeEmbeddingConfig()


def _call_schedule_robust(
    schedule: CoinScheduleLike,
    *,
    X: Tensor,
    edge_index: Tensor,
    T: int,
    edge_weight: Tensor | None = None,
    outdeg: Tensor | None = None,
    indeg: Tensor | None = None,
) -> Tensor:
    """
    Call a schedule using a best-effort set of signatures.

    This allows your coins.py to evolve without breaking policies.py,
    and also allows quick experiments with simple callables.

    Preferred call:
        schedule(X, edge_index, T=T, edge_weight=edge_weight, outdeg=outdeg, indeg=indeg)

    Fallbacks try removing kwargs progressively or using positional args.
    """
    # Try the “rich” kwargs form first.
    try:
        return schedule(
            X,
            edge_index,
            T=int(T),
            edge_weight=edge_weight,
            outdeg=outdeg,
            indeg=indeg,
        )
    except TypeError:
        pass

    # Without indeg/outdeg
    try:
        return schedule(X, edge_index, T=int(T), edge_weight=edge_weight)
    except TypeError:
        pass

    # Without edge_weight
    try:
        return schedule(X, edge_index, T=int(T))
    except TypeError:
        pass

    # Positional T
    try:
        return schedule(X, edge_index, int(T))
    except TypeError:
        pass

    # Just (T, N) (very simple schedules)
    try:
        return schedule(int(T), int(X.shape[0]))
    except TypeError as e:
        raise TypeError(
            "Could not call coin schedule with any supported signature. "
            "Expected something like schedule(X, edge_index, *, T=...). "
            f"Schedule type: {type(schedule)}"
        ) from e


class BaselinePolicy(nn.Module):
    """
    Wrap a baseline coin schedule into a “policy-like” nn.Module.

    Public contract (matches the learned policy call-site):
        theta = model(X, edge_index, T=T, edge_weight=edge_weight)

    Additionally, for your evaluation harness:
        emb = model.encode_nodes(X, edge_index, T=T, edge_weight=edge_weight)

    This means your embeddings extractor can use method="encode_nodes" for baselines,
    avoiding fragile hook-based logic.
    """

    def __init__(self, schedule: CoinScheduleLike, *, cfg: BaselinePolicyConfig):
        super().__init__()
        self.schedule = schedule
        self.cfg = cfg

        # Lazy-initialized embedding adapter (depends on X feature dim).
        self._emb_adapter: NodeEmbeddingAdapter | None = None
        self._emb_in_dim: int | None = None

    def _get_emb_adapter(self, X: Tensor) -> NodeEmbeddingAdapter:
        if X.ndim != 2:
            raise ValueError(f"Expected X (N,F), got {tuple(X.shape)}")
        F = int(X.shape[1])
        if self._emb_adapter is None or self._emb_in_dim != F:
            self._emb_adapter = NodeEmbeddingAdapter(F, self.cfg.embedding)
            self._emb_in_dim = F
        return self._emb_adapter

    @torch.no_grad()
    def encode_nodes(
        self,
        X: Tensor,
        edge_index: Tensor,
        *,
        T: int | None = None,
        edge_weight: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        """
        Baseline embedding extraction.

        We intentionally do NOT depend on edge_index/T here, but we accept them
        so callers can use a uniform interface across learned models and baselines.
        """
        adapter = self._get_emb_adapter(X)
        return adapter(X)

    def forward(
        self,
        X: Tensor,
        edge_index: Tensor,
        *,
        T: int,
        edge_weight: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Emit baseline coin parameters theta.

        Expected output:
            theta: (T, N, P)

        The actual meaning of P is schedule-dependent (SU(2) Euler angles, exp params, etc.).
        """
        if X.ndim != 2:
            raise ValueError(f"Expected X (N,F), got {tuple(X.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be (2,E), got {tuple(edge_index.shape)}")
        if not isinstance(T, int) or T <= 0:
            raise ValueError(f"T must be a positive int, got {T}")

        N = int(X.shape[0])

        # Degrees (helpful for many baselines; cheap enough to always compute)
        outdeg = infer_out_degree(edge_index, num_nodes=N)
        indeg = infer_in_degree(edge_index, num_nodes=N)

        # Let schedules be device/dtype-aware through X
        theta = _call_schedule_robust(
            self.schedule,
            X=X,
            edge_index=edge_index,
            T=T,
            edge_weight=edge_weight,
            outdeg=outdeg,
            indeg=indeg,
        )

        if not isinstance(theta, Tensor):
            raise TypeError(f"Schedule must return a torch.Tensor, got {type(theta)}")

        # Basic normalization: squeeze batch if someone returns (1,T,N,P)
        if theta.ndim >= 4 and theta.shape[0] == 1:
            theta = theta.squeeze(0)

        # Strict shape check (research-quality safety)
        if self.cfg.strict_theta_shape:
            if theta.ndim != 3:
                raise ValueError(
                    f"{self.cfg.name}: schedule returned theta with ndim={theta.ndim}, "
                    f"expected (T,N,P). Got {tuple(theta.shape)}"
                )
            if int(theta.shape[0]) != int(T):
                raise ValueError(
                    f"{self.cfg.name}: theta.shape[0] mismatch: got {theta.shape[0]} vs T={T}"
                )
            if int(theta.shape[1]) != int(N):
                raise ValueError(
                    f"{self.cfg.name}: theta.shape[1] mismatch: got {theta.shape[1]} vs N={N}"
                )

        # Post-processing: dtype, scaling, noise, clipping
        theta = theta.to(device=X.device, dtype=self.cfg.theta_dtype)

        if self.cfg.theta_scale != 1.0:
            theta = theta * float(self.cfg.theta_scale)

        if self.cfg.theta_noise_std > 0.0:
            # deterministic if caller sets torch seed; baseline can also be seeded inside schedule
            theta = theta + float(self.cfg.theta_noise_std) * torch.randn_like(theta)

        if self.cfg.theta_clip is not None:
            c = float(self.cfg.theta_clip)
            theta = theta.clamp(min=-c, max=c)

        return theta

    def extra_repr(self) -> str:
        return (
            f"name={self.cfg.name}, strict_theta_shape={self.cfg.strict_theta_shape}, "
            f"theta_scale={self.cfg.theta_scale}, theta_noise_std={self.cfg.theta_noise_std}, "
            f"theta_clip={self.cfg.theta_clip}, theta_dtype={self.cfg.theta_dtype}, "
            f"emb_mode={self.cfg.embedding.mode}, emb_out_dim={self.cfg.embedding.out_dim}, "
            f"emb_seed={self.cfg.embedding.seed}"
        )


# =============================================================================
# Builder / factory
# =============================================================================


def build_baseline_policy(
    *,
    schedule: CoinScheduleLike,
    cfg: BaselinePolicyConfig | None = None,
) -> BaselinePolicy:
    """
    Construct a BaselinePolicy from an explicit schedule object/callable.
    """
    cfg = cfg or BaselinePolicyConfig()
    return BaselinePolicy(schedule=schedule, cfg=cfg)


def _try_import_baseline_schedule_from_coins(
    kind: str,
    *,
    coins_kwargs: Mapping[str, Any] | None = None,
) -> CoinScheduleLike:
    """
    Convenience resolver that tries to construct a schedule from acpl/baselines/coins.py.

    We intentionally support multiple possible APIs so you can keep coins.py “research-flexible”.
    Supported patterns (any one is enough):

      1) coins.make_coin_schedule(kind, **kwargs) -> schedule
      2) coins.build(kind=..., **kwargs) -> schedule
      3) coins.<KindName>Schedule(**kwargs) -> schedule
      4) coins.<kind>(**kwargs) -> schedule (function returning schedule or schedule itself)

    If none found, raises a clear error.
    """
    coins_kwargs = dict(coins_kwargs or {})
    try:
        from acpl.baselines import coins as coins_mod  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import acpl.baselines.coins. "
            "If you want to auto-build schedules by name, ensure coins.py exists. "
            "Otherwise call build_baseline_policy(schedule=...)."
        ) from e

    # 1) make_coin_schedule
    fn = getattr(coins_mod, "make_coin_schedule", None)
    if callable(fn):
        sch = fn(kind, **coins_kwargs)
        return sch  # type: ignore[return-value]

    # 2) build
    fn = getattr(coins_mod, "build", None)
    if callable(fn):
        sch = fn(kind=kind, **coins_kwargs)
        return sch  # type: ignore[return-value]

    # 3) <KindName>Schedule class
    cls_name = "".join([p.capitalize() for p in kind.replace("-", "_").split("_")]) + "Schedule"
    cls = getattr(coins_mod, cls_name, None)
    if isinstance(cls, type):
        return cls(**coins_kwargs)  # type: ignore[return-value]

    # 4) coins.<kind>
    attr = getattr(coins_mod, kind, None)
    if callable(attr):
        # could return schedule or already be schedule-like
        try:
            sch = attr(**coins_kwargs)
            return sch  # type: ignore[return-value]
        except TypeError:
            return attr  # type: ignore[return-value]
    if attr is not None:
        return attr  # type: ignore[return-value]

    raise ValueError(
        f"Could not resolve baseline schedule '{kind}' from acpl.baselines.coins. "
        "Implement coins.make_coin_schedule(...) or a *Schedule class, or pass schedule explicitly."
    )


def build_baseline_policy_from_cfg(cfg: Mapping[str, Any]) -> BaselinePolicy:
    """
    Hydra-friendly constructor.

    Expected cfg format (example):
      cfg = {
        "name": "grover",
        "schedule": {"kind": "grover", "kwargs": {...}},
        "theta_scale": 1.0,
        "theta_noise_std": 0.0,
        "theta_clip": None,
        "embedding": {"mode": "identity", "out_dim": None, "seed": 0, "normalize": False},
      }

    Returns:
      BaselinePolicy
    """
    name = str(cfg.get("name", "baseline"))
    schedule_block = cfg.get("schedule", None)
    if schedule_block is None or not isinstance(schedule_block, Mapping):
        raise ValueError("cfg['schedule'] must be a mapping with at least {'kind': ...}")

    kind = str(schedule_block.get("kind"))
    coins_kwargs = schedule_block.get("kwargs", {}) or {}
    if not isinstance(coins_kwargs, Mapping):
        raise ValueError("cfg['schedule']['kwargs'] must be a mapping")

    schedule = _try_import_baseline_schedule_from_coins(kind, coins_kwargs=coins_kwargs)

    emb_block = cfg.get("embedding", {}) or {}
    if not isinstance(emb_block, Mapping):
        raise ValueError("cfg['embedding'] must be a mapping if provided")

    emb_cfg = NodeEmbeddingConfig(
        mode=str(emb_block.get("mode", "identity")),  # type: ignore[arg-type]
        out_dim=emb_block.get("out_dim", None),
        seed=int(emb_block.get("seed", 0)),
        normalize=bool(emb_block.get("normalize", False)),
        dtype=getattr(torch, str(emb_block.get("dtype", "float32")), torch.float32),
    )

    pol_cfg = BaselinePolicyConfig(
        name=name,
        strict_theta_shape=bool(cfg.get("strict_theta_shape", True)),
        theta_scale=float(cfg.get("theta_scale", 1.0)),
        theta_noise_std=float(cfg.get("theta_noise_std", 0.0)),
        theta_clip=cfg.get("theta_clip", None),
        theta_dtype=getattr(torch, str(cfg.get("theta_dtype", "float32")), torch.float32),
        embedding=emb_cfg,
    )

    return build_baseline_policy(schedule=schedule, cfg=pol_cfg)


def build_baseline_policy(
    kind: str,
    *,
    name: str | None = None,
    coins_kwargs: Mapping[str, Any] | None = None,
    policy_kwargs: Mapping[str, Any] | None = None,
) -> BaselinePolicy:
    """
    Single-entry convenience factory (good for scripts).

    Example:
      model = build_baseline_policy(
          "grover",
          coins_kwargs={"phase": 0.0},
          policy_kwargs={"theta_clip": 3.14159},
      )

    This expects acpl.baselines.coins to be present and able to build 'kind'.
    """
    schedule = _try_import_baseline_schedule_from_coins(kind, coins_kwargs=coins_kwargs)

    pk: MutableMapping[str, Any] = dict(policy_kwargs or {})
    emb_cfg = pk.pop("embedding", None)

    # Build BaselinePolicyConfig from kwargs with safe defaults
    cfg = BaselinePolicyConfig(
        name=str(name or kind),
        strict_theta_shape=bool(pk.pop("strict_theta_shape", True)),
        theta_scale=float(pk.pop("theta_scale", 1.0)),
        theta_noise_std=float(pk.pop("theta_noise_std", 0.0)),
        theta_clip=pk.pop("theta_clip", None),
        theta_dtype=pk.pop("theta_dtype", torch.float32),
        embedding=emb_cfg if isinstance(emb_cfg, NodeEmbeddingConfig) else BaselinePolicyConfig().embedding,
    )

    if pk:
        # Keep this strict: catching typos early is “research quality”.
        raise TypeError(f"Unknown policy_kwargs keys: {sorted(pk.keys())}")

    return BaselinePolicy(schedule=schedule, cfg=cfg)

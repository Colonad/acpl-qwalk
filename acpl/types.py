# acpl/types.py
# =============================================================================
# Phase B8 — Utils & typing hardening
# TypedDicts / Protocols for configs, batches, simulator, policies, and loops
# =============================================================================
# This module centralizes structural types used across ACPL. It favors:
#   • precision (TypedDict/Protocol + Literals),
#   • ergonomics (NotRequired/Required for partial configs),
#   • portability (no third-party typing deps beyond typing_extensions),
#   • robustness (runtime_checkable interfaces for duck-typing in tests).
#
# It does NOT import heavy subpackages (no torch_geometric, no networkx).
# Only torch.Tensor is referenced for tensor value types.
#
# References to project docs:
# - Theory: GNN-based ACPL for DTQWs (coin parameterization, arc/port basis)
# - Checklist B1..B8: simulator breadth, policies, PPO, eval, typing hardening
# =============================================================================

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    NotRequired,
    Protocol,
    Required,
    Union,
    runtime_checkable,
)

from typing_extensions import TypedDict

try:
    # Torch is a hard dependency of the project; the try/except keeps this file importable
    # for static analysis tools that stub torch, and for docs builds.
    from torch import Tensor
except Exception:  # pragma: no cover
    Tensor = Any  # type: ignore[misc,assignment]


# =============================================================================
# Basic aliases & literals (IDs, modes, layout)
# =============================================================================

VertexId = int
ArcId = int
EdgeId = int
TimeIndex = int
Seed = int

DTypeStr = Literal[
    "float32",
    "float64",
    "complex64",
    "complex128",
    "int32",
    "int64",
    "bool",
]

DeviceStr = Literal["cpu", "cuda", "mps"]

# Graph families: checklist B2
GraphFamily = Literal[
    "line",
    "cycle",
    "grid",
    "hypercube",
    "d_regular",
    "erdos_renyi",
    "watts_strogatz",
]

# GNN backbones: checklist B3 / theory §17–21
GNNKind = Literal["gcn", "graphsage", "gat", "gin", "pna"]

# Controller & coin heads (Phase A/B)
ControllerKind = Literal["gru", "transformer"]
CoinParamKind = Literal["su2", "exp", "cayley"]  # theory §9.1

# Tasks: checklist B4
TaskKind = Literal["transfer", "search", "mixing", "robust"]

# Training regimes
TrainLoopKind = Literal["backprop", "ppo"]

# Sparse edge layout (torch_geometric-like)
EdgeIndexLayout = Literal["coo"]


# =============================================================================
# Small structural types for arc/port bookkeeping and mixed-degree handling
# =============================================================================


class ArcSlice(TypedDict):
    """Contiguous slice in the arc/port basis for a single vertex.

    Example: if dv[v] = 4 and cumulative arc offset is 10, then
      start=10, stop=14  (Python half-open)
    """

    start: Required[int]
    stop: Required[int]  # exclusive


class PortMapEntry(TypedDict):
    """Flip-flop pairing across arcs (u, a') = port_map(v, a).

    'dst_vertex' is the neighbor u; 'dst_port' is a' (local index at u).
    """

    dst_vertex: Required[VertexId]
    dst_port: Required[int]


PortMap = list[list[PortMapEntry]]  # shape: [|V|][dv[v]]


# =============================================================================
# Data / Episode batch schemas (single-graph per episode; batched via lists)
# =============================================================================


class GraphTensors(TypedDict):
    """Minimal graph structure exposed to encoders and simulator."""

    # COO directed edge list, 2 × |E→|
    edge_index: Required[Tensor]
    # Per-node degree dv (int64), shape [N]
    degrees: Required[Tensor]
    # Number of nodes = N
    num_nodes: Required[int]
    # Optional: per-edge features (e.g., orientation, weight, phase), shape [|E→|, de]
    edge_attr: NotRequired[Tensor]
    # Optional: batch vector for mini-batching many small graphs (GNN), shape [N]
    node_batch: NotRequired[Tensor]
    # Optional: index of graph for each edge (if using fused batches), shape [|E→|]
    edge_batch: NotRequired[Tensor]


class NodeFeatures(TypedDict):
    """Node feature matrix and optional positional encodings."""

    # X: node features (degree, LapPE, coords/bitstrings, etc.), shape [N, F]
    x: Required[Tensor]
    # Optional positional encodings passed alongside z at control time, shape [N, Kpe]
    pos_enc: NotRequired[Tensor]
    # Optional role/structural features (boundary flags, distances), shape [N, F_role]
    role: NotRequired[Tensor]


class ArcLayout(TypedDict):
    """Bookkeeping for the arc/port basis used by the shift."""

    # Per-node arc slices into the flattened arc axis, len = N
    arc_slices: Required[list[ArcSlice]]
    # Total number of arcs D = sum_v dv (== 2|E| on simple undirected graphs)
    num_arcs: Required[int]
    # Flip-flop port pairing (for building S / applying permutation)
    port_map: Required[PortMap]


class EpisodeBatch(TypedDict):
    """Single-episode batch (one graph), used across policy & simulator."""

    graph: Required[GraphTensors]
    features: Required[NodeFeatures]
    layout: Required[ArcLayout]
    # Optional per-episode payload (for tasks, e.g. targets/marks, uniform vector, disorder law params)
    aux: NotRequired[Mapping[str, Tensor]]
    # Device/dtype hints for construction
    device: NotRequired[DeviceStr]
    float_dtype: NotRequired[DTypeStr]
    complex_dtype: NotRequired[DTypeStr]


# =============================================================================
# Policy I/O tensors (θ, coins, states, marginals)
# =============================================================================


class PolicyStepOutput(TypedDict):
    """Outputs at a single time step t from the policy stack."""

    # θ_{v,t}: coin parameters per node; padded or ragged by dv is handled by coin lift
    theta_vt: Required[Tensor]  # shape [N, p(dv)] or structured per-degree dict
    # h_{t+1}: next controller hidden state (opaque to callers; policy-specific)
    h_next: Required[Any]


class PackedCoins(TypedDict):
    """Physical coins Ct assembled from per-node coin blocks.

    Two representations are supported:
      • 'blocks': list of dv×dv unitary blocks in node order (preferred for block-diag apply)
      • 'dense': optional full dense block-diagonal matrix (for tests/diagnostics)
    """

    blocks: Required[list[Tensor]]  # list length N, each Tensor [dv, dv], complex
    # Optional: full Ct (D×D) complex (expensive, not needed in the simulator hot path)
    dense: NotRequired[Tensor]


class WalkState(TypedDict):
    """Quantum walk state containers."""

    # ψ_t in the arc basis, shape [D] complex
    psi: Required[Tensor]
    # Optional: reduced position probability P_t(v), shape [N], computed via partial trace
    pos_probs: NotRequired[Tensor]


# =============================================================================
# Config schemas (hydra/yaml → strongly typed dicts)
# =============================================================================


class DataConfig(TypedDict, total=False):
    family: GraphFamily
    # === family params ===
    # line/cycle
    N: int
    # grid
    L: int  # grid side (L×L)
    # hypercube
    n: int  # dimension, |V| = 2^n
    # random graphs
    d: int  # d-regular degree
    p: float  # ER probability
    k: int  # WS k-nearest neighbors on ring
    beta: float  # WS rewiring probability
    # noise/disorder flags (fed into objectives/robustness)
    disorder_seed: Seed
    # reproducibility
    seed: Seed


class FeaturesConfig(TypedDict, total=False):
    use_degree: bool
    use_coords: bool
    use_lappe: bool
    lappe_k: int
    use_bitstring: bool
    use_role: bool
    # normalization
    standardize: bool
    # caching strategy keys (optional)
    cache_key: str


class GNNConfig(TypedDict, total=False):
    kind: GNNKind
    hidden_dim: int
    num_layers: int
    dropout: float
    # GAT / PNA specifics
    heads: int  # GAT heads
    pna_aggregators: list[str]  # e.g., ["sum","mean","max","std"]
    pna_degree_scalers: list[str]  # e.g., ["id","amp","att"]
    norm: Literal["layer", "graph", "batch", "none"]
    use_residual: bool
    use_jk: bool


class ControllerConfig(TypedDict, total=False):
    kind: ControllerKind
    hidden_dim: int
    num_layers: int
    dropout: float


class CoinConfig(TypedDict, total=False):
    param: CoinParamKind
    # Parameterization domain:
    #   su2     → θ ∈ R^{N×3} (ZYZ Euler)
    #   exp     → coefficients for skew-Hermitian basis, per node dv → p = d_v^2
    #   cayley  → same coefficients; mapped by Cayley retraction
    basis: Literal["full_u", "su"]  # dv>2: U(d) vs. SU(d) (trace-free)
    # Smoothness / magnitude regularizers (weights)
    lambda_time_smooth: float
    lambda_magnitude: float


class TaskTransferConfig(TypedDict, total=False):
    kind: Literal["transfer"]
    source: VertexId
    target: VertexId


class TaskSearchConfig(TypedDict, total=False):
    kind: Literal["search"]
    # Boolean mask over nodes (1=marked), shape [N]; when constructing configs this is often resolved per episode
    mark_count: int


class TaskMixingConfig(TypedDict, total=False):
    kind: Literal["mixing"]
    # Evaluate TV distance at horizon T or provide curve sampling
    track_curve: bool


class RobustNoiseConfig(TypedDict, total=False):
    # Static edge phases ϕ_uv ~ law with seed; realized per episode
    edge_phase_std: float
    # Coin/position dephasing rates (per step), if modeled
    coin_dephase_p: float
    pos_dephase_p: float
    # CVaR level ∈ (0, 1]
    cvar_alpha: float


class TaskRobustConfig(TypedDict, total=False):
    kind: Literal["robust"]
    target: VertexId
    noise: RobustNoiseConfig


TaskConfig = Union[
    TaskTransferConfig,
    TaskSearchConfig,
    TaskMixingConfig,
    TaskRobustConfig,
]


class OptimConfig(TypedDict, total=False):
    # Optimizer
    name: Literal["adam", "adamw", "sgd"]
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    momentum: float
    # Gradient clipping
    clip_grad_norm: float


class TrainConfig(TypedDict, total=False):
    loop: TrainLoopKind
    T: int  # horizon
    epochs: int
    batch_size: int  # number of episodes per optimizer step (for PPO: rollout minibatch size)
    device: DeviceStr
    float_dtype: DTypeStr
    complex_dtype: DTypeStr
    seed: Seed
    # Logging/checkpoint
    log_every: int
    ckpt_every: int
    out_dir: str


class ModelConfig(TypedDict, total=False):
    gnn: GNNConfig
    controller: ControllerConfig
    coin: CoinConfig


class ExperimentConfig(TypedDict, total=False):
    """Top-level config for scripts/train.py and scripts/eval.py."""

    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    task: TaskConfig
    optim: OptimConfig
    train: TrainConfig


# =============================================================================
# Simulator & policy Protocols
# =============================================================================


@runtime_checkable
class ShiftOperator(Protocol):
    """Flip-flop shift S acting as an index permutation in the arc basis."""

    def apply(self, psi: Tensor) -> Tensor:
        """ψ ← S ψ  (psi: [D] complex)"""
        ...

    def as_permutation(self) -> Tensor:
        """Return an integer permutation vector p of length D such that
        psi_out = psi_in[p], for debugging/tests.
        """
        ...


@runtime_checkable
class CoinLift(Protocol):
    """Map coin parameters θ_{v,t} to physical per-node coin blocks and assemble Ct."""

    def from_parameters(self, theta_vt: Tensor, degrees: Tensor) -> PackedCoins:
        """θ_{v,t} ([N, p(dv)]) × dv([N]) → list of U(dv) blocks (and optional dense Ct)."""
        ...

    @property
    def param_kind(self) -> CoinParamKind:  # su2 | exp | cayley
        ...


@runtime_checkable
class Stepper(Protocol):
    """One unitary DTQW step: apply coin then shift."""

    def step(self, psi: Tensor, coins: PackedCoins) -> Tensor:
        """ψ_{t+1} = S · Ct · ψ_t  (all in arc basis)."""
        ...


@runtime_checkable
class PartialTrace(Protocol):
    """Compute reduced position probability P_t(v) from ψ_t in arc basis."""

    def position_prob(self, psi: Tensor, arc_slices: Sequence[ArcSlice]) -> Tensor:
        """Return P(v) = sum_a |ψ(v,a)|^2, shape [N]."""
        ...


@runtime_checkable
class GraphEncoder(Protocol):
    """Permutation-equivariant node encoder Φ_GNN(G, X) → Z."""

    def __call__(self, graph: GraphTensors, features: NodeFeatures) -> Tensor:
        """Return node embeddings z, shape [N, H]."""
        ...


@runtime_checkable
class TemporalController(Protocol):
    """Time-aware controller that consumes [z; τ_t; PE] and maintains hidden state."""

    def init_state(self, N: int, device: DeviceStr) -> Any:  # opaque, policy-specific
        ...

    def step(
        self,
        z: Tensor,  # [N, H]
        tau_t: float,  # normalized time t/T
        pos_enc: Tensor | None,  # [N, Kpe] or None
        h_prev: Any,
    ) -> PolicyStepOutput: ...


@runtime_checkable
class CoinHead(Protocol):
    """Linear/readout head that converts controller state to θ_{v,t}."""

    def __call__(self, h_t: Any, degrees: Tensor) -> Tensor:
        """Return θ_{v,t} with per-node parameterization; may depend on dv."""
        ...


@runtime_checkable
class Policy(Protocol):
    """End-to-end π: (G, X, t, h_prev) → θ_{v,t}, h_next."""

    def forward_step(
        self,
        batch: EpisodeBatch,
        t: TimeIndex,
        h_prev: Any,
    ) -> PolicyStepOutput: ...

    def init_state(self, batch: EpisodeBatch) -> Any: ...


@runtime_checkable
class Objective(Protocol):
    """Task objective: computes scalar loss (to minimize) and diagnostics."""

    def loss(
        self,
        psi_T: Tensor,  # [D] complex
        pos_probs_T: Tensor,  # [N] float
        batch: EpisodeBatch,
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        """Return (loss, metrics)."""
        ...


# =============================================================================
# Training loop integrations (Backprop and PPO)
# =============================================================================


class BackpropRolloutResult(TypedDict):
    """Output from a differentiable rollout (BPTT)."""

    loss: Required[Tensor]
    metrics: Required[Mapping[str, Tensor]]
    psi_T: NotRequired[Tensor]
    pos_probs_T: NotRequired[Tensor]


class Trajectory(TypedDict):
    """PPO-style on-policy trajectory buffers (flattened over time and nodes)."""

    # Policy
    log_probs: Required[Tensor]  # [T, N, A] or [T, N] if θ treated as continuous action with dist
    values: Required[Tensor]  # [T, N]
    actions: Required[Tensor]  # θ samples or compressed action codes
    # Env / rewards
    rewards: Required[Tensor]  # [T] or [T, N] depending on shaping
    masks: Required[Tensor]  # [T] (episode continuation)
    # Optional info
    extras: NotRequired[Mapping[str, Tensor]]


@runtime_checkable
class PPOUpdater(Protocol):
    """PPO surrogate update on collected trajectories."""

    def update(self, traj: Trajectory) -> Mapping[str, Tensor]: ...


# =============================================================================
# Lightweight utilities & guards
# =============================================================================


class ShapeSpec(TypedDict, total=False):
    """Declarative spec used by utils/tensor_shapes.py (phase B8 sibling)."""

    names: list[str]  # dimension names, e.g., ["N","H"]
    exact: tuple[int, ...] | None
    dtypes: list[DTypeStr] | None
    is_complex: bool | None


@dataclass(frozen=True)
class PolicyStack:
    """Convenience container to group policy parts for wiring tests."""

    encoder: GraphEncoder
    controller: TemporalController
    head: CoinHead
    lift: CoinLift


# =============================================================================
# Public export surface
# =============================================================================

__all__ = [
    # base
    "VertexId",
    "ArcId",
    "EdgeId",
    "TimeIndex",
    "Seed",
    "DTypeStr",
    "DeviceStr",
    "GraphFamily",
    "GNNKind",
    "ControllerKind",
    "CoinParamKind",
    "TaskKind",
    "TrainLoopKind",
    "EdgeIndexLayout",
    # layouts
    "ArcSlice",
    "PortMapEntry",
    "PortMap",
    "ArcLayout",
    # batches
    "GraphTensors",
    "NodeFeatures",
    "EpisodeBatch",
    # policy I/O
    "PolicyStepOutput",
    "PackedCoins",
    "WalkState",
    # configs
    "DataConfig",
    "FeaturesConfig",
    "GNNConfig",
    "ControllerConfig",
    "CoinConfig",
    "TaskTransferConfig",
    "TaskSearchConfig",
    "TaskMixingConfig",
    "RobustNoiseConfig",
    "TaskRobustConfig",
    "TaskConfig",
    "OptimConfig",
    "TrainConfig",
    "ModelConfig",
    "ExperimentConfig",
    # protocols
    "ShiftOperator",
    "CoinLift",
    "Stepper",
    "PartialTrace",
    "GraphEncoder",
    "TemporalController",
    "CoinHead",
    "Policy",
    "Objective",
    "PPOUpdater",
    # training artifacts
    "BackpropRolloutResult",
    "Trajectory",
    # utils
    "ShapeSpec",
    "PolicyStack",
]

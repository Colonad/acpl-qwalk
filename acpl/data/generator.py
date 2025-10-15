# acpl/data/generator.py
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import json
from typing import Literal, TypeAlias

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # make torch optional for NumPy-only usage

from .features import node_features_line
from .graphs import line_graph

SplitName = Literal["train", "val", "test"]

# ---------------------------------------------------------------------------
# Episode structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeNP:
    """Episode payload in NumPy form."""

    edge_index: np.ndarray  # (2, E) int64 canonical undirected
    degrees: np.ndarray  # (N,) int64
    coords: np.ndarray  # (N, C) float32 (may be empty columns)
    features: np.ndarray  # (N, F) float32
    arc_slices: np.ndarray  # (N, 2) int64 : vertex-lumped [start,end)
    seed: int
    manifest_hexdigest: str


@dataclass(frozen=True)
class EpisodeTorch:
    """Episode payload in Torch form (if requested)."""

    edge_index: torch.Tensor  # (2, E) long
    degrees: torch.Tensor  # (N,) long
    coords: torch.Tensor  # (N, C) float
    features: torch.Tensor  # (N, F) float
    arc_slices: torch.Tensor  # (N, 2) long
    seed: int
    manifest_hexdigest: str


# ---------------------------------------------------------------------------
# Deterministic hashing helpers
# ---------------------------------------------------------------------------

JSONScalar: TypeAlias = str | int | float | bool | None
JSONVal: TypeAlias = JSONScalar | list["JSONVal"] | dict[str, "JSONVal"]


def _stable_hexdigest(obj: object, *, digest_size: int = 16) -> str:
    """
    Produce a stable blake2b hex digest for a (possibly nested) Python object.
    Returns a hex string; does not mutate input.
    """

    def to_builtin(x: object) -> JSONVal:
        # numpy scalars
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        # numpy arrays
        if isinstance(x, np.ndarray):
            return x.tolist()
        # dicts (sorted)
        if isinstance(x, dict):
            return {k: to_builtin(v) for k, v in sorted(x.items(), key=lambda kv: kv[0])}
        # sequences (keep order)
        if isinstance(x, (list, tuple)):
            return [to_builtin(v) for v in x]
        # primitives and everything else
        return x  # type: ignore[return-value]

    canonical = json.dumps(
        to_builtin(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return blake2b(canonical, digest_size=digest_size).hexdigest()


# ---------------------------------------------------------------------------
# Seed derivation (deterministic per split/index)
# ---------------------------------------------------------------------------


def _derive_episode_seed(manifest_hex: str, split: str, index: int) -> int:
    """
    Deterministically derive a 32-bit seed from (manifest hash, split, index).
    """
    h = blake2b(digest_size=8)
    h.update(manifest_hex.encode("utf-8"))
    h.update(b"|")
    h.update(split.encode("utf-8"))
    h.update(b"|")
    h.update(str(int(index)).encode("utf-8"))
    raw = int.from_bytes(h.digest(), byteorder="big", signed=False)
    return int(raw % (2**31 - 1))


# ---------------------------------------------------------------------------
# Minimal graph builder dispatcher (only what's needed by tests)
# ---------------------------------------------------------------------------


def _build_graph_from_config(
    configs: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dispatch to a graph builder based on configs["family"] and its params.
    Only 'line' is supported here (tests).
    Returns: edge_index, degrees, coords, arc_slices
    """
    fam = str(configs.get("family", "line")).lower()
    if fam == "line":
        n = int(configs.get("num_nodes", 64))
        return line_graph(n, seed=seed)
    raise ValueError(f"Unknown/unsupported graph family for tests: {fam!r}")


# ---------------------------------------------------------------------------
# Portable PortMap-like object (attribute access, not dict)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PortMapLite:
    tail: np.ndarray  # (A,) int64
    head: np.ndarray  # (A,) int64
    rev: np.ndarray  # (A,) int64
    node_ptr: np.ndarray  # (N+1,) int64 CSR-style offsets
    node_arcs: np.ndarray  # (A,) int64 indices into arcs
    num_nodes: int
    num_arcs: int


def _portable_portmap_from_edges(edge_index: np.ndarray, num_nodes: int) -> _PortMapLite:
    """
    Build a portable PortMap-like structure from an undirected canonical edge_index (2, E).

    Convention:
      - arcs are emitted (u->v) then (v->u) for each undirected (u,v) in order
      - node_arcs concatenates arcs by tail vertex in encounter order
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")

    n = int(num_nodes)
    e = int(edge_index.shape[1])
    a = 2 * e

    tail = np.empty(a, dtype=np.int64)
    head = np.empty(a, dtype=np.int64)
    rev = np.empty(a, dtype=np.int64)

    uu = edge_index[0].astype(np.int64, copy=False)
    vv = edge_index[1].astype(np.int64, copy=False)

    # arcs and flip-flop pairing
    for i in range(e):
        a0, a1 = 2 * i, 2 * i + 1
        u, v = int(uu[i]), int(vv[i])
        tail[a0], head[a0] = u, v  # u -> v
        tail[a1], head[a1] = v, u  # v -> u
        rev[a0], rev[a1] = a1, a0  # flip-flop

    # Vertex-lumped ordering: gather arcs by tail vertex in encounter order
    per_v: list[list[int]] = [[] for _ in range(n)]
    for i in range(e):
        a0, a1 = 2 * i, 2 * i + 1
        per_v[int(uu[i])].append(a0)
        per_v[int(vv[i])].append(a1)

    lengths = np.array([len(lst) for lst in per_v], dtype=np.int64)
    node_ptr = np.empty(n + 1, dtype=np.int64)
    node_ptr[0] = 0
    np.cumsum(lengths, out=node_ptr[1:])

    node_arcs = np.empty(a, dtype=np.int64)
    cur = 0
    for lst in per_v:
        ln = len(lst)
        if ln:
            node_arcs[cur : cur + ln] = np.asarray(lst, dtype=np.int64)
            cur += ln

    return _PortMapLite(
        tail=tail,
        head=head,
        rev=rev,
        node_ptr=node_ptr,
        node_arcs=node_arcs,
        num_nodes=n,
        num_arcs=a,
    )


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NormConfig:
    """Normalized, hashable view of the episode generator config that affects data."""

    family: str
    num_nodes: int
    task: str
    feature_kind: str
    task_params: tuple[tuple[str, int], ...]  # sorted items for determinism

    @staticmethod
    def from_kwargs(
        *,
        family: str,
        num_nodes: int,
        task: str,
        feature_kind: str,
        task_params: dict[str, object] | None,
    ) -> _NormConfig:
        tp = tuple(sorted((task_params or {}).items()))
        return _NormConfig(
            family=str(family).lower(),
            num_nodes=int(num_nodes),
            task=str(task).lower(),
            feature_kind=str(feature_kind).lower(),
            task_params=tuple((str(k), int(v)) for k, v in tp),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "num_nodes": self.num_nodes,
            "task": self.task,
            "feature_kind": self.feature_kind,
            "task_params": list(self.task_params),
        }


class EpisodeGenerator:
    """
    Minimal on-the-fly episode sampler used in tests.

    Expected constructor kwargs (used by tests):
      - family:        str   (e.g., "line")
      - num_nodes:     int
      - task:          str   (e.g., "transfer")
      - task_params:   dict  (e.g., {"source": 0, "target": 15})
      - feature_kind:  str   (e.g., "line_deg_coord")

    Determinism: get_episode(seed) uses a NumPy RNG seeded by blake2b(config)+seed.
    """

    def __init__(
        self,
        *,
        family: str,
        num_nodes: int,
        task: str,
        task_params: dict[str, object] | None = None,
        feature_kind: str = "line_deg_coord",
    ) -> None:
        self._cfg_norm = _NormConfig.from_kwargs(
            family=family,
            num_nodes=num_nodes,
            task=task,
            feature_kind=feature_kind,
            task_params=task_params or {},
        )
        self._cfg_hash = _stable_hexdigest(self._cfg_norm.as_dict())

        # quick validation for cases we actually implement in tests
        if self._cfg_norm.family != "line":
            raise ValueError("EpisodeGenerator (tests) currently supports family='line' only.")
        if self._cfg_norm.task != "transfer":
            raise ValueError("EpisodeGenerator (tests) currently supports task='transfer' only.")
        if self._cfg_norm.feature_kind not in {"line_deg_coord"}:
            raise ValueError(
                "EpisodeGenerator (tests) supports feature_kind='line_deg_coord' only."
            )

    # ------------------------------- public API -------------------------------

    def get_episode(self, seed: int) -> dict[str, object]:
        """
        Generate a single episode deterministically from (normalized config, seed).
        Returns a dict with numpy arrays and ints; keys include:
        - 'edge_index' (2, E) int64
        - 'degrees'    (N,)   int64
        - 'coords'     (N,1)  float32
        - 'features'   (N,F)  float32
        - 'X'          (N,F)  float32  (alias for 'features')
        - 'source'     int
        - 'target'     int
        - 'arc_slices' (N,2)  int64
        - 'pm'         PortMap-like object (with attributes)
        - 'psi0'       (A,)   complex64 initial arc state
        - 'rng_seed'   int
        - 'config_hex' str
        """
        eff_seed = self._mix_seed(seed)
        rng = np.random.Generator(np.random.PCG64(eff_seed))

        n = self._cfg_norm.num_nodes

        # --------- graph family: line ---------
        edge_index, degrees, coords, arc_slices = line_graph(n, seed=int(rng.integers(0, 2**31)))

        # --------- features ---------
        if self._cfg_norm.feature_kind == "line_deg_coord":
            feats = node_features_line(degrees, coords, dtype=np.float32)
        else:  # pragma: no cover
            feats = np.zeros((n, 0), dtype=np.float32)
        x_feat = feats.astype(np.float32, copy=False)

        # --------- task params (transfer) ---------
        tp = dict(self._cfg_norm.task_params)
        src = int(tp.get("source", 0))
        tgt = int(tp.get("target", n - 1))
        if not (0 <= src < n and 0 <= tgt < n):
            raise ValueError("source/target indices out of range for current graph.")

        # --------- PortMap (portable object) ---------
        pm = _portable_portmap_from_edges(edge_index, n)

        # --------- Initial state psi0 (uniform over arcs leaving source) ---------
        num_arcs = int(pm.num_arcs)
        psi0 = np.zeros(num_arcs, dtype=np.complex64)
        start = int(pm.node_ptr[src])
        end = int(pm.node_ptr[src + 1])
        if end > start:
            arc_ids = pm.node_arcs[start:end]  # absolute arc indices
            amp = 1.0 / np.sqrt(float(end - start))
            psi0[arc_ids] = amp + 0.0j

        return {
            "edge_index": edge_index.astype(np.int64, copy=False),
            "degrees": degrees.astype(np.int64, copy=False),
            "coords": coords.astype(np.float32, copy=False),
            "features": x_feat,  # keep original name
            "X": x_feat,  # explicit alias expected by tests
            "source": src,
            "target": tgt,
            "arc_slices": arc_slices.astype(np.int64, copy=False),
            "pm": pm,  # attribute-access PortMap
            "psi0": psi0,  # complex64
            "rng_seed": int(eff_seed),
            "config_hex": self._cfg_hash,
        }

    def sample(self, *, seed: int) -> dict[str, object]:
        """Compatibility alias expected by tests."""
        return self.get_episode(seed)

    # ------------------------------ internals ------------------------------

    def _mix_seed(self, user_seed: int) -> int:
        """
        Mix user seed with config hash to get a deterministic 64-bit seed.
        """
        h = blake2b(digest_size=8)
        h.update(self._cfg_hash.encode("utf-8"))
        h.update(int(user_seed).to_bytes(8, byteorder="little", signed=False))
        return int.from_bytes(h.digest(), byteorder="little", signed=False)

# acpl/data/generator.py
from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from hashlib import blake2b
import json
import math
import os
from .splits import route_triplet, default_phase_b2_rules
import threading
import time
from typing import Any, Literal, TypeAlias

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# Splits we support
SplitName = Literal["train", "val", "test"]

from .features import FeatureSpec, build_node_features



# ---- Graphs & features from our Phase B2 modules ------------------------------
from .graphs import (
    cycle_graph,
    d_regular_random_graph,
    erdos_renyi_graph,
    grid_graph,
    hypercube_graph,
    line_graph,
    random_geometric_graph,
    watts_strogatz_graph,
    watts_strogatz_grid_graph,
    watts_strogatz_grid_graph_degree_preserving,
)

# ==============================================================================
# Episode payloads
# ==============================================================================


@dataclass(frozen=True)
class EpisodeNP:
    """
    Non-minimal episode payload (NumPy). All integer arrays are int64; reals are float32.
    A = number of oriented arcs (= edge_index.shape[1]).
    """

    # Graph (oriented)
    edge_index: np.ndarray  # (2, A) int64, oriented arcs (src,dst)
    degrees: np.ndarray  # (N,)   int64 (undirected degree)
    coords: np.ndarray  # (N, C) float32 (can be empty with C=0)
    arc_slices: np.ndarray  # (N+1,) int64 CSR pointer over arcs by source node

    # Node features
    X: np.ndarray  # (N, F) float32
    features: np.ndarray  # (N, F) float32 (alias of X)

    # Portable PortMap built from oriented arcs
    pm_tail: np.ndarray  # (A,)   int64  == edge_index[0]
    pm_head: np.ndarray  # (A,)   int64  == edge_index[1]
    pm_rev: np.ndarray  # (A,)   int64  index of the reverse arc
    pm_node_ptr: np.ndarray  # (N+1,) int64  == arc_slices
    pm_node_arcs: np.ndarray  # (A,)   int64  contiguous [ptr[u]:ptr[u+1]) are arcs with src=u
    num_nodes: int
    num_arcs: int

    # Initial state and (optional) noise recipe
    psi0: np.ndarray  # (A,) complex64
    noise: dict[
        str, Any
    ]  # sampled noise parameters (edge phases, dephasing, …) – no application here

    # Task & provenance
    task: dict[str, Any]
    rng_seed: int
    manifest_hexdigest: str


@dataclass(frozen=True)
class EpisodeTorch:
    """Torch view of an episode with device/dtype control."""

    edge_index: torch.Tensor
    degrees: torch.Tensor
    coords: torch.Tensor
    arc_slices: torch.Tensor
    X: torch.Tensor
    features: torch.Tensor
    pm_tail: torch.Tensor
    pm_head: torch.Tensor
    pm_rev: torch.Tensor
    pm_node_ptr: torch.Tensor
    pm_node_arcs: torch.Tensor
    psi0: torch.Tensor
    noise: dict[str, Any]
    task: dict[str, Any]
    rng_seed: int
    manifest_hexdigest: str
    device: torch.device | None = None


# ==============================================================================
# Stable hashing & seeds
# ==============================================================================

JSONScalar: TypeAlias = str | int | float | bool | None
JSONVal: TypeAlias = JSONScalar | list["JSONVal"] | dict[str, "JSONVal"]


def _stable_hexdigest(obj: object, *, digest_size: int = 16) -> str:
    """Stable BLAKE2b digest of arbitrary JSON-like content (NumPy supported)."""

    def to_builtin(x: object) -> JSONVal:
        if isinstance(x, (np.integer,)):  # numpy int
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: to_builtin(v) for k, v in sorted(x.items(), key=lambda kv: kv[0])}
        if isinstance(x, (list, tuple)):
            return [to_builtin(v) for v in x]
        return x  # type: ignore[return-value]

    canonical = json.dumps(
        to_builtin(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return blake2b(canonical, digest_size=digest_size).hexdigest()


def _derive_episode_seed(manifest_hex: str, split: str, index: int) -> int:
    """Deterministically derive a 31-bit seed from (manifest_hex, split, index)."""
    h = blake2b(digest_size=8)
    h.update(manifest_hex.encode("utf-8"))
    h.update(b"|")
    h.update(split.encode("utf-8"))
    h.update(b"|")
    h.update(str(int(index)).encode("utf-8"))
    raw = int.from_bytes(h.digest(), "big", signed=False)
    return int(raw % (2**31 - 1))


# ==============================================================================
# Portable PortMap from oriented arcs (no project conventions)
# ==============================================================================


@dataclass(frozen=True)
class _PortMapLite:
    src: np.ndarray
    dst: np.ndarray
    rev: np.ndarray
    node_ptr: np.ndarray
    node_arcs: np.ndarray
    num_nodes: int
    num_arcs: int


def _portable_portmap_from_oriented(
    edge_index_oriented: np.ndarray, node_ptr: np.ndarray, num_nodes: int
) -> _PortMapLite:
    """
    Build a portable PortMap from oriented arcs.

    Inputs
    ------
    edge_index_oriented : (2,A) int64  oriented arcs (src,dst) sorted by src for CSR.
    node_ptr            : (N+1,) int64 per-node CSR pointer over arcs grouped by src.
    """
    if edge_index_oriented.ndim != 2 or edge_index_oriented.shape[0] != 2:
        raise ValueError("edge_index must be (2, A) oriented arcs")
    src = edge_index_oriented[0].astype(np.int64, copy=False)
    dst = edge_index_oriented[1].astype(np.int64, copy=False)
    A = int(src.shape[0])
    N = int(num_nodes)

    # Reverse-arc map: find for each (u,v) the index of (v,u).
    # For robust performance, hash keys as u*N + v.
    key = (src.astype(np.int64) * N + dst.astype(np.int64)).tolist()
    rev = np.full(A, -1, dtype=np.int64)
    pos: dict[int, int] = {}
    for i, k in enumerate(key):
        pos[k] = i
    for i, (u, v) in enumerate(zip(src, dst, strict=False)):
        j = pos.get(int(v) * N + int(u), -1)
        if j >= 0:
            rev[i] = j
    if (rev < 0).any():
        # Some arcs have no explicit reverse (e.g. directed input). Still accept
        # but pair them with themselves to keep a valid index.
        missing = np.where(rev < 0)[0]
        rev[missing] = missing

    # node_arcs is just range(A) – CSR slices are already provided in node_ptr
    node_arcs = np.arange(A, dtype=np.int64)
    return _PortMapLite(src, dst, rev, node_ptr.astype(np.int64, copy=False), node_arcs, N, A)


# ==============================================================================
# Graph dispatcher (NON-minimal; many families)
# ==============================================================================


def _torch_to_numpy(*tensors: torch.Tensor) -> tuple[np.ndarray, ...]:  # safe helper
    return tuple(t.detach().cpu().numpy() for t in tensors)


def _build_graph_from_config(
    cfg: Mapping[str, Any], *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Use Phase B2 graph builders. All of our builders return **torch** tensors, so we
    convert to NumPy here to keep the generator torch-optional.
    """
    fam = str(cfg.get("family", "line")).lower()
    rng = np.random.Generator(np.random.PCG64(seed))

    # LINE / CYCLE
    if fam == "line":
        N = int(cfg.get("num_nodes", cfg.get("N", 64)))
        ei, deg, coords, ptr = line_graph(N, seed=int(rng.integers(0, 2**31)))
        return _torch_to_numpy(ei, deg, coords, ptr)

    if fam == "cycle":
        N = int(cfg.get("num_nodes", cfg.get("N", 64)))
        ei, deg, coords, ptr = cycle_graph(N, seed=int(rng.integers(0, 2**31)))
        return _torch_to_numpy(ei, deg, coords, ptr)

    # GRID (LxL – our graphs.py uses grid_graph(L, Ly=None))
    if fam == "grid":
        nn = cfg.get("num_nodes", cfg.get("N", 64))
        L = int(cfg.get("L", int(math.sqrt(nn)) or 8))
        Ly = int(cfg.get("Ly", L))
        ei, deg, coords, ptr = grid_graph(L, Ly, seed=int(rng.integers(0, 2**31)))
        return _torch_to_numpy(ei, deg, coords, ptr)

    # HYPERCUBE
    if fam == "hypercube":
        n = int(cfg.get("n", 6))
        ei, deg, coords, ptr = hypercube_graph(n, seed=int(rng.integers(0, 2**31)))
        return _torch_to_numpy(ei, deg, coords, ptr)

    # ERDOS-RENYI
    if fam in {"er", "erdos_renyi"}:
        N = int(cfg.get("num_nodes", cfg.get("N", 128)))
        p = float(cfg.get("p", 3.0 / max(N - 1, 1)))
        ei, deg, coords, ptr = erdos_renyi_graph(N, p, seed=int(rng.integers(0, 2**31)))
        return _torch_to_numpy(ei, deg, coords, ptr)

    # D-REGULAR
    if fam in {"d_regular", "d-regular", "regular"}:
        N = int(cfg.get("num_nodes", cfg.get("N", 128)))
        d = int(cfg.get("d", 3))
        ei, deg, coords, ptr = d_regular_random_graph(N, d, seed=int(rng.integers(0, 2**31)))
        return _torch_to_numpy(ei, deg, coords, ptr)

    # WATTS–STROGATZ (ring)
    if fam in {"ws", "watts", "watts_strogatz"}:
        N = int(cfg.get("num_nodes", cfg.get("N", 128)))
        k = int(cfg.get("k", 4))
        beta = float(cfg.get("beta", 0.2))
        ei, deg, coords, ptr = watts_strogatz_graph(N, k, beta, seed=int(rng.integers(0, 2**31)))
        return _torch_to_numpy(ei, deg, coords, ptr)

    # WS on GRID (degree-preserving swapping supported)
    if fam in {"ws_grid", "watts_strogatz_grid"}:
        L = int(cfg.get("L", 8))
        Ly = int(cfg.get("Ly", L))
        kx = int(cfg.get("kx", 1))
        ky = int(cfg.get("ky", 1))
        beta = float(cfg.get("beta", 0.1))
        degree_preserving = bool(cfg.get("degree_preserving", False))
        if degree_preserving:
            ei, deg, coords, ptr = watts_strogatz_grid_graph_degree_preserving(
                L, Ly, kx=kx, ky=ky, beta=beta, seed=int(rng.integers(0, 2**31))
            )
        else:
            ei, deg, coords, ptr = watts_strogatz_grid_graph(
                L, Ly, kx=kx, ky=ky, beta=beta, seed=int(rng.integers(0, 2**31))
            )
        return _torch_to_numpy(ei, deg, coords, ptr)

    # RANDOM GEOMETRIC
    if fam in {"rgg", "random_geometric"}:
        N = int(cfg.get("num_nodes", cfg.get("N", 128)))
        radius = float(cfg.get("radius", 0.2))
        dim = int(cfg.get("dim", 2))
        torus = bool(cfg.get("torus", False))
        ei, deg, coords, ptr = random_geometric_graph(
            N, radius, dim=dim, seed=int(rng.integers(0, 2**31)), torus=torus
        )
        return _torch_to_numpy(ei, deg, coords, ptr)

    raise ValueError(f"Unknown graph family: {fam!r}")


# ==============================================================================
# Feature spec derivation from manifest
# ==============================================================================


def _spec_from_manifest(manifest: Mapping[str, Any]) -> FeatureSpec:
    """
    Build a FeatureSpec from a (possibly sparse) 'features' block of the manifest.
    Missing fields fall back to robust defaults we used across B2.
    """
    f = dict(manifest.get("features", {}))
    return FeatureSpec(
        use_degree=bool(f.get("use_degree", True)),
        degree_norm=str(f.get("degree_norm", "inv_sqrt")),
        degree_onehot_K=int(f.get("degree_onehot_K", 0)),
        use_coords=bool(f.get("use_coords", True)),
        use_sinusoidal_coords=bool(f.get("use_sinusoidal_coords", False)),
        sinusoidal_dims=int(f.get("sinusoidal_dims", 0)),
        sinusoidal_base=float(f.get("sinusoidal_base", 10000.0)),
        use_lap_pe=bool(f.get("use_lap_pe", False)),
        lap_pe_k=int(f.get("lap_pe_k", 0)),
        lap_pe_norm=str(f.get("lap_pe_norm", "sym")),
        # Deterministic by default (matches FeatureSpec + features.py docstring)
        lap_pe_random_sign=bool(f.get("lap_pe_random_sign", False)),
        use_rwse=bool(f.get("use_rwse", False)),
        rwse_K=int(f.get("rwse_K", 0)),
        build_arcs=False,  # arc features not needed by policy here
        eps=float(f.get("eps", 1e-12)),
        seed=int(f.get("seed", 0)) if "seed" in f else None,
    )


# ==============================================================================
# Size-aware LRU cache (NumPy episodes)
# ==============================================================================


class _LRUNode:
    __slots__ = ("key", "value", "ts", "prev", "next", "nbytes")

    def __init__(self, key: Any, value: EpisodeNP, nbytes: int) -> None:
        self.key = key
        self.value = value
        self.ts = time.time()
        self.prev: _LRUNode | None = None
        self.next: _LRUNode | None = None
        self.nbytes = int(max(0, nbytes))


class EpisodeLRUCache:
    def __init__(
        self, *, capacity_bytes: int = 512 * 1024 * 1024, ttl_seconds: float | None = None
    ) -> None:
        self._cap = int(capacity_bytes)
        self._ttl = ttl_seconds
        self._map: MutableMapping[Any, _LRUNode] = {}
        self._head: _LRUNode | None = None
        self._tail: _LRUNode | None = None
        self._bytes = 0
        self._lock = threading.RLock()

    def _remove(self, node: _LRUNode) -> None:
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if self._head is node:
            self._head = node.next
        if self._tail is node:
            self._tail = node.prev
        node.prev = node.next = None
        self._bytes -= node.nbytes

    def _push_front(self, node: _LRUNode) -> None:
        node.prev = None
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node
        self._bytes += node.nbytes

    def _evict_until_fit(self) -> None:
        while self._bytes > self._cap and self._tail is not None:
            stale = self._tail
            self._remove(stale)
            self._map.pop(stale.key, None)

    def _expired(self, node: _LRUNode) -> bool:
        if self._ttl is None:
            return False
        return (time.time() - node.ts) > self._ttl

    def clear(self) -> None:
        with self._lock:
            self._map.clear()
            self._head = self._tail = None
            self._bytes = 0

    def get(self, key: Any) -> EpisodeNP | None:
        with self._lock:
            node = self._map.get(key)
            if not node:
                return None
            if self._expired(node):
                self._remove(node)
                self._map.pop(key, None)
                return None
            self._remove(node)
            self._push_front(node)
            node.ts = time.time()
            return node.value

    def put(self, key: Any, value: EpisodeNP) -> None:
        with self._lock:
            if key in self._map:
                self._remove(self._map[key])
                self._map.pop(key, None)
            node = _LRUNode(key, value, _episode_nbytes(value))
            self._push_front(node)
            self._map[key] = node
            self._evict_until_fit()


def _episode_nbytes(ep: EpisodeNP) -> int:
    total = 0
    for fld in (
        "edge_index",
        "degrees",
        "coords",
        "arc_slices",
        "X",
        "features",
        "pm_tail",
        "pm_head",
        "pm_rev",
        "pm_node_ptr",
        "pm_node_arcs",
        "psi0",
    ):
        arr = getattr(ep, fld, None)
        if isinstance(arr, np.ndarray):
            total += int(arr.nbytes)
    return total + 1024  # small overhead for dicts


# ==============================================================================
# Episode generator (manifest + splits → deterministic episodes)
# ==============================================================================


class EpisodeGenerator:
    """
    Robust on-the-fly episode sampler.

    Manifest keys (subset; extras are ignored by hashing):
      - family:        'line'|'cycle'|'grid'|'hypercube'|'er'|'d_regular'|'ws'|'ws_grid'|'rgg'
      - num_nodes/N/L/n/... graph params (can be scalar, per-split dict, or list to sample from)
      - graph_params:  additional family-specific params (k,beta,d,p, etc.)
      - features:      FeatureSpec overrides (see _spec_from_manifest)
      - task:          'transfer'|'search'|'mixing'|'robust'
      - task_params:   {source,target,marks,...}
      - T:             horizon (passed through into task)
      - noise:         {edge_phases: {"sigma": ..}, dephasing: {"gamma": ..}, ...}
    """

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        split: SplitName = "train",
        cache: EpisodeLRUCache | None = None,
        cache_capacity_bytes: int = 512 * 1024 * 1024,
        cache_ttl_seconds: float | None = None,
    ) -> None:
        self._split: SplitName = split
        self._manifest_raw = dict(manifest)
        self._manifest_hex = _stable_hexdigest(self._data_affecting_view(self._manifest_raw))
        self._manifest_hexdigest = self._manifest_hex  # alias to avoid AttributeError
        # ---- Phase B2 canonical router config (triplet router) ----
        router_cfg = self._manifest_raw.get("router", self._manifest_raw.get("split_router", {}))
        if not isinstance(router_cfg, Mapping):
            router_cfg = {}

        self._router_master_seed = int(
            router_cfg.get("master_seed", self._manifest_raw.get("router_master_seed", 0)) or 0
        )

        rules = router_cfg.get("rules", None)
        self._router_rules: Mapping[str, Any] = rules if isinstance(rules, Mapping) else default_phase_b2_rules()
        self._cache = cache or EpisodeLRUCache(
            capacity_bytes=cache_capacity_bytes, ttl_seconds=cache_ttl_seconds
        )
        # Cache for CI eval index lists (keyed by (split, count, base_offset)).
        self._eval_index_cache: dict[tuple[SplitName, int, int], list[tuple[int, int]]] = {}
 
        # Split-local indexing support:
        # tests expect episode(i) to mean "the i-th episode of this split",
        # not "global episode index i".
        self._local_index_lock = threading.RLock()
        self._local_to_global: dict[SplitName, list[int]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self._local_scan_cursor: dict[SplitName, int] = {
            "train": 0,
            "val": 0,
            "test": 0,
        }



    # ---------------- Public API ----------------

    @property
    def manifest_hexdigest(self) -> str:
        return self._manifest_hex

    @property
    def split(self) -> SplitName:
        return self._split

    def _global_index_from_local(self, split: SplitName, local_index: int) -> int:
        """
        Deterministically map a split-local index to a global episode index that
        routes to `split`, by scanning global indices in increasing order.

        This is required because routing is defined on global indices, but our public
        API (and tests) treat episode(i) as split-local.
        """
        li = int(local_index)
        if li < 0:
            raise ValueError("local_index must be >= 0")

        with self._local_index_lock:
            buf = self._local_to_global[split]
            g = int(self._local_scan_cursor[split])

            # Safety bound: if someone sets router ratios to 0 for a split, we’d loop forever.
            max_scan = max(10_000, (li + 1) * 1_000)
            scanned = 0

            while len(buf) <= li:
                routed_split, _ep_seed, _cfg, _task, _base_seed = self._route_triplet_for_global(g)
                if routed_split == split:
                    buf.append(g)
                g += 1
                scanned += 1
                if scanned > max_scan:
                    raise RuntimeError(
                        f"Unable to materialize split-local index {li} for split={split!r} "
                        f"after scanning {scanned} global indices. "
                        "Check router rules/ratios (a split may effectively have probability 0)."
                    )

            self._local_scan_cursor[split] = g
            return int(buf[li])


















    def episode(
        self,
        index: int,
        *,
        use_cache: bool = True,
        index_space: Literal["local", "global", "auto"] = "local",
    ) -> EpisodeNP:
        """
        Build a deterministic EpisodeNP.

        index_space:
          - "local" (default): `index` is split-local (0..), i.e. the i-th episode in this split.
          - "global": `index` is a global episode index routed via Phase B2 router.
          - "auto": treat as global if it routes to this split; otherwise treat as local.
        """
        idx = int(index)
        if idx < 0:
            raise ValueError("index must be >= 0")

        # Resolve index to a *global* episode id for routing + seed derivation.
        if index_space == "global":
            gidx = idx
        elif index_space == "local":
            gidx = self._global_index_from_local(self._split, idx)
        elif index_space == "auto":
            routed_split, _ep_seed, _cfg, _task, _base_seed = self._route_triplet_for_global(idx)
            if routed_split == self._split:
                gidx = idx
            else:
                gidx = self._global_index_from_local(self._split, idx)
        else:
            raise ValueError("index_space must be one of {'local','global','auto'}")

        key = (self._manifest_hex, self._split, int(gidx))
        if use_cache:
            hit = self._cache.get(key)
            if hit is not None:
                return hit

        # Canonical routing (global index -> split, ep_seed, cfg/task)
        routed_split, ep_seed, g_cfg, task, _base_seed = self._route_triplet_for_global(int(gidx))
        if routed_split != self._split:
            # This should only happen if the user forced index_space="global"
            raise ValueError(
                f"Global episode index {gidx} routes to split={routed_split!r}, "
                f"but this generator is split={self._split!r}. "
                f"Use episode(..., index_space='local') for split-local indexing."
            )

        # Episode RNG: drive graph realization, ψ0 randomization, noise sampling, etc.
        rng = np.random.Generator(np.random.PCG64(int(ep_seed)))

        # Build graph (as NumPy arrays)
        edge_index, degrees, coords, arc_ptr = _build_graph_from_config(
            g_cfg, seed=int(rng.integers(0, 2**31))
        )
        N = int(degrees.shape[0])
        A = int(edge_index.shape[1])
        assert arc_ptr.shape[0] == N + 1, "arc_slices must be (N+1,) CSR pointer"

        # PortMap (portable)
        pm = _portable_portmap_from_oriented(edge_index, arc_ptr, N)

        # Node features
        spec = _spec_from_manifest(self._manifest_raw)
        if torch is None:
            raise RuntimeError("Torch is required for building features in this project.")
        t_edge = torch.from_numpy(edge_index)
        t_deg = torch.from_numpy(degrees)
        t_coords = torch.from_numpy(coords)
        X_t, _ = build_node_features(t_edge, t_deg, t_coords, spec=spec)
        feats = X_t.detach().cpu().numpy().astype(np.float32, copy=False)

        # ψ0 + noise using episode RNG
        psi0 = self._build_initial_state(task, pm=pm, coords=coords, rng=rng)
        noise = self._sample_noise(self._manifest_raw.get("noise", {}), A=A, rng=rng)

        ep = EpisodeNP(
            edge_index=edge_index.astype(np.int64, copy=False),
            degrees=degrees.astype(np.int64, copy=False),
            coords=coords.astype(np.float32, copy=False),
            arc_slices=arc_ptr.astype(np.int64, copy=False),
            X=feats,
            features=feats,
            pm_tail=pm.src,
            pm_head=pm.dst,
            pm_rev=pm.rev,
            pm_node_ptr=pm.node_ptr,
            pm_node_arcs=pm.node_arcs,
            num_nodes=pm.num_nodes,
            num_arcs=pm.num_arcs,
            psi0=psi0.astype(np.complex64, copy=False),
            noise=noise,
            task=task,
            rng_seed=int(ep_seed),
            manifest_hexdigest=self._manifest_hex,
        )

        if use_cache:
            self._cache.put(key, ep)
        return ep

    def episode_global(self, global_index: int, *, use_cache: bool = True) -> EpisodeNP:
        return self.episode(global_index, use_cache=use_cache, index_space="global")




    def get(self, index: int) -> EpisodeNP:
        return self.episode(index)

    def to_torch(
        self,
        ep: EpisodeNP,
        *,
        device: torch.device | str | None = None,
        float_dtype: torch.dtype | None = None,
        complex_dtype: torch.dtype | None = None,
    ) -> EpisodeTorch:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch not available.")
        td = device if isinstance(device, (torch.device, str)) else None
        fdt = float_dtype or torch.float32
        cdt = complex_dtype or torch.complex64

        def T(x: np.ndarray, *, dtype: torch.dtype | None = None, is_complex: bool = False):
            t = torch.from_numpy(x)
            if is_complex:
                return t.to(device=td, dtype=cdt)
            return t.to(device=td, dtype=dtype or fdt)

        return EpisodeTorch(
            edge_index=T(ep.edge_index, dtype=torch.long),
            degrees=T(ep.degrees, dtype=torch.long),
            coords=T(ep.coords, dtype=fdt),
            arc_slices=T(ep.arc_slices, dtype=torch.long),
            X=T(ep.X, dtype=fdt),
            features=T(ep.features, dtype=fdt),
            pm_tail=T(ep.pm_tail, dtype=torch.long),
            pm_head=T(ep.pm_head, dtype=torch.long),
            pm_rev=T(ep.pm_rev, dtype=torch.long),
            pm_node_ptr=T(ep.pm_node_ptr, dtype=torch.long),
            pm_node_arcs=T(ep.pm_node_arcs, dtype=torch.long),
            psi0=T(ep.psi0, is_complex=True),
            noise=dict(ep.noise),
            task=dict(ep.task),
            rng_seed=ep.rng_seed,
            manifest_hexdigest=ep.manifest_hexdigest,
            device=torch.device(td) if td is not None else None,
        )

    # ---------------- Internals ----------------

    @staticmethod
    def _data_affecting_view(m: Mapping[str, Any]) -> Mapping[str, Any]:
        allow = {
            "family",
            "num_nodes",
            "N",
            "L",
            "Ly",
            "n",
            "p",
            "d",
            "k",
            "kx",
            "ky",
            "beta",
            "radius",
            "dim",
            "torus",
            "degree_preserving",
            "graph_params",
            "features",
            "task",
            "task_params",
            "T",
            "noise",
            "train",
            "val",
            "test",
        }
        return {k: m[k] for k in m.keys() if k in allow}

    @staticmethod
    def _choose_from_split_field(field: Any, *, split: SplitName, rng: np.random.Generator) -> Any:
        """
        Resolve a manifeseed = _derive_episode_seed(self._manifest_hex, self._split, int(index))st value that can be:
          • scalar,
          • a {split: value} mapping,
          • a list/tuple to sample from,
        possibly nested (e.g., {"train": [64, 96, 128]}).
        This function repeatedly unwraps until a scalar is obtained.
        """
        value = field
        while True:
            # Split-specific override
            if isinstance(value, Mapping) and split in value:
                value = value[split]
                continue
            # Sample from a candidate set
            if isinstance(value, (list, tuple)):
                if not value:
                    raise ValueError("Empty list in manifest field.")
                value = value[int(rng.integers(0, len(value)))]
                continue
            # Scalar / terminal
            return value


    # ---- Phase B2: canonical (global) materialization + triplet routing ----

    @staticmethod
    def _choose_global_field(field: Any, *, rng: np.random.Generator) -> Any:
        """
        Like _choose_from_split_field, but DOES NOT depend on self._split.
        If the field is a {train/val/test/all: ...} mapping, pick a stable canonical branch
        (all > train > val > test). Otherwise behave like normal sampling (lists/tuples).
        """
        value = field
        while True:
            if isinstance(value, Mapping):
                keys = set(value.keys())
                split_keys = {"train", "val", "test", "all"}
                if keys and keys.issubset(split_keys):
                    if "all" in value:
                        value = value["all"]
                        continue
                    if "train" in value:
                        value = value["train"]
                        continue
                    if "val" in value:
                        value = value["val"]
                        continue
                    value = value.get("test")
                    continue
                # non split-map: terminal for this unwrapping step
                return value

            if isinstance(value, (list, tuple)):
                if not value:
                    raise ValueError("Empty list in manifest field.")
                value = value[int(rng.integers(0, len(value)))]
                continue

            return value

    def _materialize_graph_config_global(
        self, m: Mapping[str, Any], *, rng: np.random.Generator
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        cfg["family"] = str(self._choose_global_field(m.get("family", "line"), rng=rng)).lower()

        for k in (
            "num_nodes",
            "N",
            "L",
            "Ly",
            "n",
            "p",
            "d",
            "k",
            "kx",
            "ky",
            "beta",
            "radius",
            "dim",
            "torus",
            "degree_preserving",
        ):
            if k in m:
                cfg[k] = self._choose_global_field(m.get(k), rng=rng)

        if "graph_params" in m and isinstance(m["graph_params"], Mapping):
            for k, v in m["graph_params"].items():
                cfg.setdefault(k, self._choose_global_field(v, rng=rng))

        return cfg

    def _materialize_task_global(self, *, N: int, rng: np.random.Generator) -> dict[str, Any]:
        m = self._manifest_raw
        name = str(self._choose_global_field(m.get("task", "transfer"), rng=rng)).lower()
        T = int(self._choose_global_field(m.get("T", max(64, N)), rng=rng))

        params: dict[str, Any] = {}
        if isinstance(m.get("task_params"), Mapping):
            for k, v in m["task_params"].items():
                params[k] = self._choose_global_field(v, rng=rng)

        if name == "transfer":
            src = int(params.get("source", 0))
            tgt = int(params.get("target", N - 1))
            return {"name": "transfer", "source": max(0, min(N - 1, src)), "target": max(0, min(N - 1, tgt)), "T": T}

        if name == "search":
            marks = params.get("marks")
            if not isinstance(marks, Iterable):
                marks = [int(rng.integers(0, N))]
            marks = sorted({int(max(0, min(N - 1, x))) for x in marks})
            return {"name": "search", "marks": marks, "T": T}

        if name == "mixing":
            return {"name": "mixing", "T": T}

        if name == "robust":
            tgt = int(params.get("target", N - 1))
            return {"name": "robust", "target": max(0, min(N - 1, tgt)), "T": T, "noise": params.get("noise", {})}

        return {"name": name, "T": T, **params}

    def _route_triplet_for_global(
        self, gidx: int
    ) -> tuple[SplitName, int, dict[str, Any], dict[str, Any], int]:
        """
        Canonical Phase B2 routing:
          global index gidx -> (split, ep_seed, cfg, task, base_seed)
        """
        g = int(gidx)
        if g < 0:
            raise ValueError("global episode index must be >= 0")

        base_seed = self._canonical_seed_for_global(g)

        # Coarse identity (cfg/task) from base_seed (independent of split)
        rng0 = np.random.Generator(np.random.PCG64(int(base_seed)))
        cfg = self._materialize_graph_config_global(self._manifest_raw, rng=rng0)

        # Infer N for task materialization (matches _make_cfg_task_for_global logic)
        N = None
        if "num_nodes" in cfg:
            N = int(cfg["num_nodes"])
        elif "N" in cfg:
            N = int(cfg["N"])
        elif "L" in cfg:
            L = int(cfg["L"])
            Ly = int(cfg.get("Ly", L))
            N = int(L * Ly)
        elif "n" in cfg and str(cfg.get("family", "")).lower() == "hypercube":
            N = 2 ** int(cfg["n"])
        else:
            N = 64

        task = self._materialize_task_global(N=int(N), rng=rng0)

        fam, size_kv = self._family_and_size_kv(cfg)
        tname = str(task.get("name", "transfer")).lower()

        split, ep_seed = route_triplet(
            family=fam,
            size_kv=size_kv,
            task=tname,
            base_seed=int(base_seed),
            episode_id=g,
            rules=self._router_rules,
            master_seed=int(self._router_master_seed),
        )
        return split, int(ep_seed), cfg, task, int(base_seed)




    def _materialize_graph_config(
        self, m: Mapping[str, Any], *, rng: np.random.Generator
    ) -> dict[str, Any]:
        split = self._split
        cfg: dict[str, Any] = {}
        cfg["family"] = str(
            self._choose_from_split_field(m.get("family", "line"), split=split, rng=rng)
        ).lower()

        # Size knobs
        def choose(name: str, default: Any) -> Any:
            return self._choose_from_split_field(m.get(name, default), split=split, rng=rng)

        for k in (
            "num_nodes",
            "N",
            "L",
            "Ly",
            "n",
            "p",
            "d",
            "k",
            "kx",
            "ky",
            "beta",
            "radius",
            "dim",
            "torus",
            "degree_preserving",
        ):
            if k in m:
                cfg[k] = choose(k, m[k])

        if "graph_params" in m and isinstance(m["graph_params"], Mapping):
            for k, v in m["graph_params"].items():
                cfg.setdefault(k, self._choose_from_split_field(v, split=split, rng=rng))

        return cfg

    def _materialize_task(self, *, N: int, rng: np.random.Generator) -> dict[str, Any]:
        m = self._manifest_raw
        name = str(
            self._choose_from_split_field(m.get("task", "transfer"), split=self._split, rng=rng)
        ).lower()
        T = int(self._choose_from_split_field(m.get("T", max(64, N)), split=self._split, rng=rng))
        params: dict[str, Any] = {}
        if isinstance(m.get("task_params"), Mapping):
            for k, v in m["task_params"].items():
                params[k] = self._choose_from_split_field(v, split=self._split, rng=rng)

        if name == "transfer":
            src = int(params.get("source", 0))
            tgt = int(params.get("target", N - 1))
            return {
                "name": "transfer",
                "source": max(0, min(N - 1, src)),
                "target": max(0, min(N - 1, tgt)),
                "T": T,
            }

        if name == "search":
            marks = params.get("marks")
            if not isinstance(marks, Iterable):
                marks = [int(rng.integers(0, N))]
            marks = sorted({int(max(0, min(N - 1, x))) for x in marks})
            return {"name": "search", "marks": marks, "T": T}

        if name == "mixing":
            return {"name": "mixing", "T": T}

        if name == "robust":
            tgt = int(params.get("target", N - 1))
            return {
                "name": "robust",
                "target": max(0, min(N - 1, tgt)),
                "T": T,
                "noise": params.get("noise", {}),
            }

        # Fallback
        return {"name": name, "T": T, **params}

        # ---- Helpers needed by export_eval_jsonl (deterministic, CI-only) ----

    def _canonical_seed_for_global(self, gidx: int) -> int:
        """
        Deterministically derive a small (31-bit) integer seed for a global index.
        Stable across processes and OSes; tied to this manifest.
        """
        h = blake2b(digest_size=8)
        h.update(self._manifest_hex.encode("utf-8"))
        h.update(b"|global|")
        h.update(str(int(gidx)).encode("utf-8"))
        raw = int.from_bytes(h.digest(), "big", signed=False)
        return int(raw % (2**31 - 1))

    def _make_cfg_task_for_global(self, gidx: int) -> tuple[dict, dict]:
        """
        Produce a *materialized* (graph-config, task) pair for a given global index,
        without constructing the episode. This mirrors the per-episode sampling logic
        but uses a canonical RNG derived from the manifest + global index.
        """
        seed = self._canonical_seed_for_global(gidx)
        rng = np.random.Generator(np.random.PCG64(seed))
        cfg = self._materialize_graph_config_global(self._manifest_raw, rng=rng)
        # Create a task consistent with the chosen size (N) using the same RNG stream.
        # We only need the *shape* of the task for metadata, not ψ0.
        # Use a small proxy N if not inferable (e.g., pure 'grid' with only L).
        # Try to infer N from cfg when possible.
        N = None
        if "num_nodes" in cfg:
            N = int(cfg["num_nodes"])
        elif "N" in cfg:
            N = int(cfg["N"])
        elif "L" in cfg and "Ly" in cfg:
            N = int(cfg["L"]) * int(cfg["Ly"])
        elif "L" in cfg:
            N = int(cfg["L"]) * int(cfg.get("Ly", cfg["L"]))
        elif "n" in cfg and str(cfg.get("family", "")).lower() == "hypercube":
            N = 2 ** int(cfg["n"])
        else:
            N = 64  # harmless, only for metadata if size can’t be inferred

        task = self._materialize_task_global(N=N, rng=rng)
        return dict(cfg), dict(task)

    def _family_and_size_kv(self, cfg: Mapping[str, Any]) -> tuple[str, dict]:
        """
        Extract (family, size_dict) for compact JSONL metadata. The size_dict has a
        single salient knob per family when possible (e.g., {'N': 128} or {'L': 8, 'Ly': 8}).
        """
        fam = str(cfg.get("family", "line")).lower()
        size = {}
        # Prefer the most natural control for each family
        if fam in {
            "line",
            "cycle",
            "er",
            "erdos_renyi",
            "d_regular",
            "d-regular",
            "regular",
            "ws",
            "watts",
            "watts_strogatz",
            "rgg",
            "random_geometric",
        }:
            if "num_nodes" in cfg:
                size = {"N": int(cfg["num_nodes"])}
            elif "N" in cfg:
                size = {"N": int(cfg["N"])}
        if not size and fam in {"grid", "ws_grid", "watts_strogatz_grid"}:
            L = int(cfg.get("L", 0)) if "L" in cfg else None
            Ly = int(cfg.get("Ly", 0)) if "Ly" in cfg else None
            if L is not None and Ly is not None and Ly > 0:
                size = {"L": L, "Ly": Ly}
            elif L is not None:
                size = {"L": L}
        if not size and fam == "hypercube":
            if "n" in cfg:
                size = {"n": int(cfg["n"])}
        # Final fallback for anything else
        if not size:
            # Pick any of the common knobs if present
            for k in ("N", "num_nodes", "L", "n"):
                if k in cfg:
                    size = {k: int(cfg[k]) if isinstance(cfg[k], (int, np.integer)) else cfg[k]}
                    break
        return fam, size

    @staticmethod
    def _gaussian_packet_on_coords(
        coords: np.ndarray, center: tuple[float, ...], sigma: float
    ) -> np.ndarray:
        """Compute normalized Gaussian over nodes using 2D (or C-D) coords."""
        if coords.size == 0:
            raise ValueError("Gaussian packet requested but no coordinates available.")
        C = coords.shape[1]
        if len(center) != C:
            raise ValueError("center dimensionality must match coords.")
        diff = coords - np.asarray(center, dtype=np.float32).reshape(1, C)
        dist2 = (diff * diff).sum(axis=1)
        w = np.exp(-0.5 * dist2 / max(sigma, 1e-12) ** 2).astype(np.float32)
        s = float(w.sum()) if w.size else 1.0
        return (w / max(s, 1e-12)).astype(np.float32)

    @staticmethod
    def _uniform_over_arcs(ptr: np.ndarray) -> np.ndarray:
        A = int(ptr[-1])
        if A == 0:
            return np.zeros(0, dtype=np.float32)
        return np.full(A, 1.0 / math.sqrt(float(A)), dtype=np.float32)

    def _build_initial_state(
        self,
        task: Mapping[str, Any],
        *,
        pm: _PortMapLite,
        coords: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Construct task-aware ψ0 in the *arc basis*.
        Modes supported via task['init'] (optional): 'source_uniform', 'uniform_arcs',
        'gaussian_packet', 'custom_arcs'.
        """
        name = str(task.get("name", "transfer")).lower()
        init_mode = str(task.get("init", "source_uniform")).lower()

        A = pm.num_arcs
        psi = np.zeros(A, dtype=np.complex64)

        if init_mode == "uniform_arcs" or name == "mixing":
            amp = 1.0 / math.sqrt(float(A)) if A > 0 else 0.0
            psi[:] = amp + 0.0j
            return psi

        if init_mode == "gaussian_packet":
            # Build node distribution then split uniformly among outgoing arcs
            center = tuple(task.get("center", (0.5, 0.5)))
            sigma = float(task.get("sigma", 0.15))
            pv = self._gaussian_packet_on_coords(coords, center=center, sigma=sigma)  # (N,)
            # expand to arcs
            for u in range(pm.num_nodes):
                s, e = int(pm.node_ptr[u]), int(pm.node_ptr[u + 1])
                deg = e - s
                if deg <= 0:
                    continue
                amp = math.sqrt(float(pv[u])) / math.sqrt(float(max(deg, 1)))
                psi[pm.node_arcs[s:e]] = amp + 0.0j
            # renormalize (numerical safety)
            n2 = float((psi.conj() * psi).real.sum())
            if n2 > 0:
                psi /= math.sqrt(n2)
            return psi

        if init_mode == "custom_arcs":
            ids = np.asarray(task.get("arc_ids", []), dtype=np.int64)
            if ids.size:
                amp = 1.0 / math.sqrt(float(ids.size))
                psi[ids] = amp + 0.0j
                return psi

        # Default / transfer: uniform on outgoing arcs of source node
        src = int(task.get("source", 0))
        s, e = int(pm.node_ptr[src]), int(pm.node_ptr[src + 1])
        if e > s:
            amp = 1.0 / math.sqrt(float(e - s))
            psi[pm.node_arcs[s:e]] = amp + 0.0j
        else:
            # Source has no outgoing arcs (isolated node). Fall back to uniform over arcs.
            if A > 0:
                amp = 1.0 / math.sqrt(float(A))
                psi[:] = amp + 0.0j

        return psi

    @staticmethod
    def _sample_noise(
        noise_block: Mapping[str, Any], *, A: int, rng: np.random.Generator
    ) -> dict[str, Any]:
        """
        Prepare a noise recipe to be consumed by the simulator (optional).
        We only *sample parameters* here; no mutation of ψ0 or the graph.
        """
        out: dict[str, Any] = {}
        # Edge phases ~ N(0, sigma^2)
        if "edge_phases" in noise_block:
            sigma = float(noise_block.get("edge_phases", {}).get("sigma", 0.0))
            if sigma > 0:
                # do not generate a giant array if not needed; store seed + sigma
                out["edge_phases"] = {"sigma": sigma, "seed": int(rng.integers(0, 2**31))}
        # Dephasing rate
        if "dephasing" in noise_block:
            gamma = float(noise_block.get("dephasing", {}).get("gamma", 0.0))
            if gamma > 0:
                out["dephasing"] = {"gamma": gamma}
        return out

    # ---------------- CI reproducibility exports (new) ----------------
    def export_eval_index(
        self,
        split: SplitName,
        *,
        count: int | None = None,
        
        base_offset: int | None = None,
    ) -> list[tuple[int, int]]:
        """
        Return a deterministic list of (global_idx, episode_seed) pairs for CI-style evaluation.

        Parameters
        ----------
        split:
            Must be "val" or "test".
        count:
            Number of indices to generate. If None, we attempt to read:
                manifest["split_counts"][split]
            and fall back to 64 (val) / 128 (test).
        base_offset:
            Optional additive offset used to generate "global_idx" values:
                global_idx = base_offset + i
            If None, a stable offset is derived from the manifest hex digest.

        Important invariants
        --------------------
        * `global_idx` is the *index you should pass into* `EpisodeGenerator.episode(global_idx)`.
        * `episode_seed` is **exactly** `_derive_episode_seed(manifest_hex, split, global_idx)`.

        This makes the eval manifest self-consistent: you can always regenerate the same
        episode from (manifest_hex, split, global_idx) and obtain the stored episode_seed.

        Notes
        -----
        - This helper is CI-only: it does *not* depend on `_mat_indices` (the training sampler).
        - We intentionally keep the "global_idx" space shifted by a manifest-derived base
          offset so indices from different manifests do not trivially collide when you
          concatenate multiple eval lists.
        """
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'.")

        # Resolve count (priority: explicit arg > manifest["split_counts"] > defaults)
        if count is None:
            sc = self._manifest_raw.get("split_counts", {})
            if isinstance(sc, Mapping) and split in sc:
                try:
                    count = int(sc[split])
                except Exception:
                    count = None
        if count is None:
            count = 64 if split == "val" else 128

        count_i = int(count)
        if count_i < 0:
            raise ValueError("count must be >= 0")

        # Resolve base offset
        if base_offset is None:
            base_offset = int(int(self._manifest_hex[:16], 16) % (2**31 - 1))
        base_i = int(base_offset)

        key = (split, count_i, base_i)
        cached = self._eval_index_cache.get(key)
        if cached is not None:
            return list(cached)

        out: list[tuple[int, int]] = []
        g = int(base_i)
        while len(out) < count_i:
            routed_split, ep_seed, _cfg, _task, _base_seed = self._route_triplet_for_global(g)
            if routed_split == split:
                out.append((g, int(ep_seed)))
            g += 1

        self._eval_index_cache[key] = out
        return list(out)
    def export_eval_jsonl(
        self,
        *,
        split: SplitName,
        out_dir: str | bytes | os.PathLike,
        overwrite: bool = False,
        count: int | None = None,
        base_offset: int | None = None,
    ) -> tuple[str, str]:
        """
        Materialize the eval list and write to JSONL at:
            {out_dir}/{manifest_hex}/{split}.jsonl
        and create/update an index sidecar:
            {out_dir}/{manifest_hex}/index.json

        Returns (jsonl_path, index_path).
        """
        from hashlib import blake2b, sha256
        import json
        import os
        from pathlib import Path

        # Build entries (without constructing episodes)
        pairs = self.export_eval_index(split, count=count, base_offset=base_offset)
        entries = []
        for gidx, epseed in pairs:
            routed_split, ep_seed2, cfg, task, base_seed = self._route_triplet_for_global(gidx)
            if routed_split != split:
                raise ValueError(f"export_eval_jsonl: index {gidx} routed to {routed_split}, expected {split}")
            fam, size_kv = self._family_and_size_kv(cfg)

            entries.append(
                {
                    "manifest_hex": self._manifest_hex,
                    "router_master_seed": int(self._router_master_seed),
                    "split": split,
                    "global_idx": int(gidx),
                    "episode_seed": int(epseed),
                    "family": fam,
                    "size": dict(size_kv),
                    "task": str(task.get("name", "transfer")).lower(),
                    "base_seed": int(base_seed),
                }
            )

        # Paths
        base = Path(out_dir) / self._manifest_hex
        base.mkdir(parents=True, exist_ok=True)
        jsonl_path = base / f"{split}.jsonl"
        idx_path = base / "index.json"

        if jsonl_path.exists() and not overwrite:
            raise FileExistsError(f"{jsonl_path} exists; set overwrite=True to replace.")

        # Atomic write JSONL
        tmp = jsonl_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for e in entries:
                f.write(json.dumps(e, sort_keys=True, separators=(",", ":")))
                f.write("\n")
        os.replace(tmp, jsonl_path)

        # Compute strong digests for index
        h_b2 = blake2b(digest_size=32)
        h_sha = sha256()
        with jsonl_path.open("rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h_b2.update(chunk)
                h_sha.update(chunk)

        # Merge/update sidecar index (manifest-scoped)
        meta = {
            "schema_version": 1,
            "created_unix": int(time.time()),
            "manifest_hex": self._manifest_hex,
            "rules": "phase_b2_default_v1",
            "splits": {},
        }
        if idx_path.exists():
            try:
                old = json.loads(idx_path.read_text(encoding="utf-8"))
                if isinstance(old, dict) and old.get("manifest_hex") == self._manifest_hex:
                    meta.update({k: old[k] for k in old.keys() if k not in ("splits",)})
                    meta["splits"] = old.get("splits", {})
            except Exception:
                pass

        # Update split entry
        meta["splits"][split] = {
            "file": jsonl_path.name,
            "count": len(entries),
            "blake2b_256": h_b2.hexdigest(),
            "sha256": h_sha.hexdigest(),
        }

        # Atomic write index
        tmpi = idx_path.with_suffix(".json.tmp")
        tmpi.write_text(json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(tmpi, idx_path)

        return (str(jsonl_path), str(idx_path))


# --- BEGIN: shift matrix indices helper expected by tests ---------------------


def mat_indices_from_portmap(
    *,
    num_arcs: int,
    pm_rev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Build COO indices (row, col) for the SHIFT operator S over the arc basis:

        S |u→v⟩ = |v→u⟩

    With our portmap, this is simply a permutation matrix that maps each arc
    index i to its reverse index pm_rev[i]. The resulting sparse matrix has:

        rows = pm_rev
        cols = arange(num_arcs)
        shape = (num_arcs, num_arcs)

    This is device/framework-agnostic; tests can feed these indices into their
    own PyTorch/NumPy/SciPy builders.
    """
    A = int(num_arcs)
    if A < 0:
        raise ValueError("num_arcs must be non-negative")
    if pm_rev is None:
        raise ValueError("pm_rev is required")
    pm_rev = np.asarray(pm_rev, dtype=np.int64)
    if pm_rev.shape != (A,):
        raise ValueError(f"pm_rev must have shape ({A},), got {pm_rev.shape}")
    # Basic sanity: permutation (allow fixed points if directed arcs lack explicit reverses)
    if (pm_rev < 0).any() or (pm_rev >= A).any():
        raise ValueError("pm_rev indices out of bounds")
    cols = np.arange(A, dtype=np.int64)
    rows = pm_rev.astype(np.int64, copy=False)
    shape = (A, A)
    return rows, cols, shape


def _mat_indices_from_episode(ep: EpisodeNP) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Convenience wrapper to get shift indices directly from an EpisodeNP.
    """
    return mat_indices_from_portmap(num_arcs=ep.num_arcs, pm_rev=ep.pm_rev)


# Back-compat hook expected by tests: EpisodeGenerator._mat_indices(...)
# Provide as a staticmethod so tests can call either EpisodeGenerator._mat_indices(ep)
# or EpisodeGenerator._mat_indices(num_arcs=..., pm_rev=...).
def _eg_mat_indices_adapter(
    *, ep: EpisodeNP | None = None, num_arcs: int | None = None, pm_rev: np.ndarray | None = None
):
    if ep is not None:
        return _mat_indices_from_episode(ep)
    if num_arcs is None or pm_rev is None:
        raise ValueError("Provide either ep=EpisodeNP or both num_arcs and pm_rev.")
    return mat_indices_from_portmap(num_arcs=num_arcs, pm_rev=pm_rev)


# Attach for tests that import/expect this symbol
try:
    EpisodeGenerator._mat_indices = staticmethod(_eg_mat_indices_adapter)  # type: ignore[attr-defined]
except Exception:
    # If EpisodeGenerator isn't defined yet for some reason, ignore; typical import order defines it above.
    pass

# --- END: shift matrix indices helper -----------------------------------------

# acpl/data/manifest.py
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum, auto
from hashlib import blake2b, sha256
import json

# ---------------- New imports for CI reproducibility ----------------
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import yaml

# Generator + split type used by CI helpers
try:
    from .generator import EpisodeGenerator, SplitName  # type: ignore
except Exception:  # pragma: no cover
    EpisodeGenerator = None  # type: ignore
    SplitName = Literal["train", "val", "test"]  # type: ignore

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

__all__ = [
    # Manifest + hashing I/O
    "Manifest",
    "build_manifest",
    "hash_bytes_blake2b",
    "hash_manifest",
    "seed_from_hash",
    "save_manifest",
    "load_manifest",
    # Episode cache
    "EpisodeCache",
    "EvictionReason",
    "CacheStats",
    # CI reproducibility (new)
    "make_eval_manifest",
    "read_eval_manifest",
    "verify_manifest",
]


# ---------------------------------------------------------------------
# Dataclass for a simple, seedable manifest
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Manifest:
    """
    Seedable, hashable manifest describing one procedural episode dataset.

    Fields
    ------
    version : int
        Schema version (bump if you change canonicalization expectations).
    configs : dict[str, Any]
        Free-form config blocks (e.g., {"graph": {...}, "task": {...}, "model": {...}}).
        Must be JSON/YAML-serializable. NumPy types are allowed; they are canonicalized.
    seed : int
        Integer seed used to generate this episode (derived or provided).
    hexdigest : str
        BLAKE2b-256 hex digest over the canonicalized (version, configs, seed).
    """

    version: int
    configs: dict[str, Any]
    seed: int
    hexdigest: str


# ---------------------------------------------------------------------
# Canonicalization & hashing
# ---------------------------------------------------------------------


def _to_builtin(obj: Any) -> Any:
    """
    Convert common non-JSON-native types (NumPy scalars/arrays) to Python builtins
    so that deterministic JSON can be produced.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)  # type: ignore[no-redef]
    if isinstance(obj, (np.floating,)):
        return float(obj)  # type: ignore[no-redef]
    if isinstance(obj, (np.ndarray,)):
        if obj.ndim == 0:
            return _to_builtin(obj.item())
        return [_to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        # Canonical: sort after string-coercion to get stability
        return sorted((_to_builtin(x) for x in obj), key=lambda z: json.dumps(z, sort_keys=True))
    if isinstance(obj, Mapping):
        return {str(k): _to_builtin(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    # Fallback: stable string form via repr
    return repr(obj)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """
    Deterministic JSON bytes: sorted keys, compact separators, ASCII only.
    """
    builtin = _to_builtin(payload)
    data = json.dumps(builtin, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return data.encode("utf-8")


def hash_bytes_blake2b(data: bytes, *, digest_bytes: int = 32, salt: bytes | None = None) -> bytes:
    """
    BLAKE2b hash (default 256-bit) with optional salt.
    """
    h = blake2b(digest_size=digest_bytes, salt=salt or b"")
    h.update(data)
    return h.digest()


def _manifest_payload_for_hash(version: int, configs: Mapping[str, Any], seed: int) -> bytes:
    return _canonical_json_bytes({"version": int(version), "configs": configs, "seed": int(seed)})


def hash_manifest(version: int, configs: Mapping[str, Any], seed: int) -> str:
    """
    Hex digest over (version, configs, seed) using BLAKE2b-256.
    """
    payload = _manifest_payload_for_hash(version, configs, seed)
    return hash_bytes_blake2b(payload, digest_bytes=32).hex()


def seed_from_hash(hexdigest: str, *, mod: int = 2**32) -> int:
    """
    Map a hex digest to a nonnegative integer seed in [0, mod).
    Suitable for deterministic-but-uniform-ish seed derivation from a manifest.
    """
    value = int(hexdigest, 16)
    return int(value % mod)


# ---------------------------------------------------------------------
# Build, save, load
# ---------------------------------------------------------------------


def build_manifest(
    *,
    version: int,
    configs: Mapping[str, Any],
    seed: int | None = None,
    derive_seed_if_none: bool = True,
) -> Manifest:
    """
    Build a Manifest with a stable hash. If seed is None and derive_seed_if_none=True,
    derive the seed from the hash of (version, configs, seed=0) to keep determinism.
    """
    cfgs = dict(configs)  # shallow copy
    base_seed = 0 if seed is None else int(seed)
    # Provisional hash used to derive seed if needed
    provisional_hex = hash_manifest(version, cfgs, base_seed)
    final_seed = base_seed
    if seed is None and derive_seed_if_none:
        final_seed = seed_from_hash(provisional_hex)

    final_hex = hash_manifest(version, cfgs, final_seed)
    return Manifest(
        version=int(version),
        configs=cfgs,
        seed=int(final_seed),
        hexdigest=final_hex,
    )


def _infer_format(path: str | Path) -> str:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".yml", ".yaml"}:
        return "yaml"
    if suf == ".json":
        return "json"
    # Default to yaml if unknown
    return "yaml"


def save_manifest(path: str | Path, manifest: Manifest, *, fmt: str | None = None) -> None:
    """
    Save to YAML (default) or JSON. Format inferred from suffix if fmt is None.
    """
    out_fmt = (fmt or _infer_format(path)).lower()
    payload = {
        "version": manifest.version,
        "seed": manifest.seed,
        "hexdigest": manifest.hexdigest,
        "configs": _to_builtin(manifest.configs),
    }
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if out_fmt == "json":
        txt = json.dumps(payload, sort_keys=True, indent=2)
        path_obj.write_text(txt, encoding="utf-8")
    else:
        txt = yaml.safe_dump(payload, sort_keys=True)
        path_obj.write_text(txt, encoding="utf-8")


def load_manifest(path: str | Path) -> Manifest:
    """
    Load YAML or JSON and verify the hash. Raises ValueError on mismatch.
    """
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    if path_obj.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    version = int(data["version"])
    seed = int(data["seed"])
    hexdigest = str(data["hexdigest"])
    configs = dict(data["configs"])
    expect = hash_manifest(version, configs, seed)
    if expect != hexdigest:
        raise ValueError(
            "Manifest hash mismatch: file reports "
            f"{hexdigest}, but recomputed {expect} from content."
        )
    return Manifest(version=version, configs=configs, seed=seed, hexdigest=hexdigest)


# ---------------------------------------------------------------------
# Feature-rich in-memory LRU cache for Episodes
# ---------------------------------------------------------------------


class EvictionReason(Enum):
    """Reason an entry left the cache."""

    CAPACITY = auto()
    COST = auto()
    TTL = auto()
    EXPLICIT = auto()
    REPLACE = auto()
    CLEAR = auto()


@dataclass
class CacheStats:
    """Lightweight, cumulative cache metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    created: int = 0
    expired: int = 0
    replaced: int = 0

    def copy(self) -> CacheStats:
        return CacheStats(**asdict(self))

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        req = self.requests
        return float(self.hits) / req if req else 0.0


@dataclass
class _Item(Generic[V]):
    value: V
    cost: int
    created_at: float
    last_access: float
    hits: int = 0
    ttl: float | None = None  # seconds; None => immortal
    # expires_at is derived by accessor to avoid drift


def _now() -> float:
    return time.monotonic()


class EpisodeCache(Generic[K, V]):
    """
    A robust, thread-safe, in-memory LRU cache with:
      • fixed entry capacity (LRU eviction)
      • optional total-cost budget with user-provided cost function
      • optional TTL per entry (global default and per-insert override)
      • stats, info(), snapshot(), and flush/resize utilities
      • eviction callbacks
      • stampede protection on get_or_create (per-key locks)
      • memoize() decorator for functions producing episodes

    Notes
    -----
    - LRU discipline is maintained on successful get/peek/put paths.
    - If both capacity and max_cost are set, eviction prunes by LRU
      until both constraints are satisfied.
    """

    # ---------------- construction ----------------

    def __init__(
        self,
        capacity: int = 128,
        *,
        max_cost: int | None = None,
        cost_fn: Callable[[V], int] | None = None,
        default_ttl: float | None = None,
        refresh_on_access: bool = True,
        namespace: str | None = None,
        on_evict: Callable[[K, V, EvictionReason], None] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be >= 1")
        if max_cost is not None and max_cost <= 0:
            raise ValueError("max_cost must be None or >= 1")

        self.capacity = int(capacity)
        self.max_cost = max_cost
        self.cost_fn = cost_fn or (lambda _: 1)  # default: unit cost
        self.default_ttl = default_ttl
        self.refresh_on_access = bool(refresh_on_access)
        self.namespace = namespace or "default"
        self.on_evict = on_evict

        self._data: OrderedDict[K, _Item[V]] = OrderedDict()
        self._total_cost: int = 0
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._key_locks: dict[K, threading.Lock] = {}  # stampede guards

    # ---------------- internals ----------------

    def _evict_one(self, reason: EvictionReason) -> None:
        """Evict least-recently-used item."""
        if not self._data:
            return
        key, item = self._data.popitem(last=False)  # pop LRU
        self._total_cost -= item.cost
        self._stats.evictions += 1
        if reason == EvictionReason.TTL:
            self._stats.expired += 1
        if self.on_evict:
            try:
                self.on_evict(key, item.value, reason)
            except Exception:
                # Eviction callback should never break caching
                pass

    def _prune(self) -> None:
        """Enforce capacity and cost constraints."""
        # capacity
        while len(self._data) > self.capacity:
            self._evict_one(EvictionReason.CAPACITY)
        # cost
        if self.max_cost is not None:
            while self._total_cost > self.max_cost and self._data:
                self._evict_one(EvictionReason.COST)

    def _expired(self, item: _Item[V], now: float | None = None) -> bool:
        if item.ttl is None:
            return False
        t = now if now is not None else _now()
        return (t - item.created_at) > item.ttl

    def _touch(self, key: K, item: _Item[V], now: float | None = None) -> None:
        """Mark as most-recent; update last_access and optionally refresh TTL."""
        t = now if now is not None else _now()
        item.last_access = t
        if self.refresh_on_access and item.ttl is not None:
            # refresh 'created_at' to extend life on access
            # (sliding expiration). If you want fixed expiration,
            # set refresh_on_access=False.
            item.created_at = t
        # move to MRU end
        self._data.move_to_end(key, last=True)

    # ---------------- basic mapping-like API ----------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: K) -> bool:
        return self.peek(key) is not None

    def clear(self) -> None:
        with self._lock:
            if self.on_evict:
                # emit callbacks for all
                for k, it in list(self._data.items()):
                    try:
                        self.on_evict(k, it.value, EvictionReason.CLEAR)
                    except Exception:
                        pass
            self._data.clear()
            self._key_locks.clear()
            self._total_cost = 0
            # keep stats but expose a method to reset
            # (clear shouldn't erase history)

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = CacheStats()

    def stats(self) -> CacheStats:
        with self._lock:
            return self._stats.copy()

    def total_cost(self) -> int:
        with self._lock:
            return self._total_cost

    def keys(self) -> Iterable[K]:
        with self._lock:
            return list(self._data.keys())

    def items(self) -> Iterable[tuple[K, V]]:
        with self._lock:
            return [(k, it.value) for k, it in self._data.items()]

    # ------------- core ops -------------

    def get(self, key: K, default: V | None = None) -> V | None:
        now = _now()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._stats.misses += 1
                return default
            if self._expired(item, now):
                # Evict and count as miss
                self._data.pop(key, None)
                self._total_cost -= item.cost
                self._stats.misses += 1
                self._evict_with(key, item, EvictionReason.TTL)
                return default
            # hit
            self._stats.hits += 1
            item.hits += 1
            self._touch(key, item, now)
            return item.value

    def peek(self, key: K) -> V | None:
        """Like get but does not update LRU order or stats; still enforces TTL."""
        now = _now()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            if self._expired(item, now):
                self._data.pop(key, None)
                self._total_cost -= item.cost
                self._evict_with(key, item, EvictionReason.TTL)
                return None
            return item.value

    def put(self, key: K, value: V, *, ttl: float | None = None, replace: bool = True) -> None:
        """
        Insert or update a value.
        - ttl: per-entry TTL (seconds). If None, uses default_ttl from cache.
        - replace: if False and key exists & not expired, do nothing.
        """
        now = _now()
        with self._lock:
            existing = self._data.get(key)
            if existing is not None and not self._expired(existing, now):
                if not replace:
                    return
                # replacing existing
                old = self._data.pop(key)
                self._total_cost -= old.cost
                self._stats.replaced += 1
                self._evict_with(key, old, EvictionReason.REPLACE, emit=False)

            eff_ttl = ttl if ttl is not None else self.default_ttl
            cost = int(self.cost_fn(value))
            item = _Item(
                value=value,
                cost=cost,
                created_at=now,
                last_access=now,
                ttl=eff_ttl,
            )
            self._data[key] = item
            self._total_cost += cost
            # new insert => MRU end by default
            self._data.move_to_end(key, last=True)
            self._stats.created += 1
            self._prune()

    def delete(self, key: K) -> bool:
        with self._lock:
            item = self._data.pop(key, None)
            if item is None:
                return False
            self._total_cost -= item.cost
            self._evict_with(key, item, EvictionReason.EXPLICIT)
            return True

    def pop(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            item = self._data.pop(key, None)
            if item is None:
                return default
            self._total_cost -= item.cost
            self._evict_with(key, item, EvictionReason.EXPLICIT)
            return item.value

    def resize(self, *, capacity: int | None = None, max_cost: int | None | None = None) -> None:
        """
        Dynamically adjust constraints; triggers pruning if needed.
        """
        with self._lock:
            if capacity is not None:
                if capacity <= 0:
                    raise ValueError("capacity must be >= 1")
                self.capacity = int(capacity)
            if max_cost is not None:
                if max_cost <= 0:
                    raise ValueError("max_cost must be None or >= 1")
                self.max_cost = int(max_cost)
            self._prune()

    def get_or_create(
        self,
        key: K,
        factory: Callable[[K], V],
        *,
        ttl: float | None = None,
        replace: bool = False,
        stampede_guard: bool = True,
    ) -> V:
        """
        Retrieve value if present & fresh; otherwise compute via `factory(key)`,
        insert, and return. If stampede_guard=True, concurrent creators for the
        same key are serialized.
        """
        # Fast path: try without per-key lock
        val = self.get(key)
        if val is not None:
            return val

        # Acquire per-key creation lock if required
        lock: threading.Lock | None = None
        if stampede_guard:
            with self._lock:
                lock = self._key_locks.get(key)
                if lock is None:
                    lock = threading.Lock()
                    self._key_locks[key] = lock
        if lock is not None:
            lock.acquire()

        try:
            # Check again after acquiring the lock (double-checked)
            val2 = self.get(key)
            if val2 is not None:
                return val2
            # Create outside the main lock
            created = factory(key)
            self.put(key, created, ttl=ttl, replace=replace)
            return created
        finally:
            if lock is not None:
                lock.release()
                # Cleanup empty lock holder
                with self._lock:
                    # Only remove if still pointing at the same lock
                    if self._key_locks.get(key) is lock:
                        self._key_locks.pop(key, None)

    # ---------------- utilities ----------------

    def memoize(
        self,
        *,
        key_fn: Callable[..., K] | None = None,
        ttl: float | None = None,
        replace: bool = False,
        stampede_guard: bool = True,
    ) -> Callable[[Callable[..., V]], Callable[..., V]]:
        """
        Decorator to memoize an episode-producing function into this cache.

        Example
        -------
        @cache.memoize(ttl=600)
        def build_episode(manifest: Manifest) -> Episode: ...

        The default key is a stable JSON key over args/kwargs; override via key_fn.
        """

        def _default_key_fn(*args: Any, **kwargs: Any) -> K:
            try:
                blob = json.dumps(
                    [args, kwargs], sort_keys=True, default=_to_builtin, separators=(",", ":")
                )
            except TypeError:
                # Fallback: repr-based (less strict but stable enough in practice)
                blob = repr((args, sorted(kwargs.items(), key=lambda kv: kv[0])))
            # Namespacing by function identity to avoid collisions across different memoized functions
            return blob + "|" + getattr(fn, "__qualname__", getattr(fn, "__name__", "fn"))  # type: ignore

        key_fn_local = key_fn or _default_key_fn

        def deco(fn: Callable[..., V]) -> Callable[..., V]:
            def wrapped(*args: Any, **kwargs: Any) -> V:
                k = key_fn_local(*args, **kwargs)
                return self.get_or_create(
                    k,
                    lambda _k: fn(*args, **kwargs),
                    ttl=ttl,
                    replace=replace,
                    stampede_guard=stampede_guard,
                )

            # Preserve basic metadata
            wrapped.__name__ = getattr(fn, "__name__", "wrapped")
            wrapped.__doc__ = fn.__doc__
            wrapped.__qualname__ = getattr(fn, "__qualname__", wrapped.__name__)
            return wrapped

        return deco

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "namespace": self.namespace,
                "size": len(self._data),
                "capacity": self.capacity,
                "max_cost": self.max_cost,
                "total_cost": self._total_cost,
                "default_ttl": self.default_ttl,
                "refresh_on_access": self.refresh_on_access,
                "stats": asdict(self._stats),
            }

    def snapshot(self, include_values: bool = False) -> list[dict[str, Any]]:
        """
        Inspect current entries (LRU→MRU order). For debugging/telemetry.
        """
        with self._lock:
            out: list[dict[str, Any]] = []
            for k, it in self._data.items():
                rec = {
                    "key": k,
                    "cost": it.cost,
                    "hits": it.hits,
                    "age_sec": _now() - it.created_at,
                    "ttl": it.ttl,
                }
                if include_values:
                    rec["value"] = it.value
                out.append(rec)
            return out

    # ---------------- helpers ----------------

    def _evict_with(
        self, key: K, item: _Item[V], reason: EvictionReason, *, emit: bool = True
    ) -> None:
        if emit and self.on_evict:
            try:
                self.on_evict(key, item.value, reason)
            except Exception:
                pass


# ---------------------------------------------------------------------
# CI-grade eval manifest writer/reader/validator + CLI (NEW)
# ---------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _file_digests(path: Path) -> tuple[str, str]:
    h_b2 = blake2b(digest_size=32)
    h_sha = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h_b2.update(chunk)
            h_sha.update(chunk)
    return h_b2.hexdigest(), h_sha.hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_config(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    raise TypeError("config must be a Mapping (e.g., dict parsed from YAML/JSON).")


def make_eval_manifest(
    config: Mapping[str, Any],
    *,
    split: SplitName,
    count: int,
    out_dir: str | os.PathLike,
    overwrite: bool = False,
) -> tuple[str, str]:
    """
    Deterministically generate an eval manifest (JSONL) for 'split' using EpisodeGenerator,
    without constructing full episodes. Returns (jsonl_path, index_path).
    """
    if EpisodeGenerator is None:  # pragma: no cover
        raise RuntimeError("EpisodeGenerator is not available; cannot build eval manifest.")
    if split not in ("val", "test"):
        raise ValueError("make_eval_manifest only supports 'val' and 'test'.")

    cfg = _normalize_config(config)
    # Respect per-split knobs when previewing graph/task choices
    gen = EpisodeGenerator(cfg, split=split)

    # Ensure desired count is honored by configuring generator's split_counts
    # (so export_eval_index size is deterministic and capped).
    pairs = gen.export_eval_index(split)  # already respects split_counts in generator

    base = Path(out_dir) / gen.manifest_hexdigest
    _ensure_dir(base)
    jsonl_path = base / f"{split}.jsonl"
    idx_path = base / "index.json"

    if jsonl_path.exists() and not overwrite:
        raise FileExistsError(f"{jsonl_path} exists; use overwrite=True to replace.")

    # Build entries
    entries = []
    for gidx, epseed in pairs:
        base_seed = gen._canonical_seed_for_global(gidx)  # deterministic by design
        cfg_i, task_i = gen._make_cfg_task_for_global(gidx)
        fam, size_kv = gen._family_and_size_kv(cfg_i)
        entries.append(
            {
                "manifest_hex": gen.manifest_hexdigest,
                "router_master_seed": int(gen._router_master_seed),
                "split": split,
                "global_idx": int(gidx),
                "episode_seed": int(epseed),
                "family": fam,
                "size": dict(size_kv),
                "task": str(task_i.get("name", "transfer")).lower(),
                "base_seed": int(base_seed),
            }
        )

    # Atomic write JSONL
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True, separators=(",", ":")))
            f.write("\n")
    os.replace(tmp, jsonl_path)

    # Compute digests
    b2, sh = _file_digests(jsonl_path)

    # Update/Write index.json
    meta = {
        "schema_version": 1,
        "created_unix": int(time.time()),
        "manifest_hex": gen.manifest_hexdigest,
        "rules": "phase_b2_default_v1",
        "splits": {},
    }
    if idx_path.exists():
        try:
            old = json.loads(idx_path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and old.get("manifest_hex") == gen.manifest_hexdigest:
                meta.update({k: old[k] for k in old.keys() if k not in ("splits",)})
                meta["splits"] = old.get("splits", {})
        except Exception:
            pass

    meta["splits"][split] = {
        "file": jsonl_path.name,
        "count": len(entries),
        "blake2b_256": b2,
        "sha256": sh,
    }
    _atomic_write_text(idx_path, json.dumps(meta, sort_keys=True, indent=2))
    return (str(jsonl_path), str(idx_path))


def read_eval_manifest(index_path: str | os.PathLike, split: SplitName) -> list[dict]:
    """
    Load entries from index+jsonl, verifying basic schema presence (does not recompute digests).
    Returns list of JSON objects (one per line).
    """
    idx = Path(index_path)
    base = idx.parent
    index = json.loads(idx.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1:
        raise ValueError("Unsupported schema_version in index.json")
    if split not in index.get("splits", {}):
        raise ValueError(f"Split {split!r} not listed in index.json")

    jsonl_path = base / index["splits"][split]["file"]
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing JSONL for split: {jsonl_path}")

    out = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def verify_manifest(
    index_path: str | os.PathLike, split: SplitName, *, strict: bool = True
) -> bool:
    """
    Verify that digests in index.json match the actual JSONL content and that manifest_hex
    in entries matches index's manifest_hex. In 'strict' mode, raises on any failure;
    otherwise returns False.
    """
    idx = Path(index_path)
    base = idx.parent
    index = json.loads(idx.read_text(encoding="utf-8"))

    man_hex = index.get("manifest_hex")
    if not man_hex:
        if strict:
            raise ValueError("index.json missing 'manifest_hex'")
        return False

    srec = index.get("splits", {}).get(split)
    if not srec:
        if strict:
            raise ValueError(f"index.json missing split entry for {split}")
        return False

    jsonl_path = base / srec["file"]
    if not jsonl_path.exists():
        if strict:
            raise FileNotFoundError(f"Missing {jsonl_path}")
        return False

    b2, sh = _file_digests(jsonl_path)
    ok = (b2 == srec.get("blake2b_256")) and (sh == srec.get("sha256"))

    if not ok and strict:
        raise ValueError("Digest mismatch for JSONL manifest")

    # Validate manifest_hex inside entries
    for e in read_eval_manifest(index_path, split):
        if e.get("manifest_hex") != man_hex:
            if strict:
                raise ValueError("Entry manifest_hex does not match index manifest_hex")
            return False

    return ok


# ------------------------------------ CLI ------------------------------------- #


def _load_config(path: str) -> Mapping[str, Any]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    # try JSON first
    try:
        return json.loads(txt)
    except Exception:
        pass
    return yaml.safe_load(txt)


def _cli(argv: Sequence[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ACPL eval manifest builder/validator")
    ap.add_argument(
        "--config", type=str, required=True, help="YAML/JSON config for EpisodeGenerator"
    )
    ap.add_argument("--split", type=str, required=True, choices=["val", "test"])
    ap.add_argument("--count", type=int, default=256, help="Desired number of entries")
    ap.add_argument("--out", type=str, default="acpl/data/manifests", help="Base output directory")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing JSONL")
    ap.add_argument("--verify", action="store_true", help="Verify after writing and print status")
    args = ap.parse_args(argv)

    cfg = _load_config(args.config)
    # Ensure the generator will materialize exactly 'count' items for the split
    cfg = dict(cfg)
    sc = dict(cfg.get("split_counts", {}))
    sc[args.split] = int(args.count)
    cfg["split_counts"] = sc

    jsonl_path, index_path = make_eval_manifest(
        cfg, split=args.split, count=args.count, out_dir=args.out, overwrite=args.overwrite
    )
    print(jsonl_path)
    print(index_path)

    if args.verify:
        ok = verify_manifest(index_path, args.split, strict=True)
        print(f"verify: {ok}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli(sys.argv[1:]))

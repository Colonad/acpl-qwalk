# acpl/data/manifest.py
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from hashlib import blake2b
import json
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import yaml

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


__all__ = [
    "Manifest",
    "build_manifest",
    "hash_bytes_blake2b",
    "hash_manifest",
    "seed_from_hash",
    "save_manifest",
    "load_manifest",
    "EpisodeCache",
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
    configs: dict[str, object]
    seed: int
    hexdigest: str


# ---------------------------------------------------------------------
# Canonicalization & hashing
# ---------------------------------------------------------------------


def _to_builtin(obj: object) -> object:
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
    # Fallback: as string (stable representation via repr), but discourage exotic objects
    return repr(obj)


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
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


def _manifest_payload_for_hash(version: int, configs: Mapping[str, object], seed: int) -> bytes:
    return _canonical_json_bytes({"version": int(version), "configs": configs, "seed": int(seed)})


def hash_manifest(version: int, configs: Mapping[str, object], seed: int) -> str:
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
    configs: Mapping[str, object],
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
# Tiny in-memory LRU cache for episodes
# ---------------------------------------------------------------------


class EpisodeCache(Generic[K, V]):
    """
    A minimal OrderedDict-backed LRU for episode objects.
    """

    def __init__(self, capacity: int = 32) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._data: OrderedDict[K, V] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __contains__(self, key: K) -> bool:  # pragma: no cover - trivial
        return key in self._data

    def clear(self) -> None:
        self._data.clear()
        self.hits = 0
        self.misses = 0

    def keys(self) -> Iterable[K]:  # pragma: no cover - trivial
        return self._data.keys()

    # ------------- core ops -------------

    def get(self, key: K) -> V | None:
        if key in self._data:
            val = self._data.pop(key)
            self._data[key] = val  # move to end (most-recent)
            self.hits += 1
            return val
        self.misses += 1
        return None

    def put(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def get_or_create(self, key: K, factory: Callable[[K], V]) -> V:
        val = self.get(key)
        if val is not None:
            return val
        new_val = factory(key)
        self.put(key, new_val)
        return new_val

    # ------------- debug/info -------------

    def info(self) -> dict[str, int]:
        return {
            "size": len(self._data),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
        }

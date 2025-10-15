# acpl/data/splits.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import blake2b
import json

import numpy as np

SplitDict = dict[str, np.ndarray]


def _validate_probs(p_train: float, p_val: float, p_test: float) -> tuple[float, float, float]:
    """Validate and (lightly) normalize split probabilities."""
    if any(p < 0.0 for p in (p_train, p_val, p_test)):
        raise ValueError("split probabilities must be non-negative")
    s = p_train + p_val + p_test
    if s <= 0.0:
        raise ValueError("sum of split probabilities must be positive")
    return (p_train / s, p_val / s, p_test / s)


# ---------------------------------------------------------------------------
# Seed-based splitter (Phase A compatibility)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedSplitter:
    """
    Deterministic router from episode seed -> split.

    Determinism comes from hashing the (seed, salt) pair into a stable RNG
    (NumPy PCG64) and drawing a single U[0,1) to decide the split bucket.
    """

    p_train: float = 0.8
    p_val: float = 0.1
    p_test: float = 0.1
    salt: int = 0

    def __post_init__(self) -> None:
        pt, pv, pte = _validate_probs(self.p_train, self.p_val, self.p_test)
        object.__setattr__(self, "p_train", pt)
        object.__setattr__(self, "p_val", pv)
        object.__setattr__(self, "p_test", pte)

    def route(self, seed: int) -> str:
        """Deterministically route a single integer seed to 'train'/'val'/'test'."""
        mix = np.uint64(seed) ^ (np.uint64(self.salt) * np.uint64(0x9E3779B97F4A7C15))
        rng = np.random.Generator(np.random.PCG64(mix))
        u = float(rng.random())
        if u < self.p_train:
            return "train"
        if u < self.p_train + self.p_val:
            return "val"
        return "test"

    def assign_from_seeds(self, seeds: Iterable[int]) -> SplitDict:
        """Assign a collection of integer seeds to splits, deterministically."""
        train, val, test = [], [], []
        for s in seeds:
            bucket = self.route(int(s))
            if bucket == "train":
                train.append(int(s))
            elif bucket == "val":
                val.append(int(s))
            else:
                test.append(int(s))
        return {
            "train": np.asarray(train, dtype=np.int64),
            "val": np.asarray(val, dtype=np.int64),
            "test": np.asarray(test, dtype=np.int64),
        }

    def assign_from_count(self, n: int, *, start_seed: int = 0) -> SplitDict:
        """Assign seeds {start_seed, ..., start_seed+n-1} deterministically."""
        seeds = range(start_seed, start_seed + n)
        return self.assign_from_seeds(seeds)


def default_splitter(
    p_train: float = 0.8,
    p_val: float = 0.1,
    p_test: float = 0.1,
    *,
    salt: int = 0,
) -> SeedSplitter:
    """Factory for a SeedSplitter with given probs and salt."""
    return SeedSplitter(p_train=p_train, p_val=p_val, p_test=p_test, salt=salt)


def route_seed(
    seed: int,
    *,
    p_train: float = 0.8,
    p_val: float = 0.1,
    p_test: float = 0.1,
    salt: int = 0,
) -> str:
    """Stateless helper: route a single seed to 'train' / 'val' / 'test'."""
    return default_splitter(p_train, p_val, p_test, salt=salt).route(seed)


# ---------------------------------------------------------------------------
# Triplet-based splitter (Phase B): (graph, config, task, seed)
# ---------------------------------------------------------------------------


def _stable_json(obj: dict[str, object] | None) -> str:
    """
    Deterministically serialize config dicts to a compact JSON string
    (sorted keys, no whitespace). Accepts None -> "".
    """
    if obj is None:
        return ""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class Triplet:
    """
    Describes an episode identity for splitting.

    family : str
        Graph family name (e.g., "line", "grid", "er", ...).
    graph_cfg : dict[str, object] | None
        Graph parameter dict; must be JSON-serializable with basic types.
    task : str
        Task ID/name (e.g., "transfer", "search", ...).
    seed : int
        Base episode seed (e.g., derived from manifest or index).
    """

    family: str
    graph_cfg: dict[str, object] | None
    task: str
    seed: int


@dataclass(frozen=True)
class TripletSplitter:
    """
    Deterministic router from (graph family, graph config, task, seed) -> split.

    We hash a compact, stable JSON representation of the triplet plus `salt`
    using BLAKE2b (64-bit digest), map to U[0,1), then bucket by probs.
    """

    p_train: float = 0.8
    p_val: float = 0.1
    p_test: float = 0.1
    salt: int = 0

    def __post_init__(self) -> None:
        pt, pv, pte = _validate_probs(self.p_train, self.p_val, self.p_test)
        object.__setattr__(self, "p_train", pt)
        object.__setattr__(self, "p_val", pv)
        object.__setattr__(self, "p_test", pte)

    def _score(
        self, family: str, graph_cfg: dict[str, object] | None, task: str, seed: int
    ) -> float:
        payload = "|".join(
            [
                family.lower(),
                _stable_json(graph_cfg),
                task.lower(),
                str(int(seed)),
                str(int(self.salt)),
            ]
        ).encode("utf-8")
        h = blake2b(payload, digest_size=8)
        # Convert 64-bit digest to float in [0,1)
        val = int.from_bytes(h.digest(), byteorder="big", signed=False)
        return (val % (1 << 64)) / float(1 << 64)

    def route(self, triplet: Triplet) -> str:
        """Route a Triplet to 'train' / 'val' / 'test' deterministically."""
        u = self._score(triplet.family, triplet.graph_cfg, triplet.task, triplet.seed)
        if u < self.p_train:
            return "train"
        if u < self.p_train + self.p_val:
            return "val"
        return "test"

    def assign_from_triplets(self, items: Iterable[Triplet]) -> SplitDict:
        """Partition a collection of Triplet items."""
        train, val, test = [], [], []
        for t in items:
            bucket = self.route(t)
            if bucket == "train":
                train.append(t)
            elif bucket == "val":
                val.append(t)
            else:
                test.append(t)
        # We return indices into the original iteration order when helpful, but for
        # symmetry with SeedSplitter, we return arrays of the base seeds by default.
        # Callers needing the full Triplets can keep their own lists.
        return {
            "train": np.asarray([ti.seed for ti in train], dtype=np.int64),
            "val": np.asarray([ti.seed for ti in val], dtype=np.int64),
            "test": np.asarray([ti.seed for ti in test], dtype=np.int64),
        }


def default_triplet_splitter(
    p_train: float = 0.8,
    p_val: float = 0.1,
    p_test: float = 0.1,
    *,
    salt: int = 0,
) -> TripletSplitter:
    """Factory for a TripletSplitter with given probs and salt."""
    return TripletSplitter(p_train=p_train, p_val=p_val, p_test=p_test, salt=salt)


def route_triplet(
    family: str,
    graph_cfg: dict[str, object] | None,
    task: str,
    seed: int,
    *,
    p_train: float = 0.8,
    p_val: float = 0.1,
    p_test: float = 0.1,
    salt: int = 0,
) -> str:
    """
    Stateless helper to route a (graph, config, task, seed) identity.
    """
    splitter = default_triplet_splitter(p_train, p_val, p_test, salt=salt)
    return splitter.route(Triplet(family=family, graph_cfg=graph_cfg, task=task, seed=seed))

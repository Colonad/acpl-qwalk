# acpl/data/splits.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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


@dataclass(frozen=True)
class SeedSplitter:
    """
    Deterministic router from episode seed -> split.

    Determinism comes from hashing the (seed, salt) pair into a stable RNG
    (NumPy PCG64) and drawing a single U[0,1) to decide the split bucket.

    Parameters
    ----------
    p_train, p_val, p_test : float
        Desired proportions. They are normalized internally.
    salt : int
        Global salt controlling the partition. Change to reshuffle,
        keep fixed to preserve splits across runs.
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

    # ------------------------ core routing ------------------------

    def route(self, seed: int) -> str:
        """
        Deterministically route a single integer seed to 'train' / 'val' / 'test'.
        """
        # Hash (seed, salt) into a 64-bit seed for RNG. Use a simple, strong mix.
        mix = np.uint64(seed) ^ (np.uint64(self.salt) * np.uint64(0x9E3779B97F4A7C15))
        rng = np.random.Generator(np.random.PCG64(mix))
        u = float(rng.random())  # U[0,1)

        if u < self.p_train:
            return "train"
        if u < self.p_train + self.p_val:
            return "val"
        return "test"

    # ------------------------ bulk helpers ------------------------

    def assign_from_seeds(self, seeds: Iterable[int]) -> SplitDict:
        """
        Assign a collection of integer seeds to splits, deterministically.

        Returns
        -------
        dict: {'train': np.ndarray, 'val': np.ndarray, 'test': np.ndarray}
              Arrays contain the input seeds belonging to each split,
              in the original iteration order.
        """
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
        """
        Assign seeds {start_seed, ..., start_seed+n-1} to splits, deterministically.
        """
        seeds = range(start_seed, start_seed + n)
        return self.assign_from_seeds(seeds)


# ------------------------ convenience one-liners ------------------------


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
    """
    Stateless helper: route a single seed to 'train' / 'val' / 'test'.
    """
    return default_splitter(p_train, p_val, p_test, salt=salt).route(seed)

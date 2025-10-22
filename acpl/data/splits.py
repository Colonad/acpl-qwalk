# acpl/data/splits.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any, Literal

import numpy as np

__all__ = [
    "SplitSpec",
    "SplitIndices",
    "stable_hash64",
    "derive_seed64",
    "normalize_ratios",
    "holdout_split_indices",
    "group_holdout_split_indices",
    "stratified_group_holdout_split_indices",
    "kfold_indices",
    "group_kfold_indices",
    "EpisodeRouter",
    # New: Phase B2 triplet router
    "default_phase_b2_rules",
    "route_triplet",
]


# --------------------------------------------------------------------------------------
# Core utilities
# --------------------------------------------------------------------------------------


def stable_hash64(x: str | bytes | int | tuple | Sequence) -> int:
    """
    Deterministic 64-bit hash via SHA-256 → lower 8 bytes interpreted as unsigned.
    Accepts strings, bytes, ints, and (nested) sequences/tuples.
    """

    def _to_bytes(obj) -> bytes:
        if isinstance(obj, bytes):
            return obj
        if isinstance(obj, str):
            return obj.encode("utf-8")
        if isinstance(obj, int):
            return str(obj).encode("utf-8")
        if isinstance(obj, (tuple, list)):
            # include separators to avoid ambiguity
            b = b"["
            for i, el in enumerate(obj):
                if i > 0:
                    b += b","
                b += _to_bytes(el)
            b += b"]"
            return b
        # Fallback to repr for anything else that’s deterministic (e.g., dicts after sorting)
        return repr(obj).encode("utf-8")

    data = _to_bytes(x)
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)


def derive_seed64(*parts: str | int | bytes, base_seed: int | None = None) -> int:
    """
    Compose a 64-bit seed deterministically from arbitrary parts and an optional base_seed.
    """
    seq = list(parts)
    if base_seed is not None:
        seq.append(int(base_seed))
    return stable_hash64(tuple(seq)) & ((1 << 64) - 1)


def normalize_ratios(
    ratios: tuple[float, float, float] | None = None,
    sizes: tuple[int, int, int] | None = None,
    total: int | None = None,
) -> tuple[int, int, int]:
    """
    Convert either fractional ratios (train,val,test) or explicit sizes to integer sizes.
    Ensures non-negative, sum == total, and small rounding is handled gracefully.
    """
    if (ratios is None) == (sizes is None):
        raise ValueError("Provide exactly one of ratios or sizes.")
    if sizes is not None:
        if any(s < 0 for s in sizes):
            raise ValueError("sizes must be non-negative.")
        if total is None:
            total = sum(sizes)
        if sum(sizes) != total:
            raise ValueError("sum(sizes) must equal total.")
        return sizes
    # ratios path
    if total is None:
        raise ValueError("total is required when using ratios.")
    tr, va, te = ratios
    if tr < 0 or va < 0 or te < 0:
        raise ValueError("ratios must be non-negative.")
    s_tr = int(round(tr * total))
    s_va = int(round(va * total))
    # keep the remainder in test to ensure sum matches
    s_te = total - s_tr - s_va
    if s_tr < 0 or s_va < 0 or s_te < 0:
        raise ValueError("Ratios incompatible with total.")
    return (s_tr, s_va, s_te)


# --------------------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSpec:
    """
    Declarative split specification.

    method: "holdout" | "kfold"
    group:  if True, split by groups (all items in the same group go to the same split)
    stratify: if True, attempt to preserve label proportions (requires labels and groups for group-mode)
    k: number of folds (for kfold)
    fold: active fold index in [0, k-1] (for kfold)
    ratios: (train,val,test) for holdout
    seed: master seed for deterministic shuffles/hashes
    """

    method: Literal["holdout", "kfold"] = "holdout"
    group: bool = True
    stratify: bool = False
    k: int = 5
    fold: int = 0
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42
    # streaming holdout behavior
    # When method="holdout" and group=True:
    # - group_pin=False (default): choose split by EPISODE key (better episode-level ratios)
    # - group_pin=True:  choose split by GROUP id (keeps groups intact in streaming)
    group_pin: bool = False


@dataclass(frozen=True)
class SplitIndices:
    """
    Indices for train/val/test.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_lists(self) -> tuple[list[int], list[int], list[int]]:
        return (self.train.tolist(), self.val.tolist(), self.test.tolist())

    def counts(self) -> tuple[int, int, int]:
        return (self.train.size, self.val.size, self.test.size)


# --------------------------------------------------------------------------------------
# Holdout splitting
# --------------------------------------------------------------------------------------


def holdout_split_indices(
    n_items: int,
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> SplitIndices:
    """
    Deterministic shuffle + split over item indices [0..n_items-1].
    """
    if n_items < 0:
        raise ValueError("n_items must be non-negative.")
    sizes = normalize_ratios(ratios=ratios, total=n_items)
    gen = np.random.default_rng(derive_seed64("holdout", seed))
    perm = np.arange(n_items, dtype=np.int64)
    gen.shuffle(perm)
    t, v, _ = sizes
    train = perm[:t]
    val = perm[t : t + v]
    test = perm[t + v :]
    return SplitIndices(train=train, val=val, test=test)


def group_holdout_split_indices(
    groups: Sequence[int | str],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> SplitIndices:
    """
    Holdout which assigns *groups* to splits and then expands to items.
    All items with the same group stay in the same split.
    """
    n = len(groups)
    # Map group value -> list of indices
    group_to_ix: dict[int | str, list[int]] = {}
    for i, g in enumerate(groups):
        group_to_ix.setdefault(g, []).append(i)

    uniq_groups = np.array(list(group_to_ix.keys()))
    G = uniq_groups.size
    sizes = normalize_ratios(ratios=ratios, total=G)

    gen = np.random.default_rng(derive_seed64("group_holdout", seed))
    order = np.arange(G, dtype=np.int64)
    gen.shuffle(order)
    t, v, _ = sizes
    g_train = set(uniq_groups[order[:t]].tolist())
    g_val = set(uniq_groups[order[t : t + v]].tolist())
    g_test = set(uniq_groups[order[t + v :]].tolist())

    train, val, test = [], [], []
    for g, idxs in group_to_ix.items():
        if g in g_train:
            train.extend(idxs)
        elif g in g_val:
            val.extend(idxs)
        else:
            test.extend(idxs)

    return SplitIndices(
        train=np.array(train, dtype=np.int64),
        val=np.array(val, dtype=np.int64),
        test=np.array(test, dtype=np.int64),
    )


def stratified_group_holdout_split_indices(
    labels: Sequence[int | str],
    groups: Sequence[int | str],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> SplitIndices:
    """
    Group-aware *and* label-stratified holdout.

    Strategy:
      1) Aggregate by (group -> label histogram).
      2) Shuffle groups deterministically.
      3) Greedy bin-packing of groups into train/val/test to match target counts per label.
         (Keeps groups intact. Deterministic and fast; approximate but effective.)
    """
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length.")
    n = len(labels)
    # Enumerate labels to ints for histogram math
    label_to_int: dict[int | str, int] = {}
    y_int = np.empty(n, dtype=np.int64)
    next_id = 0
    for i, y in enumerate(labels):
        if y not in label_to_int:
            label_to_int[y] = next_id
            next_id += 1
        y_int[i] = label_to_int[y]
    L = next_id

    # Build group histograms
    group_to_ix: dict[int | str, list[int]] = {}
    for i, g in enumerate(groups):
        group_to_ix.setdefault(g, []).append(i)

    uniq_groups = np.array(list(group_to_ix.keys()))
    G = uniq_groups.size

    # Target totals per label per split
    total_per_label = np.bincount(y_int, minlength=L).astype(np.int64)
    tr, va, te = ratios
    target_tr = (total_per_label * tr).round().astype(np.int64)
    target_va = (total_per_label * va).round().astype(np.int64)
    target_te = total_per_label - target_tr - target_va

    # Prepare greedy accumulators
    acc_tr = np.zeros(L, dtype=np.int64)
    acc_va = np.zeros(L, dtype=np.int64)
    acc_te = np.zeros(L, dtype=np.int64)

    # Group order
    gen = np.random.default_rng(derive_seed64("strat_group_holdout", seed))
    order = np.arange(G, dtype=np.int64)
    gen.shuffle(order)

    g_train, g_val, g_test = [], [], []
    # Greedy: assign each group to the split that minimizes L1 gap to target after adding this group
    for gi in uniq_groups[order]:
        idxs = group_to_ix[gi]
        hist = np.bincount(y_int[idxs], minlength=L)

        def score(acc, targ):
            after = acc + hist
            return np.abs(after - targ).sum()

        s_tr = score(acc_tr, target_tr)
        s_va = score(acc_va, target_va)
        s_te = score(acc_te, target_te)

        # choose smallest score; tie-break train > val > test deterministically
        if s_tr <= s_va and s_tr <= s_te:
            acc_tr += hist
            g_train.append(gi)
        elif s_va <= s_te:
            acc_va += hist
            g_val.append(gi)
        else:
            acc_te += hist
            g_test.append(gi)

    # Expand to item indices
    train, val, test = [], [], []
    for g, idxs in group_to_ix.items():
        if g in g_train:
            train.extend(idxs)
        elif g in g_val:
            val.extend(idxs)
        else:
            test.extend(idxs)

    return SplitIndices(
        train=np.array(train, dtype=np.int64),
        val=np.array(val, dtype=np.int64),
        test=np.array(test, dtype=np.int64),
    )


# --------------------------------------------------------------------------------------
# K-fold cross-validation
# --------------------------------------------------------------------------------------


def kfold_indices(
    n_items: int,
    *,
    k: int,
    fold: int,
    seed: int = 0,
) -> SplitIndices:
    """
    Standard K-fold over items (no grouping). Fold acts as validation; test is empty.
    """
    if k <= 1:
        raise ValueError("k must be >= 2.")
    if not (0 <= fold < k):
        raise ValueError("fold must be in [0, k-1].")
    gen = np.random.default_rng(derive_seed64("kfold", seed))
    order = np.arange(n_items, dtype=np.int64)
    gen.shuffle(order)
    # split order into k blocks as equal as possible
    sizes = [n_items // k + (1 if i < (n_items % k) else 0) for i in range(k)]
    offsets = np.cumsum([0] + sizes)
    val_block = (offsets[fold], offsets[fold + 1])
    val = order[val_block[0] : val_block[1]]
    train = np.concatenate([order[: val_block[0]], order[val_block[1] :]], axis=0)
    test = np.empty(0, dtype=np.int64)
    return SplitIndices(train=train, val=val, test=test)


def group_kfold_indices(
    groups: Sequence[int | str],
    *,
    k: int,
    fold: int,
    seed: int = 0,
) -> SplitIndices:
    """
    GroupKFold: split by unique groups into k folds; validation is one fold; test empty.
    """
    if k <= 1:
        raise ValueError("k must be >= 2.")
    if not (0 <= fold < k):
        raise ValueError("fold must be in [0, k-1].")

    group_to_ix: dict[int | str, list[int]] = {}
    for i, g in enumerate(groups):
        group_to_ix.setdefault(g, []).append(i)

    uniq_groups = np.array(list(group_to_ix.keys()))
    G = uniq_groups.size

    gen = np.random.default_rng(derive_seed64("group_kfold", seed))
    order = np.arange(G, dtype=np.int64)
    gen.shuffle(order)

    sizes = [G // k + (1 if i < (G % k) else 0) for i in range(k)]
    offsets = np.cumsum([0] + sizes)

    val_groups = set(uniq_groups[order[offsets[fold] : offsets[fold + 1]]].tolist())
    train, val = [], []
    for g, idxs in group_to_ix.items():
        if g in val_groups:
            val.extend(idxs)
        else:
            train.extend(idxs)
    test = np.empty(0, dtype=np.int64)
    return SplitIndices(
        train=np.array(train, dtype=np.int64), val=np.array(val, dtype=np.int64), test=test
    )


# --------------------------------------------------------------------------------------
# Router for episodic generators
# --------------------------------------------------------------------------------------


class EpisodeRouter:
    """
    Centralized, deterministic router for episodic data.

    Typical usage:

        router = EpisodeRouter(
            spec=SplitSpec(method="holdout", group=True, stratify=True,
                           ratios=(0.8, 0.1, 0.1), seed=13)
        )

        # For each episode 'i', with metadata:
        #   key   : unique identifier for the episode (str) (e.g., JSON of cfg)
        #   group : group id to avoid leakage (e.g., graph topology hash)
        #   label : optional label for stratification (task type)
        split, ep_seed = router.route(key=..., group=..., label=..., index=i)

    The router provides:
      - split membership ("train"/"val"/"test"),
      - per-episode 64-bit seed derived from master seed + episode key/index.
    """

    def __init__(self, spec: SplitSpec):
        self.spec = spec

    # ---- split assignment helpers -------------------------------------------------- #

    def _assign_holdout(
        self,
        n_items: int,
        *,
        groups: Sequence[int | str] | None = None,
        labels: Sequence[int | str] | None = None,
    ) -> SplitIndices:
        s = self.spec
        if s.group:
            if s.stratify:
                if labels is None or groups is None:
                    raise ValueError("labels and groups are required for stratified group holdout.")
                return stratified_group_holdout_split_indices(
                    labels, groups, ratios=s.ratios, seed=s.seed
                )
            else:
                if groups is None:
                    raise ValueError("groups are required for group holdout.")
                return group_holdout_split_indices(groups, ratios=s.ratios, seed=s.seed)
        else:
            # item-level split (no groups)
            return holdout_split_indices(n_items, ratios=s.ratios, seed=s.seed)

    def _assign_kfold(
        self,
        n_items: int,
        *,
        groups: Sequence[int | str] | None = None,
    ) -> SplitIndices:
        s = self.spec
        if s.group:
            if groups is None:
                raise ValueError("groups are required for group k-fold.")
            return group_kfold_indices(groups, k=s.k, fold=s.fold, seed=s.seed)
        else:
            return kfold_indices(n_items, k=s.k, fold=s.fold, seed=s.seed)

    # ---- public API ---------------------------------------------------------------- #

    def build_splits(
        self,
        n_items: int,
        *,
        groups: Sequence[int | str] | None = None,
        labels: Sequence[int | str] | None = None,
    ) -> SplitIndices:
        """
        Materialize train/val/test indices for a static collection of n_items.
        """
        if self.spec.method == "holdout":
            return self._assign_holdout(n_items, groups=groups, labels=labels)
        elif self.spec.method == "kfold":
            return self._assign_kfold(n_items, groups=groups)
        else:
            raise ValueError(f"Unknown split method: {self.spec.method}")

    def route(
        self,
        *,
        key: str | int | bytes,
        index: int,
        group: int | str | None = None,
        label: int | str | None = None,
    ) -> tuple[str, int]:
        s = self.spec
        key_hash = stable_hash64(("route", key))

        if s.method == "kfold":
            # Group-aware: keep groups intact for kfold
            if s.group and group is not None:
                h = stable_hash64(("gkfold", group, s.seed)) % s.k
            else:
                h = stable_hash64(("kfold", key_hash, s.seed)) % s.k
            split = "val" if h == s.fold else "train"
            ep_seed = derive_seed64("episode", key_hash, index, base_seed=s.seed)
            return split, ep_seed

        # ---- Holdout (streaming) ----
        # For online episodes we target *episode-level* ratios. Even if group=True,
        # we choose the split based on the episode key (not the group) to avoid
        # coarse group-level variance that can skew ratios with few groups.
        # ---- Holdout (streaming) ----
        tr, va, te = s.ratios
        total = tr + va + te
        if total <= 0:
            raise ValueError("Invalid ratios: sum must be positive.")

        # Choose the hash base depending on group_pin behavior
        if s.group and s.group_pin and group is not None:
            # Group-pinned: every episode of this group goes to the same split.
            # Strong mixing to avoid structured-ID skews; map to full [0,1) via 64-bit float.
            h = stable_hash64(("group_holdout_router_v2", s.seed, group))
            # 64-bit multiplicative hashing (Knuth/pcg-style) for additional scrambling
            h = (h * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
            # Convert to uniform in [0,1)
            u = h / float(1 << 64)
        else:
            # Episode-mixed: split is chosen by episode key for better episode-level ratios.
            h = stable_hash64(("item_holdout_router_v2", s.seed, key_hash))
            h = (h * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
            u = h / float(1 << 64)

        c1 = tr / total
        c2 = (tr + va) / total
        if u < c1:
            split = "train"
        elif u < c2:
            split = "val"
        else:
            split = "test"

        ep_seed = derive_seed64("episode", key_hash, index, base_seed=s.seed)
        return split, ep_seed


# --------------------------------------------------------------------------------------
# Debug and invariants
# --------------------------------------------------------------------------------------


def _check_disjoint_and_cover(splits: SplitIndices, n_items: int) -> None:
    """
    Internal: ensure disjointness and coverage of splits over [0..n_items-1].
    """
    all_ix = np.concatenate([splits.train, splits.val, splits.test], axis=0)
    uniq = np.unique(all_ix)
    if uniq.size != n_items:
        missing = set(range(n_items)) - set(uniq.tolist())
        dup = [x for x in all_ix.tolist() if (all_ix == x).sum() > 1]
        raise AssertionError(
            f"Splits must cover all items exactly once. Missing={len(missing)}, dups={len(dup)}."
        )


# ======================================================================================
# Phase B2: triplet router (family / size_kv / task) → split, seed  (NEW)
# ======================================================================================


def default_phase_b2_rules() -> dict[str, Any]:
    """
    Default routing rules for Phase B2.

    Structure (kept intentionally simple & stable):
      {
        "ratios": (train, val, test),                # global default
        "overrides": {
            "task": { "search": (0.75, 0.15, 0.10) } # optional task-specific ratios
            # You can add family-specific or size-bin overrides in future if needed:
            # "family": {"grid": (0.8, 0.1, 0.1)},
        }
      }
    """
    return {
        "ratios": (0.80, 0.10, 0.10),
        "overrides": {
            "task": {
                # Slightly more validation for search to stabilize hyperparam sweeps
                "search": (0.75, 0.15, 0.10),
            }
        },
    }


def _select_ratios(
    family: str, task: str, size_kv: Mapping[str, int], rules: Mapping[str, Any]
) -> tuple[float, float, float]:
    # Start with global ratios
    tr, va, te = rules.get("ratios", (0.8, 0.1, 0.1))
    ov = rules.get("overrides", {})
    # Task override (if any)
    task_tbl = ov.get("task", {})
    if isinstance(task_tbl, Mapping) and task in task_tbl:
        tr, va, te = task_tbl[task]
    # Future hooks: family/size-bin specific overrides can be added here.
    s = tr + va + te
    if s <= 0:
        raise ValueError("Routing ratios must sum to a positive value.")
    return float(tr), float(va), float(te)


def _hash_to_unit_interval(*parts: Any) -> float:
    """
    Mix arbitrary parts into a uniform float in [0,1).
    """
    # Normalize size_kv / dicts to a sorted tuple of pairs for deterministic hashing
    norm_parts: list[Any] = []
    for p in parts:
        if isinstance(p, dict):
            norm_parts.append(tuple(sorted((str(k), int(v)) for k, v in p.items())))
        else:
            norm_parts.append(p)
    h = stable_hash64(tuple(norm_parts))
    # Extra multiplicative mixing (Knuth constant used elsewhere for consistency)
    h = (h * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    return h / float(1 << 64)


def route_triplet(
    *,
    family: str,
    size_kv: Mapping[str, int],
    task: str,
    base_seed: int,
    episode_id: int,
    rules: Mapping[str, Any] | None = None,
    master_seed: int = 0,
) -> tuple[str, int]:
    """
    Deterministically assign split and per-episode seed from (family, size_kv, task) + seeds.

    Inputs
    ------
    family : canonical graph family name (e.g., "grid", "line", "hypercube", "er")
    size_kv: minimal size descriptor (e.g., {"L": 64} or {"N": 128} or {"n": 7})
    task   : task name ("transfer", "search", "mixing", "robust", ...)
    base_seed  : canonical stream seed for this global index (from generator)
    episode_id : global episode index (non-negative)
    rules  : routing rules (see default_phase_b2_rules())
    master_seed: master knob (lets a config switch all routing deterministically)

    Returns
    -------
    split : "train" | "val" | "test"
    ep_seed : 64-bit per-episode seed derived from all routing context
    """
    fam = str(family).lower()
    tname = str(task).lower()
    skv = {str(k): int(v) for k, v in size_kv.items()} if size_kv else {}

    rr = rules or default_phase_b2_rules()
    tr, va, te = _select_ratios(fam, tname, skv, rr)
    s = tr + va + te
    c1 = tr / s
    c2 = (tr + va) / s

    # Hash to [0,1)
    u = _hash_to_unit_interval(
        "triplet_router_v1", int(master_seed), int(base_seed), fam, skv, tname, int(episode_id)
    )

    if u < c1:
        split = "train"
    elif u < c2:
        split = "val"
    else:
        split = "test"

    # Episode seed: include everything that defines the episode identity from the router’s POV
    ep_seed = derive_seed64(
        "router_ep_seed_v1",
        int(master_seed),
        int(base_seed),
        fam,
        tuple(sorted(skv.items())),
        tname,
        int(episode_id),
    )
    return split, int(ep_seed)

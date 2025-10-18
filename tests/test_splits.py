# tests/test_splits.py
import math
import numpy as np
import pytest

from acpl.data.splits import (
    SplitSpec,
    holdout_split_indices,
    group_holdout_split_indices,
    stratified_group_holdout_split_indices,
    kfold_indices,
    group_kfold_indices,
    EpisodeRouter,
)


def _assert_disjoint_and_cover(train, val, test, n):
    all_idx = np.concatenate([train, val, test], axis=0)
    assert np.unique(all_idx).size == n
    assert all_idx.size == n
    # pairwise disjoint
    assert set(train).isdisjoint(set(val))
    assert set(train).isdisjoint(set(test))
    assert set(val).isdisjoint(set(test))


def test_holdout_disjoint_cover_and_determinism():
    n = 101
    ratios = (0.7, 0.2, 0.1)
    A = holdout_split_indices(n_items=n, ratios=ratios, seed=123)
    B = holdout_split_indices(n_items=n, ratios=ratios, seed=123)

    # Disjointness + coverage
    _assert_disjoint_and_cover(A.train, A.val, A.test, n)

    # Determinism for same seed
    assert np.array_equal(A.train, B.train)
    assert np.array_equal(A.val, B.val)
    assert np.array_equal(A.test, B.test)

    # Sizes add up
    assert A.train.size + A.val.size + A.test.size == n


def test_group_holdout_keeps_groups_intact():
    # 20 groups with uneven sizes mapped to 200 items
    rng = np.random.default_rng(7)
    G = 20
    group_ids = np.arange(G)
    # draw group sizes between 2 and 15 (sum <= 200 target)
    sizes = rng.integers(2, 15, size=G)
    sizes[-1] = max(1, 200 - sizes[:-1].sum())
    items = []
    groups = []
    for g, s in zip(group_ids, sizes):
        start = len(items)
        items.extend(range(start, start + s))
        groups.extend([int(g)] * s)

    splits = group_holdout_split_indices(groups, ratios=(0.6, 0.2, 0.2), seed=99)
    _assert_disjoint_and_cover(splits.train, splits.val, splits.test, len(items))

    # Each group must be entirely in a single split
    idx_to_split = {}
    for i in splits.train:
        idx_to_split[int(i)] = "train"
    for i in splits.val:
        idx_to_split[int(i)] = "val"
    for i in splits.test:
        idx_to_split[int(i)] = "test"

    for g in group_ids:
        split_set = {idx_to_split[i] for i, gg in enumerate(groups) if gg == g}
        assert len(split_set) == 1


def test_stratified_group_holdout_balances_labels():
    # Build 30 groups; each group has 5 items with labels skewed but known
    rng = np.random.default_rng(1234)
    G = 30
    items_per_group = 5
    labels = []
    groups = []
    for g in range(G):
        # Assign a dominant label per group to make stratification nontrivial
        dominant = int(rng.integers(0, 3))
        for k in range(items_per_group):
            lbl = dominant if k < 4 else int(rng.integers(0, 3))
            labels.append(lbl)
            groups.append(g)

    splits = stratified_group_holdout_split_indices(labels, groups, ratios=(0.7, 0.15, 0.15), seed=5)
    _assert_disjoint_and_cover(splits.train, splits.val, splits.test, G * items_per_group)

    # Check label proportions are close to global proportions (within tolerance)
    y = np.array(labels)
    total_counts = np.bincount(y, minlength=3).astype(float)
    total_props = total_counts / total_counts.sum()

    def props(ix):
        c = np.bincount(y[ix], minlength=3).astype(float)
        c[c.sum() == 0] = 1.0  # avoid divide-by-zero if empty (shouldn't happen here)
        return c / c.sum()

    tol = 0.10  # 10% absolute tolerance
    for split_ix in [splits.train, splits.val, splits.test]:
        p = props(split_ix)
        assert np.all(np.abs(p - total_props) <= tol)


def test_kfold_semantics_and_coverage():
    n = 37
    k = 6
    seed = 11

    # Union of all folds equals full set; folds are disjoint
    seen_val = set()
    sizes = []
    for f in range(k):
        S = kfold_indices(n_items=n, k=k, fold=f, seed=seed)
        _assert_disjoint_and_cover(S.train, S.val, S.test, n)
        seen_val.update(S.val.tolist())
        sizes.append(S.val.size)

        # train ∪ val = all, test is empty by contract
        assert S.test.size == 0
        assert set(S.train).isdisjoint(set(S.val))
        assert S.train.size + S.val.size == n

    assert len(seen_val) == n  # every item appears as validation once
    # fold sizes differ by at most 1
    assert max(sizes) - min(sizes) <= 1


def test_group_kfold_semantics_and_coverage():
    rng = np.random.default_rng(9)
    G = 17  # groups
    # make 120 items distributed across groups
    group_sizes = rng.integers(3, 10, size=G)
    groups = []
    for g, s in enumerate(group_sizes):
        groups.extend([g] * int(s))
    groups = np.array(groups)

    k = 5
    seed = 23
    seen_val_groups = set()

    for f in range(k):
        S = group_kfold_indices(groups=groups, k=k, fold=f, seed=seed)

        # Items from the same group must not be split across train/val
        def items_in(split):
            return set(groups[split].tolist())

        g_train = items_in(S.train)
        g_val = items_in(S.val)
        assert g_train.isdisjoint(g_val)

        # Coverage: all items assigned, test empty
        _assert_disjoint_and_cover(S.train, S.val, S.test, len(groups))
        seen_val_groups |= g_val

    # Every group becomes validation exactly once across folds
    assert seen_val_groups == set(range(G))


def test_router_holdout_determinism_and_ratios():
    spec = SplitSpec(method="holdout", group=True, stratify=False, ratios=(0.6, 0.2, 0.2), seed=777)
    router = EpisodeRouter(spec)

    # Build 500 episodes with 50 distinct groups; multiple episodes per group
    n_eps = 500
    G = 50
    groups = [f"g{g}" for g in range(G)]
    assigned = {"train": 0, "val": 0, "test": 0}

    # Map of (key,index) → (split, seed) to verify determinism
    seen = {}

    for i in range(n_eps):
        g = groups[i % G]
        key = f"episode-{g}-{i%7}"
        split, ep_seed = router.route(key=key, index=i, group=g)

        # Count
        assigned[split] += 1

        # Determinism: same call yields same answer
        split2, ep_seed2 = router.route(key=key, index=i, group=g)
        assert split == split2 and ep_seed == ep_seed2

        seen[(key, i)] = (split, ep_seed)

        # Seed should change with index even for same key
        split3, ep_seed3 = router.route(key=key, index=i + 1, group=g)
        assert ep_seed3 != ep_seed

    # Roughly match target ratios (loose tolerance since routing is hash-bucketed)
    p_train = assigned["train"] / n_eps
    p_val = assigned["val"] / n_eps
    p_test = assigned["test"] / n_eps

    assert abs(p_train - 0.6) < 0.08
    assert abs(p_val - 0.2) < 0.06
    assert abs(p_test - 0.2) < 0.06


def test_router_kfold_distribution_and_determinism():
    # Grouped kfold routing: expect ~1/k of groups to be 'val'
    spec = SplitSpec(method="kfold", group=True, k=4, fold=1, seed=313)
    router = EpisodeRouter(spec)

    G = 40
    groups = [f"g{g}" for g in range(G)]
    counts = {"train": 0, "val": 0}

    # Route one episode per group (index can be 0)
    for i, g in enumerate(groups):
        split, seed = router.route(key=g, index=0, group=g)
        counts[split] += 1
        # Determinism
        split2, seed2 = router.route(key=g, index=0, group=g)
        assert split == split2 and seed == seed2

    frac_val = counts["val"] / G
    # Allow some variance due to hashing, but should be close to 1/k = 0.25
    assert abs(frac_val - 0.25) < 0.1




# tests/test_router_group_pin.py

from acpl.data.splits import (
    EpisodeRouter,
    SplitSpec,
)


def test_router_holdout_group_pin_keeps_groups_intact():
    # Group-pinned streaming: all episodes of a group map to the same split
    spec = SplitSpec(
        method="holdout", group=True, group_pin=True, ratios=(0.6, 0.2, 0.2), seed=2025
    )
    router = EpisodeRouter(spec)

    G = 60
    n_eps = 600
    groups = [f"g{g}" for g in range(G)]
    # For each group, route multiple episodes with varying keys and indexes
    split_of_group = {}
    assigned = {"train": 0, "val": 0, "test": 0}

    for i in range(n_eps):
        g = groups[i % G]
        key = f"episode-{g}-{i%13}"
        split, ep_seed = router.route(key=key, index=i, group=g)

        # First time we see this group, record its split
        if g not in split_of_group:
            split_of_group[g] = split
        # Subsequent episodes must match the group's split
        assert split_of_group[g] == split

        assigned[split] += 1

        # Determinism for same call
        split2, ep_seed2 = router.route(key=key, index=i, group=g)
        assert split == split2 and ep_seed == ep_seed2

    # Check that about the right FRACTION OF GROUPS land in each split (loose tolerance),
    # since group-pin allocates by groups, not by episodes.
    counts_groups = {"train": 0, "val": 0, "test": 0}
    for g, sp in split_of_group.items():
        counts_groups[sp] += 1

    frac_train_g = counts_groups["train"] / G
    frac_val_g = counts_groups["val"] / G
    frac_test_g = counts_groups["test"] / G

    # Allow larger variance on groups than episodes
    assert abs(frac_train_g - 0.6) < 0.15
    assert abs(frac_val_g - 0.2) < 0.12
    assert abs(frac_test_g - 0.2) < 0.12


def test_router_holdout_episode_mix_ratios_remain_tight():
    # Episode-mixed (default): ratios measured over EPISODES should be tighter
    spec = SplitSpec(
        method="holdout", group=True, group_pin=False, ratios=(0.6, 0.2, 0.2), seed=2025
    )
    router = EpisodeRouter(spec)

    n_eps = 2000
    G = 50
    groups = [f"g{g}" for g in range(G)]
    assigned = {"train": 0, "val": 0, "test": 0}

    for i in range(n_eps):
        g = groups[i % G]
        key = f"episode-{g}-{i%17}"
        split, ep_seed = router.route(key=key, index=i, group=g)
        assigned[split] += 1

        # Determinism
        split2, ep_seed2 = router.route(key=key, index=i, group=g)
        assert split == split2 and ep_seed == ep_seed2

    p_train = assigned["train"] / n_eps
    p_val = assigned["val"] / n_eps
    p_test = assigned["test"] / n_eps

    # Tighter tolerances at episode level
    assert abs(p_train - 0.6) < 0.04
    assert abs(p_val - 0.2) < 0.03
    assert abs(p_test - 0.2) < 0.03

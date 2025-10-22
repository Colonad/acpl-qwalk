# tests/test_dataset_determinism.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import threading
from typing import Any

import numpy as np
import pytest

from acpl.data.generator import EpisodeGenerator, EpisodeNP

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _assert_arrays_equal(a: np.ndarray, b: np.ndarray, *, name: str, rtol=0.0, atol=0.0) -> None:
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), f"{name}: must be numpy arrays"
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} != {b.shape}"
    if np.issubdtype(a.dtype, np.floating):
        np.testing.assert_allclose(
            a, b, rtol=rtol or 1e-7, atol=atol or 1e-8, err_msg=f"{name}: floats differ"
        )
    elif np.issubdtype(a.dtype, np.complexfloating):
        np.testing.assert_allclose(
            a.real,
            b.real,
            rtol=rtol or 1e-7,
            atol=atol or 1e-8,
            err_msg=f"{name}: real part differs",
        )
        np.testing.assert_allclose(
            a.imag,
            b.imag,
            rtol=rtol or 1e-7,
            atol=atol or 1e-8,
            err_msg=f"{name}: imag part differs",
        )
    else:
        assert a.dtype == b.dtype, f"{name}: dtype mismatch {a.dtype} != {b.dtype}"
        assert np.array_equal(a, b), f"{name}: integer/boolean arrays differ"


def _assert_noise_equal(n1: Mapping[str, Any], n2: Mapping[str, Any]) -> None:
    assert set(n1.keys()) == set(n2.keys()), "noise: keys differ"
    for k in n1:
        v1, v2 = n1[k], n2[k]
        assert type(v1) is type(v2), f"noise[{k!r}]: type mismatch {type(v1)} != {type(v2)}"
        if isinstance(v1, Mapping):
            # shallow compare
            assert set(v1.keys()) == set(v2.keys()), f"noise[{k!r}]: nested keys differ"
            for kk in v1:
                assert v1[kk] == v2[kk], f"noise[{k!r}][{kk!r}] mismatch: {v1[kk]} != {v2[kk]}"
        else:
            assert v1 == v2, f"noise[{k!r}] mismatch: {v1} != {v2}"


def _assert_task_equal(t1: Mapping[str, Any], t2: Mapping[str, Any]) -> None:
    assert set(t1.keys()) == set(t2.keys()), "task: keys differ"
    for k in t1:
        assert t1[k] == t2[k], f"task[{k!r}] mismatch: {t1[k]} != {t2[k]}"


def _episode_asdict(ep: EpisodeNP) -> Mapping[str, Any]:
    if is_dataclass(ep):
        return asdict(ep)
    raise TypeError("EpisodeNP must be a dataclass")


def assert_episode_equal(ep1: EpisodeNP, ep2: EpisodeNP, *, where: str) -> None:
    """
    Strong equivalence check across all public EpisodeNP fields.
    Uses exact equality for ints/structure and tolerance for float/complex fields.
    """
    d1, d2 = _episode_asdict(ep1), _episode_asdict(ep2)

    # Basic scalar/meta invariants
    for scalar in ("num_nodes", "num_arcs", "rng_seed", "manifest_hexdigest"):
        assert (
            d1[scalar] == d2[scalar]
        ), f"{where}: scalar '{scalar}' mismatch: {d1[scalar]} != {d2[scalar]}"

    # Graph/CSR
    _assert_arrays_equal(ep1.edge_index, ep2.edge_index, name=f"{where}: edge_index")
    _assert_arrays_equal(ep1.degrees, ep2.degrees, name=f"{where}: degrees")
    _assert_arrays_equal(ep1.coords, ep2.coords, name=f"{where}: coords", rtol=1e-7, atol=1e-8)
    _assert_arrays_equal(ep1.arc_slices, ep2.arc_slices, name=f"{where}: arc_slices")

    # Node features
    _assert_arrays_equal(ep1.X, ep2.X, name=f"{where}: X", rtol=1e-6, atol=1e-7)
    _assert_arrays_equal(
        ep1.features, ep2.features, name=f"{where}: features", rtol=1e-6, atol=1e-7
    )

    # PortMap (portable)
    _assert_arrays_equal(ep1.pm_tail, ep2.pm_tail, name=f"{where}: pm_tail")
    _assert_arrays_equal(ep1.pm_head, ep2.pm_head, name=f"{where}: pm_head")
    _assert_arrays_equal(ep1.pm_rev, ep2.pm_rev, name=f"{where}: pm_rev")
    _assert_arrays_equal(ep1.pm_node_ptr, ep2.pm_node_ptr, name=f"{where}: pm_node_ptr")
    _assert_arrays_equal(ep1.pm_node_arcs, ep2.pm_node_arcs, name=f"{where}: pm_node_arcs")

    # Initial state
    _assert_arrays_equal(ep1.psi0, ep2.psi0, name=f"{where}: psi0", rtol=1e-6, atol=1e-7)
    # ψ0 normalization (guard determinism + physics invariant)
    n1 = float((ep1.psi0.conj() * ep1.psi0).real.sum())
    n2 = float((ep2.psi0.conj() * ep2.psi0).real.sum())
    np.testing.assert_allclose(
        n1, 1.0, rtol=1e-5, atol=1e-6, err_msg=f"{where}: psi0 not normalized (ep1)"
    )
    np.testing.assert_allclose(
        n2, 1.0, rtol=1e-5, atol=1e-6, err_msg=f"{where}: psi0 not normalized (ep2)"
    )

    # Noise + task (fully deterministic under fixed manifest/split/index)
    _assert_noise_equal(ep1.noise, ep2.noise)
    _assert_task_equal(ep1.task, ep2.task)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_manifest_hash_stability_same_config(acpl_config: Mapping[str, Any]) -> None:
    """
    Same config ⇒ identical manifest_hexdigest across fresh EpisodeGenerators.
    """
    g1 = EpisodeGenerator(acpl_config, split="val")
    g2 = EpisodeGenerator(dict(acpl_config), split="val")  # shallow copy should not affect result
    assert (
        g1.manifest_hexdigest == g2.manifest_hexdigest
    ), "manifest_hexdigest must be stable for identical configs"


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("indices", [[0, 1, 7, 13], [3], [9, 10]])
def test_episode_determinism_basic(
    acpl_config: Mapping[str, Any], split: str, indices: Sequence[int]
) -> None:
    """
    Same (config, split, index) ⇒ identical EpisodeNP across distinct generator instances.
    """
    gA = EpisodeGenerator(acpl_config, split=split)
    gB = EpisodeGenerator(dict(acpl_config), split=split)

    for idx in indices:
        epA = gA.episode(idx)
        epB = gB.episode(idx)
        assert_episode_equal(epA, epB, where=f"{split}/idx={idx}")


@pytest.mark.parametrize("split", ["val", "test"])
def test_cache_transparency_and_no_cache_path(acpl_config: Mapping[str, Any], split: str) -> None:
    """
    The internal LRU cache must not affect content determinism.
    - Two cached calls must match.
    - A non-cached call (use_cache=False) must also match cached output.
    """
    g = EpisodeGenerator(acpl_config, split=split)
    idx = 5

    # First call (populates cache)
    ep1 = g.episode(idx, use_cache=True)
    # Second call (cache hit)
    ep2 = g.episode(idx, use_cache=True)
    assert_episode_equal(ep1, ep2, where=f"{split}/cache-hit idx={idx}")

    # Non-cached path should still match
    ep3 = g.episode(idx, use_cache=False)
    assert_episode_equal(ep1, ep3, where=f"{split}/no-cache idx={idx}")


@pytest.mark.parametrize("split", ["val", "test"])
def test_concurrent_access_yields_identical_episodes(
    acpl_config: Mapping[str, Any], split: str
) -> None:
    """
    Concurrent requests for the same (split, index) must produce identical episodes.
    This exercises the generator's determinism + any internal stampede protection in the cache.
    """
    g = EpisodeGenerator(acpl_config, split=split)
    idx = 11
    out: list[EpisodeNP] = []

    def worker():
        out.append(g.episode(idx))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Compare all to the first
    base = out[0]
    for i, ep in enumerate(out[1:], start=1):
        assert_episode_equal(base, ep, where=f"{split}/thread#{i} idx={idx}")


def test_split_changes_distribution_but_preserves_self_determinism(
    acpl_config: Mapping[str, Any],
) -> None:
    """
    Sanity: val vs test episodes at the same index are *not required* to be identical,
    but each split must be deterministic within itself.
    """
    idx = 7
    g_val_1 = EpisodeGenerator(acpl_config, split="val")
    g_val_2 = EpisodeGenerator(acpl_config, split="val")
    g_test_1 = EpisodeGenerator(acpl_config, split="test")
    g_test_2 = EpisodeGenerator(acpl_config, split="test")

    ep_val_1 = g_val_1.episode(idx)
    ep_val_2 = g_val_2.episode(idx)
    ep_test_1 = g_test_1.episode(idx)
    ep_test_2 = g_test_2.episode(idx)

    # Within-split determinism
    assert_episode_equal(ep_val_1, ep_val_2, where=f"val/idx={idx}")
    assert_episode_equal(ep_test_1, ep_test_2, where=f"test/idx={idx}")

    # Cross-split: allow difference, but require basic invariants
    assert (
        ep_val_1.manifest_hexdigest == ep_test_1.manifest_hexdigest
    ), "same config => same manifest hash across splits"
    assert isinstance(ep_val_1.num_nodes, int) and isinstance(ep_test_1.num_nodes, int)
    assert ep_val_1.num_nodes > 0 and ep_test_1.num_nodes > 0


@pytest.mark.parametrize("split", ["val", "test"])
def test_noise_recipe_and_task_are_deterministic(
    acpl_config: Mapping[str, Any], split: str
) -> None:
    """
    Noise recipe sampling (e.g., per-edge phase seed) and task materialization
    must be identical for equal (config, split, index).
    """
    g1 = EpisodeGenerator(acpl_config, split=split)
    g2 = EpisodeGenerator(acpl_config, split=split)

    for idx in (0, 2, 9, 17):
        e1 = g1.episode(idx)
        e2 = g2.episode(idx)

        # Noise dictionaries are expected to match exactly (including any seeds)
        _assert_noise_equal(e1.noise, e2.noise)
        # Task dict matches (including T, source/target/marks/init if present)
        _assert_task_equal(e1.task, e2.task)


@pytest.mark.parametrize("split", ["val", "test"])
def test_structural_fields_and_shapes(acpl_config: Mapping[str, Any], split: str) -> None:
    """
    Additional structural checks that complement determinism:
    - CSR pointer monotonicity
    - pm_rev is a valid index vector
    - edge_index aligns with PortMap tails/heads
    """
    g = EpisodeGenerator(acpl_config, split=split)
    ep = g.episode(3)

    # CSR monotone and ends at A
    ptr = ep.arc_slices
    assert (ptr[:-1] <= ptr[1:]).all(), f"{split}: arc_slices must be non-decreasing"
    assert int(ptr[0]) == 0 and int(ptr[-1]) == ep.num_arcs, f"{split}: CSR ptr bounds invalid"

    # Reverse arc indices are within range
    assert ep.pm_rev.shape == (ep.num_arcs,), "pm_rev length must equal num_arcs"
    assert (
        (0 <= ep.pm_rev) & (ep.pm_rev < ep.num_arcs)
    ).all(), "pm_rev contains out-of-bounds entries"

    # edge_index vs PortMap alignment
    tails = ep.edge_index[0]
    heads = ep.edge_index[1]
    _assert_arrays_equal(tails, ep.pm_tail, name=f"{split}: pm_tail vs edge_index[0]")
    _assert_arrays_equal(heads, ep.pm_head, name=f"{split}: pm_head vs edge_index[1]")

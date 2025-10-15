# tests/test_dataset_determinism.py
from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

from acpl.data.generator import EpisodeGenerator


def _allclose_np(a: np.ndarray, b: np.ndarray, *, rtol=1e-12, atol=1e-12) -> bool:
    if a.dtype.kind in {"i", "u", "b"}:
        return np.array_equal(a, b)
    return np.allclose(a, b, rtol=rtol, atol=atol)


def _allclose_torch(a, b, *, rtol=1e-7, atol=1e-7) -> bool:
    # Match dtype domain: complex vs real allowed; use allclose on real+imag if complex.
    if a.is_complex() or b.is_complex():
        a = a.to(torch.complex64)
        b = b.to(torch.complex64)
    else:
        a = a.to(torch.float32)
        b = b.to(torch.float32)
    return bool(torch.allclose(a, b, rtol=rtol, atol=atol))


def _equal_value(x, y) -> bool:
    # numpy arrays
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if x.shape != y.shape:
            return False
        return _allclose_np(x, y)

    # torch tensors
    if torch is not None and isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        if tuple(x.shape) != tuple(y.shape):
            return False
        return _allclose_torch(x, y)

    # plain python scalars / small tuples
    if isinstance(x, (int, bool, str)) and isinstance(y, (int, bool, str)):
        return x == y
    if isinstance(x, float) and isinstance(y, float):
        return math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)

    # small tuples of numbers
    if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
        return all(_equal_value(xi, yi) for xi, yi in zip(x, y, strict=False))

    # fallback strict equality
    return x == y


def test_episode_determinism_same_config_and_seed_identical():
    """
    Same (config, seed) should produce an identical episode.
    We compare the intersection of keys present in both outputs and
    use dtype-aware equality for arrays/tensors.
    """
    # Minimal, deterministic config: line graph with transfer task.
    gen = EpisodeGenerator(
        family="line",
        num_nodes=16,
        task="transfer",
        task_params={"source": 0, "target": 15},
        feature_kind="line_deg_coord",  # degree + normalized coordinate
        # Any additional defaults inside EpisodeGenerator should make this sufficient.
    )

    seed = 12345
    ep1 = gen.sample(seed=seed)
    ep2 = gen.sample(seed=seed)

    assert isinstance(ep1, dict) and isinstance(ep2, dict)

    # Ensure the basic structural keys exist
    must_have = {
        "edge_index",
        "degrees",
        "coords",
        "arc_slices",
        "pm",
        "psi0",
        "target",
        "X",
    }
    assert must_have.issubset(
        set(ep1.keys())
    ), f"Episode missing keys: {must_have - set(ep1.keys())}"
    assert must_have.issubset(
        set(ep2.keys())
    ), f"Episode missing keys: {must_have - set(ep2.keys())}"

    # Compare common keys safely
    common = set(ep1.keys()) & set(ep2.keys())
    for k in sorted(common):
        v1 = ep1[k]
        v2 = ep2[k]

        # For pm (PortMap-like), compare its array attributes if present
        if k == "pm":
            for attr in ("tail", "head", "rev", "node_ptr", "node_arcs"):
                assert hasattr(v1, attr) and hasattr(v2, attr)
                a1 = getattr(v1, attr)
                a2 = getattr(v2, attr)
                # PortMap arrays are numpy int64; use exact equality
                assert isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray)
                assert np.array_equal(a1, a2), f"pm.{attr} mismatch"
            # also compare simple scalars if exposed
            for attr in ("num_nodes", "num_arcs"):
                if hasattr(v1, attr) and hasattr(v2, attr):
                    assert getattr(v1, attr) == getattr(v2, attr), f"pm.{attr} mismatch"
            continue

        # Generic comparisons
        assert _equal_value(v1, v2), f"Mismatch for key={k!r}"


@pytest.mark.parametrize("different_seed", [7, 9999])
def test_episode_same_config_different_seed_is_stable_not_crashing(different_seed: int):
    """
    Sanity check: different seeds should still generate valid episodes deterministically.
    (We do NOT require them to differ—line graph may be seed-invariant.)
    """
    gen = EpisodeGenerator(
        family="line",
        num_nodes=12,
        task="transfer",
        task_params={"source": 0, "target": 11},
        feature_kind="line_deg_coord",
    )
    ep = gen.sample(seed=different_seed)

    # Basic validity
    assert "edge_index" in ep and isinstance(ep["edge_index"], np.ndarray)
    assert "degrees" in ep and isinstance(ep["degrees"], np.ndarray)
    assert "coords" in ep and isinstance(ep["coords"], np.ndarray)
    assert "pm" in ep
    pm = ep["pm"]
    for attr in ("tail", "head", "rev", "node_ptr", "node_arcs"):
        assert hasattr(pm, attr)
        arr = getattr(pm, attr)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1

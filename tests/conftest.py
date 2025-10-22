# tests/conftest.py
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

import pytest

from acpl.data.generator import EpisodeGenerator

# Project imports
from acpl.data.manifest import (
    make_eval_manifest,
    read_eval_manifest,
    verify_manifest,
)


# ------------------------------------------------------------------------------
# Pytest command-line options (single definition)
# ------------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("acpl", "ACPL reproducible eval options")
    group.addoption(
        "--acpl-config",
        action="append",
        default=[],
        metavar="PATH",
        help="Path(s) to YAML/JSON manifests/config fragments. Repeatable; later files override earlier ones.",
    )
    group.addoption(
        "--acpl-outdir",
        action="store",
        default=os.environ.get("ACPL_OUTDIR", ".acpl-cache"),
        metavar="DIR",
        help="Output/cache directory for generated eval entries (default: .acpl-cache).",
    )
    group.addoption(
        "--acpl-val-count",
        type=int,
        action="store",
        default=int(os.environ.get("ACPL_VAL_COUNT", "128")),
        metavar="N",
        help="How many validation entries to build/ensure (default: 128).",
    )
    group.addoption(
        "--acpl-test-count",
        type=int,
        action="store",
        default=int(os.environ.get("ACPL_TEST_COUNT", "128")),
        metavar="N",
        help="How many test entries to build/ensure (default: 128).",
    )
    group.addoption(
        "--acpl-overwrite",
        action="store_true",
        default=False,
        help="Force-regenerate eval entries even if present.",
    )
    group.addoption(
        "--acpl-verify",
        action="store_true",
        default=False,
        help="Run integrity checks over generated entries before tests.",
    )
    group.addoption(
        "--acpl-fast",
        action="store_true",
        default=False,
        help="Fast mode: smaller counts and minimal feature set for smoke runs.",
    )
    group.addoption(
        "--acpl-skip-build",
        action="store_true",
        default=False,
        help="Skip building eval manifests (assume they already exist).",
    )


# ------------------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------------------


def _default_phase_b2_config(*, fast: bool = False) -> dict[str, Any]:
    """
    Compact yet representative Phase-B2 config that exercises:
      - multiple families (line/grid/hypercube + a small irregular)
      - multiple tasks (transfer/search/mixing/robust)
      - router-compatible size keys (N / L / n)
      - Phase-B horizons (T ∈ {64, 96} via per-split lists)
    """
    val_T = [64] if fast else [64, 96]
    test_T = [96]

    return {
        "family": ["line", "grid", "hypercube", "er"],
        "N": {"train": [64, 96, 128, 160, 192], "val": [96, 160], "test": [256]},
        "L": {"train": [32, 40, 64, 80], "val": [48, 80], "test": [96]},
        "n": {"train": [6, 7, 8, 9], "val": [7, 9], "test": [10]},
        "p": {"train": 0.05, "val": 0.04, "test": 0.03},
        "num_nodes": {"train": 128, "val": 128, "test": 256},
        "task": {
            "train": ["transfer", "mixing", "search", "robust"],
            "val": ["transfer", "search"],
            "test": ["transfer", "search"],
        },
        "task_params": {
            "source": 0,
            "target": 15,
            "marks": [0, 7],
            "noise": {"edge_phases": {"sigma": 0.05}},
        },
        "T": {"train": [64, 96], "val": val_T, "test": test_T},
        "features": {
            "use_degree": True,
            "degree_norm": "inv_sqrt",
            "use_coords": True,
            "use_lap_pe": False,
            "lap_pe_k": 0,
            "seed": 0,
        },
        "split_master_seed": 0,
        "split_counts": {
            "val": 128 if not fast else 64,
            "test": 128 if not fast else 64,
        },
        "num_episodes": 10_000,
    }


def _load_one_config(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except Exception:
        import yaml as _yaml

        return _yaml.safe_load(txt)


def _merge_configs(paths: list[str]) -> dict[str, Any]:
    """
    Shallow left-to-right merge of multiple config files.
    Later files override earlier keys.
    """
    merged: dict[str, Any] = {}
    for path in paths:
        cfg = _load_one_config(path) or {}
        merged.update(cfg)
    return merged


# ------------------------------------------------------------------------------
# Session-scoped fixtures: config, output directory, and materialization
# ------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def acpl_config(pytestconfig: pytest.Config) -> dict[str, Any]:
    """
    Final config used by EpisodeGenerator in tests.
    Merges (optional) user configs over robust defaults.
    """
    fast = bool(pytestconfig.getoption("--acpl-fast"))
    base = _default_phase_b2_config(fast=fast)

    user_paths: list[str] = pytestconfig.getoption("--acpl-config") or []
    user_cfg = _merge_configs(user_paths)

    merged = dict(base)
    merged.update(user_cfg or {})

    sc = dict(merged.get("split_counts", {}))
    sc["val"] = int(pytestconfig.getoption("--acpl-val-count"))
    sc["test"] = int(pytestconfig.getoption("--acpl-test-count"))
    merged["split_counts"] = sc

    return merged


@pytest.fixture(scope="session")
def acpl_manifests_dir(
    pytestconfig: pytest.Config, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """
    Where JSONLs will be written. Defaults to .acpl-cache; can be a tmp dir if desired.
    """
    outdir = pytestconfig.getoption("--acpl-outdir")
    if outdir:
        p = Path(outdir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    return tmp_path_factory.mktemp("acpl_manifests")


@pytest.fixture(scope="session")
def acpl_materialize_eval(
    acpl_config: Mapping[str, Any], acpl_manifests_dir: Path, pytestconfig: pytest.Config
) -> dict[str, str]:
    """
    Build val/test JSONLs (unless --acpl-skip-build), and verify digests.
    Returns dict split -> index.json path. Idempotent by default: if files exist
    and --acpl-overwrite is not set, we skip rebuilding and only verify.
    """
    skip_build = bool(pytestconfig.getoption("--acpl-skip-build"))
    overwrite = bool(pytestconfig.getoption("--acpl-overwrite"))

    # Manifest root directory for this config
    gen_val = EpisodeGenerator(acpl_config, split="val")
    manifest_hex = gen_val.manifest_hexdigest
    base = acpl_manifests_dir / manifest_hex
    base.mkdir(parents=True, exist_ok=True)

    index_paths: dict[str, str] = {}
    idx_path = base / "index.json"  # shared index (tracks both splits)

    for split in ("val", "test"):
        split_jsonl = base / f"{split}.jsonl"

        if not skip_build:
            # If the JSONL already exists and we aren't overwriting, skip cleanly.
            if split_jsonl.exists() and not overwrite:
                # no-op; we'll verify below
                pass
            else:
                count = int(acpl_config.get("split_counts", {}).get(split, 128))
                try:
                    # This may raise FileExistsError if another worker created it
                    # concurrently or we re-run with the same cache; swallow safely.
                    make_eval_manifest(
                        acpl_config,
                        split=split,
                        count=count,
                        out_dir=acpl_manifests_dir,
                        overwrite=overwrite,
                    )
                except FileExistsError:
                    # Idempotent behavior: treat as success and move on to verification
                    pass

        # Always verify whatever is present; raises on mismatch
        verify_manifest(idx_path, split, strict=True)
        index_paths[split] = str(idx_path)

    return index_paths


# Optional autouse verification (no-op unless --acpl-verify was passed)
@pytest.fixture(scope="session", autouse=True)
def _acpl_optional_verify(acpl_materialize_eval: dict[str, str]) -> None:
    return None


# ------------------------------------------------------------------------------
# Split-specific convenience fixtures
# ------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def acpl_val_index_path(acpl_materialize_eval: dict[str, str]) -> str:
    return acpl_materialize_eval["val"]


@pytest.fixture(scope="session")
def acpl_test_index_path(acpl_materialize_eval: dict[str, str]) -> str:
    return acpl_materialize_eval["test"]


@pytest.fixture(scope="session")
def acpl_val_entries(acpl_val_index_path: str) -> list[dict]:
    return read_eval_manifest(acpl_val_index_path, "val")


@pytest.fixture(scope="session")
def acpl_test_entries(acpl_test_index_path: str) -> list[dict]:
    return read_eval_manifest(acpl_test_index_path, "test")


# ------------------------------------------------------------------------------
# Ready-to-use EpisodeGenerators (train/val/test)
# ------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def acpl_train_generator(acpl_config: Mapping[str, Any]) -> EpisodeGenerator:
    return EpisodeGenerator(acpl_config, split="train")


@pytest.fixture(scope="session")
def acpl_val_generator(acpl_config: Mapping[str, Any]) -> EpisodeGenerator:
    return EpisodeGenerator(acpl_config, split="val")


@pytest.fixture(scope="session")
def acpl_test_generator(acpl_config: Mapping[str, Any]) -> EpisodeGenerator:
    return EpisodeGenerator(acpl_config, split="test")


# ------------------------------------------------------------------------------
# Optional smoke: instantiate one episode for val/test (graceful if torch absent)
# ------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def acpl_smoke_episode_val(acpl_val_generator: EpisodeGenerator) -> dict | None:
    try:
        ep = acpl_val_generator.get(0)
        return {"num_nodes": ep.num_nodes, "num_arcs": ep.num_arcs, "seed": ep.rng_seed}
    except Exception:
        return None


@pytest.fixture(scope="session")
def acpl_smoke_episode_test(acpl_test_generator: EpisodeGenerator) -> dict | None:
    try:
        ep = acpl_test_generator.get(0)
        return {"num_nodes": ep.num_nodes, "num_arcs": ep.num_arcs, "seed": ep.rng_seed}
    except Exception:
        return None

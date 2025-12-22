#!/usr/bin/env python3
"""
scripts/gen_manifest.py

Generate deterministic, leak-free evaluation manifests (val/test_iid/test_ood) for ACPL experiments.

Why this exists:
- Novelty-grade results require fixed val/test/OOD episode lists so CI comparisons are meaningful.
- Training can remain procedural/on-the-fly, but evaluation must be frozen.

Supported input schemas:
(A) Phase-B experiment manifest style (recommended):
    experiment:
      name: transfer-line
    repro:
      master_seed: 0
      eval_seeds: [...]
    data:
      family: "line"
      train_sizes: { mode: fixed_list, fixed_list: [...] }  # used for leakage checks
      val_sizes: ...
      test_iid_sizes: ...
      test_ood_sizes: ...
      manifests:
        enabled: true
        root_dir: "acpl/data/manifests"
        val_manifest: "transfer-line-val.json"
        test_iid_manifest: "transfer-line-test-iid.json"
        test_ood_manifest: "transfer-line-test-ood.json"
        episodes: { val: 200, test_iid: 200, test_ood: 300 }
        build:
          seed: 1234
          horizon_policy: "per_horizon_manifest"
          per_horizon: { enabled: true, horizons: [64, 96, 128] }
          episode_seed_base: 900000

(B) Legacy "pre-registration" YAMLs (compat mode), like the one you pasted:
    goal: transfer
    train: {...}
    data:
      family: line
      train_sizes: [...]
      test_ood_sizes: [...]
      seed: 1234
    sim:
      steps: [64, 96, 128]
    log:
      run_name: "transfer-line"

Outputs:
- JSON manifests with episodes, metadata, and fingerprints.
- If per-horizon mode: split manifests into T-specific files + write an index file.

Usage:
  python scripts/gen_manifest.py --config acpl/configs/experiments/transfer-line.yaml
  python scripts/gen_manifest.py --config acpl/configs/experiments/transfer-line.yaml --dry-run --print-examples 3
  python scripts/gen_manifest.py --config path/to/legacy_transfer_line.yaml --overwrite

Notes:
- This script intentionally does NOT generate train manifests by default, because training is typically on-the-fly.
  It will still validate train/val/test leakage using train size lists/bands if provided.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import yaml


# =============================================================================
# Utilities
# =============================================================================

JSONDict = Dict[str, Any]
SplitName = Literal["val", "test_iid", "test_ood"]

SCHEMA_VERSION = "acpl.manifest.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def blake2b_hex(data: bytes, digest_size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(data)
    return h.hexdigest()


def stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deep_get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested value by dotted path, e.g. 'data.manifests.build.seed'."""
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any, *, indent: int = 2, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        obj = {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a mapping/dict (got {type(obj)}).")
    return obj
def find_repo_root(start: Path) -> Path:
    """
    Best-effort repo root discovery.

    We treat the first parent containing any of these markers as the repo root:
      - .git
      - pyproject.toml
      - setup.cfg / setup.py
      - requirements.txt
    """
    markers = [".git", "pyproject.toml", "setup.cfg", "setup.py", "requirements.txt"]
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        for m in markers:
            if (parent / m).exists():
                return parent
    return cur


# =============================================================================
# Episode specifications
# =============================================================================

@dataclass(frozen=True)
class EpisodeSpec:
    """
    Canonical, hashed episode representation.

    This is the "episode key" used for determinism and manifest identity.
    """
    family: str
    graph: JSONDict
    task: JSONDict
    sim: JSONDict
    rng: JSONDict
    disorder: Optional[JSONDict] = None

    def to_canonical_dict(self) -> JSONDict:
        d: JSONDict = {
            "family": self.family,
            "graph": self.graph,
            "task": self.task,
            "sim": self.sim,
            "rng": self.rng,
        }
        if self.disorder is not None:
            d["disorder"] = self.disorder
        return d

    def episode_id(self) -> str:
        # Stable hash of canonical dict
        payload = stable_json_dumps(self.to_canonical_dict()).encode("utf-8")
        return blake2b_hex(payload, digest_size=16)


# =============================================================================
# Parsing experiment configs (Phase-B + Legacy compat)
# =============================================================================

@dataclass
class ManifestBuildSpec:
    seed: int
    episode_seed_base: int
    horizon_policy: Literal["per_horizon_manifest", "graphs_only"]
    horizons: List[int]  # horizons involved when per_horizon_manifest is enabled


@dataclass
class SizeSpec:
    # For line: N values; for grid: L values; for hypercube: n values; etc.
    mode: Literal["fixed_list", "uniform_band"]
    fixed_list: List[int]
    band_min: Optional[int] = None
    band_max: Optional[int] = None


@dataclass
class ManifestPaths:
    root_dir: Path
    val: str
    test_iid: Optional[str]
    test_ood: str


@dataclass
class ManifestCounts:
    val: int
    test_iid: int
    test_ood: int


@dataclass
class ExperimentManifestSpec:
    name: str
    family: str
    data_block: JSONDict
    # sizes for leakage checks + to define episode generation for evaluation
    train_sizes: Optional[SizeSpec]
    val_sizes: Optional[SizeSpec]
    test_iid_sizes: Optional[SizeSpec]
    test_ood_sizes: Optional[SizeSpec]

    # episode generation
    build: ManifestBuildSpec
    paths: ManifestPaths
    counts: ManifestCounts

    # task fields (lightly used here; episode builder uses them in eval)
    task_name: Optional[str]  # transfer/search/mixing/robust if known
    task_block: JSONDict

    # disorder defaults (robust may include this)
    disorder_block: Optional[JSONDict]

    # provenance
    source_config_path: Path
    source_config_sha: str

    # extra metadata to embed
    meta: JSONDict


def _parse_size_spec_from_any(x: Any) -> Optional[SizeSpec]:
    """
    Accepts:
      - None
      - list[int]
      - dict with {mode, fixed_list} or {N_min,N_max} style
      - dict like {mode: fixed_list, fixed_list:[...]}
    """
    if x is None:
        return None
    if isinstance(x, list):
        vals = [int(v) for v in x]
        return SizeSpec(mode="fixed_list", fixed_list=vals)
    if isinstance(x, dict):
        mode = x.get("mode")
        if mode in ("fixed_list", "uniform_band"):
            fixed_list = x.get("fixed_list") or []
            fixed_list = [int(v) for v in fixed_list]
            band_min = x.get("band", {}).get("N_min") if isinstance(x.get("band"), dict) else x.get("N_min")
            band_max = x.get("band", {}).get("N_max") if isinstance(x.get("band"), dict) else x.get("N_max")
            if band_min is not None:
                band_min = int(band_min)
            if band_max is not None:
                band_max = int(band_max)
            return SizeSpec(mode=mode, fixed_list=fixed_list, band_min=band_min, band_max=band_max)
        # Legacy shortcuts
        if "N_min" in x and "N_max" in x:
            return SizeSpec(mode="uniform_band", fixed_list=[], band_min=int(x["N_min"]), band_max=int(x["N_max"]))
        if "fixed_list" in x:
            return SizeSpec(mode="fixed_list", fixed_list=[int(v) for v in x["fixed_list"]])
    # Unknown
    return None


def _coerce_manifest_build(exp: dict[str, Any]) -> ManifestBuildSpec:
    # Phase-B
    seed = deep_get(exp, "data.manifests.build.seed")
    if seed is None:
        # legacy
        seed = deep_get(exp, "data.seed", 1234)
    seed = int(seed)

    episode_seed_base = deep_get(exp, "data.manifests.build.episode_seed_base", 900000)
    episode_seed_base = int(episode_seed_base)

    horizon_policy = deep_get(exp, "data.manifests.build.horizon_policy", None)
    if horizon_policy is None:
        # legacy: per-horizon if sim.steps is list, else graphs_only
        horizon_policy = "per_horizon_manifest"
    if horizon_policy not in ("per_horizon_manifest", "graphs_only"):
        raise ValueError(f"Unsupported horizon_policy={horizon_policy}")

    # horizons list
    horizons = deep_get(exp, "data.manifests.build.per_horizon.horizons")
    if horizons is None:
        # fallback to sim curriculum schedule or sim.steps list
        steps = deep_get(exp, "sim.steps")
        if steps is None:
            steps = deep_get(exp, "sim.curriculum.schedule")
            if isinstance(steps, list) and steps and isinstance(steps[0], dict) and "horizon_T" in steps[0]:
                horizons = [int(x["horizon_T"]) for x in steps]
            else:
                horizons = None
        else:
            horizons = [int(v) for v in steps]
    if horizons is None:
        # final fallback: task.horizon_T if present, else 64
        horizons = [int(deep_get(exp, "task.horizon_T", deep_get(exp, "task.horizon_T", 64)))]
    # sanitize unique, sorted
    horizons = sorted({int(h) for h in horizons})
    return ManifestBuildSpec(
        seed=seed,
        episode_seed_base=episode_seed_base,
        horizon_policy=horizon_policy,  # type: ignore[arg-type]
        horizons=horizons,
    )


def _infer_experiment_name(exp: dict[str, Any], config_path: Path) -> str:
    name = deep_get(exp, "experiment.name")
    if name:
        return str(name)
    # legacy: log.run_name or log.run_name field
    name = deep_get(exp, "log.run_name") or deep_get(exp, "log.run_name", None)
    if name:
        return str(name)
    # legacy: goal + family
    goal = exp.get("goal")
    fam = deep_get(exp, "data.family")
    if goal and fam:
        return f"{goal}-{fam}"
    return config_path.stem


def _infer_family(exp: dict[str, Any]) -> str:
    fam = deep_get(exp, "data.family")
    if isinstance(fam, list):
        # If multiple families, pick the first for manifest generation unless explicitly overridden.
        # For multi-family experiments (mixing), manifests should encode family per-episode; we support that
        # by expanding across families if list.
        return "MULTI"
    if fam is None:
        # legacy
        fam = deep_get(exp, "data.family", "line")
    return str(fam)


def _infer_task_name(exp: dict[str, Any]) -> Optional[str]:
    # Phase-B often has task.name under "task.name" or "task: {name: ...}"
    tn = deep_get(exp, "task.name")
    if tn:
        return str(tn)
    
    # common schema: task: { name: ... }
    t = exp.get("task")
    if isinstance(t, dict) and t.get("name"):
        return str(t["name"])
    
    
    # legacy
    goal = exp.get("goal")
    if goal:
        return str(goal)
    

    # fallback: infer from experiment name
    en = deep_get(exp, "experiment.name") or deep_get(exp, "log.run_name") or ""
    if isinstance(en, str):
        low = en.lower()
        for k in ("transfer", "search", "mixing", "robust"):
            if k in low:
                return k



    return None


def _default_manifest_paths(exp_name: str, root_dir: Path) -> ManifestPaths:
    return ManifestPaths(
        root_dir=root_dir,
        val=f"{exp_name}-val.json",
        test_iid=f"{exp_name}-test-iid.json",
        test_ood=f"{exp_name}-test-ood.json",
    )


def _parse_manifest_paths(exp: dict[str, Any], exp_name: str, config_path: Path) -> ManifestPaths:
    root = deep_get(exp, "data.manifests.root_dir", None)
    if root is None:
        # legacy default root
        root = "acpl/data/manifests"

    repo_root = find_repo_root(config_path.parent)

    root_dir = Path(root)
    if not root_dir.is_absolute():
        # IMPORTANT: resolve relative to REPO ROOT, not the config directory
        root_dir = (repo_root / root_dir).resolve()
    else:
        root_dir = root_dir.resolve()

    val = deep_get(exp, "data.manifests.val_manifest", None)
    test_iid = deep_get(exp, "data.manifests.test_iid_manifest", None)
    test_ood = deep_get(exp, "data.manifests.test_ood_manifest", None)

    if val is None or test_ood is None:
        default = _default_manifest_paths(exp_name, root_dir)
        val = val or default.val
        test_iid = test_iid or default.test_iid
        test_ood = test_ood or default.test_ood

    # Allow disabling iid test
    if test_iid is None:
        test_iid = None

    return ManifestPaths(
        root_dir=root_dir,
        val=str(val),
        test_iid=str(test_iid) if test_iid else None,
        test_ood=str(test_ood),
    )


def _parse_manifest_counts(exp: dict[str, Any]) -> ManifestCounts:
    # Phase-B
    episodes = deep_get(exp, "data.manifests.episodes", None)
    if isinstance(episodes, dict):
        val = int(episodes.get("val", 200))
        test_iid = int(episodes.get("test_iid", episodes.get("test", 200)))
        test_ood = int(episodes.get("test_ood", 300))
        return ManifestCounts(val=val, test_iid=test_iid, test_ood=test_ood)

    # legacy: train.ci_episodes used for eval; pick defaults if missing
    ci_eps = deep_get(exp, "train.ci_episodes", 100)
    ci_eps = int(ci_eps)
    return ManifestCounts(val=max(ci_eps, 100), test_iid=max(ci_eps, 100), test_ood=max(ci_eps, 150))


def _parse_sizes(exp: dict[str, Any], key: str) -> Optional[SizeSpec]:
    # Phase-B uses e.g. data.val_sizes, data.test_ood_sizes etc.
    x = deep_get(exp, f"data.{key}", None)
    if x is None:
        # legacy keys: train_sizes, test_ood_sizes
        if key == "train_sizes":
            x = deep_get(exp, "data.train_sizes", None)
        elif key == "test_ood_sizes":
            x = deep_get(exp, "data.test_ood_sizes", None)
        elif key == "val_sizes":
            x = deep_get(exp, "data.val_sizes", None)  # may not exist
        elif key == "test_iid_sizes":
            x = deep_get(exp, "data.test_iid_sizes", None)
    return _parse_size_spec_from_any(x)


def parse_experiment_manifest(config_path: Path) -> ExperimentManifestSpec:
    exp = read_yaml(config_path)

    # provenance
    raw = config_path.read_bytes()
    sha = blake2b_hex(raw, digest_size=16)

    exp_name = _infer_experiment_name(exp, config_path)
    family = _infer_family(exp)
    task_name = _infer_task_name(exp)

    build = _coerce_manifest_build(exp)
    paths = _parse_manifest_paths(exp, exp_name, config_path)
    counts = _parse_manifest_counts(exp)

    train_sizes = _parse_sizes(exp, "train_sizes")
    val_sizes = _parse_sizes(exp, "val_sizes")
    test_iid_sizes = _parse_sizes(exp, "test_iid_sizes")
    test_ood_sizes = _parse_sizes(exp, "test_ood_sizes")

    # task block: prefer exp["task"], else legacy exp["task"] exists anyway
    task_block = exp.get("task", {})
    if not isinstance(task_block, dict):
        task_block = {}


    
    data_block = exp.get("data", {})
    if not isinstance(data_block, dict):
        data_block = {}
    # legacy puts task params at top-level "task:" too; ok.

    disorder_block = exp.get("disorder")
    if disorder_block is not None and not isinstance(disorder_block, dict):
        disorder_block = None

    # embed extra metadata
    meta: JSONDict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "source_config_path": str(config_path),
        "source_config_sha16": sha,
        "experiment_name": exp_name,
        "family": family,
        "task_name": task_name,
        "build": dataclasses.asdict(build),
    }

    return ExperimentManifestSpec(
        name=exp_name,
        family=family,
        data_block=data_block,
        train_sizes=train_sizes,
        val_sizes=val_sizes,
        test_iid_sizes=test_iid_sizes,
        test_ood_sizes=test_ood_sizes,
        build=build,
        paths=paths,
        counts=counts,
        task_name=task_name,
        task_block=task_block,
        disorder_block=disorder_block,
        source_config_path=config_path,
        source_config_sha=sha,
        meta=meta,
    )


# =============================================================================
# Episode generation per family
# =============================================================================



def _deterministic_order(values: Sequence[Any], *, salt: JSONDict) -> List[Any]:
    """
    Deterministically shuffle `values` given a salt dict.
    (Used to avoid always starting cycling from the same size.)
    """
    salt_s = stable_json_dumps(salt)
    keyed: List[Tuple[str, Any]] = []
    for v in values:
        h = blake2b_hex(
            stable_json_dumps({"salt": salt_s, "v": v}).encode("utf-8"),
            digest_size=8,
        )
        keyed.append((h, v))
    keyed.sort(key=lambda t: t[0])
    return [v for _, v in keyed]


def _cycle_pick(values: Sequence[int], n: int, *, salt: JSONDict) -> List[int]:
    """
    Return a length-n list by cycling through deterministically ordered values.
    """
    vals = [int(v) for v in values]
    if not vals or n <= 0:
        return []
    ordered = _deterministic_order(vals, salt=salt)
    return [ordered[i % len(ordered)] for i in range(n)]




def _expand_sizes(spec: Optional[SizeSpec]) -> List[int]:
    if spec is None:
        return []
    if spec.mode == "fixed_list":
        return list(spec.fixed_list)
    if spec.mode == "uniform_band":
        if spec.band_min is None or spec.band_max is None:
            return []
        # for manifests we prefer explicit sizes, but if band specified:
        # choose a small representative set (min, quartiles, max)
        mn, mx = spec.band_min, spec.band_max
        if mn == mx:
            return [mn]
        q1 = mn + (mx - mn) // 4
        q2 = mn + (mx - mn) // 2
        q3 = mn + 3 * (mx - mn) // 4
        return sorted({mn, q1, q2, q3, mx})
    return []


def _resolve_target_index(task_block: JSONDict, N: int) -> Optional[int]:
    # Supports -1 => last node convention
    ti = task_block.get("target_index")
    if ti is None:
        # Phase-B transfer task block may use task: { target_index: ... }
        ti = deep_get({"task": task_block}, "task.target_index", None)
    if ti is None:
        return None
    ti = int(ti)
    if ti < 0:
        ti = N + ti
    if ti < 0 or ti >= N:
        return None
    return ti


def _resolve_source_index(task_block: JSONDict, N: int) -> Optional[int]:
    si = task_block.get("source_index")
    if si is None:
        si = deep_get({"task": task_block}, "task.source_index", None)
    if si is None:
        return None
    si = int(si)
    if si < 0:
        si = N + si
    if si < 0 or si >= N:
        return None
    return si


def generate_episodes_line(
    *,
    split: SplitName,
    sizes: Sequence[int],
    horizon_T: int,
    n_episodes: int,
    task_block: JSONDict,
    build: ManifestBuildSpec,
    disorder_block: Optional[JSONDict] = None,   # <-- ADD THIS
) -> List[EpisodeSpec]:

    """
    Generate EpisodeSpec list for line graphs.

    Episode seeds are deterministic:
      episode_seed = episode_seed_base + hash(split, horizon_T, N, idx, build.seed)
    """
    episodes: List[EpisodeSpec] = []
    if not sizes:
        return episodes

    # Deterministic mapping from (split,horizon,episode_index) -> (N, seed)
    picked_sizes = _cycle_pick(
        [int(x) for x in sizes],
        int(n_episodes),
        salt={"family": "line", "split": split, "horizon_T": int(horizon_T), "build_seed": int(build.seed)},
    )
    for i, N in enumerate(picked_sizes):
        src = _resolve_source_index(task_block, N)
        tgt = _resolve_target_index(task_block, N)
        # We keep indices in the episode (even if None) to ensure downstream builder can use them.
        task_payload: JSONDict = {
            "name": task_block.get("name", task_block.get("task", {}).get("name", "transfer")),
            "source_index": src if src is not None else task_block.get("source_index", 0),
            "target_index": tgt if tgt is not None else task_block.get("target_index", N - 1),
        }
        # Copy over loss-ish fields if present (helps downstream debug; doesn’t affect hash unless included)
        for k in ("loss", "time_agg", "tau", "eps", "label_smoothing", "cvar_alpha", "margin", "reduce", "normalize_prob"):
            if k in task_block:
                task_payload[k] = task_block[k]

        seed_material = {
            "split": split,
            "horizon_T": int(horizon_T),
            "family": "line",
            "N": int(N),
            "i": int(i),
            "build_seed": int(build.seed),
        }
        seed_hex = blake2b_hex(stable_json_dumps(seed_material).encode("utf-8"), digest_size=8)
        seed_int = int(seed_hex, 16)
        episode_seed = (int(build.episode_seed_base) + seed_int) % (2**64)

        ep = EpisodeSpec(
            family="line",
            graph={"N": int(N)},
            task=task_payload,
            sim={"horizon_T": int(horizon_T)},
            rng={"episode_seed": int(episode_seed)},
            disorder=disorder_block,
        )
        episodes.append(ep)

    # Shuffle deterministically by episode_id to avoid size-order bias while staying stable
    episodes.sort(key=lambda e: e.episode_id())
    return episodes


def generate_episodes_grid2d(
    *,
    split: SplitName,
    L_values: Sequence[int],
    horizon_T: int,
    n_episodes: int,
    task_block: JSONDict,
    build: ManifestBuildSpec,
    boundary: str = "open",
    disorder_block: Optional[JSONDict] = None,   # <-- ADD THIS
) -> List[EpisodeSpec]:
    """
    Grid-2D episode generation. We represent graph by (L, boundary).

    This is generic enough for search/mixing/robust-on-grid. Mark/disorder details
    should be included in task_block/disorder downstream; here we only fix seeds.
    """
    episodes: List[EpisodeSpec] = []
    if not L_values:
        return episodes

    picked_L = _cycle_pick(
        [int(x) for x in L_values],
        int(n_episodes),
        salt={"family": "grid-2d", "split": split, "horizon_T": int(horizon_T), "build_seed": int(build.seed), "boundary": boundary},
    )
    for i, L in enumerate(picked_L):
            seed_material = {
                "split": split,
                "horizon_T": horizon_T,
                "family": "grid-2d",
                "L": int(L),
                "boundary": boundary,
                "i": int(i),
                "build_seed": int(build.seed),
            }
            seed_hex = blake2b_hex(stable_json_dumps(seed_material).encode("utf-8"), digest_size=8)
            seed_int = int(seed_hex, 16)
            episode_seed = (int(build.episode_seed_base) + seed_int) % (2**64)

            ep = EpisodeSpec(
                family="grid-2d",
                graph={"L": int(L), "boundary": boundary},
                task=dict(task_block),
                sim={"horizon_T": int(horizon_T)},
                rng={"episode_seed": int(episode_seed)},
                disorder=disorder_block,
            )
            episodes.append(ep)

    episodes.sort(key=lambda e: e.episode_id())
    return episodes


def generate_episodes_hypercube(
    *,
    split: SplitName,
    n_values: Sequence[int],
    horizon_T: int,
    n_episodes: int,
    task_block: JSONDict,
    build: ManifestBuildSpec,
    disorder_block: Optional[JSONDict] = None,
) -> List[EpisodeSpec]:
    """
    Hypercube episode generation. Graph by dimension 'n' (Q_n).
    """
    episodes: List[EpisodeSpec] = []
    if not n_values:
        return episodes

    picked_n = _cycle_pick(
        [int(x) for x in n_values],
        int(n_episodes),
        salt={"family": "hypercube", "split": split, "horizon_T": int(horizon_T), "build_seed": int(build.seed)},
    )
    for i, n in enumerate(picked_n):
            seed_material = {
                "split": split,
                "horizon_T": horizon_T,
                "family": "hypercube",
                "n": int(n),
                "i": int(i),
                "build_seed": int(build.seed),
            }
            seed_hex = blake2b_hex(stable_json_dumps(seed_material).encode("utf-8"), digest_size=8)
            seed_int = int(seed_hex, 16)
            episode_seed = (int(build.episode_seed_base) + seed_int) % (2**64)

            ep = EpisodeSpec(
                family="hypercube",
                graph={"n": int(n)},
                task=dict(task_block),
                sim={"horizon_T": int(horizon_T)},
                rng={"episode_seed": int(episode_seed)},
                disorder=disorder_block,
            )
            episodes.append(ep)

    episodes.sort(key=lambda e: e.episode_id())
    return episodes


def generate_episodes_irregular_stub(
    *,
    split: SplitName,
    n_values: Sequence[int],
    horizon_T: int,
    n_episodes: int,
    task_block: JSONDict,
    build: ManifestBuildSpec,
    disorder_block: Optional[JSONDict] = None,
    kinds: Sequence[str] = ("er", "d-regular", "ws"),
) -> List[EpisodeSpec]:
    """
    Irregular graph episode generation stub.

    NOTE:
    This is intentionally conservative: without a unified YAML schema for irregular params
    (ER p grid, d-regular d grid, WS k/beta grid), we generate episodes over n_values and
    store 'kind' placeholders.
    """
    episodes: List[EpisodeSpec] = []
    if not n_values:
        return episodes

    kinds_list = list(kinds) if kinds else ["er"]

    pairs: List[Tuple[int, str]] = [(int(n), str(kind)) for n in n_values for kind in kinds_list]
    if not pairs:
        return episodes

    ordered_pairs = _deterministic_order(
        pairs,
        salt={
            "family": "irregular",
            "split": split,
            "horizon_T": int(horizon_T),
            "build_seed": int(build.seed),
        },
    )

    for i in range(int(n_episodes)):
        
        n, kind = ordered_pairs[i % len(ordered_pairs)]

        seed_material = {
            "split": split,
            "horizon_T": int(horizon_T),
            "family": "irregular",
            "kind": kind,
            "n": int(n),
            "i": int(i),
            "build_seed": int(build.seed),
        }
        seed_hex = blake2b_hex(stable_json_dumps(seed_material).encode("utf-8"), digest_size=8)
        seed_int = int(seed_hex, 16)
        episode_seed = (int(build.episode_seed_base) + seed_int) % (2**64)

        ep = EpisodeSpec(
            family="irregular",
            graph={"n": int(n), "kind": kind},
            task=dict(task_block),
            sim={"horizon_T": int(horizon_T)},
            rng={"episode_seed": int(episode_seed)},
            disorder=disorder_block,
        )
        episodes.append(ep)

    episodes.sort(key=lambda e: e.episode_id())
    return episodes


def _infer_regular_degree(spec: ExperimentManifestSpec) -> int:
    """
    Extract d for d-regular graphs from the experiment YAML.

    Expected schema (your YAML):
      data:
        family: "regular"
        regular:
          d: 3
    """
    # Most direct: your exact key
    reg = spec.data_block.get("regular", {})
    if isinstance(reg, dict) and "d" in reg:
        d = reg["d"]
    else:
        # Back-compat fallbacks (if you ever rename keys)
        d = (
            deep_get({"data": spec.data_block}, "data.regular.d", None)
            or deep_get({"data": spec.data_block}, "data.regular.degree", None)
            or spec.data_block.get("d", None)
            or spec.data_block.get("degree", None)
        )

    # Handle list-valued configs like d: [3,4,6] (pick first deterministically)
    if isinstance(d, list):
        d = d[0] if d else None

    if d is None:
        raise ValueError(
            "family=regular but no degree found. Expected one of:\n"
            "  data.regular.d (preferred)\n"
            "  data.regular.degree\n"
            "  data.d / data.degree\n"
        )

    d_int = int(d)
    if d_int < 0:
        raise ValueError(f"regular degree d must be non-negative (got {d_int}).")
    return d_int



def generate_episodes_regular(
    *,
    split: SplitName,
    sizes: Sequence[int],
    degree_d: int,
    horizon_T: int,
    n_episodes: int,
    task_block: JSONDict,
    build: ManifestBuildSpec,
    disorder_block: Optional[JSONDict] = None,
) -> List[EpisodeSpec]:
    """
    d-regular graph episode generation.
    Graph is represented by (N, d).
    """
    episodes: List[EpisodeSpec] = []
    if not sizes:
        return episodes
    picked_sizes = _cycle_pick(
        [int(x) for x in sizes],
        int(n_episodes),
        salt={"family": "regular", "split": split, "horizon_T": int(horizon_T), "build_seed": int(build.seed), "d": int(degree_d)},
    )
    for i, N in enumerate(picked_sizes):
        seed_material = {
            "split": split,
            "horizon_T": int(horizon_T),
            "family": "regular",
            "N": int(N),
            "d": int(degree_d),
            "i": int(i),
            "build_seed": int(build.seed),
        }
        seed_hex = blake2b_hex(stable_json_dumps(seed_material).encode("utf-8"), digest_size=8)
        seed_int = int(seed_hex, 16)
        episode_seed = (int(build.episode_seed_base) + seed_int) % (2**64)

        ep = EpisodeSpec(
            family="regular",
            graph={"N": int(N), "d": int(degree_d)},
            task=dict(task_block),
            sim={"horizon_T": int(horizon_T)},
            rng={"episode_seed": int(episode_seed)},
            disorder=disorder_block,
        )
        episodes.append(ep)

    episodes.sort(key=lambda e: e.episode_id())
    return episodes

# =============================================================================
# Leakage and sanity checks
# =============================================================================

def check_disjoint(train: Sequence[int], other: Sequence[int], *, label: str) -> List[str]:
    train_set = set(train)
    other_set = set(other)
    overlap = sorted(train_set.intersection(other_set))
    if overlap:
        return [f"[leakage] {label} overlaps train sizes: {overlap}"]
    return []


def check_ood_outside_train_band(train_spec: Optional[SizeSpec], ood: Sequence[int]) -> List[str]:
    if train_spec is None:
        return []
    if train_spec.mode == "fixed_list":
        return check_disjoint(train_spec.fixed_list, ood, label="OOD")
    if train_spec.mode == "uniform_band" and train_spec.band_min is not None and train_spec.band_max is not None:
        bad = [x for x in ood if train_spec.band_min <= x <= train_spec.band_max]
        if bad:
            return [f"[ood] OOD sizes are inside train band [{train_spec.band_min},{train_spec.band_max}]: {bad}"]
    return []


# =============================================================================
# Manifest writer
# =============================================================================

def build_manifest_payload(
    *,
    spec: ExperimentManifestSpec,
    split: SplitName,
    horizon_T: int,
    episodes: List[EpisodeSpec],
) -> JSONDict:
    items: List[JSONDict] = []
    for ep in episodes:
        d = ep.to_canonical_dict()
        d["episode_id"] = ep.episode_id()
        items.append(d)

    # fingerprint: hash of sorted episode IDs + key meta
    fingerprint_material = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": spec.name,
        "split": split,
        "horizon_T": int(horizon_T),
        "episode_ids": [it["episode_id"] for it in sorted(items, key=lambda x: x["episode_id"])],
        "source_config_sha16": spec.source_config_sha,
        "build_seed": spec.build.seed,
    }
    fingerprint = blake2b_hex(stable_json_dumps(fingerprint_material).encode("utf-8"), digest_size=16)

    payload: JSONDict = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "experiment_name": spec.name,
        "split": split,
        "horizon_T": int(horizon_T),
        "source_config": {
            "path": str(spec.source_config_path),
            "sha16": spec.source_config_sha,
        },
        "build": dataclasses.asdict(spec.build),
        "counts": {
            "n_episodes": len(items),
        },
        "fingerprint_sha16": fingerprint,
        "episodes": items,
        "meta": spec.meta,
    }
    return payload


def derive_per_horizon_filename(base: str, horizon_T: int) -> str:
    """
    Turns "transfer-line-val.json" -> "transfer-line-val-T64.json"
    """
    p = Path(base)
    stem = p.stem
    suffix = p.suffix or ".json"
    return f"{stem}-T{int(horizon_T)}{suffix}"


def print_summary(spec: ExperimentManifestSpec, messages: List[str]) -> None:
    print("")
    print("=== gen_manifest summary ===")
    print(f"Config: {spec.source_config_path}")
    print(f"Experiment: {spec.name}")
    print(f"Family: {spec.family}")
    print(f"Task: {spec.task_name}")
    print(f"Manifest root: {spec.paths.root_dir}")
    print(f"Horizon policy: {spec.build.horizon_policy}")
    print(f"Horizons: {spec.build.horizons}")
    print(f"Counts: val={spec.counts.val}, test_iid={spec.counts.test_iid}, test_ood={spec.counts.test_ood}")
    if messages:
        print("")
        print("Warnings/Errors:")
        for m in messages:
            print(f"  - {m}")
    print("============================")
    print("")


# =============================================================================
# Main generation logic
# =============================================================================

def generate_for_split(
    *,
    spec: ExperimentManifestSpec,
    split: SplitName,
    horizon_T: int,
    sizes: Sequence[int],
    n_episodes: int,
) -> List[EpisodeSpec]:
    fam = spec.family

    if fam == "MULTI":
        # Multi-family experiments: best-effort support.
        # If data.family is a list, the config likely needs more structure.
        # Here we default to generating nothing and ask user to convert the YAML.
        raise ValueError(
            "data.family is a list (multi-family). Convert the experiment YAML to Phase-B "
            "manifest style per-family split specs (e.g., grid + irregular)."
        )

    if fam == "line":
        return generate_episodes_line(
            split=split,
            sizes=sizes,
            horizon_T=horizon_T,
            n_episodes=n_episodes,
            task_block=spec.task_block,
            build=spec.build,
            disorder_block=spec.disorder_block,
        )
    
    if fam == "regular":
        d = _infer_regular_degree(spec)
        return generate_episodes_regular(
            split=split,
            sizes=sizes,
            degree_d=d,
            horizon_T=horizon_T,
            n_episodes=n_episodes,
            task_block=spec.task_block,
            build=spec.build,
            disorder_block=spec.disorder_block,
        )


    if fam == "grid-2d":
        boundary = deep_get({"task": spec.task_block}, "task.grid.boundary", None) or spec.task_block.get("boundary", "open")
        return generate_episodes_grid2d(
            split=split,
            L_values=sizes,
            horizon_T=horizon_T,
            n_episodes=n_episodes,
            task_block=spec.task_block,
            build=spec.build,
            boundary=str(boundary),
            disorder_block=spec.disorder_block,   # <-- ADD
        )

    if fam == "hypercube":
        return generate_episodes_hypercube(
            split=split,
            n_values=sizes,
            horizon_T=horizon_T,
            n_episodes=n_episodes,
            task_block=spec.task_block,
            build=spec.build,
            disorder_block=spec.disorder_block,   # <-- ADD (recommended)
        )

    if fam == "irregular":
        kinds = (
            deep_get({"data": spec.data_block}, "data.irregular.kinds", None)
            or deep_get({"data": spec.data_block}, "data.irregular.kind", None)
        )
        if not isinstance(kinds, list):
            kinds = ["er", "d-regular", "ws"]

        return generate_episodes_irregular_stub(
            split=split,
            n_values=sizes,
            horizon_T=horizon_T,
            n_episodes=n_episodes,
            task_block=spec.task_block,
            build=spec.build,
            disorder_block=spec.disorder_block,   # <-- ADD
            kinds=[str(k) for k in kinds],
        )


    raise ValueError(f"Unsupported family '{fam}'. Supported: line, regular, grid-2d, hypercube, irregular.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate deterministic evaluation manifests for ACPL experiments.")
    ap.add_argument("--config", type=str, required=True, help="Path to experiment YAML.")
    ap.add_argument("--out-root", type=str, default=None, help="Override manifests root_dir (default from YAML).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest files.")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files; just print what would be written.")
    ap.add_argument("--print-examples", type=int, default=0, help="Print N example episode dicts per split/horizon.")
    ap.add_argument("--no-test-iid", action="store_true", help="Do not generate test_iid manifests.")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    spec = parse_experiment_manifest(config_path)

    # Override out root if requested
    if args.out_root is not None:
        spec.paths.root_dir = Path(args.out_root).resolve()

    msgs: List[str] = []

    # Leakage checks (best-effort, size-based)
    train_vals = _expand_sizes(spec.train_sizes)
    val_vals = _expand_sizes(spec.val_sizes)
    iid_vals = _expand_sizes(spec.test_iid_sizes)
    ood_vals = _expand_sizes(spec.test_ood_sizes)

    # NOTE: Overlap in *sizes* between train and IID/VAL is expected (procedural training).
    # Real leakage is overlap in frozen episode lists between evaluation splits (checked later).
    msgs.extend(check_ood_outside_train_band(spec.train_sizes, ood_vals))

    if not spec.val_sizes:
        msgs.append("[warn] No val_sizes found in config; val manifest will be empty unless you add sizes.")
    if not spec.test_ood_sizes:
        msgs.append("[warn] No test_ood_sizes found in config; OOD manifest will be empty unless you add sizes.")

    print_summary(spec, msgs)

    # Determine horizons to generate
    horizons = spec.build.horizons
    per_horizon = (spec.build.horizon_policy == "per_horizon_manifest")

    # Split filename bases
    base_val = spec.paths.val
    base_iid = spec.paths.test_iid
    base_ood = spec.paths.test_ood

    # Prepare index mapping if per-horizon
    index_payload: JSONDict = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "experiment_name": spec.name,
        "source_config": {"path": str(spec.source_config_path), "sha16": spec.source_config_sha},
        "horizon_policy": spec.build.horizon_policy,
        "files": {"val": {}, "test_iid": {}, "test_ood": {}},
    }

    def _write_one(split: SplitName, horizon_T: int, episodes: List[EpisodeSpec], filename: str) -> None:
        payload = build_manifest_payload(spec=spec, split=split, horizon_T=horizon_T, episodes=episodes)
        out_path = spec.paths.root_dir / filename
        if args.dry_run:
            print(f"[dry-run] would write {out_path} (n={len(episodes)}) fingerprint={payload['fingerprint_sha16']}")
        else:
            write_json(out_path, payload, overwrite=args.overwrite)
            print(f"[write] {out_path} (n={len(episodes)}) fingerprint={payload['fingerprint_sha16']}")
        if args.print_examples > 0:
            ex = payload["episodes"][: args.print_examples]
            print(f"\n--- examples: split={split}, T={horizon_T}, file={filename} ---")
            for e in ex:
                print(stable_json_dumps(e)[:1200])
            print("--- end examples ---\n")

    # Generate per split/horizon
    for T in horizons:
        # VAL
        val_sizes = _expand_sizes(spec.val_sizes)
        val_eps = generate_for_split(spec=spec, split="val", horizon_T=T, sizes=val_sizes, n_episodes=spec.counts.val)
        val_file = derive_per_horizon_filename(base_val, T) if per_horizon else base_val
        _write_one("val", T, val_eps, val_file)
        if per_horizon:
            index_payload["files"]["val"][str(T)] = val_file

        # TEST_IID (optional)
        if not args.no_test_iid and base_iid is not None:
            iid_sizes = _expand_sizes(spec.test_iid_sizes)
            iid_eps = generate_for_split(spec=spec, split="test_iid", horizon_T=T, sizes=iid_sizes, n_episodes=spec.counts.test_iid)
            iid_file = derive_per_horizon_filename(base_iid, T) if per_horizon else base_iid
            _write_one("test_iid", T, iid_eps, iid_file)
            if per_horizon:
                index_payload["files"]["test_iid"][str(T)] = iid_file

        # TEST_OOD
        ood_sizes = _expand_sizes(spec.test_ood_sizes)
        ood_eps = generate_for_split(spec=spec, split="test_ood", horizon_T=T, sizes=ood_sizes, n_episodes=spec.counts.test_ood)
        ood_file = derive_per_horizon_filename(base_ood, T) if per_horizon else base_ood
        _write_one("test_ood", T, ood_eps, ood_file)
        
        
        
        
        # Real eval leakage check: episode_id overlap among eval splits for this horizon
        ids_val = {e.episode_id() for e in val_eps}
        ids_iid = {e.episode_id() for e in (iid_eps if (not args.no_test_iid and base_iid is not None) else [])}
        ids_ood = {e.episode_id() for e in ood_eps}
        if ids_val.intersection(ids_ood):
            msgs.append(f"[leakage] VAL overlaps TEST_OOD for T={T}: {len(ids_val.intersection(ids_ood))} episodes")
        if ids_iid and ids_iid.intersection(ids_ood):
            msgs.append(f"[leakage] TEST_IID overlaps TEST_OOD for T={T}: {len(ids_iid.intersection(ids_ood))} episodes")
        if ids_iid and ids_val.intersection(ids_iid):
            msgs.append(f"[leakage] VAL overlaps TEST_IID for T={T}: {len(ids_val.intersection(ids_iid))} episodes")

        
        
        
        
        if per_horizon:
            index_payload["files"]["test_ood"][str(T)] = ood_file

    # Write index file for per-horizon manifests
    if per_horizon:
        index_name = f"{Path(base_val).stem}.index.json"
        out_index = spec.paths.root_dir / index_name


        # fingerprint of index (stable; excludes timestamps)
        index_fingerprint_material = {
            "schema_version": SCHEMA_VERSION,
            "experiment_name": spec.name,
            "source_config_sha16": spec.source_config_sha,
            "horizon_policy": spec.build.horizon_policy,
            "files": index_payload["files"],
            "build_seed": spec.build.seed,
        }
        index_fingerprint = blake2b_hex(stable_json_dumps(index_fingerprint_material).encode("utf-8"), digest_size=16)





        index_payload["fingerprint_sha16"] = index_fingerprint

        if args.dry_run:
            print(f"[dry-run] would write index {out_index} fingerprint={index_fingerprint}")
        else:
            write_json(out_index, index_payload, overwrite=args.overwrite)
            print(f"[write] index {out_index} fingerprint={index_fingerprint}")

    print("\n[ok] manifest generation complete.\n")


if __name__ == "__main__":
    main()

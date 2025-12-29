#!/usr/bin/env python3
# scripts/collect_results.py
from __future__ import annotations

"""
scripts/collect_results.py

Research-ready “run bookkeeping” collector for evaluation outputs produced by scripts/eval.py.

What it does
------------
- Recursively scans a root directory (or explicit --evaldir paths) for evaluation folders that contain meta.json.
- Loads each evaluation run via acpl.eval.reporting.load_eval_run (preferred).
- Extracts per-condition metrics (CI if available; otherwise summary means).
- Writes:
    1) results.json          (structured registry of runs/conditions/metrics + provenance)
    2) results.csv           (wide CSV; one row per run×condition with mean metrics)
    3) results_long.csv      (long CSV; one row per run×condition×metric with CI fields if present)
    4) runs.csv              (run-level summary; one row per eval directory)
    5) collection_meta.json  (collector provenance: args, git, counts, errors)

Why this matters
----------------
Novelty-level experimental results are not “numbers in a terminal”—they’re structured comparisons:
- consistent run IDs
- consistent metric columns
- reproducible provenance
- robust handling of partial artifacts and failures

Typical usage
-------------
# Scan a whole tree of eval outputs:
python scripts/collect_results.py --root eval --outdir eval/_collected

# Collect explicit eval dirs:
python scripts/collect_results.py --evaldirs eval/tmp eval/expA eval/expB --outdir eval/_collected

# Control metric selection:
python scripts/collect_results.py --root eval --metric-mode union --max-metrics-per-run 8 --with-ci-cols

Outputs are deterministic given identical input directories.
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ----------------------------- logging -----------------------------

log = logging.getLogger("collect_results")


# ----------------------------- small utils -----------------------------


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _as_path(p: str | os.PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(_read_text(path))
    except Exception:
        return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if not (v == v):  # NaN
            return None
        return v
    except Exception:
        return None


def _blake2b_hex(data: bytes, *, digest_size: int = 16) -> str:
    h = hashlib.blake2b(data, digest_size=digest_size)
    return h.hexdigest()


def _stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sanitize(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    return s or "x"


def _metric_priority(name: str, *, suite: str | None = None) -> tuple[int, str]:
    """
    Heuristic: lower tuple => earlier column.

    Mirrors the spirit of acpl.eval.reporting._metric_priority, but kept local
    so this script stays stable even if internals change.
    """
    s = (name or "").lower()
    if suite and "mix" in suite.lower() and "tv" in s:
        return (0, name)
    if "tv" in s:
        return (1, name)
    if ("target" in s and ("p_" in s or "prob" in s or "success" in s)) or ("p_target" in s):
        return (2, name)
    if "success" in s or "hit" in s or "found" in s:
        return (3, name)
    if "loss" in s:
        return (90, name)
    return (10, name)


def _relpath(from_dir: Path, to_path: Path) -> str:
    try:
        return str(to_path.relative_to(from_dir))
    except Exception:
        try:
            return os.path.relpath(str(to_path), str(from_dir))
        except Exception:
            return str(to_path)


def _git_info(repo_root: Path) -> dict[str, Any]:
    """
    Best-effort git provenance. Never fails the collector.
    """
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(args, cwd=str(repo_root), stderr=subprocess.DEVNULL)
            s = out.decode("utf-8", errors="replace").strip()
            return s or None
        except Exception:
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    describe = _run(["git", "describe", "--tags", "--always", "--dirty"])
    return {
        "head": head,
        "branch": branch,
        "describe": describe,
        "dirty": bool(status),
    }


# ----------------------------- reporting import (preferred) -----------------------------

_HAVE_REPORTING = False
try:
    from acpl.eval.reporting import load_eval_run, select_key_metrics  # type: ignore
    _HAVE_REPORTING = True
except Exception:
    _HAVE_REPORTING = False
    load_eval_run = None  # type: ignore
    select_key_metrics = None  # type: ignore


# ----------------------------- data model -----------------------------


@dataclass
class MetricRecord:
    mean: float | None = None
    lo: float | None = None
    hi: float | None = None
    stderr: float | None = None
    n: int | None = None
    source: str = ""  # "ci" or "summary" or ""


@dataclass
class CollectedCondition:
    cond: str
    kind: str  # "base" | "baseline" | "ablation" | "other"
    policy: str | None = None
    baseline_kind: str | None = None
    ablation_meta: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, MetricRecord] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)  # optional pointers to artifacts


@dataclass
class CollectedRun:
    evaldir: str
    run_id: str
    meta_hash: str
    suite: str | None = None
    checkpoint: str | None = None
    device: str | None = None
    episodes: int | None = None
    seeds: list[int] = field(default_factory=list)
    policy: str | None = None
    baseline_kind: str | None = None
    ablations: list[str] = field(default_factory=list)
    model_config_hash: str | None = None
    raw_meta: dict[str, Any] = field(default_factory=dict)

    conditions: dict[str, CollectedCondition] = field(default_factory=dict)


@dataclass
class CollectionResult:
    schema: str
    generated_at: str
    root: str | None
    outdir: str
    git: dict[str, Any]
    runs: list[CollectedRun]
    errors: list[dict[str, Any]]
    metric_columns: dict[str, str]  # original metric name -> sanitized column base


# ----------------------------- discovery -----------------------------


def _looks_like_evaldir(path: Path) -> bool:
    """
    “Evaldir” is a directory produced by scripts/eval.py that contains meta.json.
    Avoid false positives in raw/<cond>/ where meta.json is condition_meta.json.
    """
    if not path.is_dir():
        return False
    meta = path / "meta.json"
    if not meta.exists():
        return False
    # If it contains summary.json or raw/ or figs/, it is almost certainly eval output.
    if (path / "summary.json").exists():
        return True
    if (path / "raw").exists():
        return True
    if (path / "figs").exists():
        return True
    # still allow, but it might be a partial run directory
    return True


def find_evaldirs(
    *,
    root: Path,
    max_depth: int | None = None,
    follow_symlinks: bool = False,
    exclude_regex: str | None = None,
) -> list[Path]:
    """
    Recursively find directories under root that contain meta.json and look like eval outputs.

    We search by locating meta.json files, then taking their parent as candidate evaldir.
    """
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    rx = re.compile(exclude_regex) if exclude_regex else None

    # Note: we manually walk to allow max_depth control.
    out: list[Path] = []
    root_parts = len(root.parts)

    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=follow_symlinks):
        d = Path(dirpath)

        # depth prune
        if max_depth is not None:
            depth = len(d.parts) - root_parts
            if depth > max_depth:
                dirnames[:] = []
                continue

        if rx and rx.search(str(d)):
            dirnames[:] = []
            continue

        if "meta.json" in filenames:
            cand = d
            if _looks_like_evaldir(cand):
                out.append(cand)

            # Do not descend into this directory's children aggressively; eval trees can be large.
            # However, there may be nested evaldirs; so we *don’t* prune entirely.
            # We do prune the extremely common raw/ subtree if present.
            if "raw" in dirnames:
                # keep walking other dirs, but raw/ contains per-condition dirs, not evaldirs.
                dirnames.remove("raw")

    # dedupe + stable order
    uniq: dict[str, Path] = {}
    for p in out:
        uniq[str(p.resolve())] = p.resolve()
    return sorted(uniq.values(), key=lambda p: str(p))


# ----------------------------- loading + extraction -----------------------------


def _compute_evaldir_hash(evaldir: Path) -> str:
    """
    Stable-ish hash based on meta.json + summary.json (if present).
    """
    meta_p = evaldir / "meta.json"
    data = _read_bytes(meta_p)
    summ_p = evaldir / "summary.json"
    if summ_p.exists():
        data += b"\n" + _read_bytes(summ_p)
    return _blake2b_hex(data, digest_size=16)


def _compute_model_config_hash(meta_json: Mapping[str, Any]) -> str | None:
    cfg = meta_json.get("model_config", None)
    if isinstance(cfg, Mapping):
        return _blake2b_hex(_stable_json_bytes(cfg), digest_size=16)
    return None


def _make_run_id(evaldir: Path, meta_json: Mapping[str, Any], meta_hash: str) -> str:
    """
    Run ID should be:
    - stable across copies (relative paths shouldn’t matter)
    - collision-resistant enough for bookkeeping
    """
    suite = str(meta_json.get("suite") or "suite")
    ckpt = str(meta_json.get("checkpoint") or meta_json.get("ckpt") or "ckpt")
    ckpt_base = Path(ckpt).name if ckpt else "ckpt"
    policy = str(meta_json.get("policy") or "policy")
    episodes = meta_json.get("episodes", None)
    seeds = meta_json.get("seeds", [])
    try:
        seeds_s = ",".join(str(int(x)) for x in (seeds or []))
    except Exception:
        seeds_s = "seeds"
    eps_s = str(int(episodes)) if episodes is not None and _is_number(episodes) else "E"
    # include meta_hash suffix to prevent collisions from same ckpt/suite combos
    return _sanitize(f"{suite}__{policy}__{ckpt_base}__{eps_s}__{seeds_s}__{meta_hash[:8]}")


def _discover_condition_meta(cdir: Path) -> dict[str, Any]:
    p = cdir / "condition_meta.json"
    if p.exists():
        j = _safe_read_json(p)
        return j or {}
    return {}


def _condition_kind(cond: str, base_cond: str | None, cond_meta: Mapping[str, Any]) -> str:
    if base_cond and cond == base_cond:
        return "base"
    if cond.startswith("baseline_") or (cond_meta.get("baseline_kind") is not None):
        return "baseline"
    am = cond_meta.get("ablation_meta", None)
    if isinstance(am, Mapping) and len(am) > 0:
        return "ablation"
    return "other"


def _extract_metric_records_from_reporting_condition(cr: Any, metric_names: Iterable[str]) -> dict[str, MetricRecord]:
    """
    cr is an acpl.eval.reporting.ConditionResult instance.
    """
    out: dict[str, MetricRecord] = {}
    ci = getattr(cr, "ci", {}) or {}
    sm = getattr(cr, "summary_means", {}) or {}

    for m in metric_names:
        if m in ci:
            rec = ci[m]
            out[m] = MetricRecord(
                mean=_safe_float(getattr(rec, "mean", None)),
                lo=_safe_float(getattr(rec, "lo", None)),
                hi=_safe_float(getattr(rec, "hi", None)),
                stderr=_safe_float(getattr(rec, "stderr", None)),
                n=int(getattr(rec, "n", None)) if getattr(rec, "n", None) is not None else None,
                source="ci",
            )
        elif m in sm:
            out[m] = MetricRecord(mean=_safe_float(sm[m]), source="summary")
    return out


def _discover_fig_paths(evaldir: Path, cond: str) -> dict[str, str]:
    """
    Stable figure naming pattern (as used by scripts/eval.py and reporting.py):
      figs/Pt__<safe_cond>.png
      figs/tv__<safe_cond>.png
    """
    out: dict[str, str] = {}
    figdir = evaldir / "figs"
    if not figdir.exists():
        return out
    safe = _sanitize(cond)
    pt = figdir / f"Pt__{safe}.png"
    tv = figdir / f"tv__{safe}.png"
    if pt.exists():
        out["fig_pt"] = str(pt)
    if tv.exists():
        out["fig_tv"] = str(tv)
    return out


def _discover_artifacts(evaldir: Path, cond: str) -> dict[str, str]:
    """
    Best-effort artifact discovery. Supports common layouts:
      <evaldir>/artifacts/...
      <evaldir>/eval/artifacts/...
    and looks for embeddings stats/meta if present.

    This does not assume your embeddings artifact layout is finalized yet.
    """
    out: dict[str, str] = {}
    candidates = [
        evaldir / "artifacts",
        evaldir / "eval" / "artifacts",
    ]
    for root in candidates:
        if not root.exists():
            continue
        # Try embeddings paths
        # (these are intentionally flexible; we include what exists without enforcing structure)
        emb_dir = root / "embeddings" / _sanitize(cond)
        if emb_dir.exists():
            for name in ("embeddings.npy", "embeddings_mean.npy", "embeddings_stats.json", "embeddings.meta.json"):
                p = emb_dir / name
                if p.exists():
                    out[f"embeddings/{name}"] = str(p)
        # Also allow non-cond subdir (some writers place cond in filename)
        for name in ("embeddings_stats.json", "embeddings.meta.json"):
            p = root / "embeddings" / name
            if p.exists():
                out[f"embeddings/{name}"] = str(p)
    return out


def load_run_via_reporting(evaldir: Path) -> Any:
    if not _HAVE_REPORTING or load_eval_run is None:
        raise RuntimeError(
            "acpl.eval.reporting is not importable. "
            "Run this script from your repo environment where `acpl` is on PYTHONPATH."
        )
    return load_eval_run(str(evaldir))


# ----------------------------- metric selection -----------------------------


def _union_metrics(runs: Sequence[Any]) -> set[str]:
    names: set[str] = set()
    for run in runs:
        conds = getattr(run, "conditions", {}) or {}
        for cr in conds.values():
            ci = getattr(cr, "ci", {}) or {}
            sm = getattr(cr, "summary_means", {}) or {}
            names |= set(ci.keys()) | set(sm.keys())
    return names


def _intersection_metrics(runs: Sequence[Any]) -> set[str]:
    inter: set[str] | None = None
    for run in runs:
        names = _union_metrics([run])
        inter = names if inter is None else (inter & names)
    return inter or set()


def _select_metrics_auto(
    runs: Sequence[Any],
    *,
    max_per_run: int,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
) -> set[str]:
    """
    For each run, select headline metrics (via reporting.select_key_metrics if available),
    and union across runs.
    """
    inc = set(m.lower() for m in (include or []))
    exc = set(m.lower() for m in (exclude or []))

    out: set[str] = set()
    for run in runs:
        suite = getattr(getattr(run, "meta", None), "suite", None)
        # prefer reporting's selector; fallback to heuristic sort
        if _HAVE_REPORTING and select_key_metrics is not None:
            keys = select_key_metrics(run, max_cols=int(max_per_run), include=list(include or []), exclude=list(exclude or []))
        else:
            names = sorted(_union_metrics([run]))
            if inc:
                names = [n for n in names if n.lower() in inc]
            if exc:
                names = [n for n in names if n.lower() not in exc]
            names.sort(key=lambda n: _metric_priority(n, suite=suite))
            keys = names[: max(1, int(max_per_run))]
        out |= set(keys)
    return out


# ----------------------------- CSV writers -----------------------------


def _build_metric_column_map(metric_names: Sequence[str]) -> dict[str, str]:
    """
    Map metric name -> sanitized base column name.
    Keeps it stable and collision-aware.
    """
    used: dict[str, str] = {}
    out: dict[str, str] = {}
    for m in metric_names:
        base = _sanitize(m.replace(".", "_"))
        if base in used and used[base] != m:
            # collision: disambiguate
            base2 = _sanitize(base + "__" + _blake2b_hex(m.encode("utf-8"), digest_size=4))
            out[m] = base2
        else:
            out[m] = base
            used[base] = m
    return out


def write_results_wide_csv(
    path: Path,
    runs: Sequence[CollectedRun],
    *,
    metrics: Sequence[str],
    metric_cols: Mapping[str, str],
    with_ci_cols: bool,
) -> None:
    """
    Wide CSV: one row per run×condition, includes run meta + metric means (and optionally CI cols).
    """
    _ensure_dir(path.parent)

    base_cols = [
        "run_id",
        "evaldir",
        "suite",
        "checkpoint",
        "device",
        "episodes",
        "seeds",
        "policy",
        "baseline_kind",
        "cond",
        "cond_kind",
    ]

    metric_fieldnames: list[str] = []
    for m in metrics:
        base = metric_cols[m]
        metric_fieldnames.append(base)
        if with_ci_cols:
            metric_fieldnames.extend([f"{base}__lo", f"{base}__hi", f"{base}__stderr", f"{base}__n", f"{base}__src"])

    fieldnames = base_cols + metric_fieldnames

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for run in runs:
            for cond, cc in run.conditions.items():
                row: dict[str, Any] = {
                    "run_id": run.run_id,
                    "evaldir": run.evaldir,
                    "suite": run.suite or "",
                    "checkpoint": run.checkpoint or "",
                    "device": run.device or "",
                    "episodes": run.episodes if run.episodes is not None else "",
                    "seeds": ",".join(str(s) for s in run.seeds) if run.seeds else "",
                    "policy": run.policy or "",
                    "baseline_kind": run.baseline_kind or "",
                    "cond": cond,
                    "cond_kind": cc.kind,
                }

                for m in metrics:
                    col = metric_cols[m]
                    rec = cc.metrics.get(m, MetricRecord())
                    row[col] = "" if rec.mean is None else rec.mean
                    if with_ci_cols:
                        row[f"{col}__lo"] = "" if rec.lo is None else rec.lo
                        row[f"{col}__hi"] = "" if rec.hi is None else rec.hi
                        row[f"{col}__stderr"] = "" if rec.stderr is None else rec.stderr
                        row[f"{col}__n"] = "" if rec.n is None else rec.n
                        row[f"{col}__src"] = rec.source or ""
                w.writerow(row)


def write_results_long_csv(
    path: Path,
    runs: Sequence[CollectedRun],
    *,
    metrics: Sequence[str],
) -> None:
    """
    Long CSV: one row per run×condition×metric with CI fields if present.
    """
    _ensure_dir(path.parent)
    fieldnames = [
        "run_id",
        "evaldir",
        "suite",
        "checkpoint",
        "device",
        "episodes",
        "seeds",
        "policy",
        "baseline_kind",
        "cond",
        "cond_kind",
        "metric",
        "mean",
        "lo",
        "hi",
        "stderr",
        "n",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for run in runs:
            for cond, cc in run.conditions.items():
                for m in metrics:
                    rec = cc.metrics.get(m, MetricRecord())
                    if rec.mean is None and rec.lo is None and rec.hi is None:
                        continue
                    w.writerow(
                        {
                            "run_id": run.run_id,
                            "evaldir": run.evaldir,
                            "suite": run.suite or "",
                            "checkpoint": run.checkpoint or "",
                            "device": run.device or "",
                            "episodes": run.episodes if run.episodes is not None else "",
                            "seeds": ",".join(str(s) for s in run.seeds) if run.seeds else "",
                            "policy": run.policy or "",
                            "baseline_kind": run.baseline_kind or "",
                            "cond": cond,
                            "cond_kind": cc.kind,
                            "metric": m,
                            "mean": "" if rec.mean is None else rec.mean,
                            "lo": "" if rec.lo is None else rec.lo,
                            "hi": "" if rec.hi is None else rec.hi,
                            "stderr": "" if rec.stderr is None else rec.stderr,
                            "n": "" if rec.n is None else rec.n,
                            "source": rec.source or "",
                        }
                    )


def write_runs_csv(path: Path, runs: Sequence[CollectedRun]) -> None:
    """
    Run-level summary CSV: one row per evaldir/run_id.
    """
    _ensure_dir(path.parent)
    fieldnames = [
        "run_id",
        "evaldir",
        "meta_hash",
        "suite",
        "checkpoint",
        "device",
        "episodes",
        "seeds",
        "policy",
        "baseline_kind",
        "ablations",
        "model_config_hash",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in runs:
            w.writerow(
                {
                    "run_id": r.run_id,
                    "evaldir": r.evaldir,
                    "meta_hash": r.meta_hash,
                    "suite": r.suite or "",
                    "checkpoint": r.checkpoint or "",
                    "device": r.device or "",
                    "episodes": r.episodes if r.episodes is not None else "",
                    "seeds": ",".join(str(s) for s in r.seeds) if r.seeds else "",
                    "policy": r.policy or "",
                    "baseline_kind": r.baseline_kind or "",
                    "ablations": ",".join(r.ablations) if r.ablations else "",
                    "model_config_hash": r.model_config_hash or "",
                }
            )


# ----------------------------- collector core -----------------------------


def collect_one_evaldir(
    evaldir: Path,
    *,
    strict: bool,
    metric_names_hint: set[str] | None = None,
) -> tuple[CollectedRun | None, dict[str, Any] | None]:
    """
    Returns (CollectedRun | None, error_payload | None).
    """
    evaldir = evaldir.resolve()
    meta_path = evaldir / "meta.json"
    if not meta_path.exists():
        err = {"evaldir": str(evaldir), "error": "missing meta.json"}
        return (None, err)

    meta_json = _safe_read_json(meta_path) or {}
    meta_hash = _compute_evaldir_hash(evaldir)
    run_id = _make_run_id(evaldir, meta_json, meta_hash)
    model_cfg_hash = _compute_model_config_hash(meta_json)

    # load via reporting
    try:
        run_obj = load_run_via_reporting(evaldir)
    except Exception as e:
        err = {"evaldir": str(evaldir), "run_id": run_id, "error": f"load failed: {type(e).__name__}: {e}"}
        if strict:
            raise
        return (None, err)

    # derive base condition
    base_cond = None
    try:
        base_cond = run_obj.base_condition()
    except Exception:
        base_cond = None

    # metric names: gather union for this run if hint not provided
    if metric_names_hint is None:
        metric_names = sorted(_union_metrics([run_obj]), key=lambda n: _metric_priority(n, suite=getattr(run_obj.meta, "suite", None)))
    else:
        metric_names = sorted(metric_names_hint, key=lambda n: _metric_priority(n, suite=getattr(run_obj.meta, "suite", None)))

    # build collected run
    crun = CollectedRun(
        evaldir=str(evaldir),
        run_id=run_id,
        meta_hash=meta_hash,
        suite=str(getattr(getattr(run_obj, "meta", None), "suite", "") or meta_json.get("suite") or "") or None,
        checkpoint=str(meta_json.get("checkpoint") or meta_json.get("ckpt") or "") or None,
        device=str(meta_json.get("device") or "") or None,
        episodes=int(meta_json["episodes"]) if "episodes" in meta_json and _is_number(meta_json["episodes"]) else None,
        seeds=[int(x) for x in (meta_json.get("seeds") or []) if _is_number(x)],
        policy=str(meta_json.get("policy") or "") or None,
        baseline_kind=str(meta_json.get("baseline_kind") or "") or None,
        ablations=[str(x) for x in (meta_json.get("ablations_canonical") or meta_json.get("ablations") or [])],
        model_config_hash=model_cfg_hash,
        raw_meta=dict(meta_json),
    )

    # conditions
    conds = getattr(run_obj, "conditions", {}) or {}
    for cond_name, cond_obj in conds.items():
        cond_name = str(cond_name)

        # pull richer condition meta from filesystem if present
        cdir = None
        try:
            cdir = getattr(cond_obj, "path", None)
        except Exception:
            cdir = None
        cond_meta = {}
        if isinstance(cdir, Path) and cdir.exists():
            cond_meta = _discover_condition_meta(cdir)

        kind = _condition_kind(cond_name, base_cond, cond_meta)

        policy = cond_meta.get("policy", None)
        baseline_kind = cond_meta.get("baseline_kind", None)
        ablation_meta = cond_meta.get("ablation_meta", {})
        if not isinstance(ablation_meta, Mapping):
            ablation_meta = {}

        metrics = _extract_metric_records_from_reporting_condition(cond_obj, metric_names)

        cc = CollectedCondition(
            cond=cond_name,
            kind=kind,
            policy=str(policy) if policy is not None else None,
            baseline_kind=str(baseline_kind) if baseline_kind is not None else None,
            ablation_meta=dict(ablation_meta),
            metrics=metrics,
            paths={},
        )

        # figure pointers + optional artifacts
        cc.paths.update(_discover_fig_paths(evaldir, cond_name))
        cc.paths.update(_discover_artifacts(evaldir, cond_name))

        crun.conditions[cond_name] = cc

    return (crun, None)


def collect_all(
    evaldirs: Sequence[Path],
    *,
    strict: bool,
) -> tuple[list[CollectedRun], list[dict[str, Any]]]:
    runs: list[CollectedRun] = []
    errors: list[dict[str, Any]] = []

    for d in evaldirs:
        try:
            r, err = collect_one_evaldir(d, strict=strict)
            if r is not None:
                runs.append(r)
            if err is not None:
                errors.append(err)
        except Exception as e:
            payload = {"evaldir": str(d), "error": f"fatal: {type(e).__name__}: {e}"}
            errors.append(payload)
            if strict:
                raise

    # stable sort
    runs.sort(key=lambda r: (r.suite or "", r.run_id))
    return runs, errors


# ----------------------------- CLI -----------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect evaluation outputs into a single results registry (CSV/JSON).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=str, default="", help="Root directory to scan for eval outputs (contains meta.json).")
    src.add_argument("--evaldirs", nargs="+", default=[], help="Explicit eval output directories to collect.")

    p.add_argument("--outdir", type=str, required=True, help="Output directory for collected artifacts.")
    p.add_argument("--max-depth", type=int, default=None, help="Max scan depth under --root (None = unlimited).")
    p.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks during scanning.")
    p.add_argument(
        "--exclude-regex",
        type=str,
        default=r"(^|/)(\.git|__pycache__|wandb|mlruns|raw)(/|$)",
        help="Regex of paths to exclude during scanning (applied to directory paths).",
    )

    # metric selection
    p.add_argument(
        "--metric-mode",
        type=str,
        default="auto",
        choices=["auto", "union", "intersection"],
        help="How to choose metric columns across runs.",
    )
    p.add_argument(
        "--max-metrics-per-run",
        type=int,
        default=6,
        help="When metric-mode=auto, maximum headline metrics to select per run before unioning.",
    )
    p.add_argument(
        "--include-metric",
        action="append",
        default=[],
        help="Explicit metric name to include (repeatable). If any are provided, acts as a whitelist.",
    )
    p.add_argument(
        "--exclude-metric",
        action="append",
        default=[],
        help="Metric name to exclude (repeatable).",
    )
    p.add_argument(
        "--with-ci-cols",
        action="store_true",
        help="In results.csv, add __lo/__hi/__stderr/__n/__src columns per metric.",
    )

    # behavior
    p.add_argument("--strict", action="store_true", help="Fail-fast if any evaldir fails to load.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    outdir = _as_path(args.outdir).expanduser().resolve()
    _ensure_dir(outdir)

    # Determine evaldirs
    if args.evaldirs:
        evaldirs = [_as_path(x).expanduser().resolve() for x in args.evaldirs]
    else:
        root = _as_path(args.root).expanduser().resolve()
        evaldirs = find_evaldirs(
            root=root,
            max_depth=args.max_depth,
            follow_symlinks=bool(args.follow_symlinks),
            exclude_regex=args.exclude_regex,
        )

    # Basic sanity
    if not evaldirs:
        log.error("No evaluation directories found.")
        return 2

    # Git provenance: try repo root as cwd
    repo_root = Path.cwd().resolve()
    git = _git_info(repo_root)

    if not _HAVE_REPORTING:
        log.warning(
            "acpl.eval.reporting import failed. This collector expects to run in the repo environment."
        )
        return 2

    log.info("Found %d candidate evaldirs.", len(evaldirs))

    # Load run objects once to decide metrics (auto/union/intersection)
    # We use reporting loader for this phase to make metric selection accurate.
    loaded_run_objs: list[Any] = []
    prelim_errors: list[dict[str, Any]] = []

    for d in evaldirs:
        try:
            loaded_run_objs.append(load_run_via_reporting(d))
        except Exception as e:
            payload = {"evaldir": str(d), "error": f"preload failed: {type(e).__name__}: {e}"}
            prelim_errors.append(payload)
            if args.strict:
                raise

    if not loaded_run_objs:
        log.error("All candidate evaldirs failed to load.")
        _write_json(outdir / "collection_meta.json", {"errors": prelim_errors, "generated_at": _utc_now_iso()})
        return 2

    include = list(args.include_metric or [])
    exclude = list(args.exclude_metric or [])

    if args.metric_mode == "auto":
        metric_set = _select_metrics_auto(
            loaded_run_objs,
            max_per_run=int(args.max_metrics_per_run),
            include=include if include else None,
            exclude=exclude if exclude else None,
        )
    elif args.metric_mode == "union":
        metric_set = _union_metrics(loaded_run_objs)
    else:
        metric_set = _intersection_metrics(loaded_run_objs)

    # Apply include/exclude constraints for union/intersection too
    if include:
        inc = set(m.lower() for m in include)
        metric_set = {m for m in metric_set if m.lower() in inc}
    if exclude:
        exc = set(m.lower() for m in exclude)
        metric_set = {m for m in metric_set if m.lower() not in exc}

    suite_for_sort = None
    try:
        suite_for_sort = getattr(getattr(loaded_run_objs[0], "meta", None), "suite", None)
    except Exception:
        suite_for_sort = None

    metrics = sorted(metric_set, key=lambda n: _metric_priority(n, suite=suite_for_sort))

    if not metrics:
        log.error("No metrics selected after filtering. Check --include-metric/--exclude-metric.")
        return 2

    metric_cols = _build_metric_column_map(metrics)

    # Now collect fully (structured + file pointers)
    runs, errors = collect_all(evaldirs, strict=bool(args.strict))
    errors = prelim_errors + errors

    # Emit structured JSON registry
    coll = CollectionResult(
        schema="acpl.collect_results.v1",
        generated_at=_utc_now_iso(),
        root=str(_as_path(args.root).expanduser().resolve()) if args.root else None,
        outdir=str(outdir),
        git=git,
        runs=runs,
        errors=errors,
        metric_columns=dict(metric_cols),
    )

    # Write outputs
    results_json = outdir / "results.json"
    runs_csv = outdir / "runs.csv"
    results_csv = outdir / "results.csv"
    results_long_csv = outdir / "results_long.csv"
    meta_json = outdir / "collection_meta.json"

    # Convert dataclasses to JSON-friendly structure (no dataclasses.asdict to keep control)
    def _metric_to_dict(m: MetricRecord) -> dict[str, Any]:
        return {
            "mean": m.mean,
            "lo": m.lo,
            "hi": m.hi,
            "stderr": m.stderr,
            "n": m.n,
            "source": m.source,
        }

    def _cond_to_dict(c: CollectedCondition) -> dict[str, Any]:
        return {
            "cond": c.cond,
            "kind": c.kind,
            "policy": c.policy,
            "baseline_kind": c.baseline_kind,
            "ablation_meta": c.ablation_meta,
            "metrics": {k: _metric_to_dict(v) for k, v in c.metrics.items()},
            "paths": c.paths,
        }

    def _run_to_dict(r: CollectedRun) -> dict[str, Any]:
        return {
            "evaldir": r.evaldir,
            "run_id": r.run_id,
            "meta_hash": r.meta_hash,
            "suite": r.suite,
            "checkpoint": r.checkpoint,
            "device": r.device,
            "episodes": r.episodes,
            "seeds": r.seeds,
            "policy": r.policy,
            "baseline_kind": r.baseline_kind,
            "ablations": r.ablations,
            "model_config_hash": r.model_config_hash,
            "raw_meta": r.raw_meta,
            "conditions": {k: _cond_to_dict(v) for k, v in r.conditions.items()},
        }

    _write_json(
        results_json,
        {
            "schema": coll.schema,
            "generated_at": coll.generated_at,
            "root": coll.root,
            "outdir": coll.outdir,
            "git": coll.git,
            "metric_columns": coll.metric_columns,
            "metrics": metrics,
            "runs": [_run_to_dict(r) for r in coll.runs],
            "errors": coll.errors,
        },
    )

    write_runs_csv(runs_csv, runs)
    write_results_wide_csv(results_csv, runs, metrics=metrics, metric_cols=metric_cols, with_ci_cols=bool(args.with_ci_cols))
    write_results_long_csv(results_long_csv, runs, metrics=metrics)

    _write_json(
        meta_json,
        {
            "schema": coll.schema,
            "generated_at": coll.generated_at,
            "argv": sys.argv,
            "cwd": str(Path.cwd().resolve()),
            "git": coll.git,
            "counts": {
                "candidates": len(evaldirs),
                "loaded_runs": len(runs),
                "errors": len(errors),
                "metrics": len(metrics),
            },
            "paths": {
                "results_json": str(results_json),
                "runs_csv": str(runs_csv),
                "results_csv": str(results_csv),
                "results_long_csv": str(results_long_csv),
            },
            "metric_columns": coll.metric_columns,
        },
    )

    log.info("Wrote: %s", str(results_json))
    log.info("Wrote: %s", str(runs_csv))
    log.info("Wrote: %s", str(results_csv))
    log.info("Wrote: %s", str(results_long_csv))
    if errors:
        log.warning("Completed with %d errors (see %s).", len(errors), str(meta_json))
    else:
        log.info("Completed with zero errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# acpl/eval/dataset_audit.py
from __future__ import annotations

"""
Dataset / split / bucketization audit utilities.

This module is meant to make evaluation *defendable* by recording what data was
evaluated, how it was split, and how the episode population is distributed across:

- splits (train/val/test),
- graph families,
- tasks,
- graph-size "buckets" (binning) based on a primary size key (e.g., N nodes),
- arbitrary "bags" (bagging) defined as grouping keys (e.g., family × task × N_bin).

It is intentionally designed to work from an *evaluation manifest directory* produced
by `acpl.data.manifest.make_eval_manifest(...)`.

Expected manifest layout
------------------------
<manifest_dir>/
  index.json
  splits/
    train.jsonl
    val.jsonl
    test.jsonl

Each JSONL entry is expected to include at least:
  - split: "train"|"val"|"test"
  - family: str (graph family name)
  - task: str (task name/kind)
  - size: dict[str, Any] (graph size metadata; should include a node-count-like key)

If some fields are missing, this module degrades gracefully, but output will note
missingness.

Outputs
-------
By default, `run_dataset_audit(...)` writes:

<outdir>/
  audit.json                 # machine-readable summary (counts, stats, bins, tables)
  bucket_table.csv           # long-form bucket counts (split,family,task,bin,count)
  size_key_summary.csv       # numeric summaries per size key
  figures/
    split_counts.png
    family_counts_by_split.png
    task_counts_by_split.png
    size_hist_<key>.png
    bucket_heatmap_<row>_x_<col>.png

All outputs are deterministic given the same manifest inputs and config.

Design notes
------------
- This module does not require pandas.
- Plotting uses matplotlib in Agg mode for headless usage.
- Bootstrap CIs (optional) are nonparametric over episodes within each split.
"""

from dataclasses import dataclass
import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence


import numpy as np

# Matplotlib is optional unless plots=True
try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None  # type: ignore
    plt = None  # type: ignore


SplitName = Literal["train", "val", "test"]

_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BinSpec:
    """
    Defines numeric bin edges for a key. Bins follow numpy.histogram semantics:
    edges are inclusive on the left, exclusive on the right except the last bin.
    """

    key: str
    edges: tuple[float, ...]
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.edges) < 2:
            raise ValueError("BinSpec.edges must have at least 2 edges.")
        if any(self.edges[i] >= self.edges[i + 1] for i in range(len(self.edges) - 1)):
            raise ValueError("BinSpec.edges must be strictly increasing.")
        if self.labels is not None and len(self.labels) != (len(self.edges) - 1):
            raise ValueError("BinSpec.labels must have len(edges)-1.")


@dataclass
class AuditConfig:
    """Configuration for the audit runner."""

    splits: tuple[SplitName, ...] = ("train", "val", "test")

    # Entry limits per split (None = read all)
    max_episodes_per_split: int | None = None

    # Which categorical keys to audit (present at top-level in manifest entry)
    categorical_keys: tuple[str, ...] = ("family", "task")

    # Which numeric keys to summarize. These are resolved from:
    # - entry[size_key] (top-level)
    # - entry["size"][size_key] (nested)
    numeric_size_keys: tuple[str, ...] = ("N", "n", "num_nodes", "E", "m", "num_edges")

    # Primary size key used for binning/bucketization. If None, inferred.
    primary_size_key: str | None = None

    # If bins are not provided, they are inferred from observed values.
    bins: tuple[BinSpec, ...] = ()

    # Bucketization keys ("bags"): any of "split", "family", "task", "size_bin", ...
    bucket_keys: tuple[str, ...] = ("split", "family", "task", "size_bin")

    # Optional bootstrap; set to 0 to disable.
    bootstrap_reps: int = 0
    bootstrap_alpha: float = 0.05
    bootstrap_seed: int = 0

    # Plot generation
    plots: bool = True

    # If True, include a "missingness" section that counts missing keys.
    report_missingness: bool = True


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:  # pragma: no cover
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected dict JSON objects in {path}:{line_no}")
            yield obj


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


# --------------------------------------------------------------------------------------
# Entry normalization
# --------------------------------------------------------------------------------------


def _get_size_dict(e: Mapping[str, Any]) -> Mapping[str, Any]:
    size = e.get("size", {})
    return size if isinstance(size, Mapping) else {}


def _get_num(e: Mapping[str, Any], key: str) -> float | None:
    # Prefer nested size dict for size-related keys.
    size = _get_size_dict(e)
    v = size.get(key, e.get(key, None))
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    # Some configs store sizes as strings (rare). Try parse.
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)
        except Exception:
            return None
    return None


def _infer_primary_size_key(entries: Sequence[Mapping[str, Any]], candidates: Sequence[str]) -> str | None:
    """
    Heuristic: pick the first candidate key that appears as numeric in ≥50% of entries.
    """
    if not entries:
        return None
    n = len(entries)
    for k in candidates:
        ok = 0
        for e in entries:
            if _get_num(e, k) is not None:
                ok += 1
        if ok >= max(1, int(0.5 * n)):
            return k
    return None


def _infer_bins_from_values(values: Sequence[float]) -> tuple[float, ...]:
    """
    Heuristic binning:
      - For small integers (like N), create power-of-two-ish edges.
      - Otherwise, use quantile-based edges with de-duplication.
    """
    if not values:
        return (0.0, 1.0)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)

    vmax = float(np.max(v))
    vmin = float(np.min(v))

    # If looks like counts (nonnegative, mostly integers), use doubling bins.
    frac_int = float(np.mean(np.isclose(v, np.round(v))))
    if vmin >= 0 and frac_int >= 0.9 and vmax <= 2.0**20:
        edges = [0.0]
        # Start at 8 to avoid tiny bins; include vmax.
        x = 8.0
        while x < vmax:
            edges.append(x)
            x *= 2.0
        edges.append(max(vmax + 1.0, edges[-1] + 1.0))
        # Ensure strictly increasing
        out = [edges[0]]
        for e in edges[1:]:
            if e > out[-1]:
                out.append(e)
        if len(out) < 2:
            out = [0.0, max(1.0, vmax + 1.0)]
        return tuple(out)

    # Quantile bins
    qs = np.linspace(0.0, 1.0, num=9)  # 8 bins
    edges = np.quantile(v, qs).astype(float).tolist()
    # Make strictly increasing by tiny eps increments where necessary
    out = [edges[0]]
    eps = max(1e-12, 1e-9 * (abs(out[0]) + 1.0))
    for e in edges[1:]:
        e2 = float(e)
        if e2 <= out[-1]:
            e2 = out[-1] + eps
        out.append(e2)
    return tuple(out)


def _bin_label(edges: Sequence[float], i: int) -> str:
    lo = edges[i]
    hi = edges[i + 1]

    def _fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.3g}"

    return f"[{_fmt(lo)},{_fmt(hi)})"


def _assign_bins(values: Sequence[float | None], spec: BinSpec) -> list[str | None]:
    edges = np.asarray(spec.edges, dtype=float)
    labels = spec.labels if spec.labels is not None else tuple(_bin_label(spec.edges, i) for i in range(len(spec.edges) - 1))
    out: list[str | None] = []
    for v in values:
        if v is None or not np.isfinite(v):
            out.append(None)
            continue
        bi = int(np.digitize([v], edges, right=False)[0]) - 1
        if bi < 0:
            bi = 0
        if bi >= len(labels):
            bi = len(labels) - 1
        out.append(labels[bi])
    return out


# --------------------------------------------------------------------------------------
# Stats + tables
# --------------------------------------------------------------------------------------


def _count_by(entries: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in entries:
        v = e.get(key, None)
        if v is None:
            continue
        s = str(v)
        out[s] = out.get(s, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _missingness(entries: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {k: 0 for k in keys}
    for e in entries:
        for k in keys:
            if k not in e or e.get(k, None) is None:
                out[k] += 1
    return out


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    qs = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    qv = np.quantile(v, qs)
    out = {
        "n": float(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(qv[0]),
        "p05": float(qv[1]),
        "p25": float(qv[2]),
        "p50": float(qv[3]),
        "p75": float(qv[4]),
        "p95": float(qv[5]),
        "max": float(qv[6]),
    }
    return out


def _cross_tab(entries: Sequence[Mapping[str, Any]], row_key: str, col_key: str) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for e in entries:
        r = e.get(row_key, None)
        c = e.get(col_key, None)
        if r is None or c is None:
            continue
        rs = str(r)
        cs = str(c)
        d = rows.setdefault(rs, {})
        d[cs] = d.get(cs, 0) + 1

    def row_total(item: tuple[str, dict[str, int]]) -> tuple[int, str]:
        k, d = item
        return (-sum(d.values()), k)

    out: dict[str, dict[str, int]] = {}
    for rk, rd in sorted(rows.items(), key=row_total):
        out[rk] = dict(sorted(rd.items(), key=lambda kv: (-kv[1], kv[0])))
    return out


def _bootstrap_category_ci(
    categories: Sequence[str],
    reps: int,
    alpha: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if reps <= 0 or not categories:
        return {}

    cats = np.asarray([str(c) for c in categories], dtype=object)
    uniq = sorted(set(cats.tolist()))
    n = cats.size
    rng = np.random.default_rng(seed)

    mapping = {c: i for i, c in enumerate(uniq)}
    x = np.asarray([mapping[c] for c in cats.tolist()], dtype=np.int64)
    k = len(uniq)

    props = np.empty((reps, k), dtype=float)
    for r in range(reps):
        samp = rng.integers(0, n, size=n)
        xs = x[samp]
        counts = np.bincount(xs, minlength=k).astype(float)
        props[r] = counts / float(n)

    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    lo = np.quantile(props, lo_q, axis=0)
    hi = np.quantile(props, hi_q, axis=0)

    return {c: (float(lo[i]), float(hi[i])) for c, i in mapping.items()}


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def _ensure_matplotlib() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plots=True")


def _plot_bar(
    outpath: Path,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    xlabel: str = "",
    ylabel: str = "count",
    rotate: int = 45,
) -> None:
    _ensure_matplotlib()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(max(6.0, 0.35 * len(labels)), 4.2))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(labels)), values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_hist(
    outpath: Path,
    title: str,
    values: Sequence[float],
    edges: Sequence[float] | None = None,
    xlabel: str = "",
) -> None:
    _ensure_matplotlib()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)
    if edges is None:
        ax.hist(v, bins="auto")
    else:
        ax.hist(v, bins=np.asarray(edges, dtype=float))
    ax.set_title(title)
    ax.set_ylabel("count")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_heatmap(
    outpath: Path,
    title: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    mat: np.ndarray,
) -> None:
    _ensure_matplotlib()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(max(6.0, 0.35 * len(col_labels)), max(4.5, 0.35 * len(row_labels))))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Core runner
# --------------------------------------------------------------------------------------


def load_manifest_entries(
    manifest_dir: str | Path,
    *,
    splits: Sequence[SplitName] = ("train", "val", "test"),
    max_episodes_per_split: int | None = None,
) -> dict[SplitName, list[dict[str, Any]]]:
    md = Path(manifest_dir)
    idx_path = md / "index.json"
    if not idx_path.is_file():
        raise FileNotFoundError(f"Missing index.json in manifest_dir={md}")

    idx = _read_json(idx_path)
    if not isinstance(idx, dict):
        raise ValueError("index.json must be a JSON object/dict")

    out: dict[SplitName, list[dict[str, Any]]] = {}
    splits_meta = idx.get("splits", {})
    for sp in splits:
        sp_meta = splits_meta.get(sp, {}) if isinstance(splits_meta, dict) else {}
        rel = sp_meta.get("file", None)
        if isinstance(rel, str) and rel.strip():
            path = (md / rel).resolve()
        else:
            path = md / "splits" / f"{sp}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file for {sp}: {path}")
        entries: list[dict[str, Any]] = []
        for e in _iter_jsonl(path):
            entries.append(e)
            if max_episodes_per_split is not None and len(entries) >= int(max_episodes_per_split):
                break
        out[sp] = entries

    return out


def run_dataset_audit(
    manifest_dir: str | Path,
    outdir: str | Path,
    *,
    config: AuditConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AuditConfig()
    md = Path(manifest_dir)
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)

    per_split = load_manifest_entries(
        md,
        splits=cfg.splits,
        max_episodes_per_split=cfg.max_episodes_per_split,
    )

    all_entries: list[Mapping[str, Any]] = []
    for sp in cfg.splits:
        all_entries.extend(per_split.get(sp, []))

    primary = cfg.primary_size_key or _infer_primary_size_key(all_entries, cfg.numeric_size_keys)
    bins: dict[str, BinSpec] = {b.key: b for b in cfg.bins}

    if primary is not None and primary not in bins:
        values = [_get_num(e, primary) for e in all_entries]
        vv = [float(x) for x in values if x is not None and np.isfinite(x)]
        edges = _infer_bins_from_values(vv)
        bins[primary] = BinSpec(key=primary, edges=edges)

    derived_per_split: dict[SplitName, list[dict[str, Any]]] = {}
    if primary is not None and primary in bins:
        spec = bins[primary]
        for sp, entries in per_split.items():
            vals = [_get_num(e, primary) for e in entries]
            lab = _assign_bins(vals, spec)
            out_entries: list[dict[str, Any]] = []
            for e, b in zip(entries, lab, strict=False):
                d = dict(e)
                d["size_bin"] = b
                d["_primary_size_key"] = primary
                out_entries.append(d)
            derived_per_split[sp] = out_entries
    else:
        for sp, entries in per_split.items():
            out_entries = [dict(e) for e in entries]
            for d in out_entries:
                d["size_bin"] = None
                d["_primary_size_key"] = primary
            derived_per_split[sp] = out_entries

    split_counts = {sp: len(derived_per_split.get(sp, [])) for sp in cfg.splits}
    total = sum(split_counts.values())

    audit: dict[str, Any] = {
        "manifest_dir": str(md),
        "outdir": str(od),
        "splits": list(cfg.splits),
        "counts": {
            "total": int(total),
            "by_split": {k: int(v) for k, v in split_counts.items()},
        },
        "primary_size_key": primary,
        "bins": {k: {"edges": list(v.edges), "labels": list(v.labels) if v.labels is not None else None} for k, v in bins.items()},
        "categorical": {},
        "numeric": {},
        "bucket_table": {},
    }

    if cfg.report_missingness:
        miss_keys = list(cfg.categorical_keys) + ["size"]
        audit["missingness"] = {sp: _missingness(derived_per_split[sp], miss_keys) for sp in cfg.splits}

    for key in cfg.categorical_keys:
        by_split: dict[str, Any] = {}
        for sp in cfg.splits:
            entries = derived_per_split[sp]
            counts = _count_by(entries, key)
            by_split[sp] = {"counts": counts}
            if cfg.bootstrap_reps > 0:
                cats = [str(e[key]) for e in entries if key in e and e[key] is not None]
                ci = _bootstrap_category_ci(
                    cats,
                    reps=int(cfg.bootstrap_reps),
                    alpha=float(cfg.bootstrap_alpha),
                    seed=int(cfg.bootstrap_seed) + (hash(sp) & 0xFFFF),
                )
                if ci:
                    by_split[sp]["bootstrap_ci"] = {k: [float(ci[k][0]), float(ci[k][1])] for k in sorted(ci.keys())}
        audit["categorical"][key] = by_split

    numeric_keys_found: set[str] = set()
    for e in all_entries:
        for k in cfg.numeric_size_keys:
            if _get_num(e, k) is not None:
                numeric_keys_found.add(k)
    if primary is not None:
        numeric_keys_found.add(primary)

    for nk in sorted(numeric_keys_found):
        by_split: dict[str, Any] = {}
        for sp in cfg.splits:
            vals = [_get_num(e, nk) for e in derived_per_split[sp]]
            vv = [float(x) for x in vals if x is not None and np.isfinite(x)]
            by_split[sp] = _numeric_summary(vv)
        audit["numeric"][nk] = by_split

    if "family" in cfg.categorical_keys and "task" in cfg.categorical_keys:
        audit["crosstab_family_x_task"] = _cross_tab(all_entries, "family", "task")

    bucket_rows: list[tuple[Any, ...]] = []
    bucket_counts: dict[tuple[str, ...], int] = {}
    keys = list(cfg.bucket_keys)

    for sp in cfg.splits:
        for e in derived_per_split[sp]:
            row: list[str] = []
            for k in keys:
                if k == "split":
                    row.append(str(sp))
                else:
                    v = e.get(k, None)
                    row.append("" if v is None else str(v))
            tup = tuple(row)
            bucket_counts[tup] = bucket_counts.get(tup, 0) + 1

    for tup, c in sorted(bucket_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        bucket_rows.append(tuple(list(tup) + [c]))

    bucket_csv = od / "bucket_table.csv"
    _write_csv(bucket_csv, keys + ["count"], bucket_rows)

    audit["bucket_table"] = {
        "keys": keys,
        "rows": [list(r) for r in bucket_rows[:20000]],
        "rows_truncated": len(bucket_rows) > 20000,
        "n_rows": int(len(bucket_rows)),
    }

    size_csv_rows: list[list[Any]] = []
    size_header = ["key", "split", "n", "mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "max"]
    for nk, per in audit["numeric"].items():
        for sp in cfg.splits:
            d = per.get(sp, {})
            if not d:
                continue
            size_csv_rows.append([nk, sp] + [d.get(h, "") for h in size_header[2:]])
    _write_csv(od / "size_key_summary.csv", size_header, size_csv_rows)

    if cfg.plots:
        figdir = od / "figures"
        figdir.mkdir(parents=True, exist_ok=True)

        _plot_bar(
            figdir / "split_counts.png",
            title="Episodes per split",
            labels=list(cfg.splits),
            values=[split_counts[sp] for sp in cfg.splits],
            xlabel="split",
            ylabel="count",
            rotate=0,
        )

        for key in cfg.categorical_keys:
            cats: set[str] = set()
            for sp in cfg.splits:
                cats.update(audit["categorical"][key][sp]["counts"].keys())
            cats_sorted = sorted(cats)

            mat = np.zeros((len(cfg.splits), len(cats_sorted)), dtype=float)
            for i, sp in enumerate(cfg.splits):
                counts = audit["categorical"][key][sp]["counts"]
                for j, c in enumerate(cats_sorted):
                    mat[i, j] = float(counts.get(c, 0))

            totals = mat.sum(axis=0)
            if len(cats_sorted) > 25:
                top_idx = np.argsort(-totals)[:25]
                cats_sorted = [cats_sorted[i] for i in top_idx]
                mat = mat[:, top_idx]

            _plot_heatmap(
                figdir / f"{key}_counts_by_split.png",
                title=f"{key} counts by split",
                row_labels=list(cfg.splits),
                col_labels=cats_sorted,
                mat=mat,
            )

        if primary is not None and primary in bins:
            all_vals = []
            for sp in cfg.splits:
                all_vals.extend([_get_num(e, primary) for e in derived_per_split[sp]])
            vv = [float(x) for x in all_vals if x is not None and np.isfinite(x)]
            _plot_hist(
                figdir / f"size_hist_{primary}.png",
                title=f"Histogram of {primary} (all splits)",
                values=vv,
                edges=bins[primary].edges,
                xlabel=primary,
            )

        if "family" in keys and "size_bin" in keys:
            fams = sorted({str(e.get("family", "")) for e in all_entries if e.get("family", None) is not None})
            bins_lab = sorted({str(e.get("size_bin", "")) for e in all_entries if e.get("size_bin", None) is not None})
            if fams and bins_lab:
                mat = np.zeros((len(fams), len(bins_lab)), dtype=float)
                fam_index = {f: i for i, f in enumerate(fams)}
                bin_index = {b: i for i, b in enumerate(bins_lab)}
                for e in all_entries:
                    f = e.get("family", None)
                    b = e.get("size_bin", None)
                    if f is None or b is None:
                        continue
                    mat[fam_index[str(f)], bin_index[str(b)]] += 1.0
                _plot_heatmap(
                    figdir / "bucket_heatmap_family_x_sizebin.png",
                    title="Bucket counts: family × size_bin (all splits)",
                    row_labels=fams,
                    col_labels=bins_lab,
                    mat=mat,
                )

    _write_json(od / "audit.json", audit)
    return audit


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_bins(arg: str) -> BinSpec:
    s = arg.strip()
    if ":" not in s:
        raise ValueError("Bin spec must be KEY:edges or KEY:auto")
    key, rest = s.split(":", 1)
    key = key.strip()
    rest = rest.strip()
    if not key:
        raise ValueError("Bin spec key is empty")

    if rest.lower() == "auto":
        return BinSpec(key=key, edges=(0.0, 1.0))

    parts = [p.strip() for p in rest.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("Bin spec must provide >=2 edges")
    edges: list[float] = [float(p) for p in parts]
    return BinSpec(key=key, edges=tuple(edges))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Audit evaluation manifest splits/buckets for defendable reporting.")
    ap.add_argument("--manifest-dir", type=str, required=True, help="Path to eval manifest dir (contains index.json).")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for audit artifacts.")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Splits to audit.")
    ap.add_argument("--max-per-split", type=int, default=None, help="Limit entries per split (for quick audits).")
    ap.add_argument("--primary-size-key", type=str, default=None, help="Primary numeric size key for binning (e.g., N).")
    ap.add_argument("--bin", action="append", default=None, help="Bin spec KEY:0,16,32 or KEY:auto (repeatable).")
    ap.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    ap.add_argument("--bootstrap-reps", type=int, default=0, help="Bootstrap reps for category CI (0 disables).")
    ap.add_argument("--bootstrap-alpha", type=float, default=0.05, help="Bootstrap alpha (e.g., 0.05 for 95%% CI).")
    ap.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed.")
    ap.add_argument("--bucket-keys", nargs="*", default=["split", "family", "task", "size_bin"], help="Bucketization keys.")
    ap.add_argument("--log-level", type=str, default="INFO", help="Logging level.")

    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    splits: list[SplitName] = []
    for s in args.splits:
        ss = str(s).strip()
        if ss not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {ss}")
        splits.append(ss)  # type: ignore[arg-type]

    bins: list[BinSpec] = []
    for b in (args.bin or []):
        bins.append(_parse_bins(b))

    cfg = AuditConfig(
        splits=tuple(splits),
        max_episodes_per_split=args.max_per_split,
        primary_size_key=args.primary_size_key,
        bins=tuple(bins),
        plots=(not bool(args.no_plots)),
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_alpha=float(args.bootstrap_alpha),
        bootstrap_seed=int(args.bootstrap_seed),
        bucket_keys=tuple(str(k) for k in args.bucket_keys),
    )

    run_dataset_audit(args.manifest_dir, args.outdir, config=cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

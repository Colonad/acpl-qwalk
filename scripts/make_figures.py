#!/usr/bin/env python3
# scripts/make_figures.py
from __future__ import annotations

"""
scripts/make_figures.py

Research-ready figure bundling + stable naming for ACPL evaluation outputs
produced by scripts/eval.py (and optionally embedding artifacts).

Primary goal
------------
Turn scattered per-run plots into a clean, deterministic "figures/" bundle with:
- stable names
- run provenance (run_id, meta hash, suite, checkpoint)
- a manifest (JSON + CSV)
- a human index (Markdown) and optional LaTeX include snippet

This is the “make_figures.py” companion to collect_results.py.

Expected input layouts
----------------------
Evaluation directories produced by scripts/eval.py typically look like:

<evaldir>/
  meta.json
  summary.json
  raw/<cond>/...
  figs/
    Pt__<cond_safe>.png
    tv__<cond_safe>.png
    robustness__<cond_safe>.png      (optional)
    ... any other images ...

Optionally, you may also have “first-class” artifacts:
<evaldir>/artifacts/embeddings/<cond>/embeddings.pca2d.png
or
<evaldir>/eval/artifacts/embeddings/<cond>/embeddings.pca2d.png

This script does not require a rigid artifact layout; it discovers what exists.

Outputs
-------
<outdir>/figures/...
<outdir>/manifest.figures.json
<outdir>/manifest.figures.csv
<outdir>/FIGURES.md
<outdir>/FIGURES.tex   (optional snippet with \\includegraphics lines)

Usage
-----
# Scan a tree:
python scripts/make_figures.py --root eval --outdir eval/_bundle

# Explicit eval dirs:
python scripts/make_figures.py --evaldirs eval/tmp eval/expA --outdir eval/_bundle

# Use results.json from collect_results.py:
python scripts/make_figures.py --results-json eval/_collected/results.json --outdir eval/_bundle

# Link instead of copy:
python scripts/make_figures.py --root eval --outdir eval/_bundle --mode symlink

Design choices for defendability
--------------------------------
- Deterministic traversal and naming
- File hashing to deduplicate and detect collisions
- A manifest that can be committed and diffed (stable order)
- Best-effort parsing of condition names using the eval directory’s known conditions
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

log = logging.getLogger("make_figures")


# =============================================================================
# Small utilities
# =============================================================================


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _as_path(p: str | os.PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(_read_text(path))
    except Exception:
        return None


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )


def _sanitize(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    return s or "x"


def _relpath(from_dir: Path, to_path: Path) -> str:
    try:
        return str(to_path.relative_to(from_dir))
    except Exception:
        try:
            return os.path.relpath(str(to_path), str(from_dir))
        except Exception:
            return str(to_path)


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _blake2b_hex(data: bytes, *, digest_size: int = 16) -> str:
    h = hashlib.blake2b(data, digest_size=digest_size)
    return h.hexdigest()


def _stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _git_info(repo_root: Path) -> dict[str, Any]:
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
    return {"head": head, "branch": branch, "describe": describe, "dirty": bool(status)}


def _looks_like_evaldir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "meta.json").exists():
        return False
    # Strong hints:
    if (path / "summary.json").exists() or (path / "raw").exists() or (path / "figs").exists():
        return True
    return True


def find_evaldirs(
    *,
    root: Path,
    max_depth: int | None,
    follow_symlinks: bool,
    exclude_regex: str | None,
) -> list[Path]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    rx = re.compile(exclude_regex) if exclude_regex else None

    out: list[Path] = []
    root_parts = len(root.parts)

    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=follow_symlinks):
        d = Path(dirpath)

        # Depth pruning
        if max_depth is not None:
            depth = len(d.parts) - root_parts
            if depth > max_depth:
                dirnames[:] = []
                continue

        if rx and rx.search(str(d)):
            dirnames[:] = []
            continue

        if "meta.json" in filenames and _looks_like_evaldir(d):
            out.append(d.resolve())
            # raw/ is huge and not a source of nested evaldirs; prune it
            if "raw" in dirnames:
                dirnames.remove("raw")

    uniq: dict[str, Path] = {}
    for p in out:
        uniq[str(p)] = p
    return sorted(uniq.values(), key=lambda p: str(p))


def _compute_evaldir_hash(evaldir: Path) -> str:
    """
    Hash used for collision resistance. Uses meta.json + summary.json if present.
    """
    meta_p = evaldir / "meta.json"
    data = meta_p.read_bytes()
    summ_p = evaldir / "summary.json"
    if summ_p.exists():
        data += b"\n" + summ_p.read_bytes()
    return _blake2b_hex(data, digest_size=16)


def _make_run_id(evaldir: Path, meta_json: Mapping[str, Any], meta_hash: str) -> str:
    suite = str(meta_json.get("suite") or "suite")
    policy = str(meta_json.get("policy") or "policy")
    ckpt = str(meta_json.get("checkpoint") or meta_json.get("ckpt") or "ckpt")
    ckpt_base = Path(ckpt).name if ckpt else "ckpt"
    episodes = meta_json.get("episodes", None)
    seeds = meta_json.get("seeds", [])
    try:
        seeds_s = ",".join(str(int(x)) for x in (seeds or []))
    except Exception:
        seeds_s = "seeds"
    eps_s = str(int(episodes)) if episodes is not None and _is_number(episodes) else "E"
    return _sanitize(f"{suite}__{policy}__{ckpt_base}__{eps_s}__{seeds_s}__{meta_hash[:8]}")


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


# =============================================================================
# Optional: use reporting module if available for better condition mapping
# =============================================================================

_HAVE_REPORTING = False
try:
    from acpl.eval.reporting import load_eval_run  # type: ignore

    _HAVE_REPORTING = True
except Exception:
    _HAVE_REPORTING = False
    load_eval_run = None  # type: ignore


def _load_conditions(evaldir: Path) -> list[str]:
    """
    Discover condition names to build safe->cond mapping.
    Prefer summary.json, otherwise raw/ directory names, otherwise empty.
    """
    # 1) reporting load (best)
    if _HAVE_REPORTING and load_eval_run is not None:
        try:
            run = load_eval_run(str(evaldir))
            conds = list(getattr(run, "conditions", {}).keys())
            conds = [str(c) for c in conds]
            conds.sort()
            return conds
        except Exception:
            pass

    # 2) summary.json
    summ = _safe_read_json(evaldir / "summary.json")
    if isinstance(summ, Mapping):
        conds = summ.get("conditions", None)
        if isinstance(conds, Mapping):
            out = [str(k) for k in conds.keys()]
            out.sort()
            return out

    # 3) raw dirs
    raw = evaldir / "raw"
    if raw.exists() and raw.is_dir():
        out = [p.name for p in raw.iterdir() if p.is_dir()]
        out.sort()
        return out

    return []


def _safe_to_cond_map(conditions: Sequence[str]) -> dict[str, str]:
    """
    Map sanitize(cond) -> cond, collision-aware by storing best effort.
    """
    out: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for c in conditions:
        s = _sanitize(c)
        if s in out and out[s] != c:
            collisions.setdefault(s, [out[s]]).append(c)
        else:
            out[s] = c

    # If collisions exist, keep deterministic choice but record in logs.
    for s, cs in collisions.items():
        cs2 = sorted(set(cs))
        log.warning("Condition sanitize collision for '%s': %s (keeping '%s')", s, cs2, out.get(s))
    return out


# =============================================================================
# Figure discovery + parsing
# =============================================================================


_SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}


@dataclass(frozen=True)
class ParsedFigureName:
    kind: str             # e.g. "Pt", "tv", "robustness", "embeddings_pca2d"
    cond_safe: str        # sanitized condition token found in filename, or "_ALL_"
    tags: tuple[str, ...] # extra tokens after condition
    stem: str             # original stem


def _parse_figure_stem(stem: str) -> ParsedFigureName:
    """
    Parse a filename stem into:
      kind__cond__tag1__tag2...

    If no "__" exists, kind="fig", cond_safe="_ALL_", tags=(stem,).
    """
    if "__" not in stem:
        return ParsedFigureName(kind="fig", cond_safe="_ALL_", tags=(stem,), stem=stem)

    parts = stem.split("__")
    kind = parts[0] or "fig"
    if len(parts) == 1:
        return ParsedFigureName(kind=kind, cond_safe="_ALL_", tags=(), stem=stem)

    cond_safe = parts[1] or "_ALL_"

    # Normalize common "all-conditions" sentinel used by some writers
    if cond_safe.casefold() == "all":
        cond_safe = "_ALL_"


    tags = tuple(p for p in parts[2:] if p)
    return ParsedFigureName(kind=kind, cond_safe=cond_safe, tags=tags, stem=stem)


def _discover_eval_figs(evaldir: Path) -> list[Path]:
    """
    Discover figures under evaldir/figs.
    """
    figdir = evaldir / "figs"
    if not figdir.exists() or not figdir.is_dir():
        return []
    out = [p for p in figdir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS]
    out.sort(key=lambda p: p.name)
    return out


def _discover_embedding_figs(evaldir: Path) -> list[Path]:
    """
    Best-effort discovery of embedding PCA plots and related images.
    Supports:
      evaldir/artifacts/embeddings/<cond>/*.png
      evaldir/eval/artifacts/embeddings/<cond>/*.png
    Also includes any .png/.pdf/.jpg under embeddings directories.
    """
    roots = [
        evaldir / "artifacts" / "embeddings",
        evaldir / "eval" / "artifacts" / "embeddings",
    ]
    out: list[Path] = []
    for r in roots:
        if not r.exists() or not r.is_dir():
            continue
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS:
                out.append(p)
    out = sorted(set(out), key=lambda p: str(p))
    return out


def _infer_kind_for_embedding_path(p: Path) -> str:
    """
    Convert embedding artifact filenames into a stable kind label.
    """
    s = p.name.lower()
    if "pca" in s and ("2d" in s or "pca2d" in s):
        return "embeddings_pca2d"
    if "stats" in s:
        return "embeddings_stats"
    if "mean" in s:
        return "embeddings_mean"
    if "embeddings" in s:
        return "embeddings"
    return "embedding_artifact"


def _infer_cond_from_embedding_path(p: Path, safe_map: Mapping[str, str]) -> tuple[str, str]:
    """
    For embeddings artifacts, condition is often a directory component: .../embeddings/<cond>/...
    Return (cond_name, cond_safe_guess).
    """
    parts = list(p.parts)
    # Find "embeddings" component and take next as condition dir if present
    try:
        i = parts.index("embeddings")
        if i + 1 < len(parts):
            cond_dir = parts[i + 1]
            cond_safe = _sanitize(cond_dir)
            return (safe_map.get(cond_safe, cond_dir), cond_safe)
    except Exception:
        pass

    # fallback: unknown
    return ("_ALL_", "_ALL_")


# =============================================================================
# Bundling layout + file operations
# =============================================================================


@dataclass
class FigureEntry:
    run_id: str
    evaldir: str
    suite: str | None
    checkpoint: str | None
    meta_hash: str
    source_path: str
    source_rel: str
    dest_path: str
    dest_rel: str
    cond: str
    cond_safe: str
    kind: str
    tags: list[str] = field(default_factory=list)
    ext: str = ""
    sha256: str = ""
    bytes: int = 0
    mtime_utc: str = ""


def _file_mtime_utc(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return _dt.datetime.utcfromtimestamp(ts).replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return ""


def _copy_mode_op(src: Path, dst: Path, *, mode: str) -> None:
    """
    mode: copy | hardlink | symlink
    """
    _ensure_dir(dst.parent)

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        # Hardlink requires same filesystem
        try:
            if dst.exists():
                dst.unlink()
            os.link(str(src), str(dst))
            return
        except Exception as e:
            raise RuntimeError(f"hardlink failed ({src} -> {dst}): {e}") from e

    if mode == "symlink":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            # Use relative symlinks for portability inside bundle dir
            rel = os.path.relpath(str(src), str(dst.parent))
            os.symlink(rel, str(dst))
            return
        except Exception as e:
            raise RuntimeError(f"symlink failed ({src} -> {dst}): {e}") from e

    raise ValueError(f"Unknown mode='{mode}'")


def _resolve_collision(dst: Path, src_hash: str) -> Path:
    """
    If dst exists:
      - if identical content (hash), reuse dst
      - else append suffix __<hash8> (deterministic)
    """
    if not dst.exists():
        return dst
    try:
        h = _sha256_file(dst)
        if h == src_hash:
            return dst
    except Exception:
        pass

    base = dst.with_suffix("")  # drop ext
    ext = dst.suffix
    alt = Path(str(base) + f"__{src_hash[:8]}" + ext)
    if not alt.exists():
        return alt

    # Extremely rare: collision even with hash suffix; add numeric tail
    for i in range(2, 1000):
        cand = Path(str(base) + f"__{src_hash[:8]}__{i}" + ext)
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not resolve filename collision for: {dst}")


def _default_layout_path(
    *,
    out_fig_root: Path,
    suite: str | None,
    run_id: str,
    cond: str,
    kind: str,
    tags: Sequence[str],
    ext: str,
    layout: str,
) -> Path:
    """
    layout:
      - hier: figures/<suite>/<run_id>/<cond>/<kind>__<cond>__[tags].ext
      - flat: figures/<suite>/run__<run_id>__<kind>__<cond>__[tags].ext
      - by_kind: figures/<suite>/<kind>/run__<run_id>__<cond>__[tags].ext
    """
    suite_dir = _sanitize(suite or "suite")
    cond_safe = _sanitize(cond)

    tag_part = ""
    if tags:
        tag_part = "__" + "__".join(_sanitize(t) for t in tags if t)

    if layout == "hier":
        fname = f"{_sanitize(kind)}__{cond_safe}{tag_part}{ext}"
        return out_fig_root / suite_dir / _sanitize(run_id) / cond_safe / fname

    if layout == "by_kind":
        fname = f"run__{_sanitize(run_id)}__{cond_safe}{tag_part}{ext}"
        return out_fig_root / suite_dir / _sanitize(kind) / fname

    if layout == "flat":
        fname = f"run__{_sanitize(run_id)}__{_sanitize(kind)}__{cond_safe}{tag_part}{ext}"
        return out_fig_root / suite_dir / fname

    raise ValueError(f"Unknown layout='{layout}'")


# =============================================================================
# Index writers (Markdown + LaTeX)
# =============================================================================


def _group_entries(entries: Sequence[FigureEntry]) -> dict[str, dict[str, list[FigureEntry]]]:
    """
    Group by suite -> run_id.
    """
    out: dict[str, dict[str, list[FigureEntry]]] = {}
    for e in entries:
        suite = _sanitize(e.suite or "suite")
        out.setdefault(suite, {}).setdefault(e.run_id, []).append(e)
    # deterministic order
    for suite in out:
        for run_id in out[suite]:
            out[suite][run_id].sort(key=lambda x: (x.cond_safe, x.kind, x.dest_rel))
    return out


def write_figures_markdown_index(out_path: Path, *, bundle_root: Path, entries: Sequence[FigureEntry]) -> None:
    groups = _group_entries(entries)

    lines: list[str] = []
    lines.append("# Figures Bundle Index")
    lines.append("")
    lines.append(f"- Generated: `{_utc_now_iso()}`")
    lines.append(f"- Bundle root: `{bundle_root}`")
    lines.append(f"- Figures entries: `{len(entries)}`")
    lines.append("")

    for suite in sorted(groups.keys()):
        lines.append(f"## Suite: `{suite}`")
        lines.append("")
        for run_id in sorted(groups[suite].keys()):
            run_entries = groups[suite][run_id]
            # run meta summary from first entry (they share)
            e0 = run_entries[0]
            lines.append(f"### Run: `{run_id}`")
            lines.append("")
            if e0.checkpoint:
                lines.append(f"- Checkpoint: `{e0.checkpoint}`")
            lines.append(f"- Evaldir: `{e0.evaldir}`")
            lines.append(f"- Meta hash: `{e0.meta_hash}`")
            lines.append("")

            # show per-condition sections
            conds: dict[str, list[FigureEntry]] = {}
            for e in run_entries:
                conds.setdefault(e.cond, []).append(e)
            for cond in sorted(conds.keys(), key=lambda x: (_sanitize(x) != "ckpt_policy", _sanitize(x))):
                lines.append(f"#### Condition: `{cond}`")
                lines.append("")
                # inline images for png/jpg; link for pdf/svg
                for e in conds[cond]:
                    rel = e.dest_rel
                    if e.ext.lower() in (".png", ".jpg", ".jpeg"):
                        lines.append(f"- **{e.kind}** `{Path(rel).name}`")
                        lines.append("")
                        lines.append(f"![{e.kind}]({rel})")
                        lines.append("")
                    else:
                        lines.append(f"- **{e.kind}** `{Path(rel).name}` — `{rel}`")
                lines.append("")
        lines.append("")

    _write_text(out_path, "\n".join(lines).rstrip() + "\n")


def write_figures_latex_snippet(out_path: Path, *, entries: Sequence[FigureEntry], max_per_run: int = 8) -> None:
    """
    Write a LaTeX snippet with \\includegraphics lines, grouped by suite/run/cond.
    This is a helper, not a full paper.
    """
    groups = _group_entries(entries)
    lines: list[str] = []
    lines.append("% Auto-generated by scripts/make_figures.py")
    lines.append(f"% Generated: { _utc_now_iso() }")
    lines.append("% Usage: \\input{FIGURES.tex}")
    lines.append("")

    for suite in sorted(groups.keys()):
        lines.append(f"% ===================== Suite: {suite} =====================")
        for run_id in sorted(groups[suite].keys()):
            run_entries = groups[suite][run_id]
            lines.append(f"% ---- Run: {run_id} ----")
            # limit to avoid exploding documents
            shown = 0
            for e in run_entries:
                if shown >= max_per_run:
                    break
                if e.ext.lower() not in (".png", ".jpg", ".jpeg", ".pdf"):
                    continue
                # Use forward slashes for LaTeX portability
                rel = e.dest_rel.replace("\\", "/")
                cap = f"{suite} / {run_id} / {e.cond} / {e.kind}"
                lines.append(r"\begin{figure}[t]")
                lines.append(r"\centering")
                lines.append(rf"\includegraphics[width=0.98\linewidth]{{{rel}}}")
                lines.append(rf"\caption{{{cap}}}")
                lines.append(rf"\label{{fig:{_sanitize(suite)}:{_sanitize(run_id)}:{_sanitize(e.cond)}:{_sanitize(e.kind)}}}")
                lines.append(r"\end{figure}")
                lines.append("")
                shown += 1
        lines.append("")

    _write_text(out_path, "\n".join(lines).rstrip() + "\n")





# =============================================================================
# Paper-mode: simple copy/rename into stable names for papers/slides
# =============================================================================

_PAPER_EXT_PREF = [".pdf", ".png", ".svg", ".jpg", ".jpeg"]


def _pick_existing_by_ext(stem_base: Path) -> Path | None:
    """
    Given a base path without extension (e.g., evaldir/figs/Pt__ckpt_policy),
    return the first existing file using preferred extensions.
    """
    for ext in _PAPER_EXT_PREF:
        p = stem_base.with_suffix(ext)
        if p.exists() and p.is_file():
            return p
    return None

def _pick_existing_by_ext_casefold(figdir: Path, stem: str) -> Path | None:
    """
    Case-insensitive variant of _pick_existing_by_ext, scoped to a directory.
    """
    figdir = figdir.resolve()
    direct = _pick_existing_by_ext(figdir / stem)
    if direct is not None:
        return direct

    if not figdir.exists() or not figdir.is_dir():
        return None

    stem_cf = stem.casefold()
    for p in figdir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in _SUPPORTED_EXTS:
            continue
        if p.stem.casefold() == stem_cf:
            return p
    return None

def _choose_base_condition(evaldir: Path, *, requested: str | None) -> str:
    """
    Determine the 'base' condition for stable paper names.
    Priority:
      1) --paper-cond if provided
      2) reporting base_condition() (if available)
      3) 'ckpt_policy' if present
      4) first condition (sorted)
      5) fallback '_ALL_'
    """
    conds = _load_conditions(evaldir)

    if requested:
        return str(requested)

    # reporting base_condition()
    if _HAVE_REPORTING and load_eval_run is not None:
        try:
            run = load_eval_run(str(evaldir))
            base = getattr(run, "base_condition", None)
            if callable(base):
                b = base()
                if b:
                    return str(b)
        except Exception:
            pass

    if "ckpt_policy" in conds:
        return "ckpt_policy"
    if conds:
        return sorted(conds)[0]
    return "_ALL_"


def _discover_mask_sensitivity_figs(evaldir: Path) -> list[Path]:
    """
    Best-effort discovery of substructure/mask-sensitivity plots.

    We look in common locations that scripts/run_mask_sensitivity.py should write:
      - evaldir/mask_sensitivity/figs/*
      - evaldir/artifacts/mask_sensitivity/*
      - evaldir/figs/*mask* (fallback)
    """
    roots = [
        evaldir / "mask_sensitivity",
        evaldir / "mask_sensitivity" / "figs",
        evaldir / "artifacts" / "mask_sensitivity",
        evaldir / "artifacts" / "masks",
        evaldir / "figs",
    ]
    out: list[Path] = []
    for r in roots:
        if not r.exists():
            continue
        if r.is_file() and r.suffix.lower() in _SUPPORTED_EXTS:
            out.append(r)
            continue
        if r.is_dir():
            for p in r.rglob("*"):
                if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS:
                    
                    name = p.name.lower()
                    stem = p.stem.lower()

                    is_mask_family = (
                        stem.startswith("mask__")
                        or stem.startswith("mask_sensitivity__")
                        or ("mask" in name)
                        or ("sensitivity" in name)
                    )

                    if is_mask_family:
                        out.append(p)

    # deterministic
    out = sorted(set(out), key=lambda p: str(p))
    return out


def _paper_dest_name(
    *,
    out_fig_root: Path,
    prefix: str,
    kind: str,
    ext: str,
    cond_safe: str | None = None,
) -> Path:
    """
    Stable filenames:
      base:    <prefix>__<kind>.<ext>
      others:  <prefix>__<kind>__<cond_safe>.<ext>
    """
    prefix_s = _sanitize(prefix)
    kind_s = _sanitize(kind)
    if cond_safe:
        return out_fig_root / f"{prefix_s}__{kind_s}__{_sanitize(cond_safe)}{ext}"
    return out_fig_root / f"{prefix_s}__{kind_s}{ext}"


def bundle_paper_figures(
    evaldir: Path,
    *,
    out_fig_root: Path,
    mode: str,
    include_embeddings: bool,
    include_ablations: bool,
    paper_prefix: str | None,
    paper_cond: str | None,
) -> tuple[list[FigureEntry], list[dict[str, Any]]]:
    """
    Copy/rename a SINGLE evaldir into stable paper names.
    """
    evaldir = evaldir.resolve()
    meta_p = evaldir / "meta.json"
    meta_json = _safe_read_json(meta_p) or {}
    suite = str(meta_json.get("suite") or "suite")
    checkpoint = meta_json.get("checkpoint", None)
    meta_hash = _compute_evaldir_hash(evaldir)
    run_id = _make_run_id(evaldir, meta_json, meta_hash)

    # Choose prefix
    prefix = paper_prefix if paper_prefix else suite

    # Conditions and base condition
    conditions = _load_conditions(evaldir)
    base_cond = _choose_base_condition(evaldir, requested=paper_cond)
    safe_map = _safe_to_cond_map(conditions)

    # Which conditions to emit
    cond_list: list[str] = [base_cond]
    if include_ablations:
        # include all other conditions deterministically
        for c in sorted(set(conditions)):
            if c != base_cond:
                cond_list.append(c)

    entries: list[FigureEntry] = []
    errors: list[dict[str, Any]] = []

    # Canonical eval plot kinds from eval.py outputs
    canonical_kinds = ["Pt", "tv", "robustness"]

    # --- A) Copy canonical eval figs (Pt/tv/robustness) ---
    for cond in cond_list:
        cond_safe = _sanitize(cond)
        # match eval naming patterns: <kind>__<safe>.<ext>
        for kind in canonical_kinds:
            src = _pick_existing_by_ext_casefold(evaldir / "figs", f"{kind}__{cond_safe}")
            if src is None:
                continue
            try:
                ext = src.suffix.lower()
                dst = _paper_dest_name(
                    out_fig_root=out_fig_root,
                    prefix=prefix,
                    kind=kind,
                    ext=ext,
                    cond_safe=(None if cond == base_cond else cond_safe),
                )
                sha = _sha256_file(src)
                dst = _resolve_collision(dst, sha)
                if not dst.exists():
                    _copy_mode_op(src, dst, mode=mode)

                entries.append(
                    FigureEntry(
                        run_id=run_id,
                        evaldir=str(evaldir),
                        suite=suite,
                        checkpoint=str(checkpoint) if checkpoint is not None else None,
                        meta_hash=meta_hash,
                        source_path=str(src),
                        source_rel=_relpath(evaldir, src),
                        dest_path=str(dst),
                        dest_rel=_relpath(out_fig_root.parent, dst),
                        cond=cond,
                        cond_safe=cond_safe,
                        kind=kind,
                        tags=[],
                        ext=ext,
                        sha256=sha,
                        bytes=int(src.stat().st_size),
                        mtime_utc=_file_mtime_utc(src),
                    )
                )
            except Exception as ex:
                errors.append({"evaldir": str(evaldir), "source": str(src), "error": f"{type(ex).__name__}: {ex}"})

    # --- B) Copy mask-sensitivity figs (from scripts/run_mask_sensitivity.py outputs) ---
    mask_paths = _discover_mask_sensitivity_figs(evaldir)
    # prefer figures that look like "mask_sensitivity__<cond_safe>.*" if present
    for cond in cond_list:
        cond_safe = _sanitize(cond)
        candidates = [p for p in mask_paths if f"__{cond_safe}".lower() in p.stem.lower()]
        if not candidates:
            # fallback: base-only, take any mask sensitivity figure(s)
            if cond != base_cond:
                continue
            candidates = [p for p in mask_paths]

        # deterministic: sort and take all candidates (paper can pick later)
        for src in sorted(set(candidates), key=lambda p: str(p)):
            try:
                ext = src.suffix.lower()
                kind = "mask_sensitivity"
                # If multiple mask plots exist, tag using remaining stem after kind/cond
                tags: list[str] = []
                stem_l = src.stem
                # best-effort parse: mask_sensitivity__COND__tag...
                if "__" in stem_l:
                    parts = stem_l.split("__")
                    # drop leading kind, cond
                    if len(parts) >= 3:
                        tags = [p for p in parts[2:] if p]

                # If there are tags, append them into the filename by using cond_safe slot (keeps stable base names)
                # base: prefix__mask_sensitivity.<ext>  (first/primary)
                # extras: prefix__mask_sensitivity__tag1__tag2.<ext>
                if tags and cond == base_cond:
                    dst = out_fig_root / f"{_sanitize(prefix)}__mask_sensitivity__{'__'.join(_sanitize(t) for t in tags)}{ext}"
                else:
                    dst = _paper_dest_name(
                        out_fig_root=out_fig_root,
                        prefix=prefix,
                        kind=kind,
                        ext=ext,
                        cond_safe=(None if cond == base_cond else cond_safe),
                    )

                sha = _sha256_file(src)
                dst = _resolve_collision(dst, sha)
                if not dst.exists():
                    _copy_mode_op(src, dst, mode=mode)

                entries.append(
                    FigureEntry(
                        run_id=run_id,
                        evaldir=str(evaldir),
                        suite=suite,
                        checkpoint=str(checkpoint) if checkpoint is not None else None,
                        meta_hash=meta_hash,
                        source_path=str(src),
                        source_rel=_relpath(evaldir, src),
                        dest_path=str(dst),
                        dest_rel=_relpath(out_fig_root.parent, dst),
                        cond=cond,
                        cond_safe=cond_safe,
                        kind="mask_sensitivity",
                        tags=tags,
                        ext=ext,
                        sha256=sha,
                        bytes=int(src.stat().st_size),
                        mtime_utc=_file_mtime_utc(src),
                    )
                )
            except Exception as ex:
                errors.append({"evaldir": str(evaldir), "source": str(src), "error": f"{type(ex).__name__}: {ex}"})

    # --- C) Optional: embeddings PCA plots into stable names ---
    if include_embeddings:
        emb_paths = _discover_embedding_figs(evaldir)
        for cond in cond_list:
            cond_safe = _sanitize(cond)
            # filter to this condition directory if possible
            cond_hits = []
            for p in emb_paths:
                c_name, c_safe_guess = _infer_cond_from_embedding_path(p, safe_map)
                if _sanitize(c_name) == cond_safe or _sanitize(c_safe_guess) == cond_safe:
                    cond_hits.append(p)

            # prefer pca2d
            def _rank(p: Path) -> tuple[int, str]:
                k = _infer_kind_for_embedding_path(p)
                return (0 if k == "embeddings_pca2d" else 1, str(p))

            for src in sorted(set(cond_hits), key=_rank):
                try:
                    kind = _infer_kind_for_embedding_path(src)
                    ext = src.suffix.lower()
                    dst = _paper_dest_name(
                        out_fig_root=out_fig_root,
                        prefix=prefix,
                        kind=kind,
                        ext=ext,
                        cond_safe=(None if cond == base_cond else cond_safe),
                    )
                    sha = _sha256_file(src)
                    dst = _resolve_collision(dst, sha)
                    if not dst.exists():
                        _copy_mode_op(src, dst, mode=mode)

                    entries.append(
                        FigureEntry(
                            run_id=run_id,
                            evaldir=str(evaldir),
                            suite=suite,
                            checkpoint=str(checkpoint) if checkpoint is not None else None,
                            meta_hash=meta_hash,
                            source_path=str(src),
                            source_rel=_relpath(evaldir, src),
                            dest_path=str(dst),
                            dest_rel=_relpath(out_fig_root.parent, dst),
                            cond=cond,
                            cond_safe=cond_safe,
                            kind=kind,
                            tags=[],
                            ext=ext,
                            sha256=sha,
                            bytes=int(src.stat().st_size),
                            mtime_utc=_file_mtime_utc(src),
                        )
                    )
                except Exception as ex:
                    errors.append({"evaldir": str(evaldir), "source": str(src), "error": f"{type(ex).__name__}: {ex}"})

    # deterministic order
    entries.sort(key=lambda e: (e.kind, e.cond_safe, e.dest_rel))
    return entries, errors




# =============================================================================
# Core bundling
# =============================================================================


def _evaldirs_from_results_json(path: Path) -> list[Path]:
    j = _safe_read_json(path)
    if not isinstance(j, Mapping):
        raise ValueError(f"results.json is not a JSON object: {path}")
    runs = j.get("runs", None)
    if not isinstance(runs, list):
        raise ValueError(f"results.json missing 'runs' list: {path}")
    out: list[Path] = []
    for r in runs:
        if not isinstance(r, Mapping):
            continue
        d = r.get("evaldir", None)
        if isinstance(d, str) and d:
            out.append(Path(d).expanduser().resolve())
    uniq: dict[str, Path] = {str(p): p for p in out}
    return sorted(uniq.values(), key=lambda p: str(p))


def bundle_figures_for_evaldir(
    evaldir: Path,
    *,
    out_fig_root: Path,
    mode: str,
    layout: str,
    include_embeddings: bool,
    strict: bool,
) -> tuple[list[FigureEntry], list[dict[str, Any]]]:
    """
    Returns (entries, errors) for this evaldir.
    """
    evaldir = evaldir.resolve()
    meta_p = evaldir / "meta.json"
    if not meta_p.exists():
        err = {"evaldir": str(evaldir), "error": "missing meta.json"}
        return ([], [err])

    meta_json = _safe_read_json(meta_p) or {}
    suite = meta_json.get("suite", None)
    checkpoint = meta_json.get("checkpoint", None)
    meta_hash = _compute_evaldir_hash(evaldir)
    run_id = _make_run_id(evaldir, meta_json, meta_hash)

    conditions = _load_conditions(evaldir)
    safe_map = _safe_to_cond_map(conditions)

    entries: list[FigureEntry] = []
    errors: list[dict[str, Any]] = []

    # --- 1) evaldir/figs figures ---
    fig_paths = _discover_eval_figs(evaldir)
    for src in fig_paths:
        try:
            ext = src.suffix.lower()
            stem = src.stem
            parsed = _parse_figure_stem(stem)

            cond_safe_guess = parsed.cond_safe
            cond = "_ALL_"
            if cond_safe_guess != "_ALL_":
                cond = safe_map.get(cond_safe_guess, cond_safe_guess)

            kind = parsed.kind
            tags = list(parsed.tags)

            sha = _sha256_file(src)
            dst = _default_layout_path(
                out_fig_root=out_fig_root,
                suite=str(suite) if suite is not None else None,
                run_id=run_id,
                cond=cond,
                kind=kind,
                tags=tags,
                ext=ext,
                layout=layout,
            )
            dst = _resolve_collision(dst, sha)

            # copy/link
            if not dst.exists():
                _copy_mode_op(src, dst, mode=mode)

            e = FigureEntry(
                run_id=run_id,
                evaldir=str(evaldir),
                suite=str(suite) if suite is not None else None,
                checkpoint=str(checkpoint) if checkpoint is not None else None,
                meta_hash=meta_hash,
                source_path=str(src),
                source_rel=_relpath(evaldir, src),
                dest_path=str(dst),
                dest_rel=_relpath(out_fig_root.parent, dst),  # relative to bundle root parent (outdir)
                cond=cond,
                cond_safe=_sanitize(cond),
                kind=kind,
                tags=tags,
                ext=ext,
                sha256=sha,
                bytes=int(src.stat().st_size),
                mtime_utc=_file_mtime_utc(src),
            )
            entries.append(e)

        except Exception as ex:
            payload = {
                "evaldir": str(evaldir),
                "source": str(src),
                "error": f"{type(ex).__name__}: {ex}",
            }
            errors.append(payload)
            if strict:
                raise

    # --- 2) embeddings artifacts (optional) ---
    if include_embeddings:
        emb_paths = _discover_embedding_figs(evaldir)
        for src in emb_paths:
            try:
                ext = src.suffix.lower()
                if ext not in _SUPPORTED_EXTS:
                    continue
                kind = _infer_kind_for_embedding_path(src)
                cond, cond_safe_guess = _infer_cond_from_embedding_path(src, safe_map)
                tags: list[str] = []

                sha = _sha256_file(src)
                dst = _default_layout_path(
                    out_fig_root=out_fig_root,
                    suite=str(suite) if suite is not None else None,
                    run_id=run_id,
                    cond=cond,
                    kind=kind,
                    tags=tags,
                    ext=ext,
                    layout=layout,
                )
                dst = _resolve_collision(dst, sha)

                if not dst.exists():
                    _copy_mode_op(src, dst, mode=mode)

                e = FigureEntry(
                    run_id=run_id,
                    evaldir=str(evaldir),
                    suite=str(suite) if suite is not None else None,
                    checkpoint=str(checkpoint) if checkpoint is not None else None,
                    meta_hash=meta_hash,
                    source_path=str(src),
                    source_rel=_relpath(evaldir, src),
                    dest_path=str(dst),
                    dest_rel=_relpath(out_fig_root.parent, dst),
                    cond=cond,
                    cond_safe=_sanitize(cond),
                    kind=kind,
                    tags=tags,
                    ext=ext,
                    sha256=sha,
                    bytes=int(src.stat().st_size),
                    mtime_utc=_file_mtime_utc(src),
                )
                entries.append(e)

            except Exception as ex:
                payload = {
                    "evaldir": str(evaldir),
                    "source": str(src),
                    "error": f"{type(ex).__name__}: {ex}",
                }
                errors.append(payload)
                if strict:
                    raise

    # deterministic order
    entries.sort(key=lambda e: (e.suite or "", e.run_id, e.cond_safe, e.kind, e.dest_rel))
    return entries, errors


def write_manifest_json(path: Path, *, meta: Mapping[str, Any], entries: Sequence[FigureEntry]) -> None:
    out_entries: list[dict[str, Any]] = []
    for e in entries:
        out_entries.append(
            {
                "run_id": e.run_id,
                "evaldir": e.evaldir,
                "suite": e.suite,
                "checkpoint": e.checkpoint,
                "meta_hash": e.meta_hash,
                "cond": e.cond,
                "kind": e.kind,
                "tags": e.tags,
                "ext": e.ext,
                "sha256": e.sha256,
                "bytes": e.bytes,
                "mtime_utc": e.mtime_utc,
                "source": {"path": e.source_path, "rel": e.source_rel},
                "dest": {"path": e.dest_path, "rel": e.dest_rel},
            }
        )
    _write_json(path, {"meta": dict(meta), "entries": out_entries})


def write_manifest_csv(path: Path, *, entries: Sequence[FigureEntry]) -> None:
    _ensure_dir(path.parent)
    fieldnames = [
        "suite",
        "run_id",
        "cond",
        "kind",
        "tags",
        "ext",
        "sha256",
        "bytes",
        "mtime_utc",
        "evaldir",
        "checkpoint",
        "meta_hash",
        "source_rel",
        "dest_rel",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in entries:
            w.writerow(
                {
                    "suite": e.suite or "",
                    "run_id": e.run_id,
                    "cond": e.cond,
                    "kind": e.kind,
                    "tags": "__".join(e.tags) if e.tags else "",
                    "ext": e.ext,
                    "sha256": e.sha256,
                    "bytes": e.bytes,
                    "mtime_utc": e.mtime_utc,
                    "evaldir": e.evaldir,
                    "checkpoint": e.checkpoint or "",
                    "meta_hash": e.meta_hash,
                    "source_rel": e.source_rel,
                    "dest_rel": e.dest_rel,
                }
            )


# =============================================================================
# CLI
# =============================================================================


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bundle/organize evaluation figures into a stable figures/ directory with manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=str, default="", help="Root directory to scan for eval outputs (contains meta.json).")
    src.add_argument("--evaldirs", nargs="+", default=[], help="Explicit eval output directories to bundle figures from.")
    src.add_argument("--results-json", type=str, default="", help="Path to results.json produced by collect_results.py.")

    p.add_argument("--outdir", type=str, required=True, help="Output directory for bundle (will create figures/).")

    p.add_argument("--max-depth", type=int, default=None, help="Max scan depth under --root (None = unlimited).")
    p.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks while scanning.")
    p.add_argument(
        "--exclude-regex",
        type=str,
        default=r"(^|/)(\.git|__pycache__|wandb|mlruns|raw)(/|$)",
        help="Regex of paths to exclude during scanning.",
    )

    p.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "hardlink", "symlink"],
        help="How to materialize bundled figures.",
    )
    p.add_argument(
        "--layout",
        type=str,
        default="hier",
        choices=["hier", "flat", "by_kind"],
        help="Output directory layout for figures.",
    )
    p.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Also bundle discovered embedding artifact images (PCA plots, etc.).",
    )



    # Paper/slide stable naming: copy+rename a single evaldir into canonical figure names.
    p.add_argument(
        "--paper",
        action="store_true",
        help=(
            "Paper mode: copy/rename key figures from a SINGLE evaldir into stable names "
            "(Pt/tv/robustness/mask_sensitivity). Requires --evaldirs with exactly one path."
        ),
    )
    p.add_argument(
        "--paper-prefix",
        type=str,
        default=None,
        help=(
            "Prefix used in stable paper filenames. Default: sanitize(meta.suite). "
            "Example: 'main' -> figures/main__Pt.png, main__tv.png, ..."
        ),
    )
    p.add_argument(
        "--paper-cond",
        type=str,
        default=None,
        help=(
            "Condition name to treat as the base condition for stable names. "
            "Default: reporting base_condition() if available, else 'ckpt_policy' if present, else first condition."
        ),
    )
    p.add_argument(
        "--paper-include-ablations",
        action="store_true",
        help="Also emit stable names for non-base conditions (ablations/baselines) as <prefix>__KIND__COND.ext.",
    )




    p.add_argument("--no-index", action="store_true", help="Do not write FIGURES.md index.")
    p.add_argument("--write-tex", action="store_true", help="Also write FIGURES.tex LaTeX snippet.")
    p.add_argument("--tex-max-per-run", type=int, default=8, help="Max figures per run for FIGURES.tex snippet.")

    p.add_argument("--strict", action="store_true", help="Fail-fast if any evaldir has errors.")
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
    out_fig_root = outdir / "figures"
    _ensure_dir(out_fig_root)

    # Determine evaldirs
    if args.results_json:
        evaldirs = _evaldirs_from_results_json(_as_path(args.results_json).expanduser().resolve())
    elif args.evaldirs:
        evaldirs = [_as_path(x).expanduser().resolve() for x in args.evaldirs]
    else:
        root = _as_path(args.root).expanduser().resolve()
        evaldirs = find_evaldirs(
            root=root,
            max_depth=args.max_depth,
            follow_symlinks=bool(args.follow_symlinks),
            exclude_regex=args.exclude_regex,
        )

    evaldirs = [d for d in evaldirs if d.exists() and d.is_dir()]
    
    
    
    # ---------------- Paper mode: simple stable names ----------------
    if args.paper:
        if len(evaldirs) != 1:
            log.error("--paper requires exactly ONE evaldir (use --evaldirs <dir>). Got: %d", len(evaldirs))
            return 2

        d = evaldirs[0]
        entries, errors = bundle_paper_figures(
            d,
            out_fig_root=out_fig_root,
            mode=str(args.mode),
            include_embeddings=bool(args.include_embeddings),
            include_ablations=bool(args.paper_include_ablations),
            paper_prefix=str(args.paper_prefix) if args.paper_prefix else None,
            paper_cond=str(args.paper_cond) if args.paper_cond else None,
        )

        # Deterministic order
        entries.sort(key=lambda e: (e.kind, e.cond_safe, e.dest_rel))

        manifest_json = outdir / "manifest.figures.json"
        manifest_csv = outdir / "manifest.figures.csv"
        meta = {
            "schema": "acpl.make_figures.paper.v1",
            "generated_at": _utc_now_iso(),
            "argv": sys.argv,
            "cwd": str(Path.cwd().resolve()),
            "outdir": str(outdir),
            "figures_root": str(out_fig_root),
            "mode": str(args.mode),
            "paper_prefix": args.paper_prefix,
            "paper_cond": args.paper_cond,
            "paper_include_ablations": bool(args.paper_include_ablations),
            "include_embeddings": bool(args.include_embeddings),
            "counts": {"evaldirs": 1, "entries": len(entries), "errors": len(errors)},
            "errors": errors[:200],
        }

        write_manifest_json(manifest_json, meta=meta, entries=entries)
        write_manifest_csv(manifest_csv, entries=entries)

        if not args.no_index:
            index_md = outdir / "FIGURES.md"
            write_figures_markdown_index(index_md, bundle_root=outdir, entries=entries)
            log.info("Wrote index: %s", str(index_md))

        if args.write_tex:
            tex_path = outdir / "FIGURES.tex"
            write_figures_latex_snippet(tex_path, entries=entries, max_per_run=int(args.tex_max_per_run))
            log.info("Wrote LaTeX snippet: %s", str(tex_path))

        bundle_meta = outdir / "bundle_meta.json"
        _write_json(bundle_meta, meta)

        if errors:
            log.warning("Paper bundle completed with %d errors.", len(errors))
            return 1
        log.info("Paper bundle completed with zero errors.")
        return 0
   
    
    
    
    
    
    if not evaldirs:
        log.error("No valid eval directories found.")
        return 2

    # Git provenance: best effort, repository root = cwd
    repo_root = Path.cwd().resolve()
    git = _git_info(repo_root)

    if not _HAVE_REPORTING:
        log.warning(
            "acpl.eval.reporting could not be imported. Condition mapping will be best-effort."
        )

    log.info("Bundling figures from %d evaldirs.", len(evaldirs))

    all_entries: list[FigureEntry] = []
    all_errors: list[dict[str, Any]] = []

    for d in sorted(evaldirs, key=lambda p: str(p)):
        try:
            entries, errors = bundle_figures_for_evaldir(
                d,
                out_fig_root=out_fig_root,
                mode=str(args.mode),
                layout=str(args.layout),
                include_embeddings=bool(args.include_embeddings),
                strict=bool(args.strict),
            )
            all_entries.extend(entries)
            all_errors.extend(errors)
            log.info(
                "Evaldir: %s  -> bundled %d figures (%d errors)",
                str(d),
                len(entries),
                len(errors),
            )
        except Exception as e:
            payload = {"evaldir": str(d), "error": f"fatal: {type(e).__name__}: {e}"}
            all_errors.append(payload)
            if args.strict:
                raise
            log.error("Failed evaldir %s: %s", str(d), payload["error"])

    # Deterministic order
    all_entries.sort(key=lambda e: (e.suite or "", e.run_id, e.cond_safe, e.kind, e.dest_rel))

    # Write manifests
    manifest_json = outdir / "manifest.figures.json"
    manifest_csv = outdir / "manifest.figures.csv"
    meta = {
        "schema": "acpl.make_figures.v1",
        "generated_at": _utc_now_iso(),
        "argv": sys.argv,
        "cwd": str(Path.cwd().resolve()),
        "outdir": str(outdir),
        "figures_root": str(out_fig_root),
        "mode": str(args.mode),
        "layout": str(args.layout),
        "include_embeddings": bool(args.include_embeddings),
        "git": git,
        "counts": {
            "evaldirs": len(evaldirs),
            "entries": len(all_entries),
            "errors": len(all_errors),
        },
        "errors": all_errors[:200],  # keep manifest readable; full errors also in JSON meta
    }

    write_manifest_json(manifest_json, meta=meta, entries=all_entries)
    write_manifest_csv(manifest_csv, entries=all_entries)

    # Write index/snippet
    if not args.no_index:
        index_md = outdir / "FIGURES.md"
        write_figures_markdown_index(index_md, bundle_root=outdir, entries=all_entries)
        log.info("Wrote index: %s", str(index_md))

    if args.write_tex:
        tex_path = outdir / "FIGURES.tex"
        write_figures_latex_snippet(tex_path, entries=all_entries, max_per_run=int(args.tex_max_per_run))
        log.info("Wrote LaTeX snippet: %s", str(tex_path))

    # Write bundle meta
    bundle_meta = outdir / "bundle_meta.json"
    _write_json(bundle_meta, meta)

    log.info("Wrote manifest: %s", str(manifest_json))
    log.info("Wrote manifest: %s", str(manifest_csv))
    log.info("Wrote meta: %s", str(bundle_meta))

    if all_errors:
        log.warning("Completed with %d errors (see bundle_meta.json / manifest.figures.json).", len(all_errors))
        return 1
    log.info("Completed with zero errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

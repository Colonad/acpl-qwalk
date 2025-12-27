# acpl/eval/reporting.py
from __future__ import annotations

"""
acpl.eval.reporting

Research-ready reporting utilities for evaluation runs produced by scripts/eval.py.

This module is intentionally robust to partial artifacts:
- If summary.json exists, it is the primary structured source.
- If condition folders contain eval_ci.json / eval_ci.txt, we also ingest those.
- Figures under <evaldir>/figs are discovered and referenced in reports.

Expected eval directory layout (from scripts/eval.py):
<outdir>/
  meta.json
  summary.json
  summary.csv
  raw/
    <cond_tag>/
      condition_meta.json
      eval_ci.json
      eval_ci.txt
      logs.jsonl           (optional)
  figs/
    Pt__<cond>.png         (optional)
    tv__<cond>.png         (optional)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import argparse
import csv
import datetime as _dt
import json
import os
import re
import sys


# ----------------------------- small IO helpers -----------------------------


def _as_path(p: str | os.PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(_read_text(path))
    except Exception:
        return None


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _json_dumps(obj: Any) -> str:
    # eval.py already uses a numpy-aware encoder; here we keep it conservative.
    return json.dumps(obj, sort_keys=True, indent=2, default=str)


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _relpath(from_dir: Path, to_path: Path) -> str:
    try:
        return str(to_path.relative_to(from_dir))
    except Exception:
        try:
            return os.path.relpath(str(to_path), str(from_dir))
        except Exception:
            return str(to_path)


def _sanitize_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    return s or "cond"


# ----------------------------- core dataclasses -----------------------------


@dataclass(frozen=True)
class CIRecord:
    mean: float
    lo: float
    hi: float
    stderr: float | None = None
    n: int | None = None

    def fmt(self, *, digits: int = 4) -> str:
        # defendable: always show interval endpoints if available
        m = f"{self.mean:.{digits}f}"
        lo = f"{self.lo:.{digits}f}"
        hi = f"{self.hi:.{digits}f}"
        if self.stderr is None or self.n is None:
            return f"{m} [{lo}, {hi}]"
        return f"{m} [{lo}, {hi}] (se={self.stderr:.{digits}f}, n={self.n})"


@dataclass
class ConditionResult:
    cond: str
    policy: str | None = None
    baseline_kind: str | None = None
    ablation_meta: dict[str, Any] = field(default_factory=dict)

    # from summary.json (preferred)
    summary_means: dict[str, float] = field(default_factory=dict)
    ci: dict[str, CIRecord] = field(default_factory=dict)
    text: str | None = None

    # from filesystem
    path: Path | None = None
    logs_jsonl: Path | None = None

    def is_skipped(self) -> bool:
        # summary.json in your eval.py uses {"skipped": True, ...} for failures sometimes
        # but we keep it flexible: treat empty summary + empty ci as skipped-ish.
        return (not self.summary_means) and (not self.ci)


@dataclass
class EvalMeta:
    checkpoint: str | None = None
    policy: str | None = None
    baseline_kind: str | None = None
    baseline_coins_kwargs: dict[str, Any] | None = None
    baseline_policy_kwargs: dict[str, Any] | None = None

    suite: str | None = None
    device: str | None = None
    episodes: int | None = None
    seeds: list[int] = field(default_factory=list)

    ablations: list[str] = field(default_factory=list)
    ablations_canonical: list[str] = field(default_factory=list)

    model_config: dict[str, Any] | None = None

    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRun:
    evaldir: Path
    meta: EvalMeta
    conditions: dict[str, ConditionResult] = field(default_factory=dict)
    figs: dict[str, list[Path]] = field(default_factory=dict)  # cond_tag -> list of files

    def condition_order(self) -> list[str]:
        # stable ordering: base condition first, then others alphabetically
        keys = list(self.conditions.keys())
        if not keys:
            return []
        base = None
        # heuristics: your eval.py uses "ckpt_policy" or "baseline_*"
        for k in keys:
            if k == "ckpt_policy" or k.startswith("baseline_"):
                base = k
                break
        if base is None:
            base = sorted(keys)[0]
        rest = [k for k in keys if k != base]
        rest.sort()
        return [base] + rest

    def base_condition(self) -> str | None:
        order = self.condition_order()
        return order[0] if order else None


# ----------------------------- discovery/loaders -----------------------------


def _parse_meta(meta_json: dict[str, Any]) -> EvalMeta:
    return EvalMeta(
        checkpoint=meta_json.get("checkpoint"),
        policy=meta_json.get("policy"),
        baseline_kind=meta_json.get("baseline_kind"),
        baseline_coins_kwargs=meta_json.get("baseline_coins_kwargs"),
        baseline_policy_kwargs=meta_json.get("baseline_policy_kwargs"),
        suite=meta_json.get("suite"),
        device=meta_json.get("device"),
        episodes=int(meta_json["episodes"]) if "episodes" in meta_json and meta_json["episodes"] is not None else None,
        seeds=[int(x) for x in (meta_json.get("seeds") or [])],
        ablations=list(meta_json.get("ablations") or []),
        ablations_canonical=list(meta_json.get("ablations_canonical") or []),
        model_config=meta_json.get("model_config"),
        raw=dict(meta_json),
    )


def _parse_ci_json(ci_json: Mapping[str, Any]) -> dict[str, CIRecord]:
    out: dict[str, CIRecord] = {}
    for k, v in ci_json.items():
        if not isinstance(v, Mapping):
            continue
        try:
            mean = float(v["mean"])
            lo = float(v.get("lo", mean))
            hi = float(v.get("hi", mean))
        except Exception:
            continue
        stderr = v.get("stderr", None)
        n = v.get("n", None)
        out[k] = CIRecord(
            mean=mean,
            lo=lo,
            hi=hi,
            stderr=float(stderr) if stderr is not None else None,
            n=int(n) if n is not None else None,
        )
    return out


def _discover_condition_dirs(evaldir: Path) -> list[Path]:
    raw_dir = evaldir / "raw"
    if not raw_dir.exists():
        return []
    dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name)
    return dirs


def _discover_figs(evaldir: Path) -> dict[str, list[Path]]:
    figdir = evaldir / "figs"
    out: dict[str, list[Path]] = {}
    if not figdir.exists():
        return out
    pngs = [p for p in figdir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")]
    pngs.sort(key=lambda p: p.name)
    # heuristic mapping: filenames contain sanitized condition tag (scripts/eval.py does Pt__{safe}.png / tv__{safe}.png)
    for p in pngs:
        out.setdefault("_ALL_", []).append(p)
    return out


def load_eval_run(evaldir: str | os.PathLike) -> EvalRun:
    """
    Load an evaluation run directory produced by scripts/eval.py.

    Priority sources:
      1) meta.json + summary.json (structured)
      2) per-condition files under raw/<cond>/
    """
    root = _as_path(evaldir).resolve()


    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"[reporting] meta.json not found in evaldir: {root}\n"
            f"Expected: {meta_path}\n"
            f"Did you point --evaldir at the eval output directory created by scripts/eval.py?"
        )
    meta_json = _safe_read_json(meta_path) or {}
    meta = _parse_meta(meta_json)


    run = EvalRun(evaldir=root, meta=meta)

    # ---- summary.json (preferred) ----
    summ_path = root / "summary.json"
    summ_json = _safe_read_json(summ_path)
    if isinstance(summ_json, Mapping) and isinstance(summ_json.get("conditions"), Mapping):
        for cond, payload in summ_json["conditions"].items():
            if not isinstance(payload, Mapping):
                continue
            cr = ConditionResult(cond=str(cond))
            # expected keys: summary, ci, text, skipped...
            summ = payload.get("summary", {})
            if isinstance(summ, Mapping):
                cr.summary_means = {str(k): float(v) for k, v in summ.items() if _is_number(v)}
            ci_block = payload.get("ci", {})
            if isinstance(ci_block, Mapping):
                cr.ci = _parse_ci_json(ci_block)
            txt = payload.get("text", None)
            if isinstance(txt, str):
                cr.text = txt
            run.conditions[cr.cond] = cr

    # ---- per-condition dirs ----
    for cdir in _discover_condition_dirs(root):
        cond = cdir.name
        cr = run.conditions.get(cond) or ConditionResult(cond=cond)
        cr.path = cdir

        cm = _safe_read_json(cdir / "condition_meta.json") or {}
        if isinstance(cm, Mapping):
            cr.policy = cm.get("policy")
            cr.baseline_kind = cm.get("baseline_kind")
            am = cm.get("ablation_meta", {})
            if isinstance(am, Mapping):
                cr.ablation_meta = dict(am)

        if not cr.ci:
            ci_json = _safe_read_json(cdir / "eval_ci.json")
            if isinstance(ci_json, Mapping):
                cr.ci = _parse_ci_json(ci_json)

        if cr.text is None:
            txt_path = cdir / "eval_ci.txt"
            if txt_path.exists():
                try:
                    cr.text = _read_text(txt_path)
                except Exception:
                    cr.text = None

        logs = cdir / "logs.jsonl"
        if logs.exists():
            cr.logs_jsonl = logs

        run.conditions[cond] = cr

    run.figs = _discover_figs(root)
    return run


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


# ----------------------------- metric selection heuristics -----------------------------


def _metric_priority(name: str, *, suite: str | None = None) -> tuple[int, str]:
    """
    Lower priority value means "more important / show earlier".
    We keep it heuristic and robust across tasks.
    """
    s = (name or "").lower()
    # suite-aware nudge
    if suite and "mix" in suite.lower():
        if "tv" in s:
            return (0, name)
    if "tv" in s:
        return (1, name)
    if "target" in s and ("p_" in s or "prob" in s or "success" in s):
        return (2, name)
    if "success" in s or "hit" in s or "found" in s:
        return (3, name)
    if "loss" in s:
        return (90, name)  # usually not the thesis headline metric at eval-time
    return (10, name)


def select_key_metrics(
    run: EvalRun,
    *,
    max_cols: int = 6,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[str]:
    """
    Choose a small set of "headline" metrics present across conditions.
    """
    suite = run.meta.suite
    inc = set(m.lower() for m in (include or []))
    exc = set(m.lower() for m in (exclude or []))

    # collect all metric names
    all_names: set[str] = set()
    for cr in run.conditions.values():
        all_names |= set(cr.ci.keys()) | set(cr.summary_means.keys())

    names = sorted(all_names)
    if inc:
        names = [n for n in names if n.lower() in inc]
    if exc:
        names = [n for n in names if n.lower() not in exc]

    names.sort(key=lambda n: _metric_priority(n, suite=suite))
    return names[: max(1, int(max_cols))]


# ----------------------------- markdown report generation -----------------------------


def _md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|")


def _md_ci_cell(cr: ConditionResult, metric: str, *, digits: int = 4) -> str:
    if metric in cr.ci:
        return cr.ci[metric].fmt(digits=digits)
    if metric in cr.summary_means:
        return f"{cr.summary_means[metric]:.{digits}f}"
    return ""


def build_markdown_report(
    run: EvalRun,
    *,
    title: str | None = None,
    max_metric_cols: int = 6,
    digits: int = 4,
    include_figs: bool = True,
) -> str:
    """
    Build a defendable Markdown report with:
    - run metadata
    - headline table (conditions x key metrics)
    - per-condition CI dumps (optional)
    - discovered figures
    """
    t = title or f"ACPL Evaluation Report — {run.meta.suite or 'suite'}"
    lines: list[str] = []
    lines.append(f"# {t}")
    lines.append("")
    lines.append(f"- Generated: `{_utc_now_iso()}`")
    if run.meta.checkpoint:
        lines.append(f"- Checkpoint: `{run.meta.checkpoint}`")
    if run.meta.policy:
        lines.append(f"- Policy: `{run.meta.policy}`")
    if run.meta.baseline_kind:
        lines.append(f"- Baseline kind: `{run.meta.baseline_kind}`")
    if run.meta.device:
        lines.append(f"- Device: `{run.meta.device}`")
    if run.meta.suite:
        lines.append(f"- Suite: `{run.meta.suite}`")
    if run.meta.episodes is not None:
        lines.append(f"- Episodes per seed: `{run.meta.episodes}`")
    if run.meta.seeds:
        lines.append(f"- Seeds: `{run.meta.seeds}` (n={len(run.meta.seeds)})")
    if run.meta.ablations_canonical:
        lines.append(f"- Ablations: `{run.meta.ablations_canonical}`")
    lines.append("")

    # Headline table
    keys = select_key_metrics(run, max_cols=max_metric_cols)
    order = run.condition_order()

    lines.append("## Headline metrics")
    lines.append("")
    header = ["Condition"] + keys
    lines.append("| " + " | ".join(_md_escape(h) for h in header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for cond in order:
        cr = run.conditions[cond]
        row = [cond] + [_md_ci_cell(cr, k, digits=digits) for k in keys]
        lines.append("| " + " | ".join(_md_escape(str(x)) for x in row) + " |")
    lines.append("")

    # Ablation deltas (relative to base)
    base = run.base_condition()
    if base is not None and base in run.conditions and len(order) > 1:
        lines.append("## Deltas vs base condition")
        lines.append("")
        base_cr = run.conditions[base]
        # choose same keys
        lines.append(f"Base: `{base}`")
        lines.append("")
        lines.append("| Condition | Metric | Base | Cond | Δ | Δ% |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for cond in order:
            if cond == base:
                continue
            cr = run.conditions[cond]
            for m in keys:
                b = _get_mean(base_cr, m)
                c = _get_mean(cr, m)
                if b is None or c is None:
                    continue
                d = c - b
                dp = (d / b * 100.0) if b != 0 else None
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _md_escape(cond),
                            _md_escape(m),
                            f"{b:.{digits}f}",
                            f"{c:.{digits}f}",
                            f"{d:+.{digits}f}",
                            (f"{dp:+.{digits}f}%" if dp is not None else ""),
                        ]
                    )
                    + " |"
                )
        lines.append("")

    # Figures
    if include_figs:
        figdir = run.evaldir / "figs"
        if figdir.exists():
            lines.append("## Figures")
            lines.append("")
            # show per-condition known figure naming patterns first
            for cond in order:
                safe = _sanitize_filename(cond)
                pt = figdir / f"Pt__{safe}.png"
                tv = figdir / f"tv__{safe}.png"
                any_added = False
                if pt.exists():
                    lines.append(f"### {cond} — Position timeline")
                    lines.append("")
                    lines.append(f"![Pt]({_relpath(run.evaldir, pt)})")
                    lines.append("")
                    any_added = True
                if tv.exists():
                    lines.append(f"### {cond} — TV-to-uniform")
                    lines.append("")
                    lines.append(f"![TV]({_relpath(run.evaldir, tv)})")
                    lines.append("")
                    any_added = True
                if not any_added:
                    # nothing for this condition; skip quietly
                    pass

            # list any remaining images not captured by the pattern
            others = []
            for p in figdir.iterdir():
                if not p.is_file():
                    continue
                if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".pdf"):
                    continue
                if p.name.startswith("Pt__") or p.name.startswith("tv__"):
                    continue
                others.append(p)
            others.sort(key=lambda p: p.name)
            if others:
                lines.append("### Other figures")
                lines.append("")
                for p in others:
                    lines.append(f"- `{_relpath(run.evaldir, p)}`")
                lines.append("")

    # Per-condition CI details (defendable appendix)
    lines.append("## Appendix: per-condition CI tables")
    lines.append("")
    for cond in order:
        cr = run.conditions[cond]
        lines.append(f"### {cond}")
        lines.append("")
        if cr.text:
            lines.append("**eval_ci.txt**")
            lines.append("")
            lines.append("```")
            # keep it short-ish; still defendable
            lines.extend(cr.text.strip().splitlines())
            lines.append("```")
            lines.append("")
        # Also provide a deterministic table from eval_ci.json if present
        if cr.ci:
            lines.append("**eval_ci.json (parsed)**")
            lines.append("")
            lines.append("| Metric | Mean | 95% CI | stderr | n |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for m in sorted(cr.ci.keys(), key=lambda n: _metric_priority(n, suite=run.meta.suite)):
                ci = cr.ci[m]
                se = "" if ci.stderr is None else f"{ci.stderr:.{digits}f}"
                nn = "" if ci.n is None else str(ci.n)
                lines.append(
                    f"| {_md_escape(m)} | {ci.mean:.{digits}f} | "
                    f"[{ci.lo:.{digits}f}, {ci.hi:.{digits}f}] | {se} | {nn} |"
                )
            lines.append("")
        else:
            lines.append("_No CI records found for this condition._")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _get_mean(cr: ConditionResult, metric: str) -> float | None:
    if metric in cr.ci:
        return cr.ci[metric].mean
    if metric in cr.summary_means:
        return float(cr.summary_means[metric])
    return None


def write_markdown_report(
    evaldir: str | os.PathLike,
    out_path: str | os.PathLike,
    *,
    title: str | None = None,
    max_metric_cols: int = 6,
    digits: int = 4,
    include_figs: bool = True,
) -> Path:
    run = load_eval_run(evaldir)
    md = build_markdown_report(
        run,
        title=title,
        max_metric_cols=max_metric_cols,
        digits=digits,
        include_figs=include_figs,
    )
    outp = _as_path(out_path)
    _write_text(outp, md)
    return outp


# ----------------------------- LaTeX table generation -----------------------------


def build_latex_table(
    run: EvalRun,
    *,
    metrics: Sequence[str] | None = None,
    digits: int = 4,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """
    Build a booktabs-style LaTeX table: rows=conditions, cols=metrics (mean [lo,hi]).
    """
    metrics2 = list(metrics) if metrics is not None else select_key_metrics(run, max_cols=6)
    order = run.condition_order()

    cap = caption or f"Evaluation summary for suite {run.meta.suite or ''}."
    lab = label or f"tab:eval_{_sanitize_filename(run.meta.suite or 'suite')}"

    # column spec: left + rrr...
    colspec = "l" + ("r" * len(metrics2))

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{cap}}}")
    lines.append(rf"\label{{{lab}}}")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    header = "Condition & " + " & ".join(_latex_escape(m) for m in metrics2) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for cond in order:
        cr = run.conditions[cond]
        cells = []
        for m in metrics2:
            if m in cr.ci:
                ci = cr.ci[m]
                cells.append(
                    rf"{ci.mean:.{digits}f}~[{ci.lo:.{digits}f},{ci.hi:.{digits}f}]"
                )
            elif m in cr.summary_means:
                cells.append(rf"{cr.summary_means[m]:.{digits}f}")
            else:
                cells.append("")
        row = _latex_escape(cond) + " & " + " & ".join(cells) + r" \\"
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _latex_escape(s: str) -> str:
    # minimal escaping for table text
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("_", r"\_")
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    s = s.replace("#", r"\#")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("^", r"\^{}")
    s = s.replace("~", r"\~{}")
    return s


def write_latex_table(
    evaldir: str | os.PathLike,
    out_path: str | os.PathLike,
    *,
    metrics: Sequence[str] | None = None,
    digits: int = 4,
    caption: str | None = None,
    label: str | None = None,
) -> Path:
    run = load_eval_run(evaldir)
    tex = build_latex_table(run, metrics=metrics, digits=digits, caption=caption, label=label)
    outp = _as_path(out_path)
    _write_text(outp, tex)
    return outp


# ----------------------------- CSV exporters -----------------------------


def write_csv_wide(
    evaldir: str | os.PathLike,
    out_path: str | os.PathLike,
    *,
    prefer_ci_means: bool = True,
) -> Path:
    """
    Wide CSV: one row per condition, columns = metrics (mean only).
    """
    run = load_eval_run(evaldir)
    order = run.condition_order()

    # union metrics
    metrics: set[str] = set()
    for cr in run.conditions.values():
        metrics |= set(cr.summary_means.keys()) | set(cr.ci.keys())
    cols = ["cond"] + sorted(metrics, key=lambda n: _metric_priority(n, suite=run.meta.suite))

    outp = _as_path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for cond in order:
            cr = run.conditions[cond]
            row: dict[str, Any] = {"cond": cond}
            for m in metrics:
                if prefer_ci_means and m in cr.ci:
                    row[m] = cr.ci[m].mean
                elif m in cr.summary_means:
                    row[m] = cr.summary_means[m]
                else:
                    row[m] = ""
            w.writerow(row)
    return outp


def write_csv_long(
    evaldir: str | os.PathLike,
    out_path: str | os.PathLike,
    *,
    digits: int = 6,
) -> Path:
    """
    Long CSV: rows = (cond, metric, mean, lo, hi, stderr, n).
    """
    run = load_eval_run(evaldir)
    order = run.condition_order()

    outp = _as_path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["cond", "metric", "mean", "lo", "hi", "stderr", "n"]
        )
        w.writeheader()
        for cond in order:
            cr = run.conditions[cond]
            # prefer CI if available
            for m in sorted(set(cr.ci.keys()) | set(cr.summary_means.keys()),
                            key=lambda n: _metric_priority(n, suite=run.meta.suite)):
                if m in cr.ci:
                    ci = cr.ci[m]
                    w.writerow(
                        {
                            "cond": cond,
                            "metric": m,
                            "mean": f"{ci.mean:.{digits}f}",
                            "lo": f"{ci.lo:.{digits}f}",
                            "hi": f"{ci.hi:.{digits}f}",
                            "stderr": "" if ci.stderr is None else f"{ci.stderr:.{digits}f}",
                            "n": "" if ci.n is None else str(ci.n),
                        }
                    )
                elif m in cr.summary_means:
                    v = float(cr.summary_means[m])
                    w.writerow(
                        {
                            "cond": cond,
                            "metric": m,
                            "mean": f"{v:.{digits}f}",
                            "lo": "",
                            "hi": "",
                            "stderr": "",
                            "n": "",
                        }
                    )
    return outp


# ----------------------------- report template generator (5.4) -----------------------------


def build_report_template(run: EvalRun) -> str:
    """
    A fill-in template intended for thesis/paper writing.
    This is NOT the auto-report; it's a structured outline you can edit.
    """
    base = run.base_condition() or "ckpt_policy"
    suite = run.meta.suite or "suite"
    seeds = run.meta.seeds
    episodes = run.meta.episodes

    lines: list[str] = []
    lines.append(f"# Report Notes — {suite}")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Eval dir: `{run.evaldir}`")
    if run.meta.checkpoint:
        lines.append(f"- Checkpoint: `{run.meta.checkpoint}`")
    lines.append(f"- Base condition: `{base}`")
    if seeds:
        lines.append(f"- Seeds: `{seeds}` (n={len(seeds)})")
    if episodes is not None:
        lines.append(f"- Episodes per seed: `{episodes}`")
    if run.meta.device:
        lines.append(f"- Device: `{run.meta.device}`")
    lines.append("")
    lines.append("## Headline claim (write this last)")
    lines.append("")
    lines.append("- **Claim:** …")
    lines.append("- **Evidence:** cite Table/Fig refs produced by eval.")
    lines.append("")
    lines.append("## What changed across conditions?")
    lines.append("")
    lines.append("- ckpt_policy: trained ACPL policy")
    if run.meta.ablations_canonical:
        for a in run.meta.ablations_canonical:
            lines.append(f"- {a}: … (expected qualitative effect; sanity-check rationale)")
    lines.append("")
    lines.append("## Results summary (bullet points)")
    lines.append("")
    lines.append("- Base performance: …")
    lines.append("- Ablation deltas: …")
    lines.append("")
    lines.append("## Figures to reference")
    lines.append("")
    figdir = run.evaldir / "figs"
    if figdir.exists():
        for cond in run.condition_order():
            safe = _sanitize_filename(cond)
            pt = figdir / f"Pt__{safe}.png"
            tv = figdir / f"tv__{safe}.png"
            if pt.exists():
                lines.append(f"- `{_relpath(run.evaldir, pt)}` — Pt timeline ({cond})")
            if tv.exists():
                lines.append(f"- `{_relpath(run.evaldir, tv)}` — TV-to-uniform ({cond})")
    else:
        lines.append("- (no figs directory found)")
    lines.append("")
    lines.append("## Method notes / caveats")
    lines.append("")
    lines.append("- CI method & settings: (copy from eval_ci.txt) …")
    lines.append("- Any skipped conditions? …")
    lines.append("- Any task-specific interpretability notes (e.g., NodePermute timelines not meaningful) …")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report_template(evaldir: str | os.PathLike, out_path: str | os.PathLike) -> Path:
    run = load_eval_run(evaldir)
    txt = build_report_template(run)
    outp = _as_path(out_path)
    _write_text(outp, txt)
    return outp


# ----------------------------- CLI -----------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate research-ready reports/tables from scripts/eval.py outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--evaldir", type=str, required=True, help="Evaluation output directory (contains meta.json).")
    
    p.add_argument(
    "--outdir",
    type=str,
    default="",
    help="Directory to place outputs in. If omitted, defaults to --evaldir.",
    )

    
    
    p.add_argument("--out", type=str, default="", help="Write Markdown report to this path (optional).")
    p.add_argument("--latex", type=str, default="", help="Write LaTeX table to this path (optional).")
    p.add_argument("--csv_wide", type=str, default="", help="Write wide CSV to this path (optional).")
    p.add_argument("--csv_long", type=str, default="", help="Write long CSV to this path (optional).")
    p.add_argument("--template", type=str, default="", help="Write editable report template MD to this path (optional).")
    p.add_argument("--max_cols", type=int, default=6, help="Max metric columns in headline table.")
    p.add_argument("--digits", type=int, default=4, help="Digits for floating formatting in reports.")
    p.add_argument("--no_figs", action="store_true", help="Do not embed/discover figures in Markdown report.")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    evaldir = Path(args.evaldir)

    # Where outputs go:
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else evaldir.resolve()

    # If user provided relative filenames, place them under outdir
    def _resolve_out(p: str) -> str:
        if not p:
            return ""
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str(outdir / pp)

    # If *no* outputs requested, write a full default bundle into outdir.
    any_requested = any([args.out, args.latex, args.csv_wide, args.csv_long, args.template])
    if not any_requested:
        args.out = "REPORT.md"
        args.latex = "table.tex"
        args.csv_wide = "metrics_wide.csv"
        args.csv_long = "metrics_long.csv"
        args.template = "NOTES.template.md"

    # Resolve paths
    args.out = _resolve_out(args.out)
    args.latex = _resolve_out(args.latex)
    args.csv_wide = _resolve_out(args.csv_wide)
    args.csv_long = _resolve_out(args.csv_long)
    args.template = _resolve_out(args.template)

    did_any = False

    if args.out:
        write_markdown_report(
            str(evaldir),
            args.out,
            max_metric_cols=int(args.max_cols),
            digits=int(args.digits),
            include_figs=(not bool(args.no_figs)),
        )
        print(f"[ok] wrote markdown: {args.out}")
        did_any = True

    if args.latex:
        write_latex_table(str(evaldir), args.latex, digits=int(args.digits))
        print(f"[ok] wrote latex: {args.latex}")
        did_any = True

    if args.csv_wide:
        write_csv_wide(str(evaldir), args.csv_wide)
        print(f"[ok] wrote csv_wide: {args.csv_wide}")
        did_any = True

    if args.csv_long:
        write_csv_long(str(evaldir), args.csv_long)
        print(f"[ok] wrote csv_long: {args.csv_long}")
        did_any = True

    if args.template:
        write_report_template(str(evaldir), args.template)
        print(f"[ok] wrote template: {args.template}")
        did_any = True

    if not did_any:
        # Should not happen now, but keep a safe fallback.
        run = load_eval_run(str(evaldir))
        print(_json_dumps({"evaldir": str(run.evaldir), "conditions": run.condition_order()}))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

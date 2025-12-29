#!/usr/bin/env python3
"""
scripts/make_experiment_card.py

Write a 1-page JSON “card” per evaldir:
- run_id (stable)
- config hash
- meta (checkpoint/suite/seeds/etc.)
- headline metrics (auto-selected)
- condition summaries + CI

Output: reports/cards/<run_id>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from acpl.eval.reporting import load_eval_run, select_key_metrics


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, default=str)


def _sanitize(s: str) -> str:
    s = (s or "").strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    x = "".join(out).strip("_")
    while "__" in x:
        x = x.replace("__", "_")
    return x or "run"


def _config_hash(meta_raw: dict[str, Any]) -> str:
    blob = json.dumps(meta_raw or {}, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _run_id(meta_raw: dict[str, Any]) -> str:
    suite = str(meta_raw.get("suite") or "suite")
    ckpt = str(meta_raw.get("checkpoint") or "ckpt")
    ckpt_base = Path(ckpt).name if ckpt else "ckpt"
    h = _config_hash(meta_raw)
    return _sanitize(f"{suite}__{ckpt_base}__{h}")


def _discover_evaldirs(root: Path) -> list[Path]:
    out = [p.parent for p in root.rglob("meta.json")]
    out.sort(key=lambda p: str(p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Create experiment cards for many eval dirs.")
    ap.add_argument("--root", type=str, required=True, help="Root to scan (contains many eval/*/meta.json).")
    ap.add_argument("--outdir", type=str, default="reports/cards", help="Directory to write cards.")
    ap.add_argument("--max_cols", type=int, default=6, help="Headline metrics count.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for ed in _discover_evaldirs(root):
        try:
            run = load_eval_run(ed)
        except Exception:
            continue

        meta_raw = run.meta.raw or {}
        rid = _run_id(meta_raw)
        h = _config_hash(meta_raw)

        keys = select_key_metrics(run, max_cols=int(args.max_cols))

        card: dict[str, Any] = {
            "run_id": rid,
            "config_hash": h,
            "evaldir": str(run.evaldir),
            "meta": meta_raw,
            "headline_metrics": list(keys),
            "conditions": {},
        }

        for cond in run.condition_order():
            cr = run.conditions[cond]
            card["conditions"][cond] = {
                "policy": cr.policy,
                "baseline_kind": cr.baseline_kind,
                "ablation_meta": dict(cr.ablation_meta or {}),
                "headline": {
                    m: (cr.ci[m].mean if m in cr.ci else cr.summary_means.get(m, None))
                    for m in keys
                },
                "summary_means": dict(cr.summary_means or {}),
                "ci": {k: asdict(v) for k, v in (cr.ci or {}).items()},
            }

        outpath = outdir / f"{rid}.json"
        outpath.write_text(_json_dumps(card) + "\n", encoding="utf-8")

    print(f"[ok] cards written under: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

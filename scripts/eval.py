#!/usr/bin/env python3
"""
scripts/eval.py — Phase B7
Run evaluation suites from saved checkpoints with CI-style aggregation,
optional ablations, and plot/report export.

Highlights
- Loads checkpoints saved in a few common formats (dict, lightning-like, raw state_dict).
- Rebuilds the model from config when available (factory) or via callables baked into the checkpoint.
- Evaluates over multiple seeds and aggregates mean ± CI using acpl.eval.protocol.
- Optional ablations (NoPE, GlobalCoin, TimeFrozen, NodePermute) via acpl.eval.ablations.
- Saves JSONL per-episode logs, CSV summaries, and optional figures via acpl.eval.plots.
- Clean, explicit CLI with sensible defaults.

Example
-------
python scripts/eval.py \
  --ckpt ./runs/exp42/ckpts/last.pt \
  --suite basic_valid \
  --seeds 5 \
  --episodes 128 \
  --device cuda \
  --outdir ./eval/exp42 \
  --ablations NoPE TimeFrozen \
  --plots

"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import importlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires PyTorch (`pip install torch`)") from e

# ----------------------------- Utilities & IO ---------------------------------


def _as_path(p: str | os.PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _ensure_dir(p: str | os.PathLike) -> Path:
    path = _as_path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maybe_tqdm(iterable, desc: str = "", total: int | None = None):
    try:
        from tqdm.auto import tqdm  # lazy import

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _json_dump(obj: Any) -> str:
    class _NpEncoder(json.JSONEncoder):
        def default(self, o):  # noqa: N802
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            return super().default(o)

    return json.dumps(obj, cls=_NpEncoder)


# -------------------------- Checkpoint Loading --------------------------------


def _load_any_ckpt(ckpt_path: Path) -> dict[str, Any]:
    """
    Load a checkpoint file or directory.
    Supports:
      - torch.save dicts (with keys like 'state_dict', 'model', 'config', 'factory', etc.)
      - plain state_dict (mapping param->tensor)
      - directories containing last.pt or best.pt
    """
    if ckpt_path.is_dir():
        # prefer 'last.pt' then 'best.pt' then first *.pt/*.pth
        for name in ("last.pt", "best.pt", "checkpoint.pt", "last.pth", "best.pth"):
            cand = ckpt_path / name
            if cand.exists():
                ckpt_path = cand
                break
        else:
            pts = sorted(list(ckpt_path.glob("*.pt")) + list(ckpt_path.glob("*.pth")))
            if not pts:
                raise FileNotFoundError(f"No checkpoint file found in directory: {ckpt_path}")
            ckpt_path = pts[-1]

    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    # raw state dict stored alone
    if isinstance(obj, Mapping):
        return {"state_dict": obj}
    raise ValueError(f"Unsupported checkpoint object type: {type(obj)}")


def _try_import(path: str) -> Any:
    """
    Import a dotted path like 'acpl.models.factory.build_from_config'.
    Returns None if import fails.
    """
    try:
        mod_name, attr = path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    except Exception:
        return None


def _rebuild_model_from_ckpt(
    ckpt: dict[str, Any],
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Rebuild a model from a variety of checkpoint formats.

    Accepted patterns (best-effort, robust):
      1) ckpt['factory'] = 'acpl.models.factory.build_from_config', ckpt['config'] = {...}
      2) ckpt['model_cls'] = 'package.Class', ckpt['model_kwargs'] = {...}
      3) ckpt['model'] is a torch.nn.Module (already built)
      4) fallback to a generic factory: acpl.models.factory.build_from_config if found
      5) otherwise raise

    Always loads `state_dict` if present.
    Returns (model, meta_config).
    """
    meta_cfg = dict(ckpt.get("config", {}))

    # Case 3: a ready-built model object in the checkpoint
    if isinstance(ckpt.get("model", None), torch.nn.Module):
        model = ckpt["model"]
        sd = ckpt.get("state_dict", None)
        if sd:
            # Some frameworks save with 'module.' prefix; handle both
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                # de-prefix 'module.' if needed
                new_sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                model.load_state_dict(new_sd, strict=False)
        model.to(device)
        model.eval()
        return model, meta_cfg

    # Case 1: factory with config string path
    if "factory" in ckpt and "config" in ckpt and isinstance(ckpt["factory"], str):
        factory = _try_import(ckpt["factory"])
        if factory is None:
            raise ImportError(f"Cannot import factory: {ckpt['factory']}")
        model = factory(ckpt["config"])
        sd = ckpt.get("state_dict", None)
        if sd:
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                new_sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                model.load_state_dict(new_sd, strict=False)
        model.to(device)
        model.eval()
        return model, meta_cfg

    # Case 2: class path + kwargs
    if "model_cls" in ckpt:
        cls = _try_import(ckpt["model_cls"])
        if cls is None:
            raise ImportError(f"Cannot import model class: {ckpt['model_cls']}")
        kwargs = dict(ckpt.get("model_kwargs", {}))
        model = cls(**kwargs)
        sd = ckpt.get("state_dict", None)
        if sd:
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                new_sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                model.load_state_dict(new_sd, strict=False)
        model.to(device)
        model.eval()
        return model, meta_cfg

    # Case 4: generic factory discovery
    generic_factory = _try_import("acpl.models.factory.build_from_config")
    if generic_factory and meta_cfg:
        model = generic_factory(meta_cfg)
        sd = ckpt.get("state_dict", None)
        if sd:
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                new_sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                model.load_state_dict(new_sd, strict=False)
        model.to(device)
        model.eval()
        return model, meta_cfg

    # Case 5: only state dict without construction info
    if "state_dict" in ckpt:
        raise RuntimeError(
            "Checkpoint contains only a state_dict but no factory/class info. "
            "Please save checkpoints with either {'factory','config'} or {'model_cls','model_kwargs'}."
        )

    raise RuntimeError("Unrecognized checkpoint format; cannot rebuild the model.")


# -------------------------- Protocol: evaluation entry -------------------------


def _get_protocol_callable():
    """
    Probe acpl.eval.protocol for one of the expected entrypoints.
    We keep this permissive to match Phase B6 implementations.
    """
    try:
        proto_mod = importlib.import_module("acpl.eval.protocol")
    except Exception as e:
        raise ImportError("Could not import acpl.eval.protocol") from e

    candidates = (
        "run_ci_eval",
        "run_eval_protocol",
        "run_protocol",
        "evaluate",
        "run",
        "eval_entrypoint",
    )
    for name in candidates:
        if hasattr(proto_mod, name):
            return getattr(proto_mod, name)

    # As a fall-back, expose the module (the suite might need multiple calls)
    return proto_mod


def _maybe_get_ablations():
    try:
        return importlib.import_module("acpl.eval.ablations")
    except Exception:
        return None


def _maybe_get_plots():
    try:
        return importlib.import_module("acpl.eval.plots")
    except Exception:
        return None


# ------------------------------ CSV writer ------------------------------------


def _write_csv(rows: list[Mapping[str, Any]], out_csv: Path) -> None:
    if not rows:
        return
    # collect all keys
    header = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                header.append(k)
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for r in rows:
        line = []
        for k in header:
            v = r.get(k, "")
            if isinstance(v, (list, dict)):
                v = _json_dump(v)
            line.append(str(v))
        buf.write(",".join(line) + "\n")
    out_csv.write_text(buf.getvalue())


# ------------------------------ Main runner -----------------------------------


def run_eval(
    ckpt_path: Path,
    outdir: Path,
    suite: str,
    device: str = "cuda",
    seeds: int | Iterable[int] = 5,
    episodes: int = 128,
    ablations: list[str] | None = None,
    plots: bool = False,
    extra_overrides: dict[str, Any] | None = None,
) -> None:
    """
    High-level evaluation orchestrator.

    Parameters
    ----------
    ckpt_path: Path to checkpoint file/dir.
    outdir: Directory to write artifacts.
    suite: Name of evaluation suite defined by acpl.eval.protocol.
    device: 'cuda' or 'cpu'.
    seeds: int => range(seeds) or explicit iterable of ints.
    episodes: number of evaluation episodes per seed (if used by the suite).
    ablations: optional list of ablation names (NoPE, GlobalCoin, TimeFrozen, NodePermute, ...)
    plots: whether to generate figures (if plots module is available).
    extra_overrides: optional dict to pass through to the protocol entrypoint.

    Artifacts
    ---------
    outdir/
      ├── meta.json                  # ckpt + args
      ├── raw/
      │    └── logs.jsonl            # per-episode JSON lines
      ├── summary.csv                # flat summary across (seed, ablation)
      ├── summary.json               # structured summary (means, CIs, etc.)
      └── figs/                      # optional figures
    """
    outdir = _ensure_dir(outdir)
    (outdir / "raw").mkdir(exist_ok=True, parents=True)
    (outdir / "figs").mkdir(exist_ok=True, parents=True)

    # Load model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _load_any_ckpt(ckpt_path)
    model, model_cfg = _rebuild_model_from_ckpt(ckpt, device=device)

    # Discover protocol callable (or module)
    proto_entry = _get_protocol_callable()
    abl_mod = _maybe_get_ablations()
    plot_mod = _maybe_get_plots()

    # Normalize seeds
    if isinstance(seeds, int):
        seed_list = list(range(seeds))
    else:
        seed_list = list(seeds)

    # Common kwargs passed to protocol
    common_kwargs = {
        "model": model,
        "suite": suite,
        "device": device,
        "episodes": episodes,
        "seeds": seed_list,
    }
    if extra_overrides:
        common_kwargs.update(extra_overrides)

    # Write meta
    meta = {
        "checkpoint": str(ckpt_path),
        "device": device,
        "suite": suite,
        "episodes": episodes,
        "seeds": seed_list,
        "ablations": ablations or [],
        "model_config": model_cfg,
    }
    (outdir / "meta.json").write_text(_json_dump(meta))

    # Helper to run a single condition (possibly with an ablation applied)
    def run_one_condition(condition_tag: str, model_override: torch.nn.Module | None = None):
        tag_dir = _ensure_dir(outdir / "raw" / condition_tag)
        # call the protocol entry: support several signatures
        results = None
        kwargs = dict(common_kwargs)
        # if the protocol expects 'outdir' or 'logger' it can still log there; we pass tag_dir
        kwargs.setdefault("outdir", tag_dir)

        m_for_eval = model_override if model_override is not None else model

        if callable(proto_entry):
            # Signature variants; we detect by parameters
            try:
                results = proto_entry(model=m_for_eval, **kwargs)
            except TypeError:
                # Try a leaner signature
                try:
                    results = proto_entry(m_for_eval, **kwargs)
                except TypeError:
                    # Try without suite param (some versions might encode suite in kwargs)
                    kwargs2 = dict(kwargs)
                    kwargs2.pop("suite", None)
                    results = proto_entry(m_for_eval, **kwargs2)
        else:
            # Module-style API: prefer 'run_ci_eval' or 'run_eval_protocol' inside
            for name in ("run_ci_eval", "run_eval_protocol", "run_protocol", "evaluate", "run"):
                if hasattr(proto_entry, name):
                    fn = getattr(proto_entry, name)
                    try:
                        results = fn(model=m_for_eval, **kwargs)
                    except TypeError:
                        try:
                            results = fn(m_for_eval, **kwargs)
                        except TypeError:
                            kwargs2 = dict(kwargs)
                            kwargs2.pop("suite", None)
                            results = fn(m_for_eval, **kwargs2)
                    break
            else:
                raise RuntimeError("Could not find a runnable entrypoint in acpl.eval.protocol")

        # 'results' contract (best-effort):
        # - 'logs': list[dict] per-episode JSON objects (optional)
        # - 'summary': dict of scalar metrics (means, CIs) (required)
        # - 'per_seed': list[dict] per-seed summaries (optional)
        # Persist logs if present
        if isinstance(results, dict) and "logs" in results and results["logs"] is not None:
            jsonl = tag_dir / "logs.jsonl"
            with jsonl.open("w") as f:
                for row in results["logs"]:
                    f.write(_json_dump(row) + "\n")
        # Return summary and results for aggregation
        return results

    # Evaluate baseline (no ablation)
    summaries_for_csv: list[Mapping[str, Any]] = []
    structured_out: dict[str, Any] = {"conditions": {}}

    base_res = run_one_condition("baseline")
    base_summary = (base_res or {}).get("summary", {})
    structured_out["conditions"]["baseline"] = base_res or {}
    row = {"cond": "baseline"}
    row.update({f"metric.{k}": v for k, v in base_summary.items()})
    summaries_for_csv.append(row)

    # Ablations
    if ablations:
        for abl in ablations:
            cond_tag = f"abl_{abl}"
            model_ab = None
            if abl_mod is not None:
                try:
                    # apply_ablation(model, name) -> nn.Module (wrapped / configured)
                    if hasattr(abl_mod, "apply_ablation"):
                        model_ab = abl_mod.apply_ablation(model, abl)
                    elif hasattr(abl_mod, "get_ablation"):
                        model_ab = abl_mod.get_ablation(abl)(model)
                except Exception as e:
                    print(f"[warn] ablation '{abl}' could not be applied: {e}", file=sys.stderr)
            else:
                print(
                    "[warn] acpl.eval.ablations not found; skipping ablation models.",
                    file=sys.stderr,
                )

            res = run_one_condition(cond_tag, model_override=model_ab)
            structured_out["conditions"][cond_tag] = res or {}
            summ = (res or {}).get("summary", {})
            row = {"cond": cond_tag}
            row.update({f"metric.{k}": v for k, v in summ.items()})
            summaries_for_csv.append(row)

    # Write summaries
    _write_csv(summaries_for_csv, outdir / "summary.csv")
    (outdir / "summary.json").write_text(_json_dump(structured_out))

    # Optional plots
    if plots and plot_mod is not None:
        try:
            figdir = _ensure_dir(outdir / "figs")
            # We try a few common plotting helpers; each helper should be no-op safe
            if hasattr(plot_mod, "plot_tv_curves"):
                plot_mod.plot_tv_curves(structured_out, save_dir=figdir)
            if hasattr(plot_mod, "plot_Pt_timelines"):
                plot_mod.plot_Pt_timelines(structured_out, save_dir=figdir)
            if hasattr(plot_mod, "plot_robustness_sweeps"):
                plot_mod.plot_robustness_sweeps(structured_out, save_dir=figdir)
        except Exception as e:
            print(f"[warn] plot generation failed: {e}", file=sys.stderr)

    print(f"[OK] Evaluation complete. Artifacts at: {outdir}")


# --------------------------------- CLI ----------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run evaluation suites from saved checkpoints (Phase B7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file or directory.")
    p.add_argument(
        "--outdir", type=str, required=True, help="Directory to write evaluation artifacts."
    )
    p.add_argument("--suite", type=str, default="basic_valid", help="Evaluation suite name.")
    p.add_argument("--device", type=str, default=None, help="Device to use: 'cuda' or 'cpu'.")
    p.add_argument(
        "--seeds",
        type=str,
        default="5",
        help="Either an integer (e.g., '5' -> range(5)) or a comma-separated list (e.g., '0,1,2,5').",
    )
    p.add_argument(
        "--episodes", type=int, default=128, help="Episodes per seed (if used by the suite)."
    )
    p.add_argument(
        "--ablations",
        type=str,
        nargs="*",
        default=[],
        help="List of ablation names to evaluate (e.g., NoPE GlobalCoin TimeFrozen NodePermute).",
    )
    p.add_argument(
        "--plots", action="store_true", help="Generate figures via acpl.eval.plots if available."
    )
    p.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Extra key=val overrides passed to the evaluation protocol (e.g., task=search horizon=128).",
    )
    return p.parse_args(argv)


def _parse_seeds(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return [0]
    if "," in s:
        return [int(x) for x in s.split(",") if x != ""]
    # integer => range(N)
    try:
        n = int(s)
        return list(range(n))
    except Exception:
        # space-separated list?
        parts = s.split()
        return [int(x) for x in parts]


def _kv_overrides(pairs: list[str]) -> dict[str, Any]:
    out = {}
    for token in pairs:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Try to parse numbers/bools/json
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
            continue
        except Exception:
            pass
        # JSON?
        try:
            out[k] = json.loads(v)
            continue
        except Exception:
            pass
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ckpt_path = _as_path(args.ckpt)
    outdir = _as_path(args.outdir)
    seeds = _parse_seeds(args.seeds)
    overrides = _kv_overrides(args.override)

    run_eval(
        ckpt_path=ckpt_path,
        outdir=outdir,
        suite=args.suite,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        seeds=seeds,
        episodes=args.episodes,
        ablations=args.ablations or [],
        plots=bool(args.plots),
        extra_overrides=overrides,
    )


if __name__ == "__main__":
    main()

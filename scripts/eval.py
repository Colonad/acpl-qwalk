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
import hashlib
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


def _torch_load_cpu(path: Path):
    # Keep compatibility across torch versions regarding weights_only.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_ckpt_pointer(path: Path) -> Path | None:
    """
    Try to interpret `path` as a pointer/metadata file that references a real checkpoint.

    Supports:
      - plain text files containing a path (first non-empty line)
      - JSON containing keys like: ckpt/checkpoint/path/last/best/model
      - simple KEY=VALUE text (e.g., CKPT=model_last.pt)
    """
    if not path.exists() or not path.is_file():
        return None

    # Only try this for "pointer-ish" filetypes (keep it conservative)
    if path.suffix.lower() not in {".ckpt", ".txt", ".json"}:
        return None

    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return None

    # Try JSON first
    if path.suffix.lower() == ".json" or (txt.startswith("{") and txt.endswith("}")):
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                for key in ("ckpt", "checkpoint", "path", "last", "best", "model"):
                    v = obj.get(key, None)
                    if isinstance(v, str) and v.strip():
                        cand = Path(v.strip())
                        if not cand.is_absolute():
                            cand = (path.parent / cand).resolve()
                        return cand
        except Exception:
            pass

    # Try KEY=VALUE format
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            _, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v:
                cand = Path(v)
                if not cand.is_absolute():
                    cand = (path.parent / cand).resolve()
                return cand

    # Otherwise treat the first non-empty line as a path
    first = next((ln.strip() for ln in txt.splitlines() if ln.strip()), "")
    if not first:
        return None
    cand = Path(first)
    if not cand.is_absolute():
        cand = (path.parent / cand).resolve()
    return cand



def _peek_bytes(path: Path, n: int = 256) -> bytes:
    try:
        with path.open("rb") as f:
            return f.read(n)
    except Exception:
        return b""


def _is_git_lfs_pointer_bytes(b: bytes) -> bool:
    # Git LFS pointer files are tiny text files beginning with this line.
    return b.startswith(b"version https://git-lfs.github.com/spec/v1")


def _try_decode_text(b: bytes) -> str | None:
    try:
        return b.decode("utf-8", errors="strict")
    except Exception:
        return None


def _try_parse_text_pointer_file(path: Path) -> Path | None:
    """
    Some repos store a tiny text file that contains a relative/absolute path
    to the real checkpoint. If so, resolve it.
    """
    b = _peek_bytes(path, 1024)
    if not b:
        return None

    if _is_git_lfs_pointer_bytes(b):
        return None  # handled elsewhere with a clearer error

    s = _try_decode_text(b)
    if s is None:
        return None

    # Strip comments/blank lines; take first meaningful line as candidate path
    lines = []
    for ln in s.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        lines.append(ln)

    if not lines:
        return None

    cand = lines[0]

    # If it looks like a path, resolve relative to this file
    # (works for "model_last.pt", "./model_last.pt", "runs/.../model_last.pt", etc.)
    p = Path(cand)
    if not p.is_absolute():
        p = (path.parent / p).resolve()

    return p if p.exists() else None


def _discover_ckpt_candidates(run_dir: Path) -> list[Path]:
    """
    Return ordered candidate checkpoint paths under a run directory.
    Deduplicated and includes common subdirs.
    """
    run_dir = run_dir.resolve()
    search_dirs = [run_dir]
    for sub in ("ckpts", "ckpt", "checkpoints", "checkpoint"):
        d = run_dir / sub
        if d.exists() and d.is_dir():
            search_dirs.append(d)

    preferred_names = [
        "model_last.pt",
        "model_best.pt",
        "last.pt",
        "best.pt",
        "checkpoint.pt",
        "last.pth",
        "best.pth",
    ]

    found: list[Path] = []
    seen: set[str] = set()

    # Preferred exact names first
    for d in search_dirs:
        for nm in preferred_names:
            p = d / nm
            if p.exists() and p.is_file():
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(p)

    # Then any *.pt/*.pth (sorted)
    extra: list[Path] = []
    for d in search_dirs:
        extra.extend(list(d.glob("*.pt")))
        extra.extend(list(d.glob("*.pth")))
    for p in sorted(extra):
        if not p.exists() or not p.is_file():
            continue
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            found.append(p)

    return found


# -------------------------- Checkpoint Loading --------------------------------


def _load_any_ckpt(ckpt_path: Path) -> dict[str, Any]:
    """
    Load a checkpoint file or directory.

    Supports:
      - torch.save dict checkpoints
      - plain state_dict mappings
      - run directories (auto-pick model_last.pt / model_best.pt / etc.)
      - tiny text pointer files (containing a path)
      - Git LFS pointer detection with a clear error message
    """
    ckpt_path = _as_path(ckpt_path)

    # If a directory was passed, pick best candidate
    if ckpt_path.is_dir():
        cands = _discover_ckpt_candidates(ckpt_path)
        if not cands:
            raise FileNotFoundError(f"No checkpoint file found in directory: {ckpt_path}")
        ckpt_path = cands[0]

    # If a file was passed, it might itself be a tiny pointer-to-path file.
    if ckpt_path.exists() and ckpt_path.is_file():
        resolved = _try_parse_text_pointer_file(ckpt_path)
        if resolved is not None and resolved != ckpt_path:
            ckpt_path = resolved

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    if ckpt_path.is_dir():
        raise IsADirectoryError(f"Checkpoint path is a directory (expected file): {ckpt_path}")

    # Quick inspection to give better errors
    b = _peek_bytes(ckpt_path, 256)
    size = None
    try:
        size = ckpt_path.stat().st_size
    except Exception:
        pass

    if _is_git_lfs_pointer_bytes(b):
        # This is the #1 most common reason for "invalid magic number"
        msg = [
            f"Tried to torch.load '{ckpt_path}', but it is a Git LFS pointer file (not the real checkpoint).",
            "",
            "Fix:",
            "  1) Install Git LFS (once):  git lfs install",
            "  2) Fetch real files:        git lfs pull",
            "  3) Ensure checkout:         git lfs checkout",
            "",
            f"File size reported: {size} bytes" if size is not None else "File size: (unknown)",
        ]
        raise RuntimeError("\n".join(msg))

    # Attempt torch.load
    try:
        obj = _torch_load_cpu(ckpt_path)
    except Exception as e:
        # If it fails, give a more actionable error with file hints + nearby candidates
        parent = ckpt_path.parent
        nearby = _discover_ckpt_candidates(parent) if parent.exists() else []
        nearby_rel = [p.name for p in nearby[:10]]

        # Show a tiny preview (safe) to help diagnose corruption/pointer
        preview = b[:64]
        msg = [
            f"Tried to torch.load '{ckpt_path}', but it does not look like a torch checkpoint.",
            f"Original error: {type(e).__name__}: {e}",
            "",
            f"File size: {size} bytes" if size is not None else "File size: (unknown)",
            f"First 64 bytes: {preview!r}",
            "",
            "If this is a pointer file (like latest.ckpt), point eval to a real binary checkpoint",
            "such as model_last.pt / model_best.pt AFTER Git LFS pull, or re-run training to regenerate checkpoints.",
        ]
        if nearby_rel:
            msg.append("")
            msg.append("Nearby candidates:")
            for nm in nearby_rel:
                msg.append(f"  - {nm}")
        raise RuntimeError("\n".join(msg)) from e

    # Normalize return shape
    if isinstance(obj, dict):
        ckpt = obj
    elif isinstance(obj, Mapping):
        ckpt = {"state_dict": obj}
    else:
        raise ValueError(f"Unsupported checkpoint object type: {type(obj)}")

    # Optional: merge config from sibling meta json if present
    meta_path = ckpt_path.with_suffix(ckpt_path.suffix + ".meta.json")  # model_last.pt.meta.json
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                if "config" in meta and "config" not in ckpt:
                    ckpt["config"] = meta["config"]
                if "factory" in meta and "factory" not in ckpt:
                    ckpt["factory"] = meta["factory"]
        except Exception:
            pass

    return ckpt


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


# -------------------------- ACPL trainer-compatible eval -----------------------

import importlib.util
import inspect


_TRAIN_HELPERS = None


def _load_train_helpers():
    """
    Dynamically import scripts/train.py as a module so eval.py can reuse:
      - graph_from_config
      - datasets (SingleGraphEpisodeDataset / MarkedGraphEpisodeDataset)
      - rollouts (make_rollout_su2_dv2 / make_rollout_anydeg_exp_or_cayley)
      - dtype/device selectors, prereg coercion, override applier, etc.
    """
    global _TRAIN_HELPERS
    if _TRAIN_HELPERS is not None:
        return _TRAIN_HELPERS

    train_py = Path(__file__).resolve().parent / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"Expected training script at: {train_py}")

    spec = importlib.util.spec_from_file_location("_acpl_train_script", train_py)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load import spec for scripts/train.py")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    _TRAIN_HELPERS = mod
    return mod


def _normalize_state_dict_keys(sd: Mapping[str, Any]) -> dict[str, Any]:
    """
    Handle common prefixes (DDP 'module.' and torch.compile '_orig_mod.').
    """
    out: dict[str, Any] = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod.") :]
        out[nk] = v
    return out


def _try_apply_ema_shadow(model: torch.nn.Module, ema_shadow: Mapping[str, Any]) -> bool:
    """
    ema_shadow is stored as name->tensor from model.named_parameters() during training.
    Try to map names robustly and copy into the current model. Returns True if we applied anything.
    """
    if not ema_shadow:
        return False

    applied = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            candidates = (
                name,
                f"_orig_mod.{name}",
                name[len("_orig_mod.") :] if name.startswith("_orig_mod.") else None,
            )

            src = None
            for cand in candidates:
                if cand and cand in ema_shadow:
                    src = ema_shadow[cand]
                    break

            if src is None:
                continue

            try:
                p.copy_(src.to(device=p.device, dtype=p.dtype))
                applied += 1
            except Exception:
                continue

    return applied > 0




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


    # Prefer the repo's protocol API used by scripts/train.py
    try:
        from acpl.eval.protocol import EvalConfig, run_ci_eval, summarize_results
        from acpl.train.loops import LoopConfig
        from acpl.utils.logging import MetricLogger, MetricLoggerConfig
    except Exception:
        EvalConfig = None
        run_ci_eval = None
        summarize_results = None
        LoopConfig = None
        MetricLogger = None
        MetricLoggerConfig = None




    # Load checkpoint
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _load_any_ckpt(ckpt_path)

    # Try the ACPL-trainer-native path first (your checkpoints contain 'config')
    trainer_native_ok = (
        (run_ci_eval is not None)
        and isinstance(ckpt.get("config", None), dict)
    )

    abl_mod = _maybe_get_ablations()
    plot_mod = _maybe_get_plots()

    if trainer_native_ok:
        th = _load_train_helpers()

        # --- config ---
        cfg = dict(ckpt.get("config", {}))

        model_cfg = cfg

        # allow prereg YAML schema too (same as train.py)
        try:
            cfg = th._coerce_prereg_to_trainer_schema(cfg, cfg_path=_as_path(ckpt_path))
        except Exception:
            pass

        # apply CLI --override pairs as trainer-style dotted overrides
        if extra_overrides:
            ov_list = [f"{k}={v}" for k, v in extra_overrides.items()]
            try:
                cfg = th.apply_overrides(cfg, ov_list)
            except Exception:
                # if trainer override applier isn't present, ignore
                pass


        # Stable manifest hash (matches train.py behavior when available)
        manifest_hex = ckpt.get("manifest_hex", None)
        if not isinstance(manifest_hex, str) or not manifest_hex:
            try:
                import yaml as _yaml  # optional; train.py uses YAML for hashing
                blob = _yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
            except Exception:
                blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
            manifest_hex = hashlib.blake2b(blob, digest_size=16).hexdigest()


        seed_base = int(cfg.get("seed", 0))
        torch_device = th.select_device(device)
        dtype = th.select_dtype(cfg.get("dtype", "float32"))
        cdtype = th.select_cdtype(cfg.get("state_dtype", "complex64"))

        # --- task name / init mode ---
        task_cfg = cfg.get("task", {}) or {}
        if not isinstance(task_cfg, dict):
            task_cfg = {"name": str(task_cfg)}
        task_name = str(task_cfg.get("name", cfg.get("goal", "transfer"))).lower().strip()
        is_search = ("search" in task_name)
        is_mixing = ("mix" in task_name)
        is_robust = ("robust" in task_name)


        default_init = "uniform" if is_search else ("node" if is_robust else "node0")
        init_mode = str(task_cfg.get("init_state", default_init)).lower().strip()
        if init_mode not in ("uniform", "node0", "node"):
            init_mode = default_init
        # --- coin family ---
        coin_cfg = cfg.get("coin", {"family": "su2"}) or {}
        if not isinstance(coin_cfg, dict):
            coin_cfg = {"family": str(coin_cfg)}
        coin_family = str(coin_cfg.get("family", "su2")).lower().strip()

        # --- graph / portmap / shift (same undirected arc fix as train.py) ---
        data_cfg = cfg.get("data", {}) or {}
        if not isinstance(data_cfg, dict):
            data_cfg = {}
        g = th.graph_from_config(data_cfg, device=torch_device, dtype=dtype)

        pairs = list(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist(), strict=False))
        directed = bool((cfg.get("data", {}) or {}).get("directed", False))
        if not directed:
            pairs = list(set(pairs) | set((v, u) for (u, v) in pairs))
            pairs.sort()

        pm = th.build_portmap(pairs, num_nodes=g.N, coalesce=True)
        shift = th.build_shift(pm)

        # --- horizon T (mirror train.py behavior) ---
        sim_cfg = cfg.get("sim", {}) or {}
        steps_val = sim_cfg.get("steps", 64)
        if isinstance(steps_val, (list, tuple)) and len(steps_val) > 0:
            pref = task_cfg.get("horizon_T", None)
            T = int(pref) if pref is not None else int(max(int(x) for x in steps_val))
        else:
            T = int(task_cfg.get("horizon_T", steps_val))

        # --- node features X (match train.py: degree + coord1) ---
        N = int(g.N)
        deg_out = (pm.node_ptr[1:] - pm.node_ptr[:-1]).to(device=torch_device, dtype=dtype)
        deg_feat = deg_out.unsqueeze(1)
        coord1 = torch.arange(N, device=torch_device, dtype=dtype).unsqueeze(1) / max(N - 1, 1)
        X = torch.cat([deg_feat, coord1], dim=1)  # (N,2)

        # --- build model exactly like train.py ---
        model_cfg = cfg.get("model", {}) or {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        gnn_cfg = model_cfg.get("gnn", {}) or {}
        if not isinstance(gnn_cfg, dict):
            gnn_cfg = {}
        ctrl_cfg = model_cfg.get("controller", {}) or {}
        if not isinstance(ctrl_cfg, dict):
            ctrl_cfg = {}
        head_cfg = model_cfg.get("head", {}) or {}
        if not isinstance(head_cfg, dict):
            head_cfg = {}

        head_layernorm = bool(head_cfg.get("layernorm", True))
        if coin_family in ("exp", "cayley"):
            head_layernorm = False

        use_indicator = bool(task_cfg.get("use_indicator", is_search or is_robust))
        in_dim = 3 if use_indicator else 2

        acpl_cfg = th.ACPLPolicyConfig(
            in_dim=in_dim,
            gnn_hidden=int(gnn_cfg.get("hidden", 64)),
            gnn_out=int(gnn_cfg.get("out", gnn_cfg.get("hidden", 64))),
            gnn_activation="gelu",
            gnn_dropout=float(gnn_cfg.get("dropout", 0.0)),
            gnn_layernorm=True,
            gnn_residual=True,
            gnn_dropedge=0.0,
            controller=ctrl_cfg.get("kind", "gru"),
            ctrl_hidden=int(ctrl_cfg.get("hidden", gnn_cfg.get("hidden", 64))),
            ctrl_layers=int(ctrl_cfg.get("layers", 1)),
            ctrl_dropout=float(ctrl_cfg.get("dropout", 0.0)),
            ctrl_layernorm=True,
            ctrl_bidirectional=bool(ctrl_cfg.get("bidirectional", False)),
            time_pe_dim=int(model_cfg.get("time_pe_dim", 32)),
            time_pe_learned_scale=True,
            head_hidden=int(head_cfg.get("hidden", 0)) if "hidden" in head_cfg else 0,
            head_out_scale=1.0,
            head_layernorm=head_layernorm,
            head_dropout=float(head_cfg.get("dropout", 0.0)),
        )

        model = th.ACPLPolicy(acpl_cfg).to(device=torch_device).eval()

        # load weights (supports module./_orig_mod.)
        sd = ckpt.get("state_dict", None) or ckpt.get("model", None)
        if isinstance(sd, Mapping):
            sd2 = _normalize_state_dict_keys(sd)
            try:
                model.load_state_dict(sd2, strict=False)
            except Exception:
                model.load_state_dict(sd2, strict=False)

        # optional adaptor for exp/cayley
        adaptor = None
        if coin_family in ("exp", "cayley"):
            adaptor = th.ThetaToHermitianAdaptor(Kfreq=int(coin_cfg.get("Kfreq", 2))).to(device=torch_device).eval()
            ad_sd = ckpt.get("adaptor", None)
            if isinstance(ad_sd, Mapping):
                try:
                    adaptor.load_state_dict(ad_sd, strict=False)
                except Exception:
                    pass

        # optional EMA shadow (use if it matches enough params)
        ema_shadow = ckpt.get("ema_shadow", None)
        if isinstance(ema_shadow, Mapping):
            _try_apply_ema_shadow(model, ema_shadow)


        if is_search:
            marks_per_episode = int(data_cfg.get("marks_per_episode", task_cfg.get("marks_per_episode", 1)))
            payload = {
                "X": X,  # base (N,2); dataset appends mark channel -> (N,3)
                "edge_index": g.edge_index,
                "T": int(T),
                "targets": None,
            }

            def _make_eval_iter(seed_i: int):
                ds = th.MarkedGraphEpisodeDataset(
                    payload,
                    num_episodes=int(episodes),
                    N=g.N,
                    marks_per_episode=marks_per_episode,
                    manifest_hex=manifest_hex,
                    split="test",
                )
                ds.set_epoch(int(seed_i))  # disjoint deterministic episodes per CI seed
                return (ds[i] for i in range(len(ds)))

        elif is_robust:
            targets_per_episode = int(task_cfg.get("targets_per_episode", 1))
            random_start = bool(task_cfg.get("random_start", True))
            append_indicator = bool(task_cfg.get("use_indicator", True))
            payload = {
                "X": X,  # base (N,2); dataset may append target indicator -> (N,3)
                "edge_index": g.edge_index,
                "T": int(T),
                "targets": None,
                "start_node": 0,
            }

            def _make_eval_iter(seed_i: int):
                ds = th.RobustTargetEpisodeDataset(
                    payload,
                    num_episodes=int(episodes),
                    N=g.N,
                    targets_per_episode=targets_per_episode,
                    manifest_hex=manifest_hex,
                    split="test",
                    append_target_indicator=append_indicator,
                    random_start=random_start,
                )
                ds.set_epoch(int(seed_i))
                return (ds[i] for i in range(len(ds)))

        elif is_mixing:
            payload = {
                "X": X,
                "edge_index": g.edge_index,
                "T": int(T),
            }

            def _make_eval_iter(seed_i: int):
                ds = th.SingleGraphEpisodeDataset(payload, num_episodes=int(episodes))
                return (ds[i] for i in range(len(ds)))

        else:
            # transfer: fixed target window (only if config provides target_index)
            payload = {
                "X": X,
                "edge_index": g.edge_index,
                "T": int(T),
                "targets": None,
            }
            if "target_index" in task_cfg:
                target_index = int(task_cfg.get("target_index", g.N - 1)) % int(g.N)
                target_radius = int(task_cfg.get("target_radius", 0))
                lo = max(0, target_index - target_radius)
                hi = min(int(g.N), target_index + target_radius + 1)
                payload["targets"] = torch.arange(lo, hi, device=torch_device, dtype=torch.long)

            def _make_eval_iter(seed_i: int):
                ds = th.SingleGraphEpisodeDataset(payload, num_episodes=int(episodes))
                return (ds[i] for i in range(len(ds)))
        # --- rollout_fn (same as train.py) ---
        if coin_family == "su2":
            rollout_fn = th.make_rollout_su2_dv2(pm, shift, cdtype=cdtype, init_mode=init_mode)
            title_suffix = "SU2 (deg=2)"
        else:
            theta_scale = float(coin_cfg.get("theta_scale", 1.0))
            theta_noise_std = float(coin_cfg.get("theta_noise_std", 0.0))
            rollout_fn = th.make_rollout_anydeg_exp_or_cayley(
                pm,
                shift,
                adaptor=adaptor,
                family=coin_family,
                cdtype=cdtype,
                init_mode=init_mode,
                theta_scale=theta_scale,
                theta_noise_std=theta_noise_std,
            )
            title_suffix = f"{coin_family.upper()} (any degree)"

        # --- protocol configs ---
        if isinstance(seeds, int):
            seed_list = list(range(seeds))
        else:
            seed_list = list(seeds)

        eval_cfg = EvalConfig(
            seeds=seed_list,
            n_seeds=len(seed_list),
            device=str(torch_device),
            progress_bar=True,
            ci_method="bootstrap",
            ci_alpha=0.05,
            bootstrap_samples=1000,
            keep_per_seed_means=False,
        )

        loop_cfg = LoopConfig(
            device=str(torch_device),
            log_every=0,
            grad_clip=None,
            cvar_alpha=float(task_cfg.get("cvar_alpha", 0.1)),
            primary_on_targets=(not is_mixing),
            progress_bar=True,
            amp=False,
        )

        # we'll create one logger per condition in run_one_condition
        proto_entry = run_ci_eval

    else:
        # fallback to the original generic behavior
        model, model_cfg = _rebuild_model_from_ckpt(ckpt, device=device)
        proto_entry = _get_protocol_callable()


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
        
        
        # If trainer-native protocol is active, run_ci_eval returns a results dict keyed by metric.
        if trainer_native_ok and callable(proto_entry):
            # Create a logger that writes metrics.jsonl into this condition directory
            logger = None
            if MetricLogger is not None and MetricLoggerConfig is not None:
                logger = MetricLogger(
                    MetricLoggerConfig(
                        backend="plain",
                        log_dir=str(tag_dir),
                        project=None,
                        run_name=None,
                        step_key="step",
                        enable=True,
                    )
                )

            m_for_eval = model_override if model_override is not None else model

            results = proto_entry(
                model=m_for_eval,
                dataloader_factory=_make_eval_iter,
                rollout_fn=rollout_fn,
                loop_cfg=loop_cfg,
                eval_cfg=eval_cfg,
                logger=logger,
                step=0,
            )

            # summarize + save (match train.py artifact style)
            summary_text = ""
            if summarize_results is not None:
                summary_text = summarize_results(
                    results,
                    title=f"CI over pooled episodes — {condition_tag} — {title_suffix}",
                    show_per_seed=False,
                    ci_alpha=eval_cfg.ci_alpha,
                )
                (tag_dir / "eval_ci.txt").write_text(summary_text + "\n", encoding="utf-8")

            def _ci_to_dict(ci):
                return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "stderr": ci.stderr, "n": ci.n}

            json_payload = {k: _ci_to_dict(v["all"]) for k, v in results.items() if isinstance(v, dict) and "all" in v}
            (tag_dir / "eval_ci.json").write_text(_json_dump(json_payload), encoding="utf-8")

            if logger is not None:
                try:
                    logger.close()
                except Exception:
                    pass

            # Return in the same "dict-like" structure the rest of eval.py expects.
            return {"summary": {k: d["mean"] for k, d in json_payload.items()}, "ci": json_payload, "text": summary_text}
        
        
        
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

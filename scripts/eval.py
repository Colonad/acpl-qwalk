#!/usr/bin/env python3
"""
scripts/eval.py
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
import copy
import math
from dataclasses import dataclass

import os
from types import MethodType
from pathlib import Path
import sys
import subprocess

from typing import Any

import pickle
import re

import platform
import time
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



def _git_info(repo_root: Path) -> dict[str, Any]:
    """
    Best-effort git provenance for defendable artifacts.
    Safe on machines without git / outside a repo.
    """
    repo_root = _as_path(repo_root)
    out: dict[str, Any] = {"root": str(repo_root)}
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        out["commit"] = sha
        # dirty?
        dirty = subprocess.call(
            ["git", "-C", str(repo_root), "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        out["dirty"] = bool(dirty != 0)
    except Exception:
        out["commit"] = None
        out["dirty"] = None
    return out



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









# ----------------------- Pickle pointer / blob resolving -----------------------

_HEX_RE = re.compile(r"^[0-9a-fA-F]{16,128}$")


def _looks_like_pickle_bytes(b: bytes) -> bool:
    # pickle protocol header is typically: 0x80 <protocol>
    return len(b) >= 2 and b[0] == 0x80 and b[1] in (2, 3, 4, 5)


def _pickle_load_any(path: Path) -> Any | None:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _try_read_pickle_pointer_info(path: Path) -> tuple[str, str] | None:
    """
    Detect our common binary pointer format:
      {'codec': 'torch', 'hash': '<hex>'}  (and optionally other fields)
    Returns (codec, hash) if it matches.
    """
    head = _peek_bytes(path, 2)
    if not _looks_like_pickle_bytes(head):
        return None

    obj = _pickle_load_any(path)
    if not isinstance(obj, dict):
        return None

    codec = obj.get("codec", None)
    h = obj.get("hash", None)
    if not (isinstance(codec, str) and isinstance(h, str)):
        return None

    codec = codec.strip().lower()
    h = h.strip()
    if not _HEX_RE.match(h):
        return None

    return codec, h


def _try_resolve_pickle_pointer_file(path: Path) -> Path | None:
    """
    If `path` is a pickled pointer dict with a content hash, try to locate
    the real checkpoint blob nearby and return its Path.
    """
    info = _try_read_pickle_pointer_info(path)
    if info is None:
        return None

    codec, h = info
    # Only attempt resolution for torch-ish blobs
    if codec not in {"torch", "pytorch", "pt", "pth"}:
        return None

    # Search roots: run dir, its parent (often 'runs/'), and common subdirs.
    bases: list[Path] = []
    try:
        bases.append(path.parent.resolve())
    except Exception:
        bases.append(path.parent)

    if path.parent.parent is not None:
        bases.append(path.parent.parent)

    subdirs = [
        "", "ckpts", "ckpt", "checkpoints", "checkpoint",
        "blobs", "blob", "artifacts", "artifact",
        ".cache", "cache", "objects", "obj", "data",
    ]
    exts = (".pt", ".pth", ".bin", ".ckpt")

    search_dirs: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        for sd in subdirs:
            d = (base / sd) if sd else base
            if d.exists() and d.is_dir():
                key = str(d.resolve())
                if key not in seen:
                    seen.add(key)
                    search_dirs.append(d)

    # 1) Direct name matches: <hash>.pt / <hash>.pth / ...
    for d in search_dirs:
        for ext in exts:
            cand = d / f"{h}{ext}"
            if cand.exists() and cand.is_file():
                return cand

    # 2) Sharded storage: <dir>/<hh>/<rest> or <dir>/<hh>/<hash>.pt
    if len(h) >= 4:
        hh = h[:2]
        rest = h[2:]
        for d in search_dirs:
            cand = d / hh / rest
            if cand.exists() and cand.is_file():
                return cand
            for ext in exts:
                cand2 = d / hh / f"{h}{ext}"
                if cand2.exists() and cand2.is_file():
                    return cand2

    # 3) Any filename containing the hash
    for d in search_dirs:
        try:
            for ext in exts:
                for cand in d.glob(f"*{h}*{ext}"):
                    if cand.exists() and cand.is_file():
                        return cand
        except Exception:
            pass

    return None


def _try_load_raw_pickle_checkpoint(path: Path) -> dict[str, Any] | None:
    """
    Last-resort: if a file is a raw pickle of a state_dict (or a dict that contains one),
    load it and wrap into our normalized checkpoint shape.
    """
    head = _peek_bytes(path, 2)
    if not _looks_like_pickle_bytes(head):
        return None

    obj = _pickle_load_any(path)
    if obj is None:
        return None

    # Don't treat codec/hash pointer dicts as checkpoints
    if isinstance(obj, dict) and isinstance(obj.get("codec", None), str) and isinstance(obj.get("hash", None), str):
        return None

    # If they pickled a dict with a state_dict inside
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
            return obj
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], Mapping):
            out = dict(obj)
            out["state_dict"] = out.pop("model_state_dict")
            return out

    # If they pickled the state_dict mapping directly
    if isinstance(obj, Mapping):
        return {"state_dict": obj}

    return None














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

        resolved2 = _try_resolve_pickle_pointer_file(ckpt_path)
        if resolved2 is not None and resolved2 != ckpt_path:
            ckpt_path = resolved2


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
        # ---- NEW: handle pickled pointer/state_dict fallbacks ----
        obj2 = None

        # If this is a pickled pointer file, try to resolve to the real blob and load that.
        resolved2 = _try_resolve_pickle_pointer_file(ckpt_path)
        if resolved2 is not None and resolved2.exists() and resolved2.is_file():
            try:
                obj2 = _torch_load_cpu(resolved2)
                ckpt_path = resolved2  # keep internal path consistent after resolution
            except Exception:
                obj2 = None

        # If it’s a raw pickle checkpoint/state_dict, load it directly.
        if obj2 is None:
            obj2 = _try_load_raw_pickle_checkpoint(ckpt_path)

        if obj2 is not None:
            obj = obj2
        else:
            # If it fails, give a more actionable error with file hints + nearby candidates
            parent = ckpt_path.parent
            nearby = _discover_ckpt_candidates(parent) if parent.exists() else []
            nearby_rel = [p.name for p in nearby[:10]]

            preview = b[:64]
            ptr = _try_read_pickle_pointer_info(ckpt_path)
            ptr_note = ""
            if ptr is not None:
                codec, h = ptr
                ptr_note = f"\nDetected pickled pointer dict: codec={codec!r}, hash={h!r}\n"

            msg = [
                f"Tried to torch.load '{ckpt_path}', but it does not look like a torch checkpoint.",
                f"Original error: {type(e).__name__}: {e}",
                "",
                f"File size: {size} bytes" if size is not None else "File size: (unknown)",
                f"First 64 bytes: {preview!r}",
                ptr_note.strip(),
                "",
                "If this is a pointer/metadata file, point eval to the real binary checkpoint blob,",
                "or ensure training saves checkpoints via torch.save(...).",
                "If this was produced by a blob store, make sure the blob cache is present on this machine.",
            ]
            msg = [m for m in msg if m]  # drop empty lines

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

    # IMPORTANT: register the module before executing it.
    # dataclasses (and some typing machinery) expects sys.modules[mod.__name__] to exist.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception:
        # Avoid leaving a partially-imported module around
        sys.modules.pop(spec.name, None)
        raise

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


def _maybe_get_reporting():
    try:
        return importlib.import_module("acpl.eval.reporting")
    except Exception:
        return None




def _maybe_get_stats():
    try:
        return importlib.import_module("acpl.eval.stats")
    except Exception:
        return None


def _maybe_get_embeddings():
    try:
        return importlib.import_module("acpl.eval.embeddings")
    except Exception:
        return None






def _run_reporting_best_effort(report_mod: Any, evaldir: Path) -> bool:
    """
    Run acpl.eval.reporting in a signature-robust way.
    Returns True if something ran without error.
    """
    evaldir = _as_path(evaldir)

    # Prefer function-style APIs if present
    for fn_name in (
        "write_reports",
        "write_report",
        "generate_reports",
        "generate_report",
        "render_reports",
        "render_report",
        "build_reports",
        "build_report",
    ):
        fn = getattr(report_mod, fn_name, None)
        if callable(fn):
            try:
                _call_with_accepted_kwargs(fn, evaldir=evaldir, outdir=evaldir, path=evaldir)
                return True
            except Exception:
                pass

    # Fallback: module main(argv)
    main_fn = getattr(report_mod, "main", None)
    if callable(main_fn):
        for argv in (["--evaldir", str(evaldir)], [str(evaldir)], []):
            try:
                main_fn(argv)  # many scripts accept argv list
                return True
            except SystemExit:
                return True
            except Exception:
                continue

    return False





# --------------------------- Disorder + Robust sweeps ---------------------------

def _maybe_get_disorder():
    """
    Optional disorder plumbing module. Your Phase-6 work adds acpl/sim/disorder.py.
    Keep this best-effort so eval.py stays usable without it.
    """
    try:
        return importlib.import_module("acpl.sim.disorder")
    except Exception:
        return None


def _stable_seed_u64(*parts: Any) -> int:
    """
    Deterministic 64-bit seed derived from mixed parts (strings/numbers/dicts).
    Stable across runs/machines; safe for naming + RNG seeding.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            b = bytes(p)
        elif isinstance(p, (str, int, float, bool)) or p is None:
            b = str(p).encode("utf-8")
        else:
            # dict/list/etc -> stable json
            try:
                b = json.dumps(p, sort_keys=True, default=str).encode("utf-8")
            except Exception:
                b = repr(p).encode("utf-8")
        h.update(b)
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little", signed=False)


def _as_float_list(x: Any) -> list[float]:
    """
    Accept:
      - list/tuple of numbers/strings
      - comma-separated string "0,0.1,0.2"
      - space-separated string "0 0.1 0.2"
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # allow commas or whitespace
        toks = []
        if "," in s:
            toks = [t.strip() for t in s.split(",")]
        else:
            toks = s.split()
        out = []
        for t in toks:
            if not t:
                continue
            out.append(float(t))
        return out
    try:
        return [float(x)]
    except Exception:
        return []


def _normalize_disorder_cfg(d: Any) -> dict[str, Any]:
    """
    Ensure a dict shape. If absent, return {enabled: False}.
    """
    if isinstance(d, dict):
        dd = dict(d)
    else:
        dd = {}
    if "enabled" not in dd:
        dd["enabled"] = False
    return dd


def _override_disorder_for_sweep(
    base: dict[str, Any],
    *,
    kind: str,
    sigma: float,
) -> dict[str, Any]:
    """
    Create a new disorder cfg that enables ONLY the requested kind (edge_phase or coin_dephase)
    and sets its sigma, while preserving other knobs (mode/clamp/etc) from base.
    """
    out = copy.deepcopy(_normalize_disorder_cfg(base))
    out["enabled"] = True

    # normalize keys
    kind_l = (kind or "").strip().lower()

    # ensure subdicts exist
    out.setdefault("edge_phase", {})
    out.setdefault("coin_dephase", {})

    # disable all, then enable one
    if isinstance(out.get("edge_phase"), dict):
        out["edge_phase"]["enabled"] = False
    if isinstance(out.get("coin_dephase"), dict):
        out["coin_dephase"]["enabled"] = False

    if kind_l in ("edge_phase", "edgephase", "edge-phase"):
        if not isinstance(out["edge_phase"], dict):
            out["edge_phase"] = {}
        out["edge_phase"]["enabled"] = True
        out["edge_phase"]["sigma"] = float(sigma)
    elif kind_l in ("coin_dephase", "coindephase", "coin-dephase"):
        if not isinstance(out["coin_dephase"], dict):
            out["coin_dephase"] = {}
        out["coin_dephase"]["enabled"] = True
        out["coin_dephase"]["sigma"] = float(sigma)
    else:
        raise ValueError(f"Unknown sweep kind: {kind!r}")

    return out


def _build_shift_with_optional_disorder(
    th: Any,
    pm: Any,
    *,
    disorder_cfg: dict[str, Any] | None,
    disorder_seed: int | None,
    dis_mod: Any | None,
):
    """
    Best-effort: build shift that includes disorder.
    Supports multiple repo evolutions:
      - th.build_shift(pm, disorder=..., disorder_seed=...)
      - th.build_shift(pm, disorder=..., seed=...)
      - dis_mod.make_disordered_shift(...) / apply_to_shift(...) wrappers
    Falls back to plain th.build_shift(pm).
    """
    dcfg = _normalize_disorder_cfg(disorder_cfg)
    enabled = bool(dcfg.get("enabled", False))
    if not enabled:
        return th.build_shift(pm)

    # 1) Try passing kwargs into th.build_shift
    build = getattr(th, "build_shift", None)
    if callable(build):
        try:
            return _call_with_accepted_kwargs(build, pm=pm, disorder=dcfg, disorder_seed=disorder_seed, seed=disorder_seed)
        except TypeError:
            # older build_shift(pm) only
            pass
        except Exception:
            pass

    # 2) Fallback: build plain shift then let disorder module transform it
    shift0 = th.build_shift(pm)

    if dis_mod is not None:
        for fn_name in (
            "make_disordered_shift",
            "build_disordered_shift",
            "with_disorder",
            "apply_to_shift",
            "apply_shift_disorder",
        ):
            fn = getattr(dis_mod, fn_name, None)
            if callable(fn):
                try:
                    # Try common signatures
                    return _call_with_accepted_kwargs(
                        fn,
                        shift=shift0,
                        pm=pm,
                        disorder=dcfg,
                        cfg=dcfg,
                        seed=disorder_seed,
                        disorder_seed=disorder_seed,
                    )
                except Exception:
                    continue

    # 3) Last resort: return plain shift (but warn upstream)
    return shift0


def _build_rollout_fn_with_optional_disorder(
    th: Any,
    *,
    pm: Any,
    shift: Any,
    adaptor: Any | None,
    family: str,
    cdtype: Any,
    device: Any,
    init_mode: str,
    coin_cfg: dict[str, Any] | None,
    disorder_cfg: dict[str, Any] | None,
    disorder_seed: int | None,
):
    """
    Build rollout_fn for protocol evaluation, with best-effort disorder injection.
    We pass disorder=... and disorder_seed/seed=... through a kw-filtering caller,
    so older helpers safely ignore them.
    """
    fam = (family or "su2").lower().strip()
    ccfg = coin_cfg or {}
    dcfg = _normalize_disorder_cfg(disorder_cfg)

    if fam == "su2":
        return _call_positional_with_accepted_kwargs(
            th.make_rollout_su2_dv2,
            pm,
            shift,
            cdtype=cdtype,
            device=device,
            init_mode=init_mode,
            disorder=dcfg,
            disorder_seed=disorder_seed,
            seed=disorder_seed,
        )

    theta_scale = float(ccfg.get("theta_scale", 1.0))
    theta_noise_std = float(ccfg.get("theta_noise_std", 0.0))

    return _call_positional_with_accepted_kwargs(
        th.make_rollout_anydeg_exp_or_cayley,
        pm,
        shift,
        adaptor=adaptor,
        family=fam,
        cdtype=cdtype,
        device=device,
        init_mode=init_mode,
        theta_scale=theta_scale,
        theta_noise_std=theta_noise_std,
        disorder=dcfg,
        disorder_seed=disorder_seed,
        seed=disorder_seed,
    )


@dataclass(frozen=True)
class RobustSweepSpec:
    kind: str
    sigmas: list[float]
    trials: int
    bootstrap_samples: int


def _extract_robust_sweep_specs(
    cfg: dict[str, Any] | None,
    *,
    kinds_override: list[str] | None,
    sigmas_override: str | None,
    trials_override: int | None,
    bootstrap_samples: int,
) -> tuple[list[RobustSweepSpec], bool]:

    """
    Read cfg['eval']['robustness'] which you defined in YAML.
    Allows CLI overrides.
    """
    kinds = [k.strip() for k in (kinds_override or []) if str(k).strip()]
    if not kinds:
        # default: use whatever exists in config
        kinds = ["edge_phase", "coin_dephase"]

    sig_override = _as_float_list(sigmas_override) if sigmas_override else []

    root = {}
    if isinstance(cfg, dict):
        root = cfg.get("eval", {}) or {}
    rob = root.get("robustness", {}) if isinstance(root, dict) else {}
    enabled = bool(rob.get("enabled", False)) if isinstance(rob, dict) else False

    # If user requested sweeps via CLI, run even if YAML disabled.
    # If neither CLI nor YAML enabled, caller should skip.
    specs: list[RobustSweepSpec] = []
    for kind in kinds:
        k = kind.strip()
        k_l = k.lower()

        # pull from YAML if available
        sigmas = []
        trials = None

        if isinstance(rob, dict) and k_l in rob and isinstance(rob[k_l], dict):
            sigmas = _as_float_list(rob[k_l].get("sigma_grid", []))
            try:
                trials = int(rob[k_l].get("trials", 0))
            except Exception:
                trials = None

        if sig_override:
            sigmas = sig_override

        if not sigmas:
            # sensible default grid if none provided
            sigmas = [0.0, 0.05, 0.1, 0.15, 0.2]

        if trials_override is not None:
            trials = int(trials_override)

        if trials is None or trials <= 0:
            # robust default
            trials = 10

        specs.append(
            RobustSweepSpec(
                kind=k_l,
                sigmas=[float(s) for s in sigmas if math.isfinite(float(s)) and float(s) >= 0.0],
                trials=int(trials),
                bootstrap_samples=int(bootstrap_samples),
            )
        )

    return specs, enabled


def _call_plot_robustness_sweep_best_effort(
    fn,
    *,
    sigmas: list[float],
    metrics: dict[str, np.ndarray],
    savepath: Path,
    title: str,
    xlabel: str,
):
    """
    Best-effort adapter to call acpl.eval.plots.plot_robustness_sweep across signature variants.
    We expect metrics arrays shaped (K, M) where M=len(sigmas), K=trials (replicates).
    """
    x = np.asarray(sigmas, dtype=float)

    attempts = [
        # common keyword style
        ((), {"x": x, "metrics": metrics, "savepath": savepath, "title": title, "xlabel": xlabel}),
        ((), {"sigmas": x, "metrics": metrics, "savepath": savepath, "title": title, "xlabel": xlabel}),
        ((), {"xgrid": x, "metrics": metrics, "savepath": savepath, "title": title, "xlabel": xlabel}),
        # positional variants
        ((x, metrics), {"savepath": savepath, "title": title, "xlabel": xlabel}),
        ((x, metrics, savepath), {"title": title, "xlabel": xlabel}),
        # minimal
        ((x, metrics, savepath), {}),
        ((x, metrics), {}),
    ]

    last_err = None
    for args, kwargs in attempts:
        try:
            return _call_positional_with_accepted_kwargs(fn, *args, **kwargs)
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err


def _select_metrics_for_sweep_plot(metric_names: list[str], max_metrics: int = 6) -> list[str]:
    """
    Keep plots readable: prioritize success/tv/cvar-like metrics, then fill.
    """
    prio = []
    rest = []
    for m in metric_names:
        ml = m.lower()
        if any(k in ml for k in ("succ", "success", "tv", "cvar", "regret", "hit", "mix")):
            prio.append(m)
        else:
            rest.append(m)
    out = prio + rest
    return out[: max_metrics]


def _run_robustness_sweeps(
    *,
    outdir: Path,
    cond_runners: Mapping[str, dict[str, Any]],
    cfg: dict[str, Any],
    th: Any,
    pm: Any,
    coin_family: str,
    coin_cfg: dict[str, Any],
    adaptor: Any | None,
    cdtype: Any,
    device: Any,
    init_mode: str,
    seeds: list[int],
    episodes: int,
    loop_cfg: Any,
    EvalConfig: Any,
    proto_entry: Any,
    compute_ci: Any | None,
    plot_mod: Any | None,
    manifest_hex: str,
    seed_base: int,
    kinds_override: list[str] | None,
    sigmas_override: str | None,
    trials_override: int | None,
    bootstrap_samples: int,
    ci_alpha: float,
    max_plot_metrics: int = 6,
) -> dict[str, Any]:
    """
    Runs sigma sweeps for configured disorder kinds.
    Writes:
      outdir/robust_sweeps/<cond>/<kind>/{meta.json, trials.jsonl, summary.json, sweep.png?}
    Returns a compact dict to embed into summary.json.
    """
    outdir = _as_path(outdir)
    dis_mod = _maybe_get_disorder()

    base_disorder = _normalize_disorder_cfg(cfg.get("disorder", {}) if isinstance(cfg, dict) else {})

    specs, yaml_enabled = _extract_robust_sweep_specs(
        cfg,
        kinds_override=kinds_override,
        sigmas_override=sigmas_override,
        trials_override=trials_override,
        bootstrap_samples=bootstrap_samples,
    )

    sweep_root = _ensure_dir(outdir / "robust_sweeps")
    summary_out: dict[str, Any] = {"yaml_enabled": bool(yaml_enabled), "conditions": {}}

    for cond_tag, rr in cond_runners.items():
        safe_cond = _sanitize_filename(cond_tag)
        cond_dir = _ensure_dir(sweep_root / safe_cond)
        m = rr["model"]
        dl_factory = rr["dataloader_factory"]

        cond_block: dict[str, Any] = {"kinds": {}}

        for spec in specs:
            kind = spec.kind
            sigmas = list(spec.sigmas)
            trials = int(spec.trials)

            if not sigmas:
                continue

            kind_dir = _ensure_dir(cond_dir / kind)
            trials_jsonl = kind_dir / "trials.jsonl"

            # Build a sweep-local eval config: same seeds, fewer bootstraps if user requested.
            eval_cfg_sweep = EvalConfig(
                seeds=[int(s) for s in seeds],
                n_seeds=len(seeds),
                device=str(device),
                progress_bar=False,
                ci_method="bootstrap",
                ci_alpha=float(ci_alpha),
                bootstrap_samples=int(spec.bootstrap_samples),
                keep_per_seed_means=(len(seeds) >= 2),
            )

            meta = {
                "cond": cond_tag,
                "kind": kind,
                "sigmas": sigmas,
                "trials": trials,
                "episodes": int(episodes),
                "seeds": [int(s) for s in seeds],
                "bootstrap_samples_per_trial": int(spec.bootstrap_samples),
                "manifest_hex": manifest_hex,
                "seed_base": int(seed_base),
                "base_disorder": base_disorder,
                "note": "Each trial uses a distinct disorder_seed; episodes/seeds still control dataset randomness.",
            }
            (kind_dir / "meta.json").write_text(_json_dump(meta), encoding="utf-8")

            # Collect raw matrices: metric -> (trials, len(sigmas))
            metric_names: list[str] = []
            mats: dict[str, np.ndarray] = {}
            # To ensure consistent metric set, we discover metrics on first run.
            discovered = False

            with trials_jsonl.open("w", encoding="utf-8") as f:
                for j, sigma in enumerate(sigmas):
                    for t in range(trials):
                        disorder_seed = _stable_seed_u64(
                            "robust_sweep",
                            manifest_hex,
                            cond_tag,
                            kind,
                            float(sigma),
                            int(t),
                            int(seed_base),
                        ) & ((1 << 63) - 1)

                        dcfg = _override_disorder_for_sweep(base_disorder, kind=kind, sigma=float(sigma))

                        shift_t = _build_shift_with_optional_disorder(
                            th,
                            pm,
                            disorder_cfg=dcfg,
                            disorder_seed=int(disorder_seed),
                            dis_mod=dis_mod,
                        )

                        rollout_t = _build_rollout_fn_with_optional_disorder(
                            th,
                            pm=pm,
                            shift=shift_t,
                            adaptor=adaptor,
                            family=coin_family,
                            cdtype=cdtype,
                            device=device,
                            init_mode=init_mode,
                            coin_cfg=coin_cfg,
                            disorder_cfg=dcfg,
                            disorder_seed=int(disorder_seed),
                        )

                        # Run protocol
                        results = proto_entry(
                            model=m,
                            dataloader_factory=dl_factory,
                            rollout_fn=rollout_t,
                            loop_cfg=loop_cfg,
                            eval_cfg=eval_cfg_sweep,
                            logger=None,
                            step=0,
                        )

                        # Discover metric names and allocate matrices once
                        if not discovered:
                            metric_names = [
                                str(k) for k, v in results.items()
                                if isinstance(v, dict) and "all" in v and hasattr(v["all"], "mean")
                            ]
                            for mn in metric_names:
                                mats[mn] = np.full((trials, len(sigmas)), np.nan, dtype=float)
                            discovered = True

                        # Record per-trial pooled means (+ CI if available)
                        row = {
                            "sigma": float(sigma),
                            "trial": int(t),
                            "disorder_seed": int(disorder_seed),
                            "metrics": {},
                        }
                        for mn in metric_names:
                            vv = results.get(mn, None)
                            if isinstance(vv, dict) and "all" in vv and hasattr(vv["all"], "mean"):
                                mean_v = float(vv["all"].mean)
                                mats[mn][t, j] = mean_v
                                row["metrics"][mn] = {
                                    "mean": mean_v,
                                    "lo": float(getattr(vv["all"], "lo", mean_v)),
                                    "hi": float(getattr(vv["all"], "hi", mean_v)),
                                    "stderr": float(getattr(vv["all"], "stderr", 0.0)),
                                    "n": int(getattr(vv["all"], "n", 0)),
                                }

                        f.write(_json_dump(row) + "\n")

            # Summarize across trials at each sigma (CI over trials)
            summary_kind: dict[str, Any] = {"sigmas": sigmas, "trials": trials, "metrics": {}}
            for mn, mat in mats.items():
                means = []
                los = []
                his = []
                ns = []
                for j in range(mat.shape[1]):
                    col = mat[:, j]
                    col = col[np.isfinite(col)]
                    ns.append(int(col.shape[0]))
                    if col.shape[0] == 0:
                        means.append(float("nan"))
                        los.append(float("nan"))
                        his.append(float("nan"))
                        continue
                    if compute_ci is not None and col.shape[0] >= 2:
                        ci = compute_ci(np.asarray(col, dtype=float), method="student_t", alpha=float(ci_alpha))
                        means.append(float(ci.mean))
                        los.append(float(ci.lo))
                        his.append(float(ci.hi))
                    else:
                        # fallback: no CI
                        m0 = float(np.mean(col))
                        means.append(m0)
                        los.append(m0)
                        his.append(m0)

                summary_kind["metrics"][mn] = {
                    "mean": means,
                    "lo": los,
                    "hi": his,
                    "n": ns,
                    "ci_over": "trials",
                    "ci_method": "student_t" if compute_ci is not None else "none",
                    "alpha": float(ci_alpha),
                }

            (kind_dir / "summary.json").write_text(_json_dump(summary_kind), encoding="utf-8")

            # Plot sweep (best-effort)
            if plot_mod is not None and hasattr(plot_mod, "plot_robustness_sweep") and discovered:
                fn = getattr(plot_mod, "plot_robustness_sweep")
                chosen = _select_metrics_for_sweep_plot(metric_names, max_metrics=max_plot_metrics)
                metrics_for_plot = {mn: mats[mn] for mn in chosen if mn in mats}

                if metrics_for_plot:
                    try:
                        _call_plot_robustness_sweep_best_effort(
                            fn,
                            sigmas=sigmas,
                            metrics=metrics_for_plot,
                            savepath=(kind_dir / f"sweep__{safe_cond}__{kind}.png"),
                            title=f"Robustness sweep — {cond_tag} — {kind}",
                            xlabel=f"{kind}.sigma",
                        )
                    except Exception as e:
                        print(f"[warn] robustness sweep plot failed for {cond_tag}/{kind}: {e}", file=sys.stderr)

            cond_block["kinds"][kind] = {
                "dir": str(kind_dir.relative_to(outdir)),
                "sigmas": sigmas,
                "trials": trials,
                "metrics": list(mats.keys()),
            }

        summary_out["conditions"][cond_tag] = cond_block

    return summary_out




# ----------------------- First-class artifacts: stats + embeddings -----------------------

def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str | None:
    """
    Best-effort sha256 for provenance. Returns None if unreadable.
    """
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                b = f.read(chunk_bytes)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def _env_info() -> dict[str, Any]:
    """
    Machine/software provenance for defendable results.
    """
    out: dict[str, Any] = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "numpy": getattr(np, "__version__", None),
    }
    try:
        out["torch"] = getattr(torch, "__version__", None)
        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        out["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        out["torch_cudnn_version"] = (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    except Exception:
        pass
    return out


def _resolve_attr_path(root: Any, path: str) -> Any | None:
    """
    Resolve dotted attribute path like "gnn" or "encoder.gnn".
    Returns None if any hop fails.
    """
    cur = root
    for part in (path or "").split("."):
        part = part.strip()
        if not part:
            continue
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _unwrap_tensor(out: Any) -> torch.Tensor | None:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    return None


def _encode_nodes_best_effort(
    model: torch.nn.Module,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    T: int | None,
    cfg: dict[str, Any] | None,
) -> torch.Tensor:
    """
    Stable hook for node embeddings.

    Preferred:
      - model.encode_nodes(X, edge_index, T=...)
    Accepts fallback hook path:
      - cfg['model']['hook_module_path'] or cfg['eval']['hook_module_path'] (e.g., "gnn")

    Then tries common module/method names.
    Returns embeddings shaped (N, D).
    """
    hook_path = None
    if isinstance(cfg, dict):
        try:
            hook_path = (cfg.get("eval", {}) or {}).get("hook_module_path", None)
        except Exception:
            hook_path = None
        if hook_path is None:
            try:
                hook_path = (cfg.get("model", {}) or {}).get("hook_module_path", None)
            except Exception:
                hook_path = None
    hook_path = str(hook_path).strip() if hook_path else ""

    # Candidate (object, method_name) pairs in priority order
    cands: list[tuple[Any, str]] = []

    # 0) explicit hook module path (if provided)
    if hook_path:
        m0 = _resolve_attr_path(model, hook_path)
        if m0 is not None:
            cands.append((m0, "encode_nodes"))
            cands.append((m0, "encode"))
            cands.append((m0, "forward_features"))
            cands.append((m0, "node_embeddings"))
            cands.append((m0, "get_node_embeddings"))

    # 1) model-level hook
    cands.append((model, "encode_nodes"))
    cands.append((model, "node_embeddings"))
    cands.append((model, "get_node_embeddings"))

    # 2) common submodules
    for attr in ("gnn", "encoder", "backbone", "net", "mpnn"):
        if hasattr(model, attr):
            mm = getattr(model, attr)
            cands.append((mm, "encode_nodes"))
            cands.append((mm, "encode"))
            cands.append((mm, "forward_features"))
            cands.append((mm, "node_embeddings"))

    # Try calls with signature adaptation
    last_err: Exception | None = None
    for obj, meth in cands:
        fn = getattr(obj, meth, None)
        if not callable(fn):
            continue

        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        # positional
        attempts.append(((X, edge_index), {}))
        if T is not None:
            attempts.append(((X, edge_index), {"T": int(T)}))
        # keyword
        attempts.append(((), {"X": X, "edge_index": edge_index}))
        if T is not None:
            attempts.append(((), {"X": X, "edge_index": edge_index, "T": int(T)}))

        for args, kwargs in attempts:
            try:
                out = _call_positional_with_accepted_kwargs(fn, *args, **kwargs)
                emb = _unwrap_tensor(out)
                if emb is None:
                    continue
                # Accept (B,N,D) or (N,D)
                if emb.ndim == 3:
                    emb = emb.mean(dim=0)
                if emb.ndim != 2:
                    continue
                return emb
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(
        "Could not obtain node embeddings. "
        "Implement model.encode_nodes(X, edge_index, T=...) or set cfg.eval.hook_module_path='gnn'. "
        + (f"Last error: {type(last_err).__name__}: {last_err}" if last_err else "")
    )


def _collect_embedding_samples_for_artifacts(
    *,
    model: torch.nn.Module,
    dataloader_factory: Any,
    seeds: list[int],
    episodes: int,
    cfg: dict[str, Any] | None,
    device: Any,
    T_default: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Returns embeddings shaped (K, N, D) + meta.

    Defendable CI convention:
      - seeds>=2: K=#seeds, each sample is per-seed mean over episodes
      - seeds==1: K=#episodes (CI across episodes)
    """
    model.eval()
    seeds = list(seeds or [])
    if not seeds:
        raise RuntimeError("No seeds provided for embedding artifacts.")
    episodes = int(max(1, episodes))

    def _get_T(batch: Mapping[str, Any]) -> int | None:
        try:
            return int(batch.get("T", T_default if T_default is not None else 0))
        except Exception:
            return T_default

    with torch.no_grad():
        # --- single seed => per-episode embeddings ---
        if len(seeds) == 1:
            s = int(seeds[0])
            it = dataloader_factory(s)
            embs: list[np.ndarray] = []
            n = 0
            for batch in it:
                if not isinstance(batch, Mapping):
                    continue
                X = batch.get("X", None)
                edge_index = batch.get("edge_index", None)
                if not (isinstance(X, torch.Tensor) and isinstance(edge_index, torch.Tensor)):
                    continue
                X = X.to(device=device)
                edge_index = edge_index.to(device=device)
                T_use = _get_T(batch)
                E = _encode_nodes_best_effort(model, X, edge_index, T=T_use, cfg=cfg)
                embs.append(E.detach().cpu().numpy())
                n += 1
                if n >= episodes:
                    break
            if not embs:
                raise RuntimeError("No embeddings collected (single-seed mode).")
            arr = np.stack(embs, axis=0)  # (K=episodes, N, D)
            meta = {"ci_mode": "episodes", "n_samples": int(arr.shape[0]), "seed": s}
            return arr, meta

        # --- multi-seed => per-seed mean embeddings ---
        per_seed: list[np.ndarray] = []
        kept_seeds: list[int] = []
        for s in seeds:
            it = dataloader_factory(int(s))
            acc = None
            n = 0
            for batch in it:
                if not isinstance(batch, Mapping):
                    continue
                X = batch.get("X", None)
                edge_index = batch.get("edge_index", None)
                if not (isinstance(X, torch.Tensor) and isinstance(edge_index, torch.Tensor)):
                    continue
                X = X.to(device=device)
                edge_index = edge_index.to(device=device)
                T_use = _get_T(batch)
                E = _encode_nodes_best_effort(model, X, edge_index, T=T_use, cfg=cfg)  # (N,D)
                acc = E if acc is None else (acc + E)
                n += 1
                if n >= episodes:
                    break
            if acc is None or n == 0:
                continue
            per_seed.append((acc / float(n)).detach().cpu().numpy())
            kept_seeds.append(int(s))

        if not per_seed:
            raise RuntimeError("No embeddings collected (multi-seed mode).")
        arr = np.stack(per_seed, axis=0)  # (K=seeds, N, D)
        meta = {"ci_mode": "seeds", "n_samples": int(arr.shape[0]), "seeds": kept_seeds}
        return arr, meta


def _embedding_stats(emb: np.ndarray) -> dict[str, Any]:
    """
    Compute robust embedding summaries without ddof warnings.
    emb: (K, N, D)
    """
    K, N, D = emb.shape
    flat = emb.reshape(K * N, D)
    mean = np.nanmean(flat, axis=0)
    var = np.nanvar(flat, axis=0, ddof=1) if (flat.shape[0] > 1) else np.nanvar(flat, axis=0, ddof=0)
    norm = np.linalg.norm(flat, axis=1)
    out = {
        "K": int(K),
        "N": int(N),
        "D": int(D),
        "mean_per_dim": mean.tolist(),
        "var_per_dim": var.tolist(),
        "mean_norm": float(np.nanmean(norm)),
        "std_norm": float(np.nanstd(norm, ddof=1) if norm.shape[0] > 1 else np.nanstd(norm, ddof=0)),
        "finite_frac": float(np.isfinite(flat).mean()),
    }
    return out


def _pca2_numpy(X: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """
    PCA to 2D via SVD. Returns (N,2) coords and explained variance ratio (len=2).
    """
    X = np.asarray(X, dtype=float)
    X = X - np.mean(X, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:2].T
    ev = (S**2)
    denom = float(ev.sum()) if float(ev.sum()) > 0 else 1.0
    r = [(float(ev[i]) / denom) for i in range(min(2, len(ev)))]
    while len(r) < 2:
        r.append(0.0)
    return Z, r


def _save_pca_scatter_png(savepath: Path, Z: np.ndarray, *, title: str) -> None:
    """
    Best-effort headless scatter for PCA.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(Z[:, 0], Z[:, 1], s=6)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(savepath, dpi=200)
        plt.close()
    except Exception:
        pass


def _write_embeddings_artifacts_fallback(
    *,
    outdir: Path,
    cond: str,
    emb: np.ndarray,
    meta: dict[str, Any],
    note: str,
) -> dict[str, Any]:
    """
    Always-available embeddings artifact writer (no dependency on acpl.eval.embeddings).
    """
    root = _ensure_dir(outdir / "artifacts" / "embeddings" / _sanitize_filename(cond))
    (root / "meta.json").write_text(_json_dump({"cond": cond, "meta": meta, "note": note}), encoding="utf-8")

    np.save(root / "embeddings.npy", emb)                 # (K,N,D)
    meanE = np.mean(emb, axis=0)                          # (N,D)
    np.save(root / "embeddings_mean.npy", meanE)

    Z, evr = _pca2_numpy(meanE)
    np.save(root / "pca2.npy", Z)
    (root / "pca2_meta.json").write_text(_json_dump({"explained_var_ratio_2d": evr}), encoding="utf-8")
    _save_pca_scatter_png(root / "pca2_scatter.png", Z, title=f"PCA2 embeddings — {cond}")

    return {
        "dir": str(root.relative_to(outdir)),
        "files": ["meta.json", "embeddings.npy", "embeddings_mean.npy", "pca2.npy", "pca2_meta.json", "pca2_scatter.png"],
    }


def _write_stats_artifacts(
    *,
    outdir: Path,
    stats_payload: dict[str, Any],
) -> None:
    root = _ensure_dir(outdir / "artifacts")
    (root / "stats.json").write_text(_json_dump(stats_payload), encoding="utf-8")

    # Also produce a human-readable table (defendable for thesis appendix)
    lines: list[str] = []
    lines.append("EVAL STATS (first-class artifacts)")
    lines.append("")
    env = stats_payload.get("env", {})
    lines.append(f"UTC: {env.get('time_utc', '')}")
    lines.append(f"Python: {env.get('python', '')}")
    lines.append(f"Torch: {env.get('torch', '')} CUDA_avail={env.get('torch_cuda_available', '')}")
    lines.append("")

    conds = stats_payload.get("conditions", {})
    for cond, block in conds.items():
        lines.append(f"[{cond}]")
        evalm = block.get("eval_metrics", {})
        if evalm:
            lines.append("  Metrics:")
            for k, v in evalm.items():
                lines.append(f"    - {k}: {v}")
        emb = block.get("embeddings", None)
        if isinstance(emb, dict):
            lines.append("  Embeddings:")
            lines.append(f"    - ci_mode: {emb.get('ci_mode')}  K={emb.get('K')}  N={emb.get('N')}  D={emb.get('D')}")
            lines.append(f"    - mean_norm: {emb.get('mean_norm'):.6g}  std_norm: {emb.get('std_norm'):.6g}")
            lines.append(f"    - finite_frac: {emb.get('finite_frac'):.3g}")
        lines.append("")

    (root / "stats_table.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_first_class_artifacts(
    *,
    outdir: Path,
    cfg: dict[str, Any] | None,
    conditions: Mapping[str, dict[str, Any]],
    structured_out: dict[str, Any],
    seeds: list[int],
    episodes: int,
    device: Any,
    T_default: int | None,
    max_nodes: int,
    want_embeddings: bool,
    want_stats: bool,
    emb_mod: Any | None,
    stats_mod: Any | None,
) -> None:
    """
    Creates:
      outdir/artifacts/stats.json
      outdir/artifacts/stats_table.txt
      outdir/artifacts/embeddings/<cond>/*

    Uses acpl.eval.embeddings / acpl.eval.stats if present, else falls back to built-ins.
    """
    outdir = _as_path(outdir)
    _ensure_dir(outdir / "artifacts")

    payload: dict[str, Any] = {
        "env": _env_info(),
        "git": _git_info(Path(__file__).resolve().parents[1]),
        "cfg_manifest_hex": None,
        "conditions": {},
    }

    # Manifest hash (stable)
    if isinstance(cfg, dict):
        try:
            import yaml as _yaml
            blob = _yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
        except Exception:
            blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
        payload["cfg_manifest_hex"] = hashlib.blake2b(blob, digest_size=16).hexdigest()

    for cond, rr in conditions.items():
        block: dict[str, Any] = {}

        # Eval metrics (from structured_out)
        try:
            cres = (structured_out.get("conditions", {}) or {}).get(cond, {})
            summary = (cres or {}).get("summary", {}) or {}
            block["eval_metrics"] = {str(k): float(v) for k, v in summary.items()} if isinstance(summary, dict) else {}
        except Exception:
            block["eval_metrics"] = {}

        # Embeddings
        if want_embeddings:
            try:
                m = rr["model"]
                dl = rr["dataloader_factory"]
                # Safety: skip huge graphs
                try:
                    # probe one batch to see N
                    it0 = dl(int(seeds[0])) if seeds else dl(0)
                    b0 = next(iter(it0))
                    X0 = b0.get("X", None) if isinstance(b0, Mapping) else None
                    N0 = int(X0.shape[-2]) if isinstance(X0, torch.Tensor) and X0.ndim >= 2 else None
                    if (N0 is not None) and (N0 > int(max_nodes)):
                        raise RuntimeError(f"N={N0} exceeds embeddings_max_nodes={max_nodes}; skipping embeddings for {cond}.")
                except StopIteration:
                    raise RuntimeError("No batches available to probe embeddings.")
                except Exception:
                    # if probing fails, still try to collect (collector will error if truly impossible)
                    pass

                emb, meta = _collect_embedding_samples_for_artifacts(
                    model=m,
                    dataloader_factory=dl,
                    seeds=seeds,
                    episodes=episodes,
                    cfg=cfg,
                    device=device,
                    T_default=T_default,
                )
                estats = _embedding_stats(emb)
                estats["ci_mode"] = meta.get("ci_mode", None)
                estats["n_samples"] = meta.get("n_samples", None)

                # Prefer acpl.eval.embeddings if it exposes a writer
                wrote = None
                if emb_mod is not None:
                    for fn_name in ("write_embeddings", "save_embeddings", "export_embeddings", "run", "main"):
                        fn = getattr(emb_mod, fn_name, None)
                        if callable(fn):
                            try:
                                wrote = _call_with_accepted_kwargs(
                                    fn,
                                    outdir=outdir / "artifacts" / "embeddings" / _sanitize_filename(cond),
                                    cond=cond,
                                    embeddings=emb,
                                    meta=meta,
                                    cfg=cfg,
                                )
                                break
                            except Exception:
                                continue

                if wrote is None:
                    wrote = _write_embeddings_artifacts_fallback(
                        outdir=outdir, cond=cond, emb=emb, meta=meta, note="fallback_writer"
                    )

                block["embeddings"] = estats
                block["embeddings_artifact"] = wrote
            except Exception as e:
                block["embeddings_error"] = f"{type(e).__name__}: {e}"

        payload["conditions"][cond] = block

    if want_stats:
        # Let stats module optionally post-process the payload (best-effort)
        if stats_mod is not None:
            for fn_name in ("postprocess_stats", "finalize_stats", "augment_stats"):
                fn = getattr(stats_mod, fn_name, None)
                if callable(fn):
                    try:
                        payload = _call_with_accepted_kwargs(fn, payload=payload, outdir=outdir) or payload
                    except Exception:
                        pass

        _write_stats_artifacts(outdir=outdir, stats_payload=payload)



# ------------------------------ Ablation wiring --------------------------------

@dataclass(frozen=True)
class AblationBundle:
    """
    A single evaluation condition variant produced by an ablation.

    We support ablations that modify:
      - model (most common)
      - dataloader_factory (needed for true PermuteNodes and some NoPE variants)
      - rollout_fn (needed for GlobalCoin/TimeFrozen variants if implemented at rollout level)
    """
    name: str                 # canonical name, e.g., "NoPE"
    tag: str                  # directory tag, e.g., "ckpt_policy__abl_NoPE"
    model: torch.nn.Module
    dataloader_factory: Any
    rollout_fn: Any
    meta: dict[str, Any]


_ABLATION_ALIASES = {
    # Plan name -> accepted aliases
    "NoPE": {"nope", "no_pe", "no-pe", "noposenc", "no_posenc"},
    "GlobalCoin": {"globalcoin", "global_coin", "global-coin"},
    "TimeFrozen": {"timefrozen", "time_frozen", "time-frozen", "staticcoin", "static_coin"},
    # Prefer NodePermute as canonical, but accept older "PermuteNodes" spellings too
    "NodePermute": {
        "nodepermute", "node_permute", "node-permute",
        "permutenodes", "permute_nodes", "permute-nodes",
        "permutenodes",  # tolerate common typo
    },
}





def _zero_positional_like_channels(X: torch.Tensor, *, keep_last_indicator: bool) -> torch.Tensor:
    """
    Heuristic NoPE implementation that works with your current feature layout.

    Assumptions (matches scripts/train.py + scripts/eval.py payloads):
      - X[..., 0] is degree (keep)
      - X[..., 1] is coordinate / positional proxy (zero)
      - if indicator is used (search/robust), it is appended as the LAST channel (keep)
      - any "middle" channels are treated as positional-like and zeroed

    Works for X shape (N,F) or (B,N,F).
    """
    if not isinstance(X, torch.Tensor):
        return X
    if X.ndim not in (2, 3):
        return X
    F = int(X.shape[-1])
    if F < 2:
        return X

    Y = X.clone()
    if keep_last_indicator and F >= 3:
        # keep col0 (degree) and last col (indicator); zero middle cols
        Y[..., 1:-1] = 0
    else:
        # keep col0 (degree); zero everything else
        Y[..., 1:] = 0
    return Y

def _fallback_nope_bundle(
    *,
    base_tag: str,
    model: torch.nn.Module,
    dataloader_factory: Any,
    rollout_fn: Any,
    cfg: dict[str, Any] | None,
) -> AblationBundle:
    """
    If acpl.eval.ablations.NoPE cannot infer pe_dim, we still run a meaningful NoPE:
    remove positional signal by zeroing positional-like channels in X.
    """
    task = (cfg or {}).get("task", {}) if isinstance(cfg, dict) else {}
    tname = str((task or {}).get("name", "")).lower()
    keep_indicator = bool((task or {}).get("use_indicator", ("search" in tname) or ("robust" in tname)))

    def wrapped_factory(seed_i: int):
        it = dataloader_factory(seed_i)
        for batch in it:
            if not isinstance(batch, Mapping):
                yield batch
                continue
            b = dict(batch)
            X = b.get("X", None)
            if isinstance(X, torch.Tensor):
                b["X"] = _zero_positional_like_channels(X, keep_last_indicator=keep_indicator)
            yield b

    return AblationBundle(
        name="NoPE",
        tag=f"{base_tag}__abl_NoPE",
        model=model,
        dataloader_factory=wrapped_factory,
        rollout_fn=rollout_fn,
        meta={
            "ablation": "NoPE",
            "impl": "eval.py:fallback_zero_positional_like_channels",
            "keep_last_indicator": keep_indicator,
        },
    )




def _normalize_ablation_name(x: str) -> str:
    s = (x or "").strip()
    if not s:
        return s
    low = s.lower().replace(" ", "").replace("/", "_")
    for canon, aliases in _ABLATION_ALIASES.items():
        if low == canon.lower() or low in aliases:
            return canon
    # Preserve user input (but normalize capitalization a bit)
    # If they pass "NoPE" already, it stays.
    return s

def _call_with_accepted_kwargs(fn, /, **kwargs):
    """
    Call `fn(**kwargs)` but only pass kwargs that the function accepts.
    Supports functions with **kwargs.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(**filtered)









def _call_positional_with_accepted_kwargs(fn, /, *args, **kwargs):
    """
    Call `fn(*args, **kwargs)` but only pass kwargs that the function accepts.
    Supports functions with **kwargs.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)

def _call_plot_best_effort(fn, payload: dict[str, Any], figdir: Path):
    """
    Plot-call adapter:
      - supports plot fns that take (payload, save_dir=...), (payload, outdir=...), (payload, figdir=...)
      - supports plot fns that take (payload, figdir) positionally
      - avoids passing unexpected kwargs (fixes 'unexpected keyword argument save_dir')
    """
    attempts = [
        ((payload,), {"save_dir": figdir}),
        ((payload,), {"outdir": figdir}),
        ((payload,), {"figdir": figdir}),
        ((payload,), {"savepath": figdir}),
        ((payload,), {"save_path": figdir}),
        ((payload, figdir), {}),
        ((payload,), {}),
    ]
    last_err = None
    for args, kwargs in attempts:
        try:
            return _call_positional_with_accepted_kwargs(fn, *args, **kwargs)
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err


def _module_has_custom_forward(m: torch.nn.Module) -> bool:
    """
    True iff m.forward is not the base nn.Module.forward (which raises:
    'missing the required forward function').
    """
    f = getattr(m, "forward", None)
    if f is None:
        return False
    base = torch.nn.Module.forward
    func = getattr(f, "__func__", f)  # bound-method -> underlying function
    return func is not base


def _call_theta_method_best_effort(fn, theta_t: torch.Tensor, *, d: int | None = None) -> torch.Tensor:
    """
    Call a candidate theta->Hermitian method robustly.

    Some implementations use signatures like:
        hermitian_from_theta(theta, d)
    so we try supplying a default degree `d` when available.
    """
    dd: int | None = None
    if d is not None:
        try:
            dd = int(d)
        except Exception:
            dd = None

    def _unwrap(out: Any) -> torch.Tensor | None:
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            return out[0]
        return None

    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    # Positional: fn(theta) and fn(theta, d)
    attempts.append(((theta_t,), {}))
    if dd is not None:
        attempts.append(((theta_t, dd), {}))

    # Keyword: fn(theta=..., d=...)
    for th_kw in ("theta", "th", "x", "params"):
        attempts.append(((), {th_kw: theta_t}))
        if dd is not None:
            for d_kw in ("d", "deg", "degree", "outdeg"):
                attempts.append(((), {th_kw: theta_t, d_kw: dd}))

    last_err: Exception | None = None
    for args, kwargs in attempts:
        try:
            out = fn(*args, **kwargs)
            got = _unwrap(out)
            if got is not None:
                return got
        except TypeError as e:
            last_err = e
            continue

    raise TypeError(
        "Could not call adaptor theta->Hermitian method with supported (theta[, d]) patterns."
        + (f" Last TypeError: {last_err}" if last_err is not None else "")
    )


def _looks_like_hermitian_blocks(H: Any, theta_t: torch.Tensor) -> bool:
    if not isinstance(H, torch.Tensor):
        return False
    if H.ndim != 3:
        return False
    if H.shape[0] != theta_t.shape[0]:
        return False
    if H.shape[1] != H.shape[2]:
        return False
    if int(H.shape[1]) <= 0:
        return False
    return True

def _discover_theta2H_method_name(adaptor: Any) -> list[str]:
    """
    Heuristically rank candidate method names on adaptor that may implement theta->Hermitian.
    """
    names = []
    try:
        names = list(dir(adaptor))
    except Exception:
        return []

    scored: list[tuple[int, str]] = []
    for n in names:
        if n in ("forward", "__call__"):
            continue
        low = n.lower()
        if not callable(getattr(adaptor, n, None)):
            continue
        score = 0
        # Prefer explicit intent words
        if "theta" in low:
            score += 5
        if "herm" in low or "hermit" in low:
            score += 8
        if low in ("theta_to_hermitian", "theta_to_h", "to_hermitian", "to_h"):
            score += 50
        # Mild preference for non-private names, but allow private too
        if not low.startswith("_"):
            score += 2
        scored.append((score, n))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored]



def _ensure_forward_for_adaptor(m: Any, *, default_d: int | None = None) -> str | None:
    """
    Some older ThetaToHermitianAdaptor implementations forget to define forward().
    Patch the INSTANCE so that m(theta) works, by wiring forward -> best available method.
    Returns the method name used, or None if no patch applied.
    """
    if not isinstance(m, torch.nn.Module):
        return None
    if _module_has_custom_forward(m):
        return None

    # Stash a default degree (used by wrappers for methods that require `d`)
    try:
        setattr(m, "_theta2H_default_d", int(default_d) if default_d is not None else getattr(m, "_theta2H_default_d", None))
    except Exception:
        pass

    # Prefer known names first, then discover by heuristics.   
    
    preferred = [
        "theta_to_hermitian",
        "theta_to_H",
        "theta_to_h",
        "to_hermitian",
        "to_H",
        "to_h",
        "build_H",
        "make_H",
        "lift",
        "adapt",
        "compute_H",
        "compute_hermitian",
        "hermitian_from_theta",
    ]
    discovered = _discover_theta2H_method_name(m)
    tried = []
    for name in preferred + discovered:
        if name in tried:
            continue
        tried.append(name)
        fn = getattr(m, name, None)
        if not callable(fn):
            continue
        
        
        
        
        # Wire forward to a wrapper that only takes theta, and supplies `d` if needed.
        def _wrapped_forward(self, theta_t: torch.Tensor, _fn=fn):
            dd = getattr(self, "_theta2H_default_d", None)
            return _call_theta_method_best_effort(_fn, theta_t, d=dd)

        setattr(m, "_theta2H_cached_name", name)
        setattr(m, "forward", MethodType(_wrapped_forward, m))
        return name




    # Last resort: create a dynamic forward that discovers a callable at first use.
    def _auto_forward(self, theta_t: torch.Tensor) -> torch.Tensor:
        cand_names = _discover_theta2H_method_name(self)
        for nm in cand_names:
            fn2 = getattr(self, nm, None)
            if not callable(fn2):
                continue
            try:
                dd = getattr(self, "_theta2H_default_d", None)
                H = _call_theta_method_best_effort(fn2, theta_t, d=dd)            
            
            except Exception:
                continue
            if _looks_like_hermitian_blocks(H, theta_t):
                setattr(self, "_theta2H_cached_name", nm)
                return H
        raise RuntimeError(
            f"Adaptor {type(self).__name__} has no usable theta->Hermitian method; "
            f"candidates tried: {cand_names[:25]}"
        )

    setattr(m, "forward", MethodType(_auto_forward, m))
    return "auto_forward"

def _theta_to_hermitian_best_effort(adaptor: Any, theta_t: torch.Tensor, *, default_d: int | None = None) -> torch.Tensor:
    """
    Convert per-node theta parameters to Hermitian blocks. Works even if the adaptor
    is missing forward() by calling a known method.
    """
    if adaptor is None:
        raise RuntimeError("Any-degree plotting requires adaptor (ThetaToHermitianAdaptor).")




    dd = default_d
    if dd is None:
        dd = getattr(adaptor, "_theta2H_default_d", None)
    try:
        dd = int(dd) if dd is not None else None
    except Exception:
        dd = None




    # If we've already cached a working method name, use it.
    cached = getattr(adaptor, "_theta2H_cached_name", None)
    if isinstance(cached, str) and cached:
        fn = getattr(adaptor, cached, None)
        if callable(fn):
            H = _call_theta_method_best_effort(fn, theta_t, d=dd)
            if _looks_like_hermitian_blocks(H, theta_t):
                return H

    # If forward exists and is implemented, use adaptor(theta).
    if isinstance(adaptor, torch.nn.Module) and _module_has_custom_forward(adaptor):
        for args in ( (theta_t,), (theta_t, dd) ) if dd is not None else ( (theta_t,), ):
            try:
                H = adaptor(*args)
            except TypeError:
                continue
            if _looks_like_hermitian_blocks(H, theta_t):
                return H


    # Otherwise: discover a callable theta->H method without relying on forward().
    cand_names = _discover_theta2H_method_name(adaptor)
    for nm in cand_names:
        fn = getattr(adaptor, nm, None)
        if not callable(fn):
            continue
        try:
            H = _call_theta_method_best_effort(fn, theta_t, d=dd)
        except Exception:
            continue
        if _looks_like_hermitian_blocks(H, theta_t):
            setattr(adaptor, "_theta2H_cached_name", nm)
            return H

    # Do NOT fall back to adaptor(theta) here (it will raise the original confusing error).
    raise RuntimeError(
        f"Could not compute Hermitian blocks from adaptor={type(adaptor).__name__}. "
        f"forward implemented? {bool(isinstance(adaptor, torch.nn.Module) and _module_has_custom_forward(adaptor))}. "
        f"Callable candidates (top): {cand_names[:25]}"
    )



def _sanitize_filename(s: str) -> str:
    s = (s or "").strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    x = "".join(out)
    while "__" in x:
        x = x.replace("__", "_")
    return x.strip("_") or "cond"


def _make_rollout_timeline_fn(
    th: Any,
    *,
    pm: Any,
    shift: Any,
    family: str,
    adaptor: Any | None,
    cdtype: Any,
    device: Any = None,
    init_mode: str,
    theta_scale: float = 1.0,
    theta_noise_std: float = 0.0,
):
    """
    Return a rollout(model, batch) -> Pt tensor with shape (T+1, N).
    This is ONLY used for plotting, so it is allowed to be slower.
    """
    from acpl.sim.utils import partial_trace_coin
    from acpl.sim.step import step as _step_su2


    if device is None:
        device = torch.device("cpu")



    fam = (family or "su2").lower().strip()

    def _init_state(batch: Mapping[str, Any]) -> torch.Tensor:

        # mirror scripts/train.py initialization behavior, but be robust to
        # helpers that may/may not accept `device=...`.
        if init_mode == "uniform":
            return _call_positional_with_accepted_kwargs(
                th.init_state_uniform_arcs, pm, cdtype=cdtype, device=device
            )
        if init_mode == "node":
            start = int(batch.get("start_node", 0))
            return _call_positional_with_accepted_kwargs(
                th.init_state_node_uniform_ports, pm, start, cdtype=cdtype, device=device
            )
        # default: node0
        return _call_positional_with_accepted_kwargs(
            th.init_state_node0_uniform_ports, pm, cdtype=cdtype, device=device
        )

    def _rollout_su2(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
        X = batch["X"]
        edge_index = batch["edge_index"]
        T = int(batch["T"])
        psi = _init_state(batch)

        # IMPORTANT FOR ABLATIONS:
        # Prefer forward() -> theta so wrappers (GlobalCoin/TimeFrozen) actually change plots.
        coins = None

        # 1) Try forward() first: either returns theta (T,N,P) or already-built coins (T,N,2,2)
        try:
            out = model(X, edge_index, T=T)
            if isinstance(out, torch.Tensor):
                if out.ndim == 3 and hasattr(model, "_su2_from_euler_batch"):
                    coins = model._su2_from_euler_batch(out)  # theta -> SU2 coins
                elif out.ndim == 4 and out.shape[-1] == out.shape[-2] == 2:
                    coins = out  # already SU2 coins
        except Exception:
            coins = None

        # 2) Fallback: use coins_su2 if that’s all we have
        if coins is None:
            if hasattr(model, "coins_su2"):
                coins = model.coins_su2(X, edge_index, T=T)
            else:
                raise RuntimeError(
                    "SU2 plotting needs either model.forward->theta + _su2_from_euler_batch "
                    "or model.coins_su2."
                )

        Pt = torch.empty((T + 1, pm.num_nodes), device=psi.device, dtype=torch.float32)
        Pt[0] = partial_trace_coin(psi, pm).to(torch.float32)
        for t in range(T):
            psi = _step_su2(psi, pm, coins[t], shift=shift)
            Pt[t + 1] = partial_trace_coin(psi, pm).to(torch.float32)
        return Pt

    def _rollout_anydeg(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
        from acpl.sim.utils import partial_trace_coin
        X = batch["X"]
        edge_index = batch["edge_index"]
        T = int(batch["T"])
        psi = _init_state(batch)

        
        
        theta = None
        try:
            out = model(X, edge_index, T=T)
            if isinstance(out, torch.Tensor):
                theta = out
            elif isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                theta = out[0]
        except Exception:
            theta = None

        if theta is None:
            if hasattr(model, "coins_anydeg"):
                theta = model.coins_anydeg(X, edge_index, T=T)
            else:
                raise RuntimeError("Any-degree plotting needs model.forward(...) -> theta OR model.coins_anydeg(...).")
                
        
        
        
        
        
        if theta_scale != 1.0:
            theta = theta * float(theta_scale)
        if model.training and float(theta_noise_std) > 0.0:
            theta = theta + float(theta_noise_std) * torch.randn_like(theta)

        Pt = torch.empty((T + 1, pm.num_nodes), device=psi.device, dtype=torch.float32)
        Pt[0] = partial_trace_coin(psi, pm).to(torch.float32)

        perm = shift.perm
        d0 = int(pm.degree) if getattr(pm, "is_regular", False) else None
        fast_regular = bool(getattr(pm, "is_regular", False)) and bool(getattr(shift, "is_perm", False)) and d0 is not None



        try:
            _deg = (pm.node_ptr[1:] - pm.node_ptr[:-1])
            d_default = d0 if d0 is not None else int(_deg.max().item())
        except Exception:
            d_default = d0 if d0 is not None else None
 






        for t in range(T):
            tht = theta[t]  # (N, out_dim)
            if adaptor is None:
                raise RuntimeError("Any-degree plotting requires adaptor (ThetaToHermitianAdaptor).")
            H = _theta_to_hermitian_best_effort(adaptor, tht, default_d=d_default)  # (N, d, d)
            if fam == "exp":
                U = th.unitary_exp_iH(H)
            else:
                U = th.unitary_cayley(H)

            if fast_regular:
                Nn = pm.num_nodes
                psi2 = psi.view(Nn, d0)
                psi2 = torch.bmm(U, psi2.unsqueeze(-1)).squeeze(-1)
                psi = psi2.reshape(-1)[perm]
            else:
                # variable degree: build per-node blocks
                coins_v: list[torch.Tensor | None] = []
                start = pm.node_ptr[:-1]
                end = pm.node_ptr[1:]
                for v in range(pm.num_nodes):
                    s, e = int(start[v]), int(end[v])
                    d = e - s
                    if d <= 0:
                        coins_v.append(None)
                    else:
                        coins_v.append(U[v, :d, :d])
                psi = th.apply_blockdiag_coins_anydeg(psi, pm, coins_v)[perm]

            Pt[t + 1] = partial_trace_coin(psi, pm).to(torch.float32)

        return Pt

    if fam == "su2":
        return _rollout_su2
    return _rollout_anydeg


def _collect_Pt_samples_for_plotting(
    *,
    model: torch.nn.Module,
    dataloader_factory: Any,
    rollout_timeline_fn: Any,
    seeds: list[int],
    episodes: int,

    ckpt_path: Path | None = None,
    artifacts_request: Mapping[str, Any] | None = None,

) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return Pt samples shaped (K, T+1, N) PLUS metadata describing what K means.

    Defendable CI convention:
      - If len(seeds) >= 2: K = #seeds and each sample is the per-seed mean over episodes.
        (CI across seeds)
      - If len(seeds) == 1: K = #episodes (up to `episodes`) for that single seed.
        (CI across episodes; avoids ddof/div0 warnings when seeds=1)

    This is ONLY used for plotting.
    """
    model.eval()

    seeds = list(seeds or [])
    if not seeds:
        raise RuntimeError("No seeds provided to plotting collector.")

    with torch.no_grad():
        # ---- CI across EPISODES when only one seed ----
        if len(seeds) == 1:
            s = int(seeds[0])
            it = dataloader_factory(s)
            samples: list[np.ndarray] = []
            n = 0
            for batch in it:
                Pt = rollout_timeline_fn(model, batch)  # (T+1, N) torch
                samples.append(Pt.detach().cpu().numpy())
                n += 1
                if n >= int(episodes):
                    break

            if not samples:
                raise RuntimeError("No Pt episode samples collected for plotting (single-seed mode).")

            arr = np.stack(samples, axis=0)  # (K=episodes, T+1, N)
            meta: dict[str, Any] = {
                "ci_mode": "episodes",
                "n_samples": int(arr.shape[0]),
                "seed": s,
                "env": _env_info(),
            }
            if ckpt_path is not None:
                try:
                    p = _as_path(ckpt_path)
                    meta["checkpoint_path"] = str(p)
                    meta["checkpoint_sha256"] = _sha256_file(p) if p.exists() else None
                except Exception:
                    meta["checkpoint_sha256"] = None
            if artifacts_request is not None:
                meta["artifacts_request"] = dict(artifacts_request)
            return arr, meta

        # ---- CI across SEEDS when multiple seeds ----
        per_seed_means: list[np.ndarray] = []
        for s in seeds:
            it = dataloader_factory(int(s))
            acc = None
            n = 0
            for batch in it:
                Pt = rollout_timeline_fn(model, batch)  # (T+1, N)
                acc = Pt if acc is None else (acc + Pt)
                n += 1
                if n >= int(episodes):
                    break
            if acc is None or n == 0:
                continue
            per_seed_means.append((acc / float(n)).detach().cpu().numpy())

        if not per_seed_means:
            raise RuntimeError("No Pt data collected for plotting (multi-seed mode).")

        arr = np.stack(per_seed_means, axis=0)  # (K=seeds, T+1, N)
        meta: dict[str, Any] = {
            "ci_mode": "seeds",
            "n_samples": int(arr.shape[0]),
            "seeds": [int(x) for x in seeds],
        }



        if ckpt_path is not None:
            try:
                p = _as_path(ckpt_path)
                meta["checkpoint_path"] = str(p)
                meta["checkpoint_sha256"] = _sha256_file(p) if p.exists() else None
            except Exception:
                meta["checkpoint_sha256"] = None
        if artifacts_request is not None:
            meta["artifacts_request"] = dict(artifacts_request)



        return arr, meta




def _build_ablation_bundle(
    abl_mod: Any,
    *,
    ablation: str,
    base_tag: str,
    model: torch.nn.Module,
    dataloader_factory: Any,
    rollout_fn: Any,
    cfg: dict[str, Any] | None,
    device: Any,
) -> AblationBundle | None:
    """
    Best-effort adapter over multiple possible ablation APIs.

    Supported ablations module APIs (any one is enough):

    1) apply_ablation_bundle(...)
        - returns dict-like with optional keys:
            model, dataloader_factory, rollout_fn, tag, meta
    2) apply_ablation(model, name=...) -> model
        - model-only ablations
    3) get_ablation(name) -> callable
        - callable may accept (model) or (model, cfg=..., dataloader_factory=..., rollout_fn=...)

    If an ablation cannot be applied, returns None (caller can warn/skip).
    """
    name = _normalize_ablation_name(ablation)
    cond_tag = f"{base_tag}__abl_{name}"

    # Default bundle = identity (no changes)
    out_model = model
    out_dl = dataloader_factory
    out_rollout = rollout_fn
    meta: dict[str, Any] = {"ablation": name}

    # (1) apply_ablation_bundle
    if hasattr(abl_mod, "apply_ablation_bundle"):
        fn = getattr(abl_mod, "apply_ablation_bundle")
        payload = _call_with_accepted_kwargs(
            fn,
            name=name,
            model=model,
            dataloader_factory=dataloader_factory,
            rollout_fn=rollout_fn,
            cfg=cfg,
            device=device,
        )
        if isinstance(payload, dict):
            out_model = payload.get("model", out_model)
            out_dl = payload.get("dataloader_factory", out_dl)
            out_rollout = payload.get("rollout_fn", out_rollout)
            
            
            
            
            payload_tag = payload.get("tag", None)
            if isinstance(payload_tag, str) and payload_tag:
                # If ablations module returns "abl_X", prefix with base_tag to avoid collisions
                if payload_tag.startswith(base_tag):
                    cond_tag = payload_tag
                elif payload_tag.startswith("abl_") or payload_tag.startswith("abl-"):
                    cond_tag = f"{base_tag}__{payload_tag}"
                else:
                    cond_tag = payload_tag

                        
            
            m2 = payload.get("meta", None)
            if isinstance(m2, dict):
                meta.update(m2)
        return AblationBundle(
            name=name, tag=str(cond_tag),
            model=out_model, dataloader_factory=out_dl, rollout_fn=out_rollout,
            meta=meta,
        )

    # (2) apply_ablation (model-only OR richer signature)
    if hasattr(abl_mod, "apply_ablation"):
        fn = getattr(abl_mod, "apply_ablation")
        try:
            maybe = _call_with_accepted_kwargs(
                fn,
                model=model,
                name=name,
                cfg=cfg,
                dataloader_factory=dataloader_factory,
                rollout_fn=rollout_fn,
                device=device,
            )
        except TypeError:
            # common alternate: apply_ablation(model, ablation)
            maybe = fn(model, name)
        if isinstance(maybe, torch.nn.Module):
            out_model = maybe
            return AblationBundle(
                name=name, tag=cond_tag,
                model=out_model, dataloader_factory=out_dl, rollout_fn=out_rollout,
                meta=meta,
            )
        if isinstance(maybe, dict):
            out_model = maybe.get("model", out_model)
            out_dl = maybe.get("dataloader_factory", out_dl)
            out_rollout = maybe.get("rollout_fn", out_rollout)
            m2 = maybe.get("meta", None)
            if isinstance(m2, dict):
                meta.update(m2)
            cond_tag = maybe.get("tag", cond_tag)
            return AblationBundle(
                name=name, tag=str(cond_tag),
                model=out_model, dataloader_factory=out_dl, rollout_fn=out_rollout,
                meta=meta,
            )

    # (3) get_ablation(name) -> callable
    if hasattr(abl_mod, "get_ablation"):
        get_fn = getattr(abl_mod, "get_ablation")
        ab_fn = get_fn(name)
        if callable(ab_fn):
            maybe = _call_with_accepted_kwargs(
                ab_fn,
                model=model,
                cfg=cfg,
                dataloader_factory=dataloader_factory,
                rollout_fn=rollout_fn,
                device=device,
            )
            if isinstance(maybe, torch.nn.Module):
                out_model = maybe
                return AblationBundle(
                    name=name, tag=cond_tag,
                    model=out_model, dataloader_factory=out_dl, rollout_fn=out_rollout,
                    meta=meta,
                )
            if isinstance(maybe, dict):
                out_model = maybe.get("model", out_model)
                out_dl = maybe.get("dataloader_factory", out_dl)
                out_rollout = maybe.get("rollout_fn", out_rollout)
                m2 = maybe.get("meta", None)
                if isinstance(m2, dict):
                    meta.update(m2)
                cond_tag = maybe.get("tag", cond_tag)
                return AblationBundle(
                    name=name, tag=str(cond_tag),
                    model=out_model, dataloader_factory=out_dl, rollout_fn=out_rollout,
                    meta=meta,
                )

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
    ckpt_path: Path | None,
    outdir: Path,
    suite: str,
    device: str = "cuda",
    seeds: int | Iterable[int] = 5,
    episodes: int = 128,
    ablations: list[str] | None = None,
    plots: bool = False,
    extra_overrides: dict[str, Any] | None = None,
    *,
    policy: str = "ckpt",
    baseline_kind: str = "hadamard",
    baseline_coins_kwargs: dict[str, Any] | None = None,
    baseline_policy_kwargs: dict[str, Any] | None = None,
    report: bool = False,
    strict_ablations: bool = False,
    # Robustness sweep runner (Phase 6)
    robust_sweep: bool = False,
    robust_sweep_kinds: list[str] | None = None,
    robust_sweep_sigma: str | None = None,
    robust_sweep_trials: int | None = None,
    robust_sweep_bootstrap: int = 200,
    robust_sweep_include_ablations: bool = False,
    robust_sweep_max_plot_metrics: int = 6,


    # First-class artifacts (B7)
    artifacts: bool = True,
    artifacts_embeddings: bool = True,
    artifacts_stats: bool = True,
    artifacts_embed_episodes: int = 32,
    artifacts_embed_seed: int | None = None,
    artifacts_embeddings_max_nodes: int = 4096,


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
        from acpl.eval.protocol import EvalConfig, run_ci_eval, summarize_results, compute_ci

        from acpl.train.loops import LoopConfig
        from acpl.utils.logging import MetricLogger, MetricLoggerConfig
    except Exception:
        EvalConfig = None
        run_ci_eval = None
        summarize_results = None
        LoopConfig = None
        MetricLogger = None
        MetricLoggerConfig = None
        compute_ci = None





    # Load checkpoint (always recommended, even for baselines, to reuse config)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt_path is None:
        raise ValueError("ckpt_path is None (unexpected).")

    ckpt = _load_any_ckpt(ckpt_path)

    trainer_native_ok = (
        (run_ci_eval is not None)
        and isinstance(ckpt.get("config", None), dict)
    )


    abl_mod = _maybe_get_ablations()
    plot_mod = _maybe_get_plots()
    report_mod = _maybe_get_reporting()
    stats_mod = _maybe_get_stats()
    emb_mod = _maybe_get_embeddings()

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

        # ---- Disorder plumbing (base eval must reflect cfg['disorder']) ----
        _dis_mod = _maybe_get_disorder()
        disorder_cfg = _normalize_disorder_cfg(cfg.get("disorder", {}) if isinstance(cfg, dict) else {})
        shift = _build_shift_with_optional_disorder(
            th,
            pm,
            disorder_cfg=disorder_cfg,
            disorder_seed=int(seed_base),
            dis_mod=_dis_mod,
        )

        if bool(disorder_cfg.get("enabled", False)) and _dis_mod is None:
            print(
                "[warn] cfg.disorder.enabled=True but acpl.sim.disorder could not be imported; "
                "shift disorder may be inactive until disorder plumbing lands.",
                file=sys.stderr,
            )



        # Default degree for theta->Hermitian lifts (needed by some adaptor APIs)
        try:
            _deg = (pm.node_ptr[1:] - pm.node_ptr[:-1])
            default_d = int(_deg.max().item()) if hasattr(_deg, "max") else int(getattr(pm, "degree", 0) or 0)
        except Exception:
            default_d = int(getattr(pm, "degree", 0) or 0)










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

        # --------------------------- build policy (ckpt vs baseline) ---------------------------
        policy = (policy or "ckpt").lower().strip()
        baseline_coins_kwargs = dict(baseline_coins_kwargs or {})
        baseline_policy_kwargs = dict(baseline_policy_kwargs or {})

        if policy == "baseline":
            try:
                from acpl.baselines.policies import build_baseline_policy as _build_baseline_policy
            except Exception as e:
                raise ImportError(
                    "Could not import acpl.baselines.policies.build_baseline_policy. "
                    "Implement baselines first (acpl/baselines/coins.py + policies.py)."
                ) from e

            model = _build_baseline_policy(
                baseline_kind,
                name=f"baseline:{baseline_kind}",
                coins_kwargs=baseline_coins_kwargs,
                policy_kwargs=baseline_policy_kwargs,
            ).to(device=torch_device).eval()

        else:
            # default: use trained model from ckpt
            model = th.ACPLPolicy(acpl_cfg).to(device=torch_device).eval()

            # ---- load weights (CRITICAL) ----
            sd = ckpt.get("state_dict", None)
            if sd is None and isinstance(ckpt.get("model", None), Mapping):
                # some saves store the raw state_dict under "model"
                sd = ckpt["model"]

            if isinstance(sd, Mapping):
                try:
                    model.load_state_dict(_normalize_state_dict_keys(sd), strict=False)
                except Exception:
                    # last resort: try raw
                    model.load_state_dict(sd, strict=False)

            # optional EMA shadow (if present)
            ema_shadow = ckpt.get("ema_shadow", None)
            if isinstance(ema_shadow, Mapping):
                _try_apply_ema_shadow(model, ema_shadow)

                
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        # optional adaptor for exp/cayley
        adaptor = None
        if coin_family in ("exp", "cayley"):
            adaptor = th.ThetaToHermitianAdaptor(
                Kfreq=int(coin_cfg.get("Kfreq", 2))
            ).to(device=torch_device).eval()

            ad_sd = None
            if policy != "baseline":
                ad_sd = ckpt.get("adaptor", None)

            if isinstance(ad_sd, Mapping):
                try:
                    adaptor.load_state_dict(ad_sd, strict=False)
                except Exception:
                    pass


            # ---- IMPORTANT: patch missing forward() so adaptor(theta) works for plotting ----
            patched = _ensure_forward_for_adaptor(adaptor, default_d=default_d)
            if patched is not None:
                print(
                    
                    
                    f"[warn] ThetaToHermitianAdaptor missing forward(); patched forward -> {patched}(theta[, d]) "
                    f"with default_d={default_d} (plotting-safe).",
                    
                    
                    file=sys.stderr,
                )




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
                if hasattr(ds, "set_epoch"):
                    ds.set_epoch(int(seed_i))
                return (ds[i] for i in range(len(ds)))

        else:
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
                if hasattr(ds, "set_epoch"):
                    ds.set_epoch(int(seed_i))
                return (ds[i] for i in range(len(ds)))
            
        
        
        
        # --- rollout_fn (same as train.py) ---
        if coin_family == "su2":
            
            
            rollout_fn = _call_positional_with_accepted_kwargs(
                th.make_rollout_su2_dv2,
                pm,
                shift,
                cdtype=cdtype,
                device=torch_device,
                init_mode=init_mode,
                disorder=disorder_cfg,
                disorder_seed=int(seed_base),
                seed=int(seed_base),
            )
           
            
            
            title_suffix = "SU2 (deg=2)"
        else:
            theta_scale = float(coin_cfg.get("theta_scale", 1.0))
            theta_noise_std = float(coin_cfg.get("theta_noise_std", 0.0))
            
            
            
            rollout_fn = _call_positional_with_accepted_kwargs(
                th.make_rollout_anydeg_exp_or_cayley,
                pm,
                shift,
                adaptor=adaptor,
                family=coin_family,
                cdtype=cdtype,
                device=torch_device,
                init_mode=init_mode,
                theta_scale=theta_scale,
                theta_noise_std=theta_noise_std,
                disorder=disorder_cfg,
                disorder_seed=int(seed_base),
                seed=int(seed_base),
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
            keep_per_seed_means=(len(seed_list) >= 2),

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

        "policy": policy,
        "baseline_kind": baseline_kind if policy == "baseline" else None,
        "baseline_coins_kwargs": baseline_coins_kwargs if policy == "baseline" else None,
        "baseline_policy_kwargs": {k: str(v) for k, v in (baseline_policy_kwargs or {}).items()} if policy == "baseline" else None,


        "ablations_canonical": [_normalize_ablation_name(a) for a in (ablations or [])],



        "device": device,
        "suite": suite,
        "episodes": episodes,
        "seeds": seed_list,
        "ablations": ablations or [],
        "model_config": model_cfg,

        "git": _git_info(Path(__file__).resolve().parents[1]),
        "ci_convention": {
            "pooled_episode_ci": {
                "meaning": "CI over all episodes pooled across seeds (bootstrap), written to eval_ci.json",
            },
            "seed_level_ci": {
                "meaning": "CI over per-seed episode means (replicates=seeds), written to eval_ci_seed.json when seeds>=2",
            },
            "note": "For thesis claims, prefer seeds>=3 and cite seed-level CI.",
        },
        "disorder": disorder_cfg if trainer_native_ok else None,
        "disorder_seed_base": int(seed_base) if trainer_native_ok else None,
        "robust_sweep_request": {
            "enabled": bool(robust_sweep),
            "kinds_override": list(robust_sweep_kinds or []),
            "sigma_override": robust_sweep_sigma,
            "trials_override": robust_sweep_trials,
            "bootstrap_samples_per_trial": int(robust_sweep_bootstrap),
            "include_ablations": bool(robust_sweep_include_ablations),
            "max_plot_metrics": int(robust_sweep_max_plot_metrics),
        },





    }
    (outdir / "meta.json").write_text(_json_dump(meta))

    # Helper to run a single condition (possibly with an ablation applied)
    def run_one_condition(
        condition_tag: str,
        *,
        model_override: torch.nn.Module | None = None,
        dataloader_factory_override: Any | None = None,
        rollout_fn_override: Any | None = None,
        extra_meta: dict[str, Any] | None = None,
    ):
        tag_dir = _ensure_dir(outdir / "raw" / condition_tag)

        # Choose overrides (or base)
        m_for_eval = model_override if model_override is not None else model

        # These only exist in trainer-native mode; keep safe defaults
        dl_for_eval = dataloader_factory_override if dataloader_factory_override is not None else (
            _make_eval_iter if trainer_native_ok else None
        )
        ro_for_eval = rollout_fn_override if rollout_fn_override is not None else (
            rollout_fn if trainer_native_ok else None
        )

        # Persist condition meta (helps thesis reproducibility)
        cond_meta = {
            "cond": condition_tag,
            "policy": policy,
            "baseline_kind": baseline_kind if policy == "baseline" else None,
            "ablation_meta": extra_meta or {},
        }
        (tag_dir / "condition_meta.json").write_text(_json_dump(cond_meta), encoding="utf-8")

        # Trainer-native protocol path
        if trainer_native_ok and callable(proto_entry):
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

            results = proto_entry(
                model=m_for_eval,
                dataloader_factory=dl_for_eval,
                rollout_fn=ro_for_eval,
                loop_cfg=loop_cfg,
                eval_cfg=eval_cfg,
                logger=logger,
                step=0,
            )

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

            json_payload = {
                k: _ci_to_dict(v["all"])
                for k, v in results.items()
                if isinstance(v, dict) and "all" in v
            }
            (tag_dir / "eval_ci.json").write_text(_json_dump(json_payload), encoding="utf-8")

            # --- NEW: seed-level CI (replicates = seeds), saved separately ---
            seed_ci_payload: dict[str, Any] = {}
            seed_ci_text = ""
            if compute_ci is not None and bool(eval_cfg.keep_per_seed_means) and len(seed_list) >= 2:
                for metric, vv in results.items():
                    if not (isinstance(vv, dict) and "all" in vv):
                        continue
                    seed_means: list[float] = []
                    for s in seed_list:
                        kseed = f"seed={int(s)}"
                        if kseed in vv and hasattr(vv[kseed], "mean"):
                            try:
                                seed_means.append(float(vv[kseed].mean))
                            except Exception:
                                pass
                    if len(seed_means) >= 2:
                        ci_seed = compute_ci(np.asarray(seed_means, dtype=float), method="student_t", alpha=eval_cfg.ci_alpha)
                        seed_ci_payload[str(metric)] = _ci_to_dict(ci_seed)

                if seed_ci_payload:
                    (tag_dir / "eval_ci_seed.json").write_text(_json_dump(seed_ci_payload), encoding="utf-8")
                    seed_ci_text = (
                        "Seed-level CI (replicates = seeds; values = per-seed episode means)\n"
                        f"method=student_t alpha={eval_cfg.ci_alpha} n_seeds={len(seed_list)}\n"
                    )
                    (tag_dir / "eval_ci_seed.txt").write_text(seed_ci_text, encoding="utf-8")

            if logger is not None:
                try:
                    logger.close()
                except Exception:
                    pass

            return {
                "summary": {k: d["mean"] for k, d in json_payload.items()},     # pooled-episode mean
                "ci": json_payload,                                            # pooled-episode CI (bootstrap)
                "summary_seed": {k: v["mean"] for k, v in seed_ci_payload.items()} if seed_ci_payload else {},
                "ci_seed": seed_ci_payload,                                    # seed-level CI (student_t)
                "text": summary_text,
                "text_seed": seed_ci_text,
            }

        # Fallback (non-trainer-native): keep your existing logic, but model-only
        results = None
        kwargs = dict(common_kwargs)
        kwargs.setdefault("outdir", tag_dir)

        if callable(proto_entry):
            try:
                results = proto_entry(model=m_for_eval, **kwargs)
            except TypeError:
                try:
                    results = proto_entry(m_for_eval, **kwargs)
                except TypeError:
                    kwargs2 = dict(kwargs)
                    kwargs2.pop("suite", None)
                    results = proto_entry(m_for_eval, **kwargs2)
        else:
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

        if isinstance(results, dict) and "logs" in results and results["logs"] is not None:
            jsonl = tag_dir / "logs.jsonl"
            with jsonl.open("w") as f:
                for row in results["logs"]:
                    f.write(_json_dump(row) + "\n")
        return results


    # Evaluate main condition (ckpt policy OR selected baseline) — no ablation
    summaries_for_csv: list[Mapping[str, Any]] = []
    structured_out: dict[str, Any] = {"conditions": {}}

    plot_runners: dict[str, dict[str, Any]] = {}

    base_tag = "ckpt_policy" if policy != "baseline" else f"baseline_{baseline_kind}"
    base_res = run_one_condition(base_tag)


    if trainer_native_ok:
        plot_runners[base_tag] = {
            "model": model,
            "dataloader_factory": _make_eval_iter,
            "meta": {},  # base condition has no ablation cfg
        }


    base_summary = (base_res or {}).get("summary", {})
    structured_out["conditions"][base_tag] = base_res or {}
    row = {"cond": base_tag}
    row.update({f"metric.{k}": v for k, v in base_summary.items()})
    summaries_for_csv.append(row)


    # Ablations (causality checks)
    if ablations:
        for abl in ablations:
            canon = _normalize_ablation_name(abl)

            bundle = None
            if abl_mod is not None:
                try:
                    bundle = _build_ablation_bundle(
                        abl_mod,
                        ablation=canon,
                        base_tag=base_tag,
                        model=model,
                        dataloader_factory=_make_eval_iter if trainer_native_ok else None,
                        rollout_fn=rollout_fn if trainer_native_ok else None,
                        cfg=cfg if trainer_native_ok else None,
                        device=torch_device if trainer_native_ok else device,
                    )
                except Exception as e:
                    
                    # --- FIX: NoPE fallback if ablations module can't infer pe_dim ---
                    if trainer_native_ok and canon == "NoPE":
                        try:
                            bundle = _fallback_nope_bundle(
                                base_tag=base_tag,
                                model=model,
                                dataloader_factory=_make_eval_iter,
                                rollout_fn=rollout_fn,
                                cfg=cfg,
                            )
                            print(
                                "[warn] ablation 'NoPE' failed to build in acpl.eval.ablations; "
                                "using eval.py fallback (zero positional-like channels).",
                                file=sys.stderr,
                            )
                        except Exception as e2:
                            print(f"[warn] ablation '{abl}' failed to build: {e2}", file=sys.stderr)
                            bundle = None
                    else:
                        print(f"[warn] ablation '{abl}' failed to build: {e}", file=sys.stderr)
                        bundle = None
            
            
            
            
            
            
            
            
            
            
            
            
            
            else:
                # Optional: still allow NoPE in trainer-native mode even if ablations module is missing
                if trainer_native_ok and canon == "NoPE":
                    bundle = _fallback_nope_bundle(
                        base_tag=base_tag,
                        model=model,
                        dataloader_factory=_make_eval_iter,
                        rollout_fn=rollout_fn,
                        cfg=cfg,
                    )
                    print(
                        "[warn] acpl.eval.ablations missing; using eval.py NoPE fallback (zero positional-like channels).",
                        file=sys.stderr,
                    )
                else:
                    print("[warn] acpl.eval.ablations not found; skipping ablations.", file=sys.stderr)       
            
            
            
            
            if bundle is None:
                if strict_ablations:
                    raise RuntimeError(f"Ablation '{canon}' failed to build (strict_ablations=True).")
                cond_tag = f"{base_tag}__abl_{canon}__SKIPPED"
                res = {"summary": {}, "skipped": True, "reason": "ablation_build_failed_or_missing"}
                    
            
            
            
            else:
                res = run_one_condition(
                    bundle.tag,
                    model_override=bundle.model,
                    dataloader_factory_override=bundle.dataloader_factory,
                    rollout_fn_override=bundle.rollout_fn,
                    extra_meta=bundle.meta,
                )
                cond_tag = bundle.tag

                if trainer_native_ok:
                    plot_runners[cond_tag] = {
                        "model": bundle.model,
                        "dataloader_factory": bundle.dataloader_factory,
                        "meta": dict(bundle.meta or {}),  # <-- keep ablation_cfg for plotting
                    }

            structured_out["conditions"][cond_tag] = res or {}

            summ = (res or {}).get("summary", {})
            row = {"cond": cond_tag}
            row.update({f"metric.{k}": v for k, v in summ.items()})
            summaries_for_csv.append(row)



    # ---------------- Robustness sweeps (Phase 6) ----------------
    if trainer_native_ok and bool(robust_sweep):
        try:
            # choose which conditions to sweep
            if robust_sweep_include_ablations:
                sweep_conds = dict(plot_runners)
            else:
                sweep_conds = {base_tag: plot_runners[base_tag]} if base_tag in plot_runners else {}

            if not sweep_conds:
                print("[warn] robust_sweep requested but no sweepable conditions found; skipping.", file=sys.stderr)
            else:
                sweep_out = _run_robustness_sweeps(
                    outdir=outdir,
                    cond_runners=sweep_conds,
                    cfg=cfg if trainer_native_ok else {},
                    th=th,
                    pm=pm,
                    coin_family=coin_family,
                    coin_cfg=(cfg.get("coin", {}) or {}) if isinstance(cfg, dict) else {},
                    adaptor=adaptor,
                    cdtype=cdtype,
                    device=torch_device,
                    init_mode=init_mode,
                    seeds=seed_list,
                    episodes=int(episodes),
                    loop_cfg=loop_cfg,
                    EvalConfig=EvalConfig,
                    proto_entry=proto_entry,
                    compute_ci=compute_ci,
                    plot_mod=plot_mod,
                    manifest_hex=str(manifest_hex),
                    seed_base=int(seed_base),
                    kinds_override=robust_sweep_kinds,
                    sigmas_override=robust_sweep_sigma,
                    trials_override=robust_sweep_trials,
                    bootstrap_samples=int(robust_sweep_bootstrap),
                    ci_alpha=float(eval_cfg.ci_alpha),
                    max_plot_metrics=int(robust_sweep_max_plot_metrics),
                )
                structured_out["robust_sweeps"] = sweep_out
        except Exception as e:
            print(f"[warn] robustness sweeps failed: {e}", file=sys.stderr)



    # Write summaries
    _write_csv(summaries_for_csv, outdir / "summary.csv")
    (outdir / "summary.json").write_text(_json_dump(structured_out))






    # ---------------- First-class artifacts (B7): stats + embeddings ----------------
    if bool(artifacts):
        try:
            if trainer_native_ok:
                # Optionally force a single seed for embedding artifacts (useful for quick CI runs)
                seeds_for_emb = seed_list
                if artifacts_embed_seed is not None:
                    seeds_for_emb = [int(artifacts_embed_seed)]

                # T_default is known in trainer-native path
                T_default = int(T) if "T" in locals() else None

                _write_first_class_artifacts(
                    outdir=outdir,
                    cfg=cfg if isinstance(cfg, dict) else None,
                    conditions=plot_runners,                 # models + dataloaders per condition
                    structured_out=structured_out,
                    seeds=seeds_for_emb,
                    episodes=int(min(int(episodes), int(artifacts_embed_episodes))),
                    device=torch_device,
                    T_default=T_default,
                    max_nodes=int(artifacts_embeddings_max_nodes),
                    want_embeddings=bool(artifacts_embeddings),
                    want_stats=bool(artifacts_stats),
                    emb_mod=emb_mod,
                    stats_mod=stats_mod,
                )
            else:
                # Non-trainer-native: still emit stats.json/table from summaries (no embeddings)
                _write_first_class_artifacts(
                    outdir=outdir,
                    cfg=None,
                    conditions={},                           # no dataloaders available
                    structured_out=structured_out,
                    seeds=seed_list,
                    episodes=int(min(int(episodes), int(artifacts_embed_episodes))),
                    device=torch.device("cpu"),
                    T_default=None,
                    max_nodes=int(artifacts_embeddings_max_nodes),
                    want_embeddings=False,
                    want_stats=bool(artifacts_stats),
                    emb_mod=emb_mod,
                    stats_mod=stats_mod,
                )
        except Exception as e:
            print(f"[warn] artifact generation failed: {type(e).__name__}: {e}", file=sys.stderr)


    if plots and plot_mod is not None:
        if not trainer_native_ok:
            print("[warn] plots requested, but trainer-native eval path is unavailable; skipping plots.", file=sys.stderr)
        else:
            try:
                figdir = _ensure_dir(outdir / "figs")

                # Build a timeline rollout strictly for plotting.
                theta_scale = float((cfg.get("coin", {}) or {}).get("theta_scale", 1.0)) if "cfg" in locals() else 1.0
                theta_noise_std = float((cfg.get("coin", {}) or {}).get("theta_noise_std", 0.0)) if "cfg" in locals() else 0.0
                rollout_tl = _make_rollout_timeline_fn(
                    th,
                    pm=pm,
                    shift=shift,
                    family=coin_family,
                    adaptor=adaptor,
                    cdtype=cdtype,
                    device=torch_device,
                    init_mode=init_mode,
                    theta_scale=theta_scale,
                    theta_noise_std=theta_noise_std,
                )


                # ---------------- Plotting conventions (defendable) ----------------
                # 1) CI mode:
                #    - multi-seed: CI across seeds (per-seed mean over episodes)
                #    - single-seed: CI across episodes (avoids n=1 ddof warnings)
                if len(seed_list) < 2:
                    print(
                        "[warn] plots: only one seed provided; using CI across episodes (not across seeds). "
                        "Run >=2 seeds for seed-level CI plots.",
                        file=sys.stderr,
                    )

                # 2) Choose ONE node set ONCE (from base condition), reuse across all conditions.
                forced_nodes: list[int] = []
                try:
                    start_node = int(payload.get("start_node", 0))
                    forced_nodes.append(start_node)
                except Exception:
                    pass
                try:
                    t = payload.get("targets", None)
                    if isinstance(t, torch.Tensor) and t.numel() > 0:
                        forced_nodes.extend([int(v) for v in t.detach().cpu().tolist()])
                except Exception:
                    pass
                # dedup preserving order
                _seen = set()
                forced_nodes = [v for v in forced_nodes if (v not in _seen and not _seen.add(v))]

                nodes_shared: list[int] | None = None

                # Compute nodes_shared from the BASE condition only (stable comparisons).
                if base_tag in plot_runners:
                    rr0 = plot_runners[base_tag]
                    m0 = rr0["model"]
                    dl0 = rr0["dataloader_factory"]

                    Pt0, meta0 = _collect_Pt_samples_for_plotting(
                        model=m0,
                        dataloader_factory=dl0,
                        rollout_timeline_fn=rollout_tl,
                        seeds=seed_list,
                        episodes=int(episodes),

                        ckpt_path=ckpt_path,
                        artifacts_request=meta.get("artifacts_request", None),
                    )

                    # Need at least 1 sample to pick nodes; (>=2 only matters for CI shading)
                    Pmean0 = Pt0.mean(axis=0)  # (T+1, N)
                    score0 = Pmean0.max(axis=0)  # (N,)
                    order0 = np.argsort(-score0).tolist()
                    Nn0 = int(Pmean0.shape[1])

                    nodes_shared = []
                    for v in forced_nodes + order0:
                        v = int(v)
                        if 0 <= v < Nn0 and v not in nodes_shared:
                            nodes_shared.append(v)
                        if len(nodes_shared) >= 12:
                            break

                    # Persist node choice for thesis reproducibility
                    try:
                        (figdir / "nodes_shared.json").write_text(
                            _json_dump(
                                {
                                    "picked_from": base_tag,
                                    "forced_nodes": forced_nodes,
                                    "nodes_shared": nodes_shared,
                                    "rule": "top-by-max(Pmean_over_time) on base condition; reused for all conditions",
                                    "ci_mode_base": meta0.get("ci_mode", None),
                                    "n_samples_base": meta0.get("n_samples", None),
                                }
                            ),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass





                for cond_tag, rr in plot_runners.items():
                    safe = _sanitize_filename(cond_tag)

                    m = rr["model"]
                    dl = rr["dataloader_factory"]
                    meta_rr = rr.get("meta", {}) if isinstance(rr, dict) else {}

                    # Default: use base timeline rollout
                    rollout_tl_used = rollout_tl

                    # Optional: if ablations module provides an ablation_cfg + rollout_with_ablation,
                    # wrap the timeline rollout so plotting exactly matches the ablation semantics.
                    if abl_mod is not None and isinstance(meta_rr, Mapping):
                        ab_cfg_dict = meta_rr.get("ablation_cfg", None)
                        if (
                            isinstance(ab_cfg_dict, Mapping)
                            and hasattr(abl_mod, "AblationConfig")
                            and hasattr(abl_mod, "rollout_with_ablation")
                        ):
                            try:
                                ctor = getattr(abl_mod, "AblationConfig")
                                sig = inspect.signature(ctor)
                                allowed = set(sig.parameters.keys())
                                kwargs_cfg = {k: ab_cfg_dict[k] for k in ab_cfg_dict.keys() if k in allowed}
                                ab_cfg = ctor(**kwargs_cfg)

                                def _rollout_tl_wrapped(model0: torch.nn.Module, batch0: Mapping[str, Any]) -> torch.Tensor:
                                    # rollout_with_ablation expects rollout_fn -> (P, aux)
                                    def _inner_rollout(mm: torch.nn.Module, bb: dict) -> tuple[torch.Tensor, dict]:
                                        return rollout_tl(mm, bb), {}

                                    P_out, _aux = abl_mod.rollout_with_ablation(
                                        base_policy=model0,
                                        rollout_fn=_inner_rollout,
                                        batch=dict(batch0),
                                        cfg=ab_cfg,
                                    )
                                    return P_out

                                rollout_tl_used = _rollout_tl_wrapped
                            except Exception as e:
                                print(f"[warn] plotting ablation wrapper failed for {cond_tag}: {e}", file=sys.stderr)

                    Pt, pt_meta = _collect_Pt_samples_for_plotting(
                        model=m,
                        dataloader_factory=dl,
                        rollout_timeline_fn=rollout_tl_used,
                        seeds=seed_list,
                        episodes=int(episodes),

                        ckpt_path=ckpt_path,
                        artifacts_request=meta.get("artifacts_request", None),


                    )  # (K, T+1, N)

                    # If K<2, most CI code will warn (ddof/div0). Skip CI plots in that case.
                    if int(Pt.shape[0]) < 2:
                        print(
                            f"[warn] plots: {cond_tag} produced only K={Pt.shape[0]} sample(s); "
                            "skipping CI-based plots to avoid misleading uncertainty.",
                            file=sys.stderr,
                        )
                        continue

                    ci_note = f"CI across {pt_meta.get('ci_mode','?')} (K={pt_meta.get('n_samples','?')})"



                    # --- NEW: pick ONE consistent node set for all conditions (defendable comparisons) ---
                    if "nodes_shared" not in locals():
                        nodes_shared = None
                        forced_nodes = []
                        try:
                            start_node = int(payload.get("start_node", 0))
                            forced_nodes.append(start_node)
                        except Exception:
                            pass
                        try:
                            t = payload.get("targets", None)
                            if isinstance(t, torch.Tensor) and t.numel() > 0:
                                forced_nodes.extend([int(v) for v in t.detach().cpu().tolist()])
                        except Exception:
                            pass
                        # dedup preserving order
                        seen = set()
                        forced_nodes = [v for v in forced_nodes if (v not in seen and not seen.add(v))]

                    if nodes_shared is None:
                        Pmean = Pt.mean(axis=0)          # (T+1, N)
                        score = Pmean.max(axis=0)        # (N,)
                        order = np.argsort(-score).tolist()
                        Nn = int(Pmean.shape[1])
                        nodes_shared = []
                        for v in forced_nodes + order:
                            v = int(v)
                            if 0 <= v < Nn and v not in nodes_shared:
                                nodes_shared.append(v)
                            if len(nodes_shared) >= 12:
                                break












                    # TV curves are only meaningful/claimed for mixing suites
                    if is_mixing and hasattr(plot_mod, "plot_tv_curves"):
                        plot_mod.plot_tv_curves(
                            Pt,
                            savepath=(figdir / f"tv__{safe}.png"),
                            title=f"{suite} — {cond_tag} — TV-to-uniform — {ci_note}",
                        )

                    # NodePermute: node identities are meaningless, so timelines are typically not interpretable
                    is_nodeperm = ("nodepermute" in cond_tag.lower())
                    if (not is_nodeperm) and hasattr(plot_mod, "plot_position_timelines"):
                        plot_mod.plot_position_timelines(
                            Pt,
                            nodes=nodes_shared,
                            topk=None,
                            savepath=(figdir / f"Pt__{safe}.png"),
                            title=f"{suite} — {cond_tag} — mean Pt — {ci_note}",
                        )



                        
            except Exception as e:
                print(f"[warn] plot generation failed: {e}", file=sys.stderr)




    if report:
        if report_mod is None:
            print("[warn] report requested, but acpl.eval.reporting could not be imported; skipping.", file=sys.stderr)
        else:
            ok = _run_reporting_best_effort(report_mod, outdir)
            if not ok:
                print("[warn] report requested, but reporting entrypoint failed; skipping.", file=sys.stderr)










# --------------------------------- CLI ----------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run evaluation suites from saved checkpoints (Phase B7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    
    
    
    p.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint file or directory (recommended).")



     p.add_argument(
         "--config",
         type=str,
         default=None,
         help=(
             "Optional config YAML/JSON path. If provided and --ckpt is omitted, "
             "eval will run using this config (useful for baseline-only evaluations). "
             "If --ckpt is provided, config is loaded from the checkpoint unless overridden via --override."
         ),
     )



    p.add_argument(
        "--policy",
        type=str,
        default="ckpt",
        choices=("ckpt", "baseline"),
        help="Which policy to evaluate: 'ckpt' loads the trained model; 'baseline' builds a baseline policy.",
    )

    p.add_argument(
        "--baseline",
        type=str,
        default="hadamard",
        help="Baseline kind (e.g., hadamard, grover, random, global_schedule).",
    )

    p.add_argument(
        "--baseline_coins",
        type=str,
        nargs="*",
        default=[],
        help="Baseline coin schedule kwargs as key=val (e.g., seed=0 mode=time).",
    )

    p.add_argument(
        "--baseline_policy",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Baseline policy wrapper kwargs as key=val (e.g., theta_scale=1.0 theta_noise_std=0.0 "
            "embedding.mode=identity embedding.out_dim=32 embedding.normalize=true)."
        ),
    )
    
    
    
    
    
    
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
        "--report",
        action="store_true",
        help="Run acpl.eval.reporting (best-effort) after evaluation to produce additional tables/figures.",
    )
    p.add_argument(
        "--strict_ablations",
        action="store_true",
        help="If set, any ablation build failure is a hard error (otherwise it is skipped with a warning).",
    )
 
     # Overrides (trainer-style dotted overrides)
    p.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Trainer-style overrides as key=val (supports dotted keys). Example: model.gnn.hidden=128 seed=123",
    )


    # First-class artifacts (B7)
    p.add_argument("--no_artifacts", action="store_true", help="Disable writing outdir/artifacts/*.")
    p.add_argument("--no_embeddings", action="store_true", help="Disable embeddings artifacts (artifacts/embeddings/*).")
    p.add_argument("--no_stats", action="store_true", help="Disable stats artifacts (artifacts/stats.json + stats_table.txt).")
    p.add_argument(
        "--embeddings_episodes",
        type=int,
        default=32,
        help="Episodes per seed used to compute embedding artifacts (caps at --episodes).",
    )
    p.add_argument(
        "--embeddings_seed",
        type=int,
        default=None,
        help="If set, compute embedding artifacts using ONLY this seed id (faster quick checks).",
    )
    p.add_argument(
        "--embeddings_max_nodes",
        type=int,
        default=4096,
        help="Safety cap: skip embedding artifacts if N exceeds this value.",
    )





    p.add_argument(
        "--report",
        action="store_true",
        help="Run acpl.eval.reporting on the produced eval directory (best-effort).",
    )
    p.add_argument(
        "--strict_ablations",
        action="store_true",
        help="Fail the run if any requested ablation cannot be applied (recommended for thesis runs).",
    )


    p.add_argument(
        "--robust_sweep",
        action="store_true",
        help=(
            "Run robustness sigma sweeps defined in cfg['eval']['robustness'] "
            "(writes outdir/robust_sweeps/*)."
        ),
    )
    p.add_argument(
        "--robust_sweep_kinds",
        type=str,
        nargs="*",
        default=[],
        help="Subset of kinds to sweep (default: edge_phase coin_dephase).",
    )
    p.add_argument(
        "--robust_sweep_sigma",
        type=str,
        default=None,
        help="Override sigma grid for ALL kinds: e.g. '0,0.05,0.1,0.2'.",
    )
    p.add_argument(
        "--robust_sweep_trials",
        type=int,
        default=None,
        help="Override number of disorder trials per sigma (default from YAML or 10).",
    )
    p.add_argument(
        "--robust_sweep_bootstrap",
        type=int,
        default=200,
        help="Bootstrap samples per trial (lower=faster).",
    )
    p.add_argument(
        "--robust_sweep_include_ablations",
        action="store_true",
        help="Also sweep each ablation condition (expensive). Default: sweep base condition only.",
    )
    p.add_argument(
        "--robust_sweep_max_plot_metrics",
        type=int,
        default=6,
        help="Max number of metrics to include in each sweep plot.",
    )



    p.add_argument(
         "--override",
         type=str,
         nargs="*",
         default=[],
         help="Extra key=val overrides passed to the evaluation protocol (e.g., task=search horizon=128).",
    )




    p.add_argument(
        "--embeddings_max_nodes",
         type=int,
         default=4096,
         help="Skip embeddings artifacts when probed graph has more than this many nodes (avoid OOM).",
    )
 
     # Robustness sweeps (Phase 6)
    p.add_argument(
        "--robust_sweep",
        action="store_true",
        help="Enable robustness sigma sweeps (requires trainer-native eval path).",
    )
    p.add_argument(
         "--robust_kinds",
         type=str,
         nargs="*",
         default=[],
         help="Disorder kinds to sweep (e.g., edge_phase coin_dephase). Default uses YAML or both.",
    )
    p.add_argument(
         "--robust_sigmas",
         type=str,
         default=None,
         help="Sigma grid override as comma/space list, e.g. '0,0.05,0.1,0.2' or '0 0.05 0.1'.",
    )
    p.add_argument(
         "--robust_trials",
         type=int,
         default=None,
         help="Trials per sigma (override YAML). Each trial uses a distinct disorder_seed.",
    )
    p.add_argument(
        "--robust_bootstrap",
        type=int,
        default=200,
        help="Bootstrap samples per trial for CI in sweeps (smaller=cheaper).",
    )
    p.add_argument(
        "--robust_include_ablations",
        action="store_true",
        help="If set, also run robustness sweeps for ablation conditions (otherwise base condition only).",
    )
    p.add_argument(
        "--robust_max_plot_metrics",
        type=int,
        default=6,
        help="Max number of metrics to show on sweep plots (keep readability).",
    )





    return p.parse_args(argv)



# ------------------------------ CLI helpers -----------------------------------

def _coerce_scalar(s: str) -> Any:
     """
     Parse a scalar string into bool/int/float/json/list/dict where appropriate.
     Conservative and deterministic.
     """
     if s is None:
         return None
     x = str(s).strip()
     if x == "":
         return ""
     low = x.lower()
     if low in ("true", "false"):
         return low == "true"
     if low in ("none", "null"):
         return None
     # int
     try:
         if re.fullmatch(r"[+-]?\d+", x):
             return int(x)
     except Exception:
         pass
     # float
     try:
         if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", x):
             return float(x)
     except Exception:
         pass
     # json object/array
     if (x.startswith("{") and x.endswith("}")) or (x.startswith("[") and x.endswith("]")):
         try:
             return json.loads(x)
         except Exception:
             return x
     return x
 






def _set_nested_dotted(d: dict[str, Any], key: str, value: Any) -> None:
     """
     Set d["a"]["b"]["c"] = value for key "a.b.c".
     Creates intermediate dicts as needed.
     """
     parts = [p for p in str(key).split(".") if p]
     if not parts:
         return
     cur = d
     for p in parts[:-1]:
         nxt = cur.get(p, None)
         if not isinstance(nxt, dict):
             nxt = {}
             cur[p] = nxt
         cur = nxt
     cur[parts[-1]] = value
 

def _parse_kv_list(items: list[str]) -> dict[str, Any]:
     """
     Parse ["a=1", "b.c=true", "name=foo"] into nested dicts with type coercion.
     """
     out: dict[str, Any] = {}
     for it in items or []:
         s = str(it).strip()
         if not s:
             continue
         if "=" not in s:
             # allow bare flags like "foo" -> True
             _set_nested_dotted(out, s, True)
             continue
         k, v = s.split("=", 1)
         k = k.strip()
         v = v.strip()
         if not k:
             continue
         _set_nested_dotted(out, k, _coerce_scalar(v))
     return out

def _parse_seeds_arg(s: str) -> int | list[int]:
     """
     Seeds argument:
       - "5" -> int(5) meaning range(5)
       - "0,1,2" -> explicit list
       - "0 1 2" -> explicit list
     """
     x = str(s).strip()
     if x == "":
         return 0
     # pure int
     if re.fullmatch(r"\d+", x):
         return int(x)
     # list
     toks = [t.strip() for t in (x.split(",") if "," in x else x.split())]
     out: list[int] = []
     for t in toks:
         if not t:
             continue
         out.append(int(t))
     return out


def _load_config_file(path: Path) -> dict[str, Any]:
     p = _as_path(path)
     if not p.exists():
         raise FileNotFoundError(f"Config file not found: {p}")
     txt = p.read_text(encoding="utf-8")
     # YAML preferred if available
     if p.suffix.lower() in (".yml", ".yaml"):
         try:
             import yaml as _yaml
             obj = _yaml.safe_load(txt)
         except Exception as e:
             raise RuntimeError(f"Failed to parse YAML config: {p} ({e})")
     else:
         obj = json.loads(txt)
     if not isinstance(obj, dict):
         raise TypeError(f"Config must be a dict at top-level: {p}")
     return obj


def _parse_seeds(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return [0]
    if "," in s:
        return [int(x) for x in s.split(",") if x != ""]
    # integer => range(N), BUT avoid the very common confusion where "--seeds 1"
    # is intended to mean "seed id 1" (not "one seed: seed 0").
    try:
        n = int(s)
        if n in (0, 1):
            return [n]
        return list(range(n))
    except Exception:
        # space-separated list?
        parts = s.split()
        return [int(x) for x in parts]













def _as_torch_dtype(x: Any) -> Any:
    # Accept torch.float32, "float32", "torch.float32"
    if isinstance(x, torch.dtype):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("torch."):
            s = s[len("torch.") :]
        if hasattr(torch, s):
            dt = getattr(torch, s)
            if isinstance(dt, torch.dtype):
                return dt
    return x


def _split_prefixed(flat: dict[str, Any], prefix: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split keys with prefix 'prefix.' into a separate dict without the prefix."""
    a: dict[str, Any] = {}
    b: dict[str, Any] = {}
    pre = prefix + "."
    for k, v in flat.items():
        if k.startswith(pre):
            a[k[len(pre) :]] = v
        else:
            b[k] = v
    return a, b


def _parse_baseline_policy_kwargs(tokens: list[str]) -> dict[str, Any]:
    """
    Supports:
      theta_scale=..., theta_noise_std=..., theta_clip=..., strict_theta_shape=...
      theta_dtype=float32
      embedding.mode=identity|linear_fixed|mlp_fixed
      embedding.out_dim=...
      embedding.seed=...
      embedding.normalize=true
      embedding.dtype=float32
    """
    flat = _kv_overrides(tokens)

    # dtype normalization if provided
    if "theta_dtype" in flat:
        flat["theta_dtype"] = _as_torch_dtype(flat["theta_dtype"])

    emb_flat, rest = _split_prefixed(flat, "embedding")
    if emb_flat:
        try:
            from acpl.baselines.policies import NodeEmbeddingConfig
        except Exception as e:
            raise ImportError(
                "Baseline embedding overrides require acpl.baselines.policies.NodeEmbeddingConfig"
            ) from e

        if "dtype" in emb_flat:
            emb_flat["dtype"] = _as_torch_dtype(emb_flat["dtype"])

        emb_cfg = NodeEmbeddingConfig(
            mode=str(emb_flat.get("mode", "identity")),
            out_dim=emb_flat.get("out_dim", None),
            seed=int(emb_flat.get("seed", 0)),
            normalize=bool(emb_flat.get("normalize", False)),
            dtype=emb_flat.get("dtype", torch.float32),
        )
        rest["embedding"] = emb_cfg

    return rest


def _parse_baseline_coins_kwargs(tokens: list[str]) -> dict[str, Any]:
    # Coins kwargs are schedule-specific; just parse key=val with numbers/bools/json support.
    return _kv_overrides(tokens)

























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

    # ckpt policy must have a checkpoint
    if args.policy == "ckpt" and not args.ckpt:
        raise SystemExit("--ckpt is required when --policy=ckpt")

    # baseline mode: allow config-only runs if you added --config; otherwise require ckpt
    if args.policy == "baseline" and not args.ckpt and not args.config:
        raise SystemExit(
            "--ckpt is required when --policy=baseline unless you provide --config "
            "(baseline needs graph/task/T from config)."
        )

    ckpt_path = _as_path(args.ckpt) if args.ckpt else None
    if ckpt_path is None and args.config:
        ckpt_path = _as_path(args.config)

    if ckpt_path is None:
        raise SystemExit("Provide --ckpt (recommended) or --config (baseline-only mode).")

    outdir = _as_path(args.outdir)

    seeds = _parse_seeds(args.seeds)
    overrides = _kv_overrides(args.override)

    
    # If user provided --config, treat it as the config-source override
    # (even if --ckpt is also provided).
    if args.config:
        overrides = dict(overrides or {})
        overrides["_eval_config_path"] = str(_as_path(args.config))

    baseline_coins_kwargs = _parse_baseline_coins_kwargs(args.baseline_coins or [])
    baseline_policy_kwargs = _parse_baseline_policy_kwargs(args.baseline_policy or [])

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
        policy=str(args.policy),

        baseline_kind=str(args.baseline),
        baseline_coins_kwargs=baseline_coins_kwargs,
        baseline_policy_kwargs=baseline_policy_kwargs,
        report=bool(args.report),
        strict_ablations=bool(args.strict_ablations),

        robust_sweep=bool(args.robust_sweep),
        robust_sweep_kinds=list(args.robust_sweep_kinds or []),
        robust_sweep_sigma=args.robust_sweep_sigma,
        robust_sweep_trials=args.robust_sweep_trials,
        robust_sweep_bootstrap=int(args.robust_sweep_bootstrap),
        robust_sweep_include_ablations=bool(args.robust_sweep_include_ablations),
        robust_sweep_max_plot_metrics=int(args.robust_sweep_max_plot_metrics),

        artifacts=(not bool(args.no_artifacts)),
        artifacts_embeddings=(not bool(args.no_embeddings)),
        artifacts_stats=(not bool(args.no_stats)),
        artifacts_embed_episodes=int(args.embeddings_episodes),
        artifacts_embed_seed=(int(args.embeddings_seed) if args.embeddings_seed is not None else None),
        artifacts_embeddings_max_nodes=int(args.embeddings_max_nodes),
    )

    



if __name__ == "__main__":
    main()

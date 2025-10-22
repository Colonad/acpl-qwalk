# acpl/utils/checkpoint.py
# ACPL Phase B8 — Utils: atomic checkpoint save/load + resume helpers.
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

_LOG = logging.getLogger("acpl.checkpoint")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(logging.INFO)

# Optional deps (torch, numpy, random)
try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import numpy as _np  # type: ignore

    _HAS_NUMPY = True
except Exception:
    _np = None  # type: ignore
    _HAS_NUMPY = False

import random as _py_random

# -------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _fsync_dir(dirpath: str) -> None:
    """fsync the directory entry (important for durability on some filesystems)."""
    try:
        fd = os.open(dirpath, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # Not fatal; best effort.
        pass


def _atomic_write_bytes(path: str, data: bytes, mode: int = 0o644) -> None:
    """
    POSIX-safe atomic write:
    1) create a temp file in the same directory
    2) write+flush+fsync the file
    3) os.replace to the final path (atomic rename)
    4) fsync the directory
    """
    _ensure_dir(path)
    directory = os.path.dirname(path) or "."
    base = os.path.basename(path)
    # Use NamedTemporaryFile with delete=False to control fsync and permissions
    with tempfile.NamedTemporaryFile(prefix=f".{base}.tmp.", dir=directory, delete=False) as tmp:
        tmp_path = tmp.name
        try:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.chmod(tmp_path, mode)
            # Atomic replace
            os.replace(tmp_path, path)
            _fsync_dir(directory)
        except Exception:
            # Attempt cleanup on failure
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise


def _hash_bytes(data: bytes, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()


def _try_run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return 0, out.decode("utf-8", "ignore").strip()
    except Exception as e:
        return 1, str(e)


def _git_info(cwd: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "enabled": False,
        "commit": None,
        "dirty": None,
        "branch": None,
        "describe": None,
    }
    code, head = _try_run(["git", "rev-parse", "HEAD"])
    if code != 0:
        return info
    info["enabled"] = True
    info["commit"] = head
    code, dirty = _try_run(["git", "status", "--porcelain"])
    info["dirty"] = (dirty.strip() != "") if code == 0 else None
    code, br = _try_run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    info["branch"] = br if code == 0 else None
    code, desc = _try_run(["git", "describe", "--always", "--dirty", "--tags"])
    info["describe"] = desc if code == 0 else None
    return info


# -------------------------------------------------------------------------
# RNG state capture/restore (Python + NumPy + Torch)
# -------------------------------------------------------------------------


def capture_rng_state() -> dict[str, Any]:
    """Snapshot Python, NumPy, and Torch RNG states (when available)."""
    payload: dict[str, Any] = {}
    payload["python"] = _py_random.getstate()
    if _HAS_NUMPY:
        try:
            payload["numpy"] = _np.random.get_state()  # type: ignore[attr-defined]
        except Exception:
            payload["numpy"] = None
    if _HAS_TORCH:
        try:
            payload["torch"] = {
                "cpu": torch.get_rng_state().cpu().numpy().tolist(),  # type: ignore[attr-defined]
                "cuda": [t.cpu().numpy().tolist() for t in (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])],  # type: ignore[attr-defined]
            }
        except Exception:
            payload["torch"] = None
    return payload


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state captured by `capture_rng_state`."""
    try:
        if "python" in state and state["python"] is not None:
            _py_random.setstate(tuple(state["python"]))  # type: ignore[arg-type]
    except Exception as e:
        _LOG.warning("Failed to restore Python RNG state: %s", e)

    if _HAS_NUMPY:
        try:
            if "numpy" in state and state["numpy"] is not None:
                _np.random.set_state(tuple(state["numpy"]))  # type: ignore[attr-defined,arg-type]
        except Exception as e:
            _LOG.warning("Failed to restore NumPy RNG state: %s", e)

    if _HAS_TORCH and state.get("torch") is not None:
        try:
            tstate = state["torch"]
            if "cpu" in tstate and tstate["cpu"] is not None:
                torch.set_rng_state(torch.tensor(tstate["cpu"], dtype=torch.uint8))  # type: ignore[attr-defined]
            if "cuda" in tstate and tstate["cuda"] is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
                tensors = [
                    torch.tensor(arr, dtype=torch.uint8, device="cuda") for arr in tstate["cuda"]
                ]
                torch.cuda.set_rng_state_all(tensors)  # type: ignore[attr-defined]
        except Exception as e:
            _LOG.warning("Failed to restore Torch RNG state: %s", e)


# -------------------------------------------------------------------------
# Serialization
# -------------------------------------------------------------------------


def _torch_serialize(obj: Any) -> bytes:
    """Serialize with torch.save to a bytes buffer (zipfile format)."""
    assert _HAS_TORCH, "torch not available"
    buf = io.BytesIO()
    # new zipfile serialization is default in modern torch; keep kwargs explicit
    torch.save(obj, buf)  # type: ignore[arg-type]
    return buf.getvalue()


def _torch_deserialize(
    data: bytes, map_location: str | torch.device | dict[str, str] | None = "cpu"
) -> Any:
    assert _HAS_TORCH, "torch not available"
    buf = io.BytesIO(data)
    obj = torch.load(buf, map_location=map_location)  # type: ignore[arg-type]
    return obj


def _pickle_serialize(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_deserialize(data: bytes) -> Any:
    return pickle.loads(data)


def _serialize(obj: Any, prefer_torch: bool = True) -> tuple[bytes, str]:
    """Return (bytes, codec) where codec indicates deserializer."""
    if prefer_torch and _HAS_TORCH:
        try:
            b = _torch_serialize(obj)
            return b, "torch"
        except Exception as e:
            _LOG.warning("torch serialization failed (%s); falling back to pickle", e)
    # Pickle fallback
    return _pickle_serialize(obj), "pickle"


def _deserialize(data: bytes, codec: str, map_location: Any | None = "cpu") -> Any:
    if codec == "torch":
        if not _HAS_TORCH:
            raise RuntimeError("Checkpoint was saved with torch codec but torch is not available.")
        return _torch_deserialize(data, map_location=map_location)
    elif codec == "pickle":
        return _pickle_deserialize(data)
    else:
        raise ValueError(f"Unknown codec: {codec}")


# -------------------------------------------------------------------------
# Checkpoint payload assembly
# -------------------------------------------------------------------------


def _state_dict(obj: Any) -> Mapping[str, Any] | None:
    """Best-effort state_dict extraction."""
    if obj is None:
        return None
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        try:
            return obj.state_dict()  # type: ignore[no-any-return]
        except Exception as e:
            _LOG.warning("state_dict() failed for %r: %s", obj, e)
    return None


def _load_state_dict(obj: Any, state: Mapping[str, Any], strict: bool = True) -> None:
    if obj is None or state is None:
        return
    if hasattr(obj, "load_state_dict") and callable(obj.load_state_dict):
        obj.load_state_dict(state, strict=strict)  # type: ignore[arg-type]
    else:
        raise AttributeError(f"Object {obj} has no load_state_dict method")


def build_checkpoint_payload(
    *,
    step: int | None = None,
    epoch: int | None = None,
    meta: dict[str, Any] | None = None,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    extra: dict[str, Any] | None = None,
    capture_rng: bool = True,
) -> dict[str, Any]:
    """
    Assemble a structured checkpoint dict.
    Includes model/optimizer/scheduler/scaler state_dicts when provided,
    plus RNG and git info.
    """
    payload: dict[str, Any] = {
        "format": "acpl.ckpt.v1",
        "created_at": time.time(),
        "step": int(step) if step is not None else None,
        "epoch": int(epoch) if epoch is not None else None,
        "meta": meta or {},
        "git": _git_info(),
        "states": {},
        "rng": capture_rng_state() if capture_rng else None,
        "extra": extra or {},
    }
    states = payload["states"]
    states["model"] = _state_dict(model)
    states["optimizer"] = _state_dict(optimizer)
    states["scheduler"] = _state_dict(scheduler)
    states["scaler"] = _state_dict(scaler)
    return payload


# -------------------------------------------------------------------------
# Save / load / rotation
# -------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    payload: Mapping[str, Any],
    *,
    prefer_torch_codec: bool = True,
    write_meta_json: bool = True,
    keep_last: int = 3,
    latest_symlink: bool = True,
) -> str:
    """
    Atomically save a checkpoint payload to `path`.
    Returns the path written.

    - Uses torch.save when available for best tensor fidelity; otherwise pickle.
    - Writes a sibling small JSON (<path>.meta.json) for quick inspection (optional).
    - Keeps `keep_last` most recent checkpoints in the directory (by mtime).
    - Maintains a 'latest' symlink in the directory if requested.
    """
    # Serialize (outer envelope: bytes + header)
    b, codec = _serialize(dict(payload), prefer_torch=prefer_torch_codec)
    envelope = {
        "codec": codec,
        "hash": _hash_bytes(b),
        "size": len(b),
        "payload": b,  # raw bytes to maximize compatibility
    }
    # Write binary
    final_path = path
    _atomic_write_bytes(final_path, pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    _LOG.info(
        "Saved checkpoint: %s (codec=%s, size=%.2f MB)", final_path, codec, len(b) / (1024 * 1024)
    )

    # Optional human-readable sidecar
    if write_meta_json:
        meta = {
            "path": os.path.abspath(final_path),
            "created_at": payload.get("created_at"),
            "step": payload.get("step"),
            "epoch": payload.get("epoch"),
            "codec": codec,
            "size": len(b),
            "git": payload.get("git"),
        }
        meta_path = final_path + ".meta.json"
        try:
            _atomic_write_bytes(
                meta_path, json.dumps(meta, indent=2, sort_keys=True).encode("utf-8")
            )
        except Exception as e:
            _LOG.warning("Could not write meta JSON %s: %s", meta_path, e)

    # Maintain 'latest' symlink
    if latest_symlink:
        try:
            d = os.path.dirname(os.path.abspath(final_path)) or "."
            latest_path = os.path.join(d, "latest.ckpt")
            tmp_link = latest_path + ".tmp"
            if os.path.islink(tmp_link) or os.path.exists(tmp_link):
                try:
                    os.remove(tmp_link)
                except Exception:
                    pass
            try:
                os.symlink(os.path.basename(final_path), tmp_link)
                os.replace(tmp_link, latest_path)
            except OSError:
                # Windows or restricted FS: copy small marker file instead
                with open(tmp_link, "w", encoding="utf-8") as f:
                    f.write(os.path.basename(final_path))
                os.replace(tmp_link, latest_path)
        except Exception as e:
            _LOG.debug("latest symlink update failed: %s", e)

    # Rotation
    try:
        _rotate_checkpoints(os.path.dirname(final_path) or ".", keep_last=keep_last)
    except Exception as e:
        _LOG.debug("Rotation failed: %s", e)

    return final_path


def _rotate_checkpoints(directory: str, *, keep_last: int) -> None:
    if keep_last <= 0:
        return
    candidates = []
    for name in os.listdir(directory):
        if name.endswith(".ckpt") or name.endswith(".pt") or name.endswith(".pth"):
            path = os.path.join(directory, name)
            try:
                st = os.stat(path)
            except Exception:
                continue
            candidates.append((st.st_mtime, path))
    candidates.sort(reverse=True)  # newest first
    for _, path in candidates[keep_last:]:
        try:
            os.remove(path)
            meta = path + ".meta.json"
            if os.path.exists(meta):
                os.remove(meta)
        except Exception:
            pass


def load_checkpoint(
    path: str,
    *,
    map_location: Any | None = "cpu",
    strict: bool = True,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """
    Load a checkpoint file saved by `save_checkpoint`.

    If model/optimizer/... are passed, their state_dicts are restored.
    Returns the full payload dict (including meta, states, extra).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = open(path, "rb").read()
    envelope = pickle.loads(raw)
    if not isinstance(envelope, dict) or "payload" not in envelope or "codec" not in envelope:
        raise RuntimeError("Malformed checkpoint envelope.")
    data: bytes = envelope["payload"]
    codec: str = envelope["codec"]
    h = envelope.get("hash")
    if h is not None:
        calc = _hash_bytes(data)
        if calc != h:
            raise RuntimeError(f"Checkpoint hash mismatch (expected {h}, got {calc}).")

    obj = _deserialize(data, codec=codec, map_location=map_location)
    if not isinstance(obj, dict) or obj.get("format") != "acpl.ckpt.v1":
        # Back-compat: maybe the payload itself is the torch dict
        payload = {"format": "unknown", "states": obj}
    else:
        payload = obj

    states = payload.get("states", {}) or {}
    # Restore components if provided
    if model is not None and states.get("model") is not None:
        _load_state_dict(model, states["model"], strict=strict)
    if optimizer is not None and states.get("optimizer") is not None:
        _load_state_dict(optimizer, states["optimizer"], strict=strict)
    if scheduler is not None and states.get("scheduler") is not None:
        _load_state_dict(scheduler, states["scheduler"], strict=strict)
    if scaler is not None and states.get("scaler") is not None:
        _load_state_dict(scaler, states["scaler"], strict=strict)

    if restore_rng and payload.get("rng") is not None:
        try:
            restore_rng_state(payload["rng"])
        except Exception as e:
            _LOG.warning("RNG restore failed: %s", e)

    _LOG.info(
        "Loaded checkpoint: %s (codec=%s, step=%s, epoch=%s)",
        path,
        codec,
        payload.get("step"),
        payload.get("epoch"),
    )
    return payload


# -------------------------------------------------------------------------
# High-level resume helpers
# -------------------------------------------------------------------------

_CKPT_NUM_RE = re.compile(r".*?(\d+).*")


def _infer_step_from_name(name: str) -> int | None:
    """
    Try to parse a step number from filename (e.g., ckpt_step_000123.ckpt).
    """
    m = _CKPT_NUM_RE.match(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def latest_checkpoint_in_dir(directory: str, *, prefer_step: bool = True) -> str | None:
    if not os.path.isdir(directory):
        return None
    best_path = None
    best_key = -1
    for name in os.listdir(directory):
        if not (name.endswith(".ckpt") or name.endswith(".pt") or name.endswith(".pth")):
            continue
        p = os.path.join(directory, name)
        if not os.path.isfile(p):
            continue
        if prefer_step:
            s = _infer_step_from_name(name)
            key = s if s is not None else -1
        else:
            try:
                key = os.stat(p).st_mtime
            except Exception:
                key = -1
        if key > best_key:
            best_key = key
            best_path = p
    # Fallback to 'latest.ckpt' symlink/marker
    if best_path is None:
        cand = os.path.join(directory, "latest.ckpt")
        if os.path.exists(cand):
            try:
                if os.path.islink(cand):
                    target = os.readlink(cand)
                    best_path = os.path.join(directory, target)
                else:
                    # marker file containing the filename
                    with open(cand, encoding="utf-8") as f:
                        target = f.read().strip()
                    best_path = os.path.join(directory, target)
            except Exception:
                best_path = None
    return best_path


def resume_from_directory(
    directory: str,
    *,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: Any | None = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Find the most recent checkpoint in `directory` and load it (if present).
    Returns (payload, path). If none found, returns (None, None).
    """
    ckpt = latest_checkpoint_in_dir(directory)
    if ckpt is None:
        _LOG.info("No checkpoint found in %s; starting fresh.", directory)
        return None, None
    payload = load_checkpoint(
        ckpt,
        map_location=map_location,
        strict=strict,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        restore_rng=restore_rng,
    )
    return payload, ckpt


# -------------------------------------------------------------------------
# Friendly, opinionated single-call API for training loops
# -------------------------------------------------------------------------


def save_components(
    path: str,
    *,
    step: int | None,
    epoch: int | None,
    model: Any,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    meta: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    keep_last: int = 3,
    prefer_torch_codec: bool = True,
) -> str:
    """
    Convenience wrapper:
        save_components("runs/exp/ckpt_step_%06d.ckpt" % step, step=step, epoch=epoch, model=..., optimizer=..., ...)
    """
    payload = build_checkpoint_payload(
        step=step,
        epoch=epoch,
        meta=meta,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        extra=extra,
        capture_rng=True,
    )
    return save_checkpoint(
        path,
        payload,
        prefer_torch_codec=prefer_torch_codec,
        write_meta_json=True,
        keep_last=keep_last,
        latest_symlink=True,
    )


def try_autoresume(
    directory: str,
    *,
    model: Any,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: Any | None = "cpu",
    strict: bool = True,
) -> tuple[int, int]:
    """
    Look for a checkpoint in `directory` and restore it. Returns (start_step, start_epoch).
    If not found, returns (0, 0).
    """
    payload, path = resume_from_directory(
        directory,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=map_location,
        strict=strict,
        restore_rng=True,
    )
    if payload is None:
        return 0, 0
    step = int(payload.get("step") or 0)
    epoch = int(payload.get("epoch") or 0)
    _LOG.info("Resumed from %s (step=%d, epoch=%d)", path, step, epoch)
    return step, epoch


# -------------------------------------------------------------------------
# Minimal CLI (optional)
# -------------------------------------------------------------------------


def _print_meta(path: str) -> None:
    p = load_checkpoint(path, map_location="cpu", restore_rng=False)
    keep = {k: p.get(k) for k in ("format", "created_at", "step", "epoch", "git")}
    print(json.dumps(keep, indent=2, sort_keys=True))


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="acpl.checkpoint", description="ACPL checkpoint utility")
    sub = ap.add_subparsers(dest="cmd")

    ls = sub.add_parser("latest", help="print latest checkpoint path in a directory")
    ls.add_argument("directory", type=str)

    meta = sub.add_parser("meta", help="print checkpoint meta")
    meta.add_argument("path", type=str)

    args = ap.parse_args()
    if args.cmd == "latest":
        p = latest_checkpoint_in_dir(args.directory)
        if p is None:
            print("NONE")
            return 1
        print(p)
        return 0
    elif args.cmd == "meta":
        _print_meta(args.path)
        return 0
    else:
        ap.print_help()
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())

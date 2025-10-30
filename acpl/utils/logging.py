# acpl/utils/logging.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

__all__ = ["MetricLogger", "MetricLoggerConfig"]


def _utc_iso8601() -> str:
    return datetime.now(UTC).isoformat()


def _to_builtin(obj):
    # make JSON-safe: cast numpy/torch scalars to Python scalars
    try:
        import numpy as _np  # type: ignore
    except Exception:  # pragma: no cover
        _np = None
    try:
        import torch as _th  # type: ignore
    except Exception:  # pragma: no cover
        _th = None

    if _np is not None and isinstance(obj, _np.generic):  # e.g., np.float32
        return obj.item()
    if _th is not None and _th.is_tensor(obj):
        return _to_builtin(obj.item())
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    # last resort: try float()
    try:
        return float(obj)
    except Exception:
        return str(obj)


@dataclass
class MetricLoggerConfig:
    # core backend selection
    backend: str = "plain"  # "plain" | "tensorboard" | "wandb"
    enable: bool = True

    # locations & naming
    log_dir: str | None = None  # directory for TB/W&B artifacts AND metrics.jsonl (if enabled)
    jsonl_name: str = "metrics.jsonl"
    write_jsonl: bool = True  # write a JSONL stream alongside other backends

    # wandb specifics
    project: str | None = None
    run_name: str | None = None

    # key names / misc
    step_key: str = "step"
    echo_plain: bool = True  # print to stdout even if TB/W&B are used


class MetricLogger:
    """
    Unified tiny metric logger.

    Backends:
      - "plain":       print to stdout
      - "tensorboard": scalars to TensorBoard
      - "wandb":       scalar dicts to Weights & Biases

    Extras:
      - Optional JSONL streaming to <log_dir>/<jsonl_name> for easy post-hoc checks
      - Context-manager support: `with MetricLogger(cfg) as log: ...`
    """

    def __init__(self, cfg: MetricLoggerConfig):
        self.cfg = cfg
        self.backend = (cfg.backend or "plain").lower()
        self._tb = None
        self._wb = None
        self._jsonl_fp = None  # type: Optional[object]
        self._opened_paths: list[Path] = []

        if not cfg.enable:
            self.backend = "off"
            return

        # Ensure log_dir exists if provided / needed
        log_dir_path: Path | None = None
        if cfg.log_dir:
            log_dir_path = Path(cfg.log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize backend(s) with graceful fallback
        if self.backend in ("tensorboard", "tb"):
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                self._tb = SummaryWriter(log_dir=str(log_dir_path) if log_dir_path else None)
                self.backend = "tensorboard"
            except Exception as e:  # pragma: no cover
                print(
                    f"[MetricLogger] TensorBoard not available ({e}); falling back to plain.",
                    file=sys.stderr,
                )
                self.backend = "plain"

        if self.backend == "wandb":
            try:
                import wandb  # type: ignore

                wandb_kwargs = {}
                if cfg.project:
                    wandb_kwargs["project"] = cfg.project
                if cfg.run_name:
                    wandb_kwargs["name"] = cfg.run_name
                if log_dir_path:
                    # ensure wandb dir exists; wandb may create an internal subdir under this
                    wandb_kwargs["dir"] = str(log_dir_path)

                wandb.init(**wandb_kwargs)
                self._wb = wandb
            except Exception as e:  # pragma: no cover
                print(
                    f"[MetricLogger] wandb not available ({e}); falling back to plain.",
                    file=sys.stderr,
                )
                self.backend = "plain"

        # JSONL stream (optional, independent of backend)
        if cfg.write_jsonl and log_dir_path is not None:
            try:
                jsonl_path = log_dir_path / cfg.jsonl_name
                self._jsonl_fp = open(jsonl_path, "a", encoding="utf-8")
                self._opened_paths.append(jsonl_path)
            except Exception as e:  # pragma: no cover
                print(f"[MetricLogger] Could not open JSONL at {jsonl_path}: {e}", file=sys.stderr)
                self._jsonl_fp = None

    # --- context manager -----------------------------------------------------

    def __enter__(self) -> MetricLogger:
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        # do not suppress exceptions
        return False

    # --- internals -----------------------------------------------------------

    def _emit_plain(self, msg: str):
        # Always to stdout for consistency
        print(msg)

    def _emit_jsonl(self, record: dict):
        if not self._jsonl_fp:
            return
        try:
            self._jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_fp.flush()
        except Exception as e:  # pragma: no cover
            # don't crash training because of logging
            print(f"[MetricLogger] JSONL write failed: {e}", file=sys.stderr)

    # --- public API ----------------------------------------------------------

    def close(self):
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception:
                pass
            self._tb = None

        if self._wb is not None:
            try:
                self._wb.finish()
            except Exception:
                pass
            self._wb = None

        if self._jsonl_fp is not None:
            try:
                self._jsonl_fp.flush()
                self._jsonl_fp.close()
            except Exception:
                pass
            self._jsonl_fp = None

    def log_scalar(self, name: str, value: float, step: int):
        """
        Log a single scalar.
        """
        if self.backend == "off":
            # still write JSONL if enabled (acts as a minimal audit trail)
            self._emit_jsonl(
                {
                    "type": "scalar",
                    "time": _utc_iso8601(),
                    self.cfg.step_key: int(step),
                    "name": name,
                    "value": _to_builtin(value),
                }
            )
            return

        # echo to stdout if requested (even when using TB/W&B)
        if self.cfg.echo_plain or self.backend == "plain":
            self._emit_plain(f"[{step:08d}] {name}: {_to_builtin(value):.6f}")

        # backend-specific
        if self.backend == "tensorboard" and self._tb is not None:
            self._tb.add_scalar(name, _to_builtin(value), int(step))
        elif self.backend == "wandb" and self._wb is not None:
            self._wb.log({name: _to_builtin(value), self.cfg.step_key: int(step)})

        # JSONL record
        self._emit_jsonl(
            {
                "type": "scalar",
                "time": _utc_iso8601(),
                self.cfg.step_key: int(step),
                "name": name,
                "value": _to_builtin(value),
            }
        )

    def log_dict(self, prefix: str, scalars: dict[str, float], step: int):
        """
        Log a flat dict of scalars, optionally under a prefix.
        For TB: each item is logged as its own scalar.
        For W&B: logged as a single dict payload.
        For JSONL: a single JSON object is written.
        """
        if not scalars:
            return

        flat = {f"{prefix}{k}": _to_builtin(v) for k, v in scalars.items()}

        if self.backend != "off" and (self.cfg.echo_plain or self.backend == "plain"):
            msg = " ".join([f"{k}={_to_builtin(v):.6f}" for k, v in flat.items()])
            self._emit_plain(
                f"[{step:08d}] {msg}" if prefix == "" else f"[{step:08d}] {prefix} {msg}"
            )

        # backend-specific
        if self.backend == "tensorboard" and self._tb is not None:
            for k, v in flat.items():
                self._tb.add_scalar(k, v, int(step))
        elif self.backend == "wandb" and self._wb is not None:
            payload = dict(flat)
            payload[self.cfg.step_key] = int(step)
            self._wb.log(payload)

        # JSONL record (single line)
        self._emit_jsonl(
            {
                "type": "metrics",
                "time": _utc_iso8601(),
                self.cfg.step_key: int(step),
                "prefix": prefix,
                "metrics": flat,  # already prefixed keys
            }
        )

from __future__ import annotations

from typing import Any

# IMPORTANT:
# - __all__ must contain *strings only* (never bare names).
# - Do not eagerly import reporting here; keep imports lazy to avoid circular-import crashes.

__all__ = [
    "read_eval_index",
    "load_eval_run",
    "select_key_metrics",
    "build_markdown_report",
    "write_markdown_report",
    "build_latex_table",
    "write_latex_table",
    "write_csv_wide",
    "write_csv_long",
    "build_report_template",
    "write_report_template",
]


def _load_reporting_module():
    """
    Support both historical filenames:
      - acpl/eval/reporting.py  (current / preferred)
      - acpl/eval/reports.py    (if you accidentally named it this)
    """
    try:
        from . import reporting as _m
        return _m
    except Exception:
        from . import reports as _m  # type: ignore
        return _m


def __getattr__(name: str) -> Any:
    if name in __all__:
        _m = _load_reporting_module()
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
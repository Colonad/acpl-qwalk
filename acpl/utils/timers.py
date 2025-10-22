# acpl/utils/timers.py
# ACPL Phase B8 — Utils: CUDA-aware timers, context managers, and a tiny profiler.
from __future__ import annotations

import atexit
from collections.abc import Callable
from dataclasses import dataclass
import logging
import math
import os
import sys
import threading
import time
from typing import Any

# ---------------------------------------------------------------------------
# Optional torch (for CUDA sync + NVTX). Never hard-require it.
# ---------------------------------------------------------------------------
try:
    import torch as _torch  # type: ignore

    _HAS_TORCH = True
except Exception:
    _torch = None  # type: ignore
    _HAS_TORCH = False

_LOG = logging.getLogger("acpl.timers")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ns() -> int:
    """Monotonic wall time in nanoseconds (high precision, immune to system clock jumps)."""
    return time.perf_counter_ns()


def _sync_cuda_if_needed(force: bool | None = None) -> None:
    """
    Synchronize CUDA if requested and available.
    This ensures GPU kernels finish before reading the timer (deterministic timing).
    """
    want = force
    if want is None:
        want = os.getenv("ACPL_TIMER_CUDA_SYNC", "1") not in ("0", "false", "False", "")
    if want and _HAS_TORCH:
        try:
            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
        except Exception:
            # Swallow to avoid timing failures when CUDA context isn't ready.
            pass


class _NvtxRange:
    """Small NVTX helper usable as a context manager (no-op if unsupported)."""

    def __init__(self, msg: str, enabled: bool) -> None:
        self.msg = msg
        self.enabled = enabled and _HAS_TORCH and hasattr(_torch.cuda, "nvtx")

    def __enter__(self):
        if self.enabled:
            try:
                _torch.cuda.nvtx.range_push(self.msg)  # type: ignore[attr-defined]
            except Exception:
                self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            try:
                _torch.cuda.nvtx.range_pop()  # type: ignore[attr-defined]
            except Exception:
                pass


def _want_nvtx() -> bool:
    return os.getenv("ACPL_TIMER_NVTX", "0") in ("1", "true", "True")


def format_seconds(sec: float, *, precision: int = 3) -> str:
    """Human-friendly duration formatter: ns/µs/ms/s/min/h as appropriate."""
    if math.isnan(sec) or math.isinf(sec):
        return str(sec)
    n = sec
    if n < 1e-6:
        return f"{n*1e9:.{precision}f} ns"
    if n < 1e-3:
        return f"{n*1e6:.{precision}f} µs"
    if n < 1.0:
        return f"{n*1e3:.{precision}f} ms"
    if n < 60.0:
        return f"{n:.{precision}f} s"
    if n < 3600.0:
        m, s = divmod(n, 60.0)
        return f"{int(m)}m {s:.{precision}f}s"
    h, r = divmod(n, 3600.0)
    m, s = divmod(r, 60.0)
    return f"{int(h)}h {int(m)}m {s:.{precision}f}s"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TimerStats:
    """Aggregated statistics for a named timer."""

    total_s: float = 0.0
    count: int = 0
    last_s: float = float("nan")
    min_s: float = float("inf")
    max_s: float = 0.0
    ema_s: float = float("nan")  # exponential moving average (time-decayed)
    _ema_alpha: float = 0.1

    def update(self, duration_s: float) -> None:
        self.total_s += duration_s
        self.count += 1
        self.last_s = duration_s
        self.min_s = min(self.min_s, duration_s)
        self.max_s = max(self.max_s, duration_s)
        if math.isnan(self.ema_s):
            self.ema_s = duration_s
        else:
            self.ema_s = (1 - self._ema_alpha) * self.ema_s + self._ema_alpha * duration_s

    @property
    def mean_s(self) -> float:
        return self.total_s / self.count if self.count else float("nan")

    def as_dict(self) -> dict[str, float]:
        return dict(
            total_s=self.total_s,
            count=float(self.count),
            last_s=self.last_s,
            mean_s=self.mean_s,
            ema_s=self.ema_s,
            min_s=(self.min_s if self.min_s < float("inf") else float("nan")),
            max_s=self.max_s,
        )


@dataclass
class _ActiveSpan:
    name: str
    t0_ns: int
    sync_cuda: bool
    nvtx: _NvtxRange | None = None


# Thread-local stack to support nested timers safely across dataloader workers etc.
_TLS = threading.local()


def _tls_stack() -> list[_ActiveSpan]:
    if not hasattr(_TLS, "stack"):
        _TLS.stack = []  # type: ignore[attr-defined]
    return _TLS.stack  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class Profiler:
    """
    A tiny, thread-safe profiler that aggregates named durations.
    Use it via `with record("encode"):` or `@timeit("step")`.
    """

    def __init__(self, *, ema_alpha: float = 0.1) -> None:
        self._stats: dict[str, TimerStats] = {}
        self._lock = threading.Lock()
        self._ema_alpha = float(ema_alpha)

    def _get(self, name: str) -> TimerStats:
        st = self._stats.get(name)
        if st is None:
            st = TimerStats(_ema_alpha=self._ema_alpha)
            self._stats[name] = st
        return st

    def add(self, name: str, seconds: float) -> None:
        with self._lock:
            self._get(name).update(seconds)

    def reset(self, name: str | None = None) -> None:
        with self._lock:
            if name is None:
                self._stats.clear()
            else:
                self._stats.pop(name, None)

    # ---- Reporting ---------------------------------------------------------

    def summary(
        self, *, sort_by: str = "total_s", descending: bool = True
    ) -> dict[str, dict[str, float]]:
        with self._lock:
            items = {k: v.as_dict() for k, v in self._stats.items()}
        if sort_by:
            items = dict(
                sorted(items.items(), key=lambda kv: kv[1].get(sort_by, 0.0), reverse=descending)
            )
        return items

    def pretty(self, *, width: int = 80) -> str:
        data = self.summary()
        if not data:
            return "<Profiler: empty>"
        name_w = max(len("name"), *(len(k) for k in data.keys()))

        def fmt(x: float) -> str:
            return format_seconds(x, precision=3) if math.isfinite(x) else "NaN"

        lines = []
        header = f"{'name':<{name_w}} | count | last     | mean     | ema      | min      | max      | total"
        lines.append(header)
        lines.append("-" * min(len(header), width))
        for k, v in data.items():
            row = (
                f"{k:<{name_w}} | "
                f"{int(v['count']):>5} | "
                f"{fmt(v['last_s']):>8} | "
                f"{fmt(v['mean_s']):>8} | "
                f"{fmt(v['ema_s']):>8} | "
                f"{fmt(v['min_s']):>8} | "
                f"{fmt(v['max_s']):>8} | "
                f"{fmt(v['total_s']):>8}"
            )
            lines.append(row)
        return "\n".join(lines)

    # ---- Writers -----------------------------------------------------------

    def write_scalars(
        self, writer: Any, *, prefix: str = "timers/", step: int | None = None
    ) -> None:
        """
        Export stats to a TB/W&B-style writer.
        The writer only needs `add_scalar(tag, value, step=None)` or `log({tag: value}, step=step)`.
        """
        data = self.summary()
        for name, s in data.items():
            # Prefer add_scalar, else fallback to log()
            tag = f"{prefix}{name}"
            if hasattr(writer, "add_scalar"):
                writer.add_scalar(f"{tag}/last_s", s["last_s"], step)
                writer.add_scalar(f"{tag}/mean_s", s["mean_s"], step)
                writer.add_scalar(f"{tag}/ema_s", s["ema_s"], step)
                writer.add_scalar(f"{tag}/min_s", s["min_s"], step)
                writer.add_scalar(f"{tag}/max_s", s["max_s"], step)
                writer.add_scalar(f"{tag}/total_s", s["total_s"], step)
                writer.add_scalar(f"{tag}/count", s["count"], step)
            elif hasattr(writer, "log"):
                writer.log(
                    {
                        f"{tag}/last_s": s["last_s"],
                        f"{tag}/mean_s": s["mean_s"],
                        f"{tag}/ema_s": s["ema_s"],
                        f"{tag}/min_s": s["min_s"],
                        f"{tag}/max_s": s["max_s"],
                        f"{tag}/total_s": s["total_s"],
                        f"{tag}/count": s["count"],
                    },
                    step=step,
                )


# Global profiler used by module-level helpers
_GLOBAL_PROFILER = Profiler()


def global_profiler() -> Profiler:
    return _GLOBAL_PROFILER


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


class Timer:
    """
    Basic high-precision timer with optional CUDA synchronization and NVTX ranges.
    Usable as a context manager or manually via start()/stop().
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        sync_cuda: bool | None = None,
        to_profiler: Profiler | None = None,
        use_nvtx: bool | None = None,
        logger: logging.Logger | None = None,
        log_on_exit: bool = False,
        log_level: int = logging.DEBUG,
    ) -> None:
        self.name = name or "<unnamed>"
        self.sync_cuda = sync_cuda
        self.to_profiler = to_profiler or _GLOBAL_PROFILER
        self.use_nvtx = _want_nvtx() if use_nvtx is None else bool(use_nvtx)
        self.logger = logger or _LOG
        self.log_on_exit = bool(log_on_exit)
        self.log_level = int(log_level)
        self._running = False
        self._t0_ns = 0
        self._nvtx = None  # type: ignore

    def start(self) -> Timer:
        if self._running:
            return self
        _sync_cuda_if_needed(self.sync_cuda)
        self._nvtx = _NvtxRange(self.name, self.use_nvtx).__enter__() if self.use_nvtx else None
        self._t0_ns = _now_ns()
        self._running = True
        _tls_stack().append(_ActiveSpan(self.name, self._t0_ns, bool(self.sync_cuda), self._nvtx))
        return self

    def stop(self) -> float:
        if not self._running:
            return 0.0
        _sync_cuda_if_needed(self.sync_cuda)
        t1_ns = _now_ns()
        duration_s = (t1_ns - self._t0_ns) * 1e-9
        self._running = False
        if self._nvtx is not None:
            try:
                self._nvtx.__exit__(None, None, None)
            except Exception:
                pass
            self._nvtx = None
        # Pop from thread-local stack (best-effort)
        st = _tls_stack()
        if st and st[-1].t0_ns == self._t0_ns:
            st.pop()
        # Aggregate
        if self.to_profiler is not None:
            self.to_profiler.add(self.name, duration_s)
        if self.log_on_exit:
            self.logger.log(self.log_level, "Timer[%s]: %s", self.name, format_seconds(duration_s))
        return duration_s

    # Context manager API
    def __enter__(self) -> Timer:
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        # Do not suppress exceptions
        return False


# Convenience: context alias
def record(
    name: str,
    *,
    sync_cuda: bool | None = None,
    profiler: Profiler | None = None,
    nvtx: bool | None = None,
    logger: logging.Logger | None = None,
    log: bool = False,
    log_level: int = logging.DEBUG,
) -> Timer:
    """
    Context manager to time a named block and aggregate into the (global) profiler.

    Example:
        with record("simulate", sync_cuda=True):
            psi = sim.step(psi, coins)
    """
    return Timer(
        name,
        sync_cuda=sync_cuda,
        to_profiler=profiler or _GLOBAL_PROFILER,
        use_nvtx=nvtx,
        logger=logger,
        log_on_exit=log,
        log_level=log_level,
    )


# Decorator for functions/methods
def timeit(
    name: str | None = None,
    *,
    sync_cuda: bool | None = None,
    profiler: Profiler | None = None,
    nvtx: bool | None = None,
    logger: logging.Logger | None = None,
    log: bool = False,
    log_level: int = logging.DEBUG,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorate a function to time its execution and push into the profiler.

    Usage:
        @timeit("train.step", sync_cuda=True)
        def train_step(...): ...
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        nm = name or f"{fn.__module__}.{fn.__qualname__}"

        def wrapped(*args, **kwargs):
            with record(
                nm,
                sync_cuda=sync_cuda,
                profiler=profiler,
                nvtx=nvtx,
                logger=logger,
                log=log,
                log_level=log_level,
            ):
                return fn(*args, **kwargs)

        # Preserve minimal metadata
        wrapped.__name__ = getattr(fn, "__name__", "wrapped")  # type: ignore[attr-defined]
        wrapped.__doc__ = fn.__doc__
        wrapped.__qualname__ = getattr(fn, "__qualname__", wrapped.__name__)
        return wrapped

    return deco


# ---------------------------------------------------------------------------
# High-level helpers tailored to ACPL’s pipeline
# ---------------------------------------------------------------------------


def measure_throughput(num_items: int, seconds: float) -> float:
    """Compute items/sec with safe edge case handling."""
    if seconds <= 0.0 or not math.isfinite(seconds):
        return float("nan")
    return num_items / seconds


def log_step_summary(
    step: int,
    *,
    extra_scalars: dict[str, float] | None = None,
    writer: Any | None = None,
    logger: logging.Logger | None = None,
    profiler: Profiler | None = None,
    prefix: str = "timers/",
) -> None:
    """
    Emit a one-line summary to logging, and (optionally) write scalars
    to a TensorBoard-like writer (has add_scalar() or log()).
    """
    prof = profiler or _GLOBAL_PROFILER
    lg = logger or _LOG
    txt = prof.pretty()
    lg.info("Step %d timing summary:\n%s", step, txt)
    if writer is not None:
        prof.write_scalars(writer, prefix=prefix, step=step)
        if extra_scalars:
            # Write any additional metrics the caller wants to attach at this step.
            if hasattr(writer, "add_scalar"):
                for k, v in extra_scalars.items():
                    writer.add_scalar(k, v, step)
            elif hasattr(writer, "log"):
                writer.log(extra_scalars, step=step)


# ---------------------------------------------------------------------------
# Safety & diagnostics
# ---------------------------------------------------------------------------


def push_manual(name: str, *, sync_cuda: bool | None = None, nvtx: bool | None = None) -> None:
    """
    Manually push a named span (start). Use with pop_manual().
    This mirrors record(...).__enter__ but without profiler aggregation until pop.
    """
    _sync_cuda_if_needed(sync_cuda)
    rng = _NvtxRange(name, _want_nvtx() if nvtx is None else bool(nvtx))
    rng.__enter__()
    _tls_stack().append(_ActiveSpan(name, _now_ns(), bool(sync_cuda), rng))


def pop_manual() -> float:
    """
    Manually pop the most recent named span (stop) and return duration in seconds.
    Does NOT aggregate to the profiler (use Profiler.add if needed).
    """
    if not _tls_stack():
        return 0.0
    span = _tls_stack().pop()
    _sync_cuda_if_needed(span.sync_cuda)
    t1 = _now_ns()
    if span.nvtx is not None:
        try:
            span.nvtx.__exit__(None, None, None)
        except Exception:
            pass
    return (t1 - span.t0_ns) * 1e-9


def current_stack() -> tuple[tuple[str, float], ...]:
    """
    Inspect the current thread’s active timer stack as tuples (name, elapsed_s).
    Useful for debugging stuck regions.
    """
    out = []
    now = _now_ns()
    for s in _tls_stack():
        out.append((s.name, (now - s.t0_ns) * 1e-9))
    return tuple(out)


# ---------------------------------------------------------------------------
# Auto-dump at exit (optional)
# ---------------------------------------------------------------------------


def _auto_dump() -> None:
    if os.getenv("ACPL_TIMER_AUTODUMP", "0") in ("1", "true", "True"):
        _LOG.info("Final timing summary (auto-dump on exit):\n%s", _GLOBAL_PROFILER.pretty())


atexit.register(_auto_dump)

# ---------------------------------------------------------------------------
# Quick self test
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    p = Profiler()
    with record("sleep.10ms", profiler=p, nvtx=False):
        time.sleep(0.010)
    with record("sleep.20ms", profiler=p, nvtx=False):
        time.sleep(0.020)
    # Nested + CUDA-optional (no-op if CUDA absent)
    with record("outer", profiler=p, nvtx=False):
        with record("inner", profiler=p, nvtx=False, sync_cuda=True):
            time.sleep(0.005)
    s = p.summary()
    ok = all(k in s for k in ("sleep.10ms", "sleep.20ms", "outer", "inner"))
    return ok


if __name__ == "__main__":  # pragma: no cover
    _LOG.setLevel(logging.DEBUG)
    ok = _self_test()
    print("timers.py self-test:", "OK" if ok else "FAILED")

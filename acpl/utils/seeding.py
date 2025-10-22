# acpl/utils/seeding.py
# Copyright (c) ACPL Project.
# Phase B8: Deterministic seeding across Python / NumPy / Torch (+ CUDA/cuDNN)
# Robust, novelty-first utilities aligned with the ACPL theoretical report (Phase B).
from __future__ import annotations

from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
import logging
import os
import platform
import random
import sys
import time
from typing import Any, Union
import uuid

try:
    import numpy as _np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False
    _np = None  # type: ignore

try:
    import torch as _torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    _torch = None  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG = logging.getLogger("acpl.seeding")
if not _LOG.handlers:
    # Keep quiet by default; integrates with project-wide logging once configured.
    _handler = logging.StreamHandler(stream=sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    _LOG.addHandler(_handler)
    _LOG.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Types & constants
# ---------------------------------------------------------------------------

SeedLike = Union[int, str, bytes, "numpy.random.SeedSequence"]  # noqa: F821
_INT64_MASK = (1 << 64) - 1

# PyTorch deterministic cublas workspace modes; must be set *before* CUDA init.
# See: https://pytorch.org/docs/stable/notes/randomness.html
_CUBLAS_WORKSPACE_CHOICES = (":16:8", ":4096:8")

# ---------------------------------------------------------------------------
# Hashing helpers (stable, portable)
# ---------------------------------------------------------------------------


def _blake2b_64(data: bytes, person: bytes | None = None) -> int:
    """Hash |data| (and optional personalization) to an unsigned 64-bit int."""
    import hashlib

    h = hashlib.blake2b(data, digest_size=8, person=person or b"acpl.seed")
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def canonicalize_seed(seed: SeedLike, *, context: str | None = None) -> int:
    """
    Convert heterogeneous seed inputs to a canonical unsigned 64-bit integer.

    Accepts:
      - int (will be masked to 64-bit, negative allowed)
      - str or bytes
      - numpy.random.SeedSequence

    The mapping is **stable** and **deterministic**.
    """
    person = (context or "root").encode("utf8")

    if isinstance(seed, int):
        return seed & _INT64_MASK

    if isinstance(seed, bytes):
        return _blake2b_64(seed, person=person)

    if isinstance(seed, str):
        return _blake2b_64(seed.encode("utf8"), person=person)

    if _HAS_NUMPY:
        import numpy as np  # local import for type hints

        if isinstance(seed, np.random.SeedSequence):
            # Collapse the 128-bit entropy of SeedSequence to 64-bit deterministically.
            return _blake2b_64(
                bytes(
                    seed.entropy.tobytes()
                    if hasattr(seed.entropy, "tobytes")
                    else int(seed.entropy).to_bytes(16, "little", signed=False)
                ),
                person=person,
            )

    raise TypeError(f"Unsupported seed type: {type(seed)}")


def derive_seed(parent: SeedLike, *names: str | int | bytes) -> int:
    """
    Derive a child 64-bit seed from a parent seed and a variable-length path.

    Example:
        base = canonicalize_seed("experiment-42")
        run_seed = derive_seed(base, "run", 0)
        episode_seed = derive_seed(base, "episode", 123456789012345)
        worker_seed = derive_seed(base, "loader", "worker", 7)
    """
    base64 = canonicalize_seed(parent, context="derive.base")
    data = bytearray()
    data.extend(base64.to_bytes(8, "little", signed=False))
    for n in names:
        if isinstance(n, int):
            # use variable-length LE encoding
            v = n & ((1 << 128) - 1)  # allow very large ints but cut to 128b
            nb = 16 if v > _INT64_MASK else 8
            data.extend(v.to_bytes(nb, "little", signed=False))
        elif isinstance(n, bytes):
            data.extend(n)
        else:
            data.extend(str(n).encode("utf8"))
        data.append(0x1F)  # delimiter
    return _blake2b_64(bytes(data), person=b"acpl.seed.derive")


# ---------------------------------------------------------------------------
# Environment guards for deterministic CUDA/cuDNN/cublas
# ---------------------------------------------------------------------------


def _cuda_is_initialized() -> bool:
    if not _HAS_TORCH:
        return False
    # torch.cuda.is_initialized exists in recent versions; fall back heuristics.
    if hasattr(_torch.cuda, "is_initialized"):
        try:
            return bool(_torch.cuda.is_initialized())
        except Exception:
            pass
    # Heuristic: if any device queries succeed, context likely initialized.
    try:
        return _torch.cuda.current_device() is not None  # type: ignore[arg-type]
    except Exception:
        return False


def _set_cublas_workspace_env(request: str | None) -> None:
    """
    Set CUBLAS_WORKSPACE_CONFIG for deterministic matmul kernels.

    Must be set before CUDA context initialization; warn otherwise.
    """
    if request is None:
        # Respect environment if already set; otherwise pick a default :4096:8.
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CHOICES[1]
        return

    if request not in _CUBLAS_WORKSPACE_CHOICES:
        raise ValueError(
            f"Invalid CUBLAS_WORKSPACE_CONFIG '{request}'. "
            f"Choose one of {list(_CUBLAS_WORKSPACE_CHOICES)}."
        )

    already = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if already == request:
        return
    if _cuda_is_initialized():
        _LOG.warning(
            "CUDA context already initialized; changing CUBLAS_WORKSPACE_CONFIG "
            "may not take effect this run. (current=%r, new=%r)",
            already,
            request,
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = request


# ---------------------------------------------------------------------------
# Snapshot & state structs
# ---------------------------------------------------------------------------


@dataclass
class RNGSnapshot:
    """Snapshot of Python/NumPy/Torch RNG states so you can restore later."""

    py_state: tuple[Any, ...]
    np_state: tuple[Any, ...] | None
    torch_cpu_state: bytes | None
    torch_cuda_state: list[bytes] | None
    torch_deterministic: bool | None
    torch_cudnn_benchmark: bool | None
    torch_cudnn_deterministic: bool | None
    cublas_workspace: str | None


def snapshot_rng() -> RNGSnapshot:
    """Capture current RNG states and determinism flags."""
    py_state = random.getstate()
    np_state = _np.random.get_state() if _HAS_NUMPY else None  # type: ignore
    tc_state = None
    tcu_state = None
    t_det = None
    t_bench = None
    t_cdet = None
    cublas_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    if _HAS_TORCH:
        try:
            tc_state = _torch.get_rng_state().clone().numpy().tobytes()  # bytes ok for portability
        except Exception:
            tc_state = None
        try:
            if _torch.cuda.is_available():
                tcu_state = []
                # Each device
                for dev_id in range(_torch.cuda.device_count()):
                    gen = _torch.cuda.get_rng_state(dev_id)
                    tcu_state.append(gen.clone().numpy().tobytes())
        except Exception:
            tcu_state = None

        # Flags
        try:
            t_det = bool(_torch.are_deterministic_algorithms_enabled())  # new API
        except Exception:
            try:
                t_det = bool(_torch.use_deterministic_algorithms)  # type: ignore
            except Exception:
                t_det = None
        try:
            t_bench = bool(_torch.backends.cudnn.benchmark)  # type: ignore
            t_cdet = bool(_torch.backends.cudnn.deterministic)  # type: ignore
        except Exception:
            t_bench = t_cdet = None

    return RNGSnapshot(
        py_state=py_state,
        np_state=np_state,
        torch_cpu_state=tc_state,
        torch_cuda_state=tcu_state,
        torch_deterministic=t_det,
        torch_cudnn_benchmark=t_bench,
        torch_cudnn_deterministic=t_cdet,
        cublas_workspace=cublas_env,
    )


def restore_rng(s: RNGSnapshot) -> None:
    """Restore RNG states and determinism flags captured by snapshot_rng()."""
    random.setstate(s.py_state)
    if _HAS_NUMPY and s.np_state is not None:
        _np.random.set_state(s.np_state)  # type: ignore

    if _HAS_TORCH:
        try:
            if s.torch_cpu_state is not None:
                st = _torch.frombuffer(bytearray(s.torch_cpu_state), dtype=_torch.uint8)
                _torch.set_rng_state(st)
        except Exception:
            _LOG.warning("Failed to restore Torch CPU RNG state.", exc_info=True)
        try:
            if _torch.cuda.is_available() and s.torch_cuda_state is not None:
                for dev_id, buf in enumerate(s.torch_cuda_state):
                    st = _torch.frombuffer(bytearray(buf), dtype=_torch.uint8)
                    _torch.cuda.set_rng_state(st, device=dev_id)
        except Exception:
            _LOG.warning("Failed to restore Torch CUDA RNG state(s).", exc_info=True)

        # Determinism flags
        try:
            if s.torch_deterministic is not None:
                _torch.use_deterministic_algorithms(s.torch_deterministic)  # type: ignore
        except Exception:
            pass
        try:
            if s.torch_cudnn_benchmark is not None:
                _torch.backends.cudnn.benchmark = bool(s.torch_cudnn_benchmark)  # type: ignore
            if s.torch_cudnn_deterministic is not None:
                _torch.backends.cudnn.deterministic = bool(s.torch_cudnn_deterministic)  # type: ignore
        except Exception:
            pass

    # Environment
    if s.cublas_workspace is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = s.cublas_workspace


# ---------------------------------------------------------------------------
# Public: Set global deterministic seeds & CUDA/cuDNN flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedState:
    seed64: int
    numpy_seed: int | None
    torch_seed: int | None
    py_hash_seed: str | None
    cublas_workspace: str
    deterministic: bool


def seed_everything(
    seed: SeedLike,
    *,
    deterministic: bool = True,
    set_python_hash_seed: bool = True,
    cublas_workspace: str | None = None,
    disable_tf32: bool = True,
) -> SeedState:
    """
    Set global RNG state for Python, NumPy, and PyTorch (CPU/CUDA), and
    configure deterministic kernels (cuDNN/cublas) where applicable.

    This function is *idempotent* for the same seed and safe to call in rank0
    and worker processes (DataLoader workers should still use worker_init_fn).

    Args:
        seed: Root seed (int/str/bytes/SeedSequence). Folded to 64-bit.
        deterministic: If True, enforce deterministic algos/flags in PyTorch.
        set_python_hash_seed: If True, export PYTHONHASHSEED for hash-stable dict/set.
        cublas_workspace: One of {":16:8", ":4096:8"} or None to keep/default.
        disable_tf32: If True, disable TF32 on matmul/cuDNN for strict bitwise runs.

    Returns:
        SeedState describing the applied configuration.
    """
    s64 = canonicalize_seed(seed, context="seed_everything")
    # --- Python
    if set_python_hash_seed:
        # If already set and different, warn (takes effect only at interpreter start).
        phs_prev = os.environ.get("PYTHONHASHSEED")
        phs_new = str(s64 & 0xFFFFFFFF)
        if phs_prev and phs_prev != phs_new:
            _LOG.warning(
                "Overriding PYTHONHASHSEED from %s -> %s (effective only for new processes).",
                phs_prev,
                phs_new,
            )
        os.environ["PYTHONHASHSEED"] = phs_new
    else:
        phs_new = os.environ.get("PYTHONHASHSEED")

    random.seed(s64 & _INT64_MASK)

    # --- NumPy
    np_seed = None
    if _HAS_NUMPY:
        np_seed = (
            s64 & 0xFFFFFFFF
        )  # NumPy accepts 32-bit for RandomState; Generator can accept larger via SeedSequence
        _np.random.seed(np_seed)  # type: ignore

    # --- Torch
    torch_seed = None
    if _HAS_TORCH:
        torch_seed = s64 & _INT64_MASK
        _torch.manual_seed(torch_seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(torch_seed)

        if deterministic:
            try:
                _torch.use_deterministic_algorithms(True)  # type: ignore
            except Exception:
                pass
            try:
                _torch.backends.cudnn.deterministic = True  # type: ignore
                _torch.backends.cudnn.benchmark = False  # type: ignore
            except Exception:
                pass
            # cuBLAS workspace (must be set pre-initialization ideally)
            _set_cublas_workspace_env(cublas_workspace)
        else:
            try:
                _torch.use_deterministic_algorithms(False)  # type: ignore
            except Exception:
                pass

        # Disable TF32 for strict reproducibility if requested
        if disable_tf32:
            try:
                _torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore
            except Exception:
                pass
            try:
                _torch.backends.cudnn.allow_tf32 = False  # type: ignore
            except Exception:
                pass

    state = SeedState(
        seed64=s64,
        numpy_seed=np_seed,
        torch_seed=torch_seed,
        py_hash_seed=phs_new,
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        deterministic=bool(deterministic),
    )

    _LOG.info(
        "Seeding complete | seed64=%d | numpy=%s | torch=%s | det=%s | cublas=%r",
        state.seed64,
        state.numpy_seed,
        state.torch_seed,
        state.deterministic,
        state.cublas_workspace,
    )
    return state


# ---------------------------------------------------------------------------
# Generators & contexts
# ---------------------------------------------------------------------------


def torch_generator(device: int | str | None = None, seed: SeedLike | None = None):
    """
    Create a torch.Generator (CPU or CUDA device) with an optional seed.

    If seed is None, a deterministic child seed is derived from time+pid+uuid,
    hashed under the 'torch.generator' personalization to keep independence.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available.")
    gen = _torch.Generator(
        device="cuda" if (device is not None and str(device) != "cpu") else "cpu"
    )
    if seed is None:
        # Process-local entropy; still deterministic since hashed to 64-bit.
        entropy = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4()}"
        s = canonicalize_seed(entropy, context="torch.generator")
    else:
        s = canonicalize_seed(seed, context="torch.generator")
    gen.manual_seed(s & _INT64_MASK)
    return gen


def numpy_generator(seed: SeedLike | None = None):
    """Return a NumPy Generator seeded deterministically from |seed| (or derived)."""
    if not _HAS_NUMPY:
        raise RuntimeError("NumPy not available.")
    if seed is None:
        entropy = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4()}"
        s = canonicalize_seed(entropy, context="numpy.generator")
    else:
        s = canonicalize_seed(seed, context="numpy.generator")
    # Use SeedSequence to propagate full 64-bit entropy into PCG64
    ss = _np.random.SeedSequence(s & _INT64_MASK)  # type: ignore
    return _np.random.Generator(_np.random.PCG64(ss))  # type: ignore


@contextlib.contextmanager
def temp_seed(seed: SeedLike, *, deterministic: bool = True) -> Iterator[None]:
    """
    Context manager: temporarily set global RNG seeds/flags, then restore snapshot.

    Example:
        with temp_seed(1234):
            # reproducible block
            ...
        # original RNG states restored
    """
    snap = snapshot_rng()
    try:
        seed_everything(seed, deterministic=deterministic)
        yield
    finally:
        restore_rng(snap)


# ---------------------------------------------------------------------------
# DataLoader worker init util
# ---------------------------------------------------------------------------


def dataloader_worker_init_fn(worker_id: int) -> None:
    """
    Robust worker init for torch.utils.data.DataLoader(worker_init_fn=...).

    A global "ACPL_WORKER_BASE_SEED" should be present in the parent
    (set by your training script). We derive per-worker, per-process seeds.
    """
    base = os.environ.get("ACPL_WORKER_BASE_SEED")
    if base is None:
        # Fallback to torch initial seed to keep determinism within run.
        if _HAS_TORCH:
            base_val = int(_torch.initial_seed() & _INT64_MASK)  # type: ignore
        else:
            base_val = int(time.time_ns() & _INT64_MASK)
        base = str(base_val)
        _LOG.warning("ACPL_WORKER_BASE_SEED not set; using initial_seed fallback=%s", base)

    # Derive disjoint seed per worker
    derived = derive_seed(base, "worker", worker_id, os.getpid())
    # Python & NumPy
    random.seed(derived & _INT64_MASK)
    if _HAS_NUMPY:
        _np.random.seed(derived & 0xFFFFFFFF)  # type: ignore

    # Torch CPU/CUDA generators (worker runs on CPU unless you pin/move tensors)
    if _HAS_TORCH:
        _torch.manual_seed(derived & _INT64_MASK)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(derived & _INT64_MASK)

    _LOG.debug("Initialized DataLoader worker %d with seed=%d", worker_id, derived)


# ---------------------------------------------------------------------------
# Episode/manifest helpers (ACPL-specific niceties)
# ---------------------------------------------------------------------------


def episode_seed(manifest_seed: SeedLike, episode_index: int) -> int:
    """
    Deterministically derive a 64-bit seed for an episode from a dataset manifest seed
    and a monotonically increasing episode index.
    """
    return derive_seed(manifest_seed, "episode", episode_index)


def sampler_seed(root: SeedLike, *, kind: str, name: str | int) -> int:
    """
    Derive a seed for a specific sampler (graph/task/noise/init) using a namespaced path.
    """
    return derive_seed(root, "sampler", kind, name)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def get_seed_report() -> dict[str, Any]:
    """
    Return a human-readable dict of the current seeding/determinism configuration.
    Useful to log at the start of training/eval for auditability.
    """
    rep: dict[str, Any] = dict(
        python_version=sys.version.split()[0],
        platform=dict(
            system=platform.system(),
            machine=platform.machine(),
            python_impl=platform.python_implementation(),
        ),
        pid=os.getpid(),
        python_hash_seed=os.environ.get("PYTHONHASHSEED", None),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG", None),
        numpy_available=_HAS_NUMPY,
        torch_available=_HAS_TORCH,
    )
    if _HAS_TORCH:
        rep.update(
            torch=dict(
                version=_torch.__version__,
                cuda_available=bool(_torch.cuda.is_available()),
                cudnn_available=bool(getattr(_torch.backends, "cudnn", None)),
                det_algos_enabled=(
                    bool(_torch.are_deterministic_algorithms_enabled())
                    if hasattr(_torch, "are_deterministic_algorithms_enabled")
                    else None
                ),
                cudnn_benchmark=bool(_torch.backends.cudnn.benchmark) if hasattr(_torch.backends, "cudnn") else None,  # type: ignore
                cudnn_deterministic=bool(_torch.backends.cudnn.deterministic) if hasattr(_torch.backends, "cudnn") else None,  # type: ignore
                tf32_matmul_allowed=(
                    bool(getattr(_torch.backends.cuda.matmul, "allow_tf32", None))
                    if hasattr(_torch.backends, "cuda")
                    else None
                ),
                tf32_cudnn_allowed=(
                    bool(getattr(_torch.backends.cudnn, "allow_tf32", None))
                    if hasattr(_torch.backends, "cudnn")
                    else None
                ),
            )
        )
    return rep


# ---------------------------------------------------------------------------
# Quick self-test (optional)
# ---------------------------------------------------------------------------


def _self_test(verbose: bool = False) -> bool:
    """
    Minimal sanity check to ensure seeds produce identical sequences across calls.
    Returns True on success.
    """
    with temp_seed(123456789):
        a_py = [random.getrandbits(32) for _ in range(5)]
        a_np = _np.random.integers(0, 2**31, size=5).tolist() if _HAS_NUMPY else None  # type: ignore
        a_tc = _torch.randint(0, 2**31, (5,), dtype=_torch.int64).tolist() if _HAS_TORCH else None

    with temp_seed(123456789):
        b_py = [random.getrandbits(32) for _ in range(5)]
        b_np = _np.random.integers(0, 2**31, size=5).tolist() if _HAS_NUMPY else None  # type: ignore
        b_tc = _torch.randint(0, 2**31, (5,), dtype=_torch.int64).tolist() if _HAS_TORCH else None

    ok = (
        (a_py == b_py)
        and (a_np == b_np if _HAS_NUMPY else True)
        and (a_tc == b_tc if _HAS_TORCH else True)
    )
    if verbose:
        _LOG.info("Self-test deterministic sequences equal: %s", ok)
    return ok


# If run as a script: quick check
if __name__ == "__main__":  # pragma: no cover
    _LOG.setLevel(logging.DEBUG)
    ok = _self_test(verbose=True)
    print("OK" if ok else "FAILED")

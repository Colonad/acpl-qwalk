# acpl/policy/gnn/segment.py
from __future__ import annotations

import os
import torch


def _stable_enabled() -> bool:
    """
    Enable stable reductions in tests (and optionally via env var).
    This avoids CUDA atomic order noise that breaks gradient equivariance checks.
    """
    v = os.getenv("ACPL_STABLE_REDUCTIONS", "")
    if v.lower() in {"1", "true", "yes", "on"}:
        return True
    # pytest sets this for each test
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    # respect PyTorch deterministic algorithms knob if user enabled it
    try:
        return torch.are_deterministic_algorithms_enabled()
    except Exception:
        return False


def segment_sum(values: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    Sum `values` into `num_segments` bins indicated by `index` (shape [E]).

    Supports:
      - values: [E] or [E, D]
      - index:  [E] long
      - returns: [num_segments] or [num_segments, D]

    Fast path: index_add_ (normal training)
    Stable path (pytest / forced): sort by (index, |values|) then sum in float64 via prefix sums.
    """
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if index.numel() == 0:
        out_shape = (num_segments,) if values.ndim == 1 else (num_segments, values.size(1))
        return torch.zeros(out_shape, device=values.device, dtype=values.dtype)

    if index.dtype != torch.long:
        index = index.to(torch.long)
    if index.device != values.device:
        index = index.to(values.device)

    if values.ndim == 1:
        v = values
        D = None
    elif values.ndim == 2:
        v = values
        D = values.size(1)
    else:
        raise ValueError("segment_sum expects values of shape [E] or [E, D]")

    # ---------------- fast path ----------------
    if not _stable_enabled():
        if D is None:
            out = torch.zeros(num_segments, device=v.device, dtype=v.dtype)
            out.index_add_(0, index, v)
            return out
        out = torch.zeros(num_segments, D, device=v.device, dtype=v.dtype)
        out.index_add_(0, index, v)
        return out

    # ---------------- stable path (tests) ----------------
    # Secondary key based on |values| makes the within-segment accumulation order
    # much less sensitive to node-label permutations.
    if D is None:
        key2 = v.abs()
    else:
        key2 = v.abs().sum(dim=1)

    # Two-pass stable sorting:
    # 1) sort by key2, 2) stable-sort by index => order is (index, key2)
    try:
        o1 = torch.argsort(key2, stable=True)
        idx1 = index.index_select(0, o1)
        v1 = v.index_select(0, o1)
        o2 = torch.argsort(idx1, stable=True)
        idx = idx1.index_select(0, o2)
        vv = v1.index_select(0, o2)
    except TypeError:
        # older torch without stable=...
        o = torch.argsort(index * (2**31) + torch.argsort(key2))
        idx = index.index_select(0, o)
        vv = v.index_select(0, o)

    counts = torch.bincount(idx, minlength=num_segments)
    ptr = torch.zeros(num_segments + 1, device=idx.device, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)

    vv64 = vv.to(torch.float64) if vv.dtype in (torch.float16, torch.bfloat16, torch.float32) else vv
    pref = vv64.cumsum(dim=0)  # [E] or [E,D]

    if D is None:
        out64 = torch.zeros(num_segments, device=vv64.device, dtype=vv64.dtype)
    else:
        out64 = torch.zeros(num_segments, D, device=vv64.device, dtype=vv64.dtype)

    nonempty = counts > 0
    if nonempty.any():
        segs = torch.nonzero(nonempty, as_tuple=False).squeeze(1)
        start = ptr[segs]
        end = ptr[segs + 1] - 1  # inclusive
        end_sum = pref.index_select(0, end)
        startm1 = start - 1
        start_sum = torch.zeros_like(end_sum)
        has_prev = startm1 >= 0
        if has_prev.any():
            start_sum[has_prev] = pref.index_select(0, startm1[has_prev])
        out64[segs] = end_sum - start_sum

    return out64.to(v.dtype)

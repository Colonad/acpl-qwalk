# acpl/sim/disorder.py
"""
acpl.sim.disorder
=================

Research-ready, reproducible *disorder / noise* utilities for DTQW evaluation.

This module is intentionally **stateless** and designed to be used from rollouts:
you pass in an episode-level disorder spec (usually `batch["disorder"]`) and a
graph-induced arc involution `rev` (flip-flop shift pairing), and you receive
deterministic samples that can be applied to:

1) **Shift-edge phase disorder** (arc phases) while preserving flip-flop
   involution constraints (so that S^2 = I still holds for paired arcs).

2) **Edge dropout / bond percolation** on arc pairs (symmetric keep masks
   so that both directions of an undirected edge are dropped/kept together).

3) **Coin dephasing / phase diffusion** implemented as a random diagonal
   unitary acting on the state. This preserves norm exactly (unitary channel).

Key design goals
----------------
- **Determinism** across machines and Python processes:
  we never use Python's built-in hash for seeding.
- **Explicit RNG stream separation** via tagged seed derivation (BLAKE2b).
- **Flip-flop safety**: pairing constraints are enforced when requested.
- **Device-robust**: works on CPU and CUDA even if some torch versions have
  limitations around device generators (falls back to CPU sampling + transfer).

You typically:
- parse/normalize a dict into a `DisorderSpec`,
- create a `DisorderRNG` from it,
- sample `ShiftDisorder` modifiers (phase/mask),
- apply `apply_state_dephase(...)` inside the time loop.

This module does *not* depend on PortMap/ShiftOp directly to avoid import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Final

import hashlib
import struct

import torch
from torch import Tensor

__all__ = [
    # Specs / parsing
    "EdgePhaseSpec",
    "EdgeDropoutSpec",
    "CoinDephaseSpec",
    "DisorderSpec",
    "ShiftDisorder",
    "parse_disorder_spec",
    # Deterministic seeding
    "u64",
    "blake2b_u64",
    "derive_seed_u64",
    "DisorderRNG",
    # Sampling
    "check_involution_perm",
    "pair_heads_mask",
    "sample_flipflop_edge_phases",
    "sample_flipflop_edge_keep_mask",
    # Channels / application
    "apply_state_dephase",
    # Small helpers
    "randn_device",
    "rand_device",

    "sample_shift_disorder",
    "format_disorder_meta",
    "EpisodeDisorder",
    "build_episode_disorder",
    "maybe_apply_coin_dephase",


]


# -----------------------------------------------------------------------------
#                             Deterministic seeding
# -----------------------------------------------------------------------------

_U64_MASK: Final[int] = 0xFFFFFFFFFFFFFFFF


def u64(x: int) -> int:
    """Force an integer into unsigned 64-bit range."""
    return int(x) & _U64_MASK


def blake2b_u64(data: bytes, *, person: bytes = b"acpl.disorder") -> int:
    """
    Stable 64-bit hash via BLAKE2b.

    - `person` is the BLAKE2 personalization string (domain separation).
    - Returns an unsigned 64-bit integer.
    """
    h = hashlib.blake2b(data, digest_size=8, person=person)
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def derive_seed_u64(*parts: Any, tag: str = "default") -> int:
    """
    Derive a deterministic 64-bit seed from arbitrary parts + a tag.

    Each part is encoded stably:
      - ints -> little-endian signed 64-bit
      - floats -> IEEE754 float64 bytes
      - str/bytes -> length-prefixed UTF-8/bytes
      - bool -> 0/1
      - otherwise -> `repr(obj)` as UTF-8 (best effort; avoid for long objects)

    The `tag` provides explicit stream separation (e.g. "edge_phase", "coin_dephase").
    """
    buf = bytearray()
    buf += struct.pack("<I", len(tag))
    buf += tag.encode("utf-8")

    for p in parts:
        if isinstance(p, bool):
            buf += b"b" + struct.pack("<Q", 1 if p else 0)
        elif isinstance(p, int):
            buf += b"i" + struct.pack("<q", int(p))
        elif isinstance(p, float):
            buf += b"f" + struct.pack("<d", float(p))
        elif isinstance(p, bytes):
            buf += b"y" + struct.pack("<I", len(p)) + p
        elif isinstance(p, str):
            s = p.encode("utf-8")
            buf += b"s" + struct.pack("<I", len(s)) + s
        else:
            s = repr(p).encode("utf-8")
            buf += b"r" + struct.pack("<I", len(s)) + s

    return u64(blake2b_u64(bytes(buf)))


def _try_make_generator(seed: int, device: torch.device) -> torch.Generator:
    """
    Create a torch.Generator on the given device if supported.
    Falls back to CPU generator if necessary.
    """
    seed = u64(seed)
    try:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g
    except Exception:
        g = torch.Generator()
        g.manual_seed(seed)
        # Stash target device for our helper sampling functions.
        try:
            setattr(g, "_acpl_target_device", str(device))
        except Exception:
            pass
        return g


def _generator_device(g: torch.Generator) -> torch.device:
    """
    Best-effort retrieval of generator device. Older torch may not expose `.device`.
    """
    try:
        return torch.device(g.device)  # type: ignore[attr-defined]
    except Exception:
        # If our fallback stashed a target device, use it for intent; otherwise CPU.
        dev = getattr(g, "_acpl_target_device", None)
        return torch.device(dev) if isinstance(dev, str) else torch.device("cpu")


def randn_device(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tensor:
    """
    torch.randn with generator, robust to generator/device mismatch.

    If generator is not on `device`, we sample on CPU deterministically and move.
    """
    gdev = _generator_device(generator)
    if gdev == device:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
    x = torch.randn(shape, device=torch.device("cpu"), dtype=dtype, generator=generator)
    return x.to(device=device)


def rand_device(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tensor:
    """
    torch.rand with generator, robust to generator/device mismatch.

    If generator is not on `device`, we sample on CPU deterministically and move.
    """
    gdev = _generator_device(generator)
    if gdev == device:
        return torch.rand(shape, device=device, dtype=dtype, generator=generator)
    x = torch.rand(shape, device=torch.device("cpu"), dtype=dtype, generator=generator)
    return x.to(device=device)


class DisorderRNG:
    """
    RNG manager that produces independent (but reproducible) generator streams.

    Typical usage:
        rng = DisorderRNG(spec, device=torch.device("cuda"))
        g_edge = rng.stream("edge_phase")
        g_deph = rng.stream("coin_dephase")
    """

    def __init__(self, spec: "DisorderSpec", *, device: torch.device):
        self.spec = spec
        self.device = device

    def stream(self, tag: str, *, extra: int = 0) -> torch.Generator:
        """
        Deterministic per-tag stream derived from (seed, episode_index, trial_id, tag, extra).
        `extra` allows you to derive substreams per time step, etc.
        """
        s = derive_seed_u64(
            int(self.spec.seed),
            int(self.spec.episode_index),
            int(self.spec.trial_id),
            int(extra),
            tag=tag,
        )
        return _try_make_generator(s, self.device)


# -----------------------------------------------------------------------------
#                                  Specs
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class EdgePhaseSpec:
    """
    Edge (arc) phase disorder for flip-flop shifts.

    If `flipflop_safe=True`, sampled phases satisfy:
        phase[a] * phase[rev[a]] == 1
    which preserves the involution property of flip-flop shift composition.
    """
    enabled: bool = False
    sigma: float = 0.0
    flipflop_safe: bool = True
    clamp_abs: float | None = None  # optional clamp on |phi| to avoid extreme phases


@dataclass(frozen=True)
class EdgeDropoutSpec:
    """
    Symmetric edge dropout for undirected graphs represented as paired arcs.

    With `flipflop_safe=True`, keep mask satisfies:
        keep[a] == keep[rev[a]] in {0,1}
    """
    enabled: bool = False
    p: float = 0.0
    flipflop_safe: bool = True


@dataclass(frozen=True)
class CoinDephaseSpec:
    """
    Coin dephasing / phase diffusion acting on the quantum state.

    Implemented as a random diagonal unitary:
        psi <- exp(i * eps) ⊙ psi,  eps ~ N(0, sigma^2)

    This is norm-preserving and cheap; it models phase noise without amplitude loss.
    """
    enabled: bool = False
    sigma: float = 0.0


@dataclass(frozen=True)
class DisorderSpec:
    """
    Episode-level disorder specification.

    The trio (seed, episode_index, trial_id) defines a unique disorder realization,
    while still permitting deterministic reproduction.
    """
    seed: int = 0
    episode_index: int = 0
    trial_id: int = 0

    edge_phase: EdgePhaseSpec = EdgePhaseSpec()
    edge_dropout: EdgeDropoutSpec = EdgeDropoutSpec()
    coin_dephase: CoinDephaseSpec = CoinDephaseSpec()


@dataclass(frozen=True)
class ShiftDisorder:
    """Shift modifiers sampled for one episode."""
    phase: Tensor | None  # (A,) complex64
    mask: Tensor | None   # (A,) float32 (0/1)

@dataclass(frozen=True)
class EpisodeDisorder:
    """
    Bundle of deterministic disorder objects for a single episode.

    - `shift` is intended to be sampled ONCE per episode and reused for all t.
    - `rng` is used to derive per-step streams (e.g., coin dephase with extra=t).
    """
    spec: DisorderSpec
    rng: DisorderRNG
    shift: ShiftDisorder

def _as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
    return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def parse_disorder_spec(d: Mapping[str, Any] | None) -> DisorderSpec:
    """
    Parse a nested dict (e.g. batch["disorder"]) into a normalized DisorderSpec.

    Expected (flexible) structure:
        {
          "seed": 0,
          "episode_index": 12,
          "trial_id": 3,
          "edge_phase": {"enabled": true, "sigma": 0.1, "flipflop_safe": true},
          "edge_dropout": {"enabled": true, "p": 0.2},
          "coin_dephase": {"enabled": true, "sigma": 0.05},
        }
    """
    if not d:
        return DisorderSpec()

    seed = _as_int(d.get("seed", 0), 0)
    episode_index = _as_int(d.get("episode_index", 0), 0)
    trial_id = _as_int(d.get("trial_id", 0), 0)

    ep = d.get("edge_phase", {}) or {}
    ed = d.get("edge_dropout", {}) or {}
    cd = d.get("coin_dephase", {}) or {}

    edge_phase = EdgePhaseSpec(
        enabled=_as_bool(ep.get("enabled", False), False),
        sigma=_as_float(ep.get("sigma", 0.0), 0.0),
        flipflop_safe=_as_bool(ep.get("flipflop_safe", True), True),
        clamp_abs=(None if ep.get("clamp_abs", None) is None else _as_float(ep.get("clamp_abs"), 0.0)),
    )

    edge_dropout = EdgeDropoutSpec(
        enabled=_as_bool(ed.get("enabled", False), False),
        p=_as_float(ed.get("p", 0.0), 0.0),
        flipflop_safe=_as_bool(ed.get("flipflop_safe", True), True),
    )

    coin_dephase = CoinDephaseSpec(
        enabled=_as_bool(cd.get("enabled", False), False),
        sigma=_as_float(cd.get("sigma", 0.0), 0.0),
    )

    # clamp to safe ranges
    if edge_dropout.p < 0.0:
        edge_dropout = EdgeDropoutSpec(enabled=edge_dropout.enabled, p=0.0, flipflop_safe=edge_dropout.flipflop_safe)
    if edge_dropout.p > 1.0:
        edge_dropout = EdgeDropoutSpec(enabled=edge_dropout.enabled, p=1.0, flipflop_safe=edge_dropout.flipflop_safe)

    if edge_phase.sigma < 0.0:
        edge_phase = EdgePhaseSpec(
            enabled=edge_phase.enabled, sigma=0.0, flipflop_safe=edge_phase.flipflop_safe, clamp_abs=edge_phase.clamp_abs
        )
    if coin_dephase.sigma < 0.0:
        coin_dephase = CoinDephaseSpec(enabled=coin_dephase.enabled, sigma=0.0)

    return DisorderSpec(
        seed=seed,
        episode_index=episode_index,
        trial_id=trial_id,
        edge_phase=edge_phase,
        edge_dropout=edge_dropout,
        coin_dephase=coin_dephase,
    )


# -----------------------------------------------------------------------------
#                              Flip-flop pairing helpers
# -----------------------------------------------------------------------------

def check_involution_perm(perm: Tensor, *, name: str = "perm") -> None:
    """
    Validate that `perm` is an involution over [0..A-1]:
        perm[perm[i]] == i for all i.

    This is the required structure for flip-flop shift pairing (rev arcs).
    """
    if not isinstance(perm, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(perm)}")
    if perm.ndim != 1:
        raise ValueError(f"{name} must be 1D (A,), got shape {tuple(perm.shape)}")
    if perm.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"{name} must be int tensor, got dtype={perm.dtype}")
    A = int(perm.numel())
    if A == 0:
        return
    if perm.min().item() < 0 or perm.max().item() >= A:
        raise ValueError(f"{name} must map into [0,{A-1}]")
    inv = perm[perm]
    if not torch.equal(inv, torch.arange(A, device=perm.device, dtype=perm.dtype)):
        # Provide a small debug hint
        bad = (inv != torch.arange(A, device=perm.device, dtype=perm.dtype)).nonzero(as_tuple=False).flatten()
        k = int(bad[0].item()) if bad.numel() > 0 else -1
        raise ValueError(
            f"{name} must be an involution (perm[perm[i]]==i). "
            f"First violation at i={k}, perm[i]={int(perm[k]) if k >= 0 else '??'}, perm[perm[i]]={int(inv[k]) if k >= 0 else '??'}."
        )


def pair_heads_mask(rev: Tensor) -> Tensor:
    """
    Return a boolean mask selecting exactly one representative per unordered arc-pair (a, rev[a]),
    excluding self-loops. For involution rev:
      - head if a < rev[a]
      - tail if a > rev[a]
      - self if a == rev[a]
    """
    if rev.ndim != 1:
        raise ValueError(f"rev must be (A,), got {tuple(rev.shape)}")
    A = int(rev.numel())
    idx = torch.arange(A, device=rev.device, dtype=rev.dtype)
    is_self = rev == idx
    head = (idx < rev) & (~is_self)
    return head


# -----------------------------------------------------------------------------
#                                  Sampling
# -----------------------------------------------------------------------------

def sample_flipflop_edge_phases(
    rev: Tensor,
    sigma: float,
    *,
    generator: torch.Generator,
    clamp_abs: float | None = None,
    validate: bool = False,
) -> Tensor:
    """
    Sample arc phases for flip-flop shift with involution constraint.

    Returns:
      phase: (A,) complex64 with unit magnitude, such that for non-self arcs:
             phase[a] = exp(i phi[a]), phase[rev[a]] = exp(-i phi[a])
             => phase[a]*phase[rev[a]] = 1

    Notes:
    - For self-loops (rev[a]==a), phase[a]=1.
    - If sigma<=0, returns all ones.
    """
    if validate:
        check_involution_perm(rev, name="rev")

    sigma = float(sigma)
    if sigma <= 0.0:
        return torch.ones(int(rev.numel()), device=rev.device, dtype=torch.complex64)

    A = int(rev.numel())
    device = rev.device

    idx = torch.arange(A, device=device, dtype=rev.dtype)
    is_self = rev == idx
    head = (idx < rev) & (~is_self)
    tail = (idx > rev) & (~is_self)

    phi = torch.zeros(A, device=device, dtype=torch.float32)

    n = int(head.sum().item())
    if n > 0:
        phi_head = randn_device((n,), device=device, dtype=torch.float32, generator=generator) * sigma
        if clamp_abs is not None and float(clamp_abs) > 0:
            c = float(clamp_abs)
            phi_head = phi_head.clamp(min=-c, max=c)
        phi[head] = phi_head

        # Mirror with sign flip on paired arcs
        phi[tail] = -phi[rev[tail]]

    # self-loops remain 0
    phase = torch.exp(torch.complex(torch.zeros_like(phi), phi)).to(torch.complex64)
    return phase


def sample_flipflop_edge_keep_mask(
    rev: Tensor,
    p_drop: float,
    *,
    generator: torch.Generator,
    validate: bool = False,
) -> Tensor:
    """
    Sample a symmetric (flip-flop safe) keep mask over arc pairs.

    Returns:
      keep: (A,) float32 in {0,1}, with keep[a]==keep[rev[a]] for paired arcs.
    """
    if validate:
        check_involution_perm(rev, name="rev")

    p_drop = float(p_drop)
    if p_drop <= 0.0:
        return torch.ones(int(rev.numel()), device=rev.device, dtype=torch.float32)

    if p_drop >= 1.0:
        # Drop everything except self-loops
        A = int(rev.numel())
        idx = torch.arange(A, device=rev.device, dtype=rev.dtype)
        is_self = rev == idx
        keep = torch.zeros(A, device=rev.device, dtype=torch.float32)
        keep[is_self] = 1.0
        return keep

    A = int(rev.numel())
    device = rev.device

    idx = torch.arange(A, device=device, dtype=rev.dtype)
    is_self = rev == idx
    head = (idx < rev) & (~is_self)
    tail = (idx > rev) & (~is_self)

    keep = torch.ones(A, device=device, dtype=torch.float32)

    n = int(head.sum().item())
    if n > 0:
        u = rand_device((n,), device=device, dtype=torch.float32, generator=generator)
        keep_head = (u >= p_drop).to(torch.float32)
        keep[head] = keep_head
        keep[tail] = keep[rev[tail]]

    # self-loops kept
    return keep


# -----------------------------------------------------------------------------
#                                 Application
# -----------------------------------------------------------------------------

def apply_state_dephase(
    psi: Tensor,
    sigma: float,
    *,
    generator: torch.Generator,
) -> Tensor:
    """
    Apply phase diffusion (dephasing) to the complex state.

    Implements the unitary diagonal channel:
        psi <- exp(i * eps) ⊙ psi,   eps ~ N(0, sigma^2)

    This preserves ||psi|| exactly (up to fp roundoff) and is inexpensive.

    Works for any psi shape; we sample eps with shape psi.shape.
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        return psi
    if not torch.is_complex(psi):
        raise TypeError(f"apply_state_dephase expects a complex tensor psi, got dtype={psi.dtype}")

    eps = randn_device(tuple(psi.shape), device=psi.device, dtype=torch.float32, generator=generator) * sigma
    rot = torch.exp(torch.complex(torch.zeros_like(eps), eps)).to(psi.dtype)
    return psi * rot


# -----------------------------------------------------------------------------
#                         Convenience: one-shot sampling
# -----------------------------------------------------------------------------

def sample_shift_disorder(
    rev: Tensor,
    spec: DisorderSpec,
    *,
    rng: DisorderRNG,
    extra: int = 0,
    validate: bool = False,
) -> ShiftDisorder:

    """
    Convenience wrapper to sample phase/mask for an episode, using separate RNG streams.
    """
    phase: Tensor | None = None
    mask: Tensor | None = None

    if spec.edge_phase.enabled and spec.edge_phase.sigma > 0.0:
        g = rng.stream("edge_phase", extra=int(extra))

        if spec.edge_phase.flipflop_safe:
            phase = sample_flipflop_edge_phases(
                rev,
                spec.edge_phase.sigma,
                generator=g,
                clamp_abs=spec.edge_phase.clamp_abs,
                validate=validate,
            )
        else:
            # Unsafe iid phases (does NOT enforce phase[rev]=conj(phase))
            # Provided for research comparisons; default in config should remain flipflop_safe=True.
            phi = randn_device((int(rev.numel()),), device=rev.device, dtype=torch.float32, generator=g) * float(
                spec.edge_phase.sigma
            )
            if spec.edge_phase.clamp_abs is not None and float(spec.edge_phase.clamp_abs) > 0:
                c = float(spec.edge_phase.clamp_abs)
                phi = phi.clamp(min=-c, max=c)
            phase = torch.exp(torch.complex(torch.zeros_like(phi), phi)).to(torch.complex64)

    if spec.edge_dropout.enabled and spec.edge_dropout.p > 0.0:
        g = rng.stream("edge_dropout", extra=int(extra))
        if spec.edge_dropout.flipflop_safe:
            mask = sample_flipflop_edge_keep_mask(rev, spec.edge_dropout.p, generator=g, validate=validate)
        else:
            # Unsafe iid dropout per arc (NOT symmetric). Not recommended for flip-flop walks.
            u = rand_device((int(rev.numel()),), device=rev.device, dtype=torch.float32, generator=g)
            mask = (u >= float(spec.edge_dropout.p)).to(torch.float32)

    return ShiftDisorder(phase=phase, mask=mask)


def format_disorder_meta(spec: DisorderSpec) -> dict[str, Any]:
    """
    Small helper for logging/JSON. Keeps only plain python types.
    """
    return {
        "seed": int(spec.seed),
        "episode_index": int(spec.episode_index),
        "trial_id": int(spec.trial_id),
        "edge_phase": {
            "enabled": bool(spec.edge_phase.enabled),
            "sigma": float(spec.edge_phase.sigma),
            "flipflop_safe": bool(spec.edge_phase.flipflop_safe),
            "clamp_abs": None if spec.edge_phase.clamp_abs is None else float(spec.edge_phase.clamp_abs),
        },
        "edge_dropout": {
            "enabled": bool(spec.edge_dropout.enabled),
            "p": float(spec.edge_dropout.p),
            "flipflop_safe": bool(spec.edge_dropout.flipflop_safe),
        },
        "coin_dephase": {
            "enabled": bool(spec.coin_dephase.enabled),
            "sigma": float(spec.coin_dephase.sigma),
        },
    }






def _any_enabled(spec: DisorderSpec) -> bool:
    return bool(
        (spec.edge_phase.enabled and spec.edge_phase.sigma > 0.0)
        or (spec.edge_dropout.enabled and spec.edge_dropout.p > 0.0)
        or (spec.coin_dephase.enabled and spec.coin_dephase.sigma > 0.0)
    )


def build_episode_disorder(
    rev: Tensor,
    disorder: Mapping[str, Any] | DisorderSpec | None,
    *,
    device: torch.device,
    validate: bool = False,
) -> EpisodeDisorder | None:
    """
    Build a deterministic episode-level disorder bundle.

    Parameters
    ----------
    rev:
        Arc involution permutation (A,) mapping each arc to its flip-flop partner.
        In your simulator this is `shift.perm`.
    disorder:
        Either `None`, a dict (typically batch["disorder"]), or a DisorderSpec.
    device:
        Target device for RNG + sampled modifiers.
    validate:
        If True, checks that rev is an involution (debug aid).

    Returns
    -------
    EpisodeDisorder | None
        None if disorder is None or all channels are disabled/zero.
    """
    if disorder is None:
        return None

    spec = disorder if isinstance(disorder, DisorderSpec) else parse_disorder_spec(disorder)
    if not _any_enabled(spec):
        return None

    rev_t = rev
    if rev_t.device != device:
        rev_t = rev_t.to(device=device)

    rng = DisorderRNG(spec, device=device)

    # Sample ONCE per episode (stable across time steps).
    shift_mod = sample_shift_disorder(rev_t, spec, rng=rng, extra=0, validate=validate)

    return EpisodeDisorder(spec=spec, rng=rng, shift=shift_mod)


def maybe_apply_coin_dephase(
    psi: Tensor,
    ctx: EpisodeDisorder | None,
    *,
    t: int,
) -> Tensor:
    """
    Apply per-step coin/state dephasing if enabled.

    Uses a per-step deterministic substream keyed by `t`, so results are stable
    even if episode lengths change or early exits happen.
    """
    if ctx is None:
        return psi
    spec = ctx.spec
    if not (spec.coin_dephase.enabled and spec.coin_dephase.sigma > 0.0):
        return psi
    g = ctx.rng.stream("coin_dephase", extra=int(t))
    return apply_state_dephase(psi, spec.coin_dephase.sigma, generator=g)

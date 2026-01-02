#!/usr/bin/env python3
# scripts/train.py
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, asdict
import math
from pathlib import Path
import re
import hashlib
import inspect

import sys
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from acpl.eval.protocol import EvalConfig, run_ci_eval, summarize_results





try:
    # Required by the FeatureSpec-driven path later in this script
    from acpl.data.features import FeatureSpec, build_node_features, normalize_degree  # type: ignore
except Exception as e:
    FeatureSpec = None  # type: ignore
    build_node_features = None  # type: ignore
    normalize_degree = None  # type: ignore
    _FEATURES_IMPORT_ERR = e


def _pad_epoch(i: int) -> str:
    return f"{i:08d}"


class EpochAverager:
    """
    Collect numeric metrics across an epoch and report means.
    Usage:
      avg = EpochAverager(prefix="train/")
      avg.update({"loss": 0.12, "mix/tv": 0.4})
      ...
      means = avg.means()
    """

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix
        self._sum = defaultdict(float)
        self._cnt = defaultdict(int)

    def update(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            if v is None:
                continue
            # normalize keys to have prefix once
            key = k if not self.prefix or k.startswith(self.prefix) else f"{self.prefix}{k}"
            self._sum[key] += float(v)
            self._cnt[key] += 1

    def means(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, s in self._sum.items():
            c = max(1, self._cnt[k])
            out[k] = s / c
        return out


def _fmt_epoch_summary(epoch_idx: int, means: dict[str, float], lr: float) -> str:
    eid = _pad_epoch(epoch_idx)
    # Pull known groups (fall back to None if missing)
    get = means.get
    # Core lines
    lines = []
    lines.append(f"[{eid}] train/loss: {get('train/loss', float('nan')): .6f}")
    lines.append(f"[{eid}] train/lr: {lr:.6f}")

    # Mix metrics (group known keys if available)
    mix_keys = [
        "train/mix/tv",
        "train/mix/js",
        "train/mix/hell",
        "train/mix/l2",
        "train/mix/H",
        "train/mix/kl_pu",
    ]
    mix_parts = []
    for k in mix_keys:
        if k in means:
            name = k.split("/", 2)[-1]
            mix_parts.append(f"{name}={means[k]:.6f}")
    if mix_parts:
        lines.append(f"[{eid}] train/mix/ " + " ".join(mix_parts))

    # Target metrics
    tgt_keys = [
        "train/target/success",
        "train/target/maxp",
        "train/target/gini",
        "train/target/tv_vsU",
        "train/target/js_vsU",
        "train/target/H",
        "train/target/KLpU",
        "train/target/cvar_neglogOmega",
    ]
    tgt_parts = []
    for k in tgt_keys:
        if k in means:
            name = k.split("/", 2)[-1]
            tgt_parts.append(f"{name}={means[k]:.6f}")
    if tgt_parts:
        lines.append(f"[{eid}] train/target/ " + " ".join(tgt_parts))

    return "\n".join(lines)


# -------------------------------
# Optional Phase-B8 utils (robust fallbacks if missing)
# -------------------------------


def _seed_everything(seed: int) -> None:
    """
    Prefer acpl.utils.seeding.seed_everything and fall back to a minimal implementation.
    """
    try:
        from acpl.utils import seeding as _seeding  # type: ignore

        if hasattr(_seeding, "seed_everything"):
            # Matches acpl/utils/seeding.py in your uploaded utils.
            _seeding.seed_everything(seed=seed, deterministic=True)
            return
    except Exception:
        pass

    # Fallback (minimal)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass



def _derive_episode_seed(manifest_hex: str, split: str, index: int) -> int:
    """
    Deterministic seed from (manifest_hex, split, global_index).
    Keep this consistent with your acpl.data generator seeding.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(manifest_hex).encode("utf-8"))
    h.update(b"/")
    h.update(str(split).encode("utf-8"))
    h.update(b"/")
    h.update(str(int(index)).encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False)



# Timers
class _NullTimer:
    def __enter__(self):  # noqa: D401
        """No-op timer (fallback)."""
        return self

    def __exit__(self, *exc):
        return False


def _time_block(label: str):
    """
    Prefer acpl.utils.timers.record context manager if available.
    """
    try:
        from acpl.utils.timers import record as _record  # type: ignore
        # record(name, ...) returns a context manager
        return _record(label)
    except Exception:
        return _NullTimer()


# Checkpointing (atomic save/load; resume)
class _CheckpointShim:
    def __init__(self):
        self._ok = False
        try:
            from acpl.utils import checkpoint as _ck  # type: ignore
            self.ck = _ck
            self._ok = True
        except Exception:
            self.ck = None

    def atomic_save(self, obj: dict, path: Path) -> None:
        # If your Phase-B8 checkpointer has a truly-atomic writer, use it.
        if self._ok and hasattr(self.ck, "atomic_save"):
            self.ck.atomic_save(obj, path)  # type: ignore
            return

        # Robust atomic fallback: write temp, flush+fsync, then replace.
        import os
        tmp = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)

        with tmp.open("wb") as f:
            # Force the *zip* format (more robust/portable than legacy pickle)
            torch.save(obj, f, _use_new_zipfile_serialization=True)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    def save_train_state(self, path: Path, **state) -> None:
        # Always use atomic save for model_last.pt to avoid partial/corrupt checkpoints.
        self.atomic_save(state, path)

    def try_resume(self, path: Path) -> dict | None:
        """
        Return loaded dict if found, else None. Accepts either a file or a directory.
        """
        if self._ok and hasattr(self.ck, "load_checkpoint_resume"):
            try:
                ok, payload = self.ck.load_checkpoint_resume(path)  # type: ignore
                return payload if ok else None
            except Exception:
                return None

        if path.is_dir():
            for name in ("model_last.pt", "last.pt", "checkpoint.pt"):
                cand = path / name
                if cand.exists():
                    path = cand
                    break

        if path.exists():
            try:
                # IMPORTANT for PyTorch 2.x: be explicit
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                return None
        return None


_CK = _CheckpointShim()

# -------------------------------
# Progress
# -------------------------------
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

# -------------------------------
# Graphs / Policy / Sim / Train
# -------------------------------
from acpl.data import graphs as G
from acpl.policy.policy import ACPLPolicy, ACPLPolicyConfig
from acpl.sim.portmap import PortMap, build_portmap
from acpl.sim.shift import ShiftOp, build_shift
from acpl.train.loops import LoopConfig, RolloutFn, build_metric_pack, train_epoch
from acpl.utils.logging import MetricLogger, MetricLoggerConfig

# -------------------------------
# Utilities: device / dtype
# -------------------------------


def select_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_flag)


def select_dtype(dtype_flag: str):
    mapping = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}
    if dtype_flag not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_flag}'. Choose from {list(mapping)}.")
    return mapping[dtype_flag]


def select_cdtype(state_flag: str):
    s = str(state_flag).lower().strip()
    mapping = {"complex64": torch.complex64, "complex128": torch.complex128}
    if s not in mapping:
        raise ValueError(f"Unsupported state dtype '{state_flag}'. Choose from {list(mapping)}.")
    return mapping[s]



def configure_cuda_perf(*, deterministic: bool = False, capture_scalar_outputs: bool = True) -> None:
    """Enable Ampere-friendly GPU performance knobs (TF32, cuDNN benchmarking, Dynamo scalar capture)."""
    if not torch.cuda.is_available():
        return

    # Enable TF32 tensor cores for float32 matmuls/conv (removes the warning + speeds up).
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Legacy flags (work on stable PyTorch)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # cuDNN autotuning: faster, but nondeterministic
    try:
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    except Exception:
        pass

    # Reduce torch.compile graph breaks from Tensor.item() (optional)
    if capture_scalar_outputs:
        try:
            import torch._dynamo as _dynamo  # type: ignore
            _dynamo.config.capture_scalar_outputs = True
        except Exception:
            pass

# -------------------------------
# Config IO and overrides
# -------------------------------


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _coerce_scalar(s: str) -> Any:
    if isinstance(s, (int, float, bool)):
        return s
    if not isinstance(s, str):
        return s
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Accept both Hydra-like 'key=value' and our prior '++key=value'.
    Includes a small alias map so common keys from older configs work.
    """
    alias = {
        # CLI → config path
        "logging.dir": "train.run_dir",  # your smoke test
        "dataset.num_episodes": "train.episodes_per_epoch",
        "train.max_steps": "train.episodes_per_epoch",  # best-effort: treat as episode count
        # optional convenience
        "dtype": "dtype",
        "device": "device",
        "seed": "seed",
    }

    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        # drop any leading '+' (supports both '++k=v' and 'k=v')
        key = key.lstrip("+")
        # normalize & alias
        key = alias.get(key, key)
        val = _coerce_scalar(val)

        # walk & set
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = val
    return cfg

def _feature_spec_from_cfg(cfg: dict, *, seed: int | None) -> FeatureSpec | None:
    """
    If cfg contains a 'features:' dict, interpret it as FeatureSpec kwargs.
    If missing/None, return None meaning "legacy features".
    Unknown keys are ignored (so configs don’t hard-break).
    """
    feat_cfg = cfg.get("features", None)
    if feat_cfg is None:
        return None
    


    if FeatureSpec is None:
        raise RuntimeError(
            f"Config contains 'features:' but acpl.data.features could not be imported: {_FEATURES_IMPORT_ERR}"
        )


    if not isinstance(feat_cfg, dict):
        raise TypeError("cfg['features'] must be a dict if provided.")

    # Only keep keys that exist on FeatureSpec
    allowed = set(FeatureSpec.__dataclass_fields__.keys())
    clean = {k: v for k, v in feat_cfg.items() if k in allowed}

    # If user didn’t set a seed, inherit master seed for randomized options (e.g. random_sign)
    if clean.get("seed", None) is None and seed is not None:
        clean["seed"] = int(seed)

    return FeatureSpec(**clean)


def _default_coords_1d(N: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Deterministic 1D normalized coordinate in [0,1], shape (N,1).
    This is the legacy behavior you had before (coord1).
    """
    return (torch.arange(N, device=device, dtype=dtype).unsqueeze(1) / max(N - 1, 1)).to(dtype)


def _coerce_prereg_to_trainer_schema(cfg: dict, *, cfg_path: Path) -> dict:
    """
    Coerce "pre-registration" experiment YAMLs (goal/graph/horizon/policy/optimization/...)
    into the trainer schema expected by this script (task/data/sim/model/coin/train/optim/log/...).

    If cfg already looks like a trainer-schema config, this is a no-op.
    """
    if not isinstance(cfg, dict):
        return cfg

    # Heuristic: already trainer-like
    if "data" in cfg and "sim" in cfg and ("model" in cfg or "coin" in cfg) and ("train" in cfg or "optim" in cfg):
        return cfg

    # Heuristic: prereg-like
    prereg_like = ("goal" in cfg) and (("graph" in cfg) or ("horizon" in cfg) or ("policy" in cfg) or ("optimization" in cfg))
    if not prereg_like:
        return cfg

    goal = str(cfg.get("goal", "transfer")).lower().strip()

    # ---- seed / device / dtype ----
    rnd = cfg.get("randomness", {}) or {}
    cb = cfg.get("compute_budget", {}) or {}
    cb_dtype = (cb.get("dtype", {}) or {}) if isinstance(cb.get("dtype", {}), dict) else {}

    cfg.setdefault("seed", int(rnd.get("master_seed", cfg.get("seed", 0) or 0)))
    cfg.setdefault("device", cb.get("device", "auto"))

    # encoder dtype (this script's `dtype`)
    cfg.setdefault("dtype", str(cb_dtype.get("encoder", cfg.get("dtype", "float32"))))

    # state dtype (new optional field used below)
    cfg.setdefault("state_dtype", str(cb_dtype.get("state", cfg.get("state_dtype", "complex64"))))

    # ---- task ----
    t = cfg.get("task", None)
    if not isinstance(t, dict):
        # prereg has `task: spatial_search` (string); we keep it as a name hint
        t = {"name": goal}
        cfg["task"] = t
    t.setdefault("name", goal)

    # ---- data (from graph) ----
    graph = cfg.get("graph", {}) or {}
    data = cfg.get("data", {}) or {}
    if not isinstance(data, dict):
        data = {}
    cfg["data"] = data

    fam = str(graph.get("family", data.get("family", "line"))).lower().strip()
    data.setdefault("family", fam)

    # carry marks_per_episode for search
    if "marks_per_episode" in graph and "marks_per_episode" not in data:
        data["marks_per_episode"] = int(graph.get("marks_per_episode", 1))

    # grid sizing
    if fam in ("grid", "grid-2d", "grid2d"):
        data.setdefault("grid", {})
        if not isinstance(data["grid"], dict):
            data["grid"] = {}
        g = data["grid"]

        # pick a default L if not provided (prefer 64 if present)
        L = None
        train_L = graph.get("train_L", None)
        if isinstance(train_L, (list, tuple)) and len(train_L) > 0:
            vals = [int(x) for x in train_L]
            if 64 in vals:
                L = 64
            else:
                L = vals[len(vals) // 2]
        if L is None:
            # fallback to any existing L/Lx/Ly, else 32
            if "L" in g:
                L = int(g["L"])
            elif "Lx" in g and "Ly" in g:
                L = int(g["Lx"])
            else:
                L = int(graph.get("L", 32))

        g.setdefault("Lx", int(L))
        g.setdefault("Ly", int(L))

    # hypercube sizing (map to this script's 'cube' builder)
    if fam in ("hypercube", "cube"):
        data.setdefault("cube", {})
        if not isinstance(data["cube"], dict):
            data["cube"] = {}
        # prereg uses n as dimension; this script uses cube.d
        n = None
        hc = graph.get("hypercube", {}) if isinstance(graph.get("hypercube", {}), dict) else {}
        if "n" in hc:
            n = int(hc["n"])
        data["cube"].setdefault("d", int(n) if n is not None else int(data["cube"].get("d", 6)))
        data["family"] = "cube"

    # ---- sim (from horizon) ----
    horizon = cfg.get("horizon", {}) or {}
    sim = cfg.get("sim", {}) or {}
    if not isinstance(sim, dict):
        sim = {}
    cfg["sim"] = sim

    # Keep curriculum list in sim.steps, but also set task.horizon_T preference if possible
    cand = horizon.get("T_candidates", None)
    if isinstance(cand, (list, tuple)) and len(cand) > 0:
        sim.setdefault("steps", [int(x) for x in cand])
        # prefer 128 if present; else median
        cvals = [int(x) for x in cand]
        pref = 128 if 128 in cvals else cvals[len(cvals) // 2]
        t.setdefault("horizon_T", int(pref))
    else:
        # fallback: explicit horizon_T if present
        sim.setdefault("steps", int(t.get("horizon_T", 128)))

    # ---- model/coin (from policy) ----
    policy = cfg.get("policy", {}) or {}
    model = cfg.get("model", {}) or {}
    if not isinstance(model, dict):
        model = {}
    cfg["model"] = model

    pg = policy.get("gnn", {}) or {}
    pc = policy.get("controller", {}) or {}
    if "gnn" not in model:
        model["gnn"] = {}
    if "controller" not in model:
        model["controller"] = {}
    if not isinstance(model["gnn"], dict):
        model["gnn"] = {}
    if not isinstance(model["controller"], dict):
        model["controller"] = {}

    model["gnn"].setdefault("hidden", int(pg.get("hidden", 128)))
    model["gnn"].setdefault("dropout", float(pg.get("dropout", 0.1)))
    model["controller"].setdefault("kind", str(pc.get("kind", "gru")).lower())
    model["controller"].setdefault("hidden", int(pc.get("hidden", 128)))
    model["controller"].setdefault("layers", int(pc.get("layers", 1)))
    model["controller"].setdefault("dropout", float(pc.get("dropout", 0.0)))

    coin = cfg.get("coin", {}) or {}
    if not isinstance(coin, dict):
        coin = {"family": str(coin)}
    cfg["coin"] = coin

    pcoin = policy.get("coin", {}) or {}
    cmap = str(pcoin.get("map", coin.get("family", "exp"))).lower().strip()
    # map prereg "cayley" -> this script's coin.family
    if cmap in ("cayley", "exp", "su2"):
        coin.setdefault("family", cmap)
    else:
        coin.setdefault("family", "exp")

    # ---- train/optim (from optimization) ----
    opt = cfg.get("optimization", {}) or {}
    train = cfg.get("train", {}) or {}
    optim = cfg.get("optim", {}) or {}
    if not isinstance(train, dict):
        train = {}
    if not isinstance(optim, dict):
        optim = {}
    cfg["train"] = train
    cfg["optim"] = optim

    train.setdefault("epochs", int(opt.get("epochs", 160)))

    # this script is "episode-per-step" right now; set a sane default episodes_per_epoch
    # (we can’t truly batch episodes without refactoring rollouts)
    batch_eps = int(opt.get("batch_episodes", 32))
    train.setdefault("batch_size", 1)
    train.setdefault("episodes_per_epoch", int(opt.get("episodes_per_epoch", max(64, 8 * batch_eps))))

    optim.setdefault("name", str(opt.get("optimizer", "adam")).lower())
    optim.setdefault("lr", float(opt.get("lr", 2.5e-4)))
    optim.setdefault("weight_decay", float(opt.get("weight_decay", 1e-5)))
    optim.setdefault("grad_clip", float(opt.get("grad_clip", 1.0)))

    # ---- log ----
    log = cfg.get("log", {}) or {}
    if not isinstance(log, dict):
        log = {}
    cfg["log"] = log
    log.setdefault("backend", "plain")
    log.setdefault("interval", 50)
    log.setdefault("run_name", cfg_path.stem)

    return cfg



def _to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    return d


# --- Small helper: remove an accidental leading batch dim (e.g., (1,N,F) -> (N,F)) ---
def _maybe_squeeze_leading(x):
    if isinstance(x, torch.Tensor) and x.ndim >= 2 and x.size(0) == 1:
        return x.squeeze(0)
    return x


# -------------------------------
# Minimal dataset wrapper (single episode)
# -------------------------------


class SingleGraphEpisodeDataset(Dataset):
    def __init__(self, payload: dict, num_episodes: int = 1000):
        super().__init__()
        self.payload = payload
        self.num_episodes = int(num_episodes)

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict:
        return self.payload
class MarkedGraphEpisodeDataset(Dataset):
    """
    Per-episode marked set Ω sampling + mark-indicator node feature.

    For search:
      - sample marks_per_episode nodes as Ω
      - set batch["targets"] = Ω
      - append a 0/1 mark indicator to X (so policy conditions on Ω)

    The sampling is deterministic per (base_seed, epoch, idx) for reproducibility.
    """

    def __init__(
        self,
        payload: dict,
        *,
        num_episodes: int,
        N: int,
        marks_per_episode: int,
        manifest_hex: str,
        split: str,
        append_mark_indicator: bool = True,
    ):
        super().__init__()
        self.payload = payload
        self.num_episodes = int(num_episodes)
        self.N = int(N)
        self.k = int(max(1, marks_per_episode))
        self.manifest_hex = str(manifest_hex)
        self.split = str(split)
        self.append_mark_indicator = bool(append_mark_indicator)
        self.epoch = 0

        X = payload["X"]
        if not isinstance(X, torch.Tensor):
            raise TypeError("payload['X'] must be a torch.Tensor")
        self.device = X.device


    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_episodes

    def _episode_seed(self, idx: int) -> int:
        # Global index for this split/epoch so each epoch uses a fresh, deterministic set
        gidx = int(self.epoch) * int(self.num_episodes) + int(idx)
        return _derive_episode_seed(self.manifest_hex, self.split, gidx)

    def __getitem__(self, idx: int) -> dict:
        # Deterministic sampling for this episode
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self._episode_seed(idx))

        # Sample Ω without replacement
        perm = torch.randperm(self.N, generator=gen)
        marks = perm[: self.k].to(device=self.device, dtype=torch.long)

        X_base: torch.Tensor = self.payload["X"]  # (N, F_base)
        out = dict(self.payload)

        if self.append_mark_indicator:
            mark_feat = torch.zeros((self.N, 1), device=self.device, dtype=X_base.dtype)
            mark_feat.index_fill_(0, marks, 1.0)
            out["X"] = torch.cat([X_base, mark_feat], dim=1)  # (N, F_base+1)
        else:
            out["X"] = X_base

        out["targets"] = marks  # define Ω as "targets" for loss/metrics
        out["marks"] = marks
        return out

class RobustTargetEpisodeDataset(Dataset):
    """
    Robust transfer-style episodes:
      - sample k target nodes per episode as Ω (targets)
      - optionally append a 0/1 target-indicator channel to X
      - optionally sample a start_node per episode (rollout can use it)

    Deterministic per (base_seed, epoch, idx).
    """

    def __init__(
        self,
        payload: dict,
        *,
        num_episodes: int,
        N: int,
        targets_per_episode: int = 1,
        manifest_hex: str,
        split: str,
        append_target_indicator: bool = True,
        random_start: bool = True,
    ):
        super().__init__()
        self.payload = payload
        self.num_episodes = int(num_episodes)
        self.N = int(N)
        self.k = int(max(1, targets_per_episode))
        self.manifest_hex = str(manifest_hex)
        self.split = str(split)
        self.append_target_indicator = bool(append_target_indicator)
        self.random_start = bool(random_start)
        self.epoch = 0

        X = payload["X"]
        if not isinstance(X, torch.Tensor):
            raise TypeError("payload['X'] must be a torch.Tensor")
        self.device = X.device

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_episodes

    def _episode_seed(self, idx: int) -> int:
        # Global index for this split/epoch so each epoch uses a fresh, deterministic set
        gidx = int(self.epoch) * int(self.num_episodes) + int(idx)
        return _derive_episode_seed(self.manifest_hex, self.split, gidx)
    def __getitem__(self, idx: int) -> dict:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self._episode_seed(idx))

        perm = torch.randperm(self.N, generator=gen)

        # Targets Ω
        targets = perm[: self.k].to(device=self.device, dtype=torch.long)

        # Optional random start node
        if self.random_start:
            start_node = int(perm[self.k].item()) if (self.k < self.N) else int(perm[0].item())
        else:
            start_node = int(self.payload.get("start_node", 0))

        out = dict(self.payload)
        out["targets"] = targets
        out["start_node"] = start_node

        if self.append_target_indicator:
            X_base: torch.Tensor = self.payload["X"]  # (N,2) expected
            tfeat = torch.zeros((self.N, 1), device=self.device, dtype=X_base.dtype)
            tfeat.index_fill_(0, targets, 1.0)
            out["X"] = torch.cat([X_base, tfeat], dim=1)  # (N,3)

        return out

# -------------------------------
# Initial state helpers
# -------------------------------


def init_state_node0_uniform_ports(pm: PortMap, *, cdtype, device) -> torch.Tensor:
    A = pm.src.numel()
    psi = torch.zeros(A, dtype=cdtype, device=device)
    s0, e0 = int(pm.node_ptr[0]), int(pm.node_ptr[1])
    d0 = max(1, e0 - s0)
    amp = 1.0 / math.sqrt(d0)
    psi[s0:e0] = amp
    return psi


def init_state_node_uniform_ports(pm: PortMap, node: int, *, cdtype, device) -> torch.Tensor:
    """
    Uniform superposition over the outgoing ports of a chosen node.
    """
    node = int(node) % int(pm.num_nodes)
    A = pm.src.numel()
    psi = torch.zeros(A, dtype=cdtype, device=device)
    s, e = int(pm.node_ptr[node]), int(pm.node_ptr[node + 1])
    d = max(1, e - s)
    amp = 1.0 / math.sqrt(d)
    psi[s:e] = amp
    return psi


def init_state_uniform_arcs(pm: PortMap, *, cdtype, device) -> torch.Tensor:
    """
    Uniform superposition over *all arcs/ports* (the natural |s> state for spatial search).
    """
    A = int(pm.src.numel())
    amp = 1.0 / math.sqrt(max(1, A))
    psi = torch.ones(A, dtype=cdtype, device=device) * amp
    return psi


# -------------------------------
# Graph from config (dynamic dispatch)
# -------------------------------


@dataclass
class GraphData:
    edge_index: torch.Tensor  # (2, E)
    degrees: torch.Tensor  # (N,)
    coords: torch.Tensor | None  # (N,2) or None
    arc_slices: torch.Tensor  # (N+1,)
    N: int


def _resolve_builder(names: list[str]) -> Callable:
    for n in names:
        fn = getattr(G, n, None)
        if callable(fn):
            return fn
    raise ImportError(
        f"None of the expected graph builders {names} were found in acpl.data.graphs. "
        f"Check acpl/data/graphs.py for exported names (__all__)."
    )


def graph_from_config(data_cfg: dict, *, device, dtype) -> GraphData:
    family = data_cfg.get("family", data_cfg.get("graph", "line")).lower().strip()




   # --- alias normalization (fixes config family names like 'grid-2d') ---
    if family in ("grid-2d", "grid2d", "grid_2d"):
        family = "grid"
    # (optional but useful future-proofing)
    elif family in ("hypercube",):
        family = "cube"
    elif family in ("d-regular", "d_regular"):
        family = "regular"
    elif family in ("erdos-renyi", "erdos_renyi"):
        family = "er"
    # ---------------------------------------------------------------------





    seed = int(data_cfg.get("seed", 1234))

    if family == "line":
        N = int(data_cfg.get("N", data_cfg.get("num_nodes", 64)))
        fn = _resolve_builder(["line_graph"])
        ei, deg, coords, arc = fn(N)
    elif family == "cycle":
        N = int(data_cfg.get("N", data_cfg.get("num_nodes", 64)))
        fn = _resolve_builder(["cycle_graph"])
        ei, deg, coords, arc = fn(N)
    elif family == "grid":
        g = data_cfg.get("grid", {})
        Lx, Ly = int(g.get("Lx", 8)), int(g.get("Ly", 8))
        fn = _resolve_builder(["grid_graph"])
        ei, deg, coords, arc = fn(Lx, Ly)
        N = Lx * Ly
    elif family == "cube":
        try:
            fn = _resolve_builder(["cube_graph"])
        except ImportError:
            raise RuntimeError(
                "Requested family 'cube' but 'cube_graph' is not exported by acpl.data.graphs. "
                "Implement it or choose another family."
            )
        d = int(data_cfg.get("cube", {}).get("d", 6))
        ei, deg, coords, arc = fn(d)
        N = 2**d
    elif family == "regular":
        r = data_cfg.get("regular", {})
        N = int(r.get("N", data_cfg.get("N", 64)))
        d = int(r.get("d", 3))
        fn = _resolve_builder(
            ["d_regular_random_graph", "d_regular_graph", "regular_graph", "regular_d_graph"]
        )
        ei, deg, coords, arc = fn(N, d, seed=seed)
    elif family == "er":
        r = data_cfg.get("er", {})
        N = int(r.get("N", data_cfg.get("N", 64)))
        p = float(r.get("p", 0.05))
        fn = _resolve_builder(["er_graph", "erdos_renyi_graph"])
        ei, deg, coords, arc = fn(N, p, seed=seed)
    elif family == "ws":
        r = data_cfg.get("ws", {})
        Lx, Ly = int(r.get("Lx", 10)), int(r.get("Ly", 10))
        kx, ky = int(r.get("kx", 1)), int(r.get("ky", 1))
        beta = float(r.get("beta", 0.2))
        torus = bool(r.get("torus", False))
        fn = _resolve_builder(["watts_strogatz_grid_graph", "ws_grid_graph"])
        try:
            ei, deg, coords, arc = fn(Lx, Ly, kx, ky, beta, seed=seed, torus=torus)
        except TypeError:
            ei, deg, coords, arc = fn(Lx, Ly, kx, ky, beta, seed=seed)
        N = Lx * Ly
    else:
        raise ValueError(f"Unknown graph family '{family}'")

    return GraphData(
        edge_index=ei.to(device),
        degrees=deg.to(device),
        coords=(coords.to(device) if coords is not None else None),
        arc_slices=arc.to(device),
        N=int(N),
    )


# -------------------------------
# Port-Fourier → Hermitian adaptor (Phase B)
# -------------------------------


class ThetaToHermitianAdaptor(nn.Module):
    def __init__(self, Kfreq: int = 2):
        super().__init__()
        self.Kfreq = int(Kfreq)
        self.B = 2 * self.Kfreq + 1
        self.proj = nn.Linear(3, self.B, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Cache basis matrices per (d, device, dtype) to avoid recomputing
        # sinusoid + outer products every time in the rollout loop.
        # key: (int_d, torch.device, torch.dtype) -> Tensor(B, d, d)
        self._basis_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    @staticmethod
    def _port_features(
        d: int,
        Kfreq: int,
        device=None,
        dtype=torch.float32,
    ) -> torch.Tensor:
        import math as _m

        j = torch.arange(d, device=device, dtype=dtype).unsqueeze(1)
        cols = [torch.ones(d, 1, device=device, dtype=dtype)]
        for k in range(1, Kfreq + 1):
            ang = 2.0 * _m.pi * k * j / max(d, 1)
            cols.append(torch.sin(ang))
            cols.append(torch.cos(ang))
        return torch.cat(cols, dim=1)  # (d, 2K+1)

    def _basis_matrices(self, d: int, device, dtype) -> torch.Tensor:
        """
        Return cached basis matrices M_k (B, d, d) for this d/device/dtype.
        """
        d_int = int(d)
        device = torch.device(device)
        key = (d_int, device, dtype)

        cached = self._basis_cache.get(key, None)
        if cached is not None:
            return cached

        Phi = self._port_features(d_int, self.Kfreq, device=device, dtype=dtype)  # (d,B)
        Ms = []
        for k in range(Phi.shape[-1]):
            phi = Phi[:, k : k + 1]
            Ms.append(phi @ phi.t())  # (d,d)
        M = torch.stack(Ms, dim=0)  # (B,d,d)

        self._basis_cache[key] = M
        return M

    def hermitian_from_theta(self, theta_vt: torch.Tensor, d: int) -> torch.Tensor:
        """
        Map theta_vt (...,3) to a Hermitian matrix H (..., d, d) using cached bases.
        """
        *batch, _ = theta_vt.shape
        # Project theta into B coefficients
        w = self.proj(theta_vt.reshape(-1, 3)).view(*batch, self.B)  # (...,B)

        M = self._basis_matrices(d, device=theta_vt.device, dtype=torch.float32)  # (B,d,d)
        M = M.to(dtype=w.dtype)

        H = torch.tensordot(w, M, dims=([-1], [0]))  # (..., d, d)
        H = 0.5 * (H + H.transpose(-1, -2))  # ensure Hermitian numerically
        return H




    def forward(self, theta: torch.Tensor, d: int | None = None) -> torch.Tensor:
        """
        Make the adaptor callable: H = adaptor(theta[, d]).
        This prevents 'Module [...] is missing the required "forward" function'.
        """
        # Prefer the most explicit method names if they exist in your implementation.
        if hasattr(self, "hermitian_from_theta") and callable(getattr(self, "hermitian_from_theta")):
            fn = getattr(self, "hermitian_from_theta")
            try:
                return fn(theta, d=d)
            except TypeError:
                return fn(theta, d)

        if hasattr(self, "theta_to_hermitian") and callable(getattr(self, "theta_to_hermitian")):
            fn = getattr(self, "theta_to_hermitian")
            try:
                return fn(theta, d=d)
            except TypeError:
                return fn(theta, d)

        if hasattr(self, "to_hermitian") and callable(getattr(self, "to_hermitian")):
            fn = getattr(self, "to_hermitian")
            try:
                return fn(theta, d=d)
            except TypeError:
                return fn(theta, d)

        raise NotImplementedError(
            f"{type(self).__name__} has no known theta->Hermitian method to call from forward()."
        )









def unitary_exp_iH(H: torch.Tensor, cdtype=torch.complex64) -> torch.Tensor:
    n = H.size(-1)
    Hs = 0.5 * (H + H.transpose(-1, -2))

    # RTX cards: keep FP32 on CUDA (FP64 is extremely slow)
    if Hs.is_cuda:
        Hwork = Hs.to(torch.float32)
    else:
        Hwork = Hs.to(torch.float64)

    eps = 1e-8
    Hwork = Hwork + torch.eye(n, dtype=Hwork.dtype, device=Hwork.device) * eps

    try:
        evals, evecs = torch.linalg.eigh(Hwork)  # supports batched (...,n,n)
        evals = torch.clamp(evals, min=-1e6, max=1e6)
        D = torch.diag_embed(torch.complex(torch.cos(evals), torch.sin(evals)).to(cdtype))
        Q = evecs.to(cdtype)
        return Q @ D @ Q.transpose(-1, -2).conj()
    except RuntimeError:
        return torch.matrix_exp(1j * Hs.to(cdtype))



def unitary_cayley(H: torch.Tensor, cdtype=torch.complex64) -> torch.Tensor:
    n = H.size(-1)
    Hs = 0.5 * (H + H.transpose(-1, -2))
    I = torch.eye(n, dtype=cdtype, device=H.device)
    iH = 1j * Hs.to(cdtype)
    eps = 1e-8
    A = I - iH + eps * I
    B = I + iH
    return torch.linalg.solve(A, B)


def apply_blockdiag_coins_anydeg(
    psi: torch.Tensor,
    pm: PortMap,
    coins_v: list[torch.Tensor | None],
) -> torch.Tensor:
    out = psi.clone()
    start = pm.node_ptr[:-1]
    end = pm.node_ptr[1:]



    N = pm.num_nodes
    for v in range(N):
        s, e = int(start[v]), int(end[v])
        d = e - s
        if d <= 0:
            continue
        U = coins_v[v]
        if U is None:
            continue
        out[s:e] = U @ psi[s:e]
    return out


# -------------------------------
# Rollouts
# -------------------------------


def make_rollout_su2_dv2(pm: PortMap, shift: ShiftOp, *, cdtype, init_mode: str = "node0") -> RolloutFn:
    from acpl.sim.step import step
    from acpl.sim.utils import partial_trace_coin

    def rollout(model: nn.Module, batch: dict):
        X: torch.Tensor = _maybe_squeeze_leading(batch["X"])
        edge_index: torch.Tensor = _maybe_squeeze_leading(batch["edge_index"])
        T: int = int(
            _maybe_squeeze_leading(batch["T"]).item()
            if isinstance(batch["T"], torch.Tensor)
            else batch["T"]
        )
        targets = batch.get("targets", None)
        if isinstance(targets, torch.Tensor) and targets.ndim > 1 and targets.size(0) == 1:
            targets = targets.squeeze(0)

        coins = model.coins_su2(X, edge_index, T=T)  # (T,N,2,2) complex
        if init_mode == "uniform":
            psi = init_state_uniform_arcs(pm, cdtype=coins.dtype, device=X.device)
        elif init_mode == "node":
            s0 = batch.get("start_node", 0)
            s0 = int(s0.item()) if isinstance(s0, torch.Tensor) else int(s0)
            psi = init_state_node_uniform_ports(pm, s0, cdtype=coins.dtype, device=X.device)
        else:
            psi = init_state_node0_uniform_ports(pm, cdtype=coins.dtype, device=X.device)

        
        for t in range(T):
            psi = step(psi, pm, coins[t], shift=shift)
        P = partial_trace_coin(psi, pm).unsqueeze(0).to(dtype=torch.float32)
        return P, {"targets": targets}

    return rollout


def make_rollout_anydeg_exp_or_cayley(
    pm: PortMap,
    shift: ShiftOp,
    *,
    adaptor: ThetaToHermitianAdaptor,
    family: str,
    cdtype,
    init_mode: str = "node0",
    theta_scale: float = 1.0,
    theta_noise_std: float = 0.0,
) -> RolloutFn:


    from acpl.sim.utils import partial_trace_coin

    family = family.lower().strip()
    if family not in ("exp", "cayley"):
        raise ValueError("family must be 'exp' or 'cayley'")

    start = pm.node_ptr[:-1]
    end = pm.node_ptr[1:]

    # --- Fast path detection (closure constants for this pm) ---
    deg = (end - start).to(torch.int64)
    is_regular = bool(deg.numel() > 0 and (deg == deg[0]).all().item())
    d0 = int(deg[0].item()) if is_regular else -1
    fast_regular = bool(is_regular and d0 > 0 and int(pm.src.numel()) == int(pm.num_nodes * d0))


    def rollout(model: nn.Module, batch: dict):
        X: torch.Tensor = _maybe_squeeze_leading(batch["X"])
        edge_index: torch.Tensor = _maybe_squeeze_leading(batch["edge_index"])
        T: int = int(
            _maybe_squeeze_leading(batch["T"]).item()
            if isinstance(batch["T"], torch.Tensor)
            else batch["T"]
        )
        targets = batch.get("targets", None)
        if isinstance(targets, torch.Tensor) and targets.ndim > 1 and targets.size(0) == 1:
            targets = targets.squeeze(0)

        theta = model(X, edge_index, T=T)  # (T,N,3)


        if theta_scale != 1.0:
            theta = theta * float(theta_scale)
        if model.training and theta_noise_std > 0:
            theta = theta + float(theta_noise_std) * torch.randn_like(theta)




        if init_mode == "uniform":
            psi = init_state_uniform_arcs(pm, cdtype=cdtype, device=X.device)
        elif init_mode == "node":
            s0 = batch.get("start_node", 0)
            s0 = int(s0.item()) if isinstance(s0, torch.Tensor) else int(s0)
            psi = init_state_node_uniform_ports(pm, s0, cdtype=cdtype, device=X.device)
        else:
            psi = init_state_node0_uniform_ports(pm, cdtype=cdtype, device=X.device)

        
        
        # Put perm on the right device once (avoid per-step .to())
        perm = shift.perm
        if perm.device != psi.device:
            perm = perm.to(psi.device)

        if fast_regular:
            Nn = pm.num_nodes
            # theta[t] is (N,3) and adaptor will return (N,d0,d0)
            for t in range(T):
                H = adaptor.hermitian_from_theta(theta[t], d=d0)  # (Nn, d0, d0)
                U = unitary_exp_iH(H, cdtype) if family == "exp" else unitary_cayley(H, cdtype)  # (Nn,d0,d0)

                # psi is length A = Nn*d0, arranged in contiguous node blocks -> view as (Nn,d0)
                psi_v = psi.reshape(Nn, d0) # (Nn, d0)
                psi_v = torch.bmm(U, psi_v.unsqueeze(-1)).squeeze(-1)  # (Nn, d0)
                psi = psi_v.reshape(-1)  # (A,)

                psi = psi.index_select(0, perm)
        else:
            # Fallback: generic variable-degree path (your original code)
            for t in range(T):
                coins_v: list[torch.Tensor | None] = []
                for v in range(pm.num_nodes):
                    s, e = int(start[v]), int(end[v])
                    d = e - s
                    if d <= 0:
                        coins_v.append(None)
                        continue
                    H = adaptor.hermitian_from_theta(theta[t, v : v + 1, :], d=d)[0]
                    U = unitary_exp_iH(H, cdtype) if family == "exp" else unitary_cayley(H, cdtype)
                    coins_v.append(U)
                psi = apply_blockdiag_coins_anydeg(psi, pm, coins_v)
                psi = psi.index_select(0, perm)


        P = partial_trace_coin(psi, pm).unsqueeze(0).to(dtype=torch.float32)
        return P, {"targets": targets}

    return rollout


def make_transfer_loss(
    loss_kind: str = "nll",
    reduction: str = "mean",
    renorm: bool = True,
    eps: float = 1e-8,
    margin: float = 0.0,
):
    """
    Returns a callable: loss_builder(P, aux) -> loss

    P: (B, N) probability-like tensor (will be renormalized if renorm=True)
    aux: dict that must contain 'targets' (LongTensor indices of target nodes)
    """
    lk = str(loss_kind).lower().strip()
    red = str(reduction).lower().strip()

    def loss_builder(P: torch.Tensor, aux: dict, batch: dict | None = None) -> torch.Tensor:
        # ensure real + float
        if isinstance(P, torch.Tensor) and P.is_complex():
            P = P.real
        P = P.to(dtype=torch.float32)

        # renormalize into a simplex if requested
        if renorm:
            P = P / (P.sum(dim=-1, keepdim=True).clamp_min(1e-12))

        targets = aux.get("targets", None)
        if targets is None:
            raise ValueError("Transfer loss requires 'targets' in aux.")

        # normalize targets to a 1D LongTensor on same device
        if isinstance(targets, int):
            targets = torch.tensor([targets], device=P.device, dtype=torch.long)
        elif isinstance(targets, (list, tuple)):
            targets = torch.tensor(list(targets), device=P.device, dtype=torch.long)
        elif isinstance(targets, torch.Tensor):
            if targets.ndim == 0:
                targets = targets.view(1)
            targets = targets.to(device=P.device, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported targets type: {type(targets)}")

        # total prob mass on target set
        mass = P.index_select(dim=-1, index=targets).sum(dim=-1)  # (B,)

        if lk == "nll":
            # IMPORTANT: do not clamp_min before log; it kills gradients when mass < eps.
            loss = -torch.log(mass + eps)
        elif lk in ("neg_prob", "negprob", "prob", "neg_success", "neg_success_t", "neg_success_terminal"):
            loss = -mass
        elif lk == "hinge":
            if targets.numel() != 1:
                raise ValueError("hinge loss currently supports a single target.")
            t = int(targets[0].item())
            best_other = torch.cat([P[..., :t], P[..., t + 1 :]], dim=-1).max(dim=-1).values
            loss = torch.relu(float(margin) - (mass - best_other))
        else:
            raise ValueError(f"Unknown transfer loss '{loss_kind}'")

        if red == "mean":
            return loss.mean()
        if red == "sum":
            return loss.sum()
        if red == "none":
            return loss
        raise ValueError(f"Unknown reduction '{reduction}'")

    return loss_builder



def _mix_uniform(N: int, device, dtype=torch.float32) -> torch.Tensor:
    u = torch.full((N,), 1.0 / max(1, N), device=device, dtype=dtype)
    return u


def mixing_objective(P: torch.Tensor, kind: str = "tv", eps: float = 1e-12) -> torch.Tensor:
    """
    Return a per-batch objective value (B,).
    Lower is better for: tv, js, hell, l2, kl_pu
    Higher is better for: H (entropy)
    """
    kind = str(kind).lower().strip()
    if P.is_complex():
        P = P.real
    P = P.to(torch.float32)
    P = P / P.sum(dim=-1, keepdim=True).clamp_min(eps)

    B, N = P.shape
    U = _mix_uniform(N, device=P.device, dtype=P.dtype).expand(B, N)

    if kind in ("tv", "total_variation"):
        return 0.5 * (P - U).abs().sum(dim=-1)

    if kind in ("l2",):
        return torch.sqrt(((P - U) ** 2).sum(dim=-1).clamp_min(eps))

    if kind in ("hell", "hellinger"):
        # H^2 = 1 - sum sqrt(p_i u_i)
        bc = torch.sqrt(P.clamp_min(eps) * U).sum(dim=-1)
        return torch.sqrt((1.0 - bc).clamp_min(0.0))

    if kind in ("kl_pu", "kl"):
        # KL(P||U) = sum p log(p/u)
        return (P.clamp_min(eps) * (torch.log(P.clamp_min(eps)) - torch.log(U.clamp_min(eps)))).sum(dim=-1)

    if kind in ("js", "jensen_shannon"):
        M = 0.5 * (P + U)
        kl_pm = (P.clamp_min(eps) * (torch.log(P.clamp_min(eps)) - torch.log(M.clamp_min(eps)))).sum(dim=-1)
        kl_um = (U.clamp_min(eps) * (torch.log(U.clamp_min(eps)) - torch.log(M.clamp_min(eps)))).sum(dim=-1)
        return 0.5 * (kl_pm + kl_um)

    if kind in ("h", "entropy"):
        return -(P.clamp_min(eps) * torch.log(P.clamp_min(eps))).sum(dim=-1)

    raise ValueError(f"Unknown mixing objective kind '{kind}'")


def make_mixing_loss(kind: str = "tv", reduction: str = "mean", eps: float = 1e-12):
    """
    Loss for mixing:
      - for tv/js/hell/l2/kl_pu: minimize distance-to-uniform
      - for entropy: maximize entropy (so loss is -H)
    """
    kind = str(kind).lower().strip()
    reduction = str(reduction).lower().strip()

    def loss_builder(P: torch.Tensor, aux: dict, batch: dict | None = None) -> torch.Tensor:
        v = mixing_objective(P, kind=kind, eps=eps)  # (B,) for distances OR (B,) entropy
        if kind in ("h", "entropy"):
            # maximize entropy -> minimize -H
            v = -v

        if reduction == "mean":
            return v.mean()
        if reduction == "sum":
            return v.sum()
        if reduction == "none":
            return v
        raise ValueError(f"Unknown reduction '{reduction}'")

    return loss_builder



# -------------------------------
# Optimizer: param groups, EMA, schedulers
# -------------------------------


def _named_params_with_prefix(
    mod: nn.Module, prefix: str = ""
) -> Iterable[tuple[str, nn.Parameter]]:
    for n, p in mod.named_parameters():
        yield (f"{prefix}{n}", p)


def build_param_groups(
    base_lr: float,
    base_wd: float,
    named_params: list[tuple[str, nn.Parameter]],
    cfg: dict | None,
):
    """Apply regex rules to create param groups with lr multipliers and no-decay."""
    if not cfg or not cfg.get("enabled", True):
        # single group
        return [{"params": [p for _, p in named_params], "lr": base_lr, "weight_decay": base_wd}]

    rules = cfg.get("rules", [])
    groups: list[dict] = []

    def _match_any(name: str):
        for r in rules:
            pat = r.get("pattern", None)
            if not pat:
                continue
            if re.fullmatch(pat, name):
                return r
        return None

    for name, p in named_params:
        rule = _match_any(name)
        lr = base_lr
        wd = base_wd
        if rule is not None:
            if "lr_mult" in rule and rule["lr_mult"] is not None:
                lr = base_lr * float(rule["lr_mult"])
            if "weight_decay" in rule and rule["weight_decay"] is not None:
                wd = float(rule["weight_decay"])
        groups.append({"params": [p], "lr": lr, "weight_decay": wd})

    return groups


class EMAHelper:
    def __init__(
        self,
        model: nn.Module,
        decay: float,
        warmup_steps: int = 0,
        update_every: int = 1,
        pin_to_device: bool = True,
    ):
        self.decay = float(decay)
        self.warm = int(max(0, warmup_steps))
        self.every = int(max(1, update_every))
        self.step = 0
        self.shadow = {}
        self.pin = bool(pin_to_device)
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()
        self.device = next(model.parameters()).device

    def _decay_t(self) -> float:
        if self.step < self.warm:
            return float(self.step / max(1, self.warm)) * self.decay
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        if (self.step % self.every) != 0:
            return
        d = self._decay_t()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            buf = self.shadow.get(n, None)
            if buf is None:
                buf = p.detach().clone()
                self.shadow[n] = buf
            buf.mul_(d).add_(p.detach(), alpha=(1.0 - d))
        if self.pin:
            for k in list(self.shadow.keys()):
                self.shadow[k] = self.shadow[k].to(self.device)

    @torch.no_grad()
    def store(self, model: nn.Module):
        """Swap model weights -> EMA (save current to _tmp)."""
        self._tmp = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self._tmp[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if not hasattr(self, "_tmp"):
            return
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._tmp:
                p.data.copy_(self._tmp[n].data)
        del self._tmp


def build_optimizer_and_ema(model: nn.Module, adaptor: nn.Module | None, optim_cfg: dict):
    name = str(optim_cfg.get("name", "adam")).lower()
    base_lr = float(optim_cfg.get("lr", 3e-3))
    base_wd = float(optim_cfg.get("weight_decay", 0.0))
    betas = tuple(optim_cfg.get("betas", [0.9, 0.999]))
    eps = float(optim_cfg.get("eps", 1e-8))
    momentum = float(optim_cfg.get("momentum", 0.9))
    nesterov = bool(optim_cfg.get("nesterov", True))

    named_params: list[tuple[str, nn.Parameter]] = list(_named_params_with_prefix(model))
    if adaptor is not None:
        named_params += list(_named_params_with_prefix(adaptor, prefix="adaptor."))

    pg_cfg = optim_cfg.get("param_groups", {})
    groups = build_param_groups(base_lr, base_wd, named_params, pg_cfg)

    if name in ("adamw", "adamw_fused"):
        Optim = torch.optim.AdamW
        kw = {"betas": betas, "eps": eps}
    elif name == "adam":
        Optim = torch.optim.Adam
        kw = {"betas": betas, "eps": eps}
    elif name == "sgd":
        Optim = torch.optim.SGD
        kw = {"momentum": momentum, "nesterov": nesterov}
    else:
        raise ValueError(f"Unknown optimizer '{name}'")

    # Try fused optimizer on CUDA (PyTorch supports this for some opts/builds)
    use_fused = (next(model.parameters()).device.type == "cuda")
    try:
        optimizer = Optim(groups, lr=base_lr, weight_decay=base_wd, fused=use_fused, **kw)
    except TypeError:
        optimizer = Optim(groups, lr=base_lr, weight_decay=base_wd, **kw)

    ema_cfg = optim_cfg.get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema = None
    if ema_enabled:
        ema = EMAHelper(
            model,
            decay=float(ema_cfg.get("decay", 0.999)),
            warmup_steps=int(ema_cfg.get("warmup_steps", 100)),
            update_every=int(ema_cfg.get("update_every", 1)),
            pin_to_device=bool(ema_cfg.get("pin_to_device", True)),
        )
        orig_step = optimizer.step

        from types import MethodType

        def _step_with_ema(self, *args, **kwargs):
            out = orig_step(*args, **kwargs)
            ema.update(model)
            return out

        optimizer.step = MethodType(_step_with_ema, optimizer)  # type: ignore[assignment]

    return optimizer, ema


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: dict,
    *,
    steps_per_epoch: int | None,
    total_epochs: int | None,
):
    kind = str(sched_cfg.get("kind", "none")).lower()

    total_steps = None
    if steps_per_epoch is not None and total_epochs is not None:
        total_steps = max(1, steps_per_epoch * total_epochs)

    if kind == "none":
        return None

    if kind == "cosine":
        c = sched_cfg.get("cosine", {})
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(c.get("t_max", 1000)),
            eta_min=float(c.get("eta_min", 1e-5)),
        )

    if kind == "cosine_warmup":
        c = sched_cfg.get("cosine_warmup", {})
        warm = int(c.get("warmup_steps", 100))
        tmax = int(c.get("t_max", 1000))
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        sf = max(1e-8, float(c.get("start_factor", 0.0)))
        warmup = LinearLR(optimizer, start_factor=sf, end_factor=1.0, total_iters=warm)
        cosine = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=float(c.get("eta_min", 1e-5)))
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warm])

    if kind == "cosine_restart":
        c = sched_cfg.get("cosine_restart", {})
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(c.get("first_cycle_steps", 200)),
            T_mult=float(c.get("cycle_mult", 2.0)),
            eta_min=float(c.get("eta_min", 1e-5)),
        )

    if kind == "step":
        s = sched_cfg.get("step", {})
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(s.get("step_size", 200)),
            gamma=float(s.get("gamma", 0.5)),
        )

    if kind == "plateau":
        p = sched_cfg.get("plateau", {})
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(p.get("mode", "min")),
            factor=float(p.get("factor", 0.5)),
            patience=int(p.get("patience", 10)),
            threshold=float(p.get("threshold", 1e-3)),
            threshold_mode=str(p.get("threshold_mode", "rel")),
            cooldown=int(p.get("cooldown", 0)),
            min_lr=float(p.get("min_lr", 1e-6)),
        )

    if kind == "onecycle":
        o = sched_cfg.get("onecycle", {})
        if total_steps is None:
            raise ValueError(
                "onecycle scheduler requires total_steps; cannot infer from dataloader."
            )
        max_lr_mult = float(o.get("max_lr_mult", 1.0))
        max_lr = [pg["lr"] * max_lr_mult for pg in optimizer.param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=float(o.get("pct_start", 0.1)),
            anneal_strategy=str(o.get("anneal_strategy", "cos")),
            div_factor=float(o.get("div_factor", 25.0)),
            final_div_factor=float(o.get("final_div_factor", 10000.0)),
        )

    if kind == "linear_warmup_decay":
        l = sched_cfg.get("linear_warmup_decay", {})
        warm = int(l.get("warmup_steps", 100))
        tot = int(l.get("total_steps", total_steps or 1000))
        from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR

        sf = max(1e-8, float(l.get("start_factor", 0.0)))
        warmup = LinearLR(optimizer, start_factor=sf, end_factor=1.0, total_iters=warm)

        def decay_lambda(step):
            rem = max(1, tot - warm)
            t = max(0, step - warm)
            frac = max(0.0, 1.0 - t / rem)
            return frac

        decay = LambdaLR(optimizer, lr_lambda=decay_lambda)
        return SequentialLR(optimizer, [warmup, decay], milestones=[warm])

    raise ValueError(f"Unknown scheduler kind '{kind}'")


# -------------------------------
# Plot helper
# -------------------------------


def plot_series(xs: list[int], ys: list[float], *, out_png: Path, title: str):
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
    except Exception as e:
        print(f"[plot] matplotlib not available: {e}")
        return
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------------------------------
# Main
# -------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ACPL Training — Phases A/B with advanced optim + B8 utils"
    )
    parser.add_argument(
        "--config", type=str, default="acpl/configs/train.yaml", help="Top-level train config YAML"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--N", type=int, default=None, help="Override number of nodes (if applicable)"
    )
    parser.add_argument("--T", type=int, default=None, help="Override horizon (steps)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--run_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file or run dir")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    parser.add_argument(
        "--ci_only",
        action="store_true",
        help="Skip training; run the CI evaluation only (uses EMA weights if available). Requires --resume.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Shorthand for acpl/configs/experiments/<name>.yaml (e.g., transfer-line)",
    )

    args, unknown = parser.parse_known_args()

    # Resolve config path (supports --experiment OR a full --config path)
    if args.experiment:
        cfg_path = Path(f"acpl/configs/experiments/{args.experiment}.yaml")
    else:
        cfg_path = Path(args.config)

    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_yaml(cfg_path)

    # allow pre-registration experiment YAMLs (goal/graph/horizon/policy/...) to run
    cfg = _coerce_prereg_to_trainer_schema(cfg, cfg_path=cfg_path)

    # IMPORTANT: actually apply CLI overrides like coin.family=exp optim.lr=1e-3 ...
    cfg = apply_overrides(cfg, unknown)

    # Build a stable manifest hash for deterministic episode seeding (includes overrides)
    _manifest_blob = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    manifest_hex = hashlib.blake2b(_manifest_blob, digest_size=16).hexdigest()

    
    # ---------------- Task (define EARLY; used for horizon preference + model in_dim) ----------------
    task_cfg = cfg.get("task", {}) or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {"name": str(task_cfg)}

    task_name = str(task_cfg.get("name", cfg.get("goal", "transfer"))).lower().strip()
    is_search = ("search" in task_name)  # catches "search", "spatial_search", etc.

    is_mixing = ("mix" in task_name)  # matches "mixing", "mix", "mixing-grid", etc.
    is_robust = ("robust" in task_name)
    is_transfer_like = (not is_mixing)  # transfer + search + robust (transfer-style)



    # Search should start from uniform state unless overridden
    default_init = "uniform" if is_search else ("node" if ("robust" in task_name) else "node0")
    init_mode = str(task_cfg.get("init_state", default_init)).lower().strip()
    if init_mode not in ("uniform", "node0", "node"):
        raise ValueError(f"init_state must be 'uniform', 'node0', or 'node' (got {init_mode!r})")


    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    # ---------------- Coin family (resolve early; used in multiple places) ----------------
    coin_cfg = cfg.get("coin", {"family": "su2"}) or {}
    if not isinstance(coin_cfg, dict):
        # allow configs like: coin: exp
        coin_cfg = {"family": str(coin_cfg)}
    coin_family = str(coin_cfg.get("family", "su2")).lower().strip()
    # -------------------------------------------------------------------------------



    if any(
        ("logging.dir=" in x) or ("dataset.num_episodes=" in x) or ("train.max_steps=" in x)
        for x in unknown
    ):
        print(
            "[overrides] applied CLI aliases (logging.dir → train.run_dir, "
            "dataset.num_episodes/train.max_steps → train.episodes_per_epoch)"
        )

    # Runtime / deterministic seed (Phase B8)
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)
    device = select_device(cfg.get("device", "auto"))
    dtype = select_dtype(cfg.get("dtype", "float32"))
    
    cdtype = select_cdtype(cfg.get("state_dtype", "complex64"))

    if device.type == "cuda":
        # Set deterministic=True if you need strict reproducibility (slower).
        configure_cuda_perf(deterministic=False, capture_scalar_outputs=True)

    # ---------------- Outputs (create early so logger can write JSONL here) ----------------
    train_cfg = cfg.get("train", {})
    run_dir = Path(train_cfg.get("run_dir")) if train_cfg.get("run_dir") else None
    if run_dir is None:
        run_dir = (
            Path(args.run_dir)
            if args.run_dir
            else Path("runs")
            / f"B_{coin_family}_{cfg.get('data',{}).get('N', cfg.get('data',{}).get('num_nodes', 64))}nodes_{cfg.get('sim',{}).get('steps',64)}steps_seed{seed}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_path = run_dir / "pt_target.png"
    ckpt_last = run_dir / "model_last.pt"
    ckpt_best = run_dir / "model_best.pt"
    # ---------------------------------------------------------------------------------------

    # Graph/data
    data_cfg = cfg.get("data", {})
    if args.N is not None:
        data_cfg["N"] = int(args.N)
        data_cfg["num_nodes"] = int(args.N)
    with _time_block("build_graph"):
        g = graph_from_config(data_cfg, device=device, dtype=dtype)

    # Port map / shift
    with _time_block("build_shift"):
        pairs = list(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist(), strict=False))

        # --- FIX: ensure undirected graphs are represented with BOTH arc directions ---
        # If your graph builders already return both directions, set() will dedupe.
        data_cfg0 = cfg.get("data", {}) or {}
        directed = bool(data_cfg0.get("directed", False))  # allow override if you ever need directed
        if not directed:
            pairs = list(set(pairs) | set((v, u) for (u, v) in pairs))
            pairs.sort()  # deterministic

        pm = build_portmap(pairs, num_nodes=g.N, coalesce=True)



        shift = build_shift(pm)

    # Sim horizon / task
    sim_cfg = cfg.get("sim", {}) or {}
    if not isinstance(sim_cfg, dict):
        sim_cfg = {}

    # Accept int or list for curriculum; default to max if list
    steps_val = sim_cfg.get("steps", 64)

    if args.T is not None:
        T = int(args.T)
    else:
        if isinstance(steps_val, (list, tuple)):
            if len(steps_val) == 0:
                T = 64
            else:
                # Prefer task.horizon_T if provided; else fall back to max(sim.steps)
                pref = task_cfg.get("horizon_T", None)
                if pref is not None:
                    T = int(pref)
                    print(f"[setup] sim.steps candidates={list(steps_val)}; using task.horizon_T={T}.")
                else:
                    T = int(max(steps_val))
                    print(f"[setup] sim.steps provided as {list(steps_val)}; using T={T} for this run.")
        else:
            # also accept task.horizon_T when sim.steps is scalar-ish
            T = int(task_cfg.get("horizon_T", steps_val))

    # task may be a string like "transfer" from CLI; we only need defaults here
    task_cfg = cfg.get("task", {}) or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    target_index = int(task_cfg.get("target_index", g.N - 1))

    # Normalize target_index so -1 means "last node", and clamp within [0, N-1]
    if not (0 <= target_index < g.N):
        old = target_index
        target_index = int(target_index % g.N)
        print(f"[setup] normalized target_index {old} -> {target_index} (N={g.N})")


    # ---------------- Node features (NEW: FeatureSpec-driven) ----------------
    # Legacy behavior if cfg has no 'features:' block:
    #   X_base = [deg_norm(inv_sqrt), coord1]  -> (N,2)
    # New behavior if cfg.features exists:
    #   X_base = build_node_features(..., spec=FeatureSpec(...)) -> (N,F_base)

    spec = _feature_spec_from_cfg(cfg, seed=seed)

    # Degrees are derived from PortMap (outgoing ports per node), stable & task-agnostic
    deg_int = (pm.node_ptr[1:] - pm.node_ptr[:-1]).to(device=device, dtype=torch.long)  # (N,)

    if spec is None:
        # --- Legacy: exactly what you had (degree + 1D coord) ---
        deg_feat = normalize_degree(deg_int.to(torch.float32), mode="inv_sqrt").view(-1, 1)
        coord1 = _default_coords_1d(g.N, device=device, dtype=torch.float32)
        X_base = torch.cat([deg_feat, coord1], dim=1).to(device=device, dtype=dtype)  # (N,2)
        feat_index = {"deg_norm": (0, 1), "coord1": (1, 2)}
    else:
        # --- FeatureSpec path: consume coords if present, else fall back to coord1 ---
        if g.coords is not None and isinstance(g.coords, torch.Tensor) and g.coords.numel() > 0:
            coords = g.coords.to(device=device, dtype=torch.float32)
        else:
            coords = _default_coords_1d(g.N, device=device, dtype=torch.float32)

        Xb, idx = build_node_features(
            edge_index=g.edge_index.to(device),
            degrees=deg_int.to(device),
            coords=coords,
            spec=spec,
            device=device,
        )
        X_base = Xb.to(device=device, dtype=dtype)
        feat_index = idx

    # Decide whether we need an indicator channel:
    # - search/robust: indicator is appended per-episode by dataset (dynamic Ω)
    # - transfer (fixed Ω): we can append a static indicator ONCE here if enabled
    use_indicator = bool(task_cfg.get("use_indicator", (is_search or is_robust)))
    if is_mixing:
        use_indicator = False  # indicator is meaningless for pure mixing

    # For transfer-like (but not search/robust), allow static target indicator if enabled
    X_static = X_base
    transfer_targets_static = None
    if use_indicator and (not is_search) and (not is_robust) and (not is_mixing):
        target_radius = int(task_cfg.get("target_radius", 0))
        lo = max(0, target_index - target_radius)
        hi = min(g.N, target_index + target_radius + 1)
        transfer_targets_static = torch.arange(lo, hi, device=device, dtype=torch.long)

        tfeat = torch.zeros((g.N, 1), device=device, dtype=X_base.dtype)
        tfeat.index_fill_(0, transfer_targets_static, 1.0)
        X_static = torch.cat([X_base, tfeat], dim=1).to(device=device, dtype=dtype)

    # Compute the model input dimension correctly
    base_dim = int(X_base.shape[1])
    if use_indicator and (is_search or is_robust):
        in_dim = base_dim + 1  # dataset appends per-episode indicator
    elif use_indicator and (not is_mixing) and (transfer_targets_static is not None):
        in_dim = int(X_static.shape[1])  # we appended it once
    else:
        in_dim = base_dim

    # Persist feature metadata for reproducibility
    try:
        import json as _json
        meta = {
            "manifest_hex": manifest_hex,
            "use_indicator": bool(use_indicator),
            "base_dim": base_dim,
            "in_dim": int(in_dim),
            "feature_index": feat_index,
            "feature_spec": (asdict(spec) if spec is not None else None),
        }
        (run_dir / "features.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[warn] could not write features.json: {e}")
    # ------------------------------------------------------------------------


    # ---------------- Loss config (supports task.loss being dict OR string) ----------------
    task_loss_raw = task_cfg.get("loss", None)
    task_loss_cfg: dict[str, Any] = {}
    if isinstance(task_loss_raw, dict):
        task_loss_cfg = dict(task_loss_raw)
    elif isinstance(task_loss_raw, str):
        task_loss_cfg = {"kind": task_loss_raw}
    elif task_loss_raw is not None:
        raise TypeError(f"task.loss must be a dict or string (got {type(task_loss_raw)}).")

    loss_cfg = cfg.get("loss", {}) or {}
    if isinstance(loss_cfg, dict):
        loss_cfg = dict(loss_cfg)
    elif isinstance(loss_cfg, str):
        loss_cfg = {"kind": loss_cfg}
    else:
        loss_cfg = {}

    # Precedence: top-level loss overrides task.loss (so experiment-wide overrides win)
    loss_cfg = {**task_loss_cfg, **loss_cfg}

    # Now read everything ONLY from loss_cfg (no accidental dict->str)
    reduction = str(loss_cfg.get("reduction", "mean")).lower().strip()
    renorm = bool(loss_cfg.get("normalize_prob", True))
    loss_kind = str(loss_cfg.get("kind", "nll")).lower().strip()
    eps = float(loss_cfg.get("eps", 1e-8))
    margin = float(loss_cfg.get("margin", 0.0))

    # Used by metric pack + LoopConfig; safe default even if you don't use CVaR loss yet
    cvar_alpha = float(loss_cfg.get("cvar_alpha", 0.1))

    # Optional: store resolved loss back into cfg for logging/checkpoint reproducibility
    cfg["loss"] = dict(loss_cfg)
    # -------------------------------------------------------------------------------




    # Model config (each subkey may also be a string; coerce to dicts)
    model_cfg = cfg.get("model", {}) or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    gnn_cfg = model_cfg.get("gnn", {}) or {}
    if not isinstance(gnn_cfg, dict):
        # user might pass model.gnn=gcn; we don't use the string anywhere numeric, so ignore
        gnn_cfg = {}
    ctrl_cfg = model_cfg.get("controller", {}) or {}
    if not isinstance(ctrl_cfg, dict):
        ctrl_cfg = {}
    head_cfg = model_cfg.get("head", {}) or {}
    if not isinstance(head_cfg, dict):
        head_cfg = {}



    head_layernorm = bool(head_cfg.get("layernorm", True))
    # Workaround: Inductor/Triton can fail compiling LayerNorm on tiny feature dims (e.g., 3)
    if coin_family in ("exp", "cayley"):

        head_layernorm = False



    acpl_cfg = ACPLPolicyConfig(
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
    with _time_block("build_model"):
        model = ACPLPolicy(acpl_cfg).to(device=device)

        # -------------------------------
    # Optional: torch.compile (SAFE)
    # -------------------------------
    compile_cfg = cfg.get("compile", {}) or {}
    compile_enabled = bool(compile_cfg.get("enabled", False))
    compile_mode = str(compile_cfg.get("mode", "reduce-overhead"))
    requested_backend = str(compile_cfg.get("backend", "inductor")).lower().strip()

    # If true, user is explicitly asking to risk Inductor/Triton even on CUDA.
    force_inductor = bool(compile_cfg.get("force_inductor", False))

    # Build example inputs up front (used for warmup / validation)
    dummy_X = torch.zeros((g.N, acpl_cfg.in_dim), device=device, dtype=dtype)

    if compile_enabled:
        model_eager = model

        # Workaround: Inductor+Triton can hit "zuf0 is not defined" on some installs.
        # Default to aot_eager on CUDA unless force_inductor=true.
        backend = requested_backend
        if device.type == "cuda" and backend in ("inductor", "default") and not force_inductor:
            backend = "aot_eager"
            print("[setup] torch.compile: switching backend to aot_eager (safe mode; avoids Triton)")

        # If we *are* using Inductor, make it as synchronous as possible
        if backend in ("inductor", "default"):
            try:
                import torch._inductor.config as _ic  # type: ignore
                _ic.async_compile = False
                _ic.compile_threads = 1
            except Exception:
                pass

        try:
            compile_kwargs = {"backend": backend}
            # Only Inductor uses/accepts `mode` reliably; aot_eager may crash on some torch versions.
            if backend in ("inductor", "default"):
                compile_kwargs["mode"] = compile_mode

            compiled = torch.compile(model_eager, **compile_kwargs)

            # Warmup forces any compilation NOW (so we can fall back cleanly)
            with torch.no_grad():
                _ = compiled(dummy_X, g.edge_index, T=T)


            model = compiled  

            mode_str = compile_mode if backend in ("inductor", "default") else "<omitted>"
            print(f"[setup] torch.compile enabled for ACPLPolicy (backend={backend}, mode={mode_str})")

        except Exception as e:
            model = model_eager
            print(f"[setup] torch.compile disabled (failed; falling back to eager): {e}")
    else:
        print("[setup] torch.compile disabled (compile.enabled=false)")





    # Optim (advanced)
    optim_cfg = cfg.get("optim", {})
    if args.lr is not None:
        optim_cfg["lr"] = float(args.lr)

    adaptor = None
    if coin_family in ("exp", "cayley"):
        adaptor = ThetaToHermitianAdaptor(Kfreq=2).to(device)

    optimizer, ema = build_optimizer_and_ema(model, adaptor, optim_cfg)

    # Grad clip & AMP mapping
    grad_clip_cfg = optim_cfg.get("grad_clip", optim_cfg.get("grad_clip", None))
    if isinstance(grad_clip_cfg, (int, float)) or grad_clip_cfg is None:
        max_norm = None if (not grad_clip_cfg or grad_clip_cfg <= 0) else float(grad_clip_cfg)
    else:
        max_norm = None
        if grad_clip_cfg.get("enabled", True) and grad_clip_cfg.get("mode", "norm") == "norm":
            max_norm = float(grad_clip_cfg.get("max_norm", 1.0))

    precision_cfg = optim_cfg.get("precision", {})
    amp_flag = str(precision_cfg.get("amp", "auto")).lower()
    amp_enabled = amp_flag != "off"

    # Data loaders
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 1))
    batch_size = int(train_cfg.get("batch_size", 1))
    episodes_per_epoch = max(1, int(train_cfg.get("episodes_per_epoch", 200)))





    if batch_size != 1:
        raise ValueError(
            f"scripts/train.py currently assumes batch_size==1 (collate_fn returns xs[0]). Got batch_size={batch_size}."
        )


    if is_search:
        marks_per_episode = int(data_cfg.get("marks_per_episode", task_cfg.get("marks_per_episode", 1)))

        payload = {
            "X": X_base,  # base (N,F_base); dataset may append -> (N,F_base+1)
            "edge_index": g.edge_index,
            "T": int(T),
            "targets": None,
        }

        train_ds = MarkedGraphEpisodeDataset(
            payload,
            num_episodes=episodes_per_epoch,
            N=g.N,
            marks_per_episode=marks_per_episode,
            manifest_hex=manifest_hex,
            split="train",
            append_mark_indicator=use_indicator,
        )
        eval_ds = MarkedGraphEpisodeDataset(
            payload,
            num_episodes=max(1, episodes_per_epoch // 4),
            N=g.N,
            marks_per_episode=marks_per_episode,
            manifest_hex=manifest_hex,
            split="val",
            append_mark_indicator=use_indicator,
        )


    elif is_robust:
        # robust transfer-style: random Ω per episode, optional random start node
        targets_per_episode = int(task_cfg.get("targets_per_episode", 1))
        random_start = bool(task_cfg.get("random_start", True))
        append_indicator = bool(task_cfg.get("use_indicator", True))

        payload = {
            "X": X_base,  
            "edge_index": g.edge_index,
            "T": int(T),
            "targets": None,      # dataset sets this
            "start_node": 0,      # dataset may override
        }

        train_ds = RobustTargetEpisodeDataset(
            payload,
            num_episodes=episodes_per_epoch,
            N=g.N,
            targets_per_episode=targets_per_episode,
            manifest_hex=manifest_hex,
            split="train",
            append_target_indicator=append_indicator,
            random_start=random_start,
        )
        eval_ds = RobustTargetEpisodeDataset(
            payload,
            num_episodes=max(1, episodes_per_epoch // 4),
            N=g.N,
            targets_per_episode=targets_per_episode,
            manifest_hex=manifest_hex,
            split="val",
            append_target_indicator=append_indicator,
            random_start=random_start,
        )


    elif is_mixing:
        # mixing: no targets needed
        payload = {
            "X": X_base,  # (N,2) typically
            "edge_index": g.edge_index,
            "T": int(T),
        }
        train_ds = SingleGraphEpisodeDataset(payload, num_episodes=episodes_per_epoch)
        eval_ds = SingleGraphEpisodeDataset(payload, num_episodes=max(1, episodes_per_epoch // 4))

    else:
        # transfer: fixed target window
        target_radius = int(task_cfg.get("target_radius", 0))
        lo = max(0, target_index - target_radius)
        hi = min(g.N, target_index + target_radius + 1)
        targets = torch.arange(lo, hi, device=device, dtype=torch.long)

        payload = {
            "X": X_static if (use_indicator and transfer_targets_static is not None) else X_base,
            "edge_index": g.edge_index,
            "T": int(T),
            "targets": targets,
        }


        train_ds = SingleGraphEpisodeDataset(payload, num_episodes=episodes_per_epoch)
        eval_ds = SingleGraphEpisodeDataset(payload, num_episodes=max(1, episodes_per_epoch // 4))




    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda xs: xs[0]
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda xs: xs[0]
    )

    steps_per_epoch = math.ceil(len(train_ds) / max(1, batch_size))
    scheduler_cfg = optim_cfg.get("scheduler", {"kind": "none"})
    scheduler = build_scheduler(
        optimizer, scheduler_cfg, steps_per_epoch=steps_per_epoch, total_epochs=epochs
    )

    # Rollout + loss
    if coin_family == "su2":
        rollout_fn = make_rollout_su2_dv2(pm, shift, cdtype=cdtype, init_mode=init_mode)
        title_suffix = "SU2 (deg=2)"
    else:

        theta_scale = float(coin_cfg.get("theta_scale", 1.0))
        theta_noise_std = float(coin_cfg.get("theta_noise_std", 0.0))

        print(f"[setup] theta_scale={theta_scale} theta_noise_std={theta_noise_std}")


        rollout_fn = make_rollout_anydeg_exp_or_cayley(
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
    if is_mixing:
        mix_kind = str(loss_cfg.get("mix_kind", task_cfg.get("mix_kind", "tv"))).lower().strip()
        mix_eps = float(loss_cfg.get("mix_eps", task_cfg.get("mix_eps", 1e-12)))
        loss_builder = make_mixing_loss(
            kind=mix_kind,
            reduction=reduction,
            eps=mix_eps,
        )
    else:
        loss_builder = make_transfer_loss(
            loss_kind=loss_kind,
            reduction=reduction,
            renorm=renorm,
            eps=eps,
            margin=margin,
        )


    # Logging — write JSONL under run_dir (works with TB/W&B too)
    log_cfg = cfg.get("log", {})
    log_interval = int(log_cfg.get("interval", 50))
    backend_map = {
        "console": "plain",
        "tensorboard": "tensorboard",
        "wandb": "wandb",
        "plain": "plain",
    }
    backend = backend_map.get(str(log_cfg.get("backend", "plain")).lower(), "plain")
    wb_cfg = log_cfg.get("wandb") or {}
    project = wb_cfg.get("project", log_cfg.get("project"))
    run_name = wb_cfg.get("run_name", log_cfg.get("run_name"))

    ml_cfg = MetricLoggerConfig(
        backend=backend,
        log_dir=str(run_dir),  # << ensure metrics.jsonl goes to the run folder
        project=project,
        run_name=run_name,
        step_key="step",
        enable=True,
    )
    logger = MetricLogger(ml_cfg)

    loop_cfg = LoopConfig(
        device=str(device),
        log_every=log_interval,  # uses log.interval from config
        grad_clip=max_norm,
        cvar_alpha=float(cvar_alpha),
        primary_on_targets=(not is_mixing),
        progress_bar=True,
        amp=amp_enabled,
    )


    # Optional resume (B8 atomic-aware)
    start_epoch = 0
    best_metric = None
    if getattr(args, "resume", None):
        with _time_block("resume_load"):
            ckpt_payload = _CK.try_resume(Path(args.resume))
            if ckpt_payload:
                sd = ckpt_payload.get("state_dict") or ckpt_payload.get("model")
                if sd is not None:
                    try:
                        model.load_state_dict(sd, strict=False)
                    except Exception:
                        new_sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                        model.load_state_dict(new_sd, strict=False)

                ad_sd = ckpt_payload.get("adaptor", None)
                if adaptor is not None and isinstance(ad_sd, dict):
                    try:
                        adaptor.load_state_dict(ad_sd, strict=False)
                    except Exception:
                        pass


                opt_sd = ckpt_payload.get("optimizer", None)
                if isinstance(opt_sd, dict):
                    try:
                        optimizer.load_state_dict(opt_sd)
                    except Exception:
                        pass


                if "epoch" in ckpt_payload:
                    start_epoch = int(ckpt_payload["epoch"])
                best_metric = ckpt_payload.get("best_metric", None)

                ema_shadow = ckpt_payload.get("ema_shadow", None)
                if ema is not None and isinstance(ema_shadow, dict) and len(ema_shadow) > 0:
                    try:
                        # move to current device
                        ema.shadow = {
                            k: v.to(next(model.parameters()).device) for k, v in ema_shadow.items()
                        }
                        print(f"[resume] EMA shadow restored ({len(ema.shadow)} tensors).")
                    except Exception as e:
                        print(f"[resume] EMA shadow restore failed: {e}")

                print(f"[resume] Loaded checkpoint from {args.resume} @ epoch {start_epoch}")

    # ---------------- CI-ONLY MODE ----------------
    if args.ci_only:
        if not args.resume:
            print("[ci_only] --resume is required to load a checkpoint.", file=sys.stderr)
            try:
                logger.close()
            except Exception:
                pass
            sys.exit(2)

        print(
            "[ci_only] Running pooled CI evaluation on checkpointed weights "
            f"{'(EMA)' if ema is not None and len(getattr(ema, 'shadow', {}))>0 else '(raw)'}"
        )

        # eval config knobs (same defaults as training CI block)
        ci_n_seeds = int(train_cfg.get("ci_n_seeds", 5))
        ci_episodes = int(
            train_cfg.get("ci_episodes", max(1, train_cfg.get("episodes_per_epoch", 200) // 2))
        )
        ci_bootstrap = int((cfg.get("log", {}) or {}).get("ci_bootstrap_samples", 1000))
        ci_alpha = float((cfg.get("log", {}) or {}).get("ci_alpha", 0.05))

        # make a tiny iterable of dict-batches for each seed
        def _make_eval_iter(seed_i: int):
            # Use epoch=seed_i to get a deterministic, disjoint set per CI seed
            if is_search:
                ds = MarkedGraphEpisodeDataset(
                    payload,
                    num_episodes=ci_episodes,
                    N=g.N,
                    marks_per_episode=int(data_cfg.get("marks_per_episode", task_cfg.get("marks_per_episode", 1))),
                    manifest_hex=manifest_hex,
                    split="test",
                )
                ds.set_epoch(seed_i)
                return (ds[i] for i in range(len(ds)))

            if is_robust:
                ds = RobustTargetEpisodeDataset(
                    payload,
                    num_episodes=ci_episodes,
                    N=g.N,
                    targets_per_episode=int(task_cfg.get("targets_per_episode", 1)),
                    manifest_hex=manifest_hex,
                    split="test",
                    append_target_indicator=bool(task_cfg.get("use_indicator", True)),
                    random_start=bool(task_cfg.get("random_start", True)),
                )
                ds.set_epoch(seed_i)
                return (ds[i] for i in range(len(ds)))

            # transfer or mixing: deterministic episodes (unless you later add noise)
            ds = SingleGraphEpisodeDataset(payload, num_episodes=ci_episodes)
            return (ds[i] for i in range(len(ds)))

        eval_cfg = EvalConfig(
            seeds=[],
            n_seeds=ci_n_seeds,
            device=str(device),
            progress_bar=True,
            ci_method="bootstrap",
            ci_alpha=ci_alpha,
            bootstrap_samples=ci_bootstrap,
            keep_per_seed_means=False,
        )

        # Swap to EMA if we actually have a shadow
        used_ema = False
        if (
            ema is not None
            and isinstance(getattr(ema, "shadow", None), dict)
            and len(ema.shadow) > 0
        ):
            ema.store(model)
            used_ema = True
        try:
            results = run_ci_eval(
                model=model,
                dataloader_factory=_make_eval_iter,
                rollout_fn=rollout_fn,
                loop_cfg=loop_cfg,
                eval_cfg=eval_cfg,
                logger=logger,
                step=0,  # tag as step 0 in ci_only mode
            )
        finally:
            if used_ema:
                ema.restore(model)

        summary = summarize_results(
            results,
            title=(
                "CI over pooled episodes (EMA weights)" if used_ema else "CI over pooled episodes"
            ),
            show_per_seed=False,
            ci_alpha=eval_cfg.ci_alpha,
        )
        print("\n" + summary + "\n")

        # save artifacts
        (run_dir / "eval_ci.txt").write_text(summary + "\n", encoding="utf-8")
        import json as _json

        def _ci_to_dict(ci):
            return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "stderr": ci.stderr, "n": ci.n}

        json_payload = {k: _ci_to_dict(v["all"]) for k, v in results.items() if "all" in v}
        (run_dir / "eval_ci.json").write_text(_json.dumps(json_payload, indent=2), encoding="utf-8")
        print(f"[ci_only] wrote {run_dir/'eval_ci.txt'} and {run_dir/'eval_ci.json'}")

        try:
            logger.close()
        except Exception:
            pass
        return

    # Train/Eval
    step = 0 if start_epoch == 0 else start_epoch * steps_per_epoch
    pt_curve_x: list[int] = []
    pt_curve_y: list[float] = []

    print(
        f"[setup] device={device}, N={g.N}, T={T}, target={target_index}, family={coin_family}, epochs={epochs}"
    )
    print(
        f"[setup] optimizer={optimizer.__class__.__name__}, groups={len(optimizer.param_groups)}; grad_clip={max_norm}"
    )
    if scheduler is not None:
        print(f"[setup] scheduler={scheduler.__class__.__name__}")
    if ema is not None:
        print(f"[setup] EMA enabled: decay={ema.decay}, warmup_steps={ema.warm}, every={ema.every}")
    print(f"[setup] run_dir={run_dir} (metrics.jsonl, checkpoints, plots)")

    pbar_epochs = (
        _tqdm(range(start_epoch + 1, epochs + 1), desc="epochs")
        if _tqdm is not None
        else range(start_epoch + 1, epochs + 1)
    )
    for epoch in pbar_epochs:
        # If the dataset supports per-epoch deterministic sampling (search/robust),
        # update its internal epoch counter. Otherwise do nothing (transfer/mixing).
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        if hasattr(eval_ds, "set_epoch"):
            eval_ds.set_epoch(epoch)


        # --- per-epoch collector (no prints during batches) ---
        avg = EpochAverager(prefix="")  # keys we push already include 'train/...'
        metric_pack = None

        def _collect_hook(*, P, aux, loss, step, lr, **_):
            nonlocal metric_pack
            # lazily build the metric pack (matches loops.py semantics)
            if metric_pack is None:
                with_targets = ("targets" in aux) and (aux["targets"] is not None)
                metric_pack = build_metric_pack(
                    with_targets=with_targets,
                    cvar_alpha=float(task_cfg.get("cvar_alpha", 0.1)),
                )
            # base scalars
            m = {
                "train/loss": float(loss.detach().item()),
                "train/lr": float(lr),
            }

            # raw target probability mass (mean over batch) if targets exist
            tt = aux.get("targets", None)
            if tt is not None:
                with torch.no_grad():
                    if isinstance(tt, int):
                        tt = torch.tensor([tt], device=P.device)
                    elif isinstance(tt, (list, tuple)):
                        tt = torch.tensor(list(tt), device=P.device)
                    elif isinstance(tt, torch.Tensor):
                        tt = tt.to(device=P.device, dtype=torch.long)

                    mass = P.index_select(dim=-1, index=tt).sum(dim=-1).mean().item()

                m["train/target/mass"] = float(mass)
                m["train/target/log10_mass"] = float(math.log10(max(mass, 1e-300)))




            # metric groups: mix/* and (if targets) target/*
            for name, fn in metric_pack.items():
                scalars = fn(P.detach(), **aux)  # returns dict[str, float]
                for k, v in scalars.items():
                    m[f"train/{name}/{k}"] = float(v)
            avg.update(m)

        with _time_block(f"epoch_{epoch}_train"):
            step = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                logger=logger,
                loop_cfg=loop_cfg,
                rollout_fn=rollout_fn,
                loss_builder=loss_builder,
                step_start=step,
                scheduler=(
                    None
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    else scheduler
                ),
                hooks={"after_forward": _collect_hook},  # <- collect, don't print
            )

           # with torch.no_grad():
           #     s = 0.0
           #     for p in model.parameters():
           #         if p.requires_grad:
           #             s += float(p.abs().sum().item())
           #     print(f"[debug] |params|_1 = {s:.6f}")

           #  removed: debug param-norm print (kept training output clean)
        # ---- print one training summary line block before eval ----
        means = avg.means()
        try:
            lr_now = optimizer.param_groups[0]["lr"]
        except Exception:
            lr_now = float("nan")
        print(_fmt_epoch_summary(epoch_idx=epoch, means=means, lr=lr_now))

        # --------- Eval ---------
        # Eval (EMA swap if configured for eval-only averaging)
        with torch.no_grad():
            if ema is not None:
                ema.store(model)
            with _time_block(f"epoch_{epoch}_eval"):
                vals = []
                for batch in eval_loader:
                    P, aux = rollout_fn(model, batch)
                    P = P / (P.sum(dim=-1, keepdim=True).clamp_min(1e-12))

                    if is_mixing:
                        mix_kind = str(loss_cfg.get("mix_kind", task_cfg.get("mix_kind", "tv"))).lower().strip()
                        v = mixing_objective(P, kind=mix_kind, eps=1e-12)  # distances OR entropy
                        # unify: "higher is better" score
                        if mix_kind in ("h", "entropy"):
                            score = v
                        else:
                            score = -v
                        vals.extend(score.detach().cpu().tolist())
                    else:
                        tt = aux.get("targets", None)
                        if tt is None:
                            raise RuntimeError("Expected targets for transfer/search/robust eval, but aux['targets'] is None.")
                        if isinstance(tt, int):
                            tt = torch.tensor([tt], device=P.device)
                        elif isinstance(tt, (list, tuple)):
                            tt = torch.tensor(list(tt), device=P.device)
                        elif isinstance(tt, torch.Tensor):
                            tt = tt.to(device=P.device, dtype=torch.long)

                        pt = P.index_select(dim=-1, index=tt).sum(dim=-1)
                        vals.extend(pt.detach().cpu().tolist())

                primary_avg = float(sum(vals) / max(1, len(vals)))
                pt_curve_x.append(epoch)
                pt_curve_y.append(primary_avg)

            if ema is not None:
                ema.restore(model)

        # Plateau step if used
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            mode = scheduler_cfg.get("plateau", {}).get("mode", "min")
            metric = primary_avg if mode == "max" else -primary_avg
            scheduler.step(metric)

        # Print + plot + save (atomic)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            if is_mixing:
                mix_kind = str(loss_cfg.get("mix_kind", task_cfg.get("mix_kind", "tv"))).lower().strip()
                print(f"[epoch {epoch:04d}] mixing_score({mix_kind})={primary_avg:.4f} (higher is better)")
            else:
                print(f"[epoch {epoch:04d}] P_T[target]={primary_avg:.4f}")

           
           
           
            with _time_block("plot_update"):
                plot_series(
                    pt_curve_x, pt_curve_y, out_png=plot_path, title=f"P_T[target] — {title_suffix}"
                )

        
        raw_model = getattr(model, "_orig_mod", model)  # torch.compile wraps modules
        ckpt_state = {
            "state_dict": raw_model.state_dict(),
            "adaptor": (adaptor.state_dict() if adaptor is not None else None),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg,
            "manifest_hex": manifest_hex,
            "best_metric": best_metric if best_metric is not None else -1.0,
            "ema_shadow": (ema.shadow if ema is not None else None),
        }


        _CK.save_train_state(ckpt_last, **ckpt_state)


        # Optional: fail-fast if a checkpoint ever becomes unreadable.
        if bool(train_cfg.get("verify_checkpoints", True)):
            try:
                _ = torch.load(ckpt_last, map_location="cpu", weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Checkpoint verification failed for {ckpt_last}: {e}") from e




        if (best_metric is None) or (primary_avg > best_metric):
            best_metric = primary_avg
            ckpt_state["best_metric"] = best_metric
            _CK.atomic_save(ckpt_state, ckpt_best)

        if _tqdm is not None:
            try:
                lr_now = optimizer.param_groups[0]["lr"]
                pbar_epochs.set_postfix({"primary": f"{primary_avg:.3f}", "lr": f"{lr_now:.2e}"})
            except Exception:
                pass

    print(f"[done] final P_T[target]={pt_curve_y[-1]:.4f}")
    print(f"[artifacts] plot={plot_path}, ckpt_last={ckpt_last}, ckpt_best={ckpt_best}")

    # ---------------- CI EVALUATION (pooled across seeds; uses EMA weights if available) ----------------
    try:
        # CI knobs (override via train.yaml: train.ci_n_seeds / train.ci_episodes / log.ci_bootstrap_samples)
        ci_n_seeds = int(train_cfg.get("ci_n_seeds", 5))
        ci_episodes = int(train_cfg.get("ci_episodes", max(1, episodes_per_epoch // 2)))
        ci_bootstrap = int((cfg.get("log", {}) or {}).get("ci_bootstrap_samples", 1000))
        ci_alpha = float((cfg.get("log", {}) or {}).get("ci_alpha", 0.05))

        # Build an iterable of batches for a given seed
        def _make_eval_iter(seed_i: int):
            if is_search:
                ds = MarkedGraphEpisodeDataset(
                    payload,
                    num_episodes=ci_episodes,
                    N=g.N,
                    marks_per_episode=int(data_cfg.get("marks_per_episode", task_cfg.get("marks_per_episode", 1))),
                    manifest_hex=manifest_hex,
                    split="test",
                    append_mark_indicator=use_indicator,
                )
                ds.set_epoch(seed_i)
                return (ds[i] for i in range(len(ds)))

            if is_robust:
                ds = RobustTargetEpisodeDataset(
                    payload,
                    num_episodes=ci_episodes,
                    N=g.N,
                    targets_per_episode=int(task_cfg.get("targets_per_episode", 1)),
                    manifest_hex=manifest_hex,
                    split="test",
                    append_target_indicator=bool(task_cfg.get("use_indicator", True)),
                    random_start=bool(task_cfg.get("random_start", True)),
                )
                ds.set_epoch(seed_i)
                return (ds[i] for i in range(len(ds)))

            ds = SingleGraphEpisodeDataset(payload, num_episodes=ci_episodes)
            return (ds[i] for i in range(len(ds)))


        eval_cfg = EvalConfig(
            seeds=[],  # auto-generate [0..n_seeds-1]
            n_seeds=ci_n_seeds,
            device=str(device),
            progress_bar=True,
            ci_method="bootstrap",
            ci_alpha=ci_alpha,
            bootstrap_samples=ci_bootstrap,
            keep_per_seed_means=False,
        )

        # --- Run CI on EMA weights if available ---
        if ema is not None:
            ema.store(model)
        try:
            results = run_ci_eval(
                model=model,
                dataloader_factory=_make_eval_iter,
                rollout_fn=rollout_fn,
                loop_cfg=loop_cfg,
                eval_cfg=eval_cfg,
                logger=logger,  # logs eval_CI/* into metrics.jsonl (and TB/W&B if enabled)
                step=step,  # tag with the last global step
            )
        finally:
            if ema is not None:
                ema.restore(model)

        # Pretty print + save artifacts
        summary = summarize_results(
            results,
            title=(
                "Final CI over pooled episodes (EMA weights)"
                if ema is not None
                else "Final CI over pooled episodes"
            ),
            show_per_seed=False,
            ci_alpha=eval_cfg.ci_alpha,
        )
        print("\n" + summary + "\n")

        # Save .txt
        (run_dir / "eval_ci.txt").write_text(summary + "\n", encoding="utf-8")

        # Save .json (mean/lo/hi/stderr/n per key, pooled "all")
        import json as _json

        def _ci_to_dict(ci):
            return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "stderr": ci.stderr, "n": ci.n}

        json_payload = {k: _ci_to_dict(v["all"]) for k, v in results.items() if "all" in v}
        (run_dir / "eval_ci.json").write_text(_json.dumps(json_payload, indent=2), encoding="utf-8")

        print(f"[eval/ci] wrote {run_dir/'eval_ci.txt'} and {run_dir/'eval_ci.json'}")
    except Exception as e:
        print(f"[eval/ci] skipped due to error: {e}")

    # ensure logger flushes JSONL / TB / W&B
    try:
        logger.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

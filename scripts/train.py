# scripts/train.py
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

# -------------------------------
# Optional Phase-B8 utils (robust fallbacks if missing)
# -------------------------------


# Seeding (deterministic + flags)
def _seed_everything(seed: int) -> None:
    """
    Prefer acpl.utils.seeding (Phase B8) and fall back to a minimal implementation.
    """
    try:
        from acpl.utils import seeding as _seeding  # type: ignore

        # new API (Phase B8): seed_everything handles python/numpy/torch + cudnn flags
        if hasattr(_seeding, "seed_everything"):
            _seeding.seed_everything(seed=seed, deterministic=True, warn_only=True)
            return
        # older API variants
        if hasattr(_seeding, "set_deterministic"):
            _seeding.set_deterministic(seed)  # type: ignore
            return
    except Exception:
        pass

    # Fallback (minimal)
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as _np  # noqa: WPS433

        _np.random.seed(seed)
    except Exception:
        pass
    try:
        # conservative flags for determinism
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass


# Timers
class _NullTimer:
    def __enter__(self):  # noqa: D401
        """No-op timer (fallback)."""
        return self

    def __exit__(self, *exc):
        return False


def _time_block(label: str):
    """
    Prefer acpl.utils.timers.time_block context manager if available.
    """
    try:
        from acpl.utils.timers import time_block as _tb  # type: ignore

        return _tb(label)
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
        if self._ok and hasattr(self.ck, "atomic_save"):
            self.ck.atomic_save(obj, path)  # type: ignore
        else:
            torch.save(obj, path)

    def save_train_state(self, path: Path, **state) -> None:
        # Phase B8 API if present, else plain save
        if self._ok and hasattr(self.ck, "save_checkpoint"):
            self.ck.save_checkpoint(path=path, **state)  # type: ignore
        else:
            torch.save(state, path)

    def try_resume(self, path: Path) -> dict | None:
        """
        Return loaded dict if found, else None. Accepts either a file or a directory.
        """
        # Prefer Phase B8 'load_checkpoint_resume'
        if self._ok and hasattr(self.ck, "load_checkpoint_resume"):
            try:
                ok, payload = self.ck.load_checkpoint_resume(path)  # type: ignore
                return payload if ok else None
            except Exception:
                return None

        # Fallback: accept directories with 'model_last.pt'
        if path.is_dir():
            for name in ("model_last.pt", "last.pt", "checkpoint.pt"):
                cand = path / name
                if cand.exists():
                    path = cand
                    break

        if path.exists():
            try:
                return torch.load(path, map_location="cpu")
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
from acpl.train.loops import LoopConfig, RolloutFn, train_epoch
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
    for ov in overrides:
        if not ov.startswith("++") or "=" not in ov:
            continue
        key, val = ov[2:].split("=", 1)
        val = _coerce_scalar(val)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = val
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

    @staticmethod
    def _port_features(d: int, Kfreq: int, device=None, dtype=torch.float32) -> torch.Tensor:
        import math as _m

        j = torch.arange(d, device=device, dtype=dtype).unsqueeze(1)
        cols = [torch.ones(d, 1, device=device, dtype=dtype)]
        for k in range(1, Kfreq + 1):
            ang = 2.0 * _m.pi * k * j / max(d, 1)
            cols.append(torch.sin(ang))
            cols.append(torch.cos(ang))
        return torch.cat(cols, dim=1)  # (d, 2K+1)

    def _basis_matrices(self, d: int, device, dtype) -> torch.Tensor:
        Phi = self._port_features(d, self.Kfreq, device=device, dtype=dtype)  # (d,B)
        Ms = []
        for k in range(Phi.shape[-1]):
            phi = Phi[:, k : k + 1]
            Ms.append(phi @ phi.t())
        return torch.stack(Ms, dim=0)  # (B,d,d)

    def hermitian_from_theta(self, theta_vt: torch.Tensor, d: int) -> torch.Tensor:
        *batch, _ = theta_vt.shape
        w = self.proj(theta_vt.reshape(-1, 3)).view(*batch, self.B)  # (...,B)
        M = self._basis_matrices(d, device=theta_vt.device, dtype=theta_vt.dtype)  # (B,d,d)
        H = torch.tensordot(w, M, dims=([-1], [0]))
        H = 0.5 * (H + H.transpose(-1, -2))
        return H


def unitary_exp_iH(H: torch.Tensor, cdtype=torch.complex64) -> torch.Tensor:
    n = H.size(-1)
    Hs = 0.5 * (H + H.transpose(-1, -2))
    H64 = Hs.to(torch.float64)
    eps = 1e-8
    H64 = H64 + torch.eye(n, dtype=H64.dtype, device=H64.device) * eps
    try:
        evals, evecs = torch.linalg.eigh(H64)
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


def make_rollout_su2_dv2(pm: PortMap, shift: ShiftOp, *, cdtype) -> RolloutFn:
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
    family: str,  # "exp" | "cayley"
    cdtype,
) -> RolloutFn:
    from acpl.sim.utils import partial_trace_coin

    family = family.lower().strip()
    if family not in ("exp", "cayley"):
        raise ValueError("family must be 'exp' or 'cayley'")

    start = pm.node_ptr[:-1]
    end = pm.node_ptr[1:]

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
        psi = init_state_node0_uniform_ports(pm, cdtype=cdtype, device=X.device)

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
            psi = psi.index_select(0, shift.perm)

        P = partial_trace_coin(psi, pm).unsqueeze(0).to(dtype=torch.float32)
        return P, {"targets": targets}

    return rollout


def make_transfer_loss(reduction: str = "mean", renorm: bool = True):
    def loss_builder(P: torch.Tensor, aux: dict, batch: dict) -> torch.Tensor:
        if renorm:
            P = P / (P.sum(dim=-1, keepdim=True).clamp_min(1e-12))
        targets = aux.get("targets", None)
        if targets is None:
            raise ValueError("Transfer loss requires 'targets' in aux.")
        mass = P.index_select(dim=-1, index=targets).sum(dim=-1)
        if reduction == "mean":
            return -(mass.mean())
        elif reduction == "sum":
            return -(mass.sum())
        elif reduction == "none":
            return -(mass)
        else:
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

        def step_with_ema(*args, **kwargs):
            out = orig_step(*args, **kwargs)
            ema.update(model)
            return out

        optimizer.step = step_with_ema  # type: ignore[assignment]

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

        warmup = LinearLR(optimizer, start_factor=0.0, end_factor=1.0, total_iters=warm)
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

        warmup = LinearLR(optimizer, start_factor=0.0, end_factor=1.0, total_iters=warm)

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
    plt.ylabel("P_T[target]")
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
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint file/dir (B8 atomic)."
    )
    args, unknown = parser.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = load_yaml(cfg_path)
    cfg = apply_overrides(cfg, unknown)

    # Runtime / deterministic seed (Phase B8)
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)
    device = select_device(cfg.get("device", "auto"))
    dtype = select_dtype(cfg.get("dtype", "float32"))

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
        pm = build_portmap(pairs, num_nodes=g.N, coalesce=True)
        shift = build_shift(pm)

    # Sim horizon / task
    sim_cfg = cfg.get("sim", {})
    T = int(sim_cfg.get("steps", 64))
    if args.T is not None:
        T = int(args.T)
    task_cfg = cfg.get("task", {})
    target_index = int(task_cfg.get("target_index", g.N - 1))
    reduction = task_cfg.get("reduction", "mean")
    renorm = bool(task_cfg.get("renorm", True))

    # Model config
    model_cfg = cfg.get("model", {})
    gnn_cfg = model_cfg.get("gnn", {})
    ctrl_cfg = model_cfg.get("controller", {})
    head_cfg = model_cfg.get("head", {})
    acpl_cfg = ACPLPolicyConfig(
        in_dim=2,
        gnn_hidden=int(gnn_cfg.get("hidden", 64)),
        gnn_out=int(gnn_cfg.get("hidden", 64)),
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
        head_layernorm=True,
        head_dropout=float(head_cfg.get("dropout", 0.0)),
    )
    with _time_block("build_model"):
        model = ACPLPolicy(acpl_cfg).to(device=device)

    # Simple node features (degree + coord)
    N = g.N
    deg_feat = g.degrees.to(dtype=dtype).unsqueeze(1)  # (N,1)
    coord1 = torch.arange(N, device=device, dtype=dtype).unsqueeze(1) / max(N - 1, 1)
    X = torch.cat([deg_feat, coord1], dim=1)  # (N,2)

    # Coin family
    coin_cfg = cfg.get("coin", {"family": "su2"})
    family = str(coin_cfg.get("family", "su2")).lower().strip()

    # Optim (advanced)
    optim_cfg = cfg.get("optim", {})
    if args.lr is not None:
        optim_cfg["lr"] = float(args.lr)

    adaptor = None
    if family in ("exp", "cayley"):
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
    train_cfg = cfg.get("train", {})
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 1))
    batch_size = int(train_cfg.get("batch_size", 1))
    episodes_per_epoch = max(1, int(train_cfg.get("episodes_per_epoch", 200)))
    targets = torch.tensor([target_index], dtype=torch.long, device=device)

    payload = {
        "X": X,
        "edge_index": g.edge_index,
        "T": torch.tensor(T, dtype=torch.long, device=device),
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
    if family == "su2":
        rollout_fn = make_rollout_su2_dv2(pm, shift, cdtype=torch.complex64)
        title_suffix = "SU2 (deg=2)"
    else:
        rollout_fn = make_rollout_anydeg_exp_or_cayley(
            pm, shift, adaptor=adaptor, family=family, cdtype=torch.complex64
        )
        title_suffix = f"{family.upper()} (any degree)"
    loss_builder = make_transfer_loss(reduction=reduction, renorm=renorm)

    # Logging
    log_cfg = cfg.get("log", {})
    log_interval = int(log_cfg.get("interval", 50))
    backend_map = {
        "console": "plain",
        "tensorboard": "tensorboard",
        "wandb": "wandb",
        "plain": "plain",
    }
    backend = backend_map.get(str(log_cfg.get("backend", "plain")).lower(), "plain")
    tb_dir = (log_cfg.get("tensorboard") or {}).get("log_dir", None)
    wb_cfg = log_cfg.get("wandb") or {}
    project = wb_cfg.get("project", log_cfg.get("project"))
    run_name = wb_cfg.get("run_name", log_cfg.get("run_name"))

    ml_cfg = MetricLoggerConfig(
        backend=backend,
        log_dir=tb_dir,
        project=project,
        run_name=run_name,
        step_key="step",
        enable=True,
    )
    logger = MetricLogger(ml_cfg)

    loop_cfg = LoopConfig(
        device=str(device),
        log_every=log_interval,
        grad_clip=max_norm,
        cvar_alpha=float(cfg.get("task", {}).get("cvar_alpha", 0.1)),
        primary_on_targets=True,
        progress_bar=True,
        amp=amp_enabled,
    )

    # Outputs
    run_dir = Path(train_cfg.get("run_dir")) if train_cfg.get("run_dir") else None
    if run_dir is None:
        run_dir = (
            Path(args.run_dir)
            if args.run_dir
            else Path("runs") / f"B_{family}_{g.N}nodes_{T}steps_seed{seed}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_path = run_dir / "pt_target.png"
    ckpt_last = run_dir / "model_last.pt"
    ckpt_best = run_dir / "model_best.pt"

    # Optional resume (B8 atomic-aware)
    start_epoch = 0
    best_metric = None
    if args.resume:
        with _time_block("resume_load"):
            payload = _CK.try_resume(Path(args.resume))
            if payload:
                # Accept a few common layouts
                sd = payload.get("state_dict") or payload.get("model")
                if sd is not None:
                    try:
                        model.load_state_dict(sd, strict=False)
                    except Exception:
                        # remove 'module.' prefix if present
                        new_sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                        model.load_state_dict(new_sd, strict=False)
                if adaptor is not None and isinstance(payload.get("adaptor", None), dict):
                    try:
                        adaptor.load_state_dict(payload["adaptor"], strict=False)
                    except Exception:
                        pass
                opt_sd = payload.get("optimizer", None)
                if isinstance(opt_sd, dict):
                    try:
                        optimizer.load_state_dict(opt_sd)
                    except Exception:
                        pass
                if "epoch" in payload:
                    start_epoch = int(payload["epoch"])
                best_metric = payload.get("best_metric", None)
                print(f"[resume] Loaded checkpoint from {args.resume} @ epoch {start_epoch}")

    # Train/Eval
    step = 0 if start_epoch == 0 else start_epoch * steps_per_epoch
    pt_curve_x: list[int] = []
    pt_curve_y: list[float] = []

    print(
        f"[setup] device={device}, N={g.N}, T={T}, target={target_index}, family={family}, epochs={epochs}"
    )
    print(
        f"[setup] optimizer={optimizer.__class__.__name__}, groups={len(optimizer.param_groups)}; grad_clip={max_norm}"
    )
    if scheduler is not None:
        print(f"[setup] scheduler={scheduler.__class__.__name__}")
    if ema is not None:
        print(f"[setup] EMA enabled: decay={ema.decay}, warmup_steps={ema.warm}, every={ema.every}")

    pbar_epochs = (
        _tqdm(range(start_epoch + 1, epochs + 1), desc="epochs")
        if _tqdm is not None
        else range(start_epoch + 1, epochs + 1)
    )
    for epoch in pbar_epochs:
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
            )

        # Eval (EMA swap if configured for eval-only averaging)
        with torch.no_grad():
            if ema is not None:
                ema.store(model)
            with _time_block(f"epoch_{epoch}_eval"):
                vals = []
                for batch in eval_loader:
                    P, aux = rollout_fn(model, batch)
                    P = P / (P.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                    pt = P.index_select(dim=-1, index=aux["targets"]).sum(dim=-1)
                    vals.extend(pt.cpu().tolist())
                pt_avg = float(sum(vals) / max(1, len(vals)))
                pt_curve_x.append(epoch)
                pt_curve_y.append(pt_avg)
            if ema is not None:
                ema.restore(model)

        # Plateau step if used
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            mode = scheduler_cfg.get("plateau", {}).get("mode", "min")
            metric = pt_avg if mode == "max" else -pt_avg
            scheduler.step(metric)

        # Print + plot + save (atomic)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"[epoch {epoch:04d}] P_T[target]={pt_avg:.4f}")
            with _time_block("plot_update"):
                plot_series(
                    pt_curve_x, pt_curve_y, out_png=plot_path, title=f"P_T[target] — {title_suffix}"
                )

        # Save last every epoch; update best if improved
        payload = {
            "state_dict": model.state_dict(),
            "adaptor": (adaptor.state_dict() if adaptor is not None else None),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg,
            "best_metric": best_metric if best_metric is not None else -1.0,
        }
        _CK.save_train_state(ckpt_last, **payload)

        if (best_metric is None) or (pt_avg > best_metric):
            best_metric = pt_avg
            payload["best_metric"] = best_metric
            _CK.atomic_save(payload, ckpt_best)

        if _tqdm is not None:
            try:
                lr_now = optimizer.param_groups[0]["lr"]
                pbar_epochs.set_postfix({"P_T[target]": f"{pt_avg:.3f}", "lr": f"{lr_now:.2e}"})
            except Exception:
                pass

    print(f"[done] final P_T[target]={pt_curve_y[-1]:.4f}")
    print(f"[artifacts] plot={plot_path}, ckpt_last={ckpt_last}, ckpt_best={ckpt_best}")


if __name__ == "__main__":
    main()

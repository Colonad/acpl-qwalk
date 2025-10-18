# scripts/train.py
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

# Graphs: import module and resolve builders dynamically
from acpl.data import graphs as G

# Policy & loops
from acpl.policy.policy import ACPLPolicy, ACPLPolicyConfig

# Sim
from acpl.sim.portmap import PortMap, build_portmap
from acpl.sim.shift import ShiftOp, build_shift
from acpl.train.loops import LoopConfig, RolloutFn, train_epoch
from acpl.utils.logging import MetricLogger, MetricLoggerConfig

# add this with the other imports at the top
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


# -------------------------------
# Utilities: device / dtype / seeding
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


def set_seed(seed: int):
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
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
    """
    Return the first callable present in acpl.data.graphs whose name matches one of `names`.
    Raise a clear error if none found.
    """
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
        # Optional: treat gracefully if not present
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
        j = torch.arange(d, device=device, dtype=dtype).unsqueeze(1)
        cols = [torch.ones(d, 1, device=device, dtype=dtype)]
        for k in range(1, Kfreq + 1):
            ang = 2.0 * math.pi * k * j / max(d, 1)
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
        H = 0.5 * (H + H.transpose(-1, -2))  # <-- add this line
        return H


def unitary_exp_iH(H: torch.Tensor, cdtype=torch.complex64) -> torch.Tensor:
    # Enforce exact symmetry, promote to float64, add tiny jitter on the diag
    n = H.size(-1)
    Hs = 0.5 * (H + H.transpose(-1, -2))
    H64 = Hs.to(torch.float64)
    eps = 1e-8
    H64 = H64 + torch.eye(n, dtype=H64.dtype, device=H64.device) * eps
    try:
        evals, evecs = torch.linalg.eigh(H64)
        # Clamp spectrum (not strictly necessary, but avoids extreme angles)
        evals = torch.clamp(evals, min=-1e6, max=1e6)
        D = torch.diag_embed(torch.complex(torch.cos(evals), torch.sin(evals)).to(cdtype))
        Q = evecs.to(cdtype)
        return Q @ D @ Q.transpose(-1, -2).conj()
    except RuntimeError:
        # Fallback: stable matrix exponential
        return torch.matrix_exp(1j * Hs.to(cdtype))


def unitary_cayley(H: torch.Tensor, cdtype=torch.complex64) -> torch.Tensor:
    n = H.size(-1)
    Hs = 0.5 * (H + H.transpose(-1, -2))  # enforce symmetry
    I = torch.eye(n, dtype=cdtype, device=H.device)
    iH = 1j * Hs.to(cdtype)
    # Damping to avoid near-singularity of (I - iH)
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
        # Squeeze away a possible leading batch dimension added by DataLoader
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
        # Squeeze away a possible leading batch dimension added by DataLoader
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


def plot_series(xs: list[int], ys: list[float], *, out_png: Path, title: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available: {e}")
        return
    import matplotlib

    matplotlib.use("Agg")
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
        description="ACPL Training — Phase A (SU2) + Phase B (exp/Cayley)"
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
    args, unknown = parser.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = load_yaml(cfg_path)
    cfg = apply_overrides(cfg, unknown)

    # Runtime
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = select_device(cfg.get("device", "auto"))
    dtype = select_dtype(cfg.get("dtype", "float32"))

    # Graph/data
    data_cfg = cfg.get("data", {})
    if args.N is not None:
        data_cfg["N"] = int(args.N)
        data_cfg["num_nodes"] = int(args.N)
    g = graph_from_config(data_cfg, device=device, dtype=dtype)

    # Port map / shift
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
    model = ACPLPolicy(acpl_cfg).to(device=device)

    # Simple node features (degree + coord)
    N = g.N
    deg_feat = g.degrees.to(dtype=dtype).unsqueeze(1)  # (N,1)
    coord1 = torch.arange(N, device=device, dtype=dtype).unsqueeze(1) / max(N - 1, 1)
    X = torch.cat([deg_feat, coord1], dim=1)  # (N,2)

    # Coin family
    coin_cfg = cfg.get("coin", {"family": "su2"})
    family = str(coin_cfg.get("family", "su2")).lower().strip()

    # Optim
    optim_cfg = cfg.get("optim", {})
    lr = float(args.lr if args.lr is not None else optim_cfg.get("lr", 3e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    betas = tuple(optim_cfg.get("betas", [0.9, 0.999]))

    params = list(model.parameters())
    adaptor = None
    if family in ("exp", "cayley"):
        adaptor = ThetaToHermitianAdaptor(Kfreq=2).to(device)
        params += list(adaptor.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    grad_clip = optim_cfg.get("grad_clip", 1.0)
    if isinstance(grad_clip, (int, float)) and grad_clip <= 0:
        grad_clip = None

    # Scheduler
    scheduler_cfg = optim_cfg.get("scheduler", {"name": "none"})
    name = scheduler_cfg.get("name", "none")
    if name == "cosine":
        cosine = scheduler_cfg.get("cosine", {"t_max": 1000, "eta_min": 1e-5})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cosine.get("t_max", 1000)),
            eta_min=float(cosine.get("eta_min", 1e-5)),
        )
    elif name == "step":
        step_sched = scheduler_cfg.get("step", {"step_size": 200, "gamma": 0.5})
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(step_sched.get("step_size", 200)),
            gamma=float(step_sched.get("gamma", 0.5)),
        )
    else:
        scheduler = None

    # Loaders
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

    # Logging (backward-compatible)
    log_cfg = cfg.get("log", {})
    log_interval = int(log_cfg.get("interval", 50))

    # Map our YAML names to the logger's expectation
    backend_map = {
        "console": "plain",
        "tensorboard": "tensorboard",
        "wandb": "wandb",
        "plain": "plain",
    }
    backend = backend_map.get(str(log_cfg.get("backend", "plain")).lower(), "plain")

    tb_dir = (log_cfg.get("tensorboard") or {}).get("log_dir", None)
    wb_cfg = log_cfg.get("wandb") or {}
    project = wb_cfg.get("project", log_cfg.get("project"))  # allow top-level fallback
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
        grad_clip=grad_clip,
        cvar_alpha=float(cfg.get("task", {}).get("cvar_alpha", 0.1)),
        primary_on_targets=True,
        progress_bar=True,  # or False to disable
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
    ckpt_path = run_dir / "model_last.pt"

    # Train/Eval
    step = 0
    pt_curve_x: list[int] = []
    pt_curve_y: list[float] = []
    epoch_times: list[float] = []
    t_epoch_start = time.perf_counter()

    print(
        f"[setup] device={device}, N={g.N}, T={T}, target={target_index}, family={family}, epochs={epochs}"
    )
    print(f"[setup] optimizer=Adam(lr={lr}), grad_clip={grad_clip}")

    pbar_epochs = (
        _tqdm(range(1, epochs + 1), desc="epochs") if _tqdm is not None else range(1, epochs + 1)
    )
    for epoch in pbar_epochs:
        step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            logger=logger,
            loop_cfg=loop_cfg,
            rollout_fn=rollout_fn,
            loss_builder=loss_builder,
            step_start=step,
        )
        if scheduler is not None:
            scheduler.step()

        # Eval P_T[target]
        with torch.no_grad():
            vals = []
            for batch in eval_loader:
                P, aux = rollout_fn(model, batch)
                P = P / (P.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                pt = P.index_select(dim=-1, index=aux["targets"]).sum(dim=-1)
                vals.extend(pt.cpu().tolist())
            pt_avg = float(sum(vals) / max(1, len(vals)))
            pt_curve_x.append(epoch)
            pt_curve_y.append(pt_avg)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"[epoch {epoch:04d}] P_T[target]={pt_avg:.4f}")
            plot_series(
                pt_curve_x, pt_curve_y, out_png=plot_path, title=f"P_T[target] — {title_suffix}"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "adaptor": (adaptor.state_dict() if adaptor is not None else None),
                    "epoch": epoch,
                },
                ckpt_path,
            )

            # right after pt_avg is computed
        if _tqdm is not None:
            try:
                # show latest metric(s) and lr on the epoch bar
                lr_now = optimizer.param_groups[0]["lr"]
                pbar_epochs.set_postfix({"P_T[target]": f"{pt_avg:.3f}", "lr": f"{lr_now:.2e}"})
            except Exception:
                pass

    print(f"[done] final P_T[target]={pt_curve_y[-1]:.4f}")
    print(f"[artifacts] plot={plot_path}, ckpt={ckpt_path}")

    # --- timing / ETA ---
    t_now = time.perf_counter()
    epoch_time = t_now - t_epoch_start
    epoch_times.append(epoch_time)
    avg = sum(epoch_times[-5:]) / min(len(epoch_times), 5)  # rolling avg of last 5 epochs
    remaining = max(0, epochs - epoch) * avg

    def _fmt(sec):
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    print(f"[time] last_epoch={_fmt(epoch_time)}  avg={_fmt(avg)}  ETA={_fmt(remaining)}")
    t_epoch_start = time.perf_counter()


if __name__ == "__main__":
    main()

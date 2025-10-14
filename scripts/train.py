# scripts/train.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import optim

from acpl.data.features import node_features_line
from acpl.data.graphs import line_graph
from acpl.objectives.transfer import success_prob
from acpl.policy.policy import GNNTemporalPolicy
from acpl.sim.portmap import PortMap, build_portmap
from acpl.sim.utils import complex_dtype_for, portmap_tensors
from acpl.train.backprop import rollout_and_loss
from acpl.train.loops import TrainHooks, train_one_epoch

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _select_device(name: str) -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name in {"cuda", "gpu"}:
        return torch.device("cuda")
    return torch.device("cpu")


def _build_initial_state(pm: PortMap, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Phase-A default: put the walker at node 0, uniform over its outgoing ports,
    with global phase 0 and ||psi||_2 = 1.
    """
    pt = portmap_tensors(pm, device=device)
    start = int(pt.node_ptr[0].item())
    end = int(pt.node_ptr[1].item())
    deg0 = max(1, end - start)
    a = pm.num_arcs

    cdt = complex_dtype_for(dtype)
    psi = torch.zeros(a, device=device, dtype=cdt)
    amp = 1.0 / math.sqrt(float(deg0))
    psi[start:end] = torch.tensor(amp, device=device, dtype=cdt)
    return psi


def _ascii_bar(p: float, width: int = 40) -> str:
    ticks = max(0, min(width, int(round(p * width))))
    return "[" + "#" * ticks + "-" * (width - ticks) + "]"


# ---------------------------------------------------------------------
# Tiny in-memory “loader” for Phase A (single episode per batch)
# ---------------------------------------------------------------------


@dataclass
class _EpisodeBatch:
    x: torch.Tensor  # (N, F)
    psi0: torch.Tensor  # (A,) or (A,B)
    target: torch.Tensor | int  # scalar int or (B,)
    pos_enc: torch.Tensor | None  # (N, P) or None


def _make_loader(batch: _EpisodeBatch, batch_size: int, num_batches: int) -> Iterable[dict]:
    """
    Yields `num_batches` identical dicts. Good enough for Phase A single-episode runs.
    """

    def _to_dict() -> dict[str, torch.Tensor]:
        d: dict[str, torch.Tensor] = {
            "x": batch.x,
            "psi0": batch.psi0,
            "target": (
                batch.target
                if isinstance(batch.target, torch.Tensor)
                else torch.tensor(batch.target, device=batch.x.device, dtype=torch.long)
            ),
        }
        if batch.pos_enc is not None:
            d["pos_enc"] = batch.pos_enc
        return d

    for _ in range(max(1, num_batches)):
        # Ignore batch_size in Phase A; everything is single-episode.
        yield _to_dict()


# ---------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------


@hydra.main(config_path="../acpl/configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    print("=== acpl-qwalk: Phase-A training ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # ------------------ device / dtype ------------------
    device = _select_device(cfg.device)
    torch.set_default_dtype(torch.float32 if str(cfg.dtype) == "float32" else torch.float64)
    print(f"[device] {device}, cuda={torch.cuda.is_available()}")

    # ------------------ graph / features ------------------
    if cfg.data.graph != "line":
        raise ValueError("Phase A train.py only supports data.graph: line")
    n = int(cfg.data.num_nodes)
    edge_index_np, degrees_np, coords_np, _slices = line_graph(n, seed=int(cfg.seed))
    # torch edge_index
    edge_index = torch.from_numpy(edge_index_np).to(device=device, dtype=torch.long)
    # node features: degree + normalized coord -> (N,2)
    feats_np = node_features_line(degrees_np, coords_np, dtype=torch.float32)
    x = torch.from_numpy(feats_np).to(device=device)

    # ------------------ PortMap & initial state ------------------
    pm = build_portmap(edge_index_np)
    psi0 = _build_initial_state(pm, device=device, dtype=x.dtype)

    # ------------------ model (policy) ------------------
    fx = x.size(1)  # input feature dim
    gcfg = cfg.model.gnn
    ccfg = cfg.model.controller
    head_range = cfg.model.head.angle_range
    policy = GNNTemporalPolicy(
        in_dim=fx,
        gnn_hidden=int(gcfg.hidden),
        gnn_dropout=float(gcfg.dropout),
        controller_hidden=int(ccfg.hidden) if ccfg.hidden is not None else None,
        controller_layers=int(ccfg.layers),
        controller_dropout=float(ccfg.dropout),
        controller_bidirectional=bool(ccfg.bidirectional),
        head_angle_range=str(head_range),
        head_dropout=float(cfg.model.head.dropout),
        use_time_embed=bool(cfg.model.use_time_embed),
    ).to(device)

    # ------------------ optimizer / scheduler ------------------
    ocfg = cfg.optim
    if str(ocfg.name).lower() != "adam":
        raise ValueError("optim.name must be 'adam' in Phase A config")
    optimizer = optim.Adam(
        policy.parameters(),
        lr=float(ocfg.lr),
        weight_decay=float(ocfg.weight_decay),
        betas=tuple(ocfg.betas),
        eps=float(ocfg.eps),
    )
    scheduler = None  # keep simple for Phase A
    grad_clip = float(ocfg.grad_clip) if ocfg.grad_clip is not None else None

    # ------------------ data loader ------------------
    target_idx = int(cfg.task.target_index)
    batch = _EpisodeBatch(x=x, psi0=psi0, target=target_idx, pos_enc=None)
    loader = _make_loader(batch, batch_size=int(cfg.train.batch_size), num_batches=1)

    # ------------------ hooks (console logging) ------------------
    def _on_batch_end(loss: torch.Tensor, step_idx: int, info: dict[str, torch.Tensor]) -> None:
        if cfg.log.backend == "console" and (step_idx % int(cfg.log.interval) == 0):
            print(f"[batch {step_idx:03d}] loss={float(loss.item()):.6f}")

    def _on_epoch_end(metrics: dict[str, float]) -> None:
        if cfg.log.backend == "console":
            print(f"[epoch end] {metrics}")

    hooks = TrainHooks(on_batch_end=_on_batch_end, on_epoch_end=_on_epoch_end)

    # ------------------ train epochs ------------------
    steps = int(cfg.sim.steps)
    epochs = int(cfg.train.epochs)

    for epoch in range(epochs):
        train_one_epoch(
            policy=policy,
            pm=pm,
            edge_index=edge_index,
            data_loader=loader,
            optimizer=optimizer,
            steps=steps,
            device=device,
            grad_clip=grad_clip,
            scheduler=scheduler,
            reduction=str(cfg.task.reduction),
            record_traj=bool(cfg.train.record_traj),
            hooks=hooks,
        )

        # Every 10 epochs (and the last), print P[target]
        if (epoch % 10 == 0) or (epoch == epochs - 1):
            policy.eval()
            with torch.no_grad():
                loss_eval, info = rollout_and_loss(
                    policy=policy,
                    pm=pm,
                    edge_index=edge_index,
                    x=x,
                    pos_enc=None,
                    steps=steps,
                    psi0=psi0,
                    target_index=target_idx,
                    reduction="mean",
                    record_traj=False,
                )
                p_final = info["p_T"]  # (V,)
                p_target = success_prob(p_final, target_idx, check_prob=False, renorm=True)
                p_val = float(p_target.item()) if p_target.ndim == 0 else float(p_target.mean())
                bar = _ascii_bar(p_val)
                print(
                    f"[epoch {epoch:03d}] P[target={target_idx}] = {p_val:.4f} {bar} "
                    f"| loss={float(loss_eval.item()):.6f}"
                )

    print("Training complete.")


if __name__ == "__main__":
    main()

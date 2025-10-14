# acpl/train/loops.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

from acpl.policy.policy import GNNTemporalPolicy
from acpl.sim.portmap import PortMap

from .backprop import rollout_and_loss

# -----------------------------------------------------------------------------
# Hook types
# -----------------------------------------------------------------------------


@dataclass
class TrainHooks:
    """
    Optional callbacks fired during training.

    All callbacks are optional. Return values are ignored.
    - on_batch_end(loss, step_idx, info): called after each optimization step.
    - on_epoch_end(metrics): called once per epoch with aggregate metrics.
    """

    on_batch_end: Callable[[torch.Tensor, int, dict[str, torch.Tensor]], None] | None = None
    on_epoch_end: Callable[[dict[str, float]], None] | None = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _move_to_device(x: torch.Tensor | None, device: torch.device | None) -> torch.Tensor | None:
    if x is None or device is None:
        return x
    return x.to(device=device)


def _grad_total_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2).item()
        total += param_norm * param_norm
    return float(total**0.5)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train_one_epoch(
    policy: GNNTemporalPolicy,
    pm: PortMap,
    edge_index: torch.Tensor,
    data_loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    device: torch.device | None = None,
    grad_clip: float | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    reduction: str = "mean",
    record_traj: bool = False,
    hooks: TrainHooks | None = None,
) -> dict[str, float]:
    """
    One epoch over `data_loader`.

    Expected batch dict keys
    ------------------------
    - "x":       (N, F) node features (torch.float)
    - "psi0":    (A,) or (A,B) initial arc state (complex or real)
    - "target":  int (scalar) or (B,) tensor of vertex indices
    - "pos_enc": optional (N, P) extra node features (torch.float)

    Parameters
    ----------
    policy : GNNTemporalPolicy
        The composed policy (GCN -> GRU -> Head).
    pm : PortMap
        Port map for the simulator.
    edge_index : torch.Tensor
        Graph connectivity for the GCN, shape (2, E), dtype long.
    data_loader : iterable of dicts
        Each element is a batch dict as described above.
    optimizer : torch.optim.Optimizer
        Optimizer for the policy parameters.
    steps : int
        Number of DTQW steps per batch rollout.
    device : torch.device, optional
        If provided, move inputs and policy to this device for training.
    grad_clip : float, optional
        If provided, clip gradient norm (L2) to this value.
    scheduler : lr scheduler, optional
        Stepped once per batch (typical for simple schedulers).
    reduction : {"mean","sum","none"}
        Reduction applied in `rollout_and_loss` when batches are (A,B).
    record_traj : bool
        If True, `rollout_and_loss` also returns per-step trajectories.
    hooks : TrainHooks, optional
        Callbacks for batch/epoch logging.

    Returns
    -------
    dict[str, float]
        Aggregate metrics for the epoch: {"loss_mean", "loss_sum", "num_batches",
        "grad_norm_last"}.
    """
    if device is not None:
        policy.to(device)
        edge_index = edge_index.to(device=device, dtype=torch.long)

    policy.train()
    loss_sum = 0.0
    num_batches = 0
    grad_norm_last = 0.0

    for step_idx, batch in enumerate(data_loader):
        num_batches += 1
        x = _move_to_device(batch.get("x"), device)
        psi0 = _move_to_device(batch.get("psi0"), device)
        target = batch.get("target")
        pos_enc = _move_to_device(batch.get("pos_enc"), device)

        if x is None or psi0 is None or target is None:
            raise ValueError("batch must contain 'x', 'psi0', and 'target' keys.")

        # Forward + loss
        optimizer.zero_grad(set_to_none=True)
        loss, info = rollout_and_loss(
            policy=policy,
            pm=pm,
            edge_index=edge_index,
            x=x,  # type: ignore[arg-type]
            pos_enc=pos_enc,  # type: ignore[arg-type]
            steps=steps,
            psi0=psi0,  # type: ignore[arg-type]
            target_index=target,  # type: ignore[arg-type]
            reduction=reduction,
            record_traj=record_traj,
        )

        # Backward
        loss.backward()

        # Optional grad clipping
        if grad_clip is not None and grad_clip > 0.0:
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(grad_clip))

        grad_norm_last = _grad_total_norm(policy.parameters())

        # Step optimizer (+ scheduler if given)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_sum += float(loss.detach().item())

        # Hook
        if hooks is not None and hooks.on_batch_end is not None:
            hooks.on_batch_end(loss.detach(), step_idx, info)

    loss_mean = loss_sum / max(1, num_batches)
    metrics = {
        "loss_mean": loss_mean,
        "loss_sum": loss_sum,
        "num_batches": float(num_batches),
        "grad_norm_last": grad_norm_last,
    }
    if hooks is not None and hooks.on_epoch_end is not None:
        hooks.on_epoch_end(metrics)

    return metrics

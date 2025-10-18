# acpl/train/loops.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from acpl.objectives.metrics import (
    cvar,
    mixing_summary,
    success_on_targets,
    targeting_summary,
)
from acpl.utils.logging import MetricLogger

try:
    from tqdm.auto import tqdm
except Exception:
    # no-op fallback if tqdm isn't installed
    def tqdm(x, **kwargs):
        return x


__all__ = [
    "LoopConfig",
    "build_metric_pack",
    "log_metric_pack",
    "train_epoch",
    "eval_epoch",
]


try:
    from tqdm.auto import tqdm
except Exception:
    # no-op fallback if tqdm isn't installed
    def tqdm(x, **kwargs):
        return x


# --------------------------------------------------------------------------------------
#                                       Config
# --------------------------------------------------------------------------------------


@dataclass
class LoopConfig:
    device: str = "cuda"
    log_every: int = 50

    # Optimization niceties
    grad_clip: float | None = None  # e.g., 1.0
    accum_steps: int = 1  # gradient accumulation; 1 = disabled
    amp: bool = False  # use torch.cuda.amp for forward/backward
    log_grad_norm_every: int = 0  # if >0, log grad-norm every k steps (cheap)

    # Optional scheduler stepping
    scheduler_on: bool = False  # if True, call scheduler.step() each optimizer step

    # Risk/diagnostics
    cvar_alpha: float = 0.1  # CVaR parameter for diagnostics

    # When targets exist, treat success as primary scalar
    primary_on_targets: bool = True

    # UI
    progress_bar: bool = True


# --------------------------------------------------------------------------------------
#                                 Metric pack helpers
# --------------------------------------------------------------------------------------


def build_metric_pack(
    *,
    with_targets: bool,
    cvar_alpha: float = 0.1,
):
    """
    Returns a dict of callables f(P, **aux) -> Dict[str, float], run under no_grad.

    Expected aux keys (optional):
      - "targets": LongTensor (K,) indices
    """

    def pack_mixing(P: torch.Tensor, **aux) -> dict[str, float]:
        m = mixing_summary(P)  # TV/JS/Hellinger/L2/Entropy/KL(P||U)
        return {
            "tv": float(m.tv),
            "js": float(m.js),
            "hell": float(m.hellinger),
            "l2": float(m.l2),
            "H": float(m.entropy_p),
            "kl_pu": float(m.kl_pu),
        }

    def pack_targeting(P: torch.Tensor, **aux) -> dict[str, float]:
        targets = aux.get("targets", None)
        if targets is None:
            return {}
        m = targeting_summary(P, targets=targets)
        # success is the main signal; add CVaR(-log P[Ω]) optionally
        with torch.no_grad():
            p_omega = success_on_targets(P, targets, reduction="none")  # (batch,) or scalar
            p_omega = torch.clamp(p_omega, min=1e-8)
            neglog = (-p_omega.log()).reshape(-1)
            cv = float(cvar(neglog, alpha=cvar_alpha).item())
        return {
            "success": float(m.success),
            "maxp": float(m.maxp),
            "gini": float(m.gini),
            "tv_vsU": float(m.tv),
            "js_vsU": float(m.js),
            "H": float(m.entropy_p),
            "KLpU": float(m.kl_pu),
            "cvar_neglogOmega": cv,
        }

    if with_targets:
        return {"mix": pack_mixing, "target": pack_targeting}
    else:
        return {"mix": pack_mixing}


@torch.no_grad()
def log_metric_pack(
    logger: MetricLogger,
    pack: dict[str, Callable],
    *,
    P: torch.Tensor,
    step: int,
    prefix: str,
    **aux,
):
    for name, fn in pack.items():
        scalars = fn(P, **aux)
        if scalars:
            logger.log_dict(f"{prefix}{name}/", scalars, step=step)


# --------------------------------------------------------------------------------------
#                          Generic train/eval epoch loops
# --------------------------------------------------------------------------------------

# Contracts:
#   rollout_fn:  P, aux = rollout_fn(model, batch)
#       P   : (B, N) or (N,)
#       aux : arbitrary dict (may include "targets" and/or "loss")
#
#   loss_builder (optional): loss = loss_builder(P, aux, batch)
RolloutFn = Callable[[nn.Module, dict], tuple[torch.Tensor, dict]]
LossBuilder = Callable[[torch.Tensor, dict, dict], torch.Tensor]


def _to_device_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _is_finite(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item() if isinstance(x, torch.Tensor) else True


def _grad_norm(parameters, norm_type: float = 2.0) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total += float(param_norm.item() ** 2)
    return total**0.5


def train_epoch(
    model: nn.Module,
    dataloader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    logger: MetricLogger,
    loop_cfg: LoopConfig,
    rollout_fn: RolloutFn,
    loss_builder: LossBuilder | None = None,
    *,
    step_start: int = 0,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> int:
    """
    One training epoch with:
      • optional AMP (mixed precision)
      • optional grad accumulation
      • optional grad clipping
      • periodic metric logging (mixing, targeting if targets exist)
      • optional scheduler stepping

    Returns next global step.
    """
    model.train()
    device = torch.device(loop_cfg.device)
    _device_type = device.type  # "cuda" | "cpu" | etc.
    step = step_start

    use_amp = bool(loop_cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(_device_type, enabled=use_amp)

    metric_pack = None
    accum_steps = max(1, int(loop_cfg.accum_steps))
    assert accum_steps >= 1

    optimizer.zero_grad(set_to_none=True)

    iterator = dataloader
    if getattr(loop_cfg, "progress_bar", True):
        total = len(dataloader) if hasattr(dataloader, "__len__") else None
        iterator = tqdm(dataloader, total=total, desc="train", leave=False)

    for bidx, batch in enumerate(iterator, start=1):
        step += 1
        batch = _to_device_batch(batch, device)

        with torch.amp.autocast(device_type=_device_type, enabled=use_amp):
            P, aux = rollout_fn(model, batch)  # P: (B,N) or (N,)
            if isinstance(P, torch.Tensor) and P.ndim == 1:
                P = P.unsqueeze(0)

            # Lazily init metric pack based on presence of targets
            if metric_pack is None:
                with_targets = ("targets" in aux) and (aux["targets"] is not None)
                metric_pack = build_metric_pack(
                    with_targets=with_targets, cvar_alpha=loop_cfg.cvar_alpha
                )

            if "loss" in aux and aux["loss"] is not None:
                loss = aux["loss"]
            else:
                if loss_builder is None:
                    raise ValueError("No loss provided by rollout_fn and no loss_builder supplied.")
                loss = loss_builder(P, aux, batch)

            if not _is_finite(loss):
                logger.log_text(
                    "train/warn",
                    f"Non-finite loss detected at step {step}. Skipping update.",
                    step=step,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            # Normalize if accumulating
            loss_scaled_for_accum = loss / float(accum_steps)

        # Backward (AMP-aware)
        if use_amp:
            scaler.scale(loss_scaled_for_accum).backward()
        else:
            loss_scaled_for_accum.backward()

        # Optimizer step every accum_steps
        do_step = bidx % accum_steps == 0
        if do_step:
            # Grad clip (AMP-safe)
            if loop_cfg.grad_clip is not None and loop_cfg.grad_clip > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), loop_cfg.grad_clip)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if loop_cfg.scheduler_on and scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % loop_cfg.log_every == 0:
            logger.log_scalar("train/loss", float(loss.detach().item()), step=step)
            log_metric_pack(logger, metric_pack, P=P.detach(), step=step, prefix="train/", **aux)

            if loop_cfg.log_grad_norm_every and (step % loop_cfg.log_grad_norm_every == 0):
                try:
                    gnorm = _grad_norm(model.parameters())
                    logger.log_scalar("train/grad_norm", gnorm, step=step)
                except Exception:
                    pass  # best-effort; don't break training

    return step


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader: Iterable[dict],
    logger: MetricLogger,
    loop_cfg: LoopConfig,
    rollout_fn: RolloutFn,
    *,
    step: int,
) -> None:
    """
    Evaluation epoch:
      • averages loss if aux["loss"] is provided by rollout
      • aggregates metrics over batches and logs once
    """
    model.eval()
    device = torch.device(loop_cfg.device)

    metric_pack = None
    loss_acc = 0.0
    loss_count = 0

    # Accumulate simple averages
    agg: dict[str, float] = {}
    n_batches = 0

    def _merge(prefix: str, d: dict[str, float]):
        for k, v in d.items():
            key = f"{prefix}{k}"
            agg[key] = agg.get(key, 0.0) + float(v)

    iterator = dataloader
    if getattr(loop_cfg, "progress_bar", True):
        total = len(dataloader) if hasattr(dataloader, "__len__") else None
        iterator = tqdm(dataloader, total=total, desc="eval", leave=False)

    for batch in iterator:

        n_batches += 1
        batch = _to_device_batch(batch, device)
        P, aux = rollout_fn(model, batch)
        if isinstance(P, torch.Tensor) and P.ndim == 1:
            P = P.unsqueeze(0)

        if metric_pack is None:
            with_targets = ("targets" in aux) and (aux["targets"] is not None)
            metric_pack = build_metric_pack(
                with_targets=with_targets, cvar_alpha=loop_cfg.cvar_alpha
            )

        # Per-batch metrics
        for name, fn in metric_pack.items():
            scalars = fn(P, **aux)
            _merge(f"{name}/", scalars)

        if "loss" in aux and aux["loss"] is not None:
            loss_acc += float(aux["loss"].detach().item())
            loss_count += 1

    if n_batches == 0:
        return

    # Averages
    for k in list(agg.keys()):
        agg[k] /= float(n_batches)

    if loss_count > 0:
        logger.log_scalar("eval/loss", loss_acc / max(1, loss_count), step=step)

    logger.log_dict("eval/", agg, step=step)

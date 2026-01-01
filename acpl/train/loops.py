# acpl/train/loops.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import random
import time
from typing import Any

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
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x


__all__ = [
    # Config / state
    "LoopConfig",
    "CheckpointConfig",
    "LoopState",
    # Metrics helpers
    "build_metric_pack",
    "log_metric_pack",
    # Epoch loops
    "train_epoch",
    "eval_epoch",
    # Orchestrator
    "fit",
    # Checkpoint utilities
    "save_checkpoint",
    "load_checkpoint",
    "best_is_better",
]


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

    # Orchestration (fit)
    epochs: int = 1
    eval_every: int = 1  # run eval every k train epochs

    # Early stopping
    early_stop_patience: int | None = None
    best_key: str = "eval/target/success"  # metric to track for early stop & best ckpt
    best_mode: str = "max"  # "max" or "min"


@dataclass
class CheckpointConfig:
    # Where / when
    dir: str | None = None
    save_every_steps: int | None = None  # save snapshot every K steps
    save_every_epochs: int | None = 1  # save snapshot every K epochs
    keep_last_k: int = 3  # rolling window
    filename: str = "ckpt_step{step}_epoch{epoch}.pt"
    best_filename: str = "best.pt"
    resume: bool = False  # auto-resume from latest in dir
    include_rng: bool = True  # save/restore RNG state (torch, cuda, random, numpy)
    strict_model_load: bool = True  # load state_dict strictly


@dataclass
class LoopState:
    # Global counters
    step: int = 0
    epoch: int = 0
    best_metric: float | None = None
    best_path: str | None = None
    # Housekeeping
    wall_start: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "best_path": self.best_path,
        }


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


def _lr_of(optimizer: torch.optim.Optimizer) -> float:
    try:
        return float(optimizer.param_groups[0]["lr"])
    except Exception:
        return float("nan")


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
    hooks: dict[str, Callable] | None = None,  # optional callbacks (see below)
) -> int:
    """
    One training epoch with:
      • optional AMP (mixed precision)
      • optional grad accumulation
      • optional grad clipping
      • periodic metric logging (mixing, targeting if targets exist)
      • optional scheduler stepping

    Hooks (all optional), each signature hook(**ctx):
      - "before_batch": called right after moving batch to device
      - "after_forward": called with {"P": P, "aux": aux, "loss": loss, "step": step}
      - "after_backward": called after backward (gradients ready)
      - "after_step": called after optimizer step (or skipped if accumulating)
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

    total = len(dataloader) if hasattr(dataloader, "__len__") else None
    iterator = tqdm(
        dataloader,
        total=total,
        desc="train",
        leave=False,
        disable=not loop_cfg.progress_bar,  # <- respect LoopConfig.progress_bar
    )

    for bidx, batch in enumerate(iterator, start=1):
        step += 1
        batch = _to_device_batch(batch, device)


        # Optional: fill missing disorder metadata for deterministic sampling.
        # Does NOT override existing keys.
        d = batch.get("disorder", None)
        if isinstance(d, dict):
            d.setdefault("episode_index", int(step))
            d.setdefault("trial_id", 0)
            d.setdefault("seed", int(d.get("seed", 0)))




        if hooks and "before_batch" in hooks:
            try:
                hooks["before_batch"](batch=batch, step=step)
            except Exception:
                pass

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

        if hooks and "after_forward" in hooks:
            try:
                hooks["after_forward"](P=P, aux=aux, loss=loss, step=step, lr=_lr_of(optimizer))
            except Exception:
                pass

        # Backward (AMP-aware)
        if use_amp:
            scaler.scale(loss_scaled_for_accum).backward()
        else:
            loss_scaled_for_accum.backward()

        if hooks and "after_backward" in hooks:
            try:
                hooks["after_backward"](step=step)
            except Exception:
                pass

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

            if hooks and "after_step" in hooks:
                try:
                    hooks["after_step"](step=step, lr=_lr_of(optimizer))
                except Exception:
                    pass

        # Logging
        if step % loop_cfg.log_every == 0:
            logger.log_scalar("train/loss", float(loss.detach().item()), step=step)
            logger.log_scalar("train/lr", _lr_of(optimizer), step=step)
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
    hooks: dict[str, Callable] | None = None,
) -> dict[str, float]:
    """
    Evaluation epoch:
      • averages loss if aux["loss"] is provided by rollout
      • aggregates metrics over batches and logs once

    Returns a dict of aggregated scalars that includes the primary metric
    chosen via LoopConfig (useful for early stopping).
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

    total = len(dataloader) if hasattr(dataloader, "__len__") else None
    iterator = tqdm(
        dataloader,
        total=total,
        desc="eval",
        leave=False,
        disable=not loop_cfg.progress_bar,  # <- respect LoopConfig.progress_bar
    )

    for batch in iterator:
        n_batches += 1
        batch = _to_device_batch(batch, device)



        # Optional: fill missing disorder metadata for deterministic sampling.
        d = batch.get("disorder", None)
        if isinstance(d, dict):
            # `step` is constant during eval_epoch, so use batch index as episode_index fallback.
            d.setdefault("episode_index", int(n_batches))  # n_batches already incremented
            d.setdefault("trial_id", 0)
            d.setdefault("seed", int(d.get("seed", 0)))



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
        return {}

    # Averages
    for k in list(agg.keys()):
        agg[k] /= float(n_batches)

    if loss_count > 0:
        agg["eval/loss"] = loss_acc / max(1, loss_count)
        logger.log_scalar("eval/loss", agg["eval/loss"], step=step)

    logger.log_dict("eval/", agg, step=step)

    if hooks and "after_eval" in hooks:
        try:
            hooks["after_eval"](step=step, scalars=agg)
        except Exception:
            pass

    return agg


# --------------------------------------------------------------------------------------
#                                      Checkpoints
# --------------------------------------------------------------------------------------


def _rng_state_pack(include_cuda: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    try:
        import numpy as _np  # type: ignore

        out["numpy"] = _np.random.get_state()
    except Exception:
        pass
    if include_cuda and torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def _rng_state_load(state: dict[str, Any]) -> None:
    if not state:
        return
    try:
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            import numpy as _np  # type: ignore

            _np.random.set_state(state["numpy"])
    except Exception:
        # Never break training on RNG restore
        pass


def save_checkpoint(
    *,
    path: str | os.PathLike,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    loop_state: LoopState | None = None,
    loop_cfg: LoopConfig | None = None,
    ckpt_cfg: CheckpointConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> str:


    """Serialize training state to `path`."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "loop_state": loop_state.to_dict() if loop_state is not None else {},
        "loop_cfg": asdict(loop_cfg) if loop_cfg is not None else {},
        "checkpoint_cfg": asdict(ckpt_cfg) if ckpt_cfg is not None else {},
        "extra": extra or {},
        "torch_version": torch.__version__,
    }
    if ckpt_cfg and ckpt_cfg.include_rng:
        payload["rng_state"] = _rng_state_pack(include_cuda=True)
    torch.save(payload, path)
    return path


def load_checkpoint(
    *,
    path: str | os.PathLike,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    map_location: str | torch.device | None = None,
    strict_model_load: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """
    Load a checkpoint and (optionally) restore model/optim/scheduler/RNG.

    Returns the checkpoint dict for further inspection.
    """
    chk = torch.load(path, map_location=map_location)
    if model is not None and "model" in chk and chk["model"] is not None:
        model.load_state_dict(chk["model"], strict=strict_model_load)
    if optimizer is not None and "optimizer" in chk and chk["optimizer"] is not None:
        optimizer.load_state_dict(chk["optimizer"])
    if scheduler is not None and "scheduler" in chk and chk["scheduler"] is not None:
        scheduler.load_state_dict(chk["scheduler"])
    if restore_rng and "rng_state" in chk:
        _rng_state_load(chk["rng_state"])
    return chk


def best_is_better(new: float, best: float | None, mode: str = "max") -> bool:
    if best is None:
        return True
    return (new > best) if mode == "max" else (new < best)


def _latest_ckpt(path: str | os.PathLike) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    files = sorted(p.glob("ckpt_step*_epoch*.pt"), key=lambda x: x.stat().st_mtime)
    return str(files[-1]) if files else None


# --------------------------------------------------------------------------------------
#                                        Fit
# --------------------------------------------------------------------------------------


def fit(
    *,
    model: nn.Module,
    train_loader: Iterable[dict],
    eval_loader: Iterable[dict] | None,
    optimizer: torch.optim.Optimizer,
    logger: MetricLogger,
    loop_cfg: LoopConfig,
    rollout_fn: RolloutFn,
    loss_builder: LossBuilder | None = None,
    ckpt_cfg: CheckpointConfig | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    state: LoopState | None = None,
    hooks: dict[str, Callable] | None = None,
) -> LoopState:
    """
    A simple but complete training orchestrator:
      • training/eval epochs
      • checkpoint save/restore
      • early stopping on a chosen metric
    """
    device = torch.device(loop_cfg.device)
    model.to(device)

    ckpt_cfg = ckpt_cfg or CheckpointConfig(dir=None)
    state = state or LoopState()
    Path(ckpt_cfg.dir).mkdir(parents=True, exist_ok=True) if ckpt_cfg.dir else None

    # Auto-resume if requested
    if ckpt_cfg.dir and ckpt_cfg.resume:
        latest = _latest_ckpt(ckpt_cfg.dir)
        if latest:
            logger.log_text("fit/info", f"Resuming from {latest}", step=state.step)
            chk = load_checkpoint(
                path=latest,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
                strict_model_load=ckpt_cfg.strict_model_load,
                restore_rng=True,
            )
            ls = chk.get("loop_state", {})
            state.step = int(ls.get("step", state.step))
            state.epoch = int(ls.get("epoch", state.epoch))
            state.best_metric = ls.get("best_metric", state.best_metric)
            state.best_path = ls.get("best_path", state.best_path)

    # Train
    patience_left = loop_cfg.early_stop_patience
    for ep in range(loop_cfg.epochs):
        state.epoch += 1
        state.step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            logger=logger,
            loop_cfg=loop_cfg,
            rollout_fn=rollout_fn,
            loss_builder=loss_builder,
            step_start=state.step,
            scheduler=scheduler,
            hooks=hooks,
        )

        # Periodic checkpoint-by-epoch
        if (
            ckpt_cfg.dir
            and ckpt_cfg.save_every_epochs
            and (state.epoch % ckpt_cfg.save_every_epochs == 0)
        ):
            out = save_checkpoint(
                path=str(
                    Path(ckpt_cfg.dir)
                    / ckpt_cfg.filename.format(step=state.step, epoch=state.epoch)
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loop_state=state,
                loop_cfg=loop_cfg,
                ckpt_cfg=ckpt_cfg,
            )

            logger.log_text("fit/ckpt", f"Saved checkpoint: {out}", step=state.step)

            # Keep only last K
            if ckpt_cfg.keep_last_k > 0:
                snaps = sorted(
                    Path(ckpt_cfg.dir).glob("ckpt_step*_epoch*.pt"), key=lambda x: x.stat().st_mtime
                )
                for old in snaps[: -ckpt_cfg.keep_last_k]:
                    try:
                        old.unlink()
                    except Exception:
                        pass

        # Eval & early stopping
        do_eval = (eval_loader is not None) and (
            loop_cfg.eval_every and state.epoch % loop_cfg.eval_every == 0
        )
        primary_value: float | None = None
        if do_eval:
            scalars = eval_epoch(
                model=model,
                dataloader=eval_loader,  # type: ignore[arg-type]
                logger=logger,
                loop_cfg=loop_cfg,
                rollout_fn=rollout_fn,
                step=state.step,
                hooks=hooks,
            )
            # Choose primary metric from config
            primary_value = scalars.get(loop_cfg.best_key, None)
            if primary_value is None:
                # fall back: try success if present
                for k in ("eval/target/success", "target/success"):
                    if k in scalars:
                        primary_value = scalars[k]
                        break

            # Update best & save best checkpoint
            if primary_value is not None and best_is_better(
                primary_value, state.best_metric, loop_cfg.best_mode
            ):
                state.best_metric = primary_value
                if ckpt_cfg.dir:
                    state.best_path = str(Path(ckpt_cfg.dir) / ckpt_cfg.best_filename)
                    save_checkpoint(
                        path=state.best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loop_state=state,
                        loop_cfg=loop_cfg,
                        ckpt_cfg=ckpt_cfg,
                    )

                    logger.log_text(
                        "fit/ckpt",
                        f"New best ({loop_cfg.best_key}={primary_value:.6f}); saved → {state.best_path}",
                        step=state.step,
                    )
                patience_left = loop_cfg.early_stop_patience  # reset patience
            elif loop_cfg.early_stop_patience is not None:
                patience_left = (patience_left or 0) - 1
                if patience_left is not None and patience_left <= 0:
                    logger.log_text(
                        "fit/info",
                        f"Early stopping at epoch {state.epoch} (no improvement on {loop_cfg.best_key}).",
                        step=state.step,
                    )
                    break

    return state

# acpl/utils/logging.py
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MetricLogger", "MetricLoggerConfig"]


@dataclass
class MetricLoggerConfig:
    backend: str = "plain"  # "plain" | "tensorboard" | "wandb"
    log_dir: str | None = None
    project: str | None = None  # for wandb
    run_name: str | None = None  # for wandb
    step_key: str = "step"
    enable: bool = True


class MetricLogger:
    """
    Unified tiny logger:
      - plain:   prints to stdout
      - tb:      logs to TensorBoard (scalars only)
      - wandb:   logs scalar dicts
    """

    def __init__(self, cfg: MetricLoggerConfig):
        self.cfg = cfg
        self.backend = (cfg.backend or "plain").lower()
        self._tb = None
        self._wb = None
        if not cfg.enable:
            self.backend = "off"
            return

        if self.backend in ("tensorboard", "tb"):
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
            except Exception as e:
                print(f"[MetricLogger] TensorBoard not available ({e}); falling back to plain.")
                self.backend = "plain"
            else:
                self._tb = SummaryWriter(log_dir=cfg.log_dir)
                self.backend = "tensorboard"

        if self.backend == "wandb":
            try:
                import wandb  # type: ignore
            except Exception as e:
                print(f"[MetricLogger] wandb not available ({e}); falling back to plain.")
                self.backend = "plain"
            else:
                wandb.init(project=cfg.project, name=cfg.run_name, dir=cfg.log_dir)
                self._wb = wandb

    def close(self):
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
            self._tb = None
        if self._wb is not None:
            try:
                self._wb.finish()
            except Exception:
                pass
            self._wb = None

    def log_scalar(self, name: str, value: float, step: int):
        if self.backend == "off":
            return
        if self.backend == "plain":
            print(f"[{step:08d}] {name}: {value:.6f}")
            return
        if self.backend == "tensorboard":
            self._tb.add_scalar(name, value, step)
            return
        if self.backend == "wandb":
            self._wb.log({name: value, self.cfg.step_key: step})
            return

    def log_dict(self, prefix: str, scalars: dict[str, float], step: int):
        if self.backend == "off":
            return
        if self.backend == "plain":
            msg = " ".join([f"{k}={v:.6f}" for k, v in scalars.items()])
            print(f"[{step:08d}] {prefix} {msg}")
            return
        if self.backend == "tensorboard":
            for k, v in scalars.items():
                self._tb.add_scalar(f"{prefix}{k}", v, step)
            return
        if self.backend == "wandb":
            payload = {f"{prefix}{k}": v for k, v in scalars.items()}
            payload[self.cfg.step_key] = step
            self._wb.log(payload)
            return

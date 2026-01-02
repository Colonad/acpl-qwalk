#!/usr/bin/env python3
"""
gradient_check_robust_combo.py

Loads make_transfer_loss from scripts/train.py and checks that
robust_combo produces nonzero gradients w.r.t. P.

Run (from repo root):
  python gradient_check_robust_combo.py
or:
  python gradient_check_robust_combo.py --train scripts/train.py --B 4 --N 64 --targets 5 9 --alpha 0.5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def load_train_module(train_py: Path):
    train_py = train_py.resolve()
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found: {train_py}")

    # Ensure repo root is importable so train.py's "import acpl...." works.
    # scripts/train.py -> repo_root is ../
    repo_root = train_py.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    spec = importlib.util.spec_from_file_location("acpl_train_script", str(train_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {train_py}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # executes train.py top-level (NOT main(), because __name__ != "__main__")
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="scripts/train.py", help="Path to scripts/train.py")
    ap.add_argument("--loss", type=str, default="robust_combo", help="loss_kind to test")
    ap.add_argument("--B", type=int, default=4, help="batch size")
    ap.add_argument("--N", type=int, default=16, help="num nodes")
    ap.add_argument("--targets", type=int, nargs="+", default=[3, 7], help="target node indices")
    ap.add_argument("--alpha", type=float, default=0.5, help="robust_combo_alpha in [0,1]")
    ap.add_argument("--entropy_beta", type=float, default=0.0, help="robust_entropy_beta")
    ap.add_argument("--l2_beta", type=float, default=0.0, help="robust_l2_beta")
    ap.add_argument("--seed", type=int, default=0, help="torch seed")
    args = ap.parse_args()

    train_py = Path(args.train)
    mod = load_train_module(train_py)

    if not hasattr(mod, "make_transfer_loss"):
        raise AttributeError(f"{train_py} does not define make_transfer_loss")

    make_transfer_loss = mod.make_transfer_loss

    torch.manual_seed(args.seed)

    # Random "probability-like" tensor. renorm=True inside loss will normalize it.
    P = torch.rand(args.B, args.N, dtype=torch.float32, requires_grad=True)

    aux = {
        "targets": args.targets,
        "robust_combo_alpha": args.alpha,
        "robust_entropy_beta": args.entropy_beta,
        "robust_l2_beta": args.l2_beta,
    }

    loss_fn = make_transfer_loss(loss_kind=args.loss, reduction="mean", renorm=True)
    loss = loss_fn(P, aux, batch=None)

    loss.backward()

    grad = P.grad
    grad_sum = float(grad.abs().sum().detach().cpu()) if grad is not None else 0.0
    grad_finite = bool(torch.isfinite(grad).all().item()) if grad is not None else False

    print(f"Loaded: {train_py}")
    print(f"loss_kind: {args.loss}")
    print(f"loss: {float(loss.detach().cpu())}")
    print(f"loss.requires_grad: {loss.requires_grad}")
    print(f"P.requires_grad: {P.requires_grad}")
    print(f"grad is None: {grad is None}")
    print(f"grad |sum|: {grad_sum}")
    print(f"grad all finite: {grad_finite}")
    print(f"PASS (nonzero grad): {grad_sum > 0.0 and grad_finite}")


if __name__ == "__main__":
    main()

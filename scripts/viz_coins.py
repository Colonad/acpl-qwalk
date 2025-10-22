#!/usr/bin/env python3
"""
scripts/viz_coins.py
Visualize learned coin schedules {C_v(t)} via heatmaps over (v, t) and spectra diagnostics.

Inputs
------
Accepts the outputs produced by scripts/export_coins.py:
  • NPZ: coins.npz (coins_unitary object array), degrees, meta
  • JSON: coins.json (coins serialized as real/imag)
You can point --coins to either the NPZ or JSON.

Outputs
-------
• Heatmaps (PNG):
    - For dv=2: alpha/beta/gamma (approx. ZYZ Euler angles) over (t, v).
    - For any dv: arg(det U), phase(trace U)/d, unitarity error ‖U†U−I‖_F, spectral spread (std of eigenphases).
• Spectra diagnostics:
    - Eigenvalue clouds on the unit circle at a few representative times.
    - Per-time eigenphase histogram (wrapped to [-π, π]).
• CSV with per-time summary stats.
• meta.json with run and dataset info.

Usage
-----
python scripts/viz_coins.py \
  --coins out/coins_run1/coins.npz \
  --out   out/coins_run1_viz \
  --times 0 -1 mid \
  --show                      # optional interactive display

Notes
-----
• Robust to variable coin dimensions across nodes (uses per-node dv).
• Euler angle extraction is heuristic and intended for diagnostics, not exact recovery.
• Heatmaps sort nodes by degree by default (toggle with --no-sort-by-degree).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------- IO helpers ----------------------------- #


@dataclass
class CoinPack:
    coins: list[list[np.ndarray]]  # coins[t][v] = np.ndarray (dv x dv) complex
    degrees: np.ndarray  # (N,) int
    meta: dict[str, Any]


def _load_coins_npz(path: str) -> CoinPack:
    z = np.load(path, allow_pickle=True)
    if "coins_unitary" not in z.files:
        raise ValueError("NPZ must contain 'coins_unitary'.")
    coins_obj = z["coins_unitary"]
    T, N = coins_obj.shape
    coins: list[list[np.ndarray]] = []
    for t in range(T):
        step = []
        for v in range(N):
            M = np.asarray(coins_obj[t, v]).astype(np.complex128)
            step.append(M)
        coins.append(step)
    degrees = (
        z["degrees"].astype(np.int64)
        if "degrees" in z.files
        else np.array([c.shape[0] for c in coins[0]], dtype=np.int64)
    )
    meta = {}
    if "meta" in z.files:
        m = z["meta"]
        try:
            meta = json.loads(str(m))
        except Exception:
            meta = {"meta_raw": str(m)}
    return CoinPack(coins=coins, degrees=degrees, meta=meta)


def _load_coins_json(path: str) -> CoinPack:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    T = int(payload["T"])
    N = int(payload["N"])
    coins: list[list[np.ndarray]] = []
    for t in range(T):
        step = []
        for v in range(N):
            block = payload["coins"][t][v]
            U = np.array(block["real"], dtype=np.float64) + 1j * np.array(
                block["imag"], dtype=np.float64
            )
            step.append(U.astype(np.complex128))
        coins.append(step)
    degrees = np.array(payload.get("degrees", [c.shape[0] for c in coins[0]]), dtype=np.int64)
    meta = payload.get("meta", {})
    return CoinPack(coins=coins, degrees=degrees, meta=meta)


def load_coins(path: str) -> CoinPack:
    path = str(path)
    if path.endswith(".npz"):
        return _load_coins_npz(path)
    if path.endswith(".json"):
        return _load_coins_json(path)
    raise ValueError("Unsupported coins file (expected .npz or .json).")


# ----------------------------- Linear algebra ----------------------------- #


def unitarity_error(U: np.ndarray) -> float:
    I = np.eye(U.shape[0], dtype=U.dtype)
    return float(np.linalg.norm(U.conj().T @ U - I, ord="fro"))


def eig_phases(U: np.ndarray) -> np.ndarray:
    """Eigenvalue phases in [-pi, pi]."""
    w = np.linalg.eigvals(U)
    ang = np.angle(w)
    # sort for consistency
    ang.sort()
    return ang


def zyz_from_su2(U: np.ndarray) -> tuple[float, float, float]:
    """
    Approximate ZYZ Euler angles (alpha, beta, gamma) for SU(2) coin U.
    Heuristic but stable for visualization. Removes global phase first.

    U ≈ Rz(α) Ry(β) Rz(γ)

    Returns angles in radians in range [-pi, pi] for α, γ and [0, π] for β.
    """
    if U.shape != (2, 2):
        raise ValueError("zyz_from_su2 expects a 2x2 matrix.")
    # Normalize global phase so det ≈ 1 and U00 is as real as possible
    det = np.linalg.det(U)
    # Theoretically det=1 for SU(2); handle drift:
    gphase = np.angle(det) / 2.0
    U1 = U * np.exp(-1j * gphase)

    a = U1[0, 0]
    b = U1[0, 1]
    c = U1[1, 0]
    d = U1[1, 1]

    # beta from magnitudes
    beta = 2.0 * math.atan2(abs(b), max(1e-12, abs(a)))
    beta = np.clip(beta, 0.0, math.pi)

    # alpha, gamma from phases (one consistent branch)
    # Relations (up to 2π wraps):
    #   arg(c) - arg(a) = α
    #   arg(b) - arg(a) = -γ   (with the chosen sign convention)
    alpha = float(np.angle(c) - np.angle(a))
    gamma = float(np.angle(a) - np.angle(b))
    # wrap to [-pi, pi]
    alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
    gamma = (gamma + math.pi) % (2 * math.pi) - math.pi
    return alpha, beta, gamma


# ----------------------------- Feature extraction ----------------------------- #


def per_coin_features(U: np.ndarray) -> dict[str, float]:
    dv = U.shape[0]
    # Phase of determinant and normalized trace
    # (for SU(2) det≈1, argdet ~ 0; for U(d) general it shows global phase dynamics)
    argdet = float(np.angle(np.linalg.det(U)))
    tr = np.trace(U) / max(1, dv)
    phase_trace = float(np.angle(tr))
    unit_err = unitarity_error(U)

    # eigenphase stats
    phases = eig_phases(U)
    mean_phase = float(np.angle(np.mean(np.exp(1j * phases))))  # circular mean
    # circular std proxy = sqrt(-2 ln |mean vector|)
    R = abs(np.mean(np.exp(1j * phases)))
    spectral_spread = float(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12)))))

    out = {
        "argdet": argdet,
        "phase_trace": phase_trace,
        "unitarity": unit_err,
        "eig_mean": mean_phase,
        "eig_spread": spectral_spread,
    }

    if dv == 2:
        try:
            a, b, g = zyz_from_su2(U)
            out.update({"alpha": a, "beta": b, "gamma": g})
        except Exception:
            # safe fallback
            out.update({"alpha": 0.0, "beta": 0.0, "gamma": 0.0})
    return out


def build_feature_grids(
    coins: list[list[np.ndarray]], degrees: np.ndarray, sort_by_degree: bool = True
) -> tuple[dict[str, np.ndarray], list[int]]:
    """
    Returns:
      grids: dict name -> array[T, N] of scalar features
      perm : node permutation used (identity or degree sort)
    """
    T = len(coins)
    N = len(coins[0]) if T > 0 else 0

    # Node permutation (by degree ascending for better visuals)
    if sort_by_degree:
        perm = np.argsort(degrees).tolist()
    else:
        perm = list(range(N))

    # Discover available keys by probing (0, 0)
    sample = per_coin_features(coins[0][0])
    keys = list(sample.keys())
    grids = {k: np.zeros((T, N), dtype=np.float64) for k in keys}

    for t in range(T):
        for col, v in enumerate(perm):
            feats = per_coin_features(coins[t][v])
            for k in keys:
                grids[k][t, col] = feats[k]
    return grids, perm


# ----------------------------- Plotting utils ----------------------------- #


def _heat(ax, arr: np.ndarray, title: str, xlabel: str, ylabel: str, cmap: str = "viridis"):
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _unit_circle(ax):
    th = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(th), np.sin(th), linewidth=1.0)
    ax.set_aspect("equal", "box")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True, alpha=0.3)


def _eigs_scatter(ax, U_list: list[np.ndarray], title: str):
    _unit_circle(ax)
    pts_real = []
    pts_imag = []
    for U in U_list:
        w = np.linalg.eigvals(U)
        pts_real.extend(np.real(w).tolist())
        pts_imag.extend(np.imag(w).tolist())
    ax.scatter(pts_real, pts_imag, s=6, alpha=0.5)
    ax.set_title(title)


# ----------------------------- CSV & metadata ----------------------------- #


def save_summary_csv(path: Path, grids: dict[str, np.ndarray]):
    T, N = next(iter(grids.values())).shape
    with open(path, "w", encoding="utf-8") as f:
        header = ["t", "name", "mean", "std", "min", "max"]
        f.write(",".join(header) + "\n")
        for name, G in grids.items():
            for t in range(T):
                row = G[t]
                f.write(
                    f"{t},{name},{row.mean():.8f},{row.std():.8f},{row.min():.8f},{row.max():.8f}\n"
                )


# ----------------------------- CLI ----------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize coin schedules via heatmaps and spectra diagnostics."
    )
    p.add_argument("--coins", type=str, required=True, help="Path to coins.npz or coins.json.")
    p.add_argument("--out", type=str, required=True, help="Output directory for figures.")
    p.add_argument(
        "--times",
        nargs="*",
        default=["0", "mid", "-1"],
        help="Times to plot spectra: integers, 'mid' for T//2.",
    )
    p.add_argument(
        "--no-sort-by-degree", action="store_true", help="Keep original node order in heatmaps."
    )
    p.add_argument("--show", action="store_true", help="Display interactively.")
    return p.parse_args(argv)


def resolve_times(specs: list[str], T: int) -> list[int]:
    out: list[int] = []
    for s in specs:
        if s == "mid":
            out.append(T // 2)
        else:
            try:
                k = int(s)
                if k < 0:
                    k = T + k
                k = max(0, min(T - 1, k))
                out.append(k)
            except Exception:
                continue
    # deduplicate and sort
    return sorted(set(out))


# ----------------------------- Main ----------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pack = load_coins(args.coins)
    coins, degrees, meta = pack.coins, pack.degrees, pack.meta
    T = len(coins)
    N = len(coins[0]) if T > 0 else 0

    grids, perm = build_feature_grids(coins, degrees, sort_by_degree=(not args.no_sort_by_degree))

    # Save which permutation was used
    with open(out_dir / "viz_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_coins": os.path.abspath(args.coins),
                "T": T,
                "N": N,
                "degree_sorted": not args.no_sort_by_degree,
                "perm": perm,
                "meta": meta,
            },
            f,
            indent=2,
        )

    # HEATMAPS
    # Prefer SU(2) angles if available; otherwise spectral proxies.
    keys_su2 = [k for k in ["alpha", "beta", "gamma"] if k in grids]
    keys_generic = ["argdet", "phase_trace", "unitarity", "eig_spread"]
    heat_keys = keys_su2 if len(keys_su2) == 3 else [k for k in keys_generic if k in grids]

    for name in heat_keys:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        _heat(
            ax,
            grids[name],
            title=f"{name} over (t, node)",
            xlabel="node (sorted)" if not args.no_sort_by_degree else "node",
            ylabel="time",
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"heat_{name}.png", dpi=150)
        plt.close(fig)

    # SUMMARY CSV
    save_summary_csv(out_dir / "summary_per_time.csv", grids)

    # SPECTRA DIAGNOSTICS
    pick_times = resolve_times(list(args.times), T)
    for t in pick_times:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        _eigs_scatter(ax, coins[t], title=f"Eigenvalues at t={t}")
        fig.tight_layout()
        fig.savefig(out_dir / f"eig_scatter_t{t}.png", dpi=160)
        plt.close(fig)

        # eigenphase histogram
        phases = np.concatenate([eig_phases(U) for U in coins[t]], axis=0)
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.hist(phases, bins=60, range=(-math.pi, math.pi), density=True, alpha=0.8)
        ax.set_title(f"Eigenphase histogram at t={t}")
        ax.set_xlabel("phase (rad)")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"eig_hist_t{t}.png", dpi=160)
        plt.close(fig)

    # OPTIONAL interactive display (show last plot group if desired)
    if args.show:
        # Re-render one representative figure for display
        name = heat_keys[0] if heat_keys else "argdet"
        fig, ax = plt.subplots(figsize=(9, 4.2))
        _heat(ax, grids[name], title=f"{name} over (t, node)", xlabel="node", ylabel="time")
        fig.tight_layout()
        plt.show()

    print(f"[viz_coins] Wrote figures to: {str(out_dir.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

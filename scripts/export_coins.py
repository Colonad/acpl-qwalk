#!/usr/bin/env python3
"""
scripts/export_coins.py
Export learned coin schedules {C_v(t)} from a trained ACPL checkpoint.

Key features
------------
• Loads a torch checkpoint and reconstructs the model (w/ device + dtype controls).
• Accepts a graph episode (edge_index or adjacency), optional positional encodings and node features.
• Robust "discovery" of how to get coins from the model:
    1) model.export_coins(...)  -> list[ T ][ list[ C_v ] ]  (preferred)
    2) model.policy.export_coins(...)
    3) model.rollout(..., return_coins=True)
    4) model.forward(..., return_params=True) then lift params -> unitaries
• Parameter lifting:
    - dv=2: SU(2) via ZYZ Euler angles: Rz(α) Ry(β) Rz(γ)
    - dv>2: U(d) via exp(K), with K†=-K; or Cayley retraction (I-K)(I+K)^{-1}
      If provided a flat vector, a canonical Hermitian basis is used (generalized Gell–Mann + I).
• Exports multiple formats:
    - NPZ: coins_unitary[t, v]  (complex128), degrees, meta.json as string, etc.
    - JSON: nested lists (angles for dv=2; real+imag for matrices) with metadata.
    - CSV summary per step/node (angles for dv=2; Fro norms & unitarity error for dv>2).
• Validates unitarity (‖U†U−I‖F) and reports maxima in a diagnostics file.
• Supports deterministic seeds and mixed CPU/CUDA execution.

Usage
-----
python scripts/export_coins.py \
    --ckpt path/to/checkpoint.pt \
    --T 128 \
    --graph path/to/graph.npz \
    --features path/to/features.npz \
    --out out/coins_run1 \
    --format npz json csv \
    --unitary-map exp   # or 'cayley'
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch


# ---------------------------------------------
# Small utility: colored prints (safe fallback)
# ---------------------------------------------
def _c(s: str, color: str) -> str:
    codes = {
        "g": "\033[92m",
        "y": "\033[93m",
        "b": "\033[94m",
        "r": "\033[91m",
        "x": "\033[0m",
    }
    return f"{codes.get(color,'')}{s}{codes['x']}"


# ---------------------------------------------
# Graph holders and helpers
# ---------------------------------------------
@dataclass
class EpisodeGraph:
    """
    Minimal episode description:
      - N nodes, degrees dv from edge_index or adjacency.
      - edge_index: (2, E) integer numpy array (directed arcs OK).
      - X: optional node features [N, F] (float).
      - pos: optional positional encodings [N, K] (float).
      - meta: dictionary for anything else (T, family, etc.).
    """

    N: int
    edge_index: np.ndarray  # (2, E)
    degrees: np.ndarray  # (N,)
    X: np.ndarray | None = None
    pos: np.ndarray | None = None
    meta: dict[str, Any] = None

    @staticmethod
    def from_files(graph_path: str | None, features_path: str | None) -> EpisodeGraph:
        """
        Accepts a .npz or .npy for graph; keys supported:
            - edge_index (2, E)
            - adj (N, N) dense or sparse-coo (3 rows: row, col, val)
            - degrees (N,)
            - pos (N, K) (optional)
            - meta (json string or dict-like)
        Features file can hold 'X' and/or 'pos'.
        """
        edge_index = None
        degrees = None
        pos = None
        X = None
        meta: dict[str, Any] = {}

        if graph_path is None and features_path is None:
            raise ValueError("At least one of --graph or --features must be provided.")

        if graph_path is not None:
            gp = np.load(graph_path, allow_pickle=True)
            keys = set(gp.files)
            if "edge_index" in keys:
                ei = gp["edge_index"]
                if isinstance(ei, np.ndarray) and ei.shape[0] == 2:
                    edge_index = ei.astype(np.int64)
                else:
                    raise ValueError("edge_index must be a numpy array with shape (2, E).")
            elif "adj" in keys:
                A = gp["adj"]
                if A.ndim == 2:
                    rows, cols = np.nonzero(A)
                else:
                    # sparse-coo: rows, cols, vals
                    rows, cols = A[0], A[1]
                edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
            else:
                raise ValueError("Graph file must contain 'edge_index' or 'adj'.")

            if "degrees" in keys:
                degrees = gp["degrees"].astype(np.int64)
            else:
                N = int(edge_index.max()) + 1
                deg = np.zeros(N, dtype=np.int64)
                for u in edge_index[0]:
                    deg[u] += 1
                degrees = deg

            if "pos" in keys:
                pos = gp["pos"].astype(np.float32)

            if "meta" in keys:
                m = gp["meta"]
                if isinstance(m, (bytes, str)):
                    try:
                        meta = json.loads(m)
                    except Exception:
                        meta = {"meta_raw": str(m)}
                elif isinstance(m, np.ndarray) and m.shape == ():
                    # single object
                    try:
                        meta = dict(m.item())
                    except Exception:
                        meta = {"meta_raw": str(m)}
                else:
                    try:
                        meta = dict(m)
                    except Exception:
                        meta = {"meta_raw": str(m)}

            N = int(edge_index.max()) + 1

        else:
            # If only features are provided, we need N from there
            fp = np.load(features_path, allow_pickle=True)
            if "X" not in fp.files:
                raise ValueError("--features provided without 'X' key; cannot infer N.")
            X = fp["X"].astype(np.float32)
            N = int(X.shape[0])
            edge_index = np.zeros((2, 0), dtype=np.int64)
            degrees = np.zeros(N, dtype=np.int64)

        if features_path is not None:
            fp = np.load(features_path, allow_pickle=True)
            if "X" in fp.files:
                X = fp["X"].astype(np.float32)
                N_from_X = int(X.shape[0])
                if edge_index is not None and (int(edge_index.max()) + 1 != N_from_X):
                    # We allow X with different N but only if bigger; otherwise error.
                    if edge_index.size > 0:
                        raise ValueError("Mismatch between N inferred from graph and X.shape[0].")
            if "pos" in fp.files:
                pos = fp["pos"].astype(np.float32)

        if degrees is None:
            N = int(edge_index.max()) + 1
            deg = np.zeros(N, dtype=np.int64)
            for u in edge_index[0]:
                deg[u] += 1
            degrees = deg

        if edge_index is None:
            raise ValueError("Could not construct edge_index.")

        if X is not None and X.shape[0] != (int(edge_index.max()) + 1):
            raise ValueError("X.shape[0] must equal number of nodes inferred from edge_index.")

        return EpisodeGraph(
            N=int(edge_index.max()) + 1,
            edge_index=edge_index,
            degrees=degrees,
            X=X,
            pos=pos,
            meta=meta or {},
        )


# ---------------------------------------------
# Linear algebra: SU(2) & U(d) constructors
# ---------------------------------------------
def Rz(theta: np.ndarray | float, dtype=np.complex128):
    c = np.cos(0.5 * float(theta))
    s = np.sin(0.5 * float(theta))
    # Rz(θ) = exp(-i θ/2 σz) = diag(e^{-iθ/2}, e^{iθ/2})
    return np.array([[c - 1j * s, 0.0], [0.0, c + 1j * s]], dtype=dtype)


def Ry(theta: np.ndarray | float, dtype=np.complex128):
    c = np.cos(0.5 * float(theta))
    s = np.sin(0.5 * float(theta))
    # Ry(θ) = [[c, -s],[s, c]]
    return np.array([[c, -s], [s, c]], dtype=dtype)


def su2_zyz_unitary(alpha: float, beta: float, gamma: float, dtype=np.complex128) -> np.ndarray:
    """
    ZYZ Euler: U = Rz(alpha) @ Ry(beta) @ Rz(gamma) ∈ SU(2)
    """
    return Rz(alpha, dtype=dtype) @ Ry(beta, dtype=dtype) @ Rz(gamma, dtype=dtype)


def _gell_mann_basis(d: int, include_identity: bool = True) -> list[np.ndarray]:
    """
    Returns a real-orthonormal (under <A,B>=Tr(A B)) Hermitian basis for C^{dxd}.
    Following generalized Gell–Mann matrices + identity (optionally).
    """
    basis: list[np.ndarray] = []
    # Symmetric off-diagonal
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=np.complex128)
            M[i, j] = 1.0
            M[j, i] = 1.0
            basis.append(M)
    # Anti-symmetric off-diagonal (Hermitian with i factors)
    for i in range(d):
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=np.complex128)
            M[i, j] = -1j
            M[j, i] = 1j
            basis.append(M)
    # Diagonal (traceless)
    for k in range(1, d):
        M = np.zeros((d, d), dtype=np.complex128)
        M[:k, :k] = np.eye(k)
        M[k, k] = -k
        M = M / math.sqrt(k * (k + 1))
        basis.append(M)
    if include_identity:
        basis.append(np.eye(d, dtype=np.complex128))
    return basis


def vector_to_skew_hermitian(
    theta: np.ndarray, d: int, include_identity: bool = True
) -> np.ndarray:
    """
    Given a real vector theta of length q (q=d^2 if include_identity else d^2-1),
    build a Hermitian H = sum_k theta_k * G_k, then return K = i * H (skew-Hermitian).
    """
    basis = _gell_mann_basis(d, include_identity=include_identity)
    q = len(basis)
    if theta.ndim != 1:
        theta = theta.reshape(-1)
    if theta.shape[0] != q:
        # Best-effort: if theta len < q, pad; if >q, truncate
        if theta.shape[0] < q:
            theta = np.concatenate([theta, np.zeros(q - theta.shape[0], dtype=theta.dtype)], axis=0)
        else:
            theta = theta[:q]
    H = np.zeros((d, d), dtype=np.complex128)
    for coeff, G in zip(theta, basis, strict=False):
        H = H + float(coeff) * G
    K = 1j * H  # skew-Hermitian
    # Make sure strictly skew-Hermitian (numerical cleanup)
    K = 0.5 * (K - K.conj().T)
    return K


def expm_skew(K: np.ndarray) -> np.ndarray:
    """
    Matrix exponential for a small dense skew-Hermitian K (unitary result).
    Uses scipy-like fallback via numpy.linalg.eig if torch/scipy are unavailable.
    """
    # Try torch for speed/stability if available
    try:
        import torch

        K_t = torch.from_numpy(K)
        U = torch.linalg.matrix_exp(K_t)
        return U.numpy()
    except Exception:
        # Numpy fallback: eigendecomposition
        vals, vecs = np.linalg.eig(K)
        # exp(K) = V diag(exp(λ)) V^{-1}
        exp_vals = np.exp(vals)
        U = (vecs * exp_vals) @ np.linalg.inv(vecs)
        return U


def cayley_retraction(K: np.ndarray) -> np.ndarray:
    """
    U = (I - K)(I + K)^{-1}, unitary if (I + K) invertible and K is skew-Hermitian.
    """
    I = np.eye(K.shape[0], dtype=K.dtype)
    return (I - K) @ np.linalg.inv(I + K)


def lift_params_to_unitary(
    theta: np.ndarray,
    d: int,
    map_kind: str = "exp",
    include_identity: bool = True,
) -> np.ndarray:
    """
    theta: (p,) real vector or (d,d) matrix-like.
    If 2x2: treat as Euler ZYZ when p==3 (α,β,γ). Otherwise attempt generator map.
    For d>2: interpret theta as coefficients for K = i Σ θ_k G_k (G_k Hermitian basis).
    """
    if d == 2:
        t = theta.reshape(-1)
        if t.shape[0] == 3:
            return su2_zyz_unitary(float(t[0]), float(t[1]), float(t[2]))
        # If user passed a 2x2 already
        if t.shape[0] == 4:
            U = t.astype(np.complex128).reshape(2, 2)
            return U
        # Otherwise best-effort: build K from vector
        K = vector_to_skew_hermitian(t, d=2, include_identity=include_identity)
        return expm_skew(K) if map_kind == "exp" else cayley_retraction(K)

    # d > 2
    t = theta
    if t.ndim == 2 and t.shape == (d, d):
        # If already a skew-Hermitian or unitary
        U_cand = t.astype(np.complex128)
        # If not unitary, try exp of skew part
        UU = U_cand.conj().T @ U_cand
        err = np.linalg.norm(UU - np.eye(d), ord="fro")
        if err < 1e-6:
            return U_cand
        K = 0.5 * (U_cand - U_cand.conj().T)
        return expm_skew(K) if map_kind == "exp" else cayley_retraction(K)

    # Flat vector coefficients
    t = t.reshape(-1)
    K = vector_to_skew_hermitian(t, d=d, include_identity=include_identity)
    return expm_skew(K) if map_kind == "exp" else cayley_retraction(K)


def unitarity_error(U: np.ndarray) -> float:
    """‖U†U − I‖_F"""
    I = np.eye(U.shape[0], dtype=U.dtype)
    return float(np.linalg.norm(U.conj().T @ U - I, ord="fro"))


# ---------------------------------------------
# Model loading and coin discovery
# ---------------------------------------------
def load_checkpoint(ckpt_path: str, device: torch.device):
    """
    Load a torch checkpoint in a robust way.
    Expects either:
      - state dict + a 'model_class' or 'model_factory' hint in checkpoint, or
      - a full scripted model.
    We try several strategies; if only a state_dict is present, we warn and expect
    that the user's environment has a model builder at `acpl.models.build_model`.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model = None
    meta = {}

    if isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
        model = ckpt["model"]
        meta = {k: v for k, v in ckpt.items() if k != "model"}
        model.to(device)
        model.eval()
        return model, meta

    # TorchScripted / pickled module?
    if hasattr(ckpt, "eval") and hasattr(ckpt, "to"):
        model = ckpt
        model.to(device)
        model.eval()
        return model, meta

    # state_dict?
    if (
        isinstance(ckpt, dict)
        and any("state_dict" in k for k in ckpt.keys())
        or isinstance(ckpt, dict)
        and "state_dict" in ckpt
    ):
        sd = ckpt.get("state_dict", ckpt)
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        # Try a factory in-code
        try:
            from acpl.models import build_model  # type: ignore

            model = build_model(meta.get("model_cfg", {}))
            model.load_state_dict(sd, strict=False)
            model.to(device)
            model.eval()
            return model, meta
        except Exception as e:
            print(_c(f"[warn] Could not auto-build model from state_dict: {e}", "y"))
            raise RuntimeError(
                "Checkpoint appears to be a plain state_dict, but no model factory was found. "
                "Please include `model` or `model_cfg` & `acpl.models.build_model` in your project."
            )

    # Fallback: maybe it's raw pickled state containing 'model'
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
        meta = {k: v for k, v in ckpt.items() if k != "model"}
        model.to(device)
        model.eval()
        return model, meta

    raise RuntimeError("Unrecognized checkpoint format.")


@torch.no_grad()
def discover_and_export_coins(
    model: Any,
    ep: EpisodeGraph,
    T: int,
    device: torch.device,
    unitary_map: str = "exp",
    include_identity: bool = True,
) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
    """
    Try multiple strategies to obtain {C_v(t)}. Returns:
      coins: list length T; each item is list of N local coins (complex numpy arrays).
      aux:   info dict with diagnostics and method used.
    """
    info: dict[str, Any] = {"strategy": None, "notes": []}

    # Helper to convert nested params -> unitaries, if needed
    def _lift_params(theta_t: Any, degrees: np.ndarray) -> list[np.ndarray]:
        """
        theta_t: could be
          - np.ndarray [N, p] or list of per-node arrays
          - torch.Tensor [N, p]
          - dict with 'params' key
          - list of unitary matrices already
        """
        # If already a list of matrices, return as is (after type conversion)
        if (
            isinstance(theta_t, (list, tuple))
            and len(theta_t) == ep.N
            and hasattr(theta_t[0], "shape")
        ):
            mats: list[np.ndarray] = []
            for v in range(ep.N):
                M = np.asarray(theta_t[v])
                if np.iscomplexobj(M) and M.shape[0] == M.shape[1]:
                    mats.append(M.astype(np.complex128))
                else:
                    d = int(degrees[v])
                    mats.append(
                        lift_params_to_unitary(M.reshape(-1), d, unitary_map, include_identity)
                    )
            return mats

        # Generic tensor/array [N, ...]
        if isinstance(theta_t, torch.Tensor):
            theta_t = theta_t.detach().cpu().numpy()
        if isinstance(theta_t, dict) and "params" in theta_t:
            theta_t = theta_t["params"]

        theta_t = np.asarray(theta_t)
        mats: list[np.ndarray] = []
        for v in range(ep.N):
            d = int(degrees[v])
            vec = theta_t[v]
            mats.append(lift_params_to_unitary(vec, d, unitary_map, include_identity))
        return mats

    # Prepare inputs in torch for the model
    edge_index = torch.as_tensor(ep.edge_index, device=device)
    X = torch.as_tensor(ep.X, device=device, dtype=torch.float32) if ep.X is not None else None
    pos = (
        torch.as_tensor(ep.pos, device=device, dtype=torch.float32) if ep.pos is not None else None
    )
    degrees_t = torch.as_tensor(ep.degrees, device=device, dtype=torch.long)

    # 1) Direct export method on the model
    for attr in ["export_coins", "coins_for_episode"]:
        fn = getattr(model, attr, None)
        if callable(fn):
            try:
                out = fn(edge_index=edge_index, X=X, pos=pos, degrees=degrees_t, T=T)  # type: ignore
                # Expect list[T] of list[N] unitary matrices or params
                coins_T: list[list[np.ndarray]] = []
                for t in range(T):
                    Ct = out[t]
                    # If Ct are params, lift them
                    Ct_mats = _lift_params(Ct, ep.degrees)
                    coins_T.append(Ct_mats)
                info["strategy"] = f"model.{attr}"
                return coins_T, info
            except Exception as e:
                info["notes"].append(f"model.{attr} failed: {e}")

    # 2) On a policy submodule
    pol = getattr(model, "policy", None)
    if pol is not None:
        for attr in ["export_coins", "coins_for_episode"]:
            fn = getattr(pol, attr, None)
            if callable(fn):
                try:
                    out = fn(edge_index=edge_index, X=X, pos=pos, degrees=degrees_t, T=T)  # type: ignore
                    coins_T: list[list[np.ndarray]] = []
                    for t in range(T):
                        Ct = out[t]
                        Ct_mats = _lift_params(Ct, ep.degrees)
                        coins_T.append(Ct_mats)
                    info["strategy"] = f"model.policy.{attr}"
                    return coins_T, info
                except Exception as e:
                    info["notes"].append(f"model.policy.{attr} failed: {e}")

    # 3) Try a rollout that returns coins/params
    for attr in ["rollout", "forward"]:
        fn = getattr(model, attr, None)
        if callable(fn):
            try:
                maybe = fn(
                    edge_index=edge_index,
                    X=X,
                    pos=pos,
                    degrees=degrees_t,
                    T=T,
                    return_coins=True,
                )
                # standardize output
                if isinstance(maybe, dict) and "coins" in maybe:
                    out = maybe["coins"]
                else:
                    out = maybe
                coins_T: list[list[np.ndarray]] = []
                for t in range(T):
                    Ct = out[t]
                    Ct_mats = _lift_params(Ct, ep.degrees)
                    coins_T.append(Ct_mats)
                info["strategy"] = f"model.{attr}(return_coins=True)"
                return coins_T, info
            except Exception as e:
                info["notes"].append(f"model.{attr} w/ return_coins failed: {e}")

            # Maybe returns params only
            try:
                maybe = fn(
                    edge_index=edge_index,
                    X=X,
                    pos=pos,
                    degrees=degrees_t,
                    T=T,
                    return_params=True,
                )
                if isinstance(maybe, dict) and "params" in maybe:
                    out = maybe["params"]
                else:
                    out = maybe
                coins_T: list[list[np.ndarray]] = []
                for t in range(T):
                    theta_t = out[t]
                    Ct_mats = _lift_params(theta_t, ep.degrees)
                    coins_T.append(Ct_mats)
                info["strategy"] = f"model.{attr}(return_params=True) + lift"
                return coins_T, info
            except Exception as e:
                info["notes"].append(f"model.{attr} w/ return_params failed: {e}")

    raise RuntimeError(
        "Could not discover a path to export coins. Tried model.export_coins, "
        "model.policy.export_coins, rollout/forward with return_coins/params. "
        f"Notes: {info.get('notes')}"
    )


# ---------------------------------------------
# Export writers
# ---------------------------------------------
def export_npz(
    out_dir: Path, coins_T: list[list[np.ndarray]], degrees: np.ndarray, meta: dict[str, Any]
):
    out_dir.mkdir(parents=True, exist_ok=True)
    T = len(coins_T)
    N = len(coins_T[0]) if T > 0 else 0

    # Pack as object arrays to support variable d_v
    coins_obj = np.empty((T, N), dtype=object)
    max_err = 0.0
    for t in range(T):
        for v in range(N):
            U = coins_T[t][v]
            coins_obj[t, v] = U
            max_err = max(max_err, unitarity_error(U))

    np.savez_compressed(
        out_dir / "coins.npz",
        coins_unitary=coins_obj,
        degrees=degrees.astype(np.int64),
        meta=json.dumps(meta),
        max_unitarity_error=max_err,
    )
    print(_c(f"[npz] wrote {out_dir/'coins.npz'} (max unitarity err={max_err:.2e})", "g"))


def export_json(
    out_dir: Path, coins_T: list[list[np.ndarray]], degrees: np.ndarray, meta: dict[str, Any]
):
    out_dir.mkdir(parents=True, exist_ok=True)
    T = len(coins_T)
    N = len(coins_T[0]) if T > 0 else 0
    payload: dict[str, Any] = {
        "T": T,
        "N": N,
        "degrees": degrees.tolist(),
        "meta": meta,
        "coins": [],
    }
    max_err = 0.0
    for t in range(T):
        step_list: list[Any] = []
        for v in range(N):
            U = coins_T[t][v]
            max_err = max(max_err, unitarity_error(U))
            # Serialize complex matrix as {real, imag}
            step_list.append(
                {
                    "real": U.real.tolist(),
                    "imag": U.imag.tolist(),
                }
            )
        payload["coins"].append(step_list)

    payload["max_unitarity_error"] = max_err
    with open(out_dir / "coins.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(_c(f"[json] wrote {out_dir/'coins.json'} (max unitarity err={max_err:.2e})", "g"))


def export_csv(out_dir: Path, coins_T: list[list[np.ndarray]], degrees: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    T = len(coins_T)
    N = len(coins_T[0]) if T > 0 else 0

    # For dv=2 include approximate Euler angles via best ZYZ (recoverable from unitary).
    # For dv>2 write Frobenius norms and unitarity errors as diagnostics.
    csv_path = out_dir / "coins_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t,node,dv,unitarity_err,diag\n")
        for t in range(T):
            for v in range(N):
                U = coins_T[t][v]
                err = unitarity_error(U)
                dv = int(degrees[v])
                if dv == 2:
                    # recover approximate angles (not unique; choose a simple branch)
                    # U ≈ Rz(α)Ry(β)Rz(γ) -> extract via standard formulae
                    # We avoid angle wrap complexity; provide crude diagnostics only.
                    # β = 2*arccos(|U00|)
                    beta = 2 * math.acos(np.clip(np.abs(U[0, 0]), 0.0, 1.0))
                    diag = f"beta≈{beta:.6f}"
                else:
                    diag = f"fro={np.linalg.norm(U, ord='fro'):.6f}"
                f.write(f"{t},{v},{dv},{err:.6e},{diag}\n")
    print(_c(f"[csv] wrote {csv_path}", "g"))


# ---------------------------------------------
# CLI / main
# ---------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export learned coin schedules {C_v(t)} from a trained ACPL model."
    )
    p.add_argument(
        "--ckpt", type=str, required=True, help="Path to model checkpoint (.pt / TorchScript)."
    )
    p.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Graph episode file (.npz/.npy) with edge_index/adj/degrees/pos/meta.",
    )
    p.add_argument(
        "--features",
        type=str,
        default=None,
        help="Optional features file (.npz) with X and/or pos.",
    )
    p.add_argument("--T", type=int, required=True, help="Horizon (number of steps).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--out", type=str, required=True, help="Output directory.")
    p.add_argument(
        "--format",
        nargs="+",
        default=["npz"],
        choices=["npz", "json", "csv"],
        help="Export formats.",
    )
    p.add_argument(
        "--unitary-map",
        type=str,
        default="exp",
        choices=["exp", "cayley"],
        help="Mapping from generators to U(d) (when lifting).",
    )
    p.add_argument(
        "--include-identity",
        action="store_true",
        help="Include identity in generator basis for dv>2.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for any stochastic parts.")
    return p.parse_args(argv)


def set_deterministic(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_deterministic(args.seed)

    device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    torch.set_default_dtype(dtype_map[args.dtype])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    t0 = time.time()
    model, ckpt_meta = load_checkpoint(args.ckpt, device=device)
    print(_c(f"Loaded model from {args.ckpt}", "g"))

    # Build episode inputs
    ep = EpisodeGraph.from_files(args.graph, args.features)
    # meta pack
    meta = {
        "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt": os.path.abspath(args.ckpt),
        "device": str(device),
        "T": int(args.T),
        "N": int(ep.N),
        "unitary_map": args.unitary_map,
        "include_identity": bool(args.include_identity),
        "ckpt_meta": {k: str(v) for k, v in ckpt_meta.items()},
        "episode_meta": ep.meta or {},
    }

    # Export
    coins_T, info = discover_and_export_coins(
        model=model,
        ep=ep,
        T=int(args.T),
        device=device,
        unitary_map=args.unitary_map,
        include_identity=args.include_identity,
    )
    meta["export_strategy"] = info.get("strategy", "unknown")
    meta["notes"] = info.get("notes", [])

    # Write outputs
    if "npz" in args.format:
        export_npz(out_dir, coins_T, ep.degrees, meta)
    if "json" in args.format:
        export_json(out_dir, coins_T, ep.degrees, meta)
    if "csv" in args.format:
        export_csv(out_dir, coins_T, ep.degrees)

    # Also write metadata/diagnostics
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Quick summary
    T_steps = len(coins_T)
    N_nodes = len(coins_T[0]) if T_steps else 0
    max_err = 0.0
    for t in range(T_steps):
        for v in range(N_nodes):
            max_err = max(max_err, unitarity_error(coins_T[t][v]))
    dt = time.time() - t0
    print(
        _c(
            f"Exported T={T_steps}, N={N_nodes} | max ‖U†U−I‖_F={max_err:.2e} | elapsed {dt:.2f}s",
            "b",
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(_c("\nInterrupted.", "r"))
        raise

# acpl/eval/ablations.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from acpl.policy.policy import ACPLPolicy

__all__ = [
    "AblationConfig",
    "apply_nope_to_X",
    "wrap_policy_for_ablations",
    "permute_graph",
    "rollout_with_ablation",
]


# --------------------------------------------------------------------------------------
#                                         Config
# --------------------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """
    Ablations for B6 evaluation, matching the theoretical report:

    - NoPE:     Remove/zero positional encodings fed to the encoder.
    - GlobalCoin: Enforce a *single* coin per time step shared by all nodes.
    - TimeFrozen: Freeze time dynamics of the policy -> use t=0 coins at all t.
    - NodePermute: Random node relabeling sanity-check (equivariance test).

    Notes
    -----
    • NoPE assumes PE channels are the *last* `pe_dim` columns in X.
      We *zero* them instead of deleting (keeps shapes and downstream norms stable).

    • GlobalCoin & TimeFrozen are implemented as light policy wrappers that only
      modify the tensor of emitted angles θ (shape (T, N, P)) and remain fully
      differentiable (if used in training). They are strictly for eval in B6.

    • NodePermute permutes the graph (X, edge_index[, targets, batch]) before
      calling the provided rollout, and unpermutes returned P and target indices.

    • You can combine these ablations; composition order is:
         NoPE (input) → Policy wrappers (GlobalCoin → TimeFrozen) → NodePermute.
    """

    # Input-space ablation
    nope: bool = False
    pe_dim: int | None = None  # number of PE channels at the *end* of X

    # Policy-output ablations (operate on θ after base forward)
    global_coin: bool = False
    time_frozen: bool = False

    # Graph permutation sanity test
    node_permute: bool = False
    perm_seed: int = 0

    # Misc
    device: str = "cuda"

    # Safety/validation
    strict_shapes: bool = True

    def validate(self) -> None:
        if self.nope and (self.pe_dim is None or self.pe_dim <= 0):
            raise ValueError("NoPE enabled but `pe_dim` is not set > 0.")
        if self.strict_shapes and self.pe_dim is not None and self.pe_dim < 0:
            raise ValueError("`pe_dim` must be None or >= 0.")


# --------------------------------------------------------------------------------------
#                                   NoPE (input side)
# --------------------------------------------------------------------------------------


def apply_nope_to_X(X: Tensor, pe_dim: int) -> Tensor:
    """
    Zero out the last `pe_dim` feature channels from X (NoPE ablation).
    Keeps tensor shape intact.

    Args
    ----
    X:       (N, F) float tensor
    pe_dim:  number of trailing channels to zero

    Returns
    -------
    X_nope: (N, F) float tensor with the last pe_dim columns zeroed.
    """
    if pe_dim <= 0:
        return X
    if X.ndim != 2:
        raise ValueError(f"Expected X to be (N,F), got shape {tuple(X.shape)}")
    F = X.size(1)
    if pe_dim > F:
        raise ValueError(f"pe_dim={pe_dim} exceeds feature dim F={F}.")
    if pe_dim == 0:
        return X

    if pe_dim > 0:
        X = X.clone()
        X[:, F - pe_dim : F] = 0.0
    return X


# --------------------------------------------------------------------------------------
#                        Policy wrappers (output-space ablations)
# --------------------------------------------------------------------------------------


class _PolicyWrapper(nn.Module):
    """
    Base wrapper to alter only the *angles/coin-parameters* emitted by a policy,
    without touching the encoder/controller internals. Assumes that the wrapped
    policy forward returns θ with shape (T, N, P) for dv=2 (SU(2) ZYZ angles),
    or more generally (T, N, P) for any per-node parameterization.

    We do *not* alter the policy's unitary lift; that remains exactly as in ACPLPolicy.
    """

    def __init__(self, base: ACPLPolicy):
        super().__init__()
        self.base = base

    def forward(
        self, X: Tensor, edge_index: Tensor, *, T: int, edge_weight: Tensor | None = None
    ) -> Tensor:
        raise NotImplementedError


class GlobalCoinPolicy(_PolicyWrapper):
    """
    Enforce the same coin per time step for all nodes:
       θ_t,global = mean_v θ_{t, v}
       θ_{t, v} := θ_t,global  for all v

    This respects time variation but removes spatial adaptivity.
    """

    def forward(
        self, X: Tensor, edge_index: Tensor, *, T: int, edge_weight: Tensor | None = None
    ) -> Tensor:
        theta = self.base(X, edge_index, T=T, edge_weight=edge_weight)  # (T, N, P)
        if theta.ndim != 3:
            raise RuntimeError(f"Expected θ of shape (T,N,P), got {tuple(theta.shape)}")
        # Average over nodes, keepdims then broadcast back
        theta_global = theta.mean(dim=1, keepdim=True)  # (T, 1, P)
        theta_out = theta_global.expand_as(theta)  # (T, N, P)
        return theta_out


class TimeFrozenPolicy(_PolicyWrapper):
    """
    Enforce time-invariant coins:
       θ_t := θ_0  for all t

    This preserves per-node spatial adaptivity but removes temporal control.
    """

    def forward(
        self, X: Tensor, edge_index: Tensor, *, T: int, edge_weight: Tensor | None = None
    ) -> Tensor:
        theta = self.base(X, edge_index, T=T, edge_weight=edge_weight)  # (T, N, P)
        if theta.ndim != 3:
            raise RuntimeError(f"Expected θ of shape (T,N,P), got {tuple(theta.shape)}")
        theta0 = theta[0:1]  # (1, N, P)
        theta_out = theta0.expand_as(theta)  # (T, N, P)
        return theta_out


def wrap_policy_for_ablations(policy: ACPLPolicy, cfg: AblationConfig) -> ACPLPolicy:
    """
    Compose requested policy wrappers. Order: GlobalCoin → TimeFrozen.

    (NoPE is handled on the batch/features side, not here.)
    """
    cfg.validate()
    wrapped: nn.Module = policy
    if cfg.global_coin:
        wrapped = GlobalCoinPolicy(wrapped)  # type: ignore[arg-type]
    if cfg.time_frozen:
        wrapped = TimeFrozenPolicy(wrapped)  # type: ignore[arg-type]

    # We preserve ACPLPolicy public API by attaching lift methods/attrs if needed
    # (the wrappers delegate to `base` which is a bona fide ACPLPolicy).
    # For mypy/runtime users of internal helpers like _su2_from_euler_batch, expose passthroughs.
    if isinstance(policy, ACPLPolicy):
        for attr in ("_su2_from_euler_batch",):
            if hasattr(policy, attr) and not hasattr(wrapped, attr):
                setattr(wrapped, attr, getattr(policy, attr))  # passthrough for backprop trainer
    return wrapped  # type: ignore[return-value]


# --------------------------------------------------------------------------------------
#                      Node permutation (equivariance sanity ablation)
# --------------------------------------------------------------------------------------


def _make_perm(n: int, *, seed: int, device: torch.device) -> Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return torch.randperm(n, generator=g, device=device)


def _invert_perm(p: Tensor) -> Tensor:
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), device=p.device)
    return inv


def permute_graph(
    *,
    X: Tensor,
    edge_index: Tensor,
    targets: Tensor | None = None,
    batch: Tensor | None = None,
    seed: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """
    Apply a single random permutation to node indices (single-graph or multi-graph with `batch`).

    For multi-graph case (mini-batch of small graphs), we permute *within each graph component*
    according to independent seeds derived from (seed + gid). This keeps graph membership intact.

    Returns
    -------
    (perm_batch, meta) where
        perm_batch: dict containing permuted tensors: X, edge_index, (optional) targets, batch
        meta: dict with 'perm': Tensor (N,), 'invperm': Tensor (N,)
    """
    if X.ndim != 2:
        raise ValueError("Expected X: (N,F).")
    N = X.size(0)
    device = X.device

    if batch is None:
        # Single graph case
        p = _make_perm(N, seed=seed, device=device)
        inv = _invert_perm(p)
        Xp = X[p]
        # Remap edge indices
        edge_index_p = p[edge_index]
        t_out = None if targets is None else p[targets] if targets.dim() == 1 else targets
        perm_batch = {"X": Xp, "edge_index": edge_index_p}
        if targets is not None:
            perm_batch["targets"] = t_out  # type: ignore[index]
        meta = {"perm": p, "invperm": inv}
        return perm_batch, meta
    else:
        # Multi-graph mini-batch; permute within each component independently
        if batch.ndim != 1 or batch.size(0) != N:
            raise ValueError("`batch` must be (N,) and align with X.")
        B = int(batch.max().item()) + 1
        p_global = torch.empty(N, dtype=torch.long, device=device)
        inv_global = torch.empty(N, dtype=torch.long, device=device)

        # Build per-graph permutations and stitch them
        start = 0
        index_lists = [torch.nonzero(batch == gid, as_tuple=False).flatten() for gid in range(B)]
        for gid, idx in enumerate(index_lists):
            n_g = idx.numel()
            p_g = _make_perm(n_g, seed=seed + gid, device=device)
            p_global[idx] = idx[p_g]
        inv_global[p_global] = torch.arange(N, device=device)

        Xp = X[p_global]
        edge_index_p = p_global[edge_index]
        perm_batch = {"X": Xp, "edge_index": edge_index_p, "batch": batch[p_global]}

        if targets is not None:
            if targets.dim() == 1:
                # Node indices as targets
                perm_batch["targets"] = p_global[targets]
            else:
                perm_batch["targets"] = targets

        meta = {"perm": p_global, "invperm": inv_global}
        return perm_batch, meta


# --------------------------------------------------------------------------------------
#                    Orchestration: run a rollout under ablations
# --------------------------------------------------------------------------------------

RolloutFn = Callable[[nn.Module, dict], tuple[Tensor, dict]]  # returns (P, aux)


def _clone_batch_shallow(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def rollout_with_ablation(
    *,
    base_policy: ACPLPolicy,
    rollout_fn: RolloutFn,
    batch: dict,
    cfg: AblationConfig,
) -> tuple[Tensor, dict]:
    """
    Execute `rollout_fn`(model, batch) under the requested ablations and return
    (P, aux) in the *original* node order for downstream metrics.

    Contracts
    ---------
    • batch must contain at least: "X": (N,F), "edge_index": (2,E).
      Optional keys: "targets": (K,) or (B, ...), "batch": (N,) for multi-graph.

    • rollout_fn: any callable that takes (model, batch) and returns
        P:   (B,N) or (N,) probabilities over nodes
        aux: dict (may include "loss", "targets", etc.)

    Application order
    -----------------
    1) If NoPE: zero last pe_dim columns of X.
    2) Wrap policy with GlobalCoin/TimeFrozen as requested.
    3) If NodePermute: permute (X, edge_index, [targets, batch]); call rollout;
       then unpermute returned P (and aux['targets'] if node-indexed) back.

    Returns
    -------
    P_out, aux_out with the *original* node order and targets re-aligned.
    """
    cfg.validate()
    device = torch.device(cfg.device)

    # 1) Prepare batch clone and apply NoPE if requested
    b = _clone_batch_shallow(batch)
    X: Tensor = b["X"]
    if not isinstance(X, Tensor):
        raise ValueError("batch['X'] must be a torch.Tensor")
    if cfg.nope:
        b["X"] = apply_nope_to_X(X, pe_dim=int(cfg.pe_dim or 0))

    # 2) Wrap the policy for output-space ablations
    model = wrap_policy_for_ablations(base_policy, cfg)

    # 3) Node permutation (graph-level)
    meta = None
    if cfg.node_permute:
        permed, meta = permute_graph(
            X=b["X"],
            edge_index=b["edge_index"],
            targets=b.get("targets", None),
            batch=b.get("batch", None),
            seed=cfg.perm_seed,
        )
        # Update batch with permuted tensors
        b.update(permed)

    # 4) Run rollout
    P, aux = rollout_fn(model, b)

    # Normalize shape (B,N) for convenience
    if P.ndim == 1:
        P = P.unsqueeze(0)

    # 5) If permuted, unpermute outputs/targets back to original node order for logging & metrics
    if cfg.node_permute and meta is not None:
        inv = meta["invperm"]  # (N,)
        # Single-graph or multi-graph unpermute along node axis
        if P.ndim == 2:
            P = P[:, inv]
        elif P.ndim == 3:
            P = P[:, inv, :]
        else:
            # Keep generic, but typical use is (B,N)
            pass

        # Restore targets if they are node indices
        if "targets" in aux and isinstance(aux["targets"], Tensor) and aux["targets"].dim() == 1:
            # Note: aux["targets"] were permuted in the ablated batch seen by the model;
            # to align with original graph labeling in metrics, invert that permutation.
            # If multiple targets: map each index through inv-perm.
            t = aux["targets"]
            aux["targets"] = inv[t]

    return P, aux

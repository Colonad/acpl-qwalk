# acpl/eval/ablations.py
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from dataclasses import dataclass, asdict


import torch
from torch import Tensor, nn

from acpl.policy.policy import ACPLPolicy


__all__ = [
    "AblationConfig",
    "apply_nope_to_X",
    "wrap_policy_for_ablations",
    "permute_graph",
    "rollout_with_ablation",
    # eval wiring
    "normalize_ablation_name",
    "apply_ablation_bundle",
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

      


      Historically we assumed PE channels are the *last* `pe_dim` columns.
      However, the project’s default X layout is often:
         [degree, coord1, (optional indicator_last)]
      In that common layout, the positional-like channels are the “middle”
      channels (index 1 .. -2) when indicator exists.
      Therefore NoPE supports an AUTO mode that zeroes positional-like channels
      while preserving degree and (optionally) the last indicator channel.


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
    pe_dim: int | None = None  # number of trailing PE channels (legacy); None => auto-mode allowed

    # NoPE behavior (needed by apply_ablation_bundle + rollout_with_ablation)
    
    nope_keep_last_indicator: bool = False
    nope_allow_auto: bool = True



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
        # NoPE requires either explicit pe_dim>0 OR auto-mode enabled.
        if self.nope and not self.nope_allow_auto:
            if self.pe_dim is None or int(self.pe_dim) <= 0:
                raise ValueError(
                    "NoPE enabled but `pe_dim` is not set > 0 and `nope_allow_auto` is False."
                )
        if self.strict_shapes and self.pe_dim is not None and int(self.pe_dim) < 0:
            raise ValueError("`pe_dim` must be None or >= 0.")

# --------------------------------------------------------------------------------------
#                                   NoPE (input side)
# --------------------------------------------------------------------------------------


def apply_nope_to_X(
    X: Tensor,
    pe_dim: int | None = None,
    *,
    keep_last_indicator: bool = False,
    allow_auto: bool = True,
) -> Tensor:
    """
    NoPE = remove positional signal from node features while keeping shapes stable.

    Two modes:
      (A) Trailing-PE mode (legacy): if pe_dim is provided (>0),
          we zero the last pe_dim columns (optionally preserving the last indicator).
      (B) Auto heuristic (recommended for this repo’s default X layout):
          keep X[:,0] (degree), and if keep_last_indicator and X has >=3 cols,
          preserve X[:,-1] (indicator), while zeroing all “middle” columns.


    Args
    ----
    X:       (N, F) float tensor
    pe_dim:  number of trailing PE channels (legacy). If None/<=0, auto-mode may be used.
    keep_last_indicator: preserve the last channel (common in search/robust).
    allow_auto: if True, use auto heuristic when pe_dim is missing/invalid.
    Returns
    -------
    X_nope: (N, F) float tensor with positional-like channels zeroed.    """


    if X.ndim != 2:
        raise ValueError(f"Expected X to be (N,F), got shape {tuple(X.shape)}")

    F = int(X.size(1))
    if F <= 1:
        return X

    # Normalize pe_dim
    pd = None
    if isinstance(pe_dim, int) and pe_dim > 0:
        pd = int(pe_dim)

    Y = X.clone()

    # (A) trailing-PE mode
    if pd is not None:
        if pd > F:
            raise ValueError(f"pe_dim={pd} exceeds feature dim F={F}.")
        if keep_last_indicator and F >= 2:
            # Preserve last column; zero a trailing block right before it.
            # Example: [deg, coord1, indicator] with pe_dim=1 -> zero coord1, keep indicator.
            start = max(1, F - pd - 1)
            end = F - 1
            if end > start:
                Y[:, start:end] = 0.0
            return Y
        else:
            Y[:, F - pd : F] = 0.0
            return Y

    # (B) auto heuristic mode
    if not allow_auto:
        return X
    if keep_last_indicator and F >= 3:
        # keep degree (0) and indicator (-1); zero everything in-between
        Y[:, 1 : F - 1] = 0.0
    else:
        # keep degree (0); zero all remaining
        Y[:, 1:F] = 0.0
    return Y


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

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(
        self, X: Tensor, edge_index: Tensor, *, T: int, edge_weight: Tensor | None = None
    ) -> Tensor:
        raise NotImplementedError

    def __getattr__(self, name: str):
        # Allow rollouts to access attributes/methods on the base policy.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

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


def wrap_policy_for_ablations(policy: nn.Module, cfg: AblationConfig) -> nn.Module:
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
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    p = torch.randperm(n, generator=g, device="cpu")
    return p.to(device)


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
        p = _make_perm(N, seed=seed, device=device)     # new -> old
        inv = _invert_perm(p)                           # old -> new
        Xp = X[p]

        # Edges: map old endpoints -> new endpoints using inv
        edge_index_p = inv[edge_index]

        # Targets: if node-index targets (K,), map old -> new
        t_out = None
        if targets is not None:
            t_out = inv[targets] if targets.dim() == 1 else targets

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

        # Build per-graph permutations and stitch them

        index_lists = [torch.nonzero(batch == gid, as_tuple=False).flatten() for gid in range(B)]
        for gid, idx in enumerate(index_lists):
            n_g = idx.numel()
            p_g = _make_perm(n_g, seed=seed + gid, device=device)
            # p_global is still "new -> old" in the global indexing space

            # p_global is still "new -> old" in the global indexing space
            p_global[idx] = idx[p_g]



        inv_global = _invert_perm(p_global)


        Xp = X[p_global]
        # Map old endpoints via inv_global (old -> new)
        edge_index_p = inv_global[edge_index]

        perm_batch = {"X": Xp, "edge_index": edge_index_p, "batch": batch[p_global]}

        if targets is not None:
            if targets.dim() == 1:
                # Node indices as targets
                perm_batch["targets"] = inv_global[targets]

            else:
                perm_batch["targets"] = targets

        meta = {"perm": p_global, "invperm": inv_global}
        return perm_batch, meta


# --------------------------------------------------------------------------------------
#                    Orchestration: run a rollout under ablations
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
#                          Eval wiring: bundle-style API
# --------------------------------------------------------------------------------------

_ABLATION_ALIASES: dict[str, set[str]] = {
    "NoPE": {"nope", "no_pe", "no-pe", "noposenc", "no_posenc"},
    "GlobalCoin": {"globalcoin", "global_coin", "global-coin"},
    "TimeFrozen": {"timefrozen", "time_frozen", "time-frozen", "staticcoin", "static_coin"},
    "NodePermute": {"nodepermute", "node_permute", "node-permute", "permutenodes", "permute_nodes", "permute-nodes"},
}

def normalize_ablation_name(name: str) -> str:
    s = (name or "").strip()
    low = s.lower().replace(" ", "")
    for canon, aliases in _ABLATION_ALIASES.items():
        if low == canon.lower() or low in aliases:
            return canon
    return s  # preserve unknown ablation names

def _infer_pe_dim_from_cfg(cfg: Mapping[str, Any] | None) -> int | None:
    """
    Best-effort: find node-PE dimensionality if present in config.
    You can expand these keys if your YAML schema differs.
    """
    if not cfg:
        return None
    # common places people store node PE dim
    candidates = [
        ("data", "pe_dim"),
        ("data", "lap_pe_dim"),
        ("model", "node_pe_dim"),
        ("model", "pe_dim"),
        ("encoder", "pe_dim"),
    ]
    for a, b in candidates:
        try:
            v = (cfg.get(a, {}) or {}).get(b, None)  # type: ignore[union-attr]
        except Exception:
            v = None
        if isinstance(v, int) and v > 0:
            return v
    return None





def _infer_keep_last_indicator_from_cfg(cfg: Mapping[str, Any] | None) -> bool:
    """
    Project convention:
      - search/robust tasks often append an indicator as the LAST feature channel.
    """
    if not cfg:
        return False
    try:
        task = cfg.get("task", {})  # type: ignore[union-attr]
    except Exception:
        task = {}
    if not isinstance(task, Mapping):
        return False
    name = str(task.get("name", cfg.get("goal", ""))).lower()
    default = ("search" in name) or ("robust" in name)
    return bool(task.get("use_indicator", default))













def apply_ablation_bundle(
    *,
    name: str,
    model: nn.Module,
    dataloader_factory: Any | None = None,
    rollout_fn: Any | None = None,
    cfg: Mapping[str, Any] | None = None,
    device: Any | None = None,
) -> dict[str, Any]:
    """
    Bundle-style entrypoint for eval.py.

    Returns a dict with optional overrides:
      - "model"
      - "dataloader_factory"
      - "rollout_fn"
      - "tag"
      - "meta"

    Design choice:
      - We implement ALL four planned ablations by overriding the *rollout_fn* via rollout_with_ablation(),
        which can apply NoPE, GlobalCoin, TimeFrozen, and PermuteNodes at rollout-time.
      - This avoids brittle dataloader rewriting and works for both ckpt and baseline policies as long
        as they are "policy-like" (forward emits theta).
    """
    canon = normalize_ablation_name(name)
    tag = f"abl_{canon}"

    # Build an AblationConfig from the name
    pe_dim = _infer_pe_dim_from_cfg(cfg)
    keep_last = _infer_keep_last_indicator_from_cfg(cfg)
    ab = AblationConfig(
        nope=(canon == "NoPE"),
        pe_dim=pe_dim,


        nope_keep_last_indicator=keep_last,
        nope_allow_auto=True,

        global_coin=(canon == "GlobalCoin"),
        time_frozen=(canon == "TimeFrozen"),
        node_permute=(canon == "NodePermute"),
        perm_seed=int((cfg.get("seed", 0) if isinstance(cfg, Mapping) else 0)),
        device=str(device) if device is not None else "cpu",
        strict_shapes=True,
    )

    # IMPORTANT: we do NOT hard-fail NoPE when pe_dim is missing.
    # The repo’s default X layout makes trailing-pe_dim brittle; we support auto-mode instead.
    # If user really wants strict pe_dim, they can set nope_allow_auto=False and provide pe_dim.


    # If we don't have a rollout_fn, we can only do model-only ablations (GlobalCoin/TimeFrozen).
    if rollout_fn is None:
        if ab.nope or ab.node_permute:
            raise ValueError(f"{canon} requires rollout_fn to transform the batch/graph.")
        wrapped = wrap_policy_for_ablations(model, ab)  # type: ignore[arg-type]
        return {
            "model": wrapped,
            "tag": tag,
            "meta": {
                "ablation": canon,
                "ablation_cfg": asdict(ab),  # <-- ADD THIS
            },
        }

    def rollout_fn_ab(m: nn.Module, batch: dict) -> tuple[Tensor, dict]:
        # rollout_with_ablation does the wrapping + batch transforms internally
        return rollout_with_ablation(
            base_policy=m,  # policy-like module
            rollout_fn=rollout_fn,
            batch=batch,
            cfg=ab,
        )

    return {
        "model": model,
        "dataloader_factory": dataloader_factory,
        "rollout_fn": rollout_fn_ab,
        "tag": tag,
        "meta": {
            "ablation": canon,
            "pe_dim": ab.pe_dim,
            "perm_seed": ab.perm_seed,
            "ablation_cfg": asdict(ab),  # <-- ADD THIS
        },
    }





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
    base_policy: nn.Module,
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


    # 1) Prepare batch clone and apply NoPE if requested
    b = _clone_batch_shallow(batch)
    X: Tensor = b["X"]
    if not isinstance(X, Tensor):
        raise ValueError("batch['X'] must be a torch.Tensor")
    if cfg.nope:
        b["X"] = apply_nope_to_X(
            X,
            pe_dim=(int(cfg.pe_dim) if isinstance(cfg.pe_dim, int) else None),
            keep_last_indicator=bool(cfg.nope_keep_last_indicator),
            allow_auto=bool(cfg.nope_allow_auto),
        )
     


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



        inv = meta["invperm"]  # old -> new
        p = meta["perm"]       # new -> old

        # Unpermute probabilities: produce P over ORIGINAL node order.
        # P_old[old] = P_new[ inv[old] ]  => gather with inv in old-order.
        
        
        
        
        
        if P.ndim == 2:
            P = P[:, inv]
        elif P.ndim == 3:
            P = P[:, inv, :]
        else:
            # Keep generic, but typical use is (B,N)
            pass

        # Restore targets if they are node indices
        if "targets" in aux and isinstance(aux["targets"], Tensor) and aux["targets"].dim() == 1:
            # aux["targets"] are in NEW labeling; map NEW -> OLD using p.
            t_new = aux["targets"]
            aux["targets"] = p[t_new]

    return P, aux

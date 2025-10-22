# acpl/train/ppo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from acpl.policy.policy import ACPLPolicy

__all__ = [
    "PPOConfig",
    "RewardFn",
    "PPORollout",
    "ppo_episode",
    "ppo_update",
    "ppo_train_epoch",
]


# ------------------------------ Protocols & Types ------------------------------ #


class StepFn(Protocol):
    """
    Differentiable-or-not one-step DTQW update (we do not rely on autograd in PPO).
    Must apply local coins then the flip-flop shift (or your chosen variant).
    """

    def __call__(self, state: Tensor, coin_t: Tensor, ctx: dict[str, Any]) -> Tensor: ...


class ProbsFn(Protocol):
    """
    Produce position marginals from the full state (see theory §12).
    Returns (N,) real vector that sums ~1.
    """

    def __call__(self, state: Tensor, ctx: dict[str, Any]) -> Tensor: ...


class RewardFn(Protocol):
    """
    Non-differentiable oracle / task scorer.

    Should return a scalar reward (to MAXIMIZE). PPO minimizes -reward internally.
    It may depend on terminal marginals P_T and/or terminal state.

    Return:
        reward: scalar Tensor
        info:   dict of scalar Tensors for logging (optional)
    """

    def __call__(
        self, P_T: Tensor | None, state_T: Tensor | None, ctx: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Tensor]]: ...


# ------------------------------ Configuration ------------------------------ #


@dataclass
class PPOConfig:
    """
    PPO hyperparameters for sequence-level policy optimization.

    We model the entire schedule θ ∈ R^{T×N×P} as one 'action' drawn from a factorized
    Normal N(μ, σ^2). The policy network outputs μ (deterministic head). We keep
    a small log-std vector per parameter channel (P) shared across (t, v), which is
    a strong inductive bias (cf. report): temporal & spatial correlations are provided
    by μ via the GNN+controller; exploration scale is shared.

    Notes on stability (Phase B5):
    - We monitor approximate KL and early-stop an update if it exceeds target_kl.
    - Global gradient clipping prevents rare spikes (matrix exponential/Cayley Jacobians
      are not used here, but we keep the guard for consistency with backprop training).
    """

    gamma: float = 1.0  # episodic, typically only terminal reward → gamma=1 is fine
    gae_lambda: float = 1.0  # GAE λ; for pure terminal reward, advantage = R - V
    clip_coef: float = 0.2  # PPO ε for clip
    ent_coef: float = 0.01  # entropy bonus coefficient
    vf_coef: float = 0.5  # value loss coefficient
    max_grad_norm: float | None = 1.0  # global grad clip
    train_iters: int = 4  # epochs over the batch
    minibatch_size: int = 8  # episodes per minibatch
    target_kl: float | None = 0.03  # early stop if mean KL > target
    angle_wrap: bool = True  # wrap SU(2) Euler angles to (-pi, pi]
    std_init: float = 0.3  # initial exploration std (radians for angles)
    std_min: float = 1e-3  # floor to avoid collapse
    std_max: float = 3.14  # cap to avoid extreme ratios
    dtype: torch.dtype = torch.float32  # policy scalar dtype for μ/std (angles are float)
    complex_dtype: torch.dtype = torch.complex64  # simulator dtype
    # Optimizer (simple Adam defaults; tune by config files)
    lr: float = 3e-4
    weight_decay: float = 0.0


# ------------------------------ Utilities ------------------------------ #


def _wrap_angles(theta: Tensor) -> Tensor:
    """
    Map unconstrained angles to (-pi, pi], elementwise.
    This preserves exploration while avoiding pathological drift.
    """
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi


def _coins_from_angles(policy: ACPLPolicy, theta: Tensor, *, dtype: torch.dtype) -> Tensor:
    """
    Map SU(2) Euler angles θ (T, N, 3) → coins (T, N, 2, 2) complex.
    """
    T_, N_, P = theta.shape
    if P != 3:
        raise RuntimeError(f"PPO currently supports SU(2) degrees only (P=3). Got P={P}.")
    flat = theta.reshape(T_ * N_, 3)
    coins_flat = policy._su2_from_euler_batch(flat, dtype=dtype)
    return coins_flat.view(T_, N_, 2, 2).contiguous()


def _simulate_episode(
    *,
    coins: Tensor,  # (T, N, 2, 2) complex
    psi0: Tensor,  # complex state
    step_fn: StepFn,
    probs_fn: ProbsFn | None,
    sim_ctx: dict[str, Any],
) -> tuple[Tensor, Tensor | None]:
    """
    Roll forward T steps with provided coins. No autograd needed in PPO.
    Returns (state_T, P_T or None).
    """
    state = psi0
    T_steps = coins.size(0)
    for t in range(T_steps):
        state = step_fn(state, coins[t], sim_ctx)
    P_T = probs_fn(state, sim_ctx) if probs_fn is not None else None
    return state, P_T


def _approx_kl_normal(
    old_mean: Tensor, old_logstd: Tensor, new_mean: Tensor, new_logstd: Tensor
) -> Tensor:
    """
    KL(N_old || N_new) for factorized Normals, summed over dims then averaged over batch.
    """
    # Shapes: [B, D] across flattened schedule dims
    old_var = (old_logstd.exp()) ** 2
    new_var = (new_logstd.exp()) ** 2
    # KL per dim
    kl = (new_logstd - old_logstd) + (old_var + (old_mean - new_mean) ** 2) / (2.0 * new_var) - 0.5
    # Sum over dims then mean over batch
    return kl.sum(dim=1).mean()


# ------------------------------ Value Baseline ------------------------------ #


class EpisodeValue(nn.Module):
    """
    Lightweight critic V(s0) ≈ expected terminal reward.

    We reuse the policy GNN encoder to get node embeddings Z, then pool them and feed
    a small MLP. This keeps permutation-equivariance and couples the baseline with the
    same structural summary driving μ (the mean schedule).

    Inputs:
        pooled: (B, H) pooled node features (we let the caller provide them)

    Forward:
        values: (B,) scalar value estimates
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x).squeeze(-1)


# ------------------------------ PPO Agent State ------------------------------ #


@dataclass
class PPORollout:
    """
    Stores a batch of sequence-level episodes for PPO.
    All tensors are episode-batched (B = number of episodes collected this round).
    """

    mean_theta: Tensor  # (B, T, N, 3)
    actions_theta: Tensor  # (B, T, N, 3)
    logprob: Tensor  # (B,)  total log-prob under old policy
    entropy: Tensor  # (B,)  total entropy (sum over dims)
    value: Tensor  # (B,)  V(s0) baseline
    reward: Tensor  # (B,)  scalar episodic reward
    pooled_feats: Tensor  # (B, H) critic inputs (pooled encoder embeddings)
    info: list[dict[str, Tensor]]  # per-episode infos for logging


class _DiagGauss:
    """
    Factorized Normal over a flattened schedule. Handles logprob/entropy.
    std is broadcast per-parameter channel (size P), shared across time+nodes.
    """

    def __init__(self, mean: Tensor, log_std: Tensor):
        # mean: (..., D) or (B, D)
        # log_std: (P,) will be expanded
        self.mean = mean
        self.log_std_param = log_std  # (P,)
        # Expand to match mean's last-dim
        P = log_std.numel()
        D = mean.shape[-1]
        if D % P != 0:
            raise RuntimeError(f"log_std size {P} must divide action dim {D}.")
        reps = D // P
        self.log_std = log_std.view(1, P).repeat(1, reps).view(D)

    def sample(self) -> tuple[Tensor, Tensor]:
        std = self.log_std.exp().clamp_min(1e-8)
        eps = torch.randn_like(self.mean)
        action = self.mean + std * eps
        return action, eps

    def log_prob(self, x: Tensor) -> Tensor:
        std = self.log_std.exp().clamp_min(1e-8)
        var = std**2
        # sum over dims
        return (
            -0.5 * (((x - self.mean) ** 2) / var + 2 * self.log_std + torch.log(2 * torch.pi))
        ).sum(dim=-1)

    def entropy(self) -> Tensor:
        # sum over dims
        D = self.mean.shape[-1]
        return (
            (0.5 + 0.5 * torch.log(2 * torch.pi) + self.log_std).sum().expand(self.mean.shape[:-1])
        )


class _StdHead(nn.Module):
    """
    Trainable log-std parameters per SU(2) angle channel (P=3), shared across nodes and time.
    """

    def __init__(self, P: int = 3, init: float = 0.3, std_min: float = 1e-3, std_max: float = 3.14):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((P,), float(init)).log())
        self.std_min = std_min
        self.std_max = std_max

    def forward(self) -> Tensor:
        # clamp in std space for numerical stability & exploration sanity
        std = self.log_std.exp().clamp(self.std_min, self.std_max)
        return std.log()


# ------------------------------ Core Episode Rollout ------------------------------ #


@torch.no_grad()
def ppo_episode(
    *,
    policy: ACPLPolicy,
    X: Tensor,  # (N, Fin)
    edge_index: Tensor,  # (2, E)
    psi0: Tensor,  # complex
    T: int,
    step_fn: StepFn,
    probs_fn: ProbsFn | None,
    reward_fn: RewardFn,
    cfg: PPOConfig,
    std_head: _StdHead,
    edge_weight: Tensor | None = None,
    sim_ctx: dict[str, Any] | None = None,
) -> PPORollout:
    """
    Sample a schedule from the stochastic policy, simulate, and score a single episode.

    Observation for critic:
      We reuse the encoder's pooled node embeddings (mean over nodes) as a permutation-equivariant summary of G.
    """
    device = X.device
    sim_ctx = {} if sim_ctx is None else sim_ctx

    # Deterministic mean schedule from the policy (T, N, 3)
    mean_theta = policy(X, edge_index, T=T, edge_weight=edge_weight).to(dtype=cfg.dtype)

    if cfg.angle_wrap:
        mean_theta = _wrap_angles(mean_theta)

    B = 1  # single-episode; we will stack outside when collecting a batch
    T_, N_, P = mean_theta.shape
    D = T_ * N_ * P

    # Build diagonal Gaussian over flattened schedule with channel-shared std
    log_std_vec = std_head().to(device)
    dist = _DiagGauss(mean_theta.reshape(1, D), log_std_vec)  # (1, D)

    act_flat, _ = dist.sample()
    actions_theta = act_flat.reshape(T_, N_, P)
    if cfg.angle_wrap:
        actions_theta = _wrap_angles(actions_theta)

    # Build coins & simulate
    coins = _coins_from_angles(policy, actions_theta, dtype=cfg.complex_dtype)
    state_T, P_T = _simulate_episode(
        coins=coins, psi0=psi0, step_fn=step_fn, probs_fn=probs_fn, sim_ctx=sim_ctx
    )

    # Reward from oracle (maximize)
    reward, info = reward_fn(P_T, state_T, sim_ctx)
    reward = reward.to(device).reshape(1)

    # Log-prob & entropy under current (old) policy
    logprob = dist.log_prob(act_flat)  # (1,)
    entropy = dist.entropy()  # (1,)

    # Critic input: pooled node embeddings from policy encoder (recompute once)
    with torch.no_grad():
        # policy exposes encoder via .encode_nodes(...) if implemented; otherwise recompute through forward path cheaply
        # We use an internal helper: ACPLPolicy should cache the last Z when calling forward; if not, fall back.
        if hasattr(policy, "cached_node_embeddings") and policy.cached_node_embeddings is not None:
            pooled = policy.cached_node_embeddings.mean(dim=0, keepdim=True)  # (1, H)
        elif hasattr(policy, "encode_nodes"):
            Z = policy.encode_nodes(X, edge_index, edge_weight=edge_weight)  # (N, H)
            pooled = Z.mean(dim=0, keepdim=True)  # (1, H)
        else:
            # lightweight: call once with T=1 and ignore angles; take internal embeddings if exposed
            Z = policy.encode_nodes(X, edge_index, edge_weight=edge_weight)  # type: ignore[attr-defined]
            pooled = Z.mean(dim=0, keepdim=True)

    return PPORollout(
        mean_theta=mean_theta.unsqueeze(0),  # (1, T, N, 3)
        actions_theta=actions_theta.unsqueeze(0),
        logprob=logprob,  # (1,)
        entropy=entropy,  # (1,)
        value=torch.zeros_like(reward),  # placeholder; filled by critic in update
        reward=reward,  # (1,)
        pooled_feats=pooled,  # (1, H)
        info=[info],
    )


# ------------------------------ PPO Update ------------------------------ #


def _compute_advantages(
    rewards: Tensor, values: Tensor, gamma: float, gae_lambda: float
) -> tuple[Tensor, Tensor]:
    """
    GAE over a sequence of length 1 (sequence-level action) reduces to A = R - V.
    We keep the general form for future per-step extensions.
    """
    # rewards, values: (B,)
    deltas = rewards - values
    advantages = deltas  # single-step
    returns = advantages + values
    return advantages, returns


def _flat(x: Tensor) -> Tensor:
    return x.reshape(x.size(0), -1)


def ppo_update(
    *,
    policy: ACPLPolicy,
    critic: EpisodeValue,
    std_head: _StdHead,
    batch: PPORollout,
    cfg: PPOConfig,
    optim: torch.optim.Optimizer | None = None,
) -> dict[str, Tensor]:
    """
    Perform multiple PPO epochs over a batch of episodes.
    The policy's mean μ is produced by its forward pass during the update (on the same X/graph),
    so the caller must provide rollouts collected *just before* this call.
    """
    device = batch.mean_theta.device
    B, T_, N_, P = batch.mean_theta.shape
    D = T_ * N_ * P

    # Construct optimizer lazily if not provided
    if optim is None:
        params = list(policy.parameters()) + list(critic.parameters()) + list(std_head.parameters())
        optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Compute baseline values for the batch (forward critic on pooled feats)
    values = critic(batch.pooled_feats.detach())
    with torch.no_grad():
        adv, ret = _compute_advantages(batch.reward, values, cfg.gamma, cfg.gae_lambda)
        # advantage normalization (recommended)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    # Freeze old stats
    old_logprob = batch.logprob.detach()
    old_mean_flat = _flat(batch.mean_theta).detach()  # (B, D)
    old_logstd = (
        std_head().detach().expand(1, P).repeat(1, D // P).view(D).unsqueeze(0)
    )  # (1, D) broadcast

    # Minibatch indices
    inds = torch.arange(B, device=device)

    logs: dict[str, Tensor] = {}
    for epoch in range(cfg.train_iters):
        # shuffle
        perm = inds[torch.randperm(B, device=device)]
        for start in range(0, B, cfg.minibatch_size):
            mb_idx = perm[start : start + cfg.minibatch_size]

            mb_actions = _flat(batch.actions_theta[mb_idx])  # (M, D)
            mb_old_logprob = old_logprob[mb_idx]  # (M,)
            mb_adv = adv[mb_idx]  # (M,)
            mb_ret = ret[mb_idx]  # (M,)
            mb_pooled = batch.pooled_feats[mb_idx]  # (M, H)

            # Critic
            new_values = critic(mb_pooled)
            v_loss = F.mse_loss(new_values, mb_ret)

            # Policy: recompute mean μ and current log_std
            # NOTE: μ depends on (G, X) and T — but the batch may contain different graphs.
            # Here, the caller collected episodes possibly on different graphs, so mean_theta
            # in the batch is used as the *current* μ proxy for KL/ratio stability (sequence-level PPO).
            # If you wish to reforward the policy, collect the episode-graph tensors and refwd here.
            new_mean_flat = _flat(batch.mean_theta[mb_idx]).detach()  # (M, D)

            log_std_vec = std_head()
            dist_new = _DiagGauss(new_mean_flat, log_std_vec)
            new_logprob = dist_new.log_prob(mb_actions)  # (M,)

            # Ratio & clipped surrogate
            ratio = torch.exp(new_logprob - mb_old_logprob)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
            pi_loss = -(torch.min(surr1, surr2)).mean()

            # Entropy bonus (use old entropy in batch to save cost; or recompute)
            ent = batch.entropy[mb_idx].mean()
            loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters())
                    + list(critic.parameters())
                    + list(std_head.parameters()),
                    cfg.max_grad_norm,
                )
            optim.step()

        # KL monitor (approximate; factorized Gaussians)
        with torch.no_grad():
            # Using first minibatch to approximate KL between old and new std/means
            new_logstd = std_head().detach().expand(1, P).repeat(1, D // P).view(D).unsqueeze(0)
            kl = _approx_kl_normal(old_mean_flat, old_logstd, old_mean_flat, new_logstd)
            logs[f"kl_epoch{epoch}"] = kl
            if cfg.target_kl is not None and kl > cfg.target_kl:
                logs["early_stop_epoch"] = torch.tensor(float(epoch))
                break

    # Final logs
    with torch.no_grad():
        logs.update(
            dict(
                loss_pi=pi_loss.detach(),
                loss_v=v_loss.detach(),
                entropy=batch.entropy.mean(),
                ret_mean=ret.mean(),
                ret_std=ret.std(unbiased=False),
                adv_mean=adv.mean(),
                adv_std=adv.std(unbiased=False),
                reward_mean=batch.reward.mean(),
                std=torch.exp(std_head().detach()).mean(),
            )
        )
    return logs


# ------------------------------ High-level Trainer ------------------------------ #


def ppo_train_epoch(
    *,
    policy: ACPLPolicy,
    critic: EpisodeValue,
    std_head: _StdHead,
    episodes: list[tuple[Tensor, Tensor, Tensor, int, Tensor | None, dict[str, Any]]],
    # episodes list items: (X, edge_index, psi0, T, edge_weight, sim_ctx)
    step_fn: StepFn,
    probs_fn: ProbsFn | None,
    reward_fn: RewardFn,
    cfg: PPOConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Tensor]:
    """
    Collect one PPO batch from the provided iterable of episode specs and update once.

    Args:
        episodes: list of per-episode tensors/specs (already on device).
                  Each item: (X, edge_index, psi0, T, edge_weight, sim_ctx)

    Returns:
        logs: dict of scalar tensors for monitoring.
    """
    # Rollout (no grad)
    batch_list: list[PPORollout] = []
    for X, edge_index, psi0, T, edge_weight, sim_ctx in episodes:
        ep = ppo_episode(
            policy=policy,
            X=X,
            edge_index=edge_index,
            psi0=psi0,
            T=T,
            step_fn=step_fn,
            probs_fn=probs_fn,
            reward_fn=reward_fn,
            cfg=cfg,
            std_head=std_head,
            edge_weight=edge_weight,
            sim_ctx=sim_ctx,
        )
        batch_list.append(ep)

    # Stack across episodes
    mean_theta = torch.cat([b.mean_theta for b in batch_list], dim=0)
    actions_theta = torch.cat([b.actions_theta for b in batch_list], dim=0)
    logprob = torch.cat([b.logprob for b in batch_list], dim=0)
    entropy = torch.cat([b.entropy for b in batch_list], dim=0)
    reward = torch.cat([b.reward for b in batch_list], dim=0)
    pooled = torch.cat([b.pooled_feats for b in batch_list], dim=0)

    batch = PPORollout(
        mean_theta=mean_theta,
        actions_theta=actions_theta,
        logprob=logprob,
        entropy=entropy,
        value=torch.zeros_like(reward),
        reward=reward,
        pooled_feats=pooled,
        info=[info for b in batch_list for info in b.info],
    )

    # Update
    logs = ppo_update(
        policy=policy,
        critic=critic,
        std_head=std_head,
        batch=batch,
        cfg=cfg,
        optim=optimizer,
    )

    # Aggregate oracle info (best-effort)
    with torch.no_grad():
        for k in batch.info[0].keys() if batch.info else []:
            try:
                logs[f"oracle_{k}"] = torch.stack([d[k] for d in batch.info]).mean()
            except Exception:
                pass

    # Useful diagnostics for schedule stats
    with torch.no_grad():
        if cfg.angle_wrap:
            delta = (_wrap_angles(actions_theta) - _wrap_angles(mean_theta)).abs()
        else:
            delta = (actions_theta - mean_theta).abs()
        logs["mean_abs_deviation"] = delta.mean()

    return logs


# ------------------------------ Convenience Builder ------------------------------ #


def build_ppo_objects(
    policy: ACPLPolicy, pooled_dim: int, cfg: PPOConfig
) -> tuple[EpisodeValue, _StdHead, torch.optim.Optimizer]:
    """
    Helper to create the critic, std head, and a default optimizer over policy+critic+std.
    """
    critic = EpisodeValue(in_dim=pooled_dim, hidden=128)
    std_head = _StdHead(P=3, init=cfg.std_init, std_min=cfg.std_min, std_max=cfg.std_max)
    params = list(policy.parameters()) + list(critic.parameters()) + list(std_head.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return critic, std_head, optim

# acpl/train/backprop.py
from __future__ import annotations

import torch

from acpl.objectives.transfer import loss_state_transfer
from acpl.policy.policy import GNNTemporalPolicy
from acpl.sim.coins import coins_su2_from_theta
from acpl.sim.portmap import PortMap
from acpl.sim.step import step as qw_step
from acpl.sim.utils import complex_dtype_for, partial_trace_position


@torch.no_grad()
def _ensure_complex(psi: torch.Tensor) -> torch.Tensor:
    """Cast state to a complex dtype if needed (preserves device)."""
    if psi.is_complex():
        return psi
    cdt = complex_dtype_for(psi.dtype)
    return psi.to(dtype=cdt)


def rollout_and_loss(
    policy: GNNTemporalPolicy,
    pm: PortMap,
    edge_index: torch.Tensor,
    x: torch.Tensor,
    *,
    pos_enc: torch.Tensor | None = None,
    steps: int,
    psi0: torch.Tensor,
    target_index: int | torch.Tensor,
    reduction: str = "mean",
    record_traj: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Unroll T steps of the DTQW with a policy-driven coin and compute transfer loss.

    Parameters
    ----------
    policy : GNNTemporalPolicy
        Composed {GCN, GRU, Head}.
    pm : PortMap
        Port map defining arcs, reverse-map, and vertex segmentation.
    edge_index : torch.Tensor, shape (2, E)
        Graph connectivity for the GCN.
    x : torch.Tensor, shape (N, F)
        Node features (optionally concat'd with positional encodings outside).
    pos_enc : torch.Tensor, optional, shape (N, P)
        Extra features that the policy can concatenate internally.
    steps : int
        Number of DTQW steps to simulate.
    psi0 : torch.Tensor, shape (A,) or (A, B)
        Initial arc state. Real tensors are promoted to complex automatically.
    target_index : int or torch.Tensor
        Target vertex (scalar) or per-batch targets of shape (B,).
    reduction : {"mean","sum","none"}
        Reduction across batch when psi is (A, B).
    record_traj : bool
        If True, also return per-step distributions and/or angles.

    Returns
    -------
    loss : torch.Tensor
        Scalar (reduction applied if batched).
    info : dict
        Tensors useful for logging:
          - "psi_T": (A,) or (A,B) final state
          - "p_T": (V,) or (V,B) final position distribution
          - "h_final": (N, H_out) final controller state
          - optionally, if record_traj=True:
                "p_traj": (steps, V) or (steps, V, B)
                "theta_traj": (steps, N, 3)
    """
    if steps <= 0:
        raise ValueError("steps must be >= 1")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if x.ndim != 2:
        raise ValueError("x must have shape (N, F)")

    dev = x.device
    psi = _ensure_complex(psi0.to(device=dev))
    dest = torch.from_numpy(pm.rev).to(device=dev, dtype=torch.long)

    n = x.size(0)
    theta_traj = []
    p_traj = []

    h_prev: torch.Tensor | None = None

    # Unroll
    for t in range(steps):
        t_over_t = float(t) / float(max(1, steps - 1)) if steps > 1 else 0.0

        # Policy → Euler angles for this step
        theta_vt, h_next = policy(
            g=edge_index,
            x=x,
            pos_enc=pos_enc,
            t_over_t=t_over_t,
            h_prev=h_prev,
        )  # theta_vt: (N,3), h_next: (N,H_out)

        # Lift to per-vertex SU(2) blocks and take one DTQW step
        c_blocks = coins_su2_from_theta(theta_vt)  # (N,2,2)
        psi = qw_step(psi, pm, c_blocks, dest=dest)  # (A,) or (A,B)

        if record_traj:
            p_t = partial_trace_position(psi, pm)  # (V,) or (V,B)
            p_traj.append(p_t)
            theta_traj.append(theta_vt.detach())  # angles are fine to detach for logging

        h_prev = h_next  # roll hidden

    # Final distribution and loss
    p_final = partial_trace_position(psi, pm)  # (V,) or (V,B)
    loss = loss_state_transfer(
        p_final,
        target_index,
        reduction=reduction,
        check_prob=False,
        renorm=True,
    )

    info: dict[str, torch.Tensor] = {
        "psi_T": psi,
        "p_T": p_final,
        "h_final": h_prev if h_prev is not None else torch.zeros(n, 0, device=dev),
    }
    if record_traj:
        # Stack along time: (steps, V[,B]) and (steps, N, 3)
        if p_traj:
            if p_final.ndim == 1:
                info["p_traj"] = torch.stack(p_traj, dim=0)  # (T, V)
            else:
                info["p_traj"] = torch.stack(p_traj, dim=0)  # (T, V, B)
        if theta_traj:
            info["theta_traj"] = torch.stack(theta_traj, dim=0)  # (T, N, 3)

    return loss, info

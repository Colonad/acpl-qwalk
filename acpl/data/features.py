# acpl/data/features.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

__all__ = [
    "FeatureSpec",
    "build_node_features",
    "node_features_line",  # (degree + 1D normalized coord)
    "laplacian_positional_encoding",
    "random_walk_structural_encoding",
    "normalize_degree",
    "degree_one_hot",
    "sinusoidal_coords",
    "build_arc_features",
]


# --------------------------------------------------------------------------------------
# Spec and simple helpers
# --------------------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """
    Configuration of node- and arc-level features.

    Node features (stacked in this order when enabled):
      1) degree statistics (raw, normalized)
      2) degree one-hot up to K (optional)
      3) geometry coords (as provided by graphs)
      4) sinusoidal coords (periodic embeddings of coords)
      5) Laplacian Positional Encodings (LapPE, top-k eigenvectors)
      6) Random-Walk Structural Encoding (RWSE) up to K steps

    Arc features (optional):
      - direction vector (dst - src), L2 normalized (+ distance)
      - local polar angle on 2D coords (if available)

    Notes
    -----
    - All tensors are returned as float32 by default except where noted.
    - LapPE is sign-deterministic via 'random_sign=False' or seeded generator.
    """

    # Node
    use_degree: bool = True
    degree_norm: Literal["none", "max", "log", "inv_sqrt"] = "inv_sqrt"
    degree_onehot_K: int = 0  # 0 disables one-hot
    use_coords: bool = True
    use_sinusoidal_coords: bool = False
    sinusoidal_dims: int = 0  # if >0 and use_sinusoidal_coords=True
    sinusoidal_base: float = 10000.0
    use_lap_pe: bool = False
    lap_pe_k: int = 0  # number of eigenvectors
    lap_pe_norm: Literal["sym", "rw"] = "sym"
    lap_pe_random_sign: bool = True
    use_rwse: bool = False
    rwse_K: int = 0  # steps for P^k
    rwse_aggregation: Literal["row"] = "row"  # reserved for future
    # Arc
    build_arcs: bool = False
    arc_use_direction: bool = True
    arc_use_distance: bool = True
    arc_use_angle2d: bool = True  # only if coords dim==2

    # Misc
    eps: float = 1e-12
    seed: int | None = None  # for deterministic LapPE sign jitter, etc.


# --------------------------------------------------------------------------------------
# Degree features and simple encodings
# --------------------------------------------------------------------------------------


def normalize_degree(
    degrees: torch.Tensor, mode: str = "inv_sqrt", eps: float = 1e-12
) -> torch.Tensor:
    """
    Normalize degrees into a single scalar feature per node.

    modes:
      - "none": return degrees as float
      - "max":  d / max(d)
      - "log":  log1p(d)
      - "inv_sqrt": 1 / sqrt(max(d,1))
    """
    d = degrees.to(torch.float32)
    if mode == "none":
        return d
    if mode == "max":
        m = torch.clamp(d.max(), min=1.0)
        return d / m
    if mode == "log":
        return torch.log1p(d)
    if mode == "inv_sqrt":
        return 1.0 / torch.sqrt(torch.clamp(d, min=1.0))
    raise ValueError(f"Unknown degree normalization mode: {mode}")


def degree_one_hot(degrees: torch.Tensor, K: int) -> torch.Tensor:
    """
    Degree one-hot up to K (clamps larger degrees into the last bin).
    Returns (N, K+1) float32 if K>0 else empty (N,0).
    """
    if K <= 0:
        return torch.zeros(degrees.numel(), 0, dtype=torch.float32, device=degrees.device)
    d = torch.clamp(degrees, min=0, max=K).to(torch.long)
    N = degrees.numel()
    out = torch.zeros(N, K + 1, dtype=torch.float32, device=degrees.device)
    out.scatter_(1, d.view(-1, 1), 1.0)
    return out


def sinusoidal_coords(coords: torch.Tensor, dims: int, base: float = 10000.0) -> torch.Tensor:
    """
    Sinusoidal positional encoding (Vaswani-style) for each coordinate dimension.
    coords : (N, C) in [0,1] or reasonable scale
    dims   : number of features per coordinate dimension (must be even, we use sin/cos pairs)
    returns: (N, C*dims)
    """
    if dims <= 0:
        return torch.zeros(coords.shape[0], 0, dtype=torch.float32, device=coords.device)
    if dims % 2 != 0:
        raise ValueError("sinusoidal_dims must be even (sin/cos pairs).")
    N, C = coords.shape
    pe_list = []
    # frequencies
    half = dims // 2
    div = torch.exp(
        torch.arange(0, half, dtype=torch.float32, device=coords.device)
        * (-math.log(base) / max(half - 1, 1))
    )
    for c in range(C):
        x = coords[:, c].unsqueeze(1)  # (N,1)
        scaled = x * div  # (N, half)
        pe_list.append(torch.sin(scaled))
        pe_list.append(torch.cos(scaled))
    return torch.cat(pe_list, dim=1)


# --------------------------------------------------------------------------------------
# Laplacian PE (robust, sparse-friendly)
# --------------------------------------------------------------------------------------


def _build_undirected_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Build a symmetric sparse adjacency (float32) from oriented arcs edge_index (2,A).
    Self-loops removed.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be (2, A)")
    src, dst = edge_index[0], edge_index[1]
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    # Take only one direction for undirected (min,max) coalescing
    a = torch.minimum(src, dst)
    b = torch.maximum(src, dst)
    undirected = torch.stack([a, b], dim=0)
    # sort/unique
    key = a * num_nodes + b
    uniq, idx = torch.unique(key, sorted=True, return_inverse=False, return_counts=False), None
    # recover pairs from uniq keys
    uu = (uniq // num_nodes).to(torch.long)
    vv = (uniq % num_nodes).to(torch.long)

    # Now build symmetric COO
    ii = torch.cat([uu, vv], dim=0)
    jj = torch.cat([vv, uu], dim=0)
    vv_data = torch.ones_like(ii, dtype=torch.float32)
    A = torch.sparse_coo_tensor(torch.stack([ii, jj]), vv_data, (num_nodes, num_nodes))
    return A.coalesce()


def _laplacian_from_adj(A: torch.Tensor, mode: str = "sym", eps: float = 1e-12) -> torch.Tensor:
    """
    Return sparse Laplacian:
      - mode "sym": L = I - D^{-1/2} A D^{-1/2}
      - mode "rw" : L_rw = I - D^{-1} A
    """
    assert A.is_sparse
    N = A.size(0)
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(0.0)  # (N,)
    I_idx = torch.arange(N, device=A.device)
    if mode == "sym":
        dinv_sqrt = (deg + eps).pow(-0.5)
        # scale values of A by D^{-1/2} row and col
        idx = A.indices()
        val = A.values()
        v = dinv_sqrt[idx[0]] * val * dinv_sqrt[idx[1]]
        S = torch.sparse_coo_tensor(idx, v, A.shape).coalesce()
        # L = I - S
        I = torch.sparse_coo_tensor(
            torch.stack([I_idx, I_idx]), torch.ones(N, device=A.device), A.shape
        )
        L = I - S
        return L.coalesce()
    elif mode == "rw":
        dinv = (deg + eps).reciprocal()
        idx = A.indices()
        val = A.values()
        v = dinv[idx[0]] * val
        P = torch.sparse_coo_tensor(idx, v, A.shape).coalesce()  # row-stochastic
        I = torch.sparse_coo_tensor(
            torch.stack([I_idx, I_idx]), torch.ones(N, device=A.device), A.shape
        )
        Lrw = I - P
        return Lrw.coalesce()
    else:
        raise ValueError("mode must be 'sym' or 'rw'.")


@torch.no_grad()
def laplacian_positional_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int,
    *,
    mode: Literal["sym", "rw"] = "sym",
    random_sign: bool = True,
    seed: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute top-k (smallest) Laplacian eigenvectors as positional encodings.

    - Works with disconnected graphs: we compute per-component eigenvectors implicitly
      via full sparse->dense fallback (small graphs) or dense on small N;
      for typical N (<= few thousands), dense eig is acceptable here.
    - Returns (N, k) float32. If k==0 → shape (N,0).
    - random_sign: multiply each eigenvector by ±1 consistently (fixes sign ambiguity).
    """
    if k <= 0:
        return torch.zeros(num_nodes, 0, dtype=torch.float32, device=edge_index.device)
    A = _build_undirected_adj(edge_index, num_nodes)
    L = _laplacian_from_adj(A, mode=mode, eps=eps)

    # Dense fallback
    Ld = L.to_dense()
    # eigh for symmetric (real)
    evals, evecs = torch.linalg.eigh(Ld)  # ascending
    # take the smallest k (skip exact zeros if needed? keep as-is; zeros capture components)
    evecs_k = evecs[:, :k].to(torch.float32)

    if random_sign:
        gen = torch.Generator(device=evecs_k.device)
        if seed is not None:
            gen.manual_seed(seed)
        signs = torch.where(
            torch.rand(1, k, generator=gen, device=evecs_k.device) > 0.5, 1.0, -1.0
        ).to(evecs_k.dtype)
        evecs_k = evecs_k * signs  # broadcast across rows

    return evecs_k


# --------------------------------------------------------------------------------------
# RWSE: Random-walk structural encoding
# --------------------------------------------------------------------------------------


@torch.no_grad()
def random_walk_structural_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    K: int,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    RWSE up to K steps using the row-stochastic transition matrix P = D^{-1}A.
    Returns (N, K) with columns corresponding to diag entries of P^k (or row-sums over powers).

    We build P in sparse COO, then multiply dense vectors iteratively.
    For simplicity and stability, we compute 'landing probability at the same node'
    across k steps: diag(P^k). Realistically, you can also use row-sums of P^k (which are 1).
    """
    if K <= 0:
        return torch.zeros(num_nodes, 0, dtype=torch.float32, device=edge_index.device)

    # Build P
    A = _build_undirected_adj(edge_index, num_nodes)  # symmetric 0/1
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(0.0)  # (N,)
    dinv = (deg + eps).reciprocal()
    idx = A.indices()
    val = A.values()
    v = dinv[idx[0]] * val
    P = torch.sparse_coo_tensor(idx, v, (num_nodes, num_nodes)).coalesce()

    # Efficient multiplication by P using sparse @ dense
    feats = torch.zeros(num_nodes, K, dtype=torch.float32, device=edge_index.device)
    # Start with basis vectors e_i one-by-one via diagonal extraction trick:
    # We track diag(P^k) without storing P^k: diag(P^k) = sum_j P^{k-1}_{i,j} * P_{j,i}
    # Implement iterative forward over columns of P and gather diagonal.
    # We'll use dense matmul for P@X with X dense (N,d).
    X = torch.eye(num_nodes, dtype=torch.float32, device=edge_index.device)  # (N,N)
    # To keep memory in check for moderate N; for larger N you can chunk X.
    for k in range(1, K + 1):
        X = torch.sparse.mm(P, X)  # (N,N)
        # extract diagonal
        feats[:, k - 1] = torch.diag(X)
    return feats


# --------------------------------------------------------------------------------------
# Node feature assembly
# --------------------------------------------------------------------------------------


def node_features_line(degrees: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Minimal line-graph features used in early Phase A2:
    - normalized degree (inv_sqrt)
    - 1D normalized coordinate

    Expects:
      degrees: (N,) int
      coords : (N,1) float in [0,1]
    Returns:
      (N, 2) float32
    """
    deg_feat = normalize_degree(degrees, mode="inv_sqrt")  # (N,)
    if coords.ndim != 2 or coords.shape[1] != 1:
        raise ValueError("coords must be (N,1) for node_features_line.")
    return torch.cat([deg_feat.view(-1, 1), coords.to(torch.float32)], dim=1)


def build_node_features(
    edge_index: torch.Tensor,
    degrees: torch.Tensor,
    coords: torch.Tensor,
    *,
    spec: FeatureSpec | None = None,
    device: torch.device | None = None,
) -> tuple(torch.Tensor, dict[str, tuple[int, int]]):
    """
    Assemble node features per FeatureSpec.
    Returns:
      X  : (N, F) float32
      idx: dict mapping feature-block name -> (start, end) column indices
    """
    if spec is None:
        spec = FeatureSpec()
    if device is None:
        device = degrees.device

    N = degrees.numel()
    blocks = []
    index: dict[str, tuple[int, int]] = {}
    col = 0

    # 1) degree stats
    if spec.use_degree:
        dnorm = normalize_degree(degrees.to(device), mode=spec.degree_norm, eps=spec.eps).view(
            -1, 1
        )
        blocks.append(dnorm)
        index["deg_norm"] = (col, col + 1)
        col += 1
        if spec.degree_onehot_K > 0:
            doh = degree_one_hot(degrees.to(device), spec.degree_onehot_K)
            blocks.append(doh)
            index["deg_onehot"] = (col, col + doh.shape[1])
            col += doh.shape[1]

    # 2) coords
    if spec.use_coords and coords.numel() > 0:
        C = coords.shape[1]
        c = coords.to(device, dtype=torch.float32)
        blocks.append(c)
        index["coords"] = (col, col + C)
        col += C

        if spec.use_sinusoidal_coords and spec.sinusoidal_dims > 0:
            sc = sinusoidal_coords(c, dims=spec.sinusoidal_dims, base=spec.sinusoidal_base)
            blocks.append(sc)
            index["coords_sin"] = (col, col + sc.shape[1])
            col += sc.shape[1]

    # 3) LapPE
    if spec.use_lap_pe and spec.lap_pe_k > 0:
        pe = laplacian_positional_encoding(
            edge_index=edge_index.to(device),
            num_nodes=N,
            k=spec.lap_pe_k,
            mode=spec.lap_pe_norm,
            random_sign=spec.lap_pe_random_sign,
            seed=spec.seed,
            eps=spec.eps,
        )
        blocks.append(pe)
        index["lap_pe"] = (col, col + pe.shape[1])
        col += pe.shape[1]

    # 4) RWSE
    if spec.use_rwse and spec.rwse_K > 0:
        rw = random_walk_structural_encoding(
            edge_index=edge_index.to(device),
            num_nodes=N,
            K=spec.rwse_K,
            eps=spec.eps,
        )
        blocks.append(rw)
        index["rwse"] = (col, col + rw.shape[1])
        col += rw.shape[1]

    if len(blocks) == 0:
        X = torch.zeros(N, 0, dtype=torch.float32, device=device)
    else:
        X = torch.cat(blocks, dim=1).to(torch.float32)

    return X, index


# --------------------------------------------------------------------------------------
# Optional arc features (useful for advanced policies / diagnostics)
# --------------------------------------------------------------------------------------


def build_arc_features(
    edge_index: torch.Tensor,
    coords: torch.Tensor,
    *,
    use_direction: bool = True,
    use_distance: bool = True,
    use_angle2d: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Simple geometric features per arc (i->j) given node coords:
      - direction: (x_j - x_i) / ||x_j - x_i|| (if use_direction)
      - distance : ||x_j - x_i||
      - angle2d  : atan2(Δy, Δx) in [-pi,pi], sin/cos pair (if 2D and use_angle2d)

    Returns (A, F) float32 (F depends on toggles and coord dimension).
    If coords are empty or 1D and angle requested, angle is skipped safely.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be (2, A)")
    src, dst = edge_index[0], edge_index[1]
    A = src.numel()

    if coords.numel() == 0:
        return torch.zeros(A, 0, dtype=torch.float32, device=edge_index.device)

    Xi = coords[src]  # (A, C)
    Xj = coords[dst]  # (A, C)
    d = Xj - Xi
    feats = []

    if use_direction:
        n = torch.linalg.norm(d, dim=1, keepdim=True).clamp_min(eps)
        dir_unit = d / n  # (A, C)
        feats.append(dir_unit)

    if use_distance:
        dist = torch.linalg.norm(d, dim=1, keepdim=True)
        feats.append(dist)

    if use_angle2d and coords.shape[1] >= 2:
        dx = d[:, 0]
        dy = d[:, 1]
        ang = torch.atan2(dy, dx)  # (-pi, pi)
        feats.append(torch.sin(ang).unsqueeze(1))
        feats.append(torch.cos(ang).unsqueeze(1))

    if len(feats) == 0:
        return torch.zeros(A, 0, dtype=torch.float32, device=edge_index.device)
    return torch.cat(feats, dim=1).to(torch.float32)

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
    "role_encodings",
    "hypercube_bit_features",
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
      1) degree statistics (raw-normalized scalar)
      2) degree one-hot up to K (optional)
      3) geometry coords (as provided by graphs)
      4) sinusoidal coords (periodic embeddings of coords)
      5) Laplacian Positional Encodings (LapPE, top-k eigenvectors; deterministic sign)
      6) Random-Walk Structural Encoding (RWSE) up to K steps
      7) Role encodings (leaf/hub/binning; grid-boundary flags if coords resemble a grid)
      8) Hypercube bitstrings (and Hamming weight), when graph is Q_n or coords are {0,1}^n

    Arc features (optional):
      - direction vector (dst - src), L2 normalized (+ distance)
      - local polar angle on 2D coords (if available)

    Notes
    -----
    - All tensors are float32 by default.
    - LapPE is sign-deterministic by default (see lap_pe_random_sign=False).
    """

    # Node: degrees & coords
    use_degree: bool = True
    degree_norm: Literal["none", "max", "log", "inv_sqrt"] = "inv_sqrt"
    degree_onehot_K: int = 0  # 0 disables one-hot
    use_coords: bool = True
    use_sinusoidal_coords: bool = False
    sinusoidal_dims: int = 0  # if >0 and use_sinusoidal_coords=True
    sinusoidal_base: float = 10000.0

    # Laplacian PE
    use_lap_pe: bool = False
    lap_pe_k: int = 0  # number of eigenvectors
    lap_pe_norm: Literal["sym", "rw"] = "sym"
    lap_pe_random_sign: bool = False  # Phase B2: deterministic sign is the default
    # RWSE
    use_rwse: bool = False
    rwse_K: int = 0  # steps for P^k
    rwse_aggregation: Literal["row"] = "row"  # reserved for future

    # Role encodings
    use_role_encodings: bool = False
    role_degree_bins_K: int = 0  # degree bin one-hot (0 disables)
    role_include_leaf: bool = True
    role_include_hub: bool = True
    role_hub_percentile: float = 0.90  # degree >= this percentile -> hub flag
    role_include_grid_boundary: bool = True  # if coords look grid-like, add edge/corner flags

    # Hypercube bitstrings
    use_bitstrings: bool = (
        False  # when True, attempt to consume coords as {0,1}^n (or a provided bit tensor)
    )
    bitstrings_center: Literal["none", "pm1"] = "none"  # if "pm1", map {0,1} -> {-1,+1}
    bitstrings_append_hamming: bool = True  # append Hamming weight / n (scalar)

    # Arc
    build_arcs: bool = False
    arc_use_direction: bool = True
    arc_use_distance: bool = True
    arc_use_angle2d: bool = True  # only if coords dim==2

    # Misc
    eps: float = 1e-12
    seed: int | None = None  # for optional randomized behaviors


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
    dims   : number of features *per coordinate* (must be even, we use sin/cos pairs)
    returns: (N, C*dims)
    """
    if dims <= 0:
        return torch.zeros(coords.shape[0], 0, dtype=torch.float32, device=coords.device)
    if dims % 2 != 0:
        raise ValueError("sinusoidal_dims must be even (sin/cos pairs).")
    N, C = coords.shape
    pe_list = []
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
# Laplacian PE (robust, sparse-friendly) with deterministic sign
# --------------------------------------------------------------------------------------


def _build_undirected_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Build a symmetric sparse adjacency (float32) from oriented arcs edge_index (2,A).
    Self-loops removed. Coalesces duplicates.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be (2, A)")
    src, dst = edge_index[0], edge_index[1]
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    # coalesce undirected edges via sorted endpoints
    a = torch.minimum(src, dst)
    b = torch.maximum(src, dst)
    key = a * num_nodes + b
    uniq = torch.unique(key, sorted=True)

    uu = (uniq // num_nodes).to(torch.long)
    vv = (uniq % num_nodes).to(torch.long)

    # symmetric COO
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
        idx = A.indices()
        val = A.values()
        v = dinv_sqrt[idx[0]] * val * dinv_sqrt[idx[1]]
        S = torch.sparse_coo_tensor(idx, v, A.shape).coalesce()
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


def _sign_fix_columns(M: torch.Tensor) -> torch.Tensor:
    """
    Deterministic sign fix for eigenvectors M (N, k):
      For each column, find index of max |entry| and flip the column so that entry is >= 0.
      If the max is numerically 0, leave as-is.
    """
    if M.numel() == 0:
        return M
    # idx of argmax |·| per column
    idx = torch.argmax(torch.abs(M), dim=0)  # (k,)
    signs = torch.sign(M.gather(0, idx.view(1, -1)).squeeze(0))
    signs[signs == 0] = 1.0
    return M * signs  # broadcast over rows


@torch.no_grad()
def laplacian_positional_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int,
    *,
    mode: Literal["sym", "rw"] = "sym",
    random_sign: bool = False,
    seed: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute top-k (smallest) Laplacian eigenvectors as positional encodings.

    - Works with disconnected graphs (zero eigenvalues allowed).
    - Returns (N, k) float32. If k==0 → shape (N,0).
    - Sign handling:
        * By default (random_sign=False) we apply a deterministic sign fix per eigenvector
          by making the entry with largest magnitude nonnegative.
        * If random_sign=True, we multiply each eigenvector by an independent ±1 (seedable)
          for experimentation/ablations.
    """
    if k <= 0:
        return torch.zeros(num_nodes, 0, dtype=torch.float32, device=edge_index.device)
    k_eff = int(min(k, num_nodes))
    if k_eff <= 0:
        return torch.zeros(num_nodes, 0, dtype=torch.float32, device=edge_index.device)

    A = _build_undirected_adj(edge_index, num_nodes)
    L = _laplacian_from_adj(A, mode=mode, eps=eps)

    # Dense eigendecomposition (moderate N); for larger N, swap to Lanczos if needed.
    Ld = L.to_dense()
    evals, evecs = torch.linalg.eigh(Ld)  # ascending
    evecs_k = evecs[:, :k_eff].to(torch.float32)

    if random_sign:
        gen = torch.Generator(device=evecs_k.device)
        if seed is not None:
            gen.manual_seed(seed)
        signs = torch.where(
            torch.rand(1, k_eff, generator=gen, device=evecs_k.device) > 0.5, 1.0, -1.0
        ).to(evecs_k.dtype)
        evecs_k = evecs_k * signs
    else:
        evecs_k = _sign_fix_columns(evecs_k)

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
    Returns (N, K) with columns corresponding to diag(P^k).
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

    # Track diag(P^k) iteratively with dense matmuls on an identity basis.
    feats = torch.zeros(num_nodes, K, dtype=torch.float32, device=edge_index.device)
    X = torch.eye(num_nodes, dtype=torch.float32, device=edge_index.device)  # (N,N)
    for k in range(1, K + 1):
        X = torch.sparse.mm(P, X)
        feats[:, k - 1] = torch.diag(X)
    return feats


# --------------------------------------------------------------------------------------
# Role encodings (leaf/hub/binning + optional grid boundary flags)
# --------------------------------------------------------------------------------------


@torch.no_grad()
def role_encodings(
    degrees: torch.Tensor,
    coords: torch.Tensor | None = None,
    *,
    degree_bins_K: int = 0,
    include_leaf: bool = True,
    include_hub: bool = True,
    hub_percentile: float = 0.90,
    include_grid_boundary: bool = True,
) -> torch.Tensor:
    """
    Build compact, task-agnostic structural/role encodings:
      - degree bin one-hot (0..K; K=0 disables)
      - leaf flag (deg==1)
      - hub flag (deg >= percentile threshold)
      - grid boundary flags if coords look like a regular 1D/2D integer grid or normalized [0,1] grid:
            corner, edge, interior (one-hot)

    Returns (N, R) float32, possibly (N,0) if everything is disabled or undetected.
    """
    device = degrees.device
    N = degrees.numel()
    blocks: list[torch.Tensor] = []

    # Degree binning
    if degree_bins_K > 0:
        blocks.append(degree_one_hot(degrees, degree_bins_K))

    # Leaf/hub flags
    if include_leaf:
        leaf = (degrees == 1).to(torch.float32).view(N, 1)
        blocks.append(leaf)
    if include_hub:
        th = torch.quantile(degrees.to(torch.float32), torch.tensor(hub_percentile, device=device))
        hub = (degrees.to(torch.float32) >= th).to(torch.float32).view(N, 1)
        blocks.append(hub)

    # Grid boundary flags (only if coords provided and grid-like)
    if include_grid_boundary and coords is not None and coords.numel() > 0:
        grid_flags = _grid_boundary_flags(coords)
        if grid_flags.numel() > 0:
            blocks.append(grid_flags)

    if len(blocks) == 0:
        return torch.zeros(N, 0, dtype=torch.float32, device=device)
    return torch.cat(blocks, dim=1).to(torch.float32)


def _grid_boundary_flags(coords: torch.Tensor) -> torch.Tensor:
    """
    Heuristically detect 1D/2D grid coordinates and return boundary one-hots:
      - 1D: end vs interior (one-hot 2)
      - 2D: corner / edge / interior (one-hot 3)
    Works if coords are integer-like (0..L-1) or normalized approx in [0,1].
    Returns (N, C) or (N,0) if not detected.
    """
    device = coords.device
    N, C = coords.shape
    if C not in (1, 2):
        return torch.zeros(N, 0, dtype=torch.float32, device=device)

    x = coords[:, 0]

    def is_close(a: torch.Tensor, b: float, tol: float = 1e-6) -> torch.Tensor:
        return (a - b).abs() <= tol

    # Normalize detection: try integer grid (0..L-1) or normalized [0,1]
    def norm_axis(ax: torch.Tensor) -> torch.Tensor:
        ax_min = ax.min()
        ax_max = ax.max()
        if ax_max - ax_min < 1e-9:
            return ax - ax_min  # constant axis; treat as degenerate
        axn = (ax - ax_min) / (ax_max - ax_min)
        return axn

    if C == 1:
        xn = norm_axis(x)
        at_min = is_close(xn, 0.0, 1e-6)
        at_max = is_close(xn, 1.0, 1e-6)
        end = (at_min | at_max).to(torch.float32).view(N, 1)
        interior = (~(at_min | at_max)).to(torch.float32).view(N, 1)
        return torch.cat([end, interior], dim=1)

    # C == 2
    y = coords[:, 1]
    xn = norm_axis(x)
    yn = norm_axis(y)
    at_l = is_close(xn, 0.0, 1e-6)
    at_r = is_close(xn, 1.0, 1e-6)
    at_b = is_close(yn, 0.0, 1e-6)
    at_t = is_close(yn, 1.0, 1e-6)

    on_edge = at_l | at_r | at_b | at_t
    on_corner = (at_l & at_b) | (at_l & at_t) | (at_r & at_b) | (at_r & at_t)

    corner = on_corner.to(torch.float32).view(N, 1)
    edge = (on_edge & (~on_corner)).to(torch.float32).view(N, 1)
    interior = (~on_edge).to(torch.float32).view(N, 1)
    return torch.cat([corner, edge, interior], dim=1)


# --------------------------------------------------------------------------------------
# Hypercube bitstring features
# --------------------------------------------------------------------------------------


@torch.no_grad()
def hypercube_bit_features(
    bits: torch.Tensor,
    *,
    center: Literal["none", "pm1"] = "none",
    append_hamming: bool = True,
) -> torch.Tensor:
    """
    Turn per-node bitstrings into features.

    bits: (N, n) with entries approximately in {0,1}. Non-binary values are rounded to nearest {0,1}.
    center:
      - "none": keep {0,1}
      - "pm1" : map {0,1} -> {-1,+1}
    append_hamming:
      - if True, append a single column with normalized Hamming weight (sum(bits)/n)

    Returns: (N, n [+ 1]) float32
    """
    if bits.numel() == 0:
        return torch.zeros(bits.shape[0], 0, dtype=torch.float32, device=bits.device)

    X = (bits >= 0.5).to(torch.float32)  # robust rounding
    if center == "pm1":
        X = 2.0 * X - 1.0
    blocks = [X]
    if append_hamming and X.shape[1] > 0:
        if center == "pm1":
            # For {-1,+1}, Hamming weight = (x+1)/2 summed; normalize by n
            w = ((X + 1.0) * 0.5).mean(dim=1, keepdim=True)
        else:
            w = X.mean(dim=1, keepdim=True)
        blocks.append(w)
    return torch.cat(blocks, dim=1).to(torch.float32)


def _maybe_bits_from_coords(coords: torch.Tensor) -> torch.Tensor:
    """
    If coords look like {0,1}^n (n>=1), return them as bitstrings; else return empty.
    """
    if coords is None:
        return torch.zeros(0, 0, dtype=torch.float32, device="cpu")
    # Preserve batch dimension even if coords have 0 columns.
    if coords.ndim != 2:
        raise ValueError("coords must be 2D (N,C)")
    N, C = coords.shape
    if C == 0 or coords.numel() == 0:
        return torch.zeros(N, 0, dtype=torch.float32, device=coords.device)
    N, C = coords.shape
    if C == 0:
        return torch.zeros(N, 0, dtype=torch.float32, device=coords.device)
    # Heuristic: values are within small tolerance of 0 or 1 across all dims
    vals0 = (coords - 0.0).abs().max().item()
    vals1 = (coords - 1.0).abs().max().item()
    # But that check alone is too strict; instead, check each element is close to 0 or 1
    close0 = (coords - 0.0).abs() <= 1e-6
    close1 = (coords - 1.0).abs() <= 1e-6
    mask_binary = (close0 | close1).all().item()
    if mask_binary:
        return coords.to(torch.float32)
    # Fallback: if coords are integers 0/1 (exact cast)
    if torch.equal(coords, coords.round()):
        uniq = torch.unique(coords)
        if uniq.numel() <= 2 and torch.all((uniq == 0) | (uniq == 1)):
            return coords.to(torch.float32)
    return torch.zeros(N, 0, dtype=torch.float32, device=coords.device)


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
) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    """
    Assemble node features per FeatureSpec.
    Returns:
      X  : (N, F) float32
      idx: dict mapping feature-block name -> (start, end) column indices (end is exclusive)
    """
    if spec is None:
        spec = FeatureSpec()
    if device is None:
        device = degrees.device

    N = degrees.numel()
    blocks: list[torch.Tensor] = []
    index: dict[str, tuple[int, int]] = {}
    col = 0

    # 0) Hypercube bitstrings (if requested and available)
    if spec.use_bitstrings:
        bits = _maybe_bits_from_coords(coords.to(device))
        if bits.numel() > 0:
            bf = hypercube_bit_features(
                bits,
                center=spec.bitstrings_center,
                append_hamming=spec.bitstrings_append_hamming,
            ).to(device)
            blocks.append(bf)
            index["bitstrings"] = (col, col + bf.shape[1])
            col += bf.shape[1]

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

    # 5) Role encodings
    if spec.use_role_encodings:
        roles = role_encodings(
            degrees.to(device),
            coords.to(device) if coords is not None else None,
            degree_bins_K=spec.role_degree_bins_K,
            include_leaf=spec.role_include_leaf,
            include_hub=spec.role_include_hub,
            hub_percentile=spec.role_hub_percentile,
            include_grid_boundary=spec.role_include_grid_boundary,
        )
        if roles.numel() > 0:
            blocks.append(roles)
            index["roles"] = (col, col + roles.shape[1])
            col += roles.shape[1]

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

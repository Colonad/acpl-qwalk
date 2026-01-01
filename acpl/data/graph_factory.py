# acpl/data/graph_factory.py
from __future__ import annotations


from typing import Any, Callable
import inspect
import torch


from acpl.data.graphs import (
    GraphData,
    cycle_graph,
    d_regular_random_graph,
    erdos_renyi_graph,
    grid_graph,
    hypercube_graph,
    line_graph,
    watts_strogatz_graph,
    watts_strogatz_grid_graph,
    watts_strogatz_grid_graph_degree_preserving,
)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Best-effort getter for OmegaConf/Namespace/dict-like configs."""
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    try:
        return cfg[key]  # OmegaConf / dict-like
    except Exception:
        return default






def _call_optional(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Call `fn(*args, **kwargs)` but drop kwargs that `fn` doesn't accept.
    This keeps graph_factory compatible even if graphs.py signatures evolve.
    """
    try:
        sig = inspect.signature(fn)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(*args, **allowed)
    except Exception:
        return fn(*args, **kwargs)










def _is_connected(edge_index: torch.Tensor, N: int, arc_slices: torch.Tensor | None = None) -> bool:
    """
    Connectivity check on an undirected graph represented as oriented arcs (2, A).

    IMPORTANT:
    - If `arc_slices` is provided (shape [N+1]), we use it (correct regardless of edge order).
    - Otherwise we fall back to sorting by src to build CSR safely.
    """

    
    
    if N <= 1:
        return True
    if edge_index.numel() == 0:
        return False

    dst = edge_index[1].to(torch.long)

    if arc_slices is not None and arc_slices.numel() == N + 1:
        ptr = arc_slices.to(torch.long)
    else:
        # Safe CSR build even if edge_index is not grouped by src
        src = edge_index[0].to(torch.long)
        order = torch.argsort(src)
        src = src[order]
        dst = dst[order]
        
        
        # IMPORTANT: CSR slicing assumes arcs are grouped by src.
        # Sort by src so this works regardless of edge_index ordering.
        perm = torch.argsort(src)
        src = src[perm]
        dst = dst[perm]

        # Build adjacency lists via CSR using counts on src (now sorted)        
            
        
        
        counts = torch.bincount(src, minlength=N)
        ptr = torch.zeros(N + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(counts, dim=0)

    # BFS
    seen = torch.zeros(N, dtype=torch.bool)
    q = [0]
    seen[0] = True
    while q:
        u = q.pop()
        a = int(ptr[u])
        b = int(ptr[u + 1])
        nbrs = dst[a:b]
        for v in nbrs.tolist():
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return bool(seen.all().item())


def _retry_until_connected(
    build_once: Callable[[int | None], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    keep_connected: bool,
    max_tries: int,
    seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


    if seed is None:
        if keep_connected:
            raise ValueError("keep_connected=true requires an explicit seed for determinism.")

    last = None
    last_exc: Exception | None = None
    for t in range(max_tries):
        # deterministic retry schedule
        s = None if seed is None else (seed + t)
        try:
            edge_index, degrees, coords, arc_slices = build_once(s)
        except Exception as e:
            last_exc = e
            continue

        last = (edge_index, degrees, coords, arc_slices)

        if not keep_connected:
            return last

        if _is_connected(edge_index, int(degrees.numel())):
            return last

    if last is not None:
        return last
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Graph construction failed without returning a candidate.")



def build_graph_tuple_from_cfg(cfg: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads `acpl/configs/data/graphs.yaml`-style config and returns:
      (edge_index, degrees, coords, arc_slices)

    Supports:
      family: line | cycle | grid | cube | regular | er | ws | ws_grid
    """
    # Accept either full cfg (with cfg.data) or the data node directly.
    data = _get(cfg, "data", None)
    if data is not None and _get(data, "family", None) is not None:
        cfg = data

    fam_raw = _get(cfg, "family", None)
    if fam_raw is None:
        raise ValueError("data.family must be set (e.g., line/cycle/grid-2d/hypercube/regular/er/ws).")

    family = str(fam_raw).strip().lower()
    # Aliases for consistency across configs
    if family in ("grid-2d", "grid2d", "grid_2d"):
        family = "grid"
    if family in ("hypercube",):
        family = "cube"
    if family in ("erdos_renyi", "erdos-renyi", "er-graph"):
        family = "er"
    
    
    
    
    
    seed = _get(cfg, "seed", None)
    seed = int(seed) if seed is not None else None



    coalesce = bool(_get(cfg, "coalesce", True))
    if not coalesce:
        raise ValueError("coalesce=false is not supported (DTQW expects a simple undirected graph).")





    directed = bool(_get(cfg, "directed", False))
    self_loops = bool(_get(cfg, "self_loops", False))
    if directed:
        raise ValueError("graphs.py builds undirected graphs; set data.directed=false.")
    if self_loops:
        raise ValueError("graphs.py drops self-loops; set data.self_loops=false.")

    N_top = int(_get(cfg, "N", 0))

    if family == "line":
        block = _get(cfg, "line", cfg)
        N = int(_get(block, "N", N_top))
        return line_graph(N, seed=seed)

    if family == "cycle":
        block = _get(cfg, "cycle", cfg)
        N = int(_get(block, "N", N_top))
        return cycle_graph(N, seed=seed)

    if family == "grid":
        block = _get(cfg, "grid", cfg)
        Lx = int(_get(block, "Lx", 1))
        Ly = _get(block, "Ly", None)
        Ly = int(Ly) if Ly is not None else None
        make_coords = bool(_get(block, "make_coords", True))
        return _call_optional(grid_graph, Lx, Ly, seed=seed, make_coords=make_coords)

    if family == "cube":
        block = _get(cfg, "cube", cfg)
        d = int(_get(block, "d", 0))
        emit_bitstrings = bool(_get(block, "emit_bitstrings", False))
        return _call_optional(hypercube_graph, d, seed=seed, emit_bitstrings=emit_bitstrings)

    if family == "regular":
        block = _get(cfg, "regular", cfg)
        N = int(_get(block, "N", N_top))
        d = int(_get(block, "d", 0))
        keep_connected = bool(_get(block, "keep_connected", False))
        max_tries = int(_get(block, "max_tries", _get(block, "max_retries", 50)))
 

        def _once(s: int | None):
            return d_regular_random_graph(N, d, seed=s)

        return _retry_until_connected(_once, keep_connected=keep_connected, max_tries=max_tries, seed=seed)

    if family == "er":
        block = _get(cfg, "er", cfg)
        N = int(_get(block, "N", N_top))
        p = float(_get(block, "p", 0.0))
        keep_connected = bool(_get(block, "keep_connected", False))
        max_tries = int(_get(block, "max_tries", 2000))

        def _once(s: int | None):
            return _call_optional(erdos_renyi_graph, N, p, seed=s, ensure_simple=True)

        return _retry_until_connected(_once, keep_connected=keep_connected, max_tries=max_tries, seed=seed)

    if family == "ws":
        block = _get(cfg, "ws", cfg)
        variant = str(_get(block, "variant", "ring"))

        if variant == "ring":
            N = int(_get(block, "N", N_top))
            k = int(_get(block, "k", 0))
            beta = float(_get(block, "beta", 0.0))
            keep_connected = bool(_get(block, "keep_connected", False))
            max_tries = int(_get(block, "max_tries", 2000))

            def _once(s: int | None):
                return _call_optional(watts_strogatz_graph, N, k, beta, seed=s)

            return _retry_until_connected(_once, keep_connected=keep_connected, max_tries=max_tries, seed=seed)

        if variant == "grid":
            Lx = int(_get(block, "Lx", 1))
            Ly = int(_get(block, "Ly", Lx))
            kx = int(_get(block, "kx", 1))
            ky = int(_get(block, "ky", 1))
            # supports either ws.beta or ws.beta_grid (your YAML has both patterns)
            beta = float(_get(block, "beta_grid", _get(block, "beta", 0.0)))
            torus = bool(_get(block, "torus", False))
            degree_preserving = bool(_get(block, "degree_preserving", False))
            sanitize = bool(_get(block, "sanitize", True))



            # optional “keep connected” for grid variant
            keep_connected = bool(_get(block, "keep_connected_grid", _get(block, "keep_connected", False)))
            max_tries = int(_get(block, "max_tries_grid", _get(block, "max_tries", 200)))

            def _once(s: int | None):
                if degree_preserving:
                    return _call_optional(
                        watts_strogatz_grid_graph_degree_preserving,
                        Lx, Ly, kx=kx, ky=ky, beta=beta, seed=s
                    )
                return _call_optional(
                    watts_strogatz_grid_graph,
                    Lx, Ly, kx=kx, ky=ky, beta=beta, seed=s, torus=torus, sanitize=sanitize
                )
            return _retry_until_connected(_once, keep_connected=keep_connected, max_tries=max_tries, seed=seed)

        raise ValueError(f"Unknown ws.variant={variant!r} (expected 'ring' or 'grid').")

    if family == "ws_grid":
        block = _get(cfg, "ws_grid", cfg)
        Lx = int(_get(block, "Lx", 1))
        Ly = int(_get(block, "Ly", Lx))
        kx = int(_get(block, "kx", 1))
        ky = int(_get(block, "ky", 1))
        beta = float(_get(block, "beta", 0.0))
        torus = bool(_get(block, "torus", False))

        keep_connected = bool(_get(block, "keep_connected", False))
        max_tries = int(_get(block, "max_tries", 200))

        def _once(s: int | None):
            return _call_optional(
                watts_strogatz_grid_graph,
                Lx, Ly, kx=kx, ky=ky, beta=beta, seed=s, torus=torus, sanitize=sanitize
            )

        return _retry_until_connected(_once, keep_connected=keep_connected, max_tries=max_tries, seed=seed)

    raise ValueError(f"Unknown data.family={family!r}.")


def build_graph_data_from_cfg(cfg: Any) -> GraphData:
    edge_index, degrees, coords, arc_slices = build_graph_tuple_from_cfg(cfg)
    return GraphData(edge_index=edge_index, degrees=degrees, coords=coords, arc_slices=arc_slices)

# acpl/data/graph_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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


def _is_connected(edge_index: torch.Tensor, N: int) -> bool:
    """Connectivity check on an undirected graph represented as oriented arcs (2, A)."""
    if N <= 1:
        return True
    if edge_index.numel() == 0:
        return False

    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)

    # Build adjacency lists via CSR using counts on src
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
    if not keep_connected:
        return build_once(seed)

    if seed is None:
        raise ValueError("keep_connected=true requires an explicit seed for determinism.")

    last = None
    for t in range(max_tries):
        # deterministic retry schedule
        edge_index, degrees, coords, arc_slices = build_once(seed + t)
        last = (edge_index, degrees, coords, arc_slices)
        if _is_connected(edge_index, int(degrees.numel())):
            return last
    assert last is not None
    return last


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

    family = str(_get(cfg, "family"))
    seed = _get(cfg, "seed", None)
    seed = int(seed) if seed is not None else None

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
        return grid_graph(Lx, Ly, seed=seed)

    if family == "cube":
        block = _get(cfg, "cube", cfg)
        d = int(_get(block, "d", 0))
        return hypercube_graph(d, seed=seed)

    if family == "regular":
        block = _get(cfg, "regular", cfg)
        N = int(_get(block, "N", N_top))
        d = int(_get(block, "d", 0))
        max_retries = int(_get(block, "max_retries", 50))

        def _once(s: int | None):
            return d_regular_random_graph(N, d, seed=s)

        # configuration-model retries are already inside graphs.py, but this respects YAML’s knob too.
        return _retry_until_connected(_once, keep_connected=False, max_tries=max_retries, seed=seed)

    if family == "er":
        block = _get(cfg, "er", cfg)
        N = int(_get(block, "N", N_top))
        p = float(_get(block, "p", 0.0))
        keep_connected = bool(_get(block, "keep_connected", False))
        max_tries = int(_get(block, "max_tries", 2000))

        def _once(s: int | None):
            return erdos_renyi_graph(N, p, seed=s, ensure_simple=True)

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
                return watts_strogatz_graph(N, k, beta, seed=s)

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

            # optional “keep connected” for grid variant
            keep_connected = bool(_get(block, "keep_connected_grid", False))
            max_tries = int(_get(block, "max_tries_grid", 200))

            def _once(s: int | None):
                if degree_preserving:
                    return watts_strogatz_grid_graph_degree_preserving(
                        Lx, Ly, kx=kx, ky=ky, beta=beta, seed=s
                    )
                return watts_strogatz_grid_graph(
                    Lx, Ly, kx=kx, ky=ky, beta=beta, seed=s, torus=torus
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
            return watts_strogatz_grid_graph(
                Lx, Ly, kx=kx, ky=ky, beta=beta, seed=s, torus=torus
            )

        return _retry_until_connected(_once, keep_connected=keep_connected, max_tries=max_tries, seed=seed)

    raise ValueError(f"Unknown data.family={family!r}.")


def build_graph_data_from_cfg(cfg: Any) -> GraphData:
    edge_index, degrees, coords, arc_slices = build_graph_tuple_from_cfg(cfg)
    return GraphData(edge_index=edge_index, degrees=degrees, coords=coords, arc_slices=arc_slices)

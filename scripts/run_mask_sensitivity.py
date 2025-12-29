#!/usr/bin/env python3
"""
scripts/run_mask_sensitivity.py

(3) Substructure masking plots (ACPL analogue)

Sweeps masking strength over several substructure definitions and plots metric sensitivity.

Artifacts (defendable):
  mask_sensitivity/run_meta.json
  mask_sensitivity/<kind>__pXXX/meta.json
  mask_sensitivity/<kind>__pXXX/episodes.jsonl
  mask_sensitivity/<kind>__pXXX/agg.json
  mask_sensitivity/results.csv + results.json
  mask_sensitivity/figs/<metric>.png  (optional)

This is intentionally non-minimal and research-ready:
- multiple masking strategies (random, hubs, ego/BFS, shortest-path)
- per-episode JSONL for auditability
- CI estimation
- plotting
- tries to reuse scripts/eval.py builders for repo-consistency
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

log = logging.getLogger("mask_sensitivity")


# ----------------------------- basic utils ---------------------------------


def _setup_logging(v: int) -> None:
    level = logging.WARNING if v <= 0 else (logging.INFO if v == 1 else logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _P(x: str | os.PathLike[str]) -> Path:
    return (x if isinstance(x, Path) else Path(str(x))).expanduser().resolve()


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _jdump(path: Path, obj: Any) -> None:
    _atomic_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _jappend(path: Path, rec: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True))
        f.write("\n")


def _sanitize(s: str) -> str:
    s = (s or "").strip()
    out = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "metric"


def _parse_seeds(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return [0]
    if "," not in s and " " not in s:
        try:
            n = int(s)
            if n > 0:
                return list(range(n))
        except Exception:
            pass
    return [int(p) for p in [q.strip() for q in s.replace(" ", ",").split(",") if q.strip()]]


def _parse_fracs(s: str) -> list[float]:
    out = [float(p) for p in [q.strip() for q in (s or "").replace(" ", ",").split(",") if q.strip()]]
    if not out:
        raise SystemExit("--mask_fractions must be non-empty")
    for f in out:
        if not (0.0 <= f <= 1.0):
            raise SystemExit(f"mask fraction {f} must be in [0,1]")
    return out


def _kv(items: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"'{it}' must be key=val")
        k, v = it.split("=", 1)
        k, v = k.strip(), v.strip()
        vv: Any = v
        if v.lower() in ("true", "false"):
            vv = v.lower() == "true"
        else:
            try:
                vv = int(v) if "." not in v else float(v)
            except Exception:
                try:
                    vv = json.loads(v)
                except Exception:
                    vv = v
        out[k] = vv
    return out


def _to_device(x: Any, dev: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, Mapping):
        return {k: _to_device(v, dev) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device(v, dev) for v in x)
    return x


def _edge2(edge_index: Any) -> torch.Tensor | None:
    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2:
        return None
    if edge_index.shape[0] == 2:
        return edge_index
    if edge_index.shape[1] == 2:
        return edge_index.t().contiguous()
    return None


def _call(fn: Callable[..., Any], /, **kw: Any) -> Any:
    sig = inspect.signature(fn)
    ps = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ps.values()):
        return fn(**kw)
    return fn(**{k: v for k, v in kw.items() if k in ps})


def _mean_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = float(x.mean())
    if n == 1:
        return (m, m, m)
    s = float(x.std(ddof=1))
    se = s / math.sqrt(n)
    tcrit = 1.96
    try:
        from scipy.stats import t as student_t  # type: ignore
        tcrit = float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))
    except Exception:
        tcrit = 1.96
    return (m, m - tcrit * se, m + tcrit * se)


def _metrics(out: Any) -> dict[str, float]:
    if isinstance(out, Mapping):
        if "metrics" in out and isinstance(out["metrics"], Mapping):
            out = out["metrics"]
        d: dict[str, float] = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                d[str(k)] = float(v.detach().cpu().item())
            elif isinstance(v, (int, float, np.integer, np.floating)):
                d[str(k)] = float(v)
        return d
    if isinstance(out, (tuple, list)):
        for it in out:
            if isinstance(it, Mapping):
                return _metrics(it)
        if out and isinstance(out[0], torch.Tensor) and out[0].numel() == 1:
            return {"value": float(out[0].detach().cpu().item())}
    if isinstance(out, torch.Tensor) and out.numel() == 1:
        return {"value": float(out.detach().cpu().item())}
    return {}


# ------------------------------- masking ------------------------------------


@dataclass(frozen=True)
class MaskCfg:
    kind: str
    frac: float
    mode: str  # node_features|edges|both
    keep_degree: bool
    keep_indicator: bool
    anchor: str  # auto|start|target|both


def _anchors(batch: Mapping[str, Any], anchor: str) -> list[int]:
    start = None
    target = None
    for k in ("start_node", "start", "source", "src", "s"):
        if k in batch:
            try:
                start = int(batch[k])
                break
            except Exception:
                pass
    for k in ("target_node", "target", "goal", "dst", "t"):
        if k in batch:
            try:
                target = int(batch[k])
                break
            except Exception:
                pass
    a = (anchor or "auto").lower()
    if a in ("start", "source"):
        return [start] if start is not None else []
    if a in ("target", "goal"):
        return [target] if target is not None else []
    out: list[int] = []
    if start is not None:
        out.append(start)
    if target is not None and target != start:
        out.append(target)
    return out


def _adj_undir(edge2: torch.Tensor, N: int) -> list[list[int]]:
    src = edge2[0].tolist()
    dst = edge2[1].tolist()
    adj: list[list[int]] = [[] for _ in range(N)]
    for u, v in zip(src, dst):
        if 0 <= u < N and 0 <= v < N:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def _bfs_budget(edge2: torch.Tensor, N: int, anchors: Sequence[int], budget: int) -> list[int]:
    if budget <= 0:
        return []
    adj = _adj_undir(edge2, N)
    seen = [False] * N
    q: list[int] = []
    for a in anchors:
        if 0 <= a < N and not seen[a]:
            seen[a] = True
            q.append(a)
    if not q and N > 0:
        seen[0] = True
        q = [0]
    out: list[int] = []
    qi = 0
    while qi < len(q) and len(out) < budget:
        u = q[qi]
        qi += 1
        out.append(u)
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
            if len(out) >= budget:
                break
    if len(out) < budget:
        for i in range(N):
            if not seen[i]:
                out.append(i)
                if len(out) >= budget:
                    break
    return out[:budget]


def _shortest_path(edge2: torch.Tensor, N: int, s: int, t: int) -> list[int]:
    if not (0 <= s < N and 0 <= t < N):
        return []
    if s == t:
        return [s]
    adj = _adj_undir(edge2, N)
    parent = [-1] * N
    q = [s]
    parent[s] = s
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        for v in adj[u]:
            if parent[v] != -1:
                continue
            parent[v] = u
            if v == t:
                qi = len(q)
                break
            q.append(v)
    if parent[t] == -1:
        return []
    path = [t]
    cur = t
    while cur != s:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _degrees(edge2: torch.Tensor, N: int) -> torch.Tensor:
    src = edge2[0].to(torch.long)
    dst = edge2[1].to(torch.long)
    deg = torch.zeros(N, dtype=torch.long, device=edge2.device)
    deg.scatter_add_(0, src.clamp_(0, N - 1), torch.ones_like(src, dtype=torch.long))
    deg.scatter_add_(0, dst.clamp_(0, N - 1), torch.ones_like(dst, dtype=torch.long))
    return deg


def _mask_nodes(cfg: MaskCfg, *, edge2: torch.Tensor, N: int, batch: Mapping[str, Any], rng: random.Random) -> list[int]:
    kind = cfg.kind.lower().strip()
    budget = max(0, min(N, int(math.ceil(cfg.frac * N))))

    if kind in ("none", "noop", "identity"):
        return []
    if kind in ("random_nodes", "node_random", "rand_nodes"):
        return rng.sample(range(N), k=budget) if budget > 0 else []
    if kind in ("high_degree_nodes", "hub_nodes", "degree_nodes"):
        if budget <= 0:
            return []
        idx = np.argsort(-_degrees(edge2, N).cpu().numpy())[:budget]
        return [int(i) for i in idx.tolist()]
    if kind in ("ego_anchor", "ego", "bfs_anchor"):
        if budget <= 0:
            return []
        anc = _anchors(batch, cfg.anchor)
        if not anc and N > 0:
            anc = [rng.randrange(N)]
        return _bfs_budget(edge2, N, anc, budget)
    if kind in ("path_anchor", "shortest_path", "path_start_target"):
        anc = _anchors(batch, "both")
        if len(anc) < 2:
            return rng.sample(range(N), k=budget) if budget > 0 else []
        path = _shortest_path(edge2, N, anc[0], anc[1])
        if not path:
            return rng.sample(range(N), k=budget) if budget > 0 else []
        if budget > 0 and len(path) > budget:
            keep = [path[0], path[-1]]
            interior = path[1:-1]
            need = max(0, budget - len(keep))
            if need > 0 and interior:
                step = max(1, len(interior) // need)
                keep += interior[::step][:need]
            return sorted(set(keep))
        return path

    # unknown => default to random
    return rng.sample(range(N), k=budget) if budget > 0 else []


def _mask_X(X: torch.Tensor, nodes: Sequence[int], keep_degree: bool, keep_indicator: bool) -> torch.Tensor:
    if not isinstance(X, torch.Tensor) or X.ndim not in (2, 3):
        return X
    idx = torch.as_tensor(list(nodes), dtype=torch.long, device=X.device)
    if idx.numel() == 0:
        return X
    Y = X.clone()
    F = int(Y.shape[-1])
    feat = torch.ones(F, dtype=torch.bool, device=X.device)
    if keep_degree and F >= 1:
        feat[0] = False
    if keep_indicator and F >= 2:
        feat[-1] = False

    if Y.ndim == 2:
        cols = torch.arange(F, device=X.device)[feat]
        if cols.numel():
            Y[idx[:, None], cols[None, :]] = 0
    else:
        Y[:, idx, :] = Y[:, idx, :].masked_fill(feat[None, None, :], 0)
    return Y


def _mask_edges(edge2: torch.Tensor, nodes: Sequence[int], edge_drop: float, rng: random.Random) -> torch.Tensor:
    E = int(edge2.shape[1])
    keep = torch.ones(E, dtype=torch.bool, device=edge2.device)
    if nodes:
        mn = torch.as_tensor(list(nodes), dtype=torch.long, device=edge2.device)
        keep &= ~(torch.isin(edge2[0], mn) | torch.isin(edge2[1], mn))
    if edge_drop > 0.0:
        edge_drop = float(max(0.0, min(1.0, edge_drop)))
        idx_keep = torch.nonzero(keep, as_tuple=False).flatten().tolist()
        drop_k = int(math.floor(edge_drop * len(idx_keep)))
        if drop_k > 0:
            drop = set(rng.sample(idx_keep, k=drop_k))
            keep &= torch.as_tensor([i not in drop for i in range(E)], device=keep.device)
    return edge2[:, keep].contiguous()


def wrap_factory(base_factory: Callable[[int], Iterable[Any]], cfg: MaskCfg) -> Callable[[int], Iterable[Any]]:
    def factory(seed_i: int):
        h = hashlib.blake2b(
            f"{seed_i}|{cfg.kind}|{cfg.frac}|{cfg.mode}".encode("utf-8"),
            digest_size=8,
        ).digest()
        base_seed = int.from_bytes(h, "little", signed=False) % (2**32 - 1)

        for ep_i, batch in enumerate(base_factory(int(seed_i))):
            if not isinstance(batch, Mapping):
                yield batch
                continue

            b = dict(batch)
            edge2 = _edge2(b.get("edge_index", None))
            if edge2 is None:
                yield b
                continue

            X = b.get("X", None)
            if isinstance(X, torch.Tensor) and X.ndim == 3:
                N = int(X.shape[-2])
            elif isinstance(X, torch.Tensor) and X.ndim == 2:
                N = int(X.shape[0])
            else:
                N = int(edge2.max().item()) + 1 if edge2.numel() else 0

            rng = random.Random((base_seed + 1315423911 * (ep_i + 1)) % (2**32 - 1))
            nodes = _mask_nodes(cfg, edge2=edge2, N=N, batch=b, rng=rng)

            b["_mask"] = {
                "kind": cfg.kind,
                "fraction": float(cfg.frac),
                "mode": cfg.mode,
                "N": int(N),
                "masked_nodes": int(len(nodes)),
                "masked_nodes_frac": (len(nodes) / N if N else 0.0),
                "seed_i": int(seed_i),
                "ep_i": int(ep_i),
            }

            mode = cfg.mode.lower()
            if mode in ("node_features", "both") and isinstance(X, torch.Tensor):
                b["X"] = _mask_X(X, nodes, cfg.keep_degree, cfg.keep_indicator)

            if mode in ("edges", "both"):
                edge2m = _mask_edges(edge2, nodes, edge_drop=float(cfg.frac), rng=rng)
                orig = b.get("edge_index", None)
                if isinstance(orig, torch.Tensor) and orig.ndim == 2 and orig.shape[1] == 2:
                    b["edge_index"] = edge2m.t().contiguous()
                else:
                    b["edge_index"] = edge2m

            yield b

    return factory


# ------------------------- runner bundle plumbing ---------------------------


@dataclass
class Bundle:
    model: torch.nn.Module
    dataloader_factory: Callable[[int], Iterable[Any]]
    rollout_fn: Callable[[torch.nn.Module, Any], Any]
    meta: dict[str, Any]


def _load_eval_mod() -> Any | None:
    path = Path(__file__).resolve().parent / "eval.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_acpl_eval_driver", str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_acpl_eval_driver"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _resolve_ckpt(p: Path) -> Path:
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(str(p))
    for c in (p / "last.pt", p / "model_last.pt", p / "checkpoint.pt"):
        if c.exists():
            return c
    pts = sorted(p.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]
    raise FileNotFoundError(f"No checkpoint in {p}")


def _import_obj(spec: str) -> Any:
    mod, name = spec.split(":", 1)
    return getattr(importlib.import_module(mod), name)


def build_bundle(
    ckpt_path: Path,
    *,
    suite: str,
    device: torch.device,
    seeds: Sequence[int],
    episodes: int,
    policy: str,
    baseline_kind: str,
    baseline_coins_kwargs: dict[str, Any],
    baseline_policy_kwargs: dict[str, Any],
    model_factory: str | None,
    dataloader_factory: str | None,
    rollout_fn: str | None,
) -> Bundle:
    ckpt_path = _resolve_ckpt(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    eval_mod = _load_eval_mod()

    # Prefer scripts/eval.py internal builder(s) so we stay consistent with repo semantics.
    if eval_mod is not None:
        for name in (
            "_build_base_bundle",
            "build_base_bundle",
            "_make_base_bundle",
            "_build_eval_components",
            "build_eval_components",
        ):
            fn = getattr(eval_mod, name, None)
            if not callable(fn):
                continue
            try:
                out = _call(
                    fn,
                    ckpt_path=ckpt_path,
                    ckpt=ckpt,
                    suite=suite,
                    device=str(device),
                    seeds=list(seeds),
                    episodes=int(episodes),
                    policy=policy,
                    baseline_kind=baseline_kind,
                    baseline_coins_kwargs=baseline_coins_kwargs,
                    baseline_policy_kwargs=baseline_policy_kwargs,
                )
            except Exception as e:
                log.debug("eval builder %s failed: %s", name, e)
                continue

            if hasattr(out, "model") and hasattr(out, "dataloader_factory") and hasattr(out, "rollout_fn"):
                return Bundle(out.model, out.dataloader_factory, out.rollout_fn, {"builder": f"scripts/eval.py:{name}"})

    # Fallback: explicit factories.
    if not (model_factory and dataloader_factory and rollout_fn):
        raise RuntimeError(
            "Could not reuse scripts/eval.py builders. Provide explicit factories:\n"
            "  --model_factory acpl.train.factory:build_policy\n"
            "  --dataloader_factory acpl.train.loops:build_dataloader_factory\n"
            "  --rollout_fn acpl.train.loops:rollout_eval\n"
        )

    mf = _import_obj(model_factory)
    dlf = _import_obj(dataloader_factory)
    rf = _import_obj(rollout_fn)

    model = _call(
        mf,
        ckpt=ckpt,
        suite=suite,
        device=str(device),
        policy=policy,
        baseline_kind=baseline_kind,
        baseline_coins_kwargs=baseline_coins_kwargs,
        baseline_policy_kwargs=baseline_policy_kwargs,
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model_factory returned {type(model)}")

    # Try to load weights if present
    sd = None
    if isinstance(ckpt, Mapping):
        for k in ("state_dict", "model_state_dict", "model", "policy_state_dict"):
            if isinstance(ckpt.get(k, None), Mapping):
                sd = dict(ckpt[k])
                break
    if sd is not None and policy == "ckpt":
        model.load_state_dict(sd, strict=False)

    dl_factory = _call(dlf, ckpt=ckpt, suite=suite, seeds=list(seeds), episodes=int(episodes))
    if not callable(dl_factory):
        raise TypeError("dataloader_factory must be callable (seed_i -> iterable of batches)")

    return Bundle(model, dl_factory, rf, {"builder": "explicit_factories"})


# ------------------------------- sweep --------------------------------------


@dataclass
class Row:
    mask_kind: str
    mask_fraction: float
    mask_mode: str
    metric: str
    mean: float
    lo: float
    hi: float
    n: int


def run_condition(
    bundle: Bundle,
    *,
    dev: torch.device,
    suite: str,
    seeds: Sequence[int],
    episodes: int,
    outdir: Path,
    cfg: MaskCfg,
    overwrite: bool,
) -> list[Row]:
    _mkdir(outdir)
    meta_p, agg_p, jsonl_p = outdir / "meta.json", outdir / "agg.json", outdir / "episodes.jsonl"

    if agg_p.exists() and not overwrite:
        agg = json.loads(agg_p.read_text(encoding="utf-8"))
        return [
            Row(cfg.kind, cfg.frac, cfg.mode, m, d["mean"], d["lo"], d["hi"], d["n"])
            for m, d in agg["metrics"].items()
        ]

    if overwrite and jsonl_p.exists():
        jsonl_p.unlink()

    _jdump(
        meta_p,
        {
            "suite": suite,
            "seeds": list(map(int, seeds)),
            "episodes": int(episodes),
            "device": str(dev),
            "mask": asdict(cfg),
            "bundle_meta": bundle.meta,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    model = bundle.model.to(dev)
    model.eval()
    masked_factory = wrap_factory(bundle.dataloader_factory, cfg)

    samples: dict[str, list[float]] = {}
    n_total = 0

    with torch.no_grad():
        for seed_i in seeds:
            for ep_i, batch in enumerate(masked_factory(int(seed_i))):
                if ep_i >= episodes:
                    break
                batch_dev = _to_device(batch, dev)
                try:
                    out = bundle.rollout_fn(model, batch_dev)
                except TypeError:
                    out = _call(bundle.rollout_fn, model=model, batch=batch_dev, device=dev)

                mets = _metrics(out)
                for k, v in mets.items():
                    samples.setdefault(k, []).append(float(v))

                _jappend(
                    jsonl_p,
                    {
                        "seed_i": int(seed_i),
                        "ep_i": int(ep_i),
                        "mask": (batch.get("_mask", None) if isinstance(batch, Mapping) else None),
                        "metrics": mets,
                    },
                )
                n_total += 1

    agg: dict[str, dict[str, Any]] = {}
    for met, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        m, lo, hi = _mean_ci(arr)
        agg[met] = {"mean": m, "lo": lo, "hi": hi, "n": int(arr.size)}

    _jdump(agg_p, {"n_total": int(n_total), "metrics": agg})

    return [
        Row(cfg.kind, cfg.frac, cfg.mode, met, d["mean"], d["lo"], d["hi"], int(d["n"]))
        for met, d in agg.items()
    ]


def plot_curves(root: Path, rows: list[Row], title_prefix: str) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa

    figdir = _mkdir(root / "figs")
    metrics = sorted({r.metric for r in rows})
    kinds = sorted({r.mask_kind for r in rows})

    for met in metrics:
        plt.figure()
        for kind in kinds:
            rs = [r for r in rows if r.metric == met and r.mask_kind == kind]
            rs.sort(key=lambda r: r.mask_fraction)
            if not rs:
                continue

            x = [r.mask_fraction for r in rs]
            y = [r.mean for r in rs]
            lo = [r.lo for r in rs]
            hi = [r.hi for r in rs]
            plt.plot(x, y, marker="o", label=kind)
            plt.fill_between(x, lo, hi, alpha=0.2)

        plt.xlabel("mask fraction")
        plt.ylabel(met)
        plt.title(f"{title_prefix}{met}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / f"{_sanitize(met)}.png", dpi=160)
        plt.close()


# -------------------------------- CLI ---------------------------------------


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mask sensitivity sweep (substructure masking).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="Checkpoint (file or directory).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--suite", default="basic_valid")
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    p.add_argument("--seeds", default="0", help="Seeds: '5' or '0,1,2'")
    p.add_argument("--episodes", type=int, default=128)

    p.add_argument("--policy", default="ckpt", choices=("ckpt", "baseline"))
    p.add_argument("--baseline", default="hadamard")
    p.add_argument("--baseline_coins", nargs="*", default=[], help="Baseline coin kwargs key=val")
    p.add_argument("--baseline_policy", nargs="*", default=[], help="Baseline policy kwargs key=val")

    p.add_argument(
        "--mask_kinds",
        nargs="*",
        default=["random_nodes", "high_degree_nodes", "ego_anchor", "path_anchor"],
    )
    p.add_argument("--mask_fractions", default="0,0.05,0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--mask_mode", default="node_features", choices=("node_features", "edges", "both"))
    p.add_argument("--mask_anchor", default="auto", choices=("auto", "start", "target", "both"))
    p.add_argument("--keep_degree", action="store_true")
    p.add_argument("--keep_indicator", action="store_true")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=0)

    p.add_argument("--model_factory", default=None, help="module:callable (only if needed)")
    p.add_argument("--dataloader_factory", default=None, help="module:callable (only if needed)")
    p.add_argument("--rollout_fn", default=None, help="module:callable (only if needed)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    a = _args(argv)
    _setup_logging(int(a.verbose))

    dev = (
        torch.device(a.device)
        if a.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    seeds = _parse_seeds(a.seeds)
    fracs = _parse_fracs(a.mask_fractions)

    outdir = _mkdir(_P(a.outdir))
    root = _mkdir(outdir / "mask_sensitivity")

    bundle = build_bundle(
        _P(a.ckpt),
        suite=str(a.suite),
        device=dev,
        seeds=seeds,
        episodes=int(a.episodes),
        policy=str(a.policy),
        baseline_kind=str(a.baseline),
        baseline_coins_kwargs=_kv(a.baseline_coins or []),
        baseline_policy_kwargs=_kv(a.baseline_policy or []),
        model_factory=a.model_factory,
        dataloader_factory=a.dataloader_factory,
        rollout_fn=a.rollout_fn,
    )

    _jdump(
        root / "run_meta.json",
        {
            "ckpt": str(_P(a.ckpt)),
            "suite": str(a.suite),
            "device": str(dev),
            "policy": str(a.policy),
            "baseline": str(a.baseline),
            "seeds": seeds,
            "episodes": int(a.episodes),
            "mask_kinds": list(a.mask_kinds),
            "mask_fractions": fracs,
            "mask_mode": str(a.mask_mode),
            "mask_anchor": str(a.mask_anchor),
            "keep_degree": bool(a.keep_degree),
            "keep_indicator": bool(a.keep_indicator),
            "bundle_meta": bundle.meta,
        },
    )

    rows: list[Row] = []
    for kind in a.mask_kinds:
        for frac in fracs:
            cond = f"{_sanitize(kind)}__p{frac:.3f}".replace(".", "p")
            cfg = MaskCfg(
                str(kind),
                float(frac),
                str(a.mask_mode),
                bool(a.keep_degree),
                bool(a.keep_indicator),
                str(a.mask_anchor),
            )
            rows.extend(
                run_condition(
                    bundle,
                    dev=dev,
                    suite=str(a.suite),
                    seeds=seeds,
                    episodes=int(a.episodes),
                    outdir=root / cond,
                    cfg=cfg,
                    overwrite=bool(a.overwrite),
                )
            )

    # Combined JSON + CSV
    _jdump(root / "results.json", {"rows": [asdict(r) for r in rows]})
    csv = ["mask_kind,mask_fraction,mask_mode,metric,mean,lo,hi,n"]
    for r in rows:
        csv.append(
            f"{r.mask_kind},{r.mask_fraction:.6f},{r.mask_mode},{r.metric},"
            f"{r.mean:.10g},{r.lo:.10g},{r.hi:.10g},{r.n}"
        )
    _atomic_text(root / "results.csv", "\n".join(csv) + "\n")

    if a.plots:
        plot_curves(root, rows, title_prefix=f"{a.suite} | ")

    log.info("done: %s", root)


if __name__ == "__main__":
    main()

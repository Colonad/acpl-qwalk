# acpl/eval/plots.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

from statistics import NormalDist

# Prefer SciPy for defensible statistical intervals when available.
try:  # pragma: no cover
    from scipy import stats as _sp_stats  # type: ignore
except Exception:  # pragma: no cover
    _sp_stats = None


__all__ = [
    "PlotStyle",
    "ensure_numpy",
    "tv_curve",
    "mean_ci",
    "plot_position_timelines",
    "plot_tv_curves",
    "plot_robustness_sweep",
    "plot_mean_pt_ci_across_episodes",
]




def _import_matplotlib_pyplot():
    # Headless-safe backend selection (important on CI/servers/WSL)
    try:
        matplotlib.use("Agg", force=False)
    except Exception:
        pass
    import matplotlib.pyplot as plt  # noqa: WPS433 (local import by design)
    return plt


# --------------------------------------------------------------------------------------
#                                   Utilities
# --------------------------------------------------------------------------------------


@dataclass
class PlotStyle:
    """
    Lightweight style hints (kept optional to avoid global rc side-effects).
    """

    figsize: tuple[float, float] = (7.0, 4.2)
    dpi: int = 120
    title_size: int = 12
    label_size: int = 11
    tick_size: int = 10
    grid: bool = True
    alpha_ci: float = 0.25
    tight_layout: bool = True


def ensure_numpy(x: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor]) -> np.ndarray:
    """
    Accept torch/numpy (or a sequence of them) and return a CPU numpy array.

    - If x is a sequence of arrays/tensors with same shape, stacks on axis=0.
      This is the common case for "K episodes" or "S seeds" collections.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("ensure_numpy: got an empty sequence.")
        arrs = [ensure_numpy(a) for a in x]
        try:
            return np.stack(arrs, axis=0)
        except Exception as e:
            shapes = [getattr(a, "shape", None) for a in arrs]
            raise ValueError(f"ensure_numpy: cannot stack sequence; shapes={shapes}") from e
    raise TypeError(f"Unsupported type: {type(x)}")



def _coerce_path(p: str | Path | None) -> str | None:
    if p is None:
        return None
    return str(p)


def _maybe_mkdir_for_file(path_str: str) -> None:
    try:
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If this fails, matplotlib will raise on save anyway; keep plotting robust.
        pass


def _savefig(fig, savepath: str | Path | None, *, close: bool = False) -> None:
    sp = _coerce_path(savepath)
    if not sp:
        return
    _maybe_mkdir_for_file(sp)
    fig.savefig(sp, bbox_inches="tight")
    if close:
        try:
            import matplotlib.pyplot as plt  # local import to respect backend
            plt.close(fig)
        except Exception:
            pass



def _as_probabilities(
    x: np.ndarray | torch.Tensor,
    *,
    imag_tol: float = 1e-9,
    clip_neg: bool = True,
    renormalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert input to a numerically-safe probability tensor.

    Handles common pipeline cases:
      - complex with tiny imaginary part -> take real
      - complex amplitudes -> convert to |.|^2
      - small negative values -> clip to 0
      - rows not summing to 1 -> renormalize along last axis
    """
    P = ensure_numpy(x)

    # complex handling
    if np.iscomplexobj(P):
        max_imag = np.nanmax(np.abs(P.imag)) if P.size else 0.0
        if max_imag <= imag_tol:
            P = P.real
        else:
            # treat as amplitudes (defensible fallback)
            P = np.abs(P) ** 2

    P = P.astype(np.float64, copy=False)

    P = np.where(np.isfinite(P), P, 0.0)


    if clip_neg:
        # clip tiny numerical negatives
        P = np.where(P < 0.0, 0.0, P)

    if renormalize and P.ndim >= 2:
        # normalize along last axis (node axis)
        denom = P.sum(axis=-1, keepdims=True)
        P = np.divide(P, np.maximum(denom, eps), out=np.zeros_like(P), where=denom > 0)

    return P



def _uniform(N: int) -> np.ndarray:
    return np.full((N,), 1.0 / max(1, N), dtype=np.float64)



def plot_mean_pt_ci_across_episodes(
    Pt_samples: np.ndarray | torch.Tensor | Sequence[np.ndarray | torch.Tensor],
    *,
    suite: str,
    policy: str,
    topk: int = 12,
    nodes: Sequence[int] | None = None,
    style: PlotStyle | None = None,
    savepath: str | Path | None = None,
    close: bool = False,
):
    P = _as_probabilities(Pt_samples)
    if P.ndim < 2:
        raise ValueError(f"Expected trailing (T,N), got {P.shape}")
    K = int(np.prod(P.shape[:-2])) if P.ndim > 2 else 1
    title = f"{suite} — {policy} — mean Pt — CI across episodes (K={K})"
    return plot_position_timelines(
        Pt_samples,
        nodes=nodes,
        topk=topk,
        style=style,
        title=title,
        savepath=savepath,
        close=close,
    )



def tv_curve(Pt: np.ndarray | torch.Tensor) -> np.ndarray:
    r"""
    Compute TV distance to uniform for each time t.

    Inputs
    ------
    Pt : shape (T, N) or (S, T, N)
         P_t distributions over N nodes at each time t.
         If a seeds dimension S is present, TV is computed per seed.

    Returns
    -------
    tv : (T,) if input was (T,N), or (S, T) if input was (S,T,N)
          TV_t = 0.5 * sum_v |P_t[v] - 1/N|
    """
    P = _as_probabilities(Pt)
    if P.ndim == 2:
        T, N = P.shape
        U = _uniform(N)
        return 0.5 * np.abs(P - U).sum(axis=1)
    elif P.ndim == 3:
        S, T, N = P.shape
        U = _uniform(N)[None, None, :]
        return 0.5 * np.abs(P - U).sum(axis=2)
    else:
        raise ValueError(f"Expected (T,N) or (S,T,N), got shape {P.shape}.")


def _two_sided_p(conf: float) -> float:
    # two-sided CI => upper quantile p = 1 - alpha/2 = 0.5 + conf/2
    return 0.5 + 0.5 * conf


def _z_crit(conf: float) -> float:
    # No lookup tables: compute via NormalDist (stdlib).
    p = _two_sided_p(conf)
    return float(NormalDist().inv_cdf(p))


def _t_crit(conf: float, df: np.ndarray) -> np.ndarray:
    """
    Compute two-sided t critical values for given df.

    Uses SciPy if available; otherwise falls back to z critical values.
    """
    df = np.asarray(df, dtype=np.float64)
    z = _z_crit(conf)

    if _sp_stats is None:
        return np.full_like(df, z, dtype=np.float64)

    p = _two_sided_p(conf)
    # SciPy supports vectorized df.
    out = _sp_stats.t.ppf(p, np.maximum(df, 1.0))
    out = np.asarray(out, dtype=np.float64)

    # For df < 1, fallback to z (undefined t).
    out = np.where(df >= 1.0, out, z)
    return out


def mean_ci(x, axis: int = 0, conf: float = 0.95):
    """
    Mean +/- CI along `axis`, robust for NaNs and n<=1.

    - Uses Student-t critical values when SciPy is available (recommended).
    - Falls back to normal critical values otherwise.
    - If n<=1: CI collapses to mean (no uncertainty estimate possible).
    - If n==0: returns NaNs.

    Returns: mean, lo, hi
    """
    if not (0.0 < conf < 1.0):
        raise ValueError(f"conf must be in (0,1); got {conf}")

    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)

    n = finite.sum(axis=axis)  # reduced shape

    # Mean without "empty slice" warnings
    x0 = np.where(finite, x, 0.0)
    sum_keep = x0.sum(axis=axis, keepdims=True)
    n_keep = finite.sum(axis=axis, keepdims=True).astype(np.float64)

    mean_keep = np.divide(
        sum_keep,
        n_keep,
        out=np.full_like(sum_keep, np.nan, dtype=np.float64),
        where=n_keep > 0,
    )

    # Squeeze only the reduced axis (we intentionally support axis:int here)
    mean = np.squeeze(mean_keep, axis=axis)

    # Unbiased sample variance: ssq/(n-1) for n>1
    diff = np.where(finite, x - mean_keep, 0.0)
    ssq = (diff * diff).sum(axis=axis)

    denom = (n - 1).astype(np.float64)
    var = np.divide(ssq, denom, out=np.zeros_like(ssq, dtype=np.float64), where=n > 1)

    # Standard error: sqrt(var/n) for n>1
    se2 = np.divide(var, n, out=np.zeros_like(var, dtype=np.float64), where=n > 1)
    se = np.sqrt(se2)

    # Critical value (t if possible)
    tcrit = _t_crit(conf, df=(n - 1).astype(np.float64))

    half = tcrit * se
    lo = np.where(n > 1, mean - half, mean)
    hi = np.where(n > 1, mean + half, mean)

    # If n==0, keep NaNs
    lo = np.where(n > 0, lo, np.nan)
    hi = np.where(n > 0, hi, np.nan)
    mean = np.where(n > 0, mean, np.nan)

    return mean, lo, hi



def _t_crit_lookup(conf: float, df: np.ndarray) -> np.ndarray:
    """
    Conservative CI critical value without SciPy.

    - For common conf in {0.90,0.95,0.99} and df<=30, use tabulated t critical values.
    - Otherwise fall back to z critical value.
    """
    z = z_value(conf)
    out = np.full_like(df, z, dtype=np.float64)

    # Two-sided t critical values for conf in {0.90, 0.95, 0.99}
    t_tables: dict[float, list[float]] = {
        0.90: [
            0.0,
            6.313751515, 2.919985580, 2.353363435, 2.131846786, 2.015048373,
            1.943180281, 1.894578605, 1.859548038, 1.833112933, 1.812461123,
            1.795884819, 1.782287556, 1.770933396, 1.761310136, 1.753050356,
            1.745883676, 1.739606726, 1.734063607, 1.729132812, 1.724718243,
            1.720742903, 1.717144374, 1.713871528, 1.710882080, 1.708140761,
            1.705617920, 1.703288446, 1.701130934, 1.699127027, 1.697260894,
        ],
        0.95: [
            0.0,
            12.706204736, 4.302652730, 3.182446305, 2.776445105, 2.570581836,
            2.446911851, 2.364624251, 2.306004135, 2.262157163, 2.228138852,
            2.200985160, 2.178812830, 2.160368657, 2.144786688, 2.131449546,
            2.119905299, 2.109815578, 2.100922040, 2.093024054, 2.085963447,
            2.079613845, 2.073873068, 2.068657610, 2.063898562, 2.059538553,
            2.055529439, 2.051830516, 2.048407142, 2.045229642, 2.042272456,
        ],
        0.99: [
            0.0,
            63.656741162, 9.924843201, 5.840909310, 4.604094871, 4.032142983,
            3.707428021, 3.499483297, 3.355387331, 3.249835544, 3.169272673,
            3.105806516, 3.054539589, 3.012275839, 2.976842734, 2.946712883,
            2.920781622, 2.898230519, 2.878440473, 2.860934606, 2.845339709,
            2.831359558, 2.818756060, 2.807335684, 2.796939505, 2.787435814,
            2.778714534, 2.770682958, 2.763262455, 2.756385903, 2.749995653,
        ],
    }

    if conf not in t_tables:
        return out

    table = np.asarray(t_tables[conf], dtype=np.float64)
    max_df = table.shape[0] - 1

    mask = (df >= 1) & (df <= max_df)
    if np.any(mask):
        out[mask] = table[df[mask]]

    return out


# --------------------------------------------------------------------------------------
#                                 Plot: P_t timelines
# --------------------------------------------------------------------------------------


def _pick_nodes_for_timelines(
    Pt: np.ndarray, topk: int | None, nodes: Sequence[int] | None
) -> list[int]:
    """
    Choose which nodes to display.

    Pt: expected shape (T, N) (probabilities).
    - if `nodes` provided: use as-is (validated)
    - else pick `topk` nodes by "ever prominent" score (max over time)
      which is more stable than final-time mass for mixing-like regimes.
    """
    Pt = np.asarray(Pt)
    if Pt.ndim != 2:
        raise ValueError(f"_pick_nodes_for_timelines expected (T,N), got {Pt.shape}")
    T, N = Pt.shape
    if N <= 0 or T <= 0:
        return []

    if nodes is not None:
        idx = list(nodes)
        if not all(isinstance(i, (int, np.integer)) for i in idx):
            raise TypeError("nodes must be integer indices.")
        if not all(0 <= int(i) < N for i in idx):
            raise ValueError(f"nodes indices out of range for N={N}.")
        return [int(i) for i in idx]

    if (topk is None) or (topk >= N):
        return list(range(N))

    # Robust for mixing/spreading: choose nodes that ever get large probability.
    score = np.nanmax(Pt, axis=0)  # (N,)
    score = np.where(np.isfinite(score), score, -np.inf)
    idx = np.argsort(-score)[: int(topk)].tolist()
    return [int(i) for i in idx]





def plot_position_timelines(
    Pt: np.ndarray | torch.Tensor,
    *,
    nodes: Sequence[int] | None = None,
    topk: int | None = 12,
    sharey: bool = True,
    style: PlotStyle | None = None,
    title: str | None = "Position probabilities over time",
    savepath: str | Path | None = None,
    close: bool = False,

) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Plot P_t[v] as a timeline for selected nodes.

    Inputs
    ------
    Pt : (T,N) or (S,T,N) — multi-seed input will show mean ± CI bands.

    Options
    -------
    nodes : explicit node indices to plot. If None, we choose `topk` nodes with
            highest mass at final time.
    topk  : number of nodes to auto-select if `nodes=None`.
    sharey: share y-axis across subplots.
    style : plotting style hints.
    title : figure title.
    savepath : optional path to save the figure (png/pdf/...).

    Returns
    -------
    fig, axs
    """
    plt = _import_matplotlib_pyplot()
    st = style or PlotStyle()
    P = _as_probabilities(Pt)

    # Normalize shapes: accept (T,N) OR (S,T,N) OR (anything..., T, N)
    if P.ndim < 2:
        raise ValueError(f"Expected at least 2D with trailing (T,N), got {P.shape}.")

    T, N = P.shape[-2], P.shape[-1]

    if P.ndim == 2:
        Pm = P.reshape(1, T, N)  # (1,T,N)
    else:
        # flatten all leading dims into a single sample axis
        S = int(np.prod(P.shape[:-2]))
        Pm = P.reshape(S, T, N)  # (S,T,N)

    # pick nodes to show
    nodes_to_plot = _pick_nodes_for_timelines(Pm.mean(axis=0), topk=topk, nodes=nodes)
    K = len(nodes_to_plot)



    if K == 0:
        raise ValueError("No nodes selected to plot (graph has N=0 or selection empty).")


    # layout
    cols = min(4, K)
    rows = int(np.ceil(K / cols))
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(st.figsize[0], max(st.figsize[1], 1.8 * rows)),
        dpi=st.dpi,
        squeeze=False,
        sharey=sharey,
    )

    tgrid = np.arange(T)

    # per-node panel
    for j, v in enumerate(nodes_to_plot):
        r, c = divmod(j, cols)
        ax = axs[r, c]

        if Pm.shape[0] <= 1:
            ax.plot(tgrid, Pm[0, :, v], lw=1.8)
        else:
            # mean ± CI across samples (episodes and/or seeds)
            m, lo, hi = mean_ci(Pm[:, :, v], axis=0, conf=0.95)
            ax.plot(tgrid, m, lw=1.8)
            ax.fill_between(tgrid, lo, hi, alpha=st.alpha_ci, linewidth=0)


        ax.set_xlabel("t", fontsize=st.label_size)
        ax.set_ylabel(f"P[v={v}]", fontsize=st.label_size)
        ax.tick_params(axis="both", labelsize=st.tick_size)
        if st.grid:
            ax.grid(True, linewidth=0.5, alpha=0.5)

    # hide empty panels
    for j in range(K, rows * cols):
        r, c = divmod(j, cols)
        axs[r, c].axis("off")

    if title:
        fig.suptitle(title, fontsize=st.title_size)
    if st.tight_layout:
        fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))

    _savefig(fig, savepath, close=close)
    return fig, axs








def _looks_like_Pt_matrix(A: np.ndarray) -> bool:
    """
    Heuristic to decide if a 2D array is Pt (T,N) vs TV curves (S,T).

    - If row sums are close to 1 and values are mostly within [0,1], treat as Pt.
    - Handles complex amplitudes by testing on |A|^2.
    """
    if A.ndim != 2 or A.size == 0:
        return False

    B = A
    if np.iscomplexobj(B):
        B = np.abs(B) ** 2
    B = B.astype(np.float64, copy=False)
    B = np.where(np.isfinite(B), B, np.nan)

    mn = np.nanmin(B)
    mx = np.nanmax(B)
    if mn < -1e-3:
        return False
    if mx > 1.5:
        return False

    row_sums = np.nansum(np.where(np.isnan(B), 0.0, np.clip(B, 0.0, None)), axis=-1)
    if row_sums.size == 0:
        return False
    med = float(np.nanmedian(row_sums))
    return abs(med - 1.0) <= 0.05


def _clean_tv(tv: np.ndarray) -> np.ndarray:
    tv = np.asarray(tv, dtype=np.float64)
    tv = np.where(np.isfinite(tv), tv, np.nan)
    # Clip only tiny numerical overshoots; keep big anomalies visible.
    eps = 1e-9
    tv = np.where(tv < -eps, tv, np.maximum(tv, 0.0))
    tv = np.where(tv > 1.0 + eps, tv, np.minimum(tv, 1.0))
    return tv







# --------------------------------------------------------------------------------------
#                                  Plot: TV curves
# --------------------------------------------------------------------------------------


def plot_tv_curves(
    Pt: np.ndarray | torch.Tensor,
    *,
    logy: bool = False,
    style: PlotStyle | None = None,
    title: str | None = "Total Variation to uniform vs time",
    savepath: str | Path | None = None,
    close: bool = False,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot total-variation distance TV(P_t, U) along the horizon.

    Inputs
    ------
    Pt : (T,N) or (S,T,N). If multiple seeds provided, shows mean ± CI band.

    Options
    -------
    logy    : use log scale on y (useful when approaching mixing).
    style   : PlotStyle.
    title   : figure title.
    savepath: optional path to save the figure.

    Returns
    -------
    fig, ax
    """
    plt = _import_matplotlib_pyplot()
    st = style or PlotStyle()





    A = ensure_numpy(Pt)

    # Accept either:
    #  - Pt probabilities/amplitudes: (T,N) or (S,T,N)
    #  - tv curves directly: (T,) or (S,T)
    if A.ndim == 1:
        tv = _clean_tv(A)
    elif A.ndim == 2:
        if _looks_like_Pt_matrix(A):
            tv = tv_curve(_as_probabilities(A))
        else:
            tv = _clean_tv(A)
    elif A.ndim == 3:
        tv = tv_curve(_as_probabilities(A))
    else:
        raise ValueError(f"Expected (T,), (S,T), (T,N) or (S,T,N); got {A.shape}.")





    if tv.ndim == 1:
        m, lo, hi = tv, None, None
    else:
        m, lo, hi = mean_ci(tv, axis=0, conf=0.95)

    T = m.shape[0]
    tgrid = np.arange(T)

    fig, ax = plt.subplots(1, 1, figsize=st.figsize, dpi=st.dpi)
    ax.plot(tgrid, m, lw=2.0)
    if lo is not None and hi is not None:
        ax.fill_between(tgrid, lo, hi, alpha=st.alpha_ci, linewidth=0)

    ax.set_xlabel("t", fontsize=st.label_size)
    ax.set_ylabel("TV(P_t, U)", fontsize=st.label_size)
    ax.tick_params(axis="both", labelsize=st.tick_size)
    if st.grid:
        ax.grid(True, linewidth=0.5, alpha=0.5)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title, fontsize=st.title_size)

    if st.tight_layout:
        fig.tight_layout()

    _savefig(fig, savepath, close=close)
    return fig, ax




# --------------------------------------------------------------------------------------
#                            Plot: Robustness sweeps
# --------------------------------------------------------------------------------------


def plot_robustness_sweep(
    xgrid: Sequence[float],
    metrics: Mapping[str, np.ndarray | torch.Tensor],
    *,
    conf: float = 0.95,
    style: PlotStyle | None = None,
    title: str | None = "Robustness sweep",
    xlabel: str = "noise level",
    ylabel: str = "metric",
    legend_loc: str = "best",
    savepath: str | Path | None = None,
    close: bool = False,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

    """
    Plot metrics over a 1-D robustness parameter (e.g., dephasing p, static edge noise σ).

    Inputs
    ------
    xgrid   : sequence of sweep values, length M.
    metrics : dict name -> array with shape (M,) or (S, M).
              If seeds dimension is present, mean±CI bands are shown.

    Options
    -------
    conf       : CI level (default 95%).
    style      : PlotStyle.
    title      : figure title.
    xlabel/ylabel : axis labels.
    legend_loc : matplotlib legend location string.
    savepath   : optional path to save.

    Returns
    -------
    fig, ax
    """
    plt = _import_matplotlib_pyplot()
    st = style or PlotStyle()
    x = np.asarray(list(xgrid), dtype=np.float64)
    M = x.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=st.figsize, dpi=st.dpi)

    for name, arr in metrics.items():
        Y = np.where(np.isfinite(ensure_numpy(arr)), ensure_numpy(arr), np.nan).astype(np.float64)

        if Y.ndim == 1:
            if Y.shape[0] != M:
                raise ValueError(
                    f"Metric '{name}' length mismatch: got {Y.shape[0]} vs len(xgrid)={M}."
                )
            m, lo, hi = Y, None, None
        elif Y.ndim == 2:
            if Y.shape[1] != M:
                raise ValueError(f"Metric '{name}' shape mismatch: {Y.shape} vs (S,{M}).")
            m, lo, hi = mean_ci(Y, axis=0, conf=conf)
        else:
            raise ValueError(f"Metric '{name}' must be (M,) or (S,M); got {Y.shape}.")

        ax.plot(x, m, lw=2.0, label=name)
        if lo is not None and hi is not None:
            ax.fill_between(x, lo, hi, alpha=st.alpha_ci, linewidth=0)

    ax.set_xlabel(xlabel, fontsize=st.label_size)
    ax.set_ylabel(ylabel, fontsize=st.label_size)
    ax.tick_params(axis="both", labelsize=st.tick_size)
    if st.grid:
        ax.grid(True, linewidth=0.5, alpha=0.5)
    if title:
        ax.set_title(title, fontsize=st.title_size)
    ax.legend(loc=legend_loc)

    if st.tight_layout:
        fig.tight_layout()
    _savefig(fig, savepath, close=close)
    return fig, ax





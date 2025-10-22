# acpl/eval/plots.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = [
    "PlotStyle",
    "ensure_numpy",
    "tv_curve",
    "mean_ci",
    "plot_position_timelines",
    "plot_tv_curves",
    "plot_robustness_sweep",
]


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


def ensure_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Accept torch/numpy and return a CPU numpy array (no copy if already numpy).
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(x)}")


def _uniform(N: int) -> np.ndarray:
    return np.full((N,), 1.0 / max(1, N), dtype=np.float64)


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
    P = ensure_numpy(Pt).astype(np.float64)
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


def mean_ci(
    X: np.ndarray | torch.Tensor,
    axis: int = 0,
    conf: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean and symmetric normal-approx confidence interval along `axis`.

    Args
    ----
    X    : array (...). Typical case is seeds × time or seeds × xgrid.
    axis : reduce dimension (defaults to 0 for seeds-first layout)
    conf : confidence level (0<conf<1). Uses normal z-value approximation.

    Returns
    -------
    mean, lo, hi : arrays with `axis` removed.
    """
    x = ensure_numpy(X).astype(np.float64)
    if not (0.0 < conf < 1.0):
        raise ValueError("conf must be in (0,1).")

    # z for normal approx
    # 0.95 -> 1.96, 0.90 -> 1.645, 0.99 -> 2.576
    from scipy_stats_fallback import z_value  # local lightweight fallback below

    m = np.nanmean(x, axis=axis)
    s = np.nanstd(x, axis=axis, ddof=1)
    n = np.sum(np.isfinite(x), axis=axis)
    se = np.divide(s, np.maximum(1, np.sqrt(n)), out=np.zeros_like(s), where=n > 0)

    z = z_value(conf)
    lo = m - z * se
    hi = m + z * se
    return m, lo, hi


# --------------------------------------------------------------------------------------
#                                 Plot: P_t timelines
# --------------------------------------------------------------------------------------


def _pick_nodes_for_timelines(
    Pt: np.ndarray, topk: int | None, nodes: Sequence[int] | None
) -> list[int]:
    """
    Choose which nodes to display:
      - if `nodes` provided: use as-is (validated)
      - else pick `topk` nodes by mass at final time T-1
      - fallback to all nodes if N<=topk or topk is None
    """
    T, N = Pt.shape
    if nodes is not None:
        idx = list(nodes)
        if not all(0 <= i < N for i in idx):
            raise ValueError("nodes indices out of range.")
        return idx
    if (topk is None) or (topk >= N):
        return list(range(N))
    # score by final-time mass
    final = Pt[-1]
    idx = np.argsort(-final)[:topk].tolist()
    return idx


def plot_position_timelines(
    Pt: np.ndarray | torch.Tensor,
    *,
    nodes: Sequence[int] | None = None,
    topk: int | None = 12,
    sharey: bool = True,
    style: PlotStyle | None = None,
    title: str | None = "Position probabilities over time",
    savepath: str | None = None,
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
    st = style or PlotStyle()
    P = ensure_numpy(Pt).astype(np.float64)

    # Normalize shapes
    if P.ndim == 2:
        T, N = P.shape
        Pm = P[None, ...]  # (1,T,N)
    elif P.ndim == 3:
        S, T, N = P.shape
        Pm = P
    else:
        raise ValueError(f"Expected (T,N) or (S,T,N), got {P.shape}.")

    # pick nodes to show
    nodes_to_plot = _pick_nodes_for_timelines(Pm.mean(axis=0), topk=topk, nodes=nodes)
    K = len(nodes_to_plot)

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

        if P.ndim == 2:
            ax.plot(tgrid, P[:, v], lw=1.8)
        else:
            # mean ± CI
            m, lo, hi = mean_ci(Pm[..., v], axis=0, conf=0.95)
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

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig, axs


# --------------------------------------------------------------------------------------
#                                  Plot: TV curves
# --------------------------------------------------------------------------------------


def plot_tv_curves(
    Pt: np.ndarray | torch.Tensor,
    *,
    logy: bool = False,
    style: PlotStyle | None = None,
    title: str | None = "Total Variation to uniform vs time",
    savepath: str | None = None,
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
    st = style or PlotStyle()
    P = ensure_numpy(Pt).astype(np.float64)
    tv = tv_curve(P)  # (T,) or (S,T)

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
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
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
    savepath: str | None = None,
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
    st = style or PlotStyle()
    x = np.asarray(list(xgrid), dtype=np.float64)
    M = x.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=st.figsize, dpi=st.dpi)

    for name, arr in metrics.items():
        Y = ensure_numpy(arr).astype(np.float64)
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
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig, ax


# --------------------------------------------------------------------------------------
#                  Minimal fallback for z-values (no scipy dependency)
# --------------------------------------------------------------------------------------


class scipy_stats_fallback:
    """
    Provide z-values for common confidence levels without importing scipy.
    """

    _lookup = {
        0.80: 1.2815515655446004,
        0.85: 1.4395314709384563,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.975: 2.241402727604947,  # 2-sided 95% per-side (rare)
        0.98: 2.3263478740408408,
        0.99: 2.5758293035489004,
        0.999: 3.2905267314919255,
    }

    @classmethod
    def z_value(cls, conf: float) -> float:
        # nearest level if not exact
        if conf in cls._lookup:
            return cls._lookup[conf]
        # linear interpolate between the two nearest common points
        keys = sorted(cls._lookup.keys())
        if conf <= keys[0]:
            return cls._lookup[keys[0]]
        if conf >= keys[-1]:
            return cls._lookup[keys[-1]]
        for i in range(len(keys) - 1):
            if keys[i] <= conf <= keys[i + 1]:
                a, b = keys[i], keys[i + 1]
                wa = (b - conf) / (b - a)
                wb = 1.0 - wa
                return wa * cls._lookup[a] + wb * cls._lookup[b]
        return cls._lookup[0.95]  # safe default


# expose helper
def z_value(conf: float) -> float:
    return scipy_stats_fallback.z_value(conf)

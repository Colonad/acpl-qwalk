# acpl/data/features.py
from __future__ import annotations

import numpy as np


def node_features_line(
    degrees: np.ndarray,
    coords: np.ndarray,
    *,
    normalize_degree: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Build node features for the Phase-A line graph.

    Features (per node)
    -------------------
    1) degree (optionally normalized by max degree)
    2) 1-D coordinate in [0, 1] (already normalized)

    Parameters
    ----------
    degrees : np.ndarray, shape (n,), int64
        Degree of each node.
    coords : np.ndarray, shape (n, 1), float
        Normalized coordinate for each node (e.g., i/(n-1)).
    normalize_degree : bool
        If True, divide degree by max(degrees) (safe when all zeros).
    dtype : np.dtype
        Output dtype for the feature matrix (default float32).

    Returns
    -------
    np.ndarray, shape (n, 2), dtype=dtype
        Feature matrix [deg_feat, coord].
    """
    if degrees.ndim != 1:
        raise ValueError(f"degrees must be 1-D (got shape {degrees.shape})")
    if coords.ndim != 2 or coords.shape[1] != 1:
        raise ValueError(f"coords must have shape (n, 1) (got {coords.shape})")
    if degrees.shape[0] != coords.shape[0]:
        raise ValueError(
            "degrees and coords must have the same number of rows: "
            f"{degrees.shape[0]} != {coords.shape[0]}"
        )

    n = degrees.shape[0]

    # Degree feature (optionally normalized)
    deg = degrees.astype(np.float64, copy=False)
    if normalize_degree:
        max_deg = float(deg.max()) if n > 0 else 0.0
        scale = max(max_deg, 1.0)  # avoid divide-by-zero
        deg = deg / scale

    # Coordinate feature (already normalized to [0,1] by graphs.py)
    coord = coords.astype(np.float64, copy=False).reshape(n)

    feats = np.stack([deg, coord], axis=1).astype(dtype, copy=False)
    return feats

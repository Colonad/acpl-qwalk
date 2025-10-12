"""
ACPL-QWalk: GNN-based Adaptive Coin Policy for Discrete-Time Quantum Walks.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("acpl-qwalk")
except PackageNotFoundError:
    # fallback for editable installs before packaging
    __version__ = "0.1.0"

# acpl/sim/__init__.py
"""
Lightweight package init for `acpl.sim`.

Intentionally avoids importing heavy submodules (e.g., torch-dependent `shift`, `coins`)
to keep package import side effects minimal. Import submodules explicitly:

    from acpl.sim.portmap import make_flipflop_portmap
    from acpl.sim.shift import build_shift_indices
    from acpl.sim.coins import build_coin_layout

This keeps pytest/import-time robust and avoids circular/early import crashes.
"""

__all__ = []  # import submodules explicitly as needed

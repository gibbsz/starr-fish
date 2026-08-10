"""Figures. Needs matplotlib, which is the ``plots`` extra rather than a core
dependency, so nothing here is imported unless you ask for it::

    pip install -e '.[plots]'

    from baystarrfish.plotting import plot_spatial
    fig = plot_spatial(data, "copy_number", cre="CRE129", copies=copies)
"""

from __future__ import annotations

from .spatial import (
    MODE_COLORS,
    SPATIAL_MODES,
    SpatialMode,
    evidence_mask,
    plot_spatial,
    spatial_values,
)

__all__ = [
    "MODE_COLORS",
    "SPATIAL_MODES",
    "SpatialMode",
    "evidence_mask",
    "plot_spatial",
    "spatial_values",
]

"""Pixelgen brand colors, palettes, and theme shared across all plotting modules.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pixelator.common.plot.colors import (
    PIXELGEN_ACCENT_COLORS,
    PIXELGEN_CELL_PALETTE,
    PIXELGEN_GRADIENTS,
    PIXELGEN_PALETTES,
    create_discrete_palette,
    pixelgen_accent_colors,
    pixelgen_colormap,
    pixelgen_colorscale,
    pixelgen_discrete_colors,
    pixelgen_divergent_colormap,
    pixelgen_gradient,
    pixelgen_palette,
    pixelgen_sequential_colormap,
)
from pixelator.common.plot.theme import (
    STRIP_BACKGROUND_COLOR,
    pixelgen_theme,
    set_pixelgen_theme,
    style_facet_strips,
)

__all__ = [
    "PIXELGEN_ACCENT_COLORS",
    "PIXELGEN_CELL_PALETTE",
    "PIXELGEN_GRADIENTS",
    "PIXELGEN_PALETTES",
    "STRIP_BACKGROUND_COLOR",
    "create_discrete_palette",
    "pixelgen_accent_colors",
    "pixelgen_colormap",
    "pixelgen_colorscale",
    "pixelgen_discrete_colors",
    "pixelgen_divergent_colormap",
    "pixelgen_gradient",
    "pixelgen_palette",
    "pixelgen_sequential_colormap",
    "pixelgen_theme",
    "set_pixelgen_theme",
    "style_facet_strips",
]

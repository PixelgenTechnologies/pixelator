"""Pixelgen brand theme shared across all plotting modules.

Python/matplotlib port of the ``theme_pixelgen()`` ggplot2 theme from
Pixelgen's internal ``themes_and_palettes.R``: a white, grid-free panel with a
light grey background for facet strip titles.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import matplotlib as mpl
import seaborn as sns
from cycler import cycler

from pixelator.common.plot.colors import pixelgen_discrete_colors

#: Background color used for facet/strip titles, matching ggplot2's
#: ``theme_bw()`` default of ``element_rect(fill = "gray95")``.
STRIP_BACKGROUND_COLOR = "#F2F2F2"


def _pixelgen_rc_params() -> dict[str, Any]:
    return {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "axes.prop_cycle": cycler(color=pixelgen_discrete_colors()),
    }


def set_pixelgen_theme() -> None:
    """Apply the Pixelgen brand theme globally (until changed again).

    This sets matplotlib rcParams for a white, grid-free panel style matching
    ggplot2's ``theme_bw()`` with blanked gridlines, and sets the default
    discrete color cycle (both matplotlib's and seaborn's) to
    `pixelgen_discrete_colors`.
    """
    mpl.rcParams.update(_pixelgen_rc_params())
    sns.set_palette(pixelgen_discrete_colors())


@contextmanager
def pixelgen_theme() -> Iterator[None]:
    """Context manager that applies the Pixelgen brand theme temporarily.

    Example:
        with pixelgen_theme():
            fig, ax = plt.subplots()
            ax.plot(...)

    Yields:
        None. rcParams and the default color palette are restored on exit.
    """
    with mpl.rc_context(rc=_pixelgen_rc_params()):
        with sns.color_palette(pixelgen_discrete_colors()):
            yield


def style_facet_strips(
    grid: sns.axisgrid.FacetGrid, color: str = STRIP_BACKGROUND_COLOR
) -> sns.axisgrid.FacetGrid:
    """Style the strip titles of a seaborn `FacetGrid` with a colored background.

    Args:
        grid: A seaborn `FacetGrid` (e.g. as returned by ``sns.relplot`` or
            ``sns.FacetGrid``) to style in place.
        color: The background color for the facet strip titles. Defaults to
            `STRIP_BACKGROUND_COLOR`.

    Returns:
        The same `grid`, styled in place.
    """
    for ax in grid.axes.flat:
        title = ax.get_title()
        if title:
            ax.set_title(title, backgroundcolor=color)
    return grid

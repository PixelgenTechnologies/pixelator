"""Tests for the Pixelgen brand theme module.

Copyright © 2026 Pixelgen Technologies AB.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pixelator.common.plot.colors import pixelgen_discrete_colors
from pixelator.common.plot.theme import (
    STRIP_BACKGROUND_COLOR,
    pixelgen_theme,
    set_pixelgen_theme,
    style_facet_strips,
)


def test_pixelgen_theme_is_grid_free_and_boxed():
    """Inside the context manager the panel should have no grid and a full box."""
    with pixelgen_theme():
        assert mpl.rcParams["axes.grid"] is False
        assert mpl.rcParams["axes.spines.top"] is True
        assert mpl.rcParams["axes.spines.right"] is True
        assert mpl.rcParams["axes.facecolor"] == "white"


def test_pixelgen_theme_restores_previous_rcparams():
    """The context manager must not leak rcParam changes past its scope."""
    mpl.rcParams["axes.grid"] = True
    try:
        with pixelgen_theme():
            assert mpl.rcParams["axes.grid"] is False
        assert mpl.rcParams["axes.grid"] is True
    finally:
        mpl.rcParams["axes.grid"] = mpl.rcParamsDefault["axes.grid"]


def test_pixelgen_theme_sets_discrete_color_cycle():
    """Plots created inside the theme should default to the brand color cycle."""
    with pixelgen_theme():
        assert list(sns.color_palette()) == [
            mpl.colors.to_rgb(c)
            for c in pixelgen_discrete_colors()[: len(sns.color_palette())]
        ]


def test_set_pixelgen_theme_applies_globally():
    """`set_pixelgen_theme` should persist until rcParams are reset."""
    try:
        set_pixelgen_theme()
        assert mpl.rcParams["axes.grid"] is False
        assert mpl.rcParams["axes.facecolor"] == "white"
    finally:
        mpl.rcdefaults()
        sns.reset_defaults()


def test_style_facet_strips_sets_background_color():
    """Facet titles should get the branded strip background color applied."""
    data = pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "g": ["a", "a", "b", "b"]}
    )
    grid = sns.FacetGrid(data, col="g")
    grid.map(sns.scatterplot, "x", "y")
    style_facet_strips(grid)
    try:
        for ax in grid.axes.flat:
            facecolor = ax.title.get_bbox_patch().get_facecolor()
            assert facecolor == mpl.colors.to_rgba(STRIP_BACKGROUND_COLOR)
    finally:
        plt.close(grid.figure)

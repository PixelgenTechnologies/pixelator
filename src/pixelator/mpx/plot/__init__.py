"""Functions for creating plots that are useful with MPX data.

Copyright © 2023 Pixelgen Technologies AB.
"""

import warnings
from typing import Optional, Tuple

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

from pixelator.mpx.plot.constants import Color

sns.set_style("whitegrid")
jet_colormap = LinearSegmentedColormap.from_list("jet_colormap", Color.JETSET)


def _plot_joint_distribution(data, x, y, show_marginal, **kargs):
    g = sns.JointGrid(data, x=x, y=y)
    g.plot_marginals(sns.kdeplot, fill=True)
    g.ax_marg_x.axis("off")
    g.ax_marg_y.axis("off")
    g.plot_joint(
        sns.scatterplot,
        legend=False,
        data=data,
        hue="density",
        palette=jet_colormap,
        size=0.1,
    )
    if not show_marginal:
        g.ax_marg_x.set_visible(False)
        g.ax_marg_y.set_visible(False)
    return g


def _add_gate_box(
    data,
    gate: pd.Series | pd.DataFrame,
    marker1,
    marker2,
    facet_row=None,
    facet_column=None,
    ax=None,
    **kargs,
):
    if ax is None:
        ax = plt.gca()

    if facet_row is not None and facet_column is not None:
        condition = (data.iloc[0][facet_row], data.iloc[0][facet_column])
    elif facet_row is not None:
        condition = data.iloc[0][facet_row]
    elif facet_column is not None:
        condition = data.iloc[0][facet_column]
    else:
        condition = None

    if isinstance(gate, pd.DataFrame):
        if condition in gate.index:
            gate = gate.loc[condition, :]
        else:
            return
    ax.add_patch(
        Rectangle(
            xy=(gate["xmin"], gate["ymin"]),
            width=gate["xmax"] - gate["xmin"],
            height=gate["ymax"] - gate["ymin"],
            fill=False,
            linestyle="--",
            linewidth=1,
            edgecolor="black",
        )
    )

    # Counting data points inside the gate box
    inside_box = (
        (data[marker1] >= gate["xmin"])
        & (data[marker1] < gate["xmax"])
        & (data[marker2] >= gate["ymin"])
        & (data[marker2] < gate["ymax"])
    )
    inside_percentage = f"{inside_box.mean() * 100:.1f}%"
    ax.text(
        x=gate["xmin"],
        y=gate["ymax"],
        s=inside_percentage,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=10,
        color="black",
    )
    return


def density_scatter_plot(
    adata: anndata.AnnData,
    marker1: str,
    marker2: str,
    layer: str | None = None,
    facet_row: str | None = None,
    facet_column: str | None = None,
    gate: pd.Series | pd.DataFrame | None = None,
    show_marginal=False,
):
    """Pseudocolor density scatter plot.

    This function creates a scatter plot of abundance data for two markers colored
    based on the density of points:
    Example usage: `density_scatter_plot(pxl.adata, "CD3E", "CD4")`.
    It is also possible to specify one or two variables for faceting the plot by
    columns and rows:
    Example usage: `density_scatter_plot(pxl.adata, "CD3E", "CD4",
    facet_row = "stimulation", facet_column = "donor")`.
    `facet_row` and `facet_column` should be names of categorical columns in `adata.obs`.
    In addition, a gate can be specified as a Series with xmin, xmax, ymin, and ymax
    to mark a range for components of interest. Alternatively, gate can be specified
    as a DataFrame to allow for different gate ranges for various conditions.
    When both facet_row and facet_column are specified, the condition becomes a
    tuple (facet_row, facet_column), if only one is specified, that parameter
    becomes the condition. The condition permuations are used as the index of
    the gate.
    Example usage:
    gate = pd.DataFrame(columns = ["xmin", "ymin", "xmax", "ymax"])
    gate.loc["Resting"] = [2, 2, 5, 4]
    gate.loc["PHA"] = [1.5, 1.5, 5, 4]
    fig, ax = density_scatter_plot(pixel.adata, "CD3E", "CD4", layer="dsb",
    facet_column="sample", gate=gate)

    Args:
        adata: Anndata object containing the marker abundance data per component.
        marker1: The first marker to plot (x-axis).
        marker2: The second marker to plot (y-axis).
        layer: The anndata layer (e.g. transformation) to use for the marker data. Defaults to None.
        facet_row: The column to use for faceting the plot by rows. Defaults to None.
        facet_column: The column to use for faceting the plot by columns. Defaults to None.
        gate: The gate to use for marking a range of interest. Defaults to None.
        show_marginal: Whether to show marginal distributions. Defaults to False.
    """
    layer_data = adata.to_df(layer)
    data = layer_data.loc[:, [marker1, marker2]]
    data.loc[:, "density"] = gaussian_kde(data.T)(data.T)
    if facet_column is not None:
        data.loc[:, facet_column] = adata.obs.loc[:, facet_column]

    if facet_row is not None:
        data.loc[:, facet_row] = adata.obs.loc[:, facet_row]

    if facet_column is not None or facet_row is not None:
        if show_marginal:
            warnings.warn("show_marginal is not supported for faceted plots.")
        plot_grid = sns.FacetGrid(data=data, col=facet_column, row=facet_row)
        plot_grid.map_dataframe(
            sns.scatterplot,
            x=marker1,
            y=marker2,
            hue="density",
            palette=jet_colormap,
            size=0.1,
        )
        if gate is not None:
            plot_grid.map_dataframe(
                _add_gate_box,
                gate=gate,
                marker1=marker1,
                marker2=marker2,
                facet_row=facet_row,
                facet_column=facet_column,
            )
        plot_grid.refline(x=0, y=0)
    else:
        plot_grid = _plot_joint_distribution(
            data, x=marker1, y=marker2, show_marginal=show_marginal
        )
        if gate is not None:
            _add_gate_box(
                data, marker1=marker1, marker2=marker2, ax=plot_grid.ax_joint, gate=gate
            )
        plot_grid.ax_joint.axhline(0, color="black", linewidth=1, linestyle="--")
        plot_grid.ax_joint.axvline(0, color="black", linewidth=1, linestyle="--")
    return plt.gcf(), plt.gca()

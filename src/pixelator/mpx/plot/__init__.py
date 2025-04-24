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
from pixelator.mpx.plot.layout_plots import (
    plot_2d_graph,
    plot_3d_from_coordinates,
    plot_3d_graph,
    plot_3d_heatmap,
)
from pixelator.mpx.plot.spatial_analysis_plots import (
    plot_colocalization_diff_heatmap,
    plot_colocalization_diff_volcano,
    plot_colocalization_heatmap,
    plot_polarity_diff_volcano,
)

sns.set_style("whitegrid")
jet_colormap = LinearSegmentedColormap.from_list("jet_colormap", Color.JETSET)


def scatter_umi_per_upia_vs_tau(
    data: pd.DataFrame, group_by: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a scatter plot of pixel content vs marker specificity (Tau).

    :param data: a pandas DataFrame with the columns 'umi_per_upia', 'tau', and 'tau_type'.
    :param group_by: a column in the DataFrame to group the plot by.

    :return: a scatter plot of pixel content vs marker specificity (Tau).
    :rtype: Tuple[plt.Figure, plt.Axes]
    :raises: ValueError if the required columns are not present in the DataFrame
    :raises: AssertionError if the data types are invalid
    """
    # Validate data
    required_columns = ["umi_per_upia", "tau", "tau_type"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(
            "'umi_per_upia', 'tau' and 'tau_type' must be available in the DataFrame"
        )
    if not all(
        [
            data["umi_per_upia"].dtype == "float64",
            data["tau"].dtype == "float64",
            data["tau_type"].dtype in ["object", "category"],
        ]
    ):
        raise AssertionError("Invalid data types")

    if group_by is not None:
        if group_by not in data.columns:
            raise ValueError(f"'{group_by}' is missing")
        assert data[group_by].dtype in [
            "object",
            "category",
        ], "'group_by' must be a character or factor"

    # Define palette
    palette = {
        "high": Color.ORANGERED2,
        "low": Color.SKYBLUE3,
        "normal": Color.LIGHTGREY,
    }

    # create plot
    grid = (
        sns.FacetGrid(
            data,
            col=group_by,
            hue="tau_type",
            palette=palette,
            height=4,
        )
        if group_by is not None
        else sns.FacetGrid(data, hue="tau_type", palette=palette, height=4)
    )
    grid.map(sns.scatterplot, "tau", "umi_per_upia")
    grid.set(yscale="log")
    grid.set_axis_labels("Marker specificity (Tau)", "Pixel content (UMI/UPIA)")
    grid.add_legend(title="Tau Type")

    return plt.gcf(), plt.gca()


def molecule_rank_plot(
    data: pd.DataFrame, group_by: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the number of molecules per component against its molecule rank.

    :param data: a pandas DataFrame with a column 'molecules' containing edge counts for MPX
    components.
    :param group_by: a column in the DataFrame to group the plot by.

    :return: a plot showing the number of molecules per component against its edge rank used
    for quality control.
    :rtype: Tuple[plt.Figure, plt.Axes]
    :raises: AssertionError if the required column(s) are not present in the DataFrame
    :raises: ValueError if the data types are invalid
    """
    if "molecules" not in data.columns and "edges" in data.columns:
        data["molecules"] = data["edges"]
    assert "molecules" in data.columns, "column 'molecules' is missing from DataFrame"
    assert (
        isinstance(data["molecules"], pd.Series) and data["molecules"].dtype == int
    ), "'molecules' must be a vector of integer values"

    if group_by is not None:
        assert group_by in data.columns, (
            f"group variable '{group_by}' not found in DataFrame"
        )

        if data[group_by].dtype not in ["object", "category"]:
            raise ValueError(
                f"Invalid class '{data[group_by].dtype}' for column '{group_by}'. "
                f"Expected a string or categorical value"
            )
        else:
            edge_rank_df = data[[group_by, "molecules"]].copy()
            edge_rank_df["rank"] = edge_rank_df.groupby([group_by])["molecules"].rank(
                ascending=False, method="first"
            )
    else:
        edge_rank_df = data[["molecules"]].copy()
        edge_rank_df["rank"] = edge_rank_df["molecules"].rank(
            ascending=False, method="first"
        )

    (
        sns.relplot(
            data=edge_rank_df, x="rank", y="molecules", hue=group_by, aspect=1.6
        )
        .set(xscale="log", yscale="log")
        .set_xlabels("Component rank (by number of molecules)")
        .set_ylabels("Number of molecules")
    )

    return plt.gcf(), plt.gca()


def edge_rank_plot(
    data: pd.DataFrame, group_by: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the number of edges per component against its edge rank.

    :param data: a pandas DataFrame with a column 'edges' containing edge counts for MPX
    components.
    :param group_by: a column in the DataFrame to group the plot by.

    :return: a plot showing the number of edges per component against its edge rank used
    for quality control.
    :rtype: Tuple[plt.Figure, plt.Axes]
    :raises: AssertionError if the required column(s) are not present in the DataFrame
    :raises: ValueError if the data types are invalid
    """
    warnings.warn(
        "edge_rank_plot is deprecated and will be removed in a future version. Use molecule_rank_plot instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    assert "edges" in data.columns, "column 'edges' is missing from DataFrame"
    assert isinstance(data["edges"], pd.Series) and data["edges"].dtype == int, (
        "'edges' must be a vector of integer values"
    )

    if group_by is not None:
        assert group_by in data.columns, (
            f"group variable '{group_by}' not found in DataFrame"
        )

        if data[group_by].dtype not in ["object", "category"]:
            raise ValueError(
                f"Invalid class '{data[group_by].dtype}' for column '{group_by}'. "
                f"Expected a string or categorical value"
            )
        else:
            edge_rank_df = data[[group_by, "edges"]].copy()
            edge_rank_df["rank"] = edge_rank_df.groupby([group_by])["edges"].rank(
                ascending=False, method="first"
            )
    else:
        edge_rank_df = data[["edges"]].copy()
        edge_rank_df["rank"] = edge_rank_df["edges"].rank(
            ascending=False, method="first"
        )

    (
        sns.relplot(data=edge_rank_df, x="rank", y="edges", hue=group_by, aspect=1.6)
        .set(xscale="log", yscale="log")
        .set_xlabels("Component rank (by number of edges)")
        .set_ylabels("Number of edges")
    )

    return plt.gcf(), plt.gca()


def cell_count_plot(
    data: pd.DataFrame,
    color_by: str,
    group_by: Optional[str] = None,
    flip_axes: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a bar plot showing the component counts by group(s).

    :param data: a pandas DataFrame with group variable(s).
    :param color_by: a column in the DataFrame to color the bars by. This will be used as
    the group variable if `group_by` is not provided.
    :param: group_by: a column in the DataFrame to group the bars by.

    :return: a bar plot with component counts by group(s).
    :rtype: Tuple[plt.Figure, plt.Axes]
    :raises: ValueError if the required grouping variables are missing in the DataFrame
    :raises: AssertionError if the data types are invalid
    """
    # Validate inputs
    if group_by is not None:
        if group_by not in data.columns:
            raise ValueError(f"'{group_by}' is missing from DataFrame")

        if not pd.api.types.is_categorical_dtype(
            data[group_by]
        ) and not pd.api.types.is_string_dtype(data[group_by]):
            raise ValueError("'group_by' must be a character or categorical")

        if group_by == color_by:
            raise ValueError("'group_by' and 'color_by' cannot be identical")

    if color_by not in data.columns:
        raise ValueError(f"'{color_by}' is missing from DataFrame")

    if not pd.api.types.is_categorical_dtype(
        data[color_by]
    ) and not pd.api.types.is_string_dtype(data[color_by]):
        raise ValueError("'color_by' must be a character or categorical")

    # Create plot
    if group_by is not None:
        grouped_data = data.groupby([group_by, color_by]).size().reset_index(name="n")
        if flip_axes:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                p = sns.barplot(
                    x="n", y=group_by, hue=color_by, data=grouped_data, dodge=True
                )
            p.set_xlabel("Count")
            p.set_ylabel(group_by)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                p = sns.barplot(
                    x=group_by, y="n", hue=color_by, data=grouped_data, dodge=True
                )
            p.set_ylabel("Count")
            p.set_xlabel(group_by)

        p.set_title(f"Cell counts per {color_by} split by {group_by}")
    else:
        grouped_data = data.groupby(color_by).size().reset_index(name="n")
        if flip_axes:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                p = sns.barplot(x="n", y=color_by, hue=color_by, data=grouped_data)
            p.set_xlabel("Count")
            p.set_ylabel(color_by)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                p = sns.barplot(x=color_by, y="n", hue=color_by, data=grouped_data)
            p.set_ylabel("Count")
            p.set_xlabel(color_by)
        p.set_title(f"Cell counts per {color_by}")

    return plt.gcf(), plt.gca()


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


    :param adata: Anndata object containing the marker abundance data per
     component.
    :param marker1: The first marker to plot (x-axis).
    :param marker2: The second marker to plot (y-axis).
    :param layer: The anndata layer (e.g. transformation) to use for the marker
     data. Defaults to None.
    :param facet_row: The column to use for faceting the plot by rows.
     Defaults to None.
    :param facet_column: The column to use for faceting the plot by columns.
     Defaults to None.
    :param gate: The gate to use for marking a range of interest. Defaults to
     None.
    :param show_marginal: Whether to show marginal distributions. Defaults to False.
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


def abundance_colocalization_plot(
    pixel,
    markers_x: list[str],
    markers_y: list[str],
    layer=None,
    colocalization_column="pearson_z",
):
    """Plot abundance of markers x and y with colocalization as color.

    :param pixel: Pixel object containing the data.
    :param markers_x: List of markers for the x-axis.
    :param markers_y: List of markers for the y-axis.
    :param layer: The anndata layer (e.g. transformation) to use for the marker data.
    :param colocalization_column: The column in the colocalization table to use for
        colocalization values. Defaults to "pearson_z".
    :return: a scatter plot of marker abundance with colocalization as color.
    """
    data = pixel.adata.to_df(layer)
    merged_data = pd.DataFrame()
    for i, mx in enumerate(markers_x):
        for j, my in enumerate(markers_y):
            marker_pair_rows = (
                (pixel.colocalization["marker_1"] == mx)
                & (pixel.colocalization["marker_2"] == my)
            ) | (
                (pixel.colocalization["marker_1"] == my)
                & (pixel.colocalization["marker_2"] == mx)
            )

            coloc_data = pixel.colocalization.loc[marker_pair_rows, :].set_index(
                "component"
            )[colocalization_column]
            data["colocalization"] = coloc_data
            data["colocalization_abs"] = data["colocalization"].abs()
            data["x_abundance"] = data[mx]
            data["y_abundance"] = data[my]
            data["marker_x"] = mx
            data["marker_y"] = my
            data.fillna(0)
            merged_data = pd.concat((merged_data, data), axis=0)
    plot_grid = sns.FacetGrid(data=merged_data, col="marker_x", row="marker_y")
    plot_grid.map_dataframe(
        sns.scatterplot,
        x="x_abundance",
        y="y_abundance",
        hue="colocalization",
        size="colocalization_abs",
        hue_norm=Normalize(
            vmin=merged_data["colocalization"].quantile(0.1),
            vmax=merged_data["colocalization"].quantile(0.9),
            clip=True,
        ),
        size_norm=Normalize(
            vmin=merged_data["colocalization_abs"].quantile(0.1),
            vmax=merged_data["colocalization_abs"].quantile(0.9),
            clip=True,
        ),
    )
    # TODO: See how the legend is determined based on the merged data to be
    # able to access actual marker sizes. Right now the _legend_data includes
    # many points with different sizes and and colors even though both are
    # normalized to the same range. So in the code below, we are only keeping
    # the first 5 points to include the color range and setting all the sizes
    # to a fixed value 5.
    for i in range(1, 6):
        if i >= len(plot_grid._legend_data):
            break
        plot_grid._legend_data[list(plot_grid._legend_data.keys())[i]].set_markersize(5)
    legend_data = {
        i: plot_grid._legend_data[i] for i in list(plot_grid._legend_data.keys())[:6]
    }
    plot_grid.add_legend(legend_data=legend_data)
    return plt.gcf(), plt.gca()

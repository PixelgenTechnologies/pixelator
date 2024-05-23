"""Functions for creating plots that are useful with MPX data.

Copyright © 2023 Pixelgen Technologies AB.
"""

import warnings
from typing import Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import scipy
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize

from pixelator.analysis.colocalization import get_differential_colocalization
from pixelator.graph import Graph
from pixelator.marks import experimental
from pixelator.pixeldataset import PixelDataset
from pixelator.plot.constants import Color

sns.set_style("whitegrid")


def _unit_sphere_surface(horizontal_resolution, vertical_resolution):
    horizontal_angles = np.linspace(0, 2 * np.pi, horizontal_resolution)
    vertical_angles = np.linspace(0, np.pi, vertical_resolution)

    X = np.outer(np.cos(horizontal_angles), np.sin(vertical_angles))
    Y = np.outer(np.sin(horizontal_angles), np.sin(vertical_angles))
    Z = np.outer(np.ones(np.size(horizontal_angles)), np.cos(vertical_angles))
    return X, Y, Z


def _calculate_distance_to_unit_sphere_zones(coordinates, unit_sphere_surface):
    X, Y, Z = unit_sphere_surface
    zones_on_sphere_surface = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
    return scipy.spatial.distance.cdist(
        zones_on_sphere_surface, coordinates, "euclidean"
    )


def _calculate_densities(coordinates, distance_cutoff, unit_sphere_surface):
    dist = _calculate_distance_to_unit_sphere_zones(
        coordinates=coordinates, unit_sphere_surface=unit_sphere_surface
    )
    raw_sums = np.sum(
        1 - (dist / distance_cutoff), where=(dist < distance_cutoff), axis=1
    )
    raw_sums[raw_sums <= 1] = 0
    densities = np.log(raw_sums, out=raw_sums, where=(raw_sums > 1))
    densities = densities / np.max(densities)
    return densities


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


def edge_rank_plot(
    data: pd.DataFrame, group_by: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the number of edges per component against its edge rank.

    :param data: a pandas DataFrame with a column 'edges' containing edge counts for MPX
    momponents.
    :param group_by: a column in the DataFrame to group the plot by.

    :return: a plot showing the number of edges per component against its edge rank used
    for quality control.
    :rtype: Tuple[plt.Figure, plt.Axes]
    :raises: AssertionError if the required column(s) are not present in the DataFrame
    :raises: ValueError if the data types are invalid
    """
    assert "edges" in data.columns, "column 'edges' is missing from DataFrame"
    assert (
        isinstance(data["edges"], pd.Series) and data["edges"].dtype == int
    ), "'edges' must be a vector of integer values"

    if group_by is not None:
        assert (
            group_by in data.columns
        ), f"group variable '{group_by}' not found in DataFrame"

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


def _get_component_graph(pxl_data: PixelDataset, component: str):
    component_edges = pxl_data.edgelist_lazy.filter(pl.col("component") == component)
    component_graph = Graph.from_edgelist(
        component_edges,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    return component_graph


def _get_coordinates(
    component_graph: Graph,
    layout_algorithm: str,
    cache_layout: bool = False,
    show_b_nodes: bool = False,
    random_seed: int | None = None,
) -> pd.DataFrame:
    coordinates = component_graph.layout_coordinates(
        layout_algorithm=layout_algorithm,  # type: ignore
        cache=cache_layout,
        only_keep_a_pixels=not show_b_nodes,
        random_seed=random_seed,
    )
    filtered_coordinates = coordinates
    filtered_coordinates["pixel_type"] = [
        component_graph.raw.nodes[ind]["pixel_type"]
        for ind in filtered_coordinates.index
    ]

    if not show_b_nodes:
        filtered_coordinates = filtered_coordinates[
            filtered_coordinates["pixel_type"] == "A"
        ]

    edgelist = pd.DataFrame(component_graph.es)
    edgelist = edgelist[edgelist[0].isin(filtered_coordinates.index)]
    edgelist = edgelist[edgelist[1].isin(filtered_coordinates.index)]
    edgelist = edgelist.loc[:, [0, 1]].to_numpy()

    return filtered_coordinates, edgelist


def _plot_for_legend(coordinates: pd.DataFrame, axis, show_b_nodes, cmap, node_size):
    a_node_example = coordinates[coordinates["pixel_type"] == "A"].iloc[0, :]
    axis.scatter(
        a_node_example["x"],
        a_node_example["y"],
        c=0,
        cmap=cmap,
        vmin=0,
        s=node_size,
        label="A-nodes",
    )
    if show_b_nodes:
        b_node_example = coordinates[coordinates["pixel_type"] == "B"].iloc[0, :]
        axis.scatter(
            b_node_example["x"],
            b_node_example["y"],
            c=1,
            cmap=cmap,
            vmin=0,
            s=node_size,
            label="B-nodes",
        )


def _decorate_plot(
    im=None,
    include_colorbar: bool = False,
    vmax: float = 0,
    cmap: str = "cool",
    ax=None,
    fig=None,
    legend_ax=None,
):
    if include_colorbar:
        if isinstance(ax, np.ndarray):
            fig.colorbar(
                im,
                ax=ax.ravel().tolist(),
                cmap=cmap,
                norm=Normalize(vmin=0, vmax=vmax),
            )
        else:
            fig.colorbar(
                im,
                ax=ax,
                cmap=cmap,
                norm=Normalize(vmin=0, vmax=vmax),
            )
    if legend_ax is not None:
        if not include_colorbar:
            handles, labels = legend_ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="right")
            plt.subplots_adjust(right=0.80)
        else:
            raise AssertionError(
                "Plotting pixel_type together with other markers is not supported."
            )


def _get_current_axis(ax, i_m, i_c, marker_list, component):
    if len(marker_list) == 1:
        if len(component) == 1:
            crnt_ax = ax
        else:
            crnt_ax = ax[i_c]
    else:
        if len(component) == 1:
            crnt_ax = ax[i_m]
        else:
            crnt_ax = ax[i_m, i_c]

    if i_c == 0:  # Set the y-label only for the first column
        crnt_ax.set_ylabel(marker_list[i_m], rotation=90, size="large")
    if i_m == 0:  # Set the title only for the first row
        crnt_ax.set_title(component[i_c])

    return crnt_ax


def _get_color_values(mark, filtered_coordinates, log_scale, vmax):
    if mark is None:
        color_val = Color.NETWORKX_NODE_COLOR
    elif mark == "pixel_type":
        color_val = filtered_coordinates["pixel_type"] == "B"
    elif mark not in filtered_coordinates.columns:
        raise AssertionError(f"Marker {mark} not found in the component graph.")
    else:
        if log_scale:
            color_val = np.log1p(filtered_coordinates.loc[:, mark])
        else:
            color_val = filtered_coordinates.loc[:, mark]
        vmax = max(vmax, np.max(color_val))

    return color_val, vmax


def plot_2d_graph(
    pxl_data: PixelDataset,
    component: Union["str", list],
    marker: str = "pixel_type",
    layout_algorithm: Literal["fruchterman_reingold", "kamada_kawai", "pmds"] = "pmds",
    show_edges: bool = False,
    log_scale: bool = True,
    node_size: float = 10.0,
    edge_width: float = 1.0,
    show_b_nodes: bool = False,
    cmap: str = "cool",
    alpha: float = 0.5,
    cache_layout: bool = False,
    random_seed: int | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a (collection of) 2D graph(s) based on the given pixel data.

    The graph can be plotted for one or a list of components.
    The graph nodes can be colored by a marker. The marker can be a (list of) marker(s) or "pixel_type".
    Example usage: plot_2d_graph(pxl, component=["PXLCMP0000000"], marker=["HLA-ABC", "HLA-RA"]).

    :param pxl_data: The pixel dataset to plot.
    :param component: The component(s) to plot. Defaults to None.
    :param marker: The marker attribute to use for coloring the nodes. Defaults to "pixel_type".
    :param layout_algorithm: The layout algorithm to use. Defaults to "pmds".
    :param show_edges: Whether to show the edges in the graph. Defaults to False.
    :param log_scale: Whether to use a logarithmic scale for the marker attribute. Defaults to True.
    :param node_size: The size of the nodes. Defaults to 10.0.
    :param edge_width: The width of the edges. Defaults to 1.0.
    :param show_b_nodes: Whether to show the B-nodes. Defaults to False.
    :param cmap: The colormap to use for coloring the nodes. Defaults to "cool".
    :param alpha: The alpha value for the nodes. Defaults to 0.7.
    :param cache_layout: Whether to cache the layout coordinates. Defaults to False.
    :param random_seed: The random seed to use for the layout algorithm. Defaults to None.

    :return: The figure and axes objects of the plot.
    :rtype: Tuple[plt.Figure, plt.Axes]

    :raises: AssertionError if the marker is not found in the component graph.
    :raises: AssertionError if no nodes are found with the specified marker.
    :raises: AssertionError if "pixel_type" is in the markers together with other markers.

    """
    if isinstance(component, str):
        component = [component]

    if isinstance(marker, str):
        marker_list = [marker]
    else:
        marker_list = marker

    fig, ax = plt.subplots(nrows=len(marker_list), ncols=len(component))

    include_colorbar = False
    vmax = 0  # maximum value for colorbar
    for i_c, comp in enumerate(component):
        component_graph = _get_component_graph(pxl_data=pxl_data, component=comp)
        coordinates, edgelist = _get_coordinates(
            component_graph=component_graph,
            layout_algorithm=layout_algorithm,
            cache_layout=cache_layout,
            show_b_nodes=show_b_nodes,
            random_seed=random_seed,
        )

        for i_m, mark in enumerate(marker_list):
            crnt_ax = _get_current_axis(ax, i_m, i_c, marker_list, component)
            color_val, vmax = _get_color_values(mark, coordinates, log_scale, vmax)
            if mark != "pixel_type":
                include_colorbar = True

            im = nx.draw_networkx(
                component_graph.raw,
                nodelist=coordinates.index,
                pos=coordinates.loc[:, ["x", "y"]].T.to_dict("list"),
                ax=crnt_ax,
                node_size=node_size,
                node_color=color_val,
                cmap=cmap,
                width=edge_width,
                with_labels=False,
                edgelist=edgelist if show_edges else [],
                label="_nolegend_",
                alpha=alpha,
            )

            if mark == "pixel_type":
                # Re-plot one point from each pixel type to add a legend
                _plot_for_legend(coordinates, crnt_ax, show_b_nodes, cmap, node_size)
                legend_ax = crnt_ax
            else:
                legend_ax = None

            crnt_ax.grid(False)
            crnt_ax.spines[:].set_visible(False)
            crnt_ax.set_xticks([])
            crnt_ax.set_yticks([])

    _decorate_plot(
        im=im,
        include_colorbar=include_colorbar,
        vmax=vmax,
        cmap=cmap,
        ax=ax,
        fig=fig,
        legend_ax=legend_ax,
    )

    return fig, ax


def plot_3d_from_coordinates(
    coordinates: pd.DataFrame,
    node_size: float = 3.0,
    opacity: float = 0.4,
    cmap: str = "Inferno",
    suppress_fig: bool = False,
) -> go.Figure:
    """Plot a 3D graph from the given coordinates.

    :param coordinates: The coordinates to plot.
    :param node_size: The size of the nodes. Defaults to 3.0.
    :param opacity: The opacity of the nodes. Defaults to 0.4.
    :param cmap: The colormap to use for coloring the nodes. Defaults to "Inferno".
    :param suppress_fig: Whether to suppress (i.e. not plot) the figure. Defaults to False.
    :return: The plotted 3D graph.
    :rtype: go.Figure

    """
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coordinates["x"],
                y=coordinates["y"],
                z=coordinates["z"],
                mode="markers",
                marker=dict(
                    size=node_size,
                    color=coordinates["color"],
                    colorscale=cmap,
                    opacity=opacity,
                    colorbar=dict(thickness=20, title="color"),
                ),
            )
        ]
    )

    if not suppress_fig:
        fig.show()
    return fig


def plot_3d_graph(
    pxl_data: PixelDataset,
    component: str,
    marker: Union[list, None] = None,
    layout_algorithm: Literal[
        "fruchterman_reingold_3d", "kamada_kawai_3d", "pmds_3d"
    ] = "fruchterman_reingold_3d",
    log_scale: bool = True,
    normalize: bool = False,
    node_size: float = 3.0,
    opacity: float = 0.4,
    show_b_nodes: bool = False,
    cmap: str = "Inferno",
    cache_layout: bool = False,
    suppress_fig: bool = False,
) -> go.Figure:
    """Plot a 3D graph of the specified component in the given PixelDataset.

    :param pxl_data: The PixelDataset containing the data.
    :param component: The component to plot.
    :param marker: The marker to use for coloring the nodes. Defaults to None.
    :param layout_algorithm: The layout algorithm to use for positioning the nodes. Defaults to "fruchterman_reingold_3d".
    :param log_scale: Whether to apply logarithmic scaling to the marker values. Defaults to True.
    :param normalize: Whether to normalize the coordinates. Defaults to False.
    :param node_size: The size of the nodes. Defaults to 3.0.
    :param opacity: The opacity of the nodes. Defaults to 0.4.
    :param show_b_nodes: Whether to show nodes of type B. Defaults to False.
    :param cmap: The colormap to use for coloring the nodes. Defaults to "Inferno".
    :param cache_layout: Whether to cache the layout coordinates. Defaults to False.
    :param suppress_fig: Whether to suppress (i.e. not plot) the figure. Defaults to False.
    :return: The plotted 3D graph.
    :rtype: go.Figure

    """
    component_graph = _get_component_graph(pxl_data=pxl_data, component=component)
    coordinates, _ = _get_coordinates(
        component_graph=component_graph,
        layout_algorithm=layout_algorithm,
        cache_layout=cache_layout,
        show_b_nodes=show_b_nodes,
    )

    if marker is not None:
        if log_scale:
            coordinates["color"] = np.log1p(coordinates[marker])
        else:
            coordinates["color"] = coordinates[marker]
    else:
        coordinates["color"] = Color.SKYBLUE3

    if normalize:
        coordinates[["x", "y", "z"]] = coordinates[["x_norm", "y_norm", "z_norm"]]

    fig = plot_3d_from_coordinates(
        coordinates=coordinates,
        node_size=node_size,
        opacity=opacity,
        cmap=cmap,
        suppress_fig=True,
    )

    fig.update_layout(title=component)

    if not suppress_fig:
        fig.show()

    return fig


def _pivot_colocalization_data(
    colocalization_data: pd.DataFrame,
    value_col: str = "pearson",
    markers: Union[list, None] = None,
):
    colocalization_data_pivot = pd.pivot_table(
        colocalization_data,
        index="marker_1",
        columns="marker_2",
        values=value_col,
        fill_value=0,
    )

    if markers is not None:
        colocalization_data_pivot = colocalization_data_pivot.loc[markers, markers]

    for m in colocalization_data_pivot.index:
        colocalization_data_pivot.loc[m, m] = 0  # remove autocorrelations

    colocalization_data_pivot = colocalization_data_pivot.fillna(0)

    return colocalization_data_pivot


def _make_colocalization_symmetric(
    colocalization_data: pd.DataFrame,
    value_col: str = "pearson",
):
    colocalization_data = pd.DataFrame(
        np.concatenate(
            [
                colocalization_data[["marker_1", "marker_2", value_col]].to_numpy(),
                colocalization_data[["marker_2", "marker_1", value_col]].to_numpy(),
            ],
        ),
        columns=["marker_1", "marker_2", value_col],
    )
    colocalization_data = (
        colocalization_data.groupby(["marker_1", "marker_2"])[value_col]
        .apply(lambda x: np.mean(x))
        .reset_index()
    )

    return colocalization_data


def plot_colocalization_heatmap(
    colocalization_data: pd.DataFrame,
    markers: Union[list, None] = None,
    cmap: str = "vlag",
    use_z_scores: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a colocalization heatmap based on the provided colocalization data.

    The colocalization_data DataFrame should contain the columns "marker_1", "marker_2", "pearson", "pearson_z".
    Example usage: plot_colocalization_heatmap(pxl.colocalization).

    :param colocalization_data: The colocalization data to plot. The colocalization data frame that can be found in a pixel variable "pxl" through pxl.colocalization. The data frame should contain the columns "marker_1", "marker_2", "pearson", "pearson_z", and "component".
    :param markers: The markers to include in the heatmap. Defaults to None.
    :param cmap: The colormap to use for the heatmap. Defaults to "vlag".
    :param use_z_scores: Whether to use z-scores. Defaults to False.

    :return: The figure and axes objects of the plot.
    :rtype: Tuple[plt.Figure, plt.Axes]

    """
    if use_z_scores:
        value_col = "pearson_z"
    else:
        value_col = "pearson"

    colocalization_data = _make_colocalization_symmetric(colocalization_data, value_col)

    colocalization_data_pivot = _pivot_colocalization_data(
        colocalization_data, value_col, markers=markers
    )
    sns.clustermap(
        colocalization_data_pivot,
        yticklabels=True,
        xticklabels=True,
        linewidths=0.1,
        method="complete",
        cmap=cmap,
    )

    return plt.gcf(), plt.gca()


def _get_top_marker_pairs(
    colocalization_data: pd.DataFrame,
    n_top_marker_pairs: int,
    value_col: str = "pearson",
) -> list:
    colocalization_data["abs_val"] = colocalization_data[value_col].abs()
    top_marker_pairs = colocalization_data.nlargest(n_top_marker_pairs, "abs_val")
    top_markers = list(
        set(top_marker_pairs["marker_1"]).union(set(top_marker_pairs["marker_2"]))
    )

    return top_markers


def plot_colocalization_diff_heatmap(
    colocalization_data: pd.DataFrame,
    target: str,
    reference: str,
    contrast_column: str = "sample",
    markers: Union[list, None] = None,
    n_top_marker_pairs: Union[int, None] = None,
    cmap: str = "vlag",
    use_z_score: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the differential colocalization between reference and target components.

    Example usage: plot_colocalization_diff_heatmap(pxl.colocalization, target:"stimulated", reference:"control", contrast_column="sample").

    :param colocalization_data: The colocalization data frame that can be found in a pixel variable "pxl" through pxl.colocalization. The data frame should contain the columns "marker_1", "marker_2", "pearson", "pearson_z", and the contrast_column.
    :param target: The label for target components in the contrast_column.
    :param reference: The label for reference components in the contrast_column.
    :param contrast_column: The column to use for the contrast. Defaults to "sample".
    :param markers: The markers to include in the heatmap. Defaults to None. At most only one of n_top_marker_pairs or markers should be provided.
    :param n_top_marker_pairs: The number of top marker pairs to include in the heatmap. Defaults to None. At most only one of n_top_marker_pairs or markers should be provided.
    :param cmap: The colormap to use for the heatmap. Defaults to "vlag".
    :param use_z_score: Whether to use the z-score. Defaults to True.

    :return: The figure and axes objects of the plot.
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    assert (
        markers is None or n_top_marker_pairs is None
    ), "Only one of markers or n_top_marker_pairs can be provided."

    if use_z_score:
        value_col = "pearson_z"
    else:
        value_col = "pearson"

    if markers is not None:
        filter_mask = (colocalization_data["marker_1"].isin(markers)) & (
            colocalization_data["marker_2"].isin(markers)
        )
        colocalization_data = colocalization_data[filter_mask]

    differential_colocalization = get_differential_colocalization(
        colocalization_data,
        target=target,
        reference=reference,
        contrast_column=contrast_column,
        use_z_score=use_z_score,
    )

    differential_colocalization = differential_colocalization.fillna(0).reset_index()

    if n_top_marker_pairs is not None:
        top_markers = _get_top_marker_pairs(
            differential_colocalization, n_top_marker_pairs, "median_difference"
        )
    else:
        top_markers = None

    # Making the differential colocalization symmetric
    differential_colocalization = _make_colocalization_symmetric(
        differential_colocalization, "median_difference"
    )

    pivoted_differential_colocalization = _pivot_colocalization_data(
        differential_colocalization,
        "median_difference",
        markers=top_markers,
    )

    max_value = np.max(np.abs(pivoted_differential_colocalization.to_numpy().flatten()))
    sns.clustermap(
        pivoted_differential_colocalization,
        yticklabels=True,
        xticklabels=True,
        method="complete",
        linewidths=0.1,
        vmin=-max_value,
        vmax=max_value,
        cmap=cmap,
    )

    return plt.gcf(), plt.gca()


def _add_top_marker_labels(
    differential_colocalization,
    ax,
    n_top_pairs: int = 5,
    min_log_p: float = 5.0,
):
    differential_colocalization = differential_colocalization.sort_values(
        "median_difference"
    )
    differential_colocalization = differential_colocalization.loc[
        -np.log10(differential_colocalization["p_adj"]) > min_log_p, :
    ]

    ## Labels for marker pair withs highest negative differential colocalization scores
    for _, row in differential_colocalization.head(n_top_pairs).iterrows():
        x, y = row[["median_difference", "p_adj"]]
        y = -np.log10(y)
        if x > 0:
            continue
        ax.text(x, y, row["markers"], horizontalalignment="left", fontsize="xx-small")

    ## Labels for marker pair with highest positive differential colocalization scores
    for _, row in differential_colocalization.tail(n_top_pairs).iterrows():
        x, y = row[["median_difference", "p_adj"]]
        y = -np.log10(y)
        if x < 0:
            continue
        ax.text(x, y, row["markers"], horizontalalignment="left", fontsize="xx-small")


def _add_target_mean_colocalizations(
    differential_colocalization,
    target_coloc,
    value_col,
):
    differential_colocalization = differential_colocalization.fillna(0).reset_index()

    target_values = (
        target_coloc.groupby(["marker_1", "marker_2"])[value_col].mean().reset_index()
    )
    differential_colocalization = pd.merge(
        target_values,
        differential_colocalization,
        on=["marker_1", "marker_2"],
        how="inner",
    )

    return differential_colocalization


def plot_colocalization_diff_volcano(
    colocalization_data: pd.DataFrame,
    target: str,
    reference: str,
    contrast_column: str = "sample",
    cmap: str = "vlag",
    use_z_score: bool = True,
    n_top_pairs: int = 5,
    min_log_p: float = 5.0,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate the volcano plot of differential colocalization between reference and target components.

    Example usage: `plot_colocalization_diff_volcano(pxl.colocalization, target:"stimulated", reference:"control", contrast_column="sample")`.

    :param colocalization_data: The colocalization data frame that can be found in a pixel variable
                                "pxl" through pxl.colocalization. The data frame should contain the
                                columns "marker_1", "marker_2", "pearson", "pearson_z", and the contrast_column.
    :param target: The label for target components in the contrast_column.
    :param reference: The label for reference components in the contrast_column.
    :param contrast_column: The column to use for the contrast. Defaults to "sample".
    :param cmap: The colormap to use for the heatmap. Defaults to "vlag".
    :param use_z_score: Whether to use the z-score. Defaults to True.
    :param n_top_pairs: Number of high value marker-pairs to label from positive and negative sides.
    :param min_log_p: marker-pairs only receive a label if -log10 of their p-value is higher than
                      this parameter.

    :return: The figure and axes objects of the plot.
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    if use_z_score:
        value_col = "pearson_z"
    else:
        value_col = "pearson"

    differential_colocalization = get_differential_colocalization(
        colocalization_data,
        target=target,
        reference=reference,
        contrast_column=contrast_column,
        use_z_score=use_z_score,
    )

    differential_colocalization = _add_target_mean_colocalizations(
        differential_colocalization,
        target_coloc=colocalization_data.loc[
            colocalization_data[contrast_column] == target, :
        ],
        value_col=value_col,
    )

    if ax is None:
        fig, ax = plt.subplots()

    p = ax.scatter(
        x=differential_colocalization["median_difference"],
        y=-np.log10(differential_colocalization["p_adj"]),
        c=differential_colocalization[value_col],
        s=20,
        marker="o",
        cmap=cmap,
    )

    ax.set(xlabel="Median difference", ylabel=r"$-\log_{10}$(adj. p-value)")
    fig = plt.gcf()
    fig.colorbar(p, label="Mean target colocalization score", cmap=cmap)
    _add_top_marker_labels(
        differential_colocalization,
        n_top_pairs=n_top_pairs,
        min_log_p=min_log_p,
        ax=ax,
    )

    return fig, ax


@experimental
def plot_3d_heatmap(
    component_graph: Graph,
    marker: str,
    distance_cutoff: float,
    layout_algorithm: Literal[
        "fruchterman_reingold_3d", "kamada_kawai_3d"
    ] = "fruchterman_reingold_3d",
    cache_layout: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 3D heatmap for the marker in the provided component.

    :param component_graph: A component graph to plot for.
    :param marker: marker to plot this for.
    :param distance_cutoff: a distance cutoff to use for determining size of
                            area to consider as close in the density calculation.
    :param layout_algorithm: Layout algorithm to use. Options are:
                            "fruchterman_reingold_3d" and "kamada_kawai_3d"
    :param cache_layout: set this to `True` to cache the layout
                         or faster computations on subsequent calls. This comes at the
                         cost of additional memory usage.
    :return: A matplotlib 3D heatmap figure, and it's associated Axes instance
    :rtype: Tuple[plt.Figure, plt.Axes]
    :raises: AssertionError if the provided `layout_algorithm` is not valid,
             or there are no with markers for the provided `marker`
    """
    coordinates = component_graph.layout_coordinates(
        layout_algorithm=layout_algorithm, cache=cache_layout
    )
    coordinates = coordinates[coordinates[marker] > 0]
    coordinates = np.array(coordinates[["x_norm", "y_norm", "z_norm"]])
    if len(coordinates) < 1:
        raise AssertionError(f"No nodes found with {marker}.")

    horizontal_resolution = 120
    vertical_resolution = 80
    X, Y, Z = _unit_sphere_surface(
        horizontal_resolution=horizontal_resolution,
        vertical_resolution=vertical_resolution,
    )

    densities = _calculate_densities(
        coordinates=coordinates,
        distance_cutoff=distance_cutoff,
        unit_sphere_surface=(X, Y, Z),
    )

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    color_scale = cm.inferno(  # type: ignore
        densities.reshape(horizontal_resolution, -1)
    )
    ax.plot_surface(  # type: ignore
        X,
        Y,
        Z,
        cstride=1,
        rstride=1,
        facecolors=color_scale,
        shade=False,
    )
    ax.set_axis_off()
    ax.set_box_aspect([1.0, 1.0, 1.0])  # type: ignore
    return fig, ax

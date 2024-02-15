"""Functions for creating plots that are useful with MPX data.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

from typing import Literal, Tuple, Union

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
import warnings

from pixelator.graph import Graph
from pixelator.analysis.colocalization import get_differential_colocalization
from pixelator.marks import experimental
from pixelator.pixeldataset import PixelDataset
from pixelator.plot.constants import NETWORKX_NODE_COLOR

sns.set_style("whitegrid")

from pixelator.plot.constants import (
    LIGHTGREY,
    ORANGERED2,
    SKYBLUE3,
)


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
    data: pd.DataFrame, group_by: str | None
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

    assert all(
        [
            data["umi_per_upia"].dtype == "float64",
            data["tau"].dtype == "float64",
            data["tau_type"].dtype in ["object", "category"],
        ]
    ), "Invalid data types"

    if group_by is not None:
        if group_by not in data.columns:
            raise ValueError(f"'{group_by}' is missing")
        assert data[group_by].dtype in [
            "object",
            "category",
        ], "'group_by' must be a character or factor"

    # Define palette
    palette = {"high": ORANGERED2, "low": SKYBLUE3, "normal": LIGHTGREY}

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


def cell_count_plot(
    data: pd.DataFrame,
    color_by: str,
    group_by: str | None,
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


def plot_2d_graph(
    pxl_data: PixelDataset,
    component: Union["str", list],
    marker: str = "pixel_type",
    layout_algorithm: Literal[
        "fruchterman_reingold", "kamada_kawai", "pmds"
    ] = "fruchterman_reingold",
    show_edges: bool = False,
    log_scale: bool = True,
    node_size: float = 10.0,
    edge_width: float = 1.0,
    show_b_nodes: bool = False,
    cmap: str = "cool",
    cache_layout: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 2D graph based on the given pixel data.

    :param pxl_data: The pixel dataset to plot.
    :param component: The component(s) to plot. Defaults to None.
    :param marker: The marker attribute to use for coloring the nodes. Defaults to "pixel_type".
    :param layout_algorithm: The layout algorithm to use. Defaults to "fruchterman_reingold".
    :param show_edges: Whether to show the edges in the graph. Defaults to False.
    :param log_scale: Whether to use a logarithmic scale for the marker attribute. Defaults to True.
    :param node_size: The size of the nodes. Defaults to 10.0.
    :param edge_width: The width of the edges. Defaults to 1.0.
    :param show_b_nodes: Whether to show the B-nodes. Defaults to False.
    :param cmap: The colormap to use for coloring the nodes. Defaults to "cool".
    :param cache_layout: Whether to cache the layout coordinates. Defaults to False.

    :return: The figure and axes objects of the plot.
    :rtype: Tuple[plt.Figure, plt.Axes]

    :raises: AssertionError if the marker is not found in the component graph.
    :raises: AssertionError if no nodes are found with the specified marker.

    """
    if isinstance(component, str):
        component_edges = pxl_data.edgelist_lazy.filter(
            pl.col("component") == component
        )
        component_graph = Graph.from_edgelist(
            component_edges,
            add_marker_counts=True,
            simplify=True,
            use_full_bipartite=True,
        )

    coordinates = component_graph.layout_coordinates(
        layout_algorithm=layout_algorithm,
        cache=cache_layout,
        only_keep_a_pixels=not show_b_nodes,
    )
    filtered_coordinates = coordinates
    filtered_coordinates["pixel_type"] = [
        component_graph.raw.nodes[ind]["pixel_type"]
        for ind in filtered_coordinates.index
    ]

    if marker is not None and marker != "pixel_type":
        if marker not in filtered_coordinates.columns:
            raise AssertionError(f"Marker {marker} not found in the component graph.")

        filtered_coordinates = filtered_coordinates.filter(
            items=np.nonzero(filtered_coordinates[marker] > 0)[0], axis=0
        )
        if len(filtered_coordinates) == 0:
            raise AssertionError(f"No nodes found with {marker}.")

    if not show_b_nodes:
        filtered_coordinates = filtered_coordinates[
            filtered_coordinates["pixel_type"] == "A"
        ]

    if show_edges:
        edgelist = pd.DataFrame(component_graph.es)
        edgelist = edgelist[edgelist[0].isin(filtered_coordinates.index)]
        edgelist = edgelist[edgelist[1].isin(filtered_coordinates.index)]
        edgelist = edgelist.loc[:, [0, 1]].to_numpy()

    color_val = NETWORKX_NODE_COLOR
    if marker is not None:
        if marker == "pixel_type":
            color_val = filtered_coordinates["pixel_type"] == "B"
        else:
            if log_scale:
                color_val = np.log1p(filtered_coordinates.loc[:, marker])
            else:
                color_val = filtered_coordinates.loc[:, marker]

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx(
        component_graph.raw,
        nodelist=filtered_coordinates.index,
        pos=filtered_coordinates.loc[:, ["x", "y"]].T.to_dict("list"),
        ax=ax,
        node_size=node_size,
        node_color=color_val,
        width=edge_width,
        with_labels=False,
        edgelist=edgelist if show_edges else [],
        cmap=cmap,
        label="_nolegend_",
    )

    # Plotting a single additional node to add a legend (Is there a better way?)
    if marker == "pixel_type":
        a_node_example = filtered_coordinates[
            filtered_coordinates["pixel_type"] == "A"
        ].iloc[0, :]
        ax.scatter(
            a_node_example["x"],
            a_node_example["y"],
            c=0,
            cmap=cmap,
            vmin=0,
            s=node_size,
            zorder=-9,
            label="A-nodes",
        )
        if show_b_nodes:
            b_node_example = filtered_coordinates[
                filtered_coordinates["pixel_type"] == "B"
            ].iloc[0, :]
            ax.scatter(
                b_node_example["x"],
                b_node_example["y"],
                c=1,
                cmap=cmap,
                vmin=0,
                s=node_size,
                zorder=-10,
                label="B-nodes",
            )
        ax.legend()
    elif marker is not None:
        plt.colorbar(
            cm.ScalarMappable(
                cmap=cmap, norm=Normalize(vmin=0, vmax=np.max(color_val))
            ),
            ax=ax,
            label=marker,
        )

    ax.axis("off")
    return fig, ax


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
    :return: The plotted 3D graph.
    :rtype: go.Figure

    """
    component_edges = pxl_data.edgelist_lazy.filter(pl.col("component") == component)
    component_graph = Graph.from_edgelist(
        component_edges,
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=True,
    )
    coordinates = component_graph.layout_coordinates(
        layout_algorithm=layout_algorithm,
        cache=cache_layout,
        only_keep_a_pixels=not show_b_nodes,
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

    if marker is not None and log_scale:
        filtered_coordinates[marker] = np.log1p(filtered_coordinates[marker])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=(
                    filtered_coordinates["x_norm"]
                    if normalize
                    else filtered_coordinates["x"]
                ),
                y=(
                    filtered_coordinates["y_norm"]
                    if normalize
                    else filtered_coordinates["y"]
                ),
                z=(
                    filtered_coordinates["z_norm"]
                    if normalize
                    else filtered_coordinates["z"]
                ),
                mode="markers",
                marker=dict(
                    size=node_size,
                    color=filtered_coordinates[marker] if marker is not None else None,
                    colorscale=cmap,
                    opacity=opacity,
                    colorbar=(
                        dict(thickness=20, title=marker) if marker is not None else None
                    ),
                ),
            )
        ]
    )

    fig.update_layout(title="component")
    fig.show()
    return fig


def plot_colocalization_heatmap(
    colocalization_data: pd.DataFrame,
    component: Union[str, None] = None,
    markers: Union[list, None] = None,
    cmap: str = "vlag",
    use_z_scores: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a colocalization heatmap based on the provided colocalization data.

    :param colocalization_data: The colocalization data to plot.
    :param component: The component to filter the colocalization data by. Defaults to None.
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

    if component is not None:
        colocalization_data = colocalization_data.set_index("component").filter(
            like=component, axis=0
        )
        colocalization_data = colocalization_data.reset_index()[
            ["marker_1", "marker_2", value_col]
        ]
    else:
        colocalization_data = colocalization_data.groupby(["marker_1", "marker_2"])[
            [value_col]
        ].apply(lambda x: np.mean(x))
        colocalization_data = colocalization_data.reset_index()

    colocalization_data.columns = [
        "marker_1",
        "marker_2",
        "pearson",
    ]

    colocalization_data_pivot = pd.pivot_table(
        colocalization_data,
        index=["marker_1"],
        columns=["marker_2"],
        values=[value_col],
        fill_value=0,
    )
    colocalization_data_pivot.columns = [
        col[1] for col in colocalization_data_pivot.columns
    ]  # Remove the term "pearson" from column indices

    if markers is not None:
        colocalization_data_pivot = colocalization_data_pivot.loc[markers, markers]

    for m in colocalization_data_pivot.index:
        colocalization_data_pivot.loc[m, m] = 0  # remove autocorrelations

    colocalization_data_pivot = colocalization_data_pivot.fillna(0)
    sns.clustermap(
        colocalization_data_pivot,
        yticklabels=True,
        xticklabels=True,
        method="complete",
        cmap=cmap,
    )

    return plt.gcf(), plt.gca()


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

    :param colocalization_data: The colocalization data frame.
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
        differential_colocalization["abs_val"] = differential_colocalization[
            value_col
        ].abs()
        top_marker_pairs = differential_colocalization.nlargest(
            n_top_marker_pairs, "abs_val"
        )
        top_markers = list(
            set(top_marker_pairs["marker_1"]).union(set(top_marker_pairs["marker_2"]))
        )

    # Making the differential colocalization symmetric
    differential_colocalization = pd.DataFrame(
        np.concatenate(
            [
                differential_colocalization[
                    ["marker_1", "marker_2", value_col]
                ].to_numpy(),
                differential_colocalization[
                    ["marker_2", "marker_1", value_col]
                ].to_numpy(),
            ],
        ),
        columns=["marker_1", "marker_2", value_col],
    )

    pivoted_differential_colocalization = pd.pivot_table(
        differential_colocalization,
        index="marker_1",
        columns="marker_2",
        values=value_col,
    )
    pivoted_differential_colocalization = (
        pivoted_differential_colocalization.infer_objects(copy=False).fillna(0)
    )

    if n_top_marker_pairs is not None:
        pivoted_differential_colocalization = pivoted_differential_colocalization.loc[
            top_markers, top_markers
        ]
    sns.clustermap(
        pivoted_differential_colocalization,
        yticklabels=True,
        xticklabels=True,
        method="complete",
        cmap=cmap,
    )

    return plt.gcf(), plt.gca()


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

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

from pixelator.graph import Graph
from pixelator.analysis.colocalization import get_differential_colocalization
from pixelator.marks import experimental
from pixelator.pixeldataset import PixelDataset

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


def plot_2d_graph(
    pxl_data: PixelDataset,
    component=Union["str", list],
    marker: str = "pixel_type",
    layout_algorithm: Literal[
        "fruchterman_reingold", "kamada_kawai", "pmds"
    ] = "fruchterman_reingold",
    show_edges=False,
    log_scale=True,
    node_size=10.0,
    edge_width=1.0,
    show_b_nodes=False,
    cmap="cool",
    cache_layout: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 2D graph based on the given pixel data.

    Args:
    ----
        pxl_data (PixelDataset): The pixel dataset to plot.
        component (Union[str, list], optional): The component(s) to plot. Defaults to None.
        marker (str, optional): The marker attribute to use for coloring the nodes. Defaults to "pixel_type".
        layout_algorithm (Literal["fruchterman_reingold", "kamada_kawai", "pmds"], optional): The layout algorithm to use. Defaults to "fruchterman_reingold".
        show_edges (bool, optional): Whether to show the edges in the graph. Defaults to False.
        log_scale (bool, optional): Whether to use a logarithmic scale for the marker attribute. Defaults to True.
        node_size (float, optional): The size of the nodes. Defaults to 10.0.
        edge_width (float, optional): The width of the edges. Defaults to 1.0.
        show_b_nodes (bool, optional): Whether to show the B-nodes. Defaults to False.
        cmap (str, optional): The colormap to use for coloring the nodes. Defaults to "cool".
        cache_layout (bool, optional): Whether to cache the layout coordinates. Defaults to False.

    Returns:
    -------
        Tuple[plt.Figure, plt.Axes]: The figure and axes objects of the plot.

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

    color_val = "#1f78b4"  # networkx default color
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
    marker=None,
    layout_algorithm: Literal[
        "fruchterman_reingold_3d", "kamada_kawai_3d", "pmds_3d"
    ] = "fruchterman_reingold_3d",
    log_scale=True,
    normalize=False,
    node_size=3.0,
    opacity=0.4,
    show_b_nodes=False,
    cmap="Inferno",
    cache_layout: bool = False,
):
    """Plot a 3D graph of the specified component in the given PixelDataset.

    Args:
    ----
        pxl_data (PixelDataset): The PixelDataset containing the data.
        component (str): The component to plot.
        marker (optional): The marker to use for coloring the nodes. Defaults to None.
        layout_algorithm (optional): The layout algorithm to use for positioning the nodes.
            Defaults to "fruchterman_reingold_3d".
        log_scale (bool): Whether to apply logarithmic scaling to the marker values. Defaults to True.
        normalize (bool): Whether to normalize the coordinates. Defaults to False.
        node_size (float): The size of the nodes. Defaults to 3.0.
        opacity (float): The opacity of the nodes. Defaults to 0.4.
        show_b_nodes (bool): Whether to show nodes of type B. Defaults to False.
        cmap (str): The colormap to use for coloring the nodes. Defaults to "Inferno".
        cache_layout (bool): Whether to cache the layout coordinates. Defaults to False.

    Returns:
    -------
        fig (go.Figure): The plotted 3D graph.

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
    cmap="vlag",
    use_z_scores=False,
):
    """Plot a colocalization heatmap based on the provided colocalization data.

    Args:
    ----
        colocalization_data (pd.DataFrame): The colocalization data to plot.
        component (Union[str, None], optional): The component to filter the colocalization data by. Defaults to None.
        markers (Union[list, None], optional): The markers to include in the heatmap. Defaults to None.
        cmap (str, optional): The colormap to use for the heatmap. Defaults to "vlag".
        use_z_scores (bool, optional): Whether to use z-scores. Defaults to False.

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
    source_mask: pd.Series,
    markers: Union[list, None] = None,
    n_top_marker_pairs: Union[int, None] = None,
    cmap="vlag",
    use_z_score=True,
):
    """Plot a colocalization differential heatmap.

    Args:
    ----
        colocalization_data (pd.DataFrame): The colocalization data.
        source_mask (pd.Series): A boolean series marking the rows that correspond the source data. False values correspond to the target data.
        markers (Union[list, None], optional): List of markers to include. Defaults to None.
        n_top_marker_pairs (Union[int, None], optional): Number of top marker pairs to include. Defaults to None.
        cmap (str, optional): The colormap to use. Defaults to "vlag".
        use_z_score (bool, optional): Whether to use z-scores. Defaults to True.

    Returns:
    -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and axes objects.

    """
    assert (
        markers is None or n_top_marker_pairs is None
    ), "Only one of markers or n_top_marker_pairs can be provided."

    if use_z_score:
        value_col = "pearson_z"
    else:
        value_col = "pearson"

    if markers is not None:
        colocalization_data = colocalization_data[
            (colocalization_data["marker_1"].isin(markers))
            & (colocalization_data["marker_2"].isin(markers))
        ]

    differential_colocalization = get_differential_colocalization(
        colocalization_data, source_mask, use_z_score=use_z_score
    )

    differential_colocalization = differential_colocalization.fillna(0)

    pivoted_differential_colocalization = pd.pivot_table(
        differential_colocalization,
        index="marker_1",
        columns="marker_2",
        values=value_col,
        fill_value=0,
    )
    print(pivoted_differential_colocalization.columns)
    print(pivoted_differential_colocalization.index)
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

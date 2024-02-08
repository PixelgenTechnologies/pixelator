"""Functions for creating plots that are useful with MPX data.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
import networkx as nx
import pandas as pd
from typing import Literal, Tuple

from pixelator.graph import Graph
from pixelator.marks import experimental


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
    component_graph: Graph,
    marker: str = None,
    layout_algorithm: Literal[
        "fruchterman_reingold", "kamada_kawai", "pmds"
    ] = "fruchterman_reingold",
    colors=Literal["lightgrey", "mistyrose", "red", "darkred"],
    plot_nodes=True,
    plot_edges=False,
    log_scale=True,
    node_size=0.5,
    edge_width=0.3,
    show_Bnodes=False,
    collect_scales=False,
    return_plot_list=False,
    cache_layout: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 2D graph for the marker in the provided component.

    :param component_graph: A component graph to plot for.
    :type component_graph: Graph
    :param marker: marker to plot this for.
    :type marker: str
    :param layout_algorithm: Layout algorithm to use. Options are:
                            "fruchterman_reingold" and "kamada_kawai"
    :type layout_algorithm: Literal["fruchterman_reingold", "kamada_kawai", "pmds"], optional
    :param colors: Colors to use for the plot, defaults to ["lightgrey", "mistyrose", "red", "darkred"]
    :type colors: Literal["lightgrey", "mistyrose", "red", "darkred"], optional
    :param plot_nodes: Whether to plot nodes, defaults to True
    :type plot_nodes: bool, optional
    :param plot_edges: Whether to plot edges, defaults to False
    :type plot_edges: bool, optional
    :param log_scale: Whether to use log scale for node size, defaults to True
    :type log_scale: bool, optional
    :param node_size: Size of the nodes, defaults to 0.5
    :type node_size: float, optional
    :param edge_width: Width of the edges, defaults to 0.3
    :type edge_width: float, optional
    :param show_Bnodes: Whether to show B-nodes, defaults to False
    :type show_Bnodes: bool, optional
    :param collect_scales: Whether to collect scales, defaults to False
    :type collect_scales: bool, optional
    :param return_plot_list: Whether to return a list of plots, defaults to False
    :type return_plot_list: bool, optional
    :param cache_layout: Whether to cache the layout for faster computations on subsequent calls, defaults to False
    :type cache_layout: bool, optional
    :return: A matplotlib 2D graph figure, and its associated Axes instance
    :rtype: Tuple[plt.Figure, plt.Axes]
    :raises AssertionError: If the provided `layout_algorithm` is not valid, or there are no nodes with the provided `marker`
    """
    coordinates = component_graph.layout_coordinates(
        layout_algorithm=layout_algorithm,
        cache=cache_layout,
        only_keep_a_pixels=not show_Bnodes,
    )
    filtered_coordinates = coordinates
    if marker is not None:
        filtered_coordinates = filtered_coordinates.filter(
            items=np.nonzero(filtered_coordinates[marker] > 0)[0]
        )
        if len(filtered_coordinates) == 0:
            raise AssertionError(f"No nodes found with {marker}.")

    fig, ax = plt.subplots(figsize=(6, 6))
    if not plot_edges or not show_Bnodes:
        edgelist = []
    else:
        edgelist = pd.DataFrame(component_graph.es)
        edgelist = edgelist[edgelist[0].isin(filtered_coordinates.index)]
        edgelist = edgelist[edgelist[1].isin(filtered_coordinates.index)]
        edgelist = edgelist.loc[:, [0, 1]].to_numpy()

    nx.draw_networkx(
        component_graph.raw,
        nodelist=filtered_coordinates.index,
        pos=coordinates.loc[:, ["x", "y"]].T.to_dict("list"),
        ax=ax,
        node_size=node_size,
        with_labels=False,
        edgelist=edgelist,
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

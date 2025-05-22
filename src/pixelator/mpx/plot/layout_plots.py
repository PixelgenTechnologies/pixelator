"""Functions for plotting component layouts for MPX data.

Copyright © 2024 Pixelgen Technologies AB.
"""

from typing import List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import scipy
from matplotlib import cm
from matplotlib.colors import Normalize

from pixelator.common.graph.backends.protocol import SupportedLayoutAlgorithm
from pixelator.common.marks import experimental
from pixelator.mpx.graph import Graph
from pixelator.mpx.pixeldataset import PixelDataset
from pixelator.mpx.plot.constants import Color


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
    pxl_data: PixelDataset,
    component: str,
    layout_algorithm: SupportedLayoutAlgorithm | None = None,
    cache_layout: bool = False,
    show_b_nodes: bool = False,
    random_seed: int | None = None,
) -> pd.DataFrame:
    component_graph = _get_component_graph(pxl_data=pxl_data, component=component)
    if (
        layout_algorithm is None
        and hasattr(pxl_data, "precomputed_layouts")
        and pxl_data.precomputed_layouts is not None
    ):
        coordinates = pxl_data.precomputed_layouts.filter(
            component_ids=component
        ).to_df()
    else:
        if layout_algorithm is None:
            layout_algorithm = "pmds_3d"
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

    return filtered_coordinates, edgelist, component_graph


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
    component: str | List[str],
    marker: str | List[str] = "pixel_type",
    layout_algorithm: SupportedLayoutAlgorithm | None = None,
    show_edges: bool = False,
    log_scale: bool = True,
    node_size: float = 10.0,
    edge_width: float = 1.0,
    show_b_nodes: bool = False,
    cmap: str = "cool",
    alpha: float = 0.5,
    cache_layout: bool = False,
    random_seed: int | None = None,
) -> Tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Plot a (collection of) 2D graph(s) based on the given pixel data.

    The graph can be plotted for one or a list of components.
    The graph nodes can be colored by a marker. The marker can be a (list of) marker(s) or "pixel_type".
    Example usage: plot_2d_graph(pxl, component=["PXLCMP0000000"], marker=["HLA-ABC", "HLA-RA"]).

    :param pxl_data: The pixel dataset to plot.
    :param component: The component(s) to plot. Defaults to None.
    :param marker: The marker attribute to use for coloring the nodes. Defaults to "pixel_type".
    :param layout_algorithm: The layout algorithm to use. Defaults to None (checking for pre-computed).
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
        coordinates, edgelist, component_graph = _get_coordinates(
            pxl_data=pxl_data,
            component=comp,
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
    layout_algorithm: SupportedLayoutAlgorithm | None = None,
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
    coordinates, _, _ = _get_coordinates(
        pxl_data=pxl_data,
        component=component,
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

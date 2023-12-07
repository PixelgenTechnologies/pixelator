"""Functions for creating plots that are useful with MPX data.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
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

"""Functions for creating plots from Pixelator data.

Copyright © 2023 Pixelgen Technologies AB.
"""

import warnings
from typing import Literal, Optional, Tuple

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

from pixelator.common.plot import (
    pixelgen_sequential_colormap,
    set_pixelgen_theme,
    style_facet_strips,
)

set_pixelgen_theme()
colormap = pixelgen_sequential_colormap("blues", min_level=5)


def _plot_joint_distribution(data, x, y, show_marginal, hue_norm=None, **kargs):
    g = sns.JointGrid(data, x=x, y=y)
    g.plot_marginals(sns.kdeplot, fill=True)
    g.ax_marg_x.axis("off")
    g.ax_marg_y.axis("off")
    scatter_kwargs = {
        "legend": False,
        "data": data,
        "hue": "density",
        "palette": colormap,
        "size": 0.1,
    }
    if hue_norm is not None:
        scatter_kwargs["hue_norm"] = hue_norm
    g.plot_joint(sns.scatterplot, **scatter_kwargs)
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


def _density_hue_norm(
    density: pd.Series, scale_to_max: bool
) -> tuple[float | None, float | None] | None:
    """Color-scale limits for a density hue.

    A 0–1 scale keeps a facet peak at the top of the colormap. Without scaling,
    limits are left to seaborn, except when every value is 0 or 1: seaborn
    would otherwise treat that hue as categorical and reject the colormap.

    Args:
        density: Density used to color the points.
        scale_to_max: Whether ``density`` was divided by its maximum.

    Returns:
        ``(0, 1)`` when scaled, ``(None, None)`` when every value is 0 or 1,
        and None when seaborn can infer a numeric scale on its own.
    """
    if scale_to_max:
        return (0.0, 1.0)
    values = density.to_numpy(dtype=float)
    if values.size and np.isin(values, [0.0, 1.0]).all():
        return (None, None)
    return None


def _kde_failure_message(n_points: int, error: Exception, coordinates: str) -> str:
    """Explain a failed kernel estimate and what to change."""
    cell = "cell" if n_points == 1 else "cells"
    subject = f"Cannot estimate the density of {coordinates} ({n_points} {cell})."
    if n_points < 2:
        return (
            f"{subject} "
            "A 2D kernel density needs at least two cells in different positions. "
            "Add cells, or facet the plot: a plot with density_scope='facet' "
            "with too few cells is drawn at a constant density instead of failing."
        )
    if isinstance(error, np.linalg.LinAlgError):
        return (
            f"{subject} "
            "The points have no area: they are identical or lie on a line. "
            "Choose two markers that vary independently. "
            "A non-faceted plot and density_scope='joint' fail in this case. "
            "A plot with density_scope='facet' that cannot be estimated "
            "is drawn at a constant density instead."
        )
    return f"{subject} {error}"


def _kde_density(
    points: np.ndarray,
    *,
    fallback: bool = False,
    coordinates: str = "these coordinates",
) -> np.ndarray:
    """Gaussian KDE at each row of ``points``.

    A facet with fewer than two points, or with no two-dimensional spread, cannot
    support a kernel estimate. When ``fallback`` is True those points get a
    constant density of 1 so that panel still draws. Otherwise a ``ValueError``
    names the coordinates and whether the problem is too few cells or no area.

    Args:
        points: Array of shape ``(n, 2)``.
        fallback: Replace a failed kernel estimate with a constant density of 1.
        coordinates: Marker names, or another label, used in the error message.

    Returns:
        Density of shape ``(n,)``.

    Raises:
        ValueError: If the kernel cannot be estimated and ``fallback`` is False.
    """
    try:
        return np.asarray(gaussian_kde(points.T)(points.T), dtype=float)
    except (ValueError, np.linalg.LinAlgError) as error:
        if not fallback:
            raise ValueError(
                _kde_failure_message(points.shape[0], error, coordinates)
            ) from error
        return np.ones(points.shape[0], dtype=float)


def _scale_density_to_max(density: np.ndarray) -> np.ndarray:
    """Divide ``density`` by its maximum so the peak is 1."""
    if density.size == 0:
        return density
    peak = float(np.max(density))
    if not np.isfinite(peak) or peak <= 0:
        return np.ones_like(density, dtype=float)
    return density / peak


def _estimate_point_density(
    points: np.ndarray,
    groups: pd.DataFrame | None = None,
    *,
    per_group: bool = False,
    scale_to_max: bool = False,
    coordinates: str = "these coordinates",
) -> np.ndarray:
    """Estimate scatter density, optionally separately inside each facet.

    Args:
        points: Array of shape ``(n, 2)`` with the plotted coordinates.
        groups: Facet labels, one row per point. Required when ``per_group``
            is True.
        per_group: Estimate a separate density for each combination of
            ``groups``.
        scale_to_max: Divide each estimate by its maximum. When ``per_group``
            is True this is the maximum inside that facet.
        coordinates: Marker names, or another label, used when a single
            estimate fails.

    Returns:
        Density of shape ``(n,)``, aligned with ``points``. A facet that cannot
        support a kernel estimate is filled with 1. A single estimate raises
        ``ValueError`` instead.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2).")

    if not per_group:
        density = _kde_density(points, fallback=False, coordinates=coordinates)
        return _scale_density_to_max(density) if scale_to_max else density

    if groups is None:
        raise ValueError("groups is required when per_group is True.")
    if len(groups) != points.shape[0]:
        raise ValueError("groups must have one row per point.")

    density = np.empty(points.shape[0], dtype=float)
    grouped = groups.reset_index(drop=True).groupby(
        list(groups.columns), sort=False, dropna=False, observed=True
    )
    for positions in grouped.indices.values():
        values = _kde_density(points[positions], fallback=True)
        if scale_to_max:
            values = _scale_density_to_max(values)
        density[positions] = values
    return density


def density_scatter_plot(
    adata: anndata.AnnData,
    marker1: str,
    marker2: str,
    layer: str | None = None,
    facet_row: str | None = None,
    facet_column: str | None = None,
    gate: pd.Series | pd.DataFrame | None = None,
    show_marginal: bool = False,
    density_scope: Literal["facet", "joint"] = "facet",
    scale_to_max: bool | None = None,
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

    Faceted plots estimate the density inside each facet and scale it to that
    facet's maximum, so each panel's colors reflect its own density.
    ``density_scope="joint"`` keeps the previous behavior: one density
    estimated on every point. Non-faceted plots are unchanged unless
    ``scale_to_max=True``. A facet with too few points, or with no
    two-dimensional spread, is drawn at a constant density. A non-faceted
    plot, and ``density_scope="joint"``, raise ``ValueError`` naming the
    markers and whether there are too few cells or the cells have no area.

    Args:
        adata: Anndata object containing the marker abundance data per component.
        marker1: The first marker to plot (x-axis).
        marker2: The second marker to plot (y-axis).
        layer: The anndata layer (e.g. transformation) to use for the marker data. Defaults to None.
        facet_row: The column to use for faceting the plot by rows. Defaults to None.
        facet_column: The column to use for faceting the plot by columns. Defaults to None.
        gate: The gate to use for marking a range of interest. Defaults to None.
        show_marginal: Whether to show marginal distributions. Defaults to False.
        density_scope: ``"facet"`` estimates density inside each facet.
            ``"joint"`` estimates one density on the whole dataset. Defaults to
            ``"facet"``. Has no effect when the plot is not faceted.
        scale_to_max: Divide each density by its maximum. Defaults to True for
            per-facet densities and False otherwise.

    Raises:
        ValueError: If ``density_scope`` is not ``"facet"`` or ``"joint"``,
            or if a non-faceted plot or ``density_scope="joint"`` has fewer
            than two cells or cells with no area.
    """
    if density_scope not in ("facet", "joint"):
        raise ValueError(
            f"density_scope must be 'facet' or 'joint', got {density_scope!r}."
        )

    layer_data = adata.to_df(layer)
    data = layer_data.loc[:, [marker1, marker2]].copy()
    facet_columns = [name for name in (facet_row, facet_column) if name is not None]
    for name in facet_columns:
        data.loc[:, name] = adata.obs.loc[:, name]

    per_group = density_scope == "facet" and bool(facet_columns)
    if scale_to_max is None:
        scale_to_max = per_group
    points = data.loc[:, [marker1, marker2]].to_numpy(dtype=float)
    groups = data.loc[:, facet_columns] if per_group else None
    data.loc[:, "density"] = _estimate_point_density(
        points,
        groups,
        per_group=per_group,
        scale_to_max=scale_to_max,
        coordinates=f"{marker1} against {marker2}",
    )

    if facet_column is not None or facet_row is not None:
        if show_marginal:
            warnings.warn("show_marginal is not supported for faceted plots.")
        plot_grid = sns.FacetGrid(data=data, col=facet_column, row=facet_row)
        # A fresh norm per panel. (0, 1) puts a scaled peak at the top of the
        # colormap. (None, None) still autoscales, but keeps a flat 0/1 density
        # numeric so seaborn does not reject the colormap.
        hue_norm = (0.0, 1.0) if scale_to_max else (None, None)
        plot_grid.map_dataframe(
            sns.scatterplot,
            x=marker1,
            y=marker2,
            hue="density",
            hue_norm=hue_norm,
            palette=colormap,
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
        style_facet_strips(plot_grid)
    else:
        plot_grid = _plot_joint_distribution(
            data,
            x=marker1,
            y=marker2,
            show_marginal=show_marginal,
            hue_norm=_density_hue_norm(data["density"], scale_to_max),
        )
        if gate is not None:
            _add_gate_box(
                data, marker1=marker1, marker2=marker2, ax=plot_grid.ax_joint, gate=gate
            )
        plot_grid.ax_joint.axhline(0, color="black", linewidth=1, linestyle="--")
        plot_grid.ax_joint.axvline(0, color="black", linewidth=1, linestyle="--")
    return plt.gcf(), plt.gca()

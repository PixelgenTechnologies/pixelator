"""Proximity heatmap plots.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import Callable, Literal, Mapping, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage

from pixelator.common.plot import pixelgen_divergent_colormap

__all__ = ["proximity_heatmap"]


def _validate_columns(data, required):
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"data is missing required column(s): {sorted(missing)}.")


def _validate_unique_pairs(data, marker1_col, marker2_col, symmetrise):
    ordered_pairs = list(zip(data[marker1_col], data[marker2_col]))
    if len(ordered_pairs) != len(set(ordered_pairs)):
        raise ValueError(
            f"data has duplicate rows for the same ({marker1_col}, {marker2_col}) "
            "pair. Each marker pair must be represented by a single row -- "
            "subset or summarize your data first, e.g. with "
            "`pixelator.pna.analysis.summarize_proximity_scores`."
        )
    if symmetrise:
        unordered_pairs = pd.Series(tuple(sorted(p)) for p in ordered_pairs)
        if unordered_pairs.duplicated().any():
            raise ValueError(
                "data contains both (marker_1, marker_2) and (marker_2, marker_1) "
                "for at least one pair, which is ambiguous when `symmetrise=True`. "
                "Provide only one direction per pair, or set `symmetrise=False`."
            )


def _symmetrise_data(data, marker1_col, marker2_col):
    mirrored = data.rename(columns={marker1_col: marker2_col, marker2_col: marker1_col})
    return pd.concat([data, mirrored], ignore_index=True).drop_duplicates(
        subset=[marker1_col, marker2_col]
    )


def _wide_matrix(data, marker1_col, marker2_col, value_col):
    wide = data.pivot(index=marker1_col, columns=marker2_col, values=value_col)
    markers = sorted(set(wide.index) | set(wide.columns))
    return wide.reindex(index=markers, columns=markers)


def _can_cluster(n_observations: int) -> bool:
    """Hierarchical clustering needs at least three observations.

    `scipy.cluster.hierarchy.linkage` cannot build a tree from a single
    observation, and both plot kinds skip clustering for 1- and 2-marker
    matrices so the ordering stays consistent.
    """
    return n_observations >= 3


def _cluster_order(wide: pd.DataFrame, metric: str, method: str) -> list:
    if not _can_cluster(wide.shape[0]):
        return list(wide.index)
    link = linkage(wide.fillna(0.0).to_numpy(), method=method, metric=metric)
    return [wide.index[i] for i in leaves_list(link)]


def _resolve_highlight_colors(
    highlight_pairs: pd.DataFrame,
    highlight_colors: Union[str, Mapping[object, str]],
    highlight_color_col: Union[str, None],
) -> list:
    n = len(highlight_pairs)
    if highlight_color_col is None:
        if not isinstance(highlight_colors, str):
            raise TypeError(
                "'highlight_colors' must be a single color string when "
                "'highlight_color_col' is not given."
            )
        return [highlight_colors] * n

    if highlight_color_col not in highlight_pairs.columns:
        raise ValueError(
            f"'highlight_color_col' {highlight_color_col!r} is not a column of "
            "'highlight_pairs'."
        )
    if not isinstance(highlight_colors, Mapping):
        raise TypeError(
            "'highlight_colors' must be a mapping from each value of "
            "'highlight_color_col' to a color when 'highlight_color_col' is given."
        )
    values = highlight_pairs[highlight_color_col]
    missing = sorted(set(values) - set(highlight_colors))
    if missing:
        raise ValueError(f"'highlight_colors' has no color for value(s): {missing}.")
    return [highlight_colors[v] for v in values]


def _highlight_positions(
    highlight_pairs: pd.DataFrame,
    marker1_col: str,
    marker2_col: str,
    colors: list,
    symmetrise: bool,
) -> list[Tuple[object, object, str]]:
    positions = list(
        zip(highlight_pairs[marker1_col], highlight_pairs[marker2_col], colors)
    )
    if symmetrise:
        positions += [(m2, m1, c) for m1, m2, c in positions]
    return positions


def _draw_highlights_tiles(ax, data2d, positions, shrink, linewidth):
    rows = list(data2d.index)
    cols = list(data2d.columns)
    for marker1, marker2, color in positions:
        if marker1 not in rows or marker2 not in cols:
            continue
        y = rows.index(marker1)
        x = cols.index(marker2)
        ax.add_patch(
            Rectangle(
                (x + shrink / 2, y + shrink / 2),
                1 - shrink,
                1 - shrink,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )


def _draw_highlights_dots(ax, row_order, col_order, positions, shrink, linewidth):
    for marker1, marker2, color in positions:
        if marker1 not in row_order or marker2 not in col_order:
            continue
        y = row_order.index(marker1)
        x = col_order.index(marker2)
        ax.add_patch(
            Rectangle(
                (x - (1 - shrink) / 2, y - (1 - shrink) / 2),
                1 - shrink,
                1 - shrink,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )


def proximity_heatmap(
    data: pd.DataFrame,
    marker1_col: str = "marker_1",
    marker2_col: str = "marker_2",
    value_col: str = "estimate",
    size_col: Union[str, None] = "p_adj",
    size_col_transform: Union[Callable[[pd.Series], pd.Series], None] = None,
    size_range: Tuple[float, float] = (20.0, 300.0),
    cmap: Union[str, Colormap, None] = None,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    clustering_metric: str = "euclidean",
    clustering_method: str = "complete",
    kind: Literal["tiles", "dots"] = "tiles",
    return_plot_data: bool = False,
    symmetrise: bool = True,
    legend_range: Union[Tuple[float, float], None] = None,
    legend_title: Union[str, None] = None,
    highlight_pairs: Union[pd.DataFrame, None] = None,
    highlight_colors: Union[str, Mapping[object, str]] = "black",
    highlight_color_col: Union[str, None] = None,
    highlight_linewidth: float = 1.2,
    highlight_shrink: float = 0.1,
    figsize: Union[Tuple[float, float], None] = None,
) -> Union[Tuple[Figure, Axes], pd.DataFrame]:
    """Plot a heatmap of summary proximity scores between marker pairs.

    This function takes a long-format table with one row per marker pair
    (e.g. the output of `pixelator.pna.analysis.summarize_proximity_scores` or
    `pixelator.pna.analysis.calculate_differential_proximity`) and plots it
    either as a clustered heatmap of tiles, or as a dot plot where dot size
    encodes a second (e.g. significance) column.

    Args:
        data: A long-format DataFrame with one row per marker pair, containing
            at least ``marker1_col``, ``marker2_col`` and ``value_col``,
            and also ``size_col`` if ``kind="dots"`` and ``size_col`` is not
            ``None``.
        marker1_col: Column with the first marker of each pair. Defaults to
            ``"marker_1"``.
        marker2_col: Column with the second marker of each pair. Defaults to
            ``"marker_2"``.
        value_col: Numeric column to map to tile/dot color. Defaults to
            ``"estimate"``.
        size_col: Numeric column to map to dot size, only used when
            ``kind="dots"``. Set to ``None`` to draw all dots at the same
            size. Defaults to ``"p_adj"``.
        size_col_transform: Optional function applied to ``data[size_col]``
            before mapping it to dot size, e.g. ``lambda p: -np.log10(p)`` to
            emphasize small p-values with larger dots. Defaults to ``None``.
        size_range: The ``(min, max)`` marker area (in points²) that
            ``size_col`` is scaled to. Defaults to ``(20.0, 300.0)``.
        cmap: The colormap used for ``value_col``. Defaults to a Pixelgen
            branded divergent blues-to-reds colormap, centered at 0.
        cluster_rows: Whether to order rows (``marker1_col`` values) by
            hierarchical clustering. Defaults to ``True``.
        cluster_cols: Whether to order columns (``marker2_col`` values) by
            hierarchical clustering. Defaults to ``True``.
        clustering_metric: The distance metric used for clustering, passed to
            `scipy.cluster.hierarchy.linkage`. Defaults to ``"euclidean"``.
        clustering_method: The linkage method used for clustering, passed to
            `scipy.cluster.hierarchy.linkage`. Defaults to ``"complete"``.
        kind: ``"tiles"`` draws a clustered heatmap (via `seaborn.clustermap`);
            ``"dots"`` draws a dot plot with color mapped to ``value_col`` and
            size mapped to ``size_col``. Defaults to ``"tiles"``.
        return_plot_data: If ``True``, return the prepared data instead of
            plotting it: the wide pivoted matrix for ``kind="tiles"``, or the
            long-format DataFrame (with ``marker1_col``/``marker2_col`` cast to
            ordered categoricals reflecting the plot order) for
            ``kind="dots"``. Defaults to ``False``.
        symmetrise: If ``True``, mirror each row (swapping ``marker1_col`` and
            ``marker2_col``) so that only one direction of each pair needs to
            be present in ``data``. Defaults to ``True``.
        legend_range: The ``(min, max)`` range of the color scale. Defaults to
            ``None``, which uses a range symmetric around 0, spanning the
            largest absolute value of ``value_col``.
        legend_title: The title of the color legend/colorbar. Defaults to
            ``None``, which uses ``value_col``.
        highlight_pairs: An optional DataFrame of marker pairs to outline with
            a colored border, with columns ``marker1_col`` and ``marker2_col``
            (and, if ``highlight_color_col`` is given, that column too).
            Pairs not present in ``data`` are silently skipped. Defaults to
            ``None``.
        highlight_colors: The outline color for highlighted pairs. Either a
            single color string (used for every row of ``highlight_pairs``),
            or, if ``highlight_color_col`` is given, a mapping from each value
            of that column to a color. Defaults to ``"black"``.
        highlight_color_col: An optional column of ``highlight_pairs`` used to
            look up a per-pair outline color from ``highlight_colors``.
            Defaults to ``None``.
        highlight_linewidth: The line width of the highlight outlines.
            Defaults to ``1.2``.
        highlight_shrink: How much to shrink the highlight outline relative to
            the full cell/grid spacing, as a fraction in ``[0, 1)``. Defaults
            to ``0.1``.
        figsize: The figure size, in inches. Defaults to ``None``, which uses
            a size proportional to the number of markers. When set for
            ``kind="dots"``, the auto-computed layout is scaled to fill the
            given size.

    Returns:
        A tuple of the created figure and its main axes (the heatmap axes for
        ``kind="tiles"``, the dot plot axes for ``kind="dots"``), unless
        ``return_plot_data`` is ``True``, in which case the prepared plot data
        is returned instead.

    Raises:
        ValueError: If a required column is missing, if ``data`` has more
            than one row for some marker pair, if ``kind`` is not ``"tiles"``
            or ``"dots"``, or if ``highlight_shrink`` is not in ``[0, 1)``.
        TypeError: If ``highlight_colors``/``highlight_color_col`` are
            inconsistent (see above).
    """
    if kind not in ("tiles", "dots"):
        raise ValueError(f"'kind' must be 'tiles' or 'dots', got {kind!r}.")
    if not (0.0 <= highlight_shrink < 1.0):
        raise ValueError(
            f"'highlight_shrink' must be in [0, 1), got {highlight_shrink}."
        )

    required_columns = [marker1_col, marker2_col, value_col]
    if kind == "dots" and size_col is not None:
        required_columns.append(size_col)
    _validate_columns(data, required_columns)
    if highlight_pairs is not None:
        _validate_columns(highlight_pairs, [marker1_col, marker2_col])

    _validate_unique_pairs(data, marker1_col, marker2_col, symmetrise)

    data = data[required_columns].copy()
    if symmetrise:
        data = _symmetrise_data(data, marker1_col, marker2_col)

    # Size mapping is only used for dots; skip the transform for tiles so a
    # default ``size_col`` (or an explicit transform) cannot look up a column
    # that was dropped from ``required_columns``. Keep the original column name
    # for the size-legend title so the internal ``{size_col}_transformed``
    # column is not shown to the user.
    size_legend_title = size_col
    if kind == "dots" and size_col is not None and size_col_transform is not None:
        transformed_col = f"{size_col}_transformed"
        data[transformed_col] = size_col_transform(data[size_col])
        size_col = transformed_col

    if legend_range is None:
        max_abs = float(np.nanmax(np.abs(data[value_col])))
        legend_range = (-max_abs, max_abs)
    if legend_title is None:
        legend_title = value_col
    if cmap is None:
        cmap = pixelgen_divergent_colormap()

    highlight_positions = None
    if highlight_pairs is not None:
        highlight_pair_colors = _resolve_highlight_colors(
            highlight_pairs, highlight_colors, highlight_color_col
        )
        highlight_positions = _highlight_positions(
            highlight_pairs, marker1_col, marker2_col, highlight_pair_colors, symmetrise
        )

    if kind == "tiles":
        return _plot_tiles(
            data,
            marker1_col,
            marker2_col,
            value_col,
            cmap,
            cluster_rows,
            cluster_cols,
            clustering_metric,
            clustering_method,
            return_plot_data,
            legend_range,
            legend_title,
            highlight_positions,
            highlight_linewidth,
            highlight_shrink,
            figsize,
        )
    return _plot_dots(
        data,
        marker1_col,
        marker2_col,
        value_col,
        size_col,
        size_legend_title,
        size_range,
        cmap,
        cluster_rows,
        cluster_cols,
        clustering_metric,
        clustering_method,
        return_plot_data,
        legend_range,
        legend_title,
        highlight_positions,
        highlight_linewidth,
        highlight_shrink,
        figsize,
    )


def _plot_tiles(
    data,
    marker1_col,
    marker2_col,
    value_col,
    cmap,
    cluster_rows,
    cluster_cols,
    clustering_metric,
    clustering_method,
    return_plot_data,
    legend_range,
    legend_title,
    highlight_positions,
    highlight_linewidth,
    highlight_shrink,
    figsize,
):
    wide = _wide_matrix(data, marker1_col, marker2_col, value_col)
    if return_plot_data:
        return wide

    grid = sns.clustermap(
        wide.fillna(0.0),
        mask=wide.isna(),
        row_cluster=cluster_rows and _can_cluster(wide.shape[0]),
        col_cluster=cluster_cols and _can_cluster(wide.shape[1]),
        metric=clustering_metric,
        method=clustering_method,
        cmap=cmap,
        vmin=legend_range[0],
        vmax=legend_range[1],
        cbar_kws={"label": legend_title},
        figsize=figsize,
    )
    if highlight_positions:
        _draw_highlights_tiles(
            grid.ax_heatmap,
            grid.data2d,
            highlight_positions,
            highlight_shrink,
            highlight_linewidth,
        )
    return grid.figure, grid.ax_heatmap


def _text_size_inches(
    text: str, fontsize: Union[float, None] = None, rotation: float = 0
) -> Tuple[float, float]:
    """Measure the rendered (width, height) of a text string, in inches.

    Used to size the dot plot's margins/legend from the *actual* text that
    will be drawn (labels, ticks, titles), rather than a fixed or
    character-count-based guess, since neither font metrics nor rotated
    bounding boxes scale simply with string length.
    """
    scratch_fig = plt.figure()
    try:
        renderer = scratch_fig.canvas.get_renderer()  # type: ignore[attr-defined]
        artist = scratch_fig.text(0, 0, text, fontsize=fontsize, rotation=rotation)
        bbox = artist.get_window_extent(renderer)
        return bbox.width / scratch_fig.dpi, bbox.height / scratch_fig.dpi
    finally:
        plt.close(scratch_fig)


def _plot_dots(
    data,
    marker1_col,
    marker2_col,
    value_col,
    size_col,
    size_legend_title,
    size_range,
    cmap,
    cluster_rows,
    cluster_cols,
    clustering_metric,
    clustering_method,
    return_plot_data,
    legend_range,
    legend_title,
    highlight_positions,
    highlight_linewidth,
    highlight_shrink,
    figsize,
):
    wide = _wide_matrix(data, marker1_col, marker2_col, value_col)
    row_order = (
        _cluster_order(wide, clustering_metric, clustering_method)
        if cluster_rows
        else list(wide.index)
    )
    col_order = (
        _cluster_order(wide.T, clustering_metric, clustering_method)
        if cluster_cols
        else list(wide.columns)
    )

    data = data.copy()
    data[marker1_col] = pd.Categorical(
        data[marker1_col], categories=row_order, ordered=True
    )
    data[marker2_col] = pd.Categorical(
        data[marker2_col], categories=col_order, ordered=True
    )

    if return_plot_data:
        return data

    if size_col is not None:
        size_values = data[size_col].to_numpy(dtype=float)
        lo, hi = np.nanmin(size_values), np.nanmax(size_values)
        if lo == hi:
            # Constant size_col: one midpoint marker, one legend entry.
            # np.interp with identical xp endpoints is undefined and would
            # produce invalid sizes plus duplicate labels.
            mid_size = float(np.mean(size_range))
            sizes = np.full(len(size_values), mid_size)
            legend_sizes = [mid_size]
            legend_labels = [f"{lo:.2g}"]
        else:
            sizes = np.interp(size_values, (lo, hi), size_range)
            legend_values = np.linspace(lo, hi, 4)
            legend_sizes = np.interp(legend_values, (lo, hi), size_range)
            legend_labels = [f"{value:.2g}" for value in legend_values]
    else:
        sizes = np.full(len(data), size_range[-1])
        legend_sizes = legend_labels = []

    x = data[marker2_col].cat.codes.to_numpy()
    y = data[marker1_col].cat.codes.to_numpy()

    # Axes and the colorbar/size-legend are placed at explicit, absolute-inch
    # positions (rather than via `plt.subplots` + `fig.tight_layout()`)
    # sized from measurements of the actual label/tick/legend text (see
    # `_text_size_inches`), because `tight_layout` only reserves space for
    # artists attached to an Axes -- it does not know about the colorbar or
    # size-legend added below, and would let them overlap or clip off the
    # edge of the figure, regardless of how the caller later saves it.
    n_cols = max(len(col_order), 1)
    n_rows = max(len(row_order), 1)
    row_label_w = max((_text_size_inches(str(m))[0] for m in row_order), default=0.3)
    col_label_sizes = [_text_size_inches(str(m), rotation=45) for m in col_order]
    col_label_w = max((w for w, _ in col_label_sizes), default=0.3)
    col_label_h = max((h for _, h in col_label_sizes), default=0.3)

    plot_width = max(0.45 * n_cols, col_label_w + 0.1)
    plot_height = 0.45 * n_rows
    left_margin = row_label_w + 0.4
    top_margin = col_label_h + 0.55
    bottom_margin = 0.3

    cbar_gap = 0.3
    cbar_width = 0.25
    tick_label_w = max(_text_size_inches(f"{v:.2g}")[0] for v in (*legend_range, 0.0))
    cbar_label_w, _ = _text_size_inches(legend_title, rotation=90)
    cbar_to_legend_gap = 0.25

    if size_col is not None:
        legend_title_w, _ = _text_size_inches(size_legend_title)
        legend_entry_w = max(_text_size_inches(lbl)[0] for lbl in legend_labels)
        marker_diameter_in = 2 * np.sqrt(max(size_range) / np.pi) / 72
        legend_width = (
            max(legend_title_w, marker_diameter_in + 0.3 + legend_entry_w) + 0.15
        )
    else:
        legend_width = 0.0
        cbar_to_legend_gap = 0.0

    right_margin = (
        cbar_gap
        + cbar_width
        + tick_label_w
        + 0.1
        + cbar_label_w
        + cbar_to_legend_gap
        + legend_width
        + 0.15
    )

    natural_width = left_margin + plot_width + right_margin
    natural_height = top_margin + plot_height + bottom_margin
    fig_width, fig_height = figsize or (natural_width, natural_height)

    # When the caller supplies figsize, scale the auto layout so axes,
    # colorbar, and size-legend keep their relative positions and fill the
    # figure instead of staying at absolute-inch coordinates (which clip on
    # smaller figures and leave unused space on larger ones).
    if figsize is not None:
        sx = fig_width / natural_width
        sy = fig_height / natural_height
        left_margin *= sx
        plot_width *= sx
        cbar_gap *= sx
        cbar_width *= sx
        tick_label_w *= sx
        cbar_label_w *= sx
        cbar_to_legend_gap *= sx
        bottom_margin *= sy
        plot_height *= sy

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(
        (
            left_margin / fig_width,
            bottom_margin / fig_height,
            plot_width / fig_width,
            plot_height / fig_height,
        )
    )
    scatter = ax.scatter(
        x,
        y,
        c=data[value_col],
        cmap=cmap,
        norm=Normalize(vmin=legend_range[0], vmax=legend_range[1]),
        s=sizes,
        edgecolors="black",
        linewidths=0.5,
    )

    ax.set_xlim(-0.5, len(col_order) - 0.5)
    ax.set_ylim(len(row_order) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="left")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xlabel(marker2_col)
    ax.set_ylabel(marker1_col)

    cax = fig.add_axes(
        (
            (left_margin + plot_width + cbar_gap) / fig_width,
            bottom_margin / fig_height,
            cbar_width / fig_width,
            plot_height / fig_height,
        )
    )
    fig.colorbar(scatter, cax=cax, label=legend_title)

    if size_col is not None:
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=np.sqrt(size),
                label=label,
            )
            for label, size in zip(legend_labels, legend_sizes)
        ]
        legend_x = (
            left_margin
            + plot_width
            + cbar_gap
            + cbar_width
            + tick_label_w
            + 0.1
            + cbar_label_w
            + cbar_to_legend_gap
        ) / fig_width
        fig.legend(
            handles=handles,
            title=size_legend_title,
            loc="center left",
            bbox_to_anchor=(legend_x, 0.5),
            frameon=False,
            numpoints=1,
        )

    if highlight_positions:
        _draw_highlights_dots(
            ax,
            row_order,
            col_order,
            highlight_positions,
            highlight_shrink,
            highlight_linewidth,
        )

    return fig, ax

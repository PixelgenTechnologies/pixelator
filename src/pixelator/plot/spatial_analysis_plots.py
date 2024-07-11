"""Functions for creating spatial analysis plots for MPX data.

Copyright © 2024 Pixelgen Technologies AB.
"""

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pixelator.analysis.colocalization import get_differential_colocalization


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

"""Functions for creating spatial analysis plots for MPX data.

Copyright © 2024 Pixelgen Technologies AB.
"""

import logging
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pixelator.mpx.analysis.colocalization import get_differential_colocalization
from pixelator.mpx.analysis.polarization import get_differential_polarity

logger = logging.getLogger(__name__)


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
    value_column: str = "pearson_z",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a colocalization heatmap based on the provided colocalization data.

    The colocalization_data DataFrame should contain the columns "marker_1",
    "marker_2", "pearson", "pearson_z".
    Example usage: plot_colocalization_heatmap(pxl.colocalization).

    :param colocalization_data: The colocalization data to plot. The
    colocalization data frame that can be found in a pixel variable "pxl"
    through pxl.colocalization. The data frame should contain the columns
    "marker_1", "marker_2", "pearson", "pearson_z", and "component".
    :param markers: The markers to include in the heatmap. Defaults to None.
    :param cmap: The colormap to use for the heatmap. Defaults to "vlag".
    :param value_column: What colocalization metric to use. Defaults to "pearson_z".

    :return: The figure and axes objects of the plot.
    :rtype: Tuple[plt.Figure, plt.Axes]

    """
    colocalization_data = _make_colocalization_symmetric(
        colocalization_data, value_column
    )

    colocalization_data_pivot = _pivot_colocalization_data(
        colocalization_data, value_column, markers=markers
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


def plot_colocalization_diff_heatmap(
    colocalization_data: pd.DataFrame,
    reference: str,
    targets: str | list[str] | None = None,
    contrast_column: str = "sample",
    top_marker_log_p: float | None = None,
    min_log_p: float = 3.0,
    cmap: str = "vlag",
    value_column: str = "pearson_z",
) -> Tuple[dict, dict]:
    """Plot the differential colocalization between reference and target components.

    Example usage: plot_colocalization_diff_heatmap(pxl.colocalization,
    reference:"control", contrast_column="sample").

    :param colocalization_data: The colocalization data frame that can be found
    in a pixel variable "pxl" through pxl.colocalization. The data frame should
    contain the columns "marker_1", "marker_2", "pearson", "pearson_z", and the
    contrast_column.
    :param reference: The label for reference components in the contrast_column.
    :param targets: label or list of labels for target components in the
    contrast_column. When not specified, all labels in the contrast_column
    except the reference label are used as targets.
    :param contrast_column: The column to use for the contrast. Defaults to "sample".
    :param top_marker_log_p: When set to a value, only markers that differentially
    colocalize with at least one other marker with a log10 p-score higher than
    top_marker_log_p are inclueded in the plot.
    :param min_log_p: The minimum log10 p-value. Pairs with lower log10 p-value
    are assigned 0.
    :param cmap: The colormap to use for the heatmap. Defaults to "vlag".
    :param value_column: What colocalization metric to use. Defaults to "pearson_z".

    :return: Two dicts mapping target names respectively to figures and axes of
    the generated plots.
    """
    if isinstance(targets, str):
        targets = [targets]
    elif targets is None:
        targets = colocalization_data[contrast_column].unique()
        targets = list(set(targets) - {reference})

    differential_colocalization = get_differential_colocalization(
        colocalization_data,
        reference=reference,
        targets=targets,
        contrast_column=contrast_column,
        value_column=value_column,
    )
    figs = {}
    axes = {}
    for target in sorted(targets):
        target_diff = differential_colocalization.loc[
            differential_colocalization["target"] == target, :
        ]

        target_diff_p_adj = -np.log10(
            target_diff.set_index(["marker_1", "marker_2"])["p_adj"].unstack()
        )
        target_diff_med_diff = target_diff.set_index(["marker_1", "marker_2"])[
            "median_difference"
        ].unstack(fill_value=0)
        target_diff_med_diff[target_diff_p_adj < min_log_p] = 0
        if top_marker_log_p is not None:
            target_diff_med_diff = target_diff_med_diff.loc[
                target_diff_p_adj.max(axis=1) > top_marker_log_p,
                target_diff_p_adj.max(axis=0) > top_marker_log_p,
            ]
            if target_diff_med_diff.shape[0] == 0:
                logger.warning(
                    "No marker pairs with log10 p-value higher than "
                    f"{top_marker_log_p} found for target {target}."
                )
                continue
        target_diff_med_diff = target_diff_med_diff.add(
            target_diff_med_diff.T, fill_value=0
        ).fillna(0)
        max_value = np.max(np.abs(target_diff_med_diff.to_numpy().flatten()))
        g = sns.clustermap(
            target_diff_med_diff,
            yticklabels=True,
            xticklabels=True,
            method="complete",
            linewidths=0.1,
            vmin=-max_value,
            vmax=max_value,
            cmap=cmap,
        )
        g.figure.suptitle(
            f"Differential colocalization between {target} and {reference}"
        )
        plt.tight_layout()
        figs[target] = plt.gcf()
        axes[target] = plt.gca()

    return figs, axes


def _add_top_marker_labels(
    plot_data,
    ax,
    n_top_pairs: int = 5,
    min_log_p: float = 5.0,
):
    plot_data = plot_data.sort_values("median_difference")
    plot_data = plot_data.loc[-np.log10(plot_data["p_adj"]) > min_log_p, :]

    # Labels for marker pair withs highest negative differential colocalization scores
    for _, row in plot_data.head(n_top_pairs).iterrows():
        if "marker_1" not in row.index:
            if "marker" in row.index:
                name = row["marker"]
            else:
                name = row.name
        else:
            name = row["marker_1"] + "/" + row["marker_2"]

        x, y = row[["median_difference", "p_adj"]]
        y = -np.log10(y)
        if x > 0:
            continue
        ax.text(
            x,
            y,
            name,
            horizontalalignment="left",
            fontsize="xx-small",
        )

    # Labels for marker pair with highest positive differential colocalization scores
    for _, row in plot_data.tail(n_top_pairs).iterrows():
        if "marker_1" not in row.index:
            if "marker" in row.index:
                name = row["marker"]
            else:
                name = row.name
        else:
            name = row["marker_1"] + "/" + row["marker_2"]
        x, y = row[["median_difference", "p_adj"]]
        y = -np.log10(y)
        if x < 0:
            continue
        ax.text(
            x,
            y,
            name,
            horizontalalignment="left",
            fontsize="xx-small",
        )


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
    reference: str,
    targets: str | list[str] | None = None,
    contrast_column: str = "sample",
    cmap: str = "vlag",
    value_column="pearson_z",
    n_top_pairs: int = 5,
    min_log_p: float = 5.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate the volcano plot of differential colocalization between reference and target(s) components.

    Example usage: `plot_colocalization_diff_volcano(pxl.colocalization, target:"stimulated", reference:"control", contrast_column="sample")`.

    :param colocalization_data: The colocalization data frame that can be found in a pixel variable
                                "pxl" through pxl.colocalization. The data frame should contain the
                                columns "marker_1", "marker_2", "pearson", "pearson_z", and the contrast_column.
    :param target: The label for target components in the contrast_column.
    :param reference: The label for reference components in the contrast_column.
    :param contrast_column: The column to use for the contrast. Defaults to "sample".
    :param cmap: The colormap to use for the heatmap. Defaults to "vlag".
    :param value_column: What colocalization metric to use. Defaults to "pearson_z".
    :param n_top_pairs: Number of high value marker-pairs to label from positive and negative sides.
    :param min_log_p: marker-pairs only receive a label if -log10 of their p-value is higher than
                      this parameter.

    :return: The figure and axes objects of the plot.
    """
    if isinstance(targets, str):
        targets = [targets]
    elif targets is None:
        targets = colocalization_data[contrast_column].unique()
        targets = list(set(targets) - {reference})

    if len(targets) > 5:
        raise ValueError(
            "Only up to 5 targets can be visualized. "
            "Number of requested targets is {len(targets)}."
            "Requested targets are: {targets}."
        )

    differential_colocalization = get_differential_colocalization(
        colocalization_data,
        targets=targets,
        reference=reference,
        contrast_column=contrast_column,
        value_column=value_column,
    )

    fig, axes = plt.subplots(1, len(targets))
    for i, target in enumerate(sorted(targets)):
        ax = axes[i] if len(targets) > 1 else axes
        target_differential_colocalization = differential_colocalization.loc[
            differential_colocalization["target"] == target, :
        ]
        target_differential_colocalization = _add_target_mean_colocalizations(
            target_differential_colocalization,
            target_coloc=colocalization_data.loc[
                colocalization_data[contrast_column] == target, :
            ],
            value_col=value_column,
        )
        p = ax.scatter(
            x=target_differential_colocalization["median_difference"],
            y=-np.log10(target_differential_colocalization["p_adj"]),
            c=target_differential_colocalization[value_column],
            s=20,
            marker="o",
            cmap=cmap,
        )

        ax.set(
            xlabel="Median difference",
            ylabel=r"$-\log_{10}$(adj. p-value)",
            title=f"Differential colocalization\nbetween {target}\nand {reference}",
        )
        ax.title.set_y(1.05)
        fig.colorbar(p, label="Mean target colocalization score", cmap=cmap)
        _add_top_marker_labels(
            target_differential_colocalization,
            n_top_pairs=n_top_pairs,
            min_log_p=min_log_p,
            ax=ax,
        )
    fig.subplots_adjust(top=0.8)
    fig.set_size_inches(6 * len(targets), 5)
    return fig, axes


def plot_polarity_diff_volcano(
    polarity_data: pd.DataFrame,
    reference: str,
    targets: str | list[str] | None = None,
    contrast_column: str = "sample",
    cmap: str = "vlag",
    value_column="morans_z",
    n_top_pairs: int = 5,
    min_log_p: float = 5.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate the volcano plot of differential polarity between reference and target(s) components.

    Example usage: `plot_polarity_diff_volcano(
                                                pxl.polariazation,target:"stimulated",
                                                reference:"control",
                                                contrast_column="sample"
                                                )`.

    :param polarity_data: The polarity data frame that can be found in a pixel variable
                            "pxl" through pxl.polarization. The data frame should contain the
                            columns "marker", the value_column (e.g. morans_z), and the contrast_column.
    :param target: The label for target components in the contrast_column.
    :param reference: The label for reference components in the contrast_column.
    :param contrast_column: The column to use for the contrast. Defaults to "sample".
    :param cmap: The colormap to use for the heatmap. Defaults to "vlag".
    :param value_column: What polarity metric to use. Defaults to "morans_z".
    :param n_top_pairs: Number of high value marker-pairs to label from positive and negative sides.
    :param min_log_p: marker-pairs only receive a label if -log10 of their p-value is higher than
                      this parameter.

    :return: The figure and axes objects of the plot.
    """
    if isinstance(targets, str):
        targets = [targets]
    elif targets is None:
        targets = polarity_data[contrast_column].unique()
        targets = list(set(targets) - {reference})

    if len(targets) > 5:
        raise ValueError(
            "Only up to 5 targets can be visualized. "
            "Number of requested targets is {len(targets)}."
            "Requested targets are: {targets}."
        )

    differential_polarity = get_differential_polarity(
        polarity_data,
        targets=targets,
        reference=reference,
        contrast_column=contrast_column,
        value_column=value_column,
    )

    fig, axes = plt.subplots(1, len(targets))
    for i, target in enumerate(sorted(targets)):
        ax = axes[i] if len(targets) > 1 else axes
        target_differential_polarity = differential_polarity.loc[
            differential_polarity["target"] == target, :
        ].set_index("marker")
        target_differential_polarity["target_mean"] = (
            polarity_data[polarity_data[contrast_column] == target]
            .groupby("marker")[value_column]
            .mean()
        )

        p = ax.scatter(
            x=target_differential_polarity["median_difference"],
            y=-np.log10(target_differential_polarity["p_adj"]),
            c=target_differential_polarity["target_mean"],
            s=20,
            marker="o",
            cmap=cmap,
        )

        ax.set(
            xlabel="Median difference",
            ylabel=r"$-\log_{10}$(adj. p-value)",
            title=f"Differential polarity\nbetween {target}\nand {reference}",
        )
        ax.title.set_y(1.05)
        fig.colorbar(p, label="Mean target polarity score", cmap=cmap)
        _add_top_marker_labels(
            target_differential_polarity,
            n_top_pairs=n_top_pairs,
            min_log_p=min_log_p,
            ax=ax,
        )
    fig.subplots_adjust(top=0.8)
    fig.set_size_inches(6 * len(targets), 5)
    return fig, axes

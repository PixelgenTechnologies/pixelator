"""Module for computing localization proximity statistics.

Copyright © 2024 Pixelgen Technologies AB
"""

from typing import Callable, List, Literal

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from pixelator.common.duckdb_utils import connect_duckdb
from pixelator.pna.analysis.analytical_proximity_query_helper import (
    DEFAULT_MIN_STD,
    add_join_count_p,
    jcs_with_analytical_stats,
)
from pixelator.pna.utils.utils import normalize_input_to_list

_PROXIMITY_METRICS = ("log2_ratio", "join_count_z")
_SUMMARY_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "median": np.median,
}
# R FilterProximityScores uses p1/p2 and count_1/count_2. Python Proximity.to_df()
# with add_marker_counts=True emits marker_*_freq and marker_*_count.
_PCT_COLUMN_PAIRS = (("p1", "p2"), ("marker_1_freq", "marker_2_freq"))
_COUNT_COLUMN_PAIRS = (("count_1", "count_2"), ("marker_1_count", "marker_2_count"))


def _get_markers_above_min_count(edgelist: pl.DataFrame, min_count: int = 0) -> set:
    """Filter out markers with low counts from the edgelist.

    Args:
        edgelist: A DataFrame representing the edgelist.
        min_count: Minimum count threshold for markers. Defaults to 0.

    Returns:
        pl.DataFrame: A filtered DataFrame with low-count markers removed.
    """
    umi1_counts = (
        edgelist.select(["umi1", "marker_1"])
        .unique()
        .group_by("marker_1")
        .len()
        .rename({"marker_1": "marker", "len": "umi1_count"})
    )
    umi2_counts = (
        edgelist.select(["umi2", "marker_2"])
        .unique()
        .group_by("marker_2")
        .len()
        .rename({"marker_2": "marker", "len": "umi2_count"})
    )
    umi_counts = (
        umi1_counts.join(umi2_counts, on="marker", how="full", coalesce=True)
        .fill_null(0)
        .with_columns(total_count=pl.col("umi1_count") + pl.col("umi2_count"))
    )

    passing_markers = umi_counts.filter(pl.col("total_count") >= min_count)

    return set(passing_markers["marker"])


def jcs_with_analytical_stats_from_edgelist(
    edgelist: pl.DataFrame,
    min_marker_count: int = 0,
    min_std: float = DEFAULT_MIN_STD,
) -> pd.DataFrame:
    """Compute join counts augmented with analytically calculated expected statistics.

    The expected join counts and their standard deviations are calculated
    exactly under the null model where the markers of the umi1 and umi2 nodes are
    independently permuted within a component (see `jcs_with_analytical_stats`).

    Args:
        edgelist: A DataFrame representing the edgelist of a single component.
        min_marker_count: Minimum marker count to consider. Defaults to 0.
        min_std: Minimum standard deviation to use when calculating the z-scores.
            Defaults to 1.0.

    Returns:
        A DataFrame with one row per marker pair containing the columns `marker_1`,
        `marker_2`, `join_count`, `join_count_expected_mean`,
        `join_count_expected_sd`, `join_count_z` and `join_count_p`.
    """
    passing_markers = _get_markers_above_min_count(edgelist, min_marker_count)

    # The query works on sample and component level, these are constant here.
    edgelist = edgelist.select(
        "umi1",
        "umi2",
        "marker_1",
        "marker_2",
        sample=pl.lit("sample"),
        component=pl.lit("component"),
    )
    sql, params = jcs_with_analytical_stats(min_std=min_std)
    with connect_duckdb() as connection:
        connection.register("edgelist", edgelist.to_arrow())
        results = connection.execute(sql, parameters=params).pl()

    results = (
        add_join_count_p(results)
        .filter(
            pl.col("marker_1").is_in(passing_markers)
            & pl.col("marker_2").is_in(passing_markers)
        )
        .select(
            "marker_1",
            "marker_2",
            "join_count",
            "join_count_expected_mean",
            "join_count_expected_sd",
            "join_count_z",
            "join_count_p",
        )
        .sort("marker_1", "marker_2")
    )
    return results.to_pandas()


def summarize_proximity_scores(
    proximity_df: pd.DataFrame,
    proximity_metric: Literal["log2_ratio", "join_count_z"] = "log2_ratio",
    group_vars: str | List[str] | None = None,
    include_missing_obs: bool = True,
    summary_stat: Literal["mean", "median"] = "mean",
    detailed: bool = False,
) -> pd.DataFrame:
    """Summarize per-component proximity scores across components, for each marker pair.

    Collapses a long-format proximity score table (one row per component and
    marker pair) into one row per marker pair (optionally stratified by
    ``group_vars``, e.g. sample or cell type), reporting the number of
    components the pair was detected in and a summary statistic of
    ``proximity_metric`` across those components.

    A marker pair is only present for a component if it passed detection in
    that component's proximity analysis. When ``include_missing_obs`` is
    ``True`` (the default), components where a pair was not detected are
    treated as having a ``proximity_metric`` value of 0 and are included when
    computing the summary statistic, since an undetected pair corresponds to
    no deviation from the expected join count.

    Args:
        proximity_df: A long-format proximity score table, e.g. from
            `Proximity.to_df`, with one row per ``component`` and marker
            pair. Must contain the columns ``component``, ``marker_1``,
            ``marker_2``, ``proximity_metric``, and (if ``detailed`` is
            ``True``) ``join_count`` and ``join_count_expected_mean``.
        proximity_metric: The proximity score column to summarize, either
            ``"log2_ratio"`` or ``"join_count_z"``. Defaults to
            ``"log2_ratio"``.
        group_vars: Additional column(s) to stratify the summary by, e.g.
            ``"sample"`` or ``["sample", "cell_type"]``. Each component must
            map to a single, consistent value of every ``group_vars`` column.
            Defaults to ``None``, which summarizes across all components in
            ``proximity_df``.
        include_missing_obs: If ``True``, pad the values used to compute the
            summary statistic with a 0 for every component where the marker
            pair was not detected. Defaults to ``True``.
        summary_stat: The summary statistic to compute, either ``"mean"`` or
            ``"median"``. Defaults to ``"mean"``.
        detailed: If ``True``, also return the per-component values that went
            into the summary statistic (``{proximity_metric}_list``,
            ``join_count_list``, and ``join_count_expected_mean_list``), so
            that other statistics (e.g. standard deviation or quantiles) can
            be computed downstream. Defaults to ``False``.

    Returns:
        A DataFrame with one row per unique combination of ``group_vars``,
        ``marker_1`` and ``marker_2``, containing ``n_cells_detected`` (the
        number of components the pair was detected in), ``n_cells`` (the
        total number of components in that ``group_vars`` stratum),
        ``n_cells_missing``, ``pct_detected``, and
        ``{summary_stat}_{proximity_metric}`` (e.g. ``mean_log2_ratio``). If
        ``detailed`` is ``True``, the per-component value list columns
        described above are also included.

    Raises:
        ValueError: If ``proximity_metric`` or ``summary_stat`` is not one of
            the supported options, if a required column is missing from
            ``proximity_df``, or if ``proximity_df`` has more than one row for
            some combination of ``component``, ``marker_1``, ``marker_2`` and
            ``group_vars``.
    """
    if proximity_metric not in _PROXIMITY_METRICS:
        raise ValueError(
            f"proximity_metric must be one of {_PROXIMITY_METRICS}, "
            f"got {proximity_metric!r}."
        )
    if summary_stat not in _SUMMARY_STATS:
        raise ValueError(
            f"summary_stat must be one of {tuple(_SUMMARY_STATS)}, "
            f"got {summary_stat!r}."
        )

    group_vars = normalize_input_to_list(group_vars) or []
    detail_columns = ["join_count", "join_count_expected_mean"] if detailed else []

    required_columns = {
        "component",
        "marker_1",
        "marker_2",
        proximity_metric,
        *detail_columns,
        *group_vars,
    }
    missing_columns = required_columns - set(proximity_df.columns)
    if missing_columns:
        raise ValueError(
            f"proximity_df is missing required column(s): {sorted(missing_columns)}."
        )

    pair_group_vars = [*group_vars, "marker_1", "marker_2"]
    duplicate_key = ["component", *pair_group_vars]
    duplicates = proximity_df.duplicated(subset=duplicate_key, keep=False)
    if duplicates.any():
        example = proximity_df.loc[duplicates, duplicate_key].iloc[0].to_dict()
        raise ValueError(
            "proximity_df must have at most one row per component, marker "
            "pair and group_vars combination, found duplicate rows, e.g. "
            f"{example}."
        )

    if group_vars:
        n_cells = proximity_df.groupby(group_vars)["component"].nunique()
        n_cells = n_cells.rename("n_cells").reset_index()
    else:
        n_cells = proximity_df["component"].nunique()

    value_columns = [proximity_metric, *detail_columns]
    summary = proximity_df.groupby(pair_group_vars).agg(
        n_cells_detected=(proximity_metric, "size"),
        **{f"{col}_list": (col, list) for col in value_columns},
    )
    summary = summary.reset_index()

    if group_vars:
        summary = summary.merge(n_cells, on=group_vars, how="left")
    else:
        summary["n_cells"] = n_cells

    summary["n_cells_missing"] = summary["n_cells"] - summary["n_cells_detected"]
    summary["pct_detected"] = summary["n_cells_detected"] / summary["n_cells"]

    if include_missing_obs:
        for col in value_columns:
            list_col = f"{col}_list"
            summary[list_col] = [
                values + [0] * n_missing
                for values, n_missing in zip(
                    summary[list_col], summary["n_cells_missing"]
                )
            ]

    metric_list_col = f"{proximity_metric}_list"
    stat_func = _SUMMARY_STATS[summary_stat]
    summary_col = f"{summary_stat}_{proximity_metric}"
    summary[summary_col] = [stat_func(values) for values in summary[metric_list_col]]

    output_columns = [
        *group_vars,
        "marker_1",
        "marker_2",
        "n_cells_detected",
        "n_cells",
        "n_cells_missing",
        "pct_detected",
        summary_col,
    ]
    if detailed:
        output_columns += [f"{col}_list" for col in value_columns]

    return summary[output_columns]


def _resolve_column_pair(
    proximity_df: pd.DataFrame,
    column_pairs: tuple[tuple[str, str], ...],
    filter_name: str,
    missing_hint: str,
) -> tuple[str, str]:
    """Return the first column pair present in ``proximity_df``.

    Args:
        proximity_df: Proximity score table to inspect.
        column_pairs: Candidate (left, right) column name pairs, in preference
            order.
        filter_name: Name of the filter that requires the columns, used in the
            error message.
        missing_hint: Extra guidance included in the error message.

    Returns:
        The first pair of column names that both exist in ``proximity_df``.

    Raises:
        ValueError: If none of the candidate pairs are fully present.
    """
    for left, right in column_pairs:
        if left in proximity_df.columns and right in proximity_df.columns:
            return left, right

    expected = " or ".join(f"{left!r} and {right!r}" for left, right in column_pairs)
    raise ValueError(f"{filter_name} requires columns {expected}. {missing_hint}")


def _validate_optional_threshold(
    name: str,
    value: float | int | None,
    *,
    minimum: float,
    maximum: float | None = None,
) -> None:
    """Validate an optional numeric threshold.

    Args:
        name: Argument name, used in the error message.
        value: Threshold value, or ``None`` if the filter is unused.
        minimum: Inclusive lower bound when ``value`` is not ``None``.
        maximum: Inclusive upper bound when not ``None``.

    Raises:
        ValueError: If ``value`` is not a real number or is outside the
            allowed range.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be within {bounds}, got {value!r}.")


def filter_proximity_scores(
    proximity_df: pd.DataFrame,
    background_threshold_pct: float | None = None,
    background_threshold_count: float | None = None,
    min_cells_count: int | None = None,
) -> pd.DataFrame:
    """Remove marker pairs with too little signal from a proximity score table.

    Proximity scores are reported for every marker pair detected in a cell,
    including pairs where one or both markers are barely present. Such pairs
    may reflect background and can distort downstream summaries and
    heatmaps, so it is usually worth dropping them before further analysis.

    At least one threshold must be given. When several are given, they are
    applied in this order:

    1. ``background_threshold_pct`` keeps a pair only if both of its markers
       make up at least this fraction of the cell's UMI counts. The isotype
       control fraction for your panel is a good choice of cutoff.
    2. ``background_threshold_count`` keeps a pair only if both of its markers
       have at least this many UMI counts.
    3. ``min_cells_count`` keeps a marker pair only if it is detected in at
       least this many cells.

    Using ``background_threshold_pct`` on its own is usually enough. Unlike
    a raw count cutoff, a fraction does not depend on how many molecules a
    cell has, so the same threshold is meaningful for both small and large cells.

    The filters need the per-cell marker abundance columns that
    :meth:`~pixelator.pna.pixeldataset.proximity.Proximity.to_df` adds by
    default (``add_marker_counts=True``): ``marker_1_freq`` and
    ``marker_2_freq`` for the fraction filter, ``marker_1_count`` and
    ``marker_2_count`` for the count filter. Proximity tables exported from
    pixelatorR also work, where the same columns are named ``p1`` and ``p2``,
    and ``count_1`` and ``count_2``.

    Args:
        proximity_df: Proximity scores with one row per cell (``component``)
            and marker pair, for example from
            ``dataset.proximity().to_df()``.
        background_threshold_pct: Minimum fraction of a cell's UMI counts that
            both markers in a pair must reach, as a value between 0 and 1. For
            example, 0.001 in a cell with 20,000 molecules requires at least
            20 counts for both markers.
        background_threshold_count: Minimum number of UMI counts that both
            markers in a pair must reach. This is an absolute cutoff, so it
            removes more pairs in small cells than in large ones.
        min_cells_count: Minimum number of cells a marker pair must be
            detected in. Useful for dropping pairs seen in only a handful of
            cells.

    Returns:
        The rows of ``proximity_df`` that pass every threshold you set, with
        the columns unchanged.

    Raises:
        ValueError: If no threshold is set, if a threshold is outside its
            allowed range, or if the table lacks the columns a requested
            filter needs.

    Examples:
        Keep the marker pairs where both markers are above the isotype
        control fraction, and see how much of the table that leaves::

            from pixelator.pna.analysis import filter_proximity_scores
            from pixelator.pna.pixeldataset import read

            dataset = read("sample.pxl")
            proximity_table = dataset.proximity().to_df()
            filtered = filter_proximity_scores(
                proximity_table,
                background_threshold_pct=0.001,
            )
            pct_rows_kept = round(
                len(filtered) / len(proximity_table) * 100, 2
            )

    See Also:
        ``FilterProximityScores`` in pixelatorR, the equivalent function for
        R users.
    """
    _validate_optional_threshold(
        "background_threshold_pct",
        background_threshold_pct,
        minimum=0,
        maximum=1,
    )
    _validate_optional_threshold(
        "background_threshold_count",
        background_threshold_count,
        minimum=0,
    )
    _validate_optional_threshold(
        "min_cells_count",
        min_cells_count,
        minimum=0,
    )

    if (
        background_threshold_pct is None
        and background_threshold_count is None
        and min_cells_count is None
    ):
        raise ValueError(
            "At least one of background_threshold_pct, "
            "background_threshold_count, or min_cells_count must be set."
        )

    filtered = proximity_df

    if background_threshold_pct is not None:
        p1_col, p2_col = _resolve_column_pair(
            filtered,
            _PCT_COLUMN_PAIRS,
            "background_threshold_pct",
            "Read the proximity table with dataset.proximity().to_df() and "
            "add_marker_counts=True (the default) to include them.",
        )
        filtered = filtered[
            np.minimum(filtered[p1_col], filtered[p2_col]) >= background_threshold_pct
        ]

    if background_threshold_count is not None:
        c1_col, c2_col = _resolve_column_pair(
            filtered,
            _COUNT_COLUMN_PAIRS,
            "background_threshold_count",
            "Read the proximity table with dataset.proximity().to_df() and "
            "add_marker_counts=True (the default) to include them.",
        )
        filtered = filtered[
            np.minimum(filtered[c1_col], filtered[c2_col]) >= background_threshold_count
        ]

    if min_cells_count is not None:
        missing = {"marker_1", "marker_2"} - set(filtered.columns)
        if missing:
            raise ValueError(
                "min_cells_count requires columns 'marker_1' and "
                f"'marker_2'. Missing: {sorted(missing)}."
            )
        n_cells = filtered.groupby(["marker_1", "marker_2"], sort=False)[
            "marker_1"
        ].transform("size")
        filtered = filtered[n_cells >= min_cells_count]

    return filtered.copy()


def _filter_target_data(
    proximity_df, contrast_column, reference, target, metric, min_n_obs
):
    target_data = proximity_df[proximity_df[contrast_column].isin([reference, target])]

    if min_n_obs > 0:
        group_counts = (
            target_data.groupby(["marker_1", "marker_2", contrast_column])
            .size()
            .unstack(fill_value=0)
        )
        valid_markers = group_counts[
            (group_counts[reference] > min_n_obs) & (group_counts[target] > min_n_obs)
        ].index
        target_data = target_data[
            target_data.set_index(["marker_1", "marker_2"]).index.isin(valid_markers)
        ]
    return target_data


def _perform_mannwhitneyu_test(ref_group, tgt_group):
    u_stat, p_value = mannwhitneyu(ref_group, tgt_group, alternative="two-sided")
    auc = u_stat / (len(ref_group) * len(tgt_group))
    tgt_median = np.median(tgt_group)
    ref_median = np.median(ref_group)
    median_diff = tgt_median - ref_median
    return u_stat, p_value, auc, median_diff, tgt_median, ref_median


def calculate_differential_proximity(
    proximity_df: pd.DataFrame,
    contrast_column: str,
    reference: str,
    targets: List[str] | None = None,
    metric: str = "join_count_z",
    metric_type: Literal["all", "self", "co"] = "all",
    min_n_obs: int = 0,
    p_adjust_method: Literal[
        "bonferroni", "holm", "hochberg", "hommel", "fdr_bh", "fdr_by", "sidak"
    ] = "bonferroni",
) -> pd.DataFrame:
    """Perform differential proximity analysis between groups of marker-pair data.

    Compare a proximity metric between a reference group and one or more target
    groups for marker pairs using the two-sided `Mann-Whitney U test <https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test>`_.
    P-values are adjusted to counteract the `multiple comparisons problem <https://en.wikipedia.org/wiki/Multiple_comparisons_problem>`_.
    See references below for resources on p-value adjustment methods.

    Args:
        proximity_df: Input data containing proximity metrics and grouping information.
            Must include ``contrast_column``, ``marker_1``, ``marker_2``,
            and the column determined by ``metric``.
        contrast_column: Column name of the grouping variable.
        reference: Reference group label in ``contrast_column``.
        targets: Target group labels to compare against ``reference``. If ``None``,
            all other groups in ``contrast_column`` except ``reference`` are used. Defaults to ``None``.
        metric: Column name of the proximity metric to analyze. Defaults to
            ``join_count_z``.
        metric_type: Marker pairs to include: ``"self"`` (includes pairs for which ``marker_1`` and
            ``marker_2`` are the same), ``"co"``
            (includes pairs for which ``marker_1`` and ``marker_2`` are different), or
            ``"all"``. This parameter defaults to ``"all"``.
        min_n_obs: When greater than zero, marker pairs must have more than this
            many observations in both ``reference`` and each target group. Defaults to 0.
        p_adjust_method: Method for adjusting p-values for multiple comparisons.
            ``p_adjust_method`` is passed to ``statsmodels.stats.multitest.multipletests``.
            Defaults to ``"bonferroni"``.

    Returns:
        DataFrame containing the results of the differential
        proximity analysis, including statistical metrics and adjusted p-values.

    Raises:
        ValueError: If ``contrast_column`` is not in ``proximity_df``.
        ValueError: If no data remains after applying ``metric_type`` filtering.

    References:
        * `Bonferroni correction <https://en.wikipedia.org/wiki/Bonferroni_correction>`_
        * `Holm–Bonferroni method <https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method>`_
        * `Hochberg method <https://www.sciencedirect.com/topics/mathematics/hochberg-method#:~:text=Hochberg's%20method%20is%20defined%20as,applicable%20when%20testing%20multiple%20hypotheses.>`_
        * `Hommel <https://academic.oup.com/biomet/article-abstract/75/2/383/292949?redirectedFrom=fulltext&login=false>`_
        * `False discovery rate <https://en.wikipedia.org/wiki/False_discovery_rate>`_
        * `Sidak correction <https://en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction>`_
    """
    if contrast_column not in proximity_df.columns:
        raise ValueError(f"{contrast_column} must be a column in the data.")

    if targets is None:
        targets = proximity_df[contrast_column].unique().tolist()
        targets.remove(reference)

    if metric_type == "self":
        proximity_df = proximity_df[
            proximity_df["marker_1"] == proximity_df["marker_2"]
        ]
    elif metric_type == "co":
        proximity_df = proximity_df[
            proximity_df["marker_1"] != proximity_df["marker_2"]
        ]

    if proximity_df.empty:
        raise ValueError("No data found for the specified metric type.")

    def calc_targets_differential():
        """Calc targets differential."""
        for target in targets:
            target_data = _filter_target_data(
                proximity_df, contrast_column, reference, target, metric, min_n_obs
            )

            for (marker_1, marker_2), group in target_data.groupby(
                ["marker_1", "marker_2"]
            ):
                ref_group = group[group[contrast_column] == reference][metric]
                tgt_group = group[group[contrast_column] == target][metric]

                if len(ref_group) == 0 or len(tgt_group) == 0:
                    continue

                u_stat, p_value, auc, median_diff, tgt_median, ref_median = (
                    _perform_mannwhitneyu_test(ref_group, tgt_group)
                )
                results = pd.Series(
                    {
                        "marker_1": marker_1,
                        "marker_2": marker_2,
                        "reference": reference,
                        "target": target,
                        "u_stat": u_stat,
                        "p_value": p_value,
                        "auc": auc,
                        "median_diff": median_diff,
                        "tgt_median": tgt_median,
                        "ref_median": ref_median,
                        "n_ref": len(ref_group),
                        "n_tgt": len(tgt_group),
                    }
                )
                yield results

    results_df = pd.DataFrame((calc_targets_differential()))
    if results_df.empty:
        return results_df
    results_df["p_adjusted"] = multipletests(
        results_df["p_value"], method=p_adjust_method
    )[1]

    return results_df

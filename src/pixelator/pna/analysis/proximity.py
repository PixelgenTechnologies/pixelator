"""Module for computing localization proximity statistics.

Copyright © 2024 Pixelgen Technologies AB
"""

from typing import Callable, List, Literal

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests

from pixelator.pna.analysis.permute import edgelist_permutations
from pixelator.pna.utils.utils import normalize_input_to_list

_PROXIMITY_METRICS = ("log2_ratio", "join_count_z")
_SUMMARY_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "median": np.median,
}


def get_join_counts(edgelist: pl.DataFrame) -> pd.DataFrame:
    """Compute the number of edges for each marker pair in the given edgelist.

    Args:
        edgelist: A DataFrame representing the edgelist with columns "marker_1" and "marker_2".

    Returns:
        DataFrame containing the number of edges for each marker pair. The resulting DataFrame
        includes columns ``marker_1``, ``marker_2``, and ``join_count``.
    """
    pair_cnt = edgelist.group_by(["marker_1", "marker_2"]).len().to_pandas()
    m1 = pair_cnt["marker_1"].astype(str)
    m2 = pair_cnt["marker_2"].astype(str)
    pair_cnt.loc[m1 > m2, "marker_1"] = m2
    pair_cnt.loc[m1 > m2, "marker_2"] = m1
    all_markers = list(set(pair_cnt["marker_1"]).union(set(pair_cnt["marker_2"])))
    pair_cnt["marker_1"] = pd.Categorical(pair_cnt["marker_1"], categories=all_markers)
    pair_cnt["marker_2"] = pd.Categorical(pair_cnt["marker_2"], categories=all_markers)
    pair_cnt = (
        pair_cnt.groupby(["marker_1", "marker_2"], observed=False)["len"]
        .sum()
        .reset_index()
    )
    pair_cnt.rename(columns={"len": "join_count"}, inplace=True)
    pair_cnt = pair_cnt[
        pair_cnt["marker_1"].astype(str) <= pair_cnt["marker_2"].astype(str)
    ]  # Each marker-pair gets only one row, where marker_1/marker_2 are in lexographic order
    return pair_cnt


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


def proximity_with_permute_stats(
    edgelist: pl.DataFrame,
    proximity_function: Callable[[pl.DataFrame], pd.DataFrame],
    result_columns: list[str],
    n_permutations: int = 100,
    seed: int | None = 42,
    min_std: float = 1.0,
    min_marker_count: int = 0,
) -> pd.DataFrame:
    """Compute proximity results augmented with statistics based on permutation tests.

    This function calculates proximity metrics for a given edgelist and augments
    the results with statistical measures derived from permutation tests. It
    supports computing z-scores and p-values for specified result columns.

    Args:
        edgelist: A DataFrame representing the edgelist.
        proximity_function: A function that computes proximity metrics for the given edgelist.
        result_columns: A list of column names for which statistics (e.g., z-scores, p-values) will
            be computed.
        n_permutations: The number of permutations to perform. Defaults to 100.
        seed: Seed for the random number generator. Defaults to 42.
        min_std: Minimum standard deviation to use when normalizing z-scores. Defaults to 1.0.
        min_marker_count: Minimum marker count threshold for filtering the edgelist. Defaults to 0.

    Returns:
        A DataFrame containing the proximity results augmented with
        statistical measures, including expected means, standard deviations,
        z-scores, and p-values for the specified result columns.
    """
    passing_markers = _get_markers_above_min_count(edgelist, min_marker_count)
    results = proximity_function(edgelist).set_index(["marker_1", "marker_2"])

    def compute_permuted_results():
        """Compute permuted results."""
        permutations = edgelist_permutations(edgelist, n_permutations, seed)
        for idx, perm in enumerate(permutations):
            perm_results = proximity_function(perm)
            perm_results["perm_idx"] = idx
            yield perm_results

    permuted_results = pd.concat(compute_permuted_results())

    for col in result_columns:
        results[f"{col}_expected_mean"] = permuted_results.groupby(
            ["marker_1", "marker_2"], observed=False
        )[col].mean()
        results[f"{col}_expected_sd"] = permuted_results.groupby(
            ["marker_1", "marker_2"], observed=False
        )[col].std()
        results[f"{col}_expected_mean"] = results[f"{col}_expected_mean"].fillna(0)
        results[f"{col}_expected_sd"] = results[f"{col}_expected_sd"].fillna(min_std)
        results[f"{col}_z"] = (
            results[col] - results[f"{col}_expected_mean"]
        ) / np.maximum(results[f"{col}_expected_sd"], min_std)
        results[f"{col}_p"] = 2 * norm.sf(np.abs(results[f"{col}_z"]))

    results = results.reset_index()
    results = results[
        results["marker_1"].isin(passing_markers)
        & results["marker_2"].isin(passing_markers)
    ]

    return results


def jcs_with_permute_stats(
    edgelist: pl.DataFrame,
    n_permutations: int = 100,
    min_marker_count: int = 0,
) -> pd.DataFrame:
    """Compute proximity results augmented with statistics based on permutation tests.

    Args:
        edgelist: A DataFrame representing the edgelist.
        n_permutations: Number of permutations to perform. Defaults to 100.
        min_marker_count: Minimum marker count to consider. Defaults to 0.

    Returns:
        A DataFrame containing the proximity statistics.
    """
    return proximity_with_permute_stats(
        edgelist,
        get_join_counts,
        ["join_count"],
        n_permutations=n_permutations,
        seed=42,
        min_std=1.0,
        min_marker_count=min_marker_count,
    )


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

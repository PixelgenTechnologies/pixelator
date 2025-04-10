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


def get_join_counts(edgelist: pl.DataFrame) -> pd.DataFrame:
    """Get the number of edges for each marker pair in the edgelist.

    :param edgelist: A DataFrame representing the edgelist
    :param marker_list: A list of markers. If not provided, it will be
    generated from the unique markers in the edgelist
    :returns: A DataFrame containing the number of edges for each marker pair
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


def proximity_with_permute_stats(
    edgelist: pl.DataFrame,
    proximity_function: Callable[[pl.DataFrame], pd.DataFrame],
    result_columns: list[str],
    n_permutations: int = 100,
    seed: int | None = 42,
    min_std: float = 1.0,
) -> pd.DataFrame:
    """Compute proximity results augmented with statistics based on permutation tests.

    :param edgelist: A DataFrame representing the edgelist
    :param proximity_function: A function that computes proximity
    :param result_columns: A list of columns to compute statistics for
    :param n_permutations: Number of permutations to perform
    :param seed: Seed for the random number generator
    :returns: A DataFrame containing the proximity statistics
    """
    results = proximity_function(edgelist).set_index(["marker_1", "marker_2"])

    def compute_permuted_results():
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
        results[f"{col}_p"] = norm.sf(np.abs(results[f"{col}_z"]))

    return results.reset_index()


def jcs_with_permute_stats(
    edgelist: pl.DataFrame, n_permutations: int = 100
) -> pd.DataFrame:
    """Compute proximity results augmented with statistics based on permutation tests.

    :param edgelist: A DataFrame representing the edgelist
    :param n_permutations: Number of permutations to perform
    :returns: A DataFrame containing the proximity statistics
    """
    return proximity_with_permute_stats(
        edgelist,
        get_join_counts,
        ["join_count"],
        n_permutations=n_permutations,
        seed=42,
        min_std=1.0,
    )


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
    """Perform differential analysis on marker-pair proximity data.

    :param proximity_df: Input data containing proximity metrics and grouping
        information. Must include columns for `contrast_column`, `marker_1`,
        `marker_2`, and the proximity metric (default: "join_count_z").
    :param contrast_column: The column name representing the grouping variable
        for contrasts.
    :param reference: The reference group in the contrast_column.
    :param targets : List of target groups to compare against the reference.
        If None, all groups in `contrast_column` except the reference are used.
    :param metric: Column name representing the proximity metric to analyze.,
        default="join_count_z"
    :param metric_type: Type of measures to analyze (self, co- or all
        proximities), default="all"
    :param min_n_obs: Minimum number of observations required for a group to be
        included in the analysis., default=0
    :param p_adjust_method: Method for adjusting p-values for multiple
        comparisons,default="bonferroni". The available methods are:
        "bonferroni", "holm", "hochberg", "hommel", "fdr_bh", "fdr_by",
        and "sidak". You can refer to statsmodels.stats.multitest documentation
        for method implementation details.

    :return: A DataFrame containing the results of the differential proximity
        analysis.
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

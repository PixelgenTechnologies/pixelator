"""Module for computing localization proximity statistics.

Copyright © 2024 Pixelgen Technologies AB
"""

from typing import Callable, List, Literal

import numpy as np
import pandas as pd
import polars as pl
from duckdb import DuckDBPyConnection
from scipy.stats import mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests

from pixelator.pna.analysis.permute import edgelist_permutations
from pixelator.pna.utils.utils import normalize_input_to_list


def get_join_counts(edgelist: pl.DataFrame) -> pd.DataFrame:
    """Compute the number of edges for each marker pair in the given edgelist.

    Args:
        edgelist (pl.DataFrame): A DataFrame representing the edgelist with
            columns "marker_1" and "marker_2".

    Returns:
        pd.DataFrame: A DataFrame containing the number of edges for each
        marker pair. The resulting DataFrame includes the columns:
            - "marker_1": The first marker in the pair.
            - "marker_2": The second marker in the pair.
            - "join_count": The number of edges between the marker pair.

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
        edgelist (pl.DataFrame): A DataFrame representing the edgelist.
        min_count (int, optional): Minimum count threshold for markers. Defaults to 0.

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
        edgelist (pl.DataFrame): A DataFrame representing the edgelist.
        proximity_function (Callable[[pl.DataFrame], pd.DataFrame]): A function
            that computes proximity metrics for the given edgelist.
        result_columns (list[str]): A list of column names for which statistics
            (e.g., z-scores, p-values) will be computed.
        n_permutations (int, optional): The number of permutations to perform.
            Defaults to 100.
        seed (int | None, optional): Seed for the random number generator.
            Defaults to 42.
        min_std (float, optional): Minimum standard deviation to use when
            normalizing z-scores. Defaults to 1.0.
        min_marker_count (int, optional): Minimum marker count threshold for
            filtering the edgelist. Defaults to 0.

    Returns:
        pd.DataFrame: A DataFrame containing the proximity results augmented with
        statistical measures, including expected means, standard deviations,
        z-scores, and p-values for the specified result columns.

    """
    passing_markers = _get_markers_above_min_count(edgelist, min_marker_count)
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
        edgelist (pl.DataFrame): A DataFrame representing the edgelist.
        n_permutations (int, optional): Number of permutations to perform. Defaults to 100.
        min_marker_count (int, optional): Minimum marker count to consider. Defaults to 0.

    Returns:
        pd.DataFrame: A DataFrame containing the proximity statistics.

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

    Args:
        proximity_df (pd.DataFrame): Input data containing proximity metrics and
            grouping information. Must include columns for `contrast_column`,
            `marker_1`, `marker_2`, and the proximity metric (default: "join_count_z").
        contrast_column (str): The column name representing the grouping variable
        reference (str): The reference group in the `contrast_column`.
        targets (List[str] | None, optional): List of target groups to compare
            against the reference. If None, all groups in `contrast_column` except
            the reference are used. Defaults to None.
        metric (str, optional): Column name representing the proximity metric to
            analyze. Defaults to "join_count_z".
        metric_type (Literal["all", "self", "co"], optional): Type of measures to
            analyze ("self", "co", or "all" proximities). Defaults to "all".
        min_n_obs (int, optional): Minimum number of observations required for a
            group to be included in the analysis. Defaults to 0.
        p_adjust_method (optional): Method for adjusting p-values
            for multiple comparisons. Defaults to "bonferroni".

    Returns:
        pd.DataFrame: A DataFrame containing the results of the differential
        proximity analysis, including statistical metrics and adjusted p-values.

    Raises:
        ValueError: If `contrast_column` is not in `proximity_df`.
        ValueError: If no data is found for the specified `metric_type`.

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


def jcs_with_analytical_stats(
    database_connection: DuckDBPyConnection,
    components: str | list[str] | set[str] | None = None,
    markers: str | list[str] | set[str] | None = None,
) -> pl.DataFrame:
    """Compute proximity results using analytical mean and standard deviation of join counts.

    Args:
        database_connection (DuckDBPyConnection): Used to submit database queries for pixel data.
        components (str | list[str] | set[str] | None): A list of components to include in the analysis.
        markers (str | list[str] | set[str] | None): A list of marker names to include in the analysis.

    Returns:
        pl.DataFrame: A DataFrame containing the proximity results with analytical statistics.

    """
    markers = normalize_input_to_list(markers)
    components = normalize_input_to_list(components)
    params = {}
    if components:
        params["components"] = components
    if markers:
        params["markers"] = markers

    get_current_edgelist = f"""
        current_edgelist AS (
            SELECT *
            FROM edgelist
            WHERE {"component IN $components" if components else "TRUE"}
        )"""

    cte_group_edges = f"""
        group_edges AS (
            SELECT sample, component, COUNT(*) as n_edges 
            FROM current_edgelist
            GROUP BY sample, component
        )"""

    cte_all_markers = f"""
        all_markers AS (
            SELECT sample, component, marker_1 AS marker FROM current_edgelist
            UNION
            SELECT sample, component, marker_2 AS marker FROM current_edgelist
        )"""

    cte_marker1_stats = f"""
        unique_m1 AS (
            SELECT DISTINCT sample, component, umi1, marker_1 FROM current_edgelist
        ),
        raw_stats_m1 AS (
            SELECT 
                sample, component, marker_1, COUNT(*) as marker_1_count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY sample, component) as f_umi1                
            FROM unique_m1
            GROUP BY sample, component, marker_1
        ),
        stats_m1 AS (
            SELECT 
                am.sample, 
                am.component, 
                am.marker AS marker_1, 
                COALESCE(rm1.marker_1_count, 0) AS marker_1_count,
                COALESCE(rm1.f_umi1, 0.0) AS f_umi1
            FROM all_markers am
            LEFT JOIN raw_stats_m1 rm1 
                ON am.sample = rm1.sample 
                AND am.component = rm1.component 
                AND am.marker = rm1.marker_1
        )"""

    cte_marker2_stats = f"""
        unique_m2 AS (
            SELECT DISTINCT sample, component, umi2, marker_2 FROM current_edgelist
        ),
        raw_stats_m2 AS (
            SELECT 
                sample, component, marker_2, COUNT(*) as marker_2_count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY sample, component) as f_umi2
            FROM unique_m2
            GROUP BY sample, component, marker_2
        ),
        stats_m2 AS (
            SELECT 
                am.sample, 
                am.component, 
                am.marker AS marker_2, 
                COALESCE(rm2.marker_2_count, 0) AS marker_2_count,
                COALESCE(rm2.f_umi2, 0.0) AS f_umi2
            FROM all_markers am
            LEFT JOIN raw_stats_m2 rm2 
                ON am.sample = rm2.sample 
                AND am.component = rm2.component 
                AND am.marker = rm2.marker_2
        )"""

    cte_expected_counts = f"""
        expected_calc AS (
            SELECT
                t1.sample,
                t1.component,
                LEAST(t1.marker_1, t2.marker_2) as marker_A,
                GREATEST(t1.marker_1, t2.marker_2) as marker_B,
                (t1.f_umi1 * t2.f_umi2 * ge.n_edges) as exp_count_raw,
                (t1.f_umi1 * t2.f_umi2 * (1 - (t1.f_umi1 * t2.f_umi2)) * ge.n_edges) as exp_count_var
            FROM stats_m1 t1
            JOIN stats_m2 t2 
            ON t1.sample = t2.sample AND t1.component = t2.component
            JOIN group_edges ge 
            ON t1.sample = ge.sample AND t1.component = ge.component
            WHERE {"marker_1 IN $markers AND marker_2 IN $markers" if markers else "TRUE"}
        ),
        expected_agg AS (
            SELECT sample, component, marker_A, marker_B, SUM(exp_count_raw) as join_count_expected_mean, SQRT(SUM(exp_count_var)) as join_count_expected_sd
            FROM expected_calc
            GROUP BY sample, component, marker_A, marker_B
        )"""

    cte_observed_counts = f"""
        observed_agg AS (
            SELECT
                sample,
                component,
                LEAST(marker_1, marker_2) as marker_A,
                GREATEST(marker_1, marker_2) as marker_B,
                COUNT(*) as join_count
            FROM current_edgelist
            WHERE {"marker_1 IN $markers AND marker_2 IN $markers" if markers else "TRUE"}
            GROUP BY sample, component, marker_A, marker_B
        )"""

    cte_final_results = """
        res AS (
            SELECT
                COALESCE(obs.sample, exp.sample) as sample,
                COALESCE(obs.component, exp.component) as component,
                COALESCE(obs.marker_A, exp.marker_A) as marker_1,
                COALESCE(obs.marker_B, exp.marker_B) as marker_2,
                COALESCE(obs.join_count, 0) as join_count,
                COALESCE(exp.join_count_expected_mean, 0) as join_count_expected_mean,
                GREATEST(COALESCE(exp.join_count_expected_sd, 0), 1e-6) as join_count_expected_sd,
                LOG2(GREATEST(COALESCE(obs.join_count, 0), 1) / GREATEST(COALESCE(exp.join_count_expected_mean, 0), 1)) AS log2_ratio,
                (COALESCE(obs.join_count, 0) - COALESCE(exp.join_count_expected_mean, 0)) / GREATEST(COALESCE(exp.join_count_expected_sd, 0), 1e-6) AS join_count_z,
                2 * (1 - dist_normal_cdf(0.0, GREATEST(COALESCE(exp.join_count_expected_sd, 0), 1e-6), ABS(COALESCE(obs.join_count, 0) - COALESCE(exp.join_count_expected_mean, 0)))) AS join_count_p
            FROM observed_agg obs
            FULL OUTER JOIN expected_agg exp 
                ON obs.sample = exp.sample
                AND obs.component = exp.component
                AND obs.marker_A = exp.marker_A 
                AND obs.marker_B = exp.marker_B
        )"""

    analysis_query = f"""
    WITH 
    {get_current_edgelist},
    {cte_group_edges},
    {cte_all_markers},
    {cte_marker1_stats},
    {cte_marker2_stats},
    {cte_expected_counts},
    {cte_observed_counts},
    {cte_final_results}
    SELECT * FROM res;
    """
    database_connection.execute("INSTALL stochastic FROM community; LOAD stochastic;")
    results = database_connection.execute(analysis_query, parameters=params).pl()
    return results

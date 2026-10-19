"""Module for constructing analytical proximity sql queries.

The expected join counts are calculated analytically under the same null model
as the marker permutation test: the markers of the umi1 nodes and the markers of
the umi2 nodes are independently shuffled within each component, keeping the
graph and the number of nodes per marker fixed.

Copyright © 2026 Pixelgen Technologies AB
"""

import numpy as np
import polars as pl
from scipy.stats import norm

from pixelator.pna.utils.utils import normalize_input_to_list

DEFAULT_MIN_STD = 1.0


def _ordered_pair_moments(side_1: str, side_2: str) -> tuple[str, str]:
    """Construct SQL expressions for the mean and variance of an ordered join count.

    The ordered join count N_XY is the number of edges where the umi1 node has
    marker X (taken from the `side_1` marker table alias) and the umi2 node has
    marker Y (taken from the `side_2` marker table alias).

    Under the permutation null, with E edges, U1 umi1 nodes (of which n1 have
    marker X) and U2 umi2 nodes (of which n2 have marker Y):

        p = n1 / U1, a = n1 (n1 - 1) / (U1 (U1 - 1))
        q = n2 / U2, b = n2 (n2 - 1) / (U2 (U2 - 1))
        E[N_XY] = E p q
        Var[N_XY] = E p q + (S1 - E) p b + (S2 - E) a q + K a b - (E p q)^2

    where S1 and S2 are the sums of squared node degrees of umi1 and umi2 nodes,
    and K = E^2 - S1 - S2 + E is the number of ordered edge pairs that share no
    node. The S1 and S2 terms capture that all edges of a node carry the same
    marker, which makes the join counts more variable than if every edge was an
    independent draw.
    """
    p = f"({side_1}.n1 / cs.n_umi1)"
    q = f"({side_2}.n2 / cs.n_umi2)"
    a = f"({side_1}.n1 * ({side_1}.n1 - 1) / cs.n_umi1_pairs)"
    b = f"({side_2}.n2 * ({side_2}.n2 - 1) / cs.n_umi2_pairs)"
    mean = f"(cs.n_edges * {p} * {q})"
    var = f"""(
        cs.n_edges * {p} * {q}
        + (cs.sum_sq_degree_1 - cs.n_edges) * {p} * {b}
        + (cs.sum_sq_degree_2 - cs.n_edges) * {a} * {q}
        + cs.n_disjoint_edge_pairs * {a} * {b}
        - {mean} * {mean}
    )"""
    return mean, var


def jcs_with_analytical_stats(
    components: str | list[str] | set[str] | None = None,
    markers: str | list[str] | set[str] | None = None,
    min_std: float = DEFAULT_MIN_STD,
) -> tuple[str, dict]:
    """Construct a SQL query for calculating proximity join counts with analytical statistics.

    The query expects an `edgelist` table and returns one row per sample, component
    and (unordered) marker pair with the columns: `sample`, `component`, `marker_1`,
    `marker_2`, `join_count`, `join_count_expected_mean`, `join_count_expected_sd`
    and `join_count_z`. Use `add_join_count_p` to add the p-values to the result.

    Args:
        components: A list of components to include in the analysis.
        markers: A list of marker names to include in the analysis.
        min_std: Minimum standard deviation used when calculating the z-scores.

    Returns:
        A tuple containing the SQL query string and a dictionary of parameters.
    """
    markers = normalize_input_to_list(markers)
    components = normalize_input_to_list(components)
    params: dict = {"min_std": float(min_std)}
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

    cte_nodes = """
        umi1_nodes AS (
            SELECT sample, component, umi1, ANY_VALUE(marker_1) AS marker, COUNT(*) AS degree
            FROM current_edgelist
            GROUP BY sample, component, umi1
        ),
        umi2_nodes AS (
            SELECT sample, component, umi2, ANY_VALUE(marker_2) AS marker, COUNT(*) AS degree
            FROM current_edgelist
            GROUP BY sample, component, umi2
        )"""

    cte_component_stats = """
        umi1_stats AS (
            SELECT
                sample, component,
                CAST(COUNT(*) AS DOUBLE) AS n_umi1,
                CAST(SUM(degree) AS DOUBLE) AS n_edges,
                CAST(SUM(degree * degree) AS DOUBLE) AS sum_sq_degree_1
            FROM umi1_nodes
            GROUP BY sample, component
        ),
        umi2_stats AS (
            SELECT
                sample, component,
                CAST(COUNT(*) AS DOUBLE) AS n_umi2,
                CAST(SUM(degree * degree) AS DOUBLE) AS sum_sq_degree_2
            FROM umi2_nodes
            GROUP BY sample, component
        ),
        component_stats AS (
            SELECT
                s1.sample, s1.component,
                s1.n_edges, s1.n_umi1, s2.n_umi2,
                s1.sum_sq_degree_1, s2.sum_sq_degree_2,
                -- Number of ordered pairs of distinct nodes (clamped to avoid division by zero,
                -- the numerators are always zero when there is only one node)
                GREATEST(s1.n_umi1 * (s1.n_umi1 - 1), 1) AS n_umi1_pairs,
                GREATEST(s2.n_umi2 * (s2.n_umi2 - 1), 1) AS n_umi2_pairs,
                -- Number of ordered pairs of edges that share neither the umi1 nor the umi2 node
                s1.n_edges * s1.n_edges - s1.sum_sq_degree_1 - s2.sum_sq_degree_2 + s1.n_edges
                    AS n_disjoint_edge_pairs
            FROM umi1_stats s1
            JOIN umi2_stats s2
                ON s1.sample = s2.sample AND s1.component = s2.component
        )"""

    cte_marker_stats = """
        all_markers AS (
            SELECT sample, component, marker FROM umi1_nodes
            UNION
            SELECT sample, component, marker FROM umi2_nodes
        ),
        marker_1_counts AS (
            SELECT sample, component, marker, COUNT(*) AS n1
            FROM umi1_nodes
            GROUP BY sample, component, marker
        ),
        marker_2_counts AS (
            SELECT sample, component, marker, COUNT(*) AS n2
            FROM umi2_nodes
            GROUP BY sample, component, marker
        ),
        marker_stats AS (
            SELECT
                am.sample,
                am.component,
                am.marker,
                CAST(COALESCE(m1.n1, 0) AS DOUBLE) AS n1,
                CAST(COALESCE(m2.n2, 0) AS DOUBLE) AS n2
            FROM all_markers am
            LEFT JOIN marker_1_counts m1
                ON am.sample = m1.sample AND am.component = m1.component AND am.marker = m1.marker
            LEFT JOIN marker_2_counts m2
                ON am.sample = m2.sample AND am.component = m2.component AND am.marker = m2.marker
        )"""

    mean_ab, var_ab = _ordered_pair_moments("ma", "mb")
    mean_ba, var_ba = _ordered_pair_moments("mb", "ma")
    # Covariance of the two ordered join counts N_AB and N_BA (A != B). Only edge pairs
    # that share no node contribute, since a node can not have both marker A and B.
    cov_ab_ba = f"""(
        cs.n_disjoint_edge_pairs
        * (ma.n1 * mb.n1 / cs.n_umi1_pairs)
        * (ma.n2 * mb.n2 / cs.n_umi2_pairs)
        - {mean_ab} * {mean_ba}
    )"""

    cte_expected = f"""
        expected AS (
            SELECT
                ma.sample,
                ma.component,
                ma.marker AS marker_1,
                mb.marker AS marker_2,
                CASE WHEN ma.marker = mb.marker
                    THEN {mean_ab}
                    ELSE {mean_ab} + {mean_ba}
                END AS join_count_expected_mean,
                SQRT(GREATEST(
                    CASE WHEN ma.marker = mb.marker
                        THEN {var_ab}
                        ELSE {var_ab} + {var_ba} + 2 * {cov_ab_ba}
                    END,
                    0.0
                )) AS join_count_expected_sd
            FROM marker_stats ma
            JOIN marker_stats mb
                ON ma.sample = mb.sample
                AND ma.component = mb.component
                AND ma.marker <= mb.marker
            JOIN component_stats cs
                ON ma.sample = cs.sample AND ma.component = cs.component
            WHERE {"ma.marker IN $markers AND mb.marker IN $markers" if markers else "TRUE"}
        )"""

    cte_observed = f"""
        observed AS (
            SELECT
                sample,
                component,
                LEAST(marker_1, marker_2) AS marker_1,
                GREATEST(marker_1, marker_2) AS marker_2,
                COUNT(*) AS join_count
            FROM current_edgelist
            WHERE {"marker_1 IN $markers AND marker_2 IN $markers" if markers else "TRUE"}
            GROUP BY sample, component, LEAST(marker_1, marker_2), GREATEST(marker_1, marker_2)
        )"""

    cte_final_results = """
        res AS (
            SELECT
                exp.sample,
                exp.component,
                exp.marker_1,
                exp.marker_2,
                CAST(COALESCE(obs.join_count, 0) AS UINTEGER) AS join_count,
                exp.join_count_expected_mean,
                exp.join_count_expected_sd,
                (COALESCE(obs.join_count, 0) - exp.join_count_expected_mean)
                    / GREATEST(exp.join_count_expected_sd, $min_std) AS join_count_z
            FROM expected exp
            LEFT JOIN observed obs
                ON obs.sample = exp.sample
                AND obs.component = exp.component
                AND obs.marker_1 = exp.marker_1
                AND obs.marker_2 = exp.marker_2
        )"""

    analysis_query = f"""
    WITH
    {get_current_edgelist},
    {cte_nodes},
    {cte_component_stats},
    {cte_marker_stats},
    {cte_expected},
    {cte_observed},
    {cte_final_results}
    SELECT * FROM res;
    """

    return analysis_query, params


def add_join_count_p(proximity: pl.DataFrame) -> pl.DataFrame:
    """Add two-sided p-values (`join_count_p`) for the `join_count_z` scores.

    Args:
        proximity: A DataFrame with a `join_count_z` column.

    Returns:
        The DataFrame with an added `join_count_p` column.
    """
    p_values = 2 * norm.sf(np.abs(proximity["join_count_z"].to_numpy()))
    return proximity.with_columns(join_count_p=pl.Series(p_values, dtype=pl.Float64))

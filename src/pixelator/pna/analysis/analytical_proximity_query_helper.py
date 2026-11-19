"""Module for constructing analytical proximity sql queries.

The expected join counts are calculated analytically under an edge-level null
model: the `marker_1` labels and the `marker_2` labels are independently shuffled
across the edges of each component, keeping the number of edges per marker on
each side fixed. Since the marker frequencies are taken over edges rather than
over nodes, markers whose nodes have a higher average degree get a
correspondingly higher expected join count.

Copyright © 2026 Pixelgen Technologies AB
"""

import numpy as np
import polars as pl
from scipy.stats import norm

from pixelator.pna.utils.utils import normalize_input_to_list

DEFAULT_MIN_STD = 1.0


def _ordered_pair_moments(side_1: str, side_2: str) -> tuple[str, str]:
    """Construct SQL expressions for the mean and variance of an ordered join count.

    The ordered join count N_XY is the number of edges where `marker_1` is X
    (taken from the `side_1` marker table alias) and `marker_2` is Y (taken
    from the `side_2` marker table alias).

    Under the edge-level permutation null, with E edges, of which e1 have
    `marker_1` X and e2 have `marker_2` Y:

        p = e1 / E, a = e1 (e1 - 1) / (E (E - 1))
        q = e2 / E, b = e2 (e2 - 1) / (E (E - 1))
        E[N_XY] = E p q
        Var[N_XY] = E p q + E (E - 1) a b - (E p q)^2

    Note that this null treats every edge as an independent draw, i.e. it does
    not account for all edges of a node carrying the same marker.
    """
    p = f"({side_1}.e1 / cs.n_edges)"
    q = f"({side_2}.e2 / cs.n_edges)"
    a = f"({side_1}.e1 * ({side_1}.e1 - 1) / cs.n_edge_pairs)"
    b = f"({side_2}.e2 * ({side_2}.e2 - 1) / cs.n_edge_pairs)"
    mean = f"(cs.n_edges * {p} * {q})"
    var = f"""(
        cs.n_edges * {p} * {q}
        + cs.n_edge_pairs * {a} * {b}
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

    cte_component_stats = """
        component_stats AS (
            SELECT
                sample, component,
                CAST(COUNT(*) AS DOUBLE) AS n_edges,
                -- Number of ordered pairs of distinct edges (clamped to avoid division
                -- by zero, the numerators are always zero when there is only one edge)
                GREATEST(CAST(COUNT(*) AS DOUBLE) * (COUNT(*) - 1), 1) AS n_edge_pairs
            FROM current_edgelist
            GROUP BY sample, component
        )"""

    cte_marker_stats = """
        marker_1_counts AS (
            SELECT sample, component, marker_1 AS marker, COUNT(*) AS e1
            FROM current_edgelist
            GROUP BY sample, component, marker_1
        ),
        marker_2_counts AS (
            SELECT sample, component, marker_2 AS marker, COUNT(*) AS e2
            FROM current_edgelist
            GROUP BY sample, component, marker_2
        ),
        marker_stats AS (
            SELECT
                COALESCE(m1.sample, m2.sample) AS sample,
                COALESCE(m1.component, m2.component) AS component,
                COALESCE(m1.marker, m2.marker) AS marker,
                CAST(COALESCE(m1.e1, 0) AS DOUBLE) AS e1,
                CAST(COALESCE(m2.e2, 0) AS DOUBLE) AS e2
            FROM marker_1_counts m1
            FULL OUTER JOIN marker_2_counts m2
                ON m1.sample = m2.sample AND m1.component = m2.component AND m1.marker = m2.marker
        )"""

    mean_ab, var_ab = _ordered_pair_moments("ma", "mb")
    mean_ba, var_ba = _ordered_pair_moments("mb", "ma")
    # Covariance of the two ordered join counts N_AB and N_BA (A != B). Only pairs of
    # distinct edges contribute, since a single edge can not have marker_1 both A and B.
    cov_ab_ba = f"""(
        cs.n_edge_pairs
        * (ma.e1 * mb.e1 / cs.n_edge_pairs)
        * (ma.e2 * mb.e2 / cs.n_edge_pairs)
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

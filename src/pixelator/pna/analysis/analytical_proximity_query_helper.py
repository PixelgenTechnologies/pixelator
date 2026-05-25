"""Module for constructing analytical proximity sql queries.

Copyright © 2026 Pixelgen Technologies AB
"""

from pixelator.pna.utils.utils import normalize_input_to_list


def jcs_with_analytical_stats(
    components: str | list[str] | set[str] | None = None,
    markers: str | list[str] | set[str] | None = None,
) -> tuple[str, dict]:
    """Construct a SQL query for calculating proximity join counts with analytical statistics.

    Args:
        components: A list of components to include in the analysis.
        markers: A list of marker names to include in the analysis.

    Returns:
        tuple[str, dict]: A tuple containing the SQL query string and a dictionary of parameters.
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
            WHERE {"t1.marker_1 IN $markers AND t2.marker_2 IN $markers" if markers else "TRUE"}
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

    return analysis_query, params

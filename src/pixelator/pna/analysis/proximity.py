"""Module for computing localization proximity statistics.

Copyright © 2024 Pixelgen Technologies AB
"""

import warnings
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
        # A group that exists only in cell_groups has no proximity rows, so
        # unstack omits its column. That is a count of 0, not a missing label.
        group_counts = group_counts.reindex(columns=[reference, target], fill_value=0)
        valid_markers = group_counts[
            (group_counts[reference] > min_n_obs) & (group_counts[target] > min_n_obs)
        ].index
        target_data = target_data[
            target_data.set_index(["marker_1", "marker_2"]).index.isin(valid_markers)
        ]
    return target_data


_DIFFERENTIAL_PROXIMITY_COLUMNS = [
    "marker_1",
    "marker_2",
    "reference",
    "target",
    "u_stat",
    "p_value",
    "auc",
    "median_diff",
    "tgt_median",
    "ref_median",
    "n_ref",
    "n_tgt",
    "pct_ref",
    "pct_tgt",
    "p_adjusted",
]


def _empty_differential_proximity_result() -> pd.DataFrame:
    """Return a zero-row result with the stable differential-proximity columns."""
    return pd.DataFrame(columns=_DIFFERENTIAL_PROXIMITY_COLUMNS)


def _collapse_duplicate_scores(proximity_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Keep one row per component and marker pair.

    Identical duplicates are collapsed. Conflicting values of ``metric`` for
    the same component and pair are rejected.
    """
    if "component" not in proximity_df.columns:
        return proximity_df
    keys = ["component", "marker_1", "marker_2"]
    if not proximity_df.duplicated(keys).any():
        return proximity_df
    n_values = proximity_df.groupby(keys, sort=False)[metric].nunique(dropna=False)
    if (n_values > 1).any():
        raise ValueError(
            f"proximity_df has conflicting {metric!r} values for the same "
            "component and marker pair."
        )
    return proximity_df.drop_duplicates(keys)


def _one_label_per_component(
    source: pd.DataFrame, contrast_column: str
) -> pd.DataFrame:
    """Return one row per component and its contrast label."""
    if contrast_column == "component":
        return source[["component"]].drop_duplicates()
    labels = source[[contrast_column, "component"]].drop_duplicates()
    conflict = labels["component"].duplicated(keep=False)
    if conflict.any():
        ids = labels.loc[conflict, "component"].unique()[:5].tolist()
        raise ValueError(
            f"A component is assigned to more than one {contrast_column!r}: {ids}."
        )
    return labels


def _metric_on_components(
    group: pd.DataFrame, metric: str, components: np.ndarray
) -> np.ndarray:
    """Align ``metric`` to ``components``, filling components with no row with 0."""
    scores = pd.Series(
        group[metric].to_numpy(),
        index=pd.Index(group["component"].to_numpy()),
    )
    return scores.reindex(components).fillna(0).to_numpy()


def _detection_rate(values: np.ndarray) -> float:
    """Fraction of values that are non-zero, rounded to 3 decimals."""
    if len(values) == 0:
        return 0.0
    return round(float(np.count_nonzero(np.asarray(values) != 0)) / len(values), 3)


def _component_groups(
    proximity_df: pd.DataFrame,
    contrast_column: str,
    cell_groups: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Return one row per cell, with its contrast label.

    ``cell_groups`` supplies cells that have no proximity rows. Otherwise the
    cells are the components present in ``proximity_df``.
    """
    if cell_groups is not None:
        required = (
            {"component"}
            if contrast_column == "component"
            else {
                "component",
                contrast_column,
            }
        )
        missing = required - set(cell_groups.columns)
        if missing:
            raise ValueError(
                f"cell_groups is missing required column(s): {sorted(missing)}."
            )
        if "component" not in proximity_df.columns:
            raise ValueError(
                "proximity_df must include a 'component' column when "
                "cell_groups is provided."
            )
        if contrast_column != "component":
            _require_no_missing_labels(cell_groups, [contrast_column], "cell_groups")
            if contrast_column in proximity_df.columns:
                _require_no_missing_labels(
                    proximity_df, [contrast_column], "proximity_df"
                )
        labels = _one_label_per_component(cell_groups, contrast_column)
        if contrast_column != "component" and contrast_column in proximity_df.columns:
            proximity_labels = _one_label_per_component(proximity_df, contrast_column)
            compared = labels.merge(
                proximity_labels, on="component", suffixes=("_groups", "_proximity")
            )
            left = f"{contrast_column}_groups"
            right = f"{contrast_column}_proximity"
            disagree = compared.loc[compared[left] != compared[right], "component"]
            if not disagree.empty:
                ids = disagree.unique()[:5].tolist()
                raise ValueError(
                    "cell_groups and proximity_df disagree on "
                    f"{contrast_column!r} for component(s): {ids}."
                )
        return labels
    if "component" in proximity_df.columns:
        if contrast_column != "component":
            _require_no_missing_labels(proximity_df, [contrast_column], "proximity_df")
        return _one_label_per_component(proximity_df, contrast_column)
    return None


def _filter_pair_results(
    results_df: pd.DataFrame,
    diff_threshold: float,
    min_pct: float,
    min_diff_pct: float,
) -> pd.DataFrame:
    """Drop pairs that fail the median-difference or detection-rate thresholds."""
    if results_df.empty:
        return results_df
    pct_max = results_df[["pct_ref", "pct_tgt"]].max(axis=1)
    pct_min = results_df[["pct_ref", "pct_tgt"]].min(axis=1)
    keep = (
        (results_df["median_diff"].abs() >= diff_threshold)
        & (pct_max >= min_pct)
        & ((pct_max - pct_min) >= min_diff_pct)
    )
    return results_df.loc[keep].reset_index(drop=True)


def _perform_mannwhitneyu_test(ref_group, tgt_group):
    u_stat, p_value = mannwhitneyu(
        ref_group,
        tgt_group,
        alternative="two-sided",
        use_continuity=True,
        method="asymptotic",
    )
    # Every value tied makes the null variance zero, so the approximation is
    # undefined. The groups do not differ.
    if pd.isna(p_value):
        p_value = 1.0
    auc = u_stat / (len(ref_group) * len(tgt_group))
    tgt_median = np.median(tgt_group)
    ref_median = np.median(ref_group)
    median_diff = tgt_median - ref_median
    return u_stat, p_value, auc, median_diff, tgt_median, ref_median


def _filter_metric_type(proximity_df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
    """Keep self pairs, co-pairs, or every pair."""
    if metric_type == "self":
        return proximity_df[proximity_df["marker_1"] == proximity_df["marker_2"]]
    if metric_type == "co":
        return proximity_df[proximity_df["marker_1"] != proximity_df["marker_2"]]
    return proximity_df


def _grouping_columns(
    group_vars: str | List[str] | None, contrast_column: str
) -> list[str]:
    """Return grouping columns, rejecting the contrast column itself."""
    columns = normalize_input_to_list(group_vars) or []
    if contrast_column in columns:
        raise ValueError(
            f"{contrast_column!r} is the contrast column and cannot be one of "
            "group_vars."
        )
    return columns


def _require_no_missing_labels(
    source: pd.DataFrame, columns: list[str], table_name: str
) -> None:
    """Reject missing values in grouping or contrast columns."""
    missing = source[columns].isna().any(axis=1)
    if not bool(missing.any()):
        return
    if "component" in source.columns:
        ids = source.loc[missing, "component"].unique()[:5].tolist()
        raise ValueError(
            f"{table_name} has missing values in {columns} for component(s): {ids}."
        )
    raise ValueError(f"{table_name} has missing values in {columns}.")


def _require_one_value_per_component(
    source: pd.DataFrame, columns: list[str], table_name: str
) -> None:
    """Reject a component that carries more than one value of ``columns``."""
    if "component" not in source.columns:
        return
    labels = source[["component", *columns]].drop_duplicates()
    conflict = labels["component"].duplicated(keep=False)
    if not conflict.any():
        return
    ids = labels.loc[conflict, "component"].unique()[:5].tolist()
    raise ValueError(
        f"A component has more than one value of {columns} in {table_name}: {ids}."
    )


def _require_group_vars(
    proximity_df: pd.DataFrame, cell_groups, columns: list[str]
) -> None:
    """Require grouping columns, with one value per component."""
    if cell_groups is not None:
        missing = [column for column in columns if column not in cell_groups.columns]
        if missing:
            raise ValueError(f"cell_groups is missing group_vars column(s): {missing}.")
        _require_no_missing_labels(cell_groups, columns, "cell_groups")
        _require_one_value_per_component(cell_groups, columns, "cell_groups")
        if all(column in proximity_df.columns for column in columns):
            _require_no_missing_labels(proximity_df, columns, "proximity_df")
            _require_one_value_per_component(proximity_df, columns, "proximity_df")
            _require_group_labels_agree(proximity_df, cell_groups, columns)
        return
    missing = [column for column in columns if column not in proximity_df.columns]
    if missing:
        raise ValueError(f"proximity_df is missing group_vars column(s): {missing}.")
    _require_no_missing_labels(proximity_df, columns, "proximity_df")
    _require_one_value_per_component(proximity_df, columns, "proximity_df")


def _require_group_labels_agree(
    proximity_df: pd.DataFrame, cell_groups: pd.DataFrame, columns: list[str]
) -> None:
    """Reject a component whose grouping values differ between the two tables."""
    if "component" not in proximity_df.columns:
        return
    compared = (
        cell_groups[["component", *columns]]
        .drop_duplicates()
        .merge(
            proximity_df[["component", *columns]].drop_duplicates(),
            on="component",
            suffixes=("_groups", "_proximity"),
        )
    )
    disagree = pd.Series(False, index=compared.index)
    for column in columns:
        disagree = disagree | (
            compared[f"{column}_groups"] != compared[f"{column}_proximity"]
        )
    if disagree.any():
        ids = compared.loc[disagree, "component"].unique()[:5].tolist()
        raise ValueError(
            f"cell_groups and proximity_df disagree on {columns} for component(s): {ids}."
        )


def _rows_matching(frame: pd.DataFrame, stratum: dict) -> pd.Series:
    """Return rows whose grouping columns equal ``stratum``."""
    mask = pd.Series(True, index=frame.index)
    for column, value in stratum.items():
        mask = mask & frame[column].eq(value)
    return mask


def _iter_group_strata(proximity_df: pd.DataFrame, cell_groups, columns: list[str]):
    """Yield each grouping combination and the cells and rows inside it."""
    source = cell_groups if cell_groups is not None else proximity_df
    keys = source[columns].drop_duplicates().sort_values(columns, kind="mergesort")
    for record in keys.itertuples(index=False, name=None):
        stratum = dict(zip(columns, record))
        if cell_groups is not None:
            cells = cell_groups.loc[_rows_matching(cell_groups, stratum)]
            proximity_part = proximity_df.loc[
                proximity_df["component"].isin(cells["component"])
            ]
            yield stratum, proximity_part, cells
        else:
            yield stratum, proximity_df.loc[_rows_matching(proximity_df, stratum)], None


def _differential_proximity_one_comparison(
    proximity_df: pd.DataFrame,
    contrast_column: str,
    reference: str,
    targets: List[str] | None = None,
    metric: str = "log2_ratio",
    min_cells_per_group: int = 10,
    min_n_obs: int = 0,
    diff_threshold: float = 0.01,
    min_pct: float = 0,
    min_diff_pct: float = float("-inf"),
    cell_groups: pd.DataFrame | None = None,
    stratum: dict | None = None,
    p_adjust_method: Literal[
        "bonferroni", "holm", "hochberg", "hommel", "fdr_bh", "fdr_by", "sidak"
    ] = "bonferroni",
) -> pd.DataFrame:
    """Compare one reference against its targets, with no extra grouping."""
    if contrast_column not in proximity_df.columns:
        raise ValueError(f"{contrast_column} must be a column in the data.")
    if metric not in proximity_df.columns:
        raise ValueError(f"{metric} must be a column in the data.")
    if min_cells_per_group < 0:
        raise ValueError("min_cells_per_group must be a non-negative integer.")

    # Check before ``label != reference``. pd.NA raises TypeError there, and
    # NaN compares as not equal, so it would be collected as its own target.
    _require_no_missing_labels(proximity_df, [contrast_column], "proximity_df")
    if cell_groups is not None and contrast_column in cell_groups.columns:
        _require_no_missing_labels(cell_groups, [contrast_column], "cell_groups")

    if targets is None:
        labels = proximity_df[contrast_column]
        if cell_groups is not None and contrast_column in cell_groups.columns:
            labels = cell_groups[contrast_column]
        targets = [label for label in labels.unique().tolist() if label != reference]

    if proximity_df.empty:
        raise ValueError("No data found for the specified metric type.")

    proximity_df = _collapse_duplicate_scores(proximity_df, metric)
    group_components = _component_groups(proximity_df, contrast_column, cell_groups)
    pad_missing = group_components is not None
    if group_components is not None and min_cells_per_group > 0:
        n_cells = group_components.groupby(contrast_column, sort=False)[
            "component"
        ].nunique()
        n_ref_cells = int(n_cells.get(reference, 0))
        kept_targets = []
        for target in targets:
            n_tgt_cells = int(n_cells.get(target, 0))
            if n_ref_cells < min_cells_per_group or n_tgt_cells < min_cells_per_group:
                where = ""
                if stratum:
                    where = (
                        " ("
                        + ", ".join(
                            f"{name}={value}" for name, value in stratum.items()
                        )
                        + ")"
                    )
                warnings.warn(
                    f"Skipping {target} vs {reference}{where}: a group has fewer "
                    f"than {min_cells_per_group} cells "
                    f"({reference}={n_ref_cells}, {target}={n_tgt_cells}).",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            kept_targets.append(target)
        targets = kept_targets

    def calc_targets_differential():
        """Yield a Mann-Whitney result for each marker pair and target.

        For each target, keep the proximity rows labeled with that target or
        the reference. When ``min_n_obs`` is positive, drop marker pairs that
        do not have more than that many detected rows in both groups. A
        target with no remaining rows is skipped. When component ids are
        available, align ``metric`` to every component in the two groups and
        fill a missing score with 0. Otherwise use the observed scores, and
        skip a pair that is missing from either group. Each remaining pair
        is tested with a two-sided Mann-Whitney U test.

        Yields:
            One Series per tested marker pair and target, with ``marker_1``,
            ``marker_2``, ``reference``, ``target``, ``u_stat``, ``p_value``,
            ``auc``, ``median_diff``, ``tgt_median``, ``ref_median``,
            ``n_ref``, ``n_tgt``, ``pct_ref``, and ``pct_tgt``.
        """
        for target in targets:
            target_data = _filter_target_data(
                proximity_df, contrast_column, reference, target, metric, min_n_obs
            )
            if target_data.empty:
                continue
            if pad_missing:
                assert group_components is not None
                comparison_components = group_components[
                    group_components[contrast_column].isin([reference, target])
                ]
                ref_components = comparison_components.loc[
                    comparison_components[contrast_column].eq(reference), "component"
                ].to_numpy()
                tgt_components = comparison_components.loc[
                    comparison_components[contrast_column].eq(target), "component"
                ].to_numpy()
            else:
                ref_components = None
                tgt_components = None

            for (marker_1, marker_2), group in target_data.groupby(
                ["marker_1", "marker_2"]
            ):
                if pad_missing:
                    ref_group = _metric_on_components(group, metric, ref_components)
                    tgt_group = _metric_on_components(group, metric, tgt_components)
                else:
                    ref_group = group.loc[group[contrast_column] == reference, metric]
                    tgt_group = group.loc[group[contrast_column] == target, metric]

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
                        "pct_ref": _detection_rate(ref_group),
                        "pct_tgt": _detection_rate(tgt_group),
                    }
                )
                yield results

    results_df = pd.DataFrame(list(calc_targets_differential()))
    if results_df.empty:
        return _empty_differential_proximity_result()
    results_df = _filter_pair_results(results_df, diff_threshold, min_pct, min_diff_pct)
    if results_df.empty:
        return _empty_differential_proximity_result()
    results_df["p_adjusted"] = multipletests(
        results_df["p_value"], method=p_adjust_method
    )[1]

    return results_df


def calculate_differential_proximity(
    proximity_df: pd.DataFrame,
    contrast_column: str,
    reference: str,
    targets: List[str] | None = None,
    metric: str = "log2_ratio",
    metric_type: Literal["all", "self", "co"] = "all",
    min_cells_per_group: int = 10,
    min_n_obs: int = 0,
    diff_threshold: float = 0.01,
    min_pct: float = 0,
    min_diff_pct: float = float("-inf"),
    cell_groups: pd.DataFrame | None = None,
    group_vars: str | List[str] | None = None,
    p_adjust_method: Literal[
        "bonferroni", "holm", "hochberg", "hommel", "fdr_bh", "fdr_by", "sidak"
    ] = "bonferroni",
) -> pd.DataFrame:
    """Compare a proximity metric between a reference group and (one or more) target groups.

    Each marker pair is tested with a two-sided
    `Mann-Whitney U test <https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test>`_.
    The p-value is the normal approximation with a continuity correction.
    ``median_diff`` is the target median minus the reference median. ``auc``
    is the fraction of reference-versus-target comparisons won by the
    reference, with ties counting as half.

    When ``proximity_df`` contains ``component``, each tested pair is aligned
    to every component in the two groups and missing values of ``metric`` are
    filled with 0. A marker pair never detected in either group of a comparison
    is left out. Without ``component`` and without ``cell_groups``, only
    observed rows are tested, and a pair missing from one of the two groups is
    skipped.

    ``group_vars`` repeats that comparison inside each combination of those
    columns. Reference and target cells come from the same combination, and
    p-values are adjusted inside it. Each result row carries the combination.

    A comparison is skipped when either group contains fewer than
    ``min_cells_per_group`` components. After the tests, pairs are removed
    when the absolute median difference is below ``diff_threshold``, when the
    larger detection rate is below ``min_pct``, or when the difference
    between the two detection rates is below ``min_diff_pct``. Adjusted
    p-values are computed on the pairs that remain, across every target in
    the comparison.

    Args:
        proximity_df: Long-format proximity table. Must contain
            ``contrast_column``, ``marker_1``, ``marker_2``, and ``metric``.
            ``component`` is optional; when present, missing scores are
            zero-padded.
        contrast_column: Column that assigns each row to a group. ``"component"``
            is allowed, and then each component is its own group.
        reference: Label of the reference group in ``contrast_column``.
        targets: Labels to compare against ``reference``. ``None`` uses every
            other label in ``contrast_column``. Defaults to ``None``.
        metric: Proximity column to test. Defaults to ``"log2_ratio"``.
        metric_type: Which pairs to test. ``"self"`` keeps pairs where
            ``marker_1`` and ``marker_2`` are equal, ``"co"`` keeps the rest,
            and ``"all"`` keeps every pair. Defaults to ``"all"``.
        min_cells_per_group: Minimum number of distinct components in each
            group. A smaller group is skipped with a warning. Components are
            counted in ``cell_groups`` when that table is given. The check is
            skipped when no component id is available. Defaults to 10.
        min_n_obs: Require more than this many detected rows in both groups
            before zero-padding. ``0`` disables the check. Defaults to 0.
        diff_threshold: Minimum absolute median difference. Pairs below it are
            dropped before p-values are adjusted. Defaults to 0.01.
        min_pct: Minimum larger detection rate. Exact zeros count as
            undetected. Defaults to 0.
        min_diff_pct: Minimum difference between the two detection rates.
            Defaults to negative infinity.
        cell_groups: Components to include, with columns ``component`` and
            ``contrast_column`` (``component`` alone when that column is the
            contrast). Required in order to zero-pad a component that has no
            proximity rows. Components missing from ``proximity_df`` contribute
            a 0. A component present in both tables must have the same
            ``contrast_column`` label in both. Defaults to ``None``.
        group_vars: Columns that split the comparison. Both groups in a
            comparison come from the same combination of these columns.
            Defaults to ``None``.
        p_adjust_method: Method forwarded to
            ``statsmodels.stats.multitest.multipletests``. Defaults to
            ``"bonferroni"``.

    Returns:
        One row per marker pair, target, and grouping combination that passed
        the thresholds, with columns ``marker_1``, ``marker_2``, ``reference``,
        ``target``, ``u_stat``, ``p_value``, ``auc``, ``median_diff``,
        ``tgt_median``, ``ref_median``, ``n_ref``, ``n_tgt``, ``pct_ref``,
        ``pct_tgt``, and ``p_adjusted``, plus one column per entry in
        ``group_vars``. ``n_ref`` and ``n_tgt`` include zero-padded components
        when a component id is available, and count observed rows otherwise.
        ``pct_ref`` and ``pct_tgt`` are the rounded detection rates. The same
        columns are present, with no rows, when every comparison is skipped
        or every pair is filtered.

    Raises:
        ValueError: If ``contrast_column`` or ``metric`` is not in
            ``proximity_df``, if ``min_cells_per_group`` is negative, if
            ``cell_groups`` is missing a required column, if ``proximity_df``
            has no ``component`` column when ``cell_groups`` is provided, if a
            component has more than one contrast or grouping label, if a
            contrast or grouping label is missing, if the two tables disagree,
            if ``group_vars`` includes ``contrast_column``,
            if the same component and marker pair has conflicting ``metric``
            values, or if ``metric_type`` removes every row.

    Warns:
        UserWarning: If a comparison is skipped because a group has fewer than
            ``min_cells_per_group`` components. The message names the grouping
            combination when ``group_vars`` is set.

    References:
        * `Bonferroni correction <https://en.wikipedia.org/wiki/Bonferroni_correction>`_
        * `Holm–Bonferroni method <https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method>`_
        * `Hochberg method <https://www.sciencedirect.com/topics/mathematics/hochberg-method>`_
        * `Hommel <https://academic.oup.com/biomet/article-abstract/75/2/383/292949>`_
        * `False discovery rate <https://en.wikipedia.org/wiki/False_discovery_rate>`_
        * `Sidak correction <https://en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction>`_
    """
    if contrast_column not in proximity_df.columns:
        raise ValueError(f"{contrast_column} must be a column in the data.")
    if metric not in proximity_df.columns:
        raise ValueError(f"{metric} must be a column in the data.")
    if min_cells_per_group < 0:
        raise ValueError("min_cells_per_group must be a non-negative integer.")

    columns = _grouping_columns(group_vars, contrast_column)
    if columns:
        _require_group_vars(proximity_df, cell_groups, columns)
    proximity_df = _filter_metric_type(proximity_df, metric_type)
    if proximity_df.empty:
        raise ValueError("No data found for the specified metric type.")
    if not columns:
        return _differential_proximity_one_comparison(
            proximity_df,
            contrast_column,
            reference,
            targets,
            metric=metric,
            min_cells_per_group=min_cells_per_group,
            min_n_obs=min_n_obs,
            diff_threshold=diff_threshold,
            min_pct=min_pct,
            min_diff_pct=min_diff_pct,
            cell_groups=cell_groups,
            p_adjust_method=p_adjust_method,
        )

    frames = []
    for stratum, proximity_part, cells_part in _iter_group_strata(
        proximity_df, cell_groups, columns
    ):
        if proximity_part.empty:
            result = _empty_differential_proximity_result()
        else:
            result = _differential_proximity_one_comparison(
                proximity_part,
                contrast_column,
                reference,
                targets,
                metric=metric,
                min_cells_per_group=min_cells_per_group,
                min_n_obs=min_n_obs,
                diff_threshold=diff_threshold,
                min_pct=min_pct,
                min_diff_pct=min_diff_pct,
                cell_groups=cells_part,
                stratum=stratum,
                p_adjust_method=p_adjust_method,
            )
        for name, value in stratum.items():
            result[name] = value
        if not result.empty:
            frames.append(result)
    if not frames:
        empty = _empty_differential_proximity_result()
        for column in columns:
            empty[column] = pd.Series(dtype="object")
        return empty
    return pd.concat(frames, ignore_index=True)

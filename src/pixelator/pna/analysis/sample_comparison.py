"""Table-level abundance and proximity comparison between two samples.

These functions take data frames. They do not read ``.pxl`` files.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AbundanceComparison:
    """Mean marker CLR values for two samples, and their correlation.

    Attributes:
        abundance: One row per marker, with the mean CLR value in each sample.
        correlation: Pearson correlation between the two samples' mean marker
            CLR values.

    """

    abundance: pd.DataFrame
    correlation: float


@dataclass
class ProximityComparison:
    """Mean marker-pair proximity scores for two samples, and their correlation.

    Attributes:
        proximity: One row per marker pair present in both samples after
            filtering, with the mean log2 ratio and the number of contributing
            components in each sample.
        correlation: Pearson correlation between the two samples' mean
            proximity log2 ratios.

    """

    proximity: pd.DataFrame
    correlation: float


def _candidate_markers(
    clr_a: pd.DataFrame,
    clr_b: pd.DataFrame,
    markers: set[str] | None,
) -> set[str]:
    if markers is not None:
        return markers
    return set(clr_a.columns) & set(clr_b.columns)


def _expressed_markers(
    clr_a: pd.DataFrame,
    clr_b: pd.DataFrame,
    markers: set[str] | None,
    min_mean_clr: float,
) -> set[str]:
    """Return markers whose mean CLR exceeds ``min_mean_clr`` in both frames."""
    candidate_markers = _candidate_markers(clr_a, clr_b, markers)
    markers_a = {m for m in candidate_markers if clr_a[m].mean() > min_mean_clr}
    markers_b = {m for m in candidate_markers if clr_b[m].mean() > min_mean_clr}
    return markers_a & markers_b


def _require_distinct_names(name_a: str, name_b: str) -> None:
    if name_a == name_b:
        raise ValueError(
            f"name_a and name_b must be different, got '{name_a}' for both samples."
        )


def _summarize_abundance(clr: pd.DataFrame, markers: set[str]) -> pd.Series:
    return clr[sorted(markers)].mean(axis=0)


def _summarize_proximity(
    proximity_df: pd.DataFrame,
    min_expected_join_count: int,
    min_n_cells: int,
) -> pd.DataFrame:
    """Summarize proximity scores by marker pair.

    Drops marker pairs whose expected join count is below
    ``min_expected_join_count``, then keeps pairs supported by at least
    ``min_n_cells`` components.
    """
    filtered = proximity_df[
        proximity_df["join_count_expected_mean"] >= min_expected_join_count
    ]

    summary = filtered.groupby(["marker_1", "marker_2"]).agg(
        mean_log2_ratio=("log2_ratio", "mean"), n_cells=("log2_ratio", "size")
    )
    summary = summary[summary["n_cells"] >= min_n_cells].reset_index()
    return summary


def compare_abundance(
    clr_a: pd.DataFrame,
    clr_b: pd.DataFrame,
    *,
    name_a: str = "a",
    name_b: str = "b",
    markers: set[str] | None = None,
) -> AbundanceComparison:
    """Compare mean marker CLR values between two samples.

    Args:
        clr_a: Component-by-marker CLR values for the first sample, for
            example ``dataset.adata().obsm["clr"]``.
        clr_b: Component-by-marker CLR values for the second sample.
        name_a: Name used to label the first sample's mean-CLR column.
            Defaults to ``"a"``.
        name_b: Name used to label the second sample's mean-CLR column.
            Defaults to ``"b"``.
        markers: Markers to include. Defaults to the markers present in both
            frames.

    Returns:
        Mean CLR per marker in each sample, and the Pearson correlation
        between those means.

    Raises:
        ValueError: If ``name_a`` and ``name_b`` are the same, or if the
            correlation is undefined.

    """
    _require_distinct_names(name_a, name_b)
    candidate_markers = _candidate_markers(clr_a, clr_b, markers)
    ordered_markers = sorted(candidate_markers)

    abundance_a = _summarize_abundance(clr_a, candidate_markers)
    abundance_b = _summarize_abundance(clr_b, candidate_markers)
    abundance = pd.DataFrame(
        {
            "marker": ordered_markers,
            f"mean_clr_{name_a}": abundance_a.loc[ordered_markers].values,
            f"mean_clr_{name_b}": abundance_b.loc[ordered_markers].values,
        }
    )
    correlation = float(
        abundance[f"mean_clr_{name_a}"].corr(abundance[f"mean_clr_{name_b}"])
    )
    if pd.isna(correlation):
        raise ValueError(
            "Abundance correlation is undefined. "
            "(need >=2 markers with non-constant values)."
        )
    return AbundanceComparison(abundance=abundance, correlation=correlation)


def compare_proximity(
    prox_a: pd.DataFrame,
    prox_b: pd.DataFrame,
    *,
    name_a: str = "a",
    name_b: str = "b",
    min_expected_join_count: int = 10,
    min_n_cells: int = 50,
) -> ProximityComparison:
    """Compare mean marker-pair proximity log2 ratios between two samples.

    Each frame has one row per component and marker pair, as returned by
    ``Proximity.to_df()``. Marker pairs are kept when their expected join
    count is at least ``min_expected_join_count`` and they are supported by
    at least ``min_n_cells`` components. Only pairs that pass in both samples
    are compared.

    Args:
        prox_a: Proximity scores for the first sample.
        prox_b: Proximity scores for the second sample.
        name_a: Name used to label the first sample's columns. Defaults to
            ``"a"``.
        name_b: Name used to label the second sample's columns. Defaults to
            ``"b"``.
        min_expected_join_count: Minimum expected join count for a marker
            pair to be included. Defaults to 10.
        min_n_cells: Minimum number of components required for a marker pair
            to be included. Defaults to 50.

    Returns:
        Mean proximity log2 ratio and component count per marker pair, and
        the Pearson correlation between the two samples' mean log2 ratios.

    Raises:
        ValueError: If ``name_a`` and ``name_b`` are the same, if no marker
            pair passes the filters in both samples, or if the correlation
            is undefined.

    """
    _require_distinct_names(name_a, name_b)

    summary_a = _summarize_proximity(
        prox_a, min_expected_join_count, min_n_cells
    ).rename(
        columns={
            "mean_log2_ratio": f"log2_ratio_{name_a}",
            "n_cells": f"n_cells_{name_a}",
        }
    )
    summary_b = _summarize_proximity(
        prox_b, min_expected_join_count, min_n_cells
    ).rename(
        columns={
            "mean_log2_ratio": f"log2_ratio_{name_b}",
            "n_cells": f"n_cells_{name_b}",
        }
    )

    proximity = summary_a.merge(summary_b, on=["marker_1", "marker_2"], how="inner")

    if proximity.empty:
        raise ValueError(
            "No marker pairs passed the proximity filtering criteria in both "
            "samples. Try lowering min_expected_join_count or min_n_cells."
        )

    correlation = float(
        proximity[f"log2_ratio_{name_a}"].corr(proximity[f"log2_ratio_{name_b}"])
    )
    if pd.isna(correlation):
        raise ValueError(
            "Proximity correlation is undefined. "
            "(need >=2 marker pairs with non-constant values). "
            "Try lowering min_expected_join_count or min_n_cells."
        )
    return ProximityComparison(proximity=proximity, correlation=correlation)

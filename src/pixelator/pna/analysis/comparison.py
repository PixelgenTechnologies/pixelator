"""Functions for comparing abundance and proximity similarity between sample pairs.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from pixelator.common.utils import logger
from pixelator.pna.analysis.gating import MarkerThreshold, gate_mask
from pixelator.pna.pixeldataset import PNAPixelDataset, read


@dataclass
class SamplePairComparisonResult:
    """The result of comparing abundance and proximity similarity between a pair of samples.

    Attributes:
        sample1_name: The name of the first sample.
        sample2_name: The name of the second sample.
        abundance: A DataFrame with one row per marker, containing the mean CLR
            value of that marker in each of the two samples.
        abundance_correlation: The Pearson correlation between the two samples'
            mean marker CLR values.
        proximity: A DataFrame with one row per marker pair, containing the mean
            proximity log2 ratio and number of contributing components for each
            of the two samples.
        proximity_correlation: The Pearson correlation between the two samples'
            mean proximity log2 ratios.
        gate: The cell-type gate applied to both samples before comparison, if
            any, e.g. ``["+CD3e", "+CD4", "-CD19"]``.

    """

    sample1_name: str
    sample2_name: str
    abundance: pd.DataFrame
    abundance_correlation: float
    proximity: pd.DataFrame
    proximity_correlation: float
    gate: list[str] | None = field(default=None)


def _load_dataset(pxl) -> PNAPixelDataset:
    """Load a PNAPixelDataset from a pxl file path (or pass one through unchanged)."""
    if isinstance(pxl, PNAPixelDataset):
        return pxl
    return read(pxl)


def _resolve_sample_name(dataset: PNAPixelDataset, provided_name: str | None) -> str:
    if provided_name is not None:
        return provided_name

    sample_names = dataset.sample_names()
    if len(sample_names) != 1:
        raise ValueError(
            "Could not infer a unique sample name from the dataset "
            f"(found {sorted(sample_names)}). Please provide a sample name explicitly."
        )
    return next(iter(sample_names))


def _summarize_abundance(clr: pd.DataFrame, markers: set[str]) -> pd.Series:
    return clr[sorted(markers)].mean(axis=0)


def _summarize_proximity(
    proximity_df: pd.DataFrame,
    min_expected_join_count: int,
    min_n_cells: int,
) -> pd.DataFrame:
    """Summarize the proximity data by marker pairs.

    Filters out marker pairs with a low expected join count, and only keeps
    marker pairs supported by at least ``min_n_cells`` components.
    """
    filtered = proximity_df[
        proximity_df["join_count_expected_mean"] >= min_expected_join_count
    ]

    summary = filtered.groupby(["marker_1", "marker_2"]).agg(
        mean_log2_ratio=("log2_ratio", "mean"), n_cells=("log2_ratio", "size")
    )
    summary = summary[summary["n_cells"] >= min_n_cells].reset_index()
    return summary


def _expressed_markers(
    clr1: pd.DataFrame,
    clr2: pd.DataFrame,
    candidate_markers: set[str],
    min_mean_clr: float,
) -> set[str]:
    markers_1 = {m for m in candidate_markers if clr1[m].mean() > min_mean_clr}
    markers_2 = {m for m in candidate_markers if clr2[m].mean() > min_mean_clr}
    return markers_1 & markers_2


def compare_sample_pair(
    pxl1,
    pxl2,
    sample1_name: str | None = None,
    sample2_name: str | None = None,
    markers: set[str] | None = None,
    min_mean_clr: float = 1.5,
    min_expected_join_count: int = 10,
    min_n_cells: int = 50,
) -> SamplePairComparisonResult:
    """Compare abundance and proximity similarity between a single pair of samples.

    Markers are first restricted to those expressed (mean CLR above
    ``min_mean_clr``) in both samples. Mean marker CLR values (abundance) and
    mean marker-pair proximity log2 ratios are then compared between the two
    samples, and the Pearson correlation is computed for each.

    Args:
        pxl1: The first sample, given as a `PNAPixelDataset`, a path to a
            ``.pxl`` file, or anything accepted by `pixelator.pna.read`.
        pxl2: The second sample, in the same forms as ``pxl1``.
        sample1_name: A name for the first sample, used for labeling the
            output. If not provided, it is inferred from the dataset.
        sample2_name: A name for the second sample, used for labeling the
            output. If not provided, it is inferred from the dataset.
        markers: If provided, restrict the comparison to this set of markers
            (still subject to the ``min_mean_clr`` filter). Defaults to all
            markers shared by the two datasets.
        min_mean_clr: The minimum mean CLR value (in both samples) for a
            marker to be included in the comparison. Defaults to 1.5.
        min_expected_join_count: The minimum expected join count for a marker
            pair to be included in the proximity comparison. Defaults to 10.
        min_n_cells: The minimum number of components required for a marker
            pair to be included in the proximity comparison. Defaults to 50.

    Returns:
        A `SamplePairComparisonResult` with the abundance and proximity
        comparison data and their correlations.

    """
    dataset1 = _load_dataset(pxl1)
    dataset2 = _load_dataset(pxl2)

    sample1_name = _resolve_sample_name(dataset1, sample1_name)
    sample2_name = _resolve_sample_name(dataset2, sample2_name)

    if sample1_name == sample2_name:
        raise ValueError(
            "sample1_name and sample2_name must be different, got "
            f"'{sample1_name}' for both samples. Pass explicit, distinct names."
        )

    clr1 = dataset1.adata().obsm["clr"]
    clr2 = dataset2.adata().obsm["clr"]

    candidate_markers = (
        markers if markers is not None else set(clr1.columns) & set(clr2.columns)
    )

    expressed_markers = _expressed_markers(clr1, clr2, candidate_markers, min_mean_clr)

    if not expressed_markers:
        raise ValueError(
            "No markers passed the min_mean_clr filter in both samples. "
            "Try lowering min_mean_clr."
        )

    abundance_1 = _summarize_abundance(clr1, candidate_markers)
    abundance_2 = _summarize_abundance(clr2, candidate_markers)
    abundance = pd.DataFrame(
        {
            "marker": sorted(candidate_markers),
            f"mean_clr_{sample1_name}": abundance_1.loc[
                sorted(candidate_markers)
            ].values,
            f"mean_clr_{sample2_name}": abundance_2.loc[
                sorted(candidate_markers)
            ].values,
        }
    )
    abundance_correlation = float(
        abundance[f"mean_clr_{sample1_name}"].corr(
            abundance[f"mean_clr_{sample2_name}"]
        )
    )
    if pd.isna(abundance_correlation):
        raise ValueError(
            "Abundance correlation is undefined. "
            "(need >=2 markers with non-constant values)."
        )

    proximity_1 = dataset1.filter(markers=expressed_markers).proximity().to_df()
    proximity_2 = dataset2.filter(markers=expressed_markers).proximity().to_df()

    summary_1 = _summarize_proximity(
        proximity_1, min_expected_join_count, min_n_cells
    ).rename(
        columns={
            "mean_log2_ratio": f"log2_ratio_{sample1_name}",
            "n_cells": f"n_cells_{sample1_name}",
        }
    )
    summary_2 = _summarize_proximity(
        proximity_2, min_expected_join_count, min_n_cells
    ).rename(
        columns={
            "mean_log2_ratio": f"log2_ratio_{sample2_name}",
            "n_cells": f"n_cells_{sample2_name}",
        }
    )

    proximity = summary_1.merge(summary_2, on=["marker_1", "marker_2"], how="inner")

    if proximity.empty:
        raise ValueError(
            "No marker pairs passed the proximity filtering criteria in both "
            "samples. Try lowering min_expected_join_count or min_n_cells."
        )

    proximity_correlation = float(
        proximity[f"log2_ratio_{sample1_name}"].corr(
            proximity[f"log2_ratio_{sample2_name}"]
        )
    )
    if pd.isna(proximity_correlation):
        raise ValueError(
            "Proximity correlation is undefined. "
            "(need >=2 marker pairs with non-constant values). "
            "Try lowering min_expected_join_count or min_n_cells."
        )

    return SamplePairComparisonResult(
        sample1_name=sample1_name,
        sample2_name=sample2_name,
        abundance=abundance,
        abundance_correlation=abundance_correlation,
        proximity=proximity,
        proximity_correlation=proximity_correlation,
    )


def compare_sample_pairs(
    pairs: Sequence[tuple],
    sample_names: Sequence[tuple[str, str]] | None = None,
    markers: set[str] | None = None,
    min_mean_clr: float = 1.5,
    min_expected_join_count: int = 10,
    min_n_cells: int = 50,
) -> list[SamplePairComparisonResult]:
    """Compare abundance and proximity similarity across a list of sample pairs.

    This is the main entry point for checking that pairs of samples are
    similar in terms of abundance and proximity patterns. For each pair, mean
    marker abundances (CLR) and mean marker-pair proximity scores are compared,
    and their correlations are reported. See `compare_sample_pair` for details
    on how a single pair is compared.

    Args:
        pairs: A list of ``(pxl1, pxl2)`` pairs, where each element is a
            `PNAPixelDataset`, a path to a ``.pxl`` file, or anything accepted
            by `pixelator.pna.read`.
        sample_names: An optional list of ``(sample1_name, sample2_name)``
            tuples, parallel to ``pairs``, used for labeling the output. If
            not provided, sample names are inferred from each dataset.
        markers: If provided, restrict the comparison to this set of markers
            for all pairs (still subject to the ``min_mean_clr`` filter).
        min_mean_clr: The minimum mean CLR value (in both samples) for a
            marker to be included in the comparison. Defaults to 1.5.
        min_expected_join_count: The minimum expected join count for a marker
            pair to be included in the proximity comparison. Defaults to 10.
        min_n_cells: The minimum number of components required for a marker
            pair to be included in the proximity comparison. Defaults to 50.

    Returns:
        A list of `SamplePairComparisonResult`, one per pair in ``pairs``.

    """
    if sample_names is not None and len(sample_names) != len(pairs):
        raise ValueError("sample_names must have the same length as pairs.")

    results = []
    for i, (pxl1, pxl2) in enumerate(pairs):
        sample1_name, sample2_name = sample_names[i] if sample_names else (None, None)
        logger.info("Comparing sample pair %d/%d", i + 1, len(pairs))
        results.append(
            compare_sample_pair(
                pxl1,
                pxl2,
                sample1_name=sample1_name,
                sample2_name=sample2_name,
                markers=markers,
                min_mean_clr=min_mean_clr,
                min_expected_join_count=min_expected_join_count,
                min_n_cells=min_n_cells,
            )
        )
    return results


def compare_sample_pairs_by_gate(
    pairs: Sequence[tuple],
    gate: list[str],
    sample_names: Sequence[tuple[str, str]] | None = None,
    min_separation_score: float = 3.0,
    markers: set[str] | None = None,
    min_mean_clr: float = 1.5,
    min_expected_join_count: int = 10,
    min_n_cells: int = 50,
) -> list[SamplePairComparisonResult]:
    """Compare abundance and proximity similarity across sample pairs, restricted to a cell-type gate.

    Since components in a `PNAPixelDataset` don't have cell types assigned,
    this wrapper first filters each pair of samples to the components matching
    a gating specification, e.g. ``["+CD3e", "+CD4", "-CD19"]``, before running
    the same comparison as `compare_sample_pairs`.

    For each pair, the positive/negative threshold for every marker in
    ``gate`` is determined from the pooled CLR distribution of both samples in
    that pair (so that both samples are gated using the exact same threshold).
    If a marker's distribution appears unimodal, a warning is issued and that
    marker is ignored when gating (see `pixelator.pna.analysis.gating.determine_marker_threshold`).

    Args:
        pairs: A list of ``(pxl1, pxl2)`` pairs, where each element is a
            `PNAPixelDataset`, a path to a ``.pxl`` file, or anything accepted
            by `pixelator.pna.read`.
        gate: A list of gating strings, e.g. ``["+CD3e", "+CD4", "-CD19"]``.
        sample_names: An optional list of ``(sample1_name, sample2_name)``
            tuples, parallel to ``pairs``, used for labeling the output.
        min_separation_score: The minimum separation score required for a
            gating marker's distribution to be considered bimodal. Defaults to 3.0.
        markers: If provided, restrict the abundance/proximity comparison to
            this set of markers for all pairs.
        min_mean_clr: The minimum mean CLR value (in both samples) for a
            marker to be included in the comparison. Defaults to 1.5.
        min_expected_join_count: The minimum expected join count for a marker
            pair to be included in the proximity comparison. Defaults to 10.
        min_n_cells: The minimum number of components required for a marker
            pair to be included in the proximity comparison. Defaults to 50.

    Returns:
        A list of `SamplePairComparisonResult`, one per pair in ``pairs``, each
        annotated with the ``gate`` that was applied.

    """
    if sample_names is not None and len(sample_names) != len(pairs):
        raise ValueError("sample_names must have the same length as pairs.")

    results = []
    for i, (pxl1, pxl2) in enumerate(pairs):
        sample1_name, sample2_name = sample_names[i] if sample_names else (None, None)

        dataset1 = _load_dataset(pxl1)
        dataset2 = _load_dataset(pxl2)

        sample1_name = _resolve_sample_name(dataset1, sample1_name)
        sample2_name = _resolve_sample_name(dataset2, sample2_name)

        clr1 = dataset1.adata().obsm["clr"]
        clr2 = dataset2.adata().obsm["clr"]

        pooled_clr = pd.concat([clr1, clr2], axis=0)
        pooled_mask, thresholds = gate_mask(
            pooled_clr, gate, min_separation_score=min_separation_score
        )
        _log_gate_thresholds(thresholds, sample1_name, sample2_name)

        mask1 = pooled_mask.loc[clr1.index]
        mask2 = pooled_mask.loc[clr2.index]

        if not mask1.any():
            raise ValueError(
                f"No components in sample '{sample1_name}' pass the gate {gate}."
            )
        if not mask2.any():
            raise ValueError(
                f"No components in sample '{sample2_name}' pass the gate {gate}."
            )

        gated_dataset1 = dataset1.filter(components=mask1[mask1].index)
        gated_dataset2 = dataset2.filter(components=mask2[mask2].index)

        logger.info(
            "Gated sample pair %d/%d (%s vs %s): %d vs %d components pass the gate %s",
            i + 1,
            len(pairs),
            sample1_name,
            sample2_name,
            int(mask1.sum()),
            int(mask2.sum()),
            gate,
        )

        result = compare_sample_pair(
            gated_dataset1,
            gated_dataset2,
            sample1_name=sample1_name,
            sample2_name=sample2_name,
            markers=markers,
            min_mean_clr=min_mean_clr,
            min_expected_join_count=min_expected_join_count,
            min_n_cells=min_n_cells,
        )
        result.gate = list(gate)
        results.append(result)

    return results


def _log_gate_thresholds(
    thresholds: list[MarkerThreshold], sample1_name: str, sample2_name: str
) -> None:
    for t in thresholds:
        if t.threshold is not None:
            logger.debug(
                "Gate threshold for marker '%s' (pooled %s/%s): %.3f "
                "(separation score %.3f)",
                t.marker,
                sample1_name,
                sample2_name,
                t.threshold,
                t.separation_score,
            )

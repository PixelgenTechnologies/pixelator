"""Tests for the pna sample pair comparison module.

Copyright © 2026 Pixelgen Technologies AB.
"""

from io import StringIO

import pandas as pd
import polars as pl
import pytest

from pixelator.pna.analysis.comparison import (
    SamplePairComparisonResult,
    compare_sample_pair,
    compare_sample_pairs,
    compare_sample_pairs_by_gate,
)
from pixelator.pna.pixeldataset import PNAPixelDataset
from tests.pna.conftest import create_pxl_file
from tests.pna.data.pxl_data import EDGELIST_DATA, PROXIMITY_DATA

# The component ids used by the shared EDGELIST_DATA/PROXIMITY_DATA fixtures,
# remapped to a distinct set of ids so a second, independent sample can be
# built from otherwise identical data (real samples never share component ids).
_SAMPLE2_ID_MAP = {
    "fc07dea9b679aca7": "fc07dea9b679acb8",
    "e7d82bca9694eea7": "e7d82bca9694eeb8",
    "4920229146151c29": "4920229146151c38",
    "3770519d30f36d18": "3770519d30f36d28",
}

_EDGELIST_SCHEMA = {
    "umi1": pl.UInt64,
    "umi2": pl.UInt64,
    "read_count": pl.UInt32,
    "uei_count": pl.UInt32,
    "marker_1": pl.Utf8,
    "marker_2": pl.Utf8,
    "component": pl.Utf8,
}


def _remap_component_ids(csv_text: str) -> str:
    for old, new in _SAMPLE2_ID_MAP.items():
        csv_text = csv_text.replace(old, new)
    return csv_text


@pytest.fixture(name="pxl_file_2", scope="module")
def pxl_file_2_fixture(tmp_path_factory, panel):
    """A second, independent pxl file with the same panel/marker structure as `pxl_file`."""
    edgelist_df = pl.read_csv(
        StringIO(_remap_component_ids(EDGELIST_DATA)), schema=_EDGELIST_SCHEMA
    )
    proximity_df = pl.read_csv(StringIO(_remap_component_ids(PROXIMITY_DATA)))

    tmp_dir = tmp_path_factory.mktemp("data2")
    edgelist_path = tmp_dir / "edgelist2.parquet"
    proximity_path = tmp_dir / "proximity2.parquet"
    edgelist_df.write_parquet(edgelist_path)
    proximity_df.write_parquet(proximity_path)

    target = tmp_dir / "file2.pxl"
    return create_pxl_file(
        target=target,
        sample_name="test_sample_2",
        edgelist_parquet_path=edgelist_path,
        proximity_parquet_path=proximity_path,
        layout_parquet_path=None,
        panel=panel,
    )


@pytest.fixture(name="pxl_dataset_2")
def pxl_dataset_2_fixture(pxl_file_2):
    """A `PNAPixelDataset` for the second sample."""
    return PNAPixelDataset.from_pxl_files(pxl_file_2)


@pytest.fixture(name="lenient_kwargs")
def lenient_kwargs_fixture():
    """Filtering kwargs lenient enough for the tiny test datasets to produce data."""
    return {
        "min_mean_clr": -100.0,
        "min_expected_join_count": 0,
        "min_n_cells": 0,
    }


def test_compare_sample_pair(pxl_dataset, pxl_dataset_2, lenient_kwargs):
    """Verify a basic pairwise comparison between two (identical, relabeled) samples."""
    result = compare_sample_pair(
        pxl_dataset,
        pxl_dataset_2,
        sample1_name="sample1",
        sample2_name="sample2",
        **lenient_kwargs,
    )

    assert isinstance(result, SamplePairComparisonResult)
    assert result.gate is None

    assert set(result.abundance.columns) == {
        "marker",
        "mean_clr_sample1",
        "mean_clr_sample2",
    }
    assert not result.abundance.empty

    assert not result.proximity.empty
    assert {"marker_1", "marker_2"}.issubset(result.proximity.columns)

    # sample1 and sample2 are built from identical underlying data (just
    # relabeled component ids), so the two samples should be perfectly
    # correlated in both abundance and proximity.
    assert result.abundance_correlation == pytest.approx(1.0)
    assert result.proximity_correlation == pytest.approx(1.0)


def test_compare_sample_pair_requires_distinct_sample_names(
    pxl_dataset, pxl_dataset_2, lenient_kwargs
):
    """Verify comparing two samples with the same name raises a ValueError."""
    with pytest.raises(ValueError, match="must be different"):
        compare_sample_pair(
            pxl_dataset,
            pxl_dataset_2,
            sample1_name="same_name",
            sample2_name="same_name",
            **lenient_kwargs,
        )


def test_compare_sample_pair_no_markers_pass_filter(pxl_dataset, pxl_dataset_2):
    """Verify an overly strict min_mean_clr raises an informative ValueError."""
    with pytest.raises(ValueError, match="No markers passed"):
        compare_sample_pair(
            pxl_dataset,
            pxl_dataset_2,
            sample1_name="sample1",
            sample2_name="sample2",
            min_mean_clr=1e6,
        )


def test_compare_sample_pair_no_marker_pairs_pass_filter(
    pxl_dataset, pxl_dataset_2, lenient_kwargs
):
    """Verify an overly strict proximity filter raises an informative ValueError."""
    kwargs = dict(lenient_kwargs)
    kwargs["min_expected_join_count"] = 1e6
    with pytest.raises(ValueError, match="No marker pairs passed"):
        compare_sample_pair(
            pxl_dataset,
            pxl_dataset_2,
            sample1_name="sample1",
            sample2_name="sample2",
            **kwargs,
        )


def test_compare_sample_pairs(pxl_dataset, pxl_dataset_2, lenient_kwargs):
    """Verify the multi-pair wrapper delegates to compare_sample_pair per pair."""
    results = compare_sample_pairs(
        [(pxl_dataset, pxl_dataset_2)],
        sample_names=[("sample1", "sample2")],
        **lenient_kwargs,
    )

    assert len(results) == 1
    assert results[0].sample1_name == "sample1"
    assert results[0].sample2_name == "sample2"


def test_compare_sample_pairs_sample_names_length_mismatch(
    pxl_dataset, pxl_dataset_2, lenient_kwargs
):
    """Verify a mismatched sample_names length raises a ValueError."""
    with pytest.raises(ValueError, match="same length"):
        compare_sample_pairs(
            [(pxl_dataset, pxl_dataset_2)],
            sample_names=[("sample1", "sample2"), ("sample3", "sample4")],
            **lenient_kwargs,
        )


def test_compare_sample_pairs_by_gate_unimodal_marker_is_noop(
    pxl_dataset, pxl_dataset_2, lenient_kwargs
):
    """Verify a gate on a marker that isn't clearly bimodal enough doesn't filter anything."""
    gate = ["+MarkerA"]

    results = compare_sample_pairs_by_gate(
        [(pxl_dataset, pxl_dataset_2)],
        gate=gate,
        sample_names=[("sample1", "sample2")],
        min_separation_score=1e6,  # nothing can be this well separated -> gate is a no-op
        **lenient_kwargs,
    )

    assert len(results) == 1
    result = results[0]
    assert result.gate == gate
    assert not result.abundance.empty
    assert not result.proximity.empty


def test_compare_sample_pairs_by_gate_filters_components(
    pxl_dataset, pxl_dataset_2, lenient_kwargs, monkeypatch
):
    """Verify the wrapper applies a pooled gate mask per-sample before comparing."""
    from pixelator.pna.analysis import comparison as comparison_module

    # The only components with proximity data available in the tiny test fixtures.
    kept_components = {"fc07dea9b679aca7", "fc07dea9b679acb8"}

    def fake_gate_mask(pooled_clr, gate, min_separation_score):
        mask = pd.Series(
            [idx in kept_components for idx in pooled_clr.index],
            index=pooled_clr.index,
        )
        return mask, []

    monkeypatch.setattr(comparison_module, "gate_mask", fake_gate_mask)

    results = compare_sample_pairs_by_gate(
        [(pxl_dataset, pxl_dataset_2)],
        gate=["+MarkerA"],
        sample_names=[("sample1", "sample2")],
        **lenient_kwargs,
    )

    result = results[0]
    assert result.gate == ["+MarkerA"]
    assert not result.abundance.empty
    assert not result.proximity.empty


def test_compare_sample_pairs_by_gate_no_components_pass(
    pxl_dataset, pxl_dataset_2, lenient_kwargs
):
    """Verify a gate that excludes all components raises rather than silently comparing unfiltered data."""
    gate = ["+MarkerA", "-MarkerA"]  # impossible to satisfy both at once

    with pytest.raises(ValueError, match="No components"):
        compare_sample_pairs_by_gate(
            [(pxl_dataset, pxl_dataset_2)],
            gate=gate,
            sample_names=[("sample1", "sample2")],
            min_separation_score=-1e6,
            **lenient_kwargs,
        )

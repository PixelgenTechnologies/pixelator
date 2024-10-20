"""Test functions related to data collection for the QC report.

Copyright © 2023 Pixelgen Technologies AB.
"""

import json

import pytest

from pixelator.report.common.json_encoder import PixelatorJSONEncoder
from pixelator.report.qcreport.collect import (
    collect_antibody_counts_data,
    collect_antibody_percentages_data,
    collect_component_ranked_component_size_data,
    collect_components_umap_data,
    collect_reads_per_molecule_frequency,
    collect_report_data,
)


def test_collect_reads_per_molecule_frequency(filtered_dataset_pxl_data, snapshot):
    csv_data = collect_reads_per_molecule_frequency(filtered_dataset_pxl_data)
    snapshot.assert_match(csv_data, "test_collect_reads_per_molecule_frequency.csv")


def test_component_ranked_component_size_data(raw_component_metrics_data, snapshot):
    csv_data = collect_component_ranked_component_size_data(
        raw_component_metrics_data, subsample_non_cell_components=False
    )
    snapshot.assert_match(csv_data, "ranked_component_size.csv")


def test_component_ranked_component_size_data_subsampled(
    raw_component_metrics_data, snapshot
):
    raw_component_metrics_data.loc[
        raw_component_metrics_data["molecules"] <= 1, "is_filtered"
    ] = False
    csv_data = collect_component_ranked_component_size_data(
        raw_component_metrics_data, subsample_non_cell_components=True
    )
    snapshot.assert_match(csv_data, "ranked_component_size_subsampled.csv")


def test_components_umap_data(filtered_dataset_pxl_data, snapshot):
    csv_data = collect_components_umap_data(filtered_dataset_pxl_data.adata)
    snapshot.assert_match(csv_data, "components_umap.csv")


def test_antibody_percentages_data(filtered_dataset_pxl_data, snapshot):
    csv_data = collect_antibody_percentages_data(filtered_dataset_pxl_data.adata)
    snapshot.assert_match(csv_data, "antibody_percentages.csv")


def test_antibody_counts_data(filtered_dataset_pxl_data, snapshot):
    csv_data = collect_antibody_counts_data(filtered_dataset_pxl_data.adata)
    snapshot.assert_match(csv_data, "antibody_counts.csv")


@pytest.fixture
def collect_report_data_workdir(
    pixelator_workdir, filtered_dataset_pxl_workdir, raw_component_metrics_workdir
):
    return pixelator_workdir


@pytest.mark.parametrize("sample_id", ["uropod_control"])
def test_collect_report_data(collect_report_data_workdir, sample_id, snapshot):
    data = collect_report_data(collect_report_data_workdir, sample_id)
    serialized_data = json.dumps(data, cls=PixelatorJSONEncoder)
    snapshot.assert_match(serialized_data, f"{sample_id}/combined_report_data.json")

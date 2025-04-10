"""Copyright © 2025 Pixelgen Technologies AB."""

import json

import pytest

from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.report.common import PixelatorPNAReporting
from pixelator.pna.report.qcreport.collect import (
    collect_antibody_counts_data,
    collect_antibody_percentages_data,
    collect_metrics_report_data,
    collect_proximity_data,
    collect_ranked_component_size_data,
    collect_report_data,
)


@pytest.mark.parametrize("sample_id", ["PNA055_Sample07_filtered_S7"])
def test_collect_metrics_report_data(
    sample_id, all_stages_all_reports_and_meta, snapshot
):
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)
    report_data = collect_metrics_report_data(reporting, sample_id)
    snapshot.assert_match(json.dumps(report_data, indent=4), f"{sample_id}.json")


@pytest.mark.parametrize("sample_id", ["PNA055_Sample07_filtered_S7"])
def test_antibody_percentages_data(
    sample_id, all_stages_all_reports_and_meta, snapshot
):
    dataset = PNAPixelDataset.from_files(
        all_stages_all_reports_and_meta.filtered_dataset(sample_id)
    )
    csv_data = collect_antibody_percentages_data(dataset.adata())
    snapshot.assert_match(csv_data, "0-antibody-percentages.csv")


@pytest.mark.parametrize("sample_id", ["PNA055_Sample07_filtered_S7"])
def test_antibody_counts_data(sample_id, all_stages_all_reports_and_meta, snapshot):
    dataset = PNAPixelDataset.from_files(
        all_stages_all_reports_and_meta.filtered_dataset(sample_id)
    )
    csv_data = collect_antibody_counts_data(dataset.adata())
    snapshot.assert_match(csv_data, "0-antibody-counts.csv")


@pytest.mark.parametrize("sample_id", ["PNA055_Sample07_filtered_S7"])
def test_collect_ranked_component_size_data(
    sample_id, all_stages_all_reports_and_meta, snapshot
):
    dataset = PNAPixelDataset.from_files(
        all_stages_all_reports_and_meta.filtered_dataset(sample_id)
    )
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)
    graph_report = reporting.graph_metrics(sample_id)

    csv_data = collect_ranked_component_size_data(dataset.adata(), graph_report)
    snapshot.assert_match(csv_data, "0-ranked-component-size.csv")


@pytest.mark.parametrize("sample_id", ["PNA055_Sample07_filtered_S7"])
@pytest.mark.slow
def test_collect_report_data(sample_id, all_stages_all_reports_and_meta, snapshot):
    data = collect_report_data(
        PixelatorPNAReporting(all_stages_all_reports_and_meta), sample_id
    )
    data_dict = data.to_dict()
    component_data = data_dict.pop("component_data")
    assert component_data.startswith(
        "component,umap1,umap2,cluster,reads_in_component,n_antibodies,n_umi,n_edges\n"
    )
    snapshot.assert_match(json.dumps(data_dict, indent=4), f"0-metrics.json")


@pytest.mark.parametrize("sample_id", ["PNA055_Sample07_filtered_S7"])
def test_collect_colocalization_data(
    sample_id, all_stages_all_reports_and_meta, snapshot
):
    dataset = PNAPixelDataset.from_files(
        all_stages_all_reports_and_meta.analysed_dataset(sample_id)
    )
    csv_data = collect_proximity_data(dataset)
    snapshot.assert_match(csv_data, f"{sample_id}.json")

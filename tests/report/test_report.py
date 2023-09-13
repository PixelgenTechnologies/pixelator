"""
Tests for the report.py module

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from anndata import AnnData
from pandas.testing import assert_frame_equal
from pixelator.cli.main import main_cli
from pixelator.pixeldataset import PixelDataset
from pixelator.report import (
    adapterqc_metrics,
    cell_calling_metrics,
    collapse_metrics,
    create_dynamic_report,
    demux_metrics,
    graph_and_annotate_metrics,
    preqc_metrics,
)
from pixelator.report.webreport.builder import WebreportBuilder
from pixelator.report.webreport.collect import generate_parameter_info
from pixelator.report.webreport.types import (
    InfoAndMetrics,
    Metrics,
    SampleInfo,
    WebreportData,
)
from pixelator.report.workdir import PixelatorWorkdir


@pytest.fixture
def markers_vertices_plot(data_root):
    json_path = data_root / "report/markers_vertices_plot.json"
    plotdata = None

    with open(json_path, "r") as f:
        plotdata = json.load(f)

    return plotdata


@pytest.fixture
def main_data(data_root) -> InfoAndMetrics:
    data_dir = data_root / "report"

    with open(data_dir / "info_and_metrics.json") as f:
        data = json.load(f)

    info = SampleInfo(**data["info"])
    #  mypy does not support spreading TypedDict yet ...
    metrics = Metrics(**data["metrics"])  # type: ignore
    return InfoAndMetrics(info=info, metrics=metrics)


@pytest.fixture
def figure_data(data_root) -> WebreportData:
    data_dir = data_root / "report"

    with open(data_dir / "component_data.csv") as f:
        embedding_data = f.read()

    with open(data_dir / "component_size_and_marker_data.csv") as f:
        ranked_component_size_data = f.read()

    with open(data_dir / "sequencing_saturation.csv") as f:
        seq_saturation = f.read()

    with open(data_dir / "antibodies_per_cell.csv") as f:
        ab_per_cell = f.read()

    with open(data_dir / "antibody_percentages.csv") as f:
        ab_pct = f.read()

    with open(data_dir / "antibody_counts.csv") as f:
        ab_counts = f.read()

    webreport_data = WebreportData(
        component_data=embedding_data,
        ranked_component_size=ranked_component_size_data,
        sequencing_saturation=seq_saturation,
        antibodies_per_cell=ab_per_cell,
        antibody_percentages=ab_pct,
        antibody_counts=ab_counts,
    )
    return webreport_data


def test_write_webreport(tmp_path, main_data, figure_data):
    output_report = tmp_path / "test_webreport.html"
    builder = WebreportBuilder()

    with open(output_report, "wb") as f:
        builder.write(
            f,
            sample_info=main_data["info"],
            metrics=main_data["metrics"],
            data=figure_data,
        )

    assert output_report.exists()


def test_preqc_metrics(data_root):
    res = preqc_metrics(str(data_root))

    assert_frame_equal(
        res,
        pd.DataFrame(
            [
                [
                    100000,
                    98866,
                    73,
                    0,
                    1061,
                    0,
                    0.01,
                ],
                [
                    200000,
                    198866,
                    83,
                    0,
                    1051,
                    0,
                    0.01,
                ],
            ],
            columns=[
                "total_reads",
                "passed_filter_reads",
                "low_quality_reads",
                "too_many_N_reads",
                "too_short_reads",
                "too_long_reads",
                "discarded",
            ],
            index=["Sample1_01", "Sample2_02"],
        ),
    )


def test_adapterqc_metrics(data_root):
    res = adapterqc_metrics(str(data_root))

    assert_frame_equal(
        res,
        pd.DataFrame(
            [
                [88866, 76257, 0.14],
                [98866, 86257, 0.13],
            ],
            columns=[
                "input",
                "output",
                "discarded",
            ],
            index=["Sample1_01", "Sample2_02"],
        ),
    )


def test_demux_metrics(data_root):
    res = demux_metrics(str(data_root))

    assert_frame_equal(
        res,
        pd.DataFrame(
            [
                [50000, 10000, 10.0, 20.0, 0.0, 30.0, 40.0, 0.8],
                [25000, 10000, 50.0, 40.0, 50.0, 10.0, 0.0, 0.6],
            ],
            columns=["input", "output", "CD1", "CD2", "CD5", "CD3", "CD4", "discarded"],
            index=["Sample1_01", "Sample2_02"],
        ),
    )


def test_collapse_metrics(data_root):
    res = collapse_metrics(str(data_root))

    assert_frame_equal(
        res,
        pd.DataFrame(
            [
                [85113, 59630, 60209, 59824, 0.29],
                [84940, 77624, 77777, 77689, 0.08],
            ],
            columns=[
                "input",
                "output_edges",
                "output_umi",
                "output_upi",
                "duplication",
            ],
            index=["Sample1_01", "Sample2_02"],
        ),
    )


def test_graph_and_annotate_metrics(data_root):
    res = graph_and_annotate_metrics(str(data_root), folder="graph")

    assert_frame_equal(
        res,
        pd.DataFrame(
            data={
                "upia": [16358, 6550],
                "upib": [16183, 6484],
                "umi": [100, 100],
                "vertices": [32541, 13034],
                "edges": [17423, 6786],
                "components": [100, 150],
                "markers": [65, 59],
                "modularity": [0.99, 0.99],
                "frac_upib_upia": [0.5, 0.5],
                "upia_degree_mean": [1.0, 1.0],
                "upia_degree_median": [1.0, 1.0],
                "frac_largest_edges": [0.1, 0.1],
                "frac_largest_vertices": [0.1, 0.1],
            },
            index=["Sample1_01", "Sample2_02"],
        ),
    )


def test_cell_calling_metrics(adata: AnnData, edgelist: pd.DataFrame, tmp_path: Path):
    dataset = PixelDataset.from_data(edgelist=edgelist, adata=adata)
    dataset.adata.obs["tau_type"] = "normal"
    dataset.adata.obs.loc["PXLCMP0000000", "tau_type"] = "high"
    dataset.adata.obs.loc["PXLCMP0000001", "tau_type"] = "high"
    dataset.adata.obs.loc["PXLCMP0000002", "tau_type"] = "low"
    dataset.adata.uns["min_size_threshold"] = 3000
    dataset.adata.uns["max_size_threshold"] = None
    dataset.adata.uns["doublet_size_threshold"] = 30000

    input_path = tmp_path / "annotate"
    input_path.mkdir(parents=True, exist_ok=True)
    dataset.save(str(input_path / "Sample1_01.dataset.pxl"))
    dataset.save(str(input_path / "Sample2_02.dataset.pxl"))

    res = cell_calling_metrics(str(tmp_path))

    assert_frame_equal(
        res,
        pd.DataFrame(
            data={
                "cells_filtered": [5, 5],
                "total_markers": [22, 22],
                "total_reads_cell": [30000, 30000],
                "median_reads_cell": [6000.0, 6000.0],
                "mean_reads_cell": [6000.0, 6000.0],
                "median_upi_cell": [1996.0, 1996.0],
                "mean_upi_cell": [1996.0, 1996.0],
                "median_upia_cell": [997.0, 997.0],
                "mean_upia_cell": [997.8, 997.8],
                "median_umi_cell": [1.0, 1.0],
                "mean_umi_cell": [1.0, 1.0],
                "median_umi_upia_cell": [6.0, 6.0],
                "mean_umi_upia_cell": [6.01, 6.01],
                "median_upia_degree_cell": [6.0, 6.0],
                "mean_upia_degree_cell": [6.01, 6.01],
                "median_markers_cell": [7.0, 7.0],
                "mean_markers_cell": [7.0, 7.0],
                "upib_per_upia": [1.0, 1.0],
                "reads_of_aggregates": [18000, 18000],
                "number_of_aggregates": [3, 3],
                "fraction_of_aggregates": [0.6, 0.6],
                "minimum_size_threshold": [3000, 3000],
                "doublet_size_threshold": [30000, 30000],
            },
            index=["Sample1_01", "Sample2_02"],
        ),
    )


def test_meta_files_collection(data_root):
    workdir = PixelatorWorkdir(data_root)
    files = workdir.metadata_files()
    assert len(files) == 4

    t1_files = workdir.metadata_files("test_data_pe_T1")
    assert len(t1_files) == 3

    t2_files = workdir.metadata_files("test_data_pe_T2")
    assert len(t2_files) == 1


def test_generate_parameter_info(data_root):
    params_files = [
        data_root / "param_files" / "test_data_pe_T1.annotate.meta.json",
        data_root / "param_files" / "test_data_pe_T1.graph.meta.json",
        data_root / "param_files" / "test_data_pe_T1.amplicon.meta.json",
    ]

    res = generate_parameter_info(main_cli, params_files)

    assert res[0].command == "pixelator single-cell amplicon"
    assert res[2].command == "pixelator single-cell annotate"
    assert res[2].options[0].name == "--panel"
    assert res[2].options[0].default_value is None
    assert res[2].options[1].name == "--min-size"
    assert res[2].options[1].value == 1
    assert res[2].options[1].default_value is None


def test_create_dynamic_report(tmp_path):
    with mock.patch(
        "pixelator.report.WebreportBuilder"
    ) as mock_builder_factory, mock.patch("pixelator.report.collect_report_data"):
        instance = mock_builder_factory.return_value

        create_dynamic_report(
            input_path="foo",
            summary_all=pd.Series({"reads": 1000, "adapterqc": 900, "duplication": 10}),
            summary_amplicon=pd.Series(
                {
                    "fraction_q30_barcode": 0.3,
                    "fraction_q30_umi": 0.3,
                    "fraction_q30_upia": 0.3,
                    "fraction_q30_upib": 0.3,
                    "fraction_q30_PBS1": 0.3,
                    "fraction_q30_PBS2": 0.3,
                    "fraction_q30": 0.3,
                },
            ),
            summary_preqc=pd.Series({"too_short_reads": 10}),
            summary_demux=pd.Series({"input": 1000, "output": 900}),
            summary_collapse=pd.Series({"input": 900}),
            summary_annotate=pd.Series([1]),
            summary_cell_calling=pd.Series(
                {
                    "total_reads_cell": 1000,
                    "reads_of_aggregates": 1,
                    "cells_filtered": 10,
                    "mean_reads_cell": 10,
                    "median_umi_cell": 11,
                    "mean_upia_cell": 3,
                    "mean_umi_upia_cell": 3,
                    "median_markers_cell": 14,
                    "total_markers": 33,
                }
            ),
            info=mock.MagicMock(),
            output_path=tmp_path,
        )
        # For now just asser that there is an attempt to write the report
        instance.write.assert_called_once()

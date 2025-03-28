"""Copyright © 2025 Pixelgen Technologies AB."""

import json

import pandas as pd
import pytest
from numpy import rec

from pixelator.pna.collapse.paired.combine_collapse import (
    combine_parquet_files,
    combine_report_files,
)
from pixelator.pna.collapse.report import CollapseSampleReport

from .conftest import PNA_DATA_ROOT


@pytest.fixture
def report_files():
    report_files_dir = PNA_DATA_ROOT / "combine_collapse"
    return list(report_files_dir.glob("*.json"))


@pytest.fixture
def report_file_1(tmp_path):
    data = {
        "reads_input": 115,
        "molecules_output": 90,
        "summary_statistics": {
            "read_counts_stats": {
                "mean": 1.3333333333333333,
                "std": 0.718664276800739,
                "min": 1.0,
                "q1": 1.0,
                "q2": 1.0,
                "q3": 1.0,
                "max": 8.0,
                "count": 4248,
                "iqr": 0.0,
            },
            "uei_stats": {
                "mean": 1.1172316384180792,
                "std": 0.4081383265670433,
                "min": 1.0,
                "q1": 1.0,
                "q2": 1.0,
                "q3": 1.0,
                "max": 6.0,
                "count": 4248,
                "iqr": 0.0,
            },
        },
        "processed_files": [
            {
                "path": "PNA055_Sample07_filtered_S7.demux.passed.part_000.demux.parquet",
                "file_size": 114479,
                "molecule_count": 4750,
            }
        ],
        "markers": [
            [
                {
                    "marker_1": "HLA-ABC",
                    "marker_2": "CD4",
                    "input_reads_count": 115,
                    "input_molecules_count": 90,
                    "corrected_reads_count": 0,
                    "cluster_size_distribution": [0, 90],
                    "collapsed_molecules_count": 90,
                    "unique_marker_links_count": 83,
                    "read_count_per_collapsed_molecule_stats": {
                        "mean": 1.2777777777777777,
                        "std": 0.5169951166280818,
                        "min": 1.0,
                        "q1": 1.0,
                        "q2": 1.0,
                        "q3": 1.0,
                        "max": 3.0,
                        "count": 90,
                        "iqr": 0.0,
                    },
                    "read_count_per_unique_marker_link_stats": {
                        "mean": 1.3855421686746987,
                        "std": 0.5773083638896838,
                        "min": 1.0,
                        "q1": 1.0,
                        "q2": 1.0,
                        "q3": 2.0,
                        "max": 3.0,
                        "count": 83,
                        "iqr": 1.0,
                    },
                    "uei_count_per_unique_marker_link_stats": {
                        "mean": 1.0843373493975903,
                        "std": 0.27789307457038065,
                        "min": 1.0,
                        "q1": 1.0,
                        "q2": 1.0,
                        "q3": 1.0,
                        "max": 2.0,
                        "count": 83,
                        "iqr": 0.0,
                    },
                    "elapsed_real_time": 0.3184335231781006,
                }
            ]
        ],
    }

    report_file = tmp_path / "test1.part_000.report.json"
    with report_file.open("w") as f:
        f.write(json.dumps(data))

    return report_file


@pytest.fixture
def report_file_2(tmp_path):
    data = {
        "reads_input": 69,
        "molecules_output": 59,
        "summary_statistics": {
            "read_counts_stats": {
                "mean": 1.2717391304347827,
                "std": 0.5870571572343816,
                "min": 1.0,
                "q1": 1.0,
                "q2": 1.0,
                "q3": 1.0,
                "max": 4.0,
                "count": 368,
                "iqr": 0.0,
            },
            "uei_stats": {
                "mean": 1.0842391304347827,
                "std": 0.27774610589236765,
                "min": 1.0,
                "q1": 1.0,
                "q2": 1.0,
                "q3": 1.0,
                "max": 2.0,
                "count": 368,
                "iqr": 0.0,
            },
        },
        "processed_files": [
            {
                "path": "PNA055_Sample07_filtered_S7.demux.passed.part_001.demux.parquet",
                "file_size": 11986,
                "molecule_count": 400,
            }
        ],
        "markers": [
            [
                {
                    "marker_1": "HLA-ABC",
                    "marker_2": "CD18",
                    "input_reads_count": 32,
                    "input_molecules_count": 25,
                    "corrected_reads_count": 0,
                    "cluster_size_distribution": [0, 25],
                    "collapsed_molecules_count": 25,
                    "unique_marker_links_count": 25,
                    "read_count_per_collapsed_molecule_stats": {
                        "mean": 1.28,
                        "std": 0.6645299090334459,
                        "min": 1.0,
                        "q1": 1.0,
                        "q2": 1.0,
                        "q3": 1.0,
                        "max": 4.0,
                        "count": 25,
                        "iqr": 0.0,
                    },
                    "read_count_per_unique_marker_link_stats": {
                        "mean": 1.28,
                        "std": 0.664529909033446,
                        "min": 1.0,
                        "q1": 1.0,
                        "q2": 1.0,
                        "q3": 1.0,
                        "max": 4.0,
                        "count": 25,
                        "iqr": 0.0,
                    },
                    "uei_count_per_unique_marker_link_stats": {
                        "mean": 1.0,
                        "std": 0.0,
                        "min": 1.0,
                        "q1": 1.0,
                        "q2": 1.0,
                        "q3": 1.0,
                        "max": 1.0,
                        "count": 25,
                        "iqr": 0.0,
                    },
                    "elapsed_real_time": 0.17090320587158203,
                },
                {
                    "marker_1": "HLA-ABC",
                    "marker_2": "TCRab",
                    "input_reads_count": 37,
                    "input_molecules_count": 34,
                    "corrected_reads_count": 0,
                    "cluster_size_distribution": [0, 34],
                    "collapsed_molecules_count": 34,
                    "unique_marker_links_count": 32,
                    "read_count_per_collapsed_molecule_stats": {
                        "mean": 1.088235294117647,
                        "std": 0.2836367870880281,
                        "min": 1.0,
                        "q1": 1.0,
                        "q2": 1.0,
                        "q3": 1.0,
                        "max": 2.0,
                        "count": 34,
                        "iqr": 0.0,
                    },
                    "read_count_per_unique_marker_link_stats": {
                        "mean": 1.15625,
                        "std": 0.4408354993645589,
                        "min": 1.0,
                        "q1": 1.0,
                        "q2": 1.0,
                        "q3": 1.0,
                        "max": 3.0,
                        "count": 32,
                        "iqr": 0.0,
                    },
                    "uei_count_per_unique_marker_link_stats": {
                        "mean": 1.0625,
                        "std": 0.24206145913796356,
                        "min": 1.0,
                        "q1": 1.0,
                        "q2": 1.0,
                        "q3": 1.0,
                        "max": 2.0,
                        "count": 32,
                        "iqr": 0.0,
                    },
                    "elapsed_real_time": 0.16115927696228027,
                },
            ]
        ],
    }

    report_file = tmp_path / "test1.part_001.report.json"
    with report_file.open("w") as f:
        f.write(json.dumps(data))

    return report_file


def test_combine_collapse_reports(report_file_1, report_file_2, snapshot):
    combined_stats = combine_report_files([report_file_1, report_file_2])
    combined_dict = combined_stats.to_dict()

    assert combined_dict["input_reads"] == 184
    assert combined_dict["output_molecules"] == 149
    assert combined_dict["corrected_reads"] == 0
    assert combined_dict["unique_marker_links"] == 140

    snapshot.assert_match(json.dumps(combined_dict), "combine_collapse_reports")


def test_combine_collapse_report_to_model(report_file_1, report_file_2):
    combined_stats = combine_report_files([report_file_1, report_file_2])
    combined_dict = combined_stats.to_dict()

    report = CollapseSampleReport(
        product_id="single-cell-pna", sample_id="test1", **combined_dict
    )

    assert report


@pytest.fixture
def parquet_files(tmp_path):
    df1_data = rec.array(
        [
            (0, "HLA-ABC", "HLA-ABC", 30675668146128039, 71192993612974932, 1, 1),
            (1, "HLA-ABC", "HLA-ABC", 44997753232109639, 19996029541346568, 1, 1),
            (2, "HLA-ABC", "HLA-ABC", 25492552756077520, 27173498434113063, 1, 1),
        ],
        dtype=[
            ("index", "<i8"),
            ("marker_1", "O"),
            ("marker_2", "O"),
            ("umi1", "<u8"),
            ("umi2", "<u8"),
            ("read_count", "<i8"),
            ("uei_count", "<i8"),
        ],
    )

    df2_data = rec.array(
        [
            (71300, "ACTB", "mIgG1", 47062695198592434, 31822590274102686, 1, 1),
            (71301, "ACTB", "mIgG1", 38938523987446592, 69438562496152639, 1, 1),
            (71302, "ACTB", "mIgG1", 60006432418681876, 40615272389271088, 1, 1),
        ],
        dtype=[
            ("index", "<i8"),
            ("marker_1", "O"),
            ("marker_2", "O"),
            ("umi1", "<u8"),
            ("umi2", "<u8"),
            ("read_count", "<i8"),
            ("uei_count", "<i8"),
        ],
    )

    edgelist1_output = tmp_path / "test1.part_000.collapse.parquet"
    edgelist2_output = tmp_path / "test1.part_001.collapse.parquet"

    df1 = pd.DataFrame.from_records(df1_data)
    df2 = pd.DataFrame.from_records(df2_data)

    df1.to_parquet(edgelist1_output, index=None)
    df2.to_parquet(edgelist2_output, index=None)

    return [edgelist1_output, edgelist2_output]


def test_combine_parquet(tmp_path, parquet_files):
    combined_parquet = tmp_path / "combined.parquet"
    combine_parquet_files(parquet_files, combined_parquet)

    assert combined_parquet.exists()

    combined_df = pd.read_parquet(combined_parquet)
    parts_dfs = []
    for p in parquet_files:
        parts_dfs.append(pd.read_parquet(p))

    check_df = pd.concat(parts_dfs, axis=0).reset_index(drop=True)
    assert combined_df.equals(check_df)

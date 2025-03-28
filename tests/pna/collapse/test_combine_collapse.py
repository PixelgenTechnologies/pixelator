"""Copyright © 2025 Pixelgen Technologies AB."""

import json

import pandas as pd
import pytest
from numpy import rec

from pixelator.pna.collapse.independent.combine_collapse import (
    CombineCollapseIndependentStats,
    combine_independent_parquet_files,
    combine_independent_report_files,
)
from pixelator.pna.collapse.independent.report import IndependentCollapseSampleReport
from pixelator.pna.collapse.paired.combine_collapse import (
    combine_parquet_files,
    combine_report_files,
)
from pixelator.pna.collapse.paired.report import CollapseSampleReport
from tests.pna.conftest import PNA_DATA_ROOT


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


@pytest.fixture
def marker1_parquet():
    m1_data = rec.array(
        [
            (
                0,
                0,
                0,
                b"\xc0\n\x14.\xba\xb9-:l\xb6\x01\x00\xae[\xc3\xeb<\xd7\x85\x0btn\x01\x00\x98\xed\x01\xf5\x86\r\x00\x00",
                1,
                36305209561225850,
            ),
            (
                1,
                0,
                0,
                b"m]\xd4X\x8bm\x9b\x8bv\x98\x01\x00\x85[\x18v\x81\x15\xa8\xbd\x0e\x18\n\x00F[w\x9d\x87\x15\x00\x00",
                1,
                40205085751253269,
            ),
            (
                2,
                0,
                0,
                b"3\x8c\x015\xbau\xc3\xea\xb9\x9d\x07\x00]\rxka\xd8\xdb6\x18E\x0b\x00\xc3\xea\x15\xf6:\x14\x00\x00",
                1,
                57786465138093603,
            ),
            (
                3,
                0,
                0,
                b"\xc5\x86uskc\xf5\xb6\xae[\x07\x00.\xd6\x156\xd0n\x1d\xb0\xb6v\x07\x00\xb5a\x18\x83\x0b\x14\x00\x00",
                1,
                62900800540892921,
            ),
            (
                4,
                0,
                0,
                b"\x1dPl\x9e\xeb\xc2[\xe1\x19v\x0b\x00\x18\xda\xc1[\xdd\xad\xee`l\xdd\x0c\x00\x83Q\xa0\xd8\x8a\x0e\x00\x00",
                1,
                22672613273565613,
            ),
            (
                5,
                0,
                0,
                b"\xb5\x0b\x0c\x83\x8d\xaeC\x81\xad\xad\x07\x00\xab\xbd\xba\xed\x00\x1b\xc0\xb0\xa2m\r\x00\xe8\x80\xad\x1d\x8c\x1a\x00\x00",
                3,
                55589776577509953,
            ),
            (
                6,
                0,
                0,
                b"[\x81\x02-\xec\xa2\x85ac\x05\x0c\x00[\xbd\x02\xeb\xec\xd6\xed\\\xa3\xb6\x0b\x00\xee\x8cz\x83\xbb\x02\x00\x00",
                1,
                11787354740467359,
            ),
            (
                7,
                0,
                0,
                b"^;\x14\xc5\x0c\x17]\r\xd8s\x01\x003\x0c\xc3\x1e\x8c\x0ek\x01x\xb3\x01\x00\xde\xb6\xce\x18\x00\x14\x00\x00",
                1,
                41387943700044636,
            ),
            (
                8,
                0,
                0,
                b"\xddP\xcc\xd8\n\xb7\x83\xe1m\xdd\x06\x00s\xb7y\x1bf`\x1d`\xd7k\x0b\x00\x06\x86z\x00:\x0c\x00\x00",
                1,
                71490844422453693,
            ),
            (
                9,
                0,
                0,
                b"\xddP\xcc\xd8\n\xb7\x83\xe1m\xdd\x06\x00s\xb7y\x1bf`\x1d`\xd7k\x0b\x00\xae1\x00\xdd\xd6\x19\x00\x00",
                2,
                71490844422453693,
            ),
        ],
        dtype=[
            ("index", "<i8"),
            ("marker_1", "<u4"),
            ("marker_2", "<u4"),
            ("molecule", "O"),
            ("read_count", "<i8"),
            ("umi1", "<u8"),
        ],
    )

    m1_parquet = pd.DataFrame.from_records(m1_data)
    return m1_parquet


@pytest.fixture
def marker2_parquet():
    m1_data = rec.array(
        [
            (
                0,
                0,
                0,
                b"\xc0\n\x14.\xba\xb9-:l\xb6\x01\x00\xae[\xc3\xeb<\xd7\x85\x0btn\x01\x00\x98\xed\x01\xf5\x86\r\x00\x00",
                1,
                36305209561225850,
            ),
            (
                1,
                0,
                0,
                b"m]\xd4X\x8bm\x9b\x8bv\x98\x01\x00\x85[\x18v\x81\x15\xa8\xbd\x0e\x18\n\x00F[w\x9d\x87\x15\x00\x00",
                1,
                40205085751253269,
            ),
            (
                2,
                0,
                0,
                b"3\x8c\x015\xbau\xc3\xea\xb9\x9d\x07\x00]\rxka\xd8\xdb6\x18E\x0b\x00\xc3\xea\x15\xf6:\x14\x00\x00",
                1,
                57786465138093603,
            ),
            (
                3,
                0,
                0,
                b"\xc5\x86uskc\xf5\xb6\xae[\x07\x00.\xd6\x156\xd0n\x1d\xb0\xb6v\x07\x00\xb5a\x18\x83\x0b\x14\x00\x00",
                1,
                62900800540892921,
            ),
            (
                4,
                0,
                0,
                b"\x1dPl\x9e\xeb\xc2[\xe1\x19v\x0b\x00\x18\xda\xc1[\xdd\xad\xee`l\xdd\x0c\x00\x83Q\xa0\xd8\x8a\x0e\x00\x00",
                1,
                22672613273565613,
            ),
            (
                5,
                0,
                0,
                b"\xb5\x0b\x0c\x83\x8d\xaeC\x81\xad\xad\x07\x00\xab\xbd\xba\xed\x00\x1b\xc0\xb0\xa2m\r\x00\xe8\x80\xad\x1d\x8c\x1a\x00\x00",
                3,
                55589776577509953,
            ),
            (
                6,
                0,
                0,
                b"[\x81\x02-\xec\xa2\x85ac\x05\x0c\x00[\xbd\x02\xeb\xec\xd6\xed\\\xa3\xb6\x0b\x00\xee\x8cz\x83\xbb\x02\x00\x00",
                1,
                11787354740467359,
            ),
            (
                7,
                0,
                0,
                b"^;\x14\xc5\x0c\x17]\r\xd8s\x01\x003\x0c\xc3\x1e\x8c\x0ek\x01x\xb3\x01\x00\xde\xb6\xce\x18\x00\x14\x00\x00",
                1,
                41387943700044636,
            ),
            (
                8,
                0,
                0,
                b"\xddP\xcc\xd8\n\xb7\x83\xe1m\xdd\x06\x00s\xb7y\x1bf`\x1d`\xd7k\x0b\x00\x06\x86z\x00:\x0c\x00\x00",
                1,
                71490844422453693,
            ),
            (
                9,
                0,
                0,
                b"\xddP\xcc\xd8\n\xb7\x83\xe1m\xdd\x06\x00s\xb7y\x1bf`\x1d`\xd7k\x0b\x00\xae1\x00\xdd\xd6\x19\x00\x00",
                2,
                71490844422453693,
            ),
        ],
        dtype=[
            ("index", "<i8"),
            ("marker_1", "<u4"),
            ("marker_2", "<u4"),
            ("molecule", "O"),
            ("read_count", "<i8"),
            ("umi1", "<u8"),
        ],
    )

    m1_parquet = pd.DataFrame.from_records(m1_data)
    return m1_parquet


def test_combine_independent_parquet(tmp_path, m1_collapsed_data, m2_collapsed_data):
    combined_parquet = tmp_path / "combined.parquet"
    stats = combine_independent_parquet_files(
        m1_collapsed_data, m2_collapsed_data, combined_parquet
    )

    assert combined_parquet.exists()
    assert stats.output_molecules == 118344
    assert stats.corrected_reads == 6409

    combined_df = pd.read_parquet(combined_parquet)

    df1 = pd.read_parquet(m1_collapsed_data)
    df2 = pd.read_parquet(m2_collapsed_data)

    assert df1.shape == df2.shape

    combined_df_read_count = combined_df["read_count"].sum()
    m1_read_count = df1["read_count"].sum()
    m2_read_count = df2["read_count"].sum()

    assert combined_df_read_count == m1_read_count == m2_read_count


def test_combine_independent_reports(
    tmp_path, m1_collapsed_report, m2_collapsed_report, snapshot
):
    sample_name = "PNA055_Sample07_filtered_S7"
    output_report_path = tmp_path / f"{sample_name}.collapse.report.json"
    combine_independent_report_files(
        [m1_collapsed_report],
        [m2_collapsed_report],
        sample_id=sample_name,
        stats=CombineCollapseIndependentStats(
            output_molecules=70937, corrected_reads=1203
        ),
        output_file=output_report_path,
    )

    assert output_report_path.exists()
    report = IndependentCollapseSampleReport.from_json(output_report_path)
    snapshot.assert_match(report.to_json(indent=4), "combine_independent_reports")

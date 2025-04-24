"""Copyright © 2025 Pixelgen Technologies AB."""

import json

import pandas as pd
import polars as pl
import pytest
from numpy.core.numeric import array_equal

from pixelator.mpx.report.models import SummaryStatistics
from pixelator.pna.collapse import MoleculeCollapser
from pixelator.pna.collapse.statistics import CollapseStatistics, MarkerLinkGroupStats
from pixelator.pna.collapse.utilities import _collect_label_array_indices, _split_chunks
from pixelator.pna.config import pna_config


def test_label_array_to_indices():
    """Test the label array to indices helper function."""
    labels = [0, 1, 1, 3, 4, 2, 4, 3, 2, 1, 2, 3, 4, 0]
    indices = _collect_label_array_indices(labels, 5)
    array_equal(indices, [[0, 13], [1, 2, 9], [5, 8, 10], [3, 7, 11], [4, 6, 12]])


def test_label_array_to_indices_uniform_shape():
    """Test the label array to indices helper function."""
    labels = [0, 1, 2, 3, 4]
    indices = _collect_label_array_indices(labels, 5)

    array_equal(indices, [[0], [1], [2], [3], [4]])
    assert indices[0] == [0]


def test_split_chunks():
    r = list(_split_chunks(10, 3))
    assert r == [(0, 3), (3, 6), (6, 9), (9, 10)]


@pytest.mark.slow
def test_collapse_from_paired_demux_data(tmp_path, testdata_paired_small_demux):
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")
    output = tmp_path / "test.parquet"

    collapser = MoleculeCollapser(assay, panel, output)

    with collapser as c:
        c.process_file(testdata_paired_small_demux)

    stats = collapser.statistics()
    assert stats

    df = pd.read_parquet(output)
    assert df.columns.tolist() == [
        "marker_1",
        "marker_2",
        "umi1",
        "umi2",
        "read_count",
        "uei_count",
    ]


def test_statistics_to_json(testdata_paired_small_demux):
    stats = CollapseStatistics()

    stats.add_input_file(testdata_paired_small_demux, molecule_count=972152)

    stats.add_marker_stats(
        "CD335",
        "ACTB",
        elapsed_time=1.211069107055664,
        cluster_stats=MarkerLinkGroupStats(
            marker_1="CD335",
            marker_2="ACTB",
            input_molecules_count=3055,
            input_reads_count=4387,
            corrected_reads_count=109,
            cluster_size_distribution=[0, 2864, 73, 10, 1, 0, 0, 0, 0, 0, 0, 1],
            collapsed_molecules_count=2949,
            unique_marker_links_count=2117,
            read_count_per_collapsed_molecule_stats=SummaryStatistics(
                mean=1.4876229230247542,
                std=0.8905514038386997,
                min=1.0,
                q1=1.0,
                q2=1.0,
                q3=2.0,
                max=15.0,
                count=2949,
                iqr=1.0,
            ),
            read_count_per_unique_marker_link_stats=SummaryStatistics(
                mean=2.072272083136514,
                std=1.7005295753657115,
                min=1.0,
                q1=1.0,
                q2=1.0,
                q3=3.0,
                max=24.0,
                count=2117,
                iqr=2.0,
            ),
            uei_count_per_unique_marker_link_stats=SummaryStatistics(
                mean=1.3930089749645724,
                std=0.8448394382060764,
                min=1.0,
                q1=1.0,
                q2=1.0,
                q3=2.0,
                max=8.0,
                count=2117,
                iqr=1.0,
            ),
        ),
    )
    lz_df = pl.DataFrame(
        {"uei_count": [1, 2, 3, 4, 5], "read_count": [1, 2, 3, 4, 5]}
    ).lazy()
    stats.add_summary_statistics(lz_df)

    dict = stats.to_dict()
    as_json = json.dumps(dict)

    assert as_json

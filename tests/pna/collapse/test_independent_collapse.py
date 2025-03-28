"""Copyright © 2025 Pixelgen Technologies AB."""

import numpy as np
import pandas as pd
import pytest

from pixelator.pna.cli.collapse import process_independent_files
from pixelator.pna.collapse.independent import MarkerCorrectionStats
from pixelator.pna.collapse.independent.collapser import (
    IndependentCollapseStatisticsCollector,
    RegionCollapser,
)
from pixelator.pna.config import pna_config
from pixelator.pna.demux.barcode_demuxer import PNAEmbedding
from pixelator.pna.utils import unpack_2bits


class TestRegionCollapserInternals:
    @pytest.fixture(autouse=True, scope="function")
    def _setup_collapser(self, umi1_partition):
        self.assay = pna_config.get_assay("pna-2")
        self.panel = pna_config.get_panel("proxiome-immuno-155")
        self.collapser = RegionCollapser(
            assay=self.assay,
            panel=self.panel,
            region_id="umi-1",
            threads=1,
            max_mismatches=1,
        )
        self.embedding = PNAEmbedding(self.assay)

    def test_extract_unique_umis(self, umi1_partition):
        """Test the _extract_unique_umis private method"""
        embedding = self.embedding
        unique_umi1, unique_umi2 = self.collapser._extract_unique_umis(
            umi1_partition["molecule"]
        )

        # Verify uniqueness
        assert np.all(np.unique(unique_umi1) == unique_umi1)
        assert np.all(np.unique(unique_umi2) == unique_umi2)

        # Check self._umi1_data and self._umi2_data attributes
        assert self.collapser._umi1_data.shape == (602,)
        assert self.collapser._umi2_data.shape == (2005,)

        # Check if the UMI1 are properly recoded from 3-bit to 2-bit encoded and both
        # can be decoded to the same original sequence
        umi1_seq = embedding.decode_umi(unique_umi1[0])
        umi1_seq2 = unpack_2bits(self.collapser._umi1_data[0], 28)

        assert umi1_seq == umi1_seq2

        # Check if the UMI2 are properly recoded from 3-bit to 2-bit encoded and both
        # can be decoded to the same original sequence
        umi2_seq = embedding.decode_umi(unique_umi2[0])
        umi2_seq2 = unpack_2bits(self.collapser._umi2_data[0], 28)

        assert umi2_seq == umi2_seq2


def test_region_collapser_umi1(tmp_path, m1_demuxed_data_part0):
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")
    output1 = tmp_path / "PNA055_Sample07_filtered_S7.collapse.m1.part_000.parquet"

    collapser = RegionCollapser(
        assay=assay, panel=panel, region_id="umi-1", threads=1, max_mismatches=1
    )

    with collapser as c1:
        c1.process_file(m1_demuxed_data_part0, output1)

    # Do some checks on the output parquet file
    df1 = pd.read_parquet(output1)
    assert df1.columns.tolist() == [
        "marker_1",
        "marker_2",
        "read_count",
        "original_umi1",
        "original_umi2",
        "uei",
        "corrected_umi1",
    ]


def test_region_collapser_umi2(tmp_path, m1_demuxed_data_part0):
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")
    output1 = tmp_path / "PNA055_Sample07_filtered_S7.collapse.m1.part_000.parquet"

    collapser = RegionCollapser(
        assay=assay, panel=panel, region_id="umi-2", threads=1, max_mismatches=1
    )

    with collapser as c2:
        c2.process_file(m1_demuxed_data_part0, output1)

    # Do some checks on the output parquet file
    df1 = pd.read_parquet(output1)
    assert df1.columns.tolist() == [
        "marker_1",
        "marker_2",
        "read_count",
        "original_umi1",
        "original_umi2",
        "uei",
        "corrected_umi2",
    ]


@pytest.mark.slow
def test_process_independent_files(tmp_path, m1_demuxed_data, m2_demuxed_data):
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")
    output1 = tmp_path / "PNA055_Sample07_filtered_S7.collapse.m1.part_000.parquet"
    output2 = tmp_path / "PNA055_Sample07_filtered_S7.collapse.m2.part_000.parquet"

    res = process_independent_files(
        umi1_files=m1_demuxed_data,
        umi2_files=m2_demuxed_data,
        assay=assay,
        panel=panel,
        collapse_output=tmp_path,
        mismatches=0.1,
        algorithm="directional",
        threads=-1,
    )


class TestIndependentCollapseStatisticsCollector:
    s1 = MarkerCorrectionStats(
        marker="CD82",
        region_id="umi-1",
        input_reads=1413845,
        input_molecules=896960,
        input_unique_umis=139132,
        corrected_reads=19868,
        corrected_unique_umis=17028,
        output_unique_umis=122104,
    )

    s2 = MarkerCorrectionStats(
        marker="ACTB",
        region_id="umi-2",
        input_reads=345,
        input_molecules=282,
        input_unique_umis=221,
        corrected_reads=3,
        corrected_unique_umis=3,
        output_unique_umis=218,
    )

    s3 = MarkerCorrectionStats(
        marker="CD8",
        region_id="umi-1",
        input_reads=184,
        input_molecules=156,
        input_unique_umis=101,
        corrected_reads=0,
        corrected_unique_umis=0,
        output_unique_umis=101,
    )

    def test_init(self):
        stats = IndependentCollapseStatisticsCollector(region_id="umi-1")
        assert stats.region_id == "umi-1"

    def test_add_umi_markers(self):
        stats = IndependentCollapseStatisticsCollector(region_id="umi-1")
        stats.add_marker_stats(self.s1)

        assert stats.markers[0] == self.s1

    def test_summary_statistics(self):
        stats = IndependentCollapseStatisticsCollector(region_id="umi-1")
        stats.add_marker_stats(self.s1)
        stats.add_marker_stats(self.s3)

        combined_stats = stats.get_combined_stats()
        assert combined_stats == {
            "input_reads": 1414029,
            "input_molecules": 897116,
            "input_unique_umis": 139233,
            "corrected_reads": 19868,
            "corrected_unique_umis": 17028,
            "output_unique_umis": 122205,
        }

    def test_to_sample_report(self):
        stats = IndependentCollapseStatisticsCollector(region_id="umi-1")
        stats.add_marker_stats(self.s1)
        stats.add_marker_stats(self.s3)

        report = stats.to_sample_report(sample_id="test")

        assert report.sample_id == "test"
        assert report.report_type == "collapse-umi"
        assert report.region_id == "umi-1"
        assert len(report.markers) == 2

    def test_to_dict(self):
        stats = IndependentCollapseStatisticsCollector(region_id="umi-1")
        stats.add_marker_stats(self.s1)
        stats.add_marker_stats(self.s3)

        dict = stats.to_dict()

        assert dict == {
            "region_id": "umi-1",
            "processed_files": [],
            "markers": [
                {
                    "marker": "CD82",
                    "region_id": "umi-1",
                    "input_reads": 1413845,
                    "input_molecules": 896960,
                    "input_unique_umis": 139132,
                    "corrected_reads": 19868,
                    "corrected_unique_umis": 17028,
                    "output_unique_umis": 122104,
                    "output_reads": 1413845,
                    "corrected_reads_fraction": 0.014052459781659234,
                    "corrected_unique_umis_fraction": 0.12238737314205216,
                },
                {
                    "marker": "CD8",
                    "region_id": "umi-1",
                    "input_reads": 184,
                    "input_molecules": 156,
                    "input_unique_umis": 101,
                    "corrected_reads": 0,
                    "corrected_unique_umis": 0,
                    "output_unique_umis": 101,
                    "output_reads": 184,
                    "corrected_reads_fraction": 0.0,
                    "corrected_unique_umis_fraction": 0.0,
                },
            ],
            "input_reads": 1414029,
            "input_molecules": 897116,
            "input_unique_umis": 139233,
            "corrected_reads": 19868,
            "corrected_unique_umis": 17028,
            "output_unique_umis": 122205,
        }

"""Copyright © 2025 Pixelgen Technologies AB."""

from collections import Counter

import pytest

from pixelator.pna.amplicon.build_amplicon import AmpliconBuilderStatistics
from pixelator.pna.amplicon.quality import QualityStatistics
from pixelator.pna.amplicon.report import AmpliconSampleReport, AmpliconStatistics


@pytest.fixture(name="amplicon_report")
def amplicon_report_fixture():
    return {
        "sample_id": "PNA055_Sample07_filtered_S7",
        "product_id": "single-cell-pna",
        "report_type": "amplicon",
        "input_reads": 99812,
        "output_reads": 97407,
        "passed_missing_lbs1_anchor": 0,
        "passed_missing_uei_reads": 0,
        "passed_partial_uei_reads": 0,
        "failed_too_many_n_reads": 0,
        "failed_partial_upi1_umi1_reads": 0,
        "failed_partial_upi2_umi2_reads": 0,
        "failed_missing_upi1_umi1_reads": 0,
        "failed_missing_upi2_umi2_reads": 0,
        "total_failed_reads": 2405,
        "q30_statistics": {
            "total": 0.9687832501504746,
            "umi1": 0.9697718996434694,
            "pid1": 0.978598047368259,
            "lbs1": 0.97016165067683,
            "uei": 0.9399345016271932,
            "lbs2": 0.978912115748876,
            "pid2": 0.976195755951831,
            "umi2": 0.9580841945944043,
        },
        "basepair_counts": {
            "input": 12177064,
            "input_read1": 4391728,
            "input_read2": 7785336,
            "quality_trimmed": 125811,
            "quality_trimmed_read1": 101807,
            "quality_trimmed_read2": 24004,
            "output": 13831794,
        },
        "failed_invalid_amplicon_reads": 0,
        "fraction_discarded_reads": 0.02409529916242536,
    }


def test_amplicon_sample_report(amplicon_report):
    report = AmpliconSampleReport(**amplicon_report)
    assert report.sample_id == "PNA055_Sample07_filtered_S7"
    assert report.product_id == "single-cell-pna"
    assert report.report_type == "amplicon"
    assert report.input_reads == 99812
    assert report.output_reads == 97407
    assert report.passed_missing_uei_reads == 0
    assert report.passed_partial_uei_reads == 0
    assert report.failed_too_many_n_reads == 0
    assert report.failed_partial_upi1_umi1_reads == 0
    assert report.failed_partial_upi2_umi2_reads == 0
    assert report.failed_missing_upi1_umi1_reads == 0
    assert report.failed_missing_upi2_umi2_reads == 0
    assert report.total_failed_reads == 2405
    assert report.q30_statistics.total == 0.9687832501504746
    assert report.q30_statistics.umi1 == 0.9697718996434694
    assert report.q30_statistics.pid1 == 0.978598047368259
    assert report.q30_statistics.lbs1 == 0.97016165067683
    assert report.q30_statistics.uei == 0.9399345016271932
    assert report.q30_statistics.lbs2 == 0.978912115748876
    assert report.q30_statistics.pid2 == 0.976195755951831
    assert report.q30_statistics.umi2 == 0.9580841945944043
    assert report.basepair_counts.input == 12177064
    assert report.basepair_counts.input_read1 == 4391728
    assert report.basepair_counts.input_read2 == 7785336
    assert report.basepair_counts.quality_trimmed == 125811
    assert report.basepair_counts.quality_trimmed_read1 == 101807
    assert report.basepair_counts.quality_trimmed_read2 == 24004
    assert report.basepair_counts.output == 13831794
    assert report.failed_invalid_amplicon_reads == 0
    assert report.fraction_discarded_reads == 0.02409529916242536


def test_amplicon_sample_report_from_amplicon_stats():
    statistics = AmpliconStatistics()
    statistics.collect(
        n=300000, total_bp1=13200000, total_bp2=23400000, steps=[], modifiers=[]
    )

    class MockReadLengths:
        def written_lengths(self):
            return (Counter({142: 222095}), Counter({}))

    read_stats = MockReadLengths()
    statistics.read_length_statistics += read_stats
    statistics.quality_trimmed_bp = [8891, 62424]
    builder_stats = AmpliconBuilderStatistics()
    builder_stats.failed_missing_upi1_umi1_reads = 54528
    builder_stats.failed_missing_upi2_umi2_reads = 21539
    builder_stats.failed_partial_upi1_umi1_reads = 0
    builder_stats.failed_partial_upi2_umi2_reads = 0
    builder_stats.failed_reads = 76067
    builder_stats.passed_missing_uei_reads = 6423
    builder_stats.passed_partial_uei_reads = 4960
    builder_stats.passed_reads = 223933

    quality_stats = QualityStatistics(
        {
            "amplicon": {
                "total_bases": 31537348,
                "sequenced_bases": 26939456,
                "q30_bases": 25881887,
            },
            "lbs-1": {
                "total_bases": 7329102,
                "sequenced_bases": 2834592,
                "q30_bases": 2740616,
            },
            "lbs-2": {
                "sequenced_bases": 3997692,
                "total_bases": 3997692,
                "q30_bases": 3817441,
            },
            "pid-1": {
                "sequenced_bases": 2220940,
                "total_bases": 2220940,
                "q30_bases": 2171855,
            },
            "pid-2": {
                "sequenced_bases": 2220940,
                "total_bases": 2220940,
                "q30_bases": 2150127,
            },
            "uei": {
                "total_bases": 3331410,
                "sequenced_bases": 3228028,
                "q30_bases": 3096228,
            },
            "umi-1": {
                "sequenced_bases": 6218632,
                "total_bases": 6218632,
                "q30_bases": 5942993,
            },
            "umi-2": {
                "sequenced_bases": 6218632,
                "total_bases": 6218632,
                "q30_bases": 5962627,
            },
        }
    )
    statistics._custom_stats["amplicon"] = builder_stats
    statistics._custom_stats["quality_profile"] = quality_stats
    statistics.filtered["too_many_n"] = 1838
    statistics.filtered["invalid_amplicon"] = 76067

    stats = statistics.as_dict()

    report = AmpliconSampleReport(
        sample_id="test", product_id="single-cell-pna", stage_name="amplicon", **stats
    )

    assert report.sample_id == "test"

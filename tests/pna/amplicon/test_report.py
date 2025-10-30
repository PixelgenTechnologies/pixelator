"""Copyright © 2025 Pixelgen Technologies AB."""

from collections import Counter

import pytest

from pixelator.pna.amplicon.build_amplicon import AmpliconBuilderStatistics
from pixelator.pna.amplicon.quality import QualityStatistics
from pixelator.pna.amplicon.report import AmpliconSampleReport, AmpliconStatistics


@pytest.fixture(name="amplicon_report")
def amplicon_report_fixture():
    return {
        "sample_id": "PNA055_Sample07_300k_S7_001",
        "product_id": "single-cell-pna",
        "report_type": "amplicon",
        "input_reads": 300000,
        "output_reads": 271272,
        "passed_missing_uei_reads": 61587,
        "passed_partial_uei_reads": 4450,
        "passed_missing_lbs1_anchor": 4315,
        "failed_too_many_n_reads": 71,
        "failed_partial_upi1_umi1_reads": 0,
        "failed_partial_upi2_umi2_reads": 0,
        "failed_missing_upi1_umi1_reads": 18243,
        "failed_missing_upi2_umi2_reads": 8265,
        "failed_lbs_detected_in_umi_reads": 2038,
        "failed_low_complexity_umi_reads": 111,
        "total_failed_reads": 28728,
        "q30_statistics": {
            "total": 0.9479181592425334,
            "umi1": 0.9770778301588706,
            "pid1": 0.9823977410127105,
            "lbs1": 0.9418085938797941,
            "uei": 0.7419050497901245,
            "lbs2": 0.817271504046038,
            "pid2": 0.9677843640331475,
            "umi2": 0.9653207850423191,
        },
        "basepair_counts": {
            "input": 36600000,
            "input_read1": 13200000,
            "input_read2": 23400000,
            "quality_trimmed": 50062,
            "quality_trimmed_read1": 2278,
            "quality_trimmed_read2": 47784,
            "output": 38520624,
        },
        "failed_invalid_amplicon_reads": 26508,
        "fraction_discarded_reads": 0.09576,
    }


def test_amplicon_sample_report(amplicon_report):
    report = AmpliconSampleReport(**amplicon_report)
    assert report.sample_id == "PNA055_Sample07_300k_S7_001"
    assert report.product_id == "single-cell-pna"
    assert report.report_type == "amplicon"
    assert report.input_reads == 300000
    assert report.output_reads == 271272
    assert report.passed_missing_uei_reads == 61587
    assert report.passed_partial_uei_reads == 4450
    assert report.failed_too_many_n_reads == 71
    assert report.failed_partial_upi1_umi1_reads == 0
    assert report.failed_partial_upi2_umi2_reads == 0
    assert report.failed_missing_upi1_umi1_reads == 18243
    assert report.failed_missing_upi2_umi2_reads == 8265
    assert report.failed_lbs_detected_in_umi_reads == 2038
    assert report.failed_low_complexity_umi_reads == 111
    assert report.total_failed_reads == 28728
    assert report.q30_statistics.total == 0.9479181592425334
    assert report.q30_statistics.umi1 == 0.9770778301588706
    assert report.q30_statistics.pid1 == 0.9823977410127105
    assert report.q30_statistics.lbs1 == 0.9418085938797941
    assert report.q30_statistics.uei == 0.7419050497901245
    assert report.q30_statistics.lbs2 == 0.817271504046038
    assert report.q30_statistics.pid2 == 0.9677843640331475
    assert report.q30_statistics.umi2 == 0.9653207850423191
    assert report.basepair_counts.input == 36600000
    assert report.basepair_counts.input_read1 == 13200000
    assert report.basepair_counts.input_read2 == 23400000
    assert report.basepair_counts.quality_trimmed == 50062
    assert report.basepair_counts.quality_trimmed_read1 == 2278
    assert report.basepair_counts.quality_trimmed_read2 == 47784
    assert report.basepair_counts.output == 38520624
    assert report.failed_invalid_amplicon_reads == 26508
    assert report.fraction_discarded_reads == 0.09576


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

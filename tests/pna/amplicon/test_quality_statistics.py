"""Copyright © 2025 Pixelgen Technologies AB."""

from pixelator.pna.amplicon.quality import QualityStatistics


def test_construction():
    stats = QualityStatistics(
        {
            "r1": {"total_bases": 100, "q30_bases": 30, "sequenced_bases": 100},
            "r2": {"total_bases": 100, "q30_bases": 10, "sequenced_bases": 100},
        }
    )

    assert stats.get_q30_fraction("r1") == 0.3
    assert stats.get_q30_fraction("r2") == 0.1


def test_merging():
    stats = QualityStatistics(
        {
            "r1": {"total_bases": 100, "q30_bases": 30, "sequenced_bases": 100},
            "r2": {"total_bases": 100, "q30_bases": 10, "sequenced_bases": 100},
        }
    )

    stats2 = QualityStatistics(
        {
            "r1": {"total_bases": 100, "q30_bases": 10, "sequenced_bases": 100},
            "r2": {"total_bases": 100, "q30_bases": 90, "sequenced_bases": 100},
        }
    )

    stats += stats2

    assert stats.get_q30_fraction("r1") == 0.2
    assert stats.get_q30_fraction("r2") == 0.5


def test_collect():
    stats = QualityStatistics(
        {
            "umi-1": {"total_bases": 100, "q30_bases": 30, "sequenced_bases": 10},
            "pid-1": {"total_bases": 100, "q30_bases": 30, "sequenced_bases": 10},
            "lbs-1": {"total_bases": 100, "q30_bases": 30, "sequenced_bases": 10},
            "uei": {"total_bases": 100, "q30_bases": 30, "sequenced_bases": 10},
            "lbs-2": {"total_bases": 100, "q30_bases": 30, "sequenced_bases": 10},
            "pid-2": {"total_bases": 100, "q30_bases": 30, "sequenced_bases": 10},
            "umi-2": {"total_bases": 100, "q30_bases": 30, "sequenced_bases": 10},
            "amplicon": {"total_bases": 100, "q30_bases": 30, "sequenced_bases": 10},
        }
    )

    r = stats.collect()

    assert r["fraction_q30_lbs1"] == 3.0
    assert r["fraction_q30_umi1"] == 3.0
    assert r["fraction_q30_pid1"] == 3.0
    assert r["fraction_q30_lbs1"] == 3.0
    assert r["fraction_q30_uei"] == 0.3
    assert r["fraction_q30_lbs2"] == 3.0
    assert r["fraction_q30_pid2"] == 3.0
    assert r["fraction_q30_umi2"] == 3.0
    assert r["fraction_q30_total"] == 3.0

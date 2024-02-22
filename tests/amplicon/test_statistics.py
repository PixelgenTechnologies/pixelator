"""Tests for amplicon statistics module.

Copyright © 2023 Pixelgen Technologies AB.
"""

import numpy as np

from pixelator.amplicon.statistics import (
    SequenceQualityStats,
    SequenceQualityStatsCollector,
)


def test_sequence_quality_stats_collector():
    _ = SequenceQualityStatsCollector(design_name="D21")


def test_sequence_quality_stats_collector_update():
    rng = np.random.default_rng(seed=1)

    collector = SequenceQualityStatsCollector(design_name="D21")
    # Make a quality array of the same length as the amplicon design
    collector.update(rng.normal(30, 5, size=132))
    collector.update(rng.normal(30, 5, size=132))

    assert collector.read_count == 2
    assert collector.stats == SequenceQualityStats(
        fraction_q30_upia=0.42,
        fraction_q30_upib=0.46,
        fraction_q30_umi=0.5,
        fraction_q30_pbs1=0.4318181818181818,
        fraction_q30_pbs2=0.5238095238095238,
        fraction_q30_bc=0.375,
        fraction_q30=0.4659090909090909,
    )

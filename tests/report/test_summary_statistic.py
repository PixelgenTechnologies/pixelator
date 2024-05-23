"""Test SummaryStatistics class.

Copyright © 2023 Pixelgen Technologies AB.
"""

from pixelator.report.models import SummaryStatistics


def test_summary_statistic_from_pandas_series(adata):
    stats = SummaryStatistics.from_series(adata.obs["reads"])

    assert stats.iqr == 0.0
    assert stats.max == 6000
    assert stats.mean == 6000.0
    assert stats.q2 == 6000.0
    assert stats.min == 6000
    assert stats.q1 == 6000.0
    assert stats.q3 == 6000.0
    assert stats.std == 0.0
    assert stats.count == 5

    assert stats.model_dump() == {
        "mean": 6000.0,
        "std": 0.0,
        "q2": 6000.0,
        "q1": 6000.0,
        "q3": 6000.0,
        "min": 6000.0,
        "max": 6000.0,
        "count": 5.0,
        "iqr": 0.0,
    }

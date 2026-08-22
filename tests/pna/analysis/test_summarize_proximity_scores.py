"""Tests for `pixelator.pna.analysis.proximity.summarize_proximity_scores`.

Copyright © 2026 Pixelgen Technologies AB.
"""

from io import StringIO

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pixelator.pna.analysis.proximity import summarize_proximity_scores

# Marker pair detection is intentionally sparse:
# - M1/M1 is detected in c1 and c2, but not c3
# - M1/M2 is detected in c1 and c3, but not c2
# - M2/M2 is detected in c2, but not c1 or c3
PROXIMITY_DATA = """component,sample,marker_1,marker_2,log2_ratio,join_count_z,join_count,join_count_expected_mean
c1,A,M1,M1,1.0,2.0,10,5
c1,A,M1,M2,0.5,1.0,8,6
c2,A,M1,M1,2.0,3.0,12,4
c2,A,M2,M2,-1.0,-2.0,2,6
c3,B,M1,M2,3.0,4.0,20,3
"""


@pytest.fixture
def proximity_df() -> pd.DataFrame:
    return pd.read_csv(StringIO(PROXIMITY_DATA))


def test_summarize_proximity_scores_default(proximity_df):
    result = summarize_proximity_scores(proximity_df)

    expected = pd.DataFrame(
        {
            "marker_1": ["M1", "M1", "M2"],
            "marker_2": ["M1", "M2", "M2"],
            "n_cells_detected": [2, 2, 1],
            "n_cells": [3, 3, 3],
            "n_cells_missing": [1, 1, 2],
            "pct_detected": [2 / 3, 2 / 3, 1 / 3],
            "mean_log2_ratio": [1.0, 3.5 / 3, -1 / 3],
        }
    )
    assert_frame_equal(result, expected)


def test_summarize_proximity_scores_median(proximity_df):
    result = summarize_proximity_scores(proximity_df, summary_stat="median")
    assert "median_log2_ratio" in result.columns
    m1_m2 = result[(result.marker_1 == "M1") & (result.marker_2 == "M2")]
    # padded values [0.5, 3.0, 0] -> median is 0.5
    assert m1_m2["median_log2_ratio"].item() == pytest.approx(0.5)


def test_summarize_proximity_scores_join_count_z(proximity_df):
    result = summarize_proximity_scores(proximity_df, proximity_metric="join_count_z")
    m1_m1 = result[(result.marker_1 == "M1") & (result.marker_2 == "M1")]
    # padded values [2.0, 3.0, 0] -> mean is 5/3
    assert m1_m1["mean_join_count_z"].item() == pytest.approx(5 / 3)


def test_summarize_proximity_scores_exclude_missing_obs(proximity_df):
    result = summarize_proximity_scores(proximity_df, include_missing_obs=False)

    m1_m1 = result[(result.marker_1 == "M1") & (result.marker_2 == "M1")]
    assert m1_m1["mean_log2_ratio"].item() == pytest.approx(1.5)

    m1_m2 = result[(result.marker_1 == "M1") & (result.marker_2 == "M2")]
    assert m1_m2["mean_log2_ratio"].item() == pytest.approx(1.75)

    m2_m2 = result[(result.marker_2 == "M2") & (result.marker_1 == "M2")]
    assert m2_m2["mean_log2_ratio"].item() == pytest.approx(-1.0)


def test_summarize_proximity_scores_group_vars(proximity_df):
    result = summarize_proximity_scores(proximity_df, group_vars="sample")

    expected = pd.DataFrame(
        {
            "sample": ["A", "A", "A", "B"],
            "marker_1": ["M1", "M1", "M2", "M1"],
            "marker_2": ["M1", "M2", "M2", "M2"],
            "n_cells_detected": [2, 1, 1, 1],
            "n_cells": [2, 2, 2, 1],
            "n_cells_missing": [0, 1, 1, 0],
            "pct_detected": [1.0, 0.5, 0.5, 1.0],
            "mean_log2_ratio": [1.5, 0.25, -0.5, 3.0],
        }
    )
    assert_frame_equal(result, expected)


def test_summarize_proximity_scores_detailed(proximity_df):
    result = summarize_proximity_scores(proximity_df, detailed=True)

    assert "log2_ratio_list" in result.columns
    assert "join_count_list" in result.columns
    assert "join_count_expected_mean_list" in result.columns

    m1_m1 = result[(result.marker_1 == "M1") & (result.marker_2 == "M1")]
    assert sorted(m1_m1["log2_ratio_list"].item()) == [0, 1.0, 2.0]
    assert sorted(m1_m1["join_count_list"].item()) == [0, 10, 12]


def test_summarize_proximity_scores_invalid_proximity_metric(proximity_df):
    with pytest.raises(ValueError, match="proximity_metric"):
        summarize_proximity_scores(proximity_df, proximity_metric="not_a_metric")


def test_summarize_proximity_scores_invalid_summary_stat(proximity_df):
    with pytest.raises(ValueError, match="summary_stat"):
        summarize_proximity_scores(proximity_df, summary_stat="not_a_stat")


def test_summarize_proximity_scores_missing_column(proximity_df):
    with pytest.raises(ValueError, match="missing required column"):
        summarize_proximity_scores(proximity_df.drop(columns=["marker_1"]))


def test_summarize_proximity_scores_missing_detail_columns(proximity_df):
    with pytest.raises(ValueError, match="missing required column"):
        summarize_proximity_scores(
            proximity_df.drop(columns=["join_count"]), detailed=True
        )


def test_summarize_proximity_scores_missing_group_var(proximity_df):
    with pytest.raises(ValueError, match="missing required column"):
        summarize_proximity_scores(proximity_df, group_vars="cell_type")


def test_summarize_proximity_scores_duplicate_rows(proximity_df):
    duplicated = pd.concat([proximity_df, proximity_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_proximity_scores(duplicated)

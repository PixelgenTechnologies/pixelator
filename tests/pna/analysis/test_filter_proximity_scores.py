"""Tests for `pixelator.pna.analysis.proximity.filter_proximity_scores`.

Copyright © 2026 Pixelgen Technologies AB.
"""

from io import StringIO

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pixelator.pna.analysis.proximity import filter_proximity_scores

# Sparse pairs with known min(p1, p2) / min(count_1, count_2):
# - M1/M1 c1: pct min=0.02, count min=20
# - M1/M2 c1: pct min=0.005, count min=5
# - M1/M1 c2: pct min=0.01, count min=10
# - M2/M2 c2: pct min=0.03, count min=30
# - M1/M2 c3: pct min=0.008, count min=8
PROXIMITY_DATA = """component,marker_1,marker_2,p1,p2,count_1,count_2,log2_ratio
c1,M1,M1,0.02,0.02,20,20,1.0
c1,M1,M2,0.02,0.005,20,5,0.5
c2,M1,M1,0.01,0.04,10,40,2.0
c2,M2,M2,0.03,0.03,30,30,-1.0
c3,M1,M2,0.008,0.05,8,50,3.0
"""

PYTHON_NAMED_DATA = """component,marker_1,marker_2,marker_1_freq,marker_2_freq,marker_1_count,marker_2_count
c1,M1,M1,0.02,0.02,20,20
c1,M1,M2,0.02,0.005,20,5
"""


@pytest.fixture
def proximity_df() -> pd.DataFrame:
    return pd.read_csv(StringIO(PROXIMITY_DATA))


def test_filter_proximity_scores_pct(proximity_df):
    result = filter_proximity_scores(proximity_df, background_threshold_pct=0.01)

    expected = proximity_df.iloc[[0, 2, 3]].reset_index(drop=True)
    assert_frame_equal(result.reset_index(drop=True), expected)


def test_filter_proximity_scores_pct_python_column_names():
    proximity_df = pd.read_csv(StringIO(PYTHON_NAMED_DATA))
    result = filter_proximity_scores(proximity_df, background_threshold_pct=0.01)

    expected = proximity_df.iloc[[0]].reset_index(drop=True)
    assert_frame_equal(result.reset_index(drop=True), expected)


def test_filter_proximity_scores_count(proximity_df):
    result = filter_proximity_scores(proximity_df, background_threshold_count=10)

    expected = proximity_df.iloc[[0, 2, 3]].reset_index(drop=True)
    assert_frame_equal(result.reset_index(drop=True), expected)


def test_filter_proximity_scores_min_cells_count(proximity_df):
    result = filter_proximity_scores(proximity_df, min_cells_count=2)

    expected = proximity_df.iloc[[0, 1, 2, 4]].reset_index(drop=True)
    assert_frame_equal(result.reset_index(drop=True), expected)


def test_filter_proximity_scores_pct_then_min_cells(proximity_df):
    result = filter_proximity_scores(
        proximity_df, background_threshold_pct=0.01, min_cells_count=2
    )

    expected = proximity_df.iloc[[0, 2]].reset_index(drop=True)
    assert_frame_equal(result.reset_index(drop=True), expected)


def test_filter_proximity_scores_no_threshold_raises(proximity_df):
    with pytest.raises(ValueError, match="At least one of"):
        filter_proximity_scores(proximity_df)


def test_filter_proximity_scores_pct_missing_columns_raises(proximity_df):
    with pytest.raises(ValueError, match="background_threshold_pct"):
        filter_proximity_scores(
            proximity_df.drop(columns=["p1", "p2"]),
            background_threshold_pct=0.01,
        )


def test_filter_proximity_scores_count_missing_columns_raises(proximity_df):
    with pytest.raises(ValueError, match="background_threshold_count"):
        filter_proximity_scores(
            proximity_df.drop(columns=["count_1", "count_2"]),
            background_threshold_count=10,
        )


def test_filter_proximity_scores_pct_out_of_range_raises(proximity_df):
    with pytest.raises(ValueError, match="background_threshold_pct"):
        filter_proximity_scores(proximity_df, background_threshold_pct=1.5)

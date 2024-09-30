"""
Tests for the normalization modules

Copyright © 2024 Pixelgen Technologies AB.
"""

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from pixelator.analysis.normalization import dsb_normalize

DATA_ROOT = Path(__file__).parents[2] / "data"


def test_dsb_normalize():
    input_data = pd.read_csv(
        str(DATA_ROOT / "dsb_normalization_test_input.csv")
    ).astype(float)
    output_data = pd.read_csv(
        str(DATA_ROOT / "dsb_normalization_test_output.csv")
    ).astype(float)
    output_data = output_data - output_data.iloc[0, :]
    result = dsb_normalize(input_data, isotype_controls=["mIgG1", "mIgG2a", "mIgG2b"])
    result = result - result.iloc[0, :]
    assert_frame_equal(result, output_data, atol=0.08)

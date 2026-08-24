"""Utility functions for (de)serializing data to and from disk formats.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def np_encoder(object: Any):
    """Encoder for JSON serialization of numpy data types."""  # noqa: D401
    if isinstance(object, np.generic):
        return object.item()


def remove_csv_whitespaces(df: pd.DataFrame) -> None:
    """Remove leading and trailing blank spaces from csv files slurped by pandas."""
    # fill NaNs as empty strings to be able to do `.str`
    df.fillna("", inplace=True)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].str.strip()

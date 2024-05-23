"""Model for a collection of summary statistics of a distribution.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pydantic

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


T = TypeVar("T")


class SummaryStatistics(pydantic.BaseModel):
    """A collection of summary statistics for a distribution."""

    mean: float
    std: float
    min: float
    q1: float
    q2: float
    q3: float
    max: float
    count: int

    @pydantic.computed_field(
        return_type=float,
        description="The interquartile range.",
    )
    def iqr(self) -> float:
        """Return the interquartile range."""
        return self.q3 - self.q1

    @classmethod
    def from_series(cls, distribution: pd.Series | pl.Series) -> Self:
        """Create a SummaryStatistics from a pandas or polars Series.

        :param distribution: The series to summarize.
        :return: A SummaryStatistics object from given series.
        """
        distribution = distribution.to_numpy()

        mean = np.mean(distribution)
        std = np.std(distribution)
        min = np.min(distribution)
        max = np.max(distribution)
        q1, q2, q3 = np.quantile(distribution, [0.25, 0.5, 0.75])
        count = len(distribution)

        return cls(
            mean=float(mean),
            std=float(std),
            q1=float(q1),
            q2=float(q2),
            q3=float(q3),
            max=float(max),
            min=float(min),
            count=int(count),
        )

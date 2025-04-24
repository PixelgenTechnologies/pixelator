"""Types associated with colocalization.

Copyright © 2023 Pixelgen Technologies AB.
"""

from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

from pixelator.mpx.analysis.types import RegionByCountsDataFrame

MarkerColocalizationResults = pd.DataFrame

TransformationTypes = Literal["raw", "log1p", "rate-diff"]


@dataclass
class CoLocalizationFunction:
    """Container for colocalization functions and their names."""

    name: str
    func: Callable[  # type: ignore
        [RegionByCountsDataFrame], MarkerColocalizationResults  # type: ignore
    ]  # type: ignore

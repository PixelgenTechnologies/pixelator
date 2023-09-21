"""Types associated with colocalization.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

RegionByCountsDataFrame = pd.DataFrame
MarkerColocalizationResults = pd.DataFrame

TransformationTypes = Literal["raw", "log1p"]


@dataclass
class CoLocalizationFunction:
    """Container for colocalization functions and their names."""

    name: str
    func: Callable[  # type: ignore
        [RegionByCountsDataFrame], MarkerColocalizationResults  # type: ignore
    ]  # type: ignore

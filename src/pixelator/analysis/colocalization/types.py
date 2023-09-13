"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""

from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

RegionByCountsDataFrame = pd.DataFrame
MarkerColocalizationResults = pd.DataFrame

TransformationTypes = Literal["raw", "clr", "log1p", "relative"]


@dataclass
class CoLocalizationFunction:
    """
    Holds the name to use for a colocalization function to make them
    easier to work with as a pair
    """

    name: str
    func: Callable[  # type: ignore
        [RegionByCountsDataFrame], MarkerColocalizationResults  # type: ignore
    ]  # type: ignore

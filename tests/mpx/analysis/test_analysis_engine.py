"""Tests for the analysis engine.

Copyright © 2024 Pixelgen Technologies AB.
"""

from types import GeneratorType
from typing import Any, Iterable
from unittest.mock import MagicMock

import pandas as pd
from pandas.testing import assert_frame_equal

from pixelator.mpx.analysis.analysis_engine import (
    PerComponentAnalysis,
    _AnalysisManager,
    edgelist_to_component_stream,
    run_analysis,
)
from pixelator.mpx.analysis.colocalization import ColocalizationAnalysis
from pixelator.mpx.analysis.polarization import PolarizationAnalysis
from pixelator.mpx.graph import Graph
from pixelator.mpx.pixeldataset import PixelDataset


class MockAnalysis(PerComponentAnalysis):
    """Represent mock analysis."""

    def __init__(self, multiplication_factor):
        """Initialize the instance.

        Args:
            multiplication_factor: multiplication factor.
        """
        self.multiplication_factor = multiplication_factor
        self.ANALYSIS_NAME = f"mock_analysis_{multiplication_factor}"

    def run_on_component(self, component: Graph, component_id: str) -> pd.DataFrame:
        """Run on component.

        Args:
            component: Component.
            component_id: Component id.
        """
        return pd.DataFrame({"component_id": [component_id], "values": [2]})

    def post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Post process data.

        Args:
            data: Data.
        """
        data["values_multiplied"] = data["values"] * self.multiplication_factor
        return data

    def add_to_pixel_dataset(
        self, data: pd.DataFrame, pxl_dataset: PixelDataset
    ) -> PixelDataset:
        """Add to pixel dataset.

        Args:
            data: Data.
            pxl_dataset: Pxl dataset.
        """
        pxl_dataset.data_slots[self.ANALYSIS_NAME] = data  # type: ignore
        return pxl_dataset

    def concatenate_data(self, data: Iterable[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate data.

        Args:
            data: Data.
        """
        scores = pd.concat(data, axis=0, ignore_index=True)
        return scores


class MockPixelDataset:
    """Represent mock pixel dataset."""

    def __init__(self) -> None:
        """Initialize the instance.

        Returns:
            Result (None).
        """
        self.data_slots = dict()  # type: ignore


class MockGraph(MagicMock):
    """Mockable Graph object, see: https://github.com/testing-cabal/mock/issues/139"""

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (MagicMock, ())


def test_analysis_manager():
    """Verify analysis manager."""
    component_stream = [
        ("component1", MockGraph()),
        ("component2", MockGraph()),
    ]

    analysis_to_run = [MockAnalysis(2), MockAnalysis(3)]

    engine = _AnalysisManager(
        analysis_to_run=analysis_to_run, component_stream=component_stream
    )
    mock_pixel_dataset = MockPixelDataset()

    result = engine.execute(pixel_dataset=mock_pixel_dataset)

    assert result.data_slots.keys() == {"mock_analysis_2", "mock_analysis_3"}
    expected_1 = pd.DataFrame(
        {
            "component_id": ["component1", "component2"],
            "values": [2, 2],
            "values_multiplied": [4, 4],
        }
    )
    expected_2 = pd.DataFrame(
        {
            "component_id": ["component1", "component2"],
            "values": [2, 2],
            "values_multiplied": [6, 6],
        }
    )
    assert_frame_equal(
        result.data_slots["mock_analysis_2"]
        .sort_values("component_id")
        .reset_index(drop=True),
        expected_1,
        check_index_type=False,
    )
    assert_frame_equal(
        result.data_slots["mock_analysis_3"]
        .sort_values("component_id")
        .reset_index(drop=True),
        expected_2,
        check_index_type=False,
    )


def test_edgelist_to_component_stream(setup_basic_pixel_dataset):
    """Verify edgelist to component stream.

    Args:
        setup_basic_pixel_dataset: setup basic pixel dataset.
    """
    (
        dataset,
        *_,
    ) = setup_basic_pixel_dataset
    result = edgelist_to_component_stream(dataset=dataset, use_full_bipartite=False)
    assert isinstance(result, GeneratorType)
    component_id, component = next(result)
    assert isinstance(component_id, str)
    assert isinstance(component, Graph)


def test_run_analysis(setup_basic_pixel_dataset):
    """Verify run analysis.

    Args:
        setup_basic_pixel_dataset: setup basic pixel dataset.
    """
    (
        dataset,
        *_,
    ) = setup_basic_pixel_dataset

    dataset.polarization = None
    dataset.colocalization = None

    analysis_functions = [
        PolarizationAnalysis("log1p", n_permutations=5, min_marker_count=1),
        ColocalizationAnalysis(
            "rate-diff",
            neighbourhood_size=3,
            n_permutations=5,
            min_region_count=3,
            min_marker_count=1,
        ),
    ]

    result = run_analysis(
        dataset, analysis_to_run=analysis_functions, use_full_bipartite=False
    )

    assert isinstance(result.colocalization, pd.DataFrame)
    assert not result.colocalization.empty

    assert isinstance(result.polarization, pd.DataFrame)
    assert not result.polarization.empty

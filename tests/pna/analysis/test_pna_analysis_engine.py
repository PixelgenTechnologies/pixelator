"""Tests for the analysis engine.

Copyright © 2024 Pixelgen Technologies AB.
"""

from typing import Iterable
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from pixelator.pna.analysis_engine import (
    AnalysisManager,
    PerComponentTask,
)
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset.io import PxlFile


class MockAnalysis(PerComponentTask):
    """Represent mock analysis."""

    def __init__(self, multiplication_factor):
        """Initialize the instance.

        Args:
            multiplication_factor: multiplication factor.
        """
        self.multiplication_factor = multiplication_factor
        self.TASK_NAME = f"mock_analysis_{multiplication_factor}"

    def run_on_component_edgelist(
        self, component: pl.LazyFrame, component_id: str
    ) -> pd.DataFrame:
        """Run on component edgelist.

        Args:
            component: Component.
            component_id: Component id.
        """
        return pd.DataFrame({"component_id": [component_id], "values": [2]})

    def run_on_component_graph(self, component: PNAGraph, component_id: str):
        """Run the analysis on this component.

        Args:
            component: The graph of the component.
            component_id: The id of the component.
        """
        raise NotImplementedError

    def post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Post process data.

        Args:
            data: Data.
        """
        data["values_multiplied"] = data["values"] * self.multiplication_factor
        return data

    def add_to_pixel_file(self, data: pd.DataFrame, pxl_file_target: PxlFile) -> None:
        """Add to pixel file.

        Args:
            data: Data.
            pxl_file_target: Pxl file target.
        """
        pxl_file_target.data_slots[self.TASK_NAME] = data  # type: ignore

    def concatenate_data(self, data: Iterable[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate data.

        Args:
            data: Data.
        """
        scores = pd.concat(data, axis=0, ignore_index=True)
        return scores


class MockGraph(MagicMock):
    """Mockable Graph object, see: https://github.com/testing-cabal/mock/issues/139"""

    def __reduce__(self):
        return (MagicMock, ())


@pytest.mark.slow
def test_analysis_manager():
    """Verify analysis manager."""

    class MockComponent:
        """Represent mock component."""

        def __init__(self, component_id, data):
            """Initialize the instance.

            Args:
                component_id: component id.
                data: data.
            """
            self.component_id = component_id
            self.frame = data

        @property
        def graph(self):
            """Graph."""
            return MockGraph()

    component_stream = [
        MockComponent("component1", pl.LazyFrame()),
        MockComponent("component2", pl.LazyFrame()),
    ]

    analysis_to_run = [MockAnalysis(2), MockAnalysis(3)]

    class MockPxlFile:
        """Represent mock pxl file."""

        def __init__(self):
            """Initialize the instance."""
            self.data_slots = {}

    class MockPixelDataset:
        """Represent mock pixel dataset."""

        def __init__(self, mock_component_stream):
            """Initialize the instance.

            Args:
                mock_component_stream: mock component stream.
            """
            self._mock_component_stream = mock_component_stream
            self.data = {}

        def edgelist(self):
            """Edgelist."""
            mock = MagicMock()
            mock.iterator.return_value = self._mock_component_stream
            return mock

    def mock_builder(pxl_files):
        mock_dataset = MockPixelDataset(None)
        mock_dataset.data = pxl_files.data_slots
        return mock_dataset

    engine = AnalysisManager(
        analysis_to_run=analysis_to_run, pxl_dataset_builder=mock_builder
    )
    mock_pixel_dataset = MockPixelDataset(component_stream)

    mock_pxl_file_target = MockPxlFile()

    result = engine.execute(
        pixel_dataset=mock_pixel_dataset, pxl_file_target=mock_pxl_file_target
    )

    assert result.data.keys() == {"mock_analysis_2", "mock_analysis_3"}
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
        result.data["mock_analysis_2"]
        .sort_values("component_id")
        .reset_index(drop=True),
        expected_1,
        check_index_type=False,
    )
    assert_frame_equal(
        result.data["mock_analysis_3"]
        .sort_values("component_id")
        .reset_index(drop=True),
        expected_2,
        check_index_type=False,
    )

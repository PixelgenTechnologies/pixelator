"""Copyright (c) 2024 Pixelgen Technologies AB."""

from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from pixelator.pixeldataset.precomputed_layouts import (
    PreComputedLayouts,
    aggregate_precomputed_layouts,
)


@pytest.fixture(name="layout_df")
def layout_df_fixture() -> pd.DataFrame:
    nbr_of_rows = 300
    components = [
        "PXLCMP0000000",
        "PXLCMP0000001",
        "PXLCMP0000002",
        "PXLCMP0000003",
        "PXLCMP0000004",
    ]
    sample = ["sample_1", "sample_2"]
    graph_projections = ["bipartite", "a-node"]
    layout_methods = ["pmds", "fr"]
    rgn = np.random.default_rng(1)
    layout_df = pd.DataFrame(
        {
            "x": rgn.random(nbr_of_rows),
            "y": rgn.random(nbr_of_rows),
            "z": rgn.random(nbr_of_rows),
            "graph_projection": rgn.choice(graph_projections, nbr_of_rows),
            "layout": rgn.choice(layout_methods, nbr_of_rows),
            "component": rgn.choice(components, nbr_of_rows),
            "sample": rgn.choice(sample, nbr_of_rows),
        }
    )
    yield layout_df


@pytest.fixture(name="precomputed_layouts")
def precomputed_layouts_fixture(layout_df) -> pd.DataFrame:
    yield PreComputedLayouts(pl.DataFrame(layout_df).lazy())


class TestPreComputedLayouts:
    def test_is_empty_returns_true_for_empty_layout(self):
        layouts_lazy = pl.DataFrame({"component": []}).lazy()
        precomputed_layouts = PreComputedLayouts(layouts_lazy)
        assert precomputed_layouts.is_empty

    def test_is_empty_returns_false_for_non_empty_layout(self):
        layouts_lazy = pl.DataFrame({"component": ["PXLCMP0000000"]}).lazy()
        precomputed_layouts = PreComputedLayouts(layouts_lazy)
        assert not precomputed_layouts.is_empty

    def test_partitioning_returns_default_partitioning(self):
        layouts_lazy = pl.DataFrame({"component": []}).lazy()
        precomputed_layouts = PreComputedLayouts(layouts_lazy)
        assert precomputed_layouts.partitioning == [
            "graph_projection",
            "layout",
            "component",
        ]

    def test_df_returns_pandas_dataframe(self, precomputed_layouts):
        df = precomputed_layouts.df
        assert isinstance(df, pd.DataFrame)
        assert set(df["component"]) == {
            "PXLCMP0000000",
            "PXLCMP0000001",
            "PXLCMP0000002",
            "PXLCMP0000003",
            "PXLCMP0000004",
        }
        assert set(df.columns) == {
            "x",
            "y",
            "z",
            "graph_projection",
            "layout",
            "component",
            "sample",
        }

    def test_lazy_returns_polars_lazy_frame(self):
        layouts_lazy = pl.DataFrame({"component": ["PXLCMP0000000"]}).lazy()
        precomputed_layouts = PreComputedLayouts(layouts_lazy)
        lazy = precomputed_layouts.lazy
        assert isinstance(lazy, pl.LazyFrame)
        assert lazy.collect().shape == (1, 1)

    def test_filter_returns_filtered_dataframe(self):
        layouts_lazy = pl.DataFrame(
            {
                "component": ["PXLCMP0000000", "PXLCMP0000001"],
                "graph_projection": ["a-node", "b-node"],
                "layout": ["pmds", "fr"],
            }
        ).lazy()

        precomputed_layouts = PreComputedLayouts(layouts_lazy)
        filtered = precomputed_layouts.filter(
            component_ids=["PXLCMP0000000"],
            graph_projection="a-node",
            layout_method="pmds",
        )

        assert isinstance(filtered, PreComputedLayouts)

        expected_df = pd.DataFrame(
            {
                "component": ["PXLCMP0000000"],
                "graph_projection": ["a-node"],
                "layout": ["pmds"],
            }
        )
        assert_frame_equal(filtered.df, expected_df)

    def test_iterator_returns_filtered_dataframes(self):
        layouts_lazy = pl.DataFrame(
            {
                "component": [
                    "PXLCMP0000000",
                    "PXLCMP0000000",
                    "PXLCMP0000001",
                    "PXLCMP0000001",
                ],
                "graph_projection": ["a-node", "a-node", "a-node", "a-node"],
                "layout": ["pmds", "pmds", "fr", "pmds"],
            }
        ).lazy()
        precomputed_layouts = PreComputedLayouts(layouts_lazy)

        result = precomputed_layouts.component_iterator(
            component_ids=["PXLCMP0000000", "PXLCMP0000001"],
            graph_projections="a-node",
            layout_methods="pmds",
        )

        expected_df1 = pd.DataFrame(
            {
                "component": ["PXLCMP0000000", "PXLCMP0000000"],
                "graph_projection": ["a-node", "a-node"],
                "layout": ["pmds", "pmds"],
            }
        )
        expected_df2 = pd.DataFrame(
            {
                "component": ["PXLCMP0000001"],
                "graph_projection": ["a-node"],
                "layout": ["pmds"],
            }
        )
        assert isinstance(result, Iterable)
        result = list(result)
        assert len(result) == 2
        # There is no guarantee for the order of the results
        try:
            assert_frame_equal(result[0], expected_df1)
            assert_frame_equal(result[1], expected_df2)
        except AssertionError:
            assert_frame_equal(result[1], expected_df1)
            assert_frame_equal(result[0], expected_df2)

    def test_aggregate_precomputed_layouts(self):
        # Create some sample PreComputedLayouts
        layout1 = PreComputedLayouts(
            pl.DataFrame(
                {
                    "x": [1, 2, 3],
                    "y": [4, 5, 6],
                    "component": ["A", "B", "C"],
                    "sample": ["sample1", "sample1", "sample1"],
                }
            ).lazy()
        )
        layout2 = PreComputedLayouts(
            pl.DataFrame(
                {
                    "x": [7, 8, 9],
                    "y": [10, 11, 12],
                    "component": ["A", "B", "C"],
                    "sample": ["sample2", "sample2", "sample2"],
                }
            ).lazy()
        )

        # Aggregate the layouts
        aggregated_layouts = aggregate_precomputed_layouts(
            [("sample1", layout1), ("sample2", layout2)],
            all_markers={"x", "y", "component", "sample"},
        )

        # Check the aggregated layout DataFrame
        expected_df = pd.DataFrame(
            {
                "x": [1, 2, 3, 7, 8, 9],
                "y": [4, 5, 6, 10, 11, 12],
                "component": ["A", "B", "C", "A", "B", "C"],
                "sample": [
                    "sample1",
                    "sample1",
                    "sample1",
                    "sample2",
                    "sample2",
                    "sample2",
                ],
            }
        )
        pd.testing.assert_frame_equal(aggregated_layouts.df, expected_df)

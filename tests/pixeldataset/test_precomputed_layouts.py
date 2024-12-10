"""Copyright © 2024 Pixelgen Technologies AB."""

from typing import Iterable
from unittest import mock

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from pixelator.pixeldataset import PixelDataset
from pixelator.pixeldataset.precomputed_layouts import (
    PreComputedLayouts,
    aggregate_precomputed_layouts,
    generate_precomputed_layouts_for_components,
)
from tests.utils import dna_seqs


def layout_df() -> pl.LazyFrame:
    nbr_of_rows = 300
    components = [
        "2ac2ca983a4b82dd",
        "6ed5d4e4cfe588bd",
        "701ec72d3bda62d5",
        "bec92437d668cfa1",
        "ce2709afa8ebd1c9",
    ]
    sample = ["sample_1", "sample_2"]
    graph_projections = ["bipartite", "a-node"]
    layout_methods = ["pmds", "fr"]
    pixel_type = ["A", "B"]
    sequences = dna_seqs(length=10, min_dist=0, n_sequences=1000)
    rgn = np.random.default_rng(1)
    layout_df = (
        pl.DataFrame(
            {
                "x": rgn.random(nbr_of_rows),
                "y": rgn.random(nbr_of_rows),
                "z": rgn.random(nbr_of_rows),
                "graph_projection": rgn.choice(graph_projections, nbr_of_rows),
                "layout": rgn.choice(layout_methods, nbr_of_rows),
                "component": rgn.choice(components, nbr_of_rows),
                "sample": rgn.choice(sample, nbr_of_rows),
                "name": rgn.choice(sequences, nbr_of_rows),
                "pixel_type": rgn.choice(pixel_type, nbr_of_rows),
            }
        )
        .with_columns(index=pl.col("name"))
        .lazy()
    )
    return layout_df


def layout_df_generator() -> Iterable[pl.LazyFrame]:
    for component in [
        "2ac2ca983a4b82dd",
        "6ed5d4e4cfe588bd",
        "701ec72d3bda62d5",
        "bec92437d668cfa1",
        "ce2709afa8ebd1c9",
    ]:
        nbr_of_rows = 300
        sample = ["sample_1", "sample_2"]
        graph_projections = ["bipartite", "a-node"]
        layout_methods = ["pmds", "fr"]
        rgn = np.random.default_rng(1)
        pixel_type = ["A", "B"]
        sequences = dna_seqs(length=10, min_dist=0, n_sequences=1000)
        layout_df = (
            pl.DataFrame(
                {
                    "x": rgn.random(nbr_of_rows),
                    "y": rgn.random(nbr_of_rows),
                    "z": rgn.random(nbr_of_rows),
                    "graph_projection": rgn.choice(graph_projections, nbr_of_rows),
                    "layout": rgn.choice(layout_methods, nbr_of_rows),
                    "component": component,
                    "sample": rgn.choice(sample, nbr_of_rows),
                    "name": rgn.choice(sequences, nbr_of_rows),
                    "pixel_type": rgn.choice(pixel_type, nbr_of_rows),
                }
            )
            .with_columns(index=pl.col("name"))
            .lazy()
        )
        yield layout_df


# We are using this to make sure we cover both cases of the PreComputedLayouts
# one where we have a DataFrame and one where we have a generator of data frames
@pytest.fixture(name="precomputed_layouts", params=["layout_df", "layout_df_generator"])
def precomputed_layouts_fixture(request) -> PreComputedLayouts:
    if request.param == "layout_df":
        return PreComputedLayouts(layout_df())  # type: ignore
    if request.param == "layout_df_generator":
        return PreComputedLayouts(layout_df_generator())
    raise Exception("We should never get here!")


class MockPixelDataset:
    def __init__(self, precomputed_layouts):
        self.precomputed_layouts = precomputed_layouts


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

    def test_to_df_returns_pandas_dataframe(self, precomputed_layouts):
        df = precomputed_layouts.to_df()
        assert isinstance(df, pd.DataFrame)
        assert set(df["component"]) == {
            "2ac2ca983a4b82dd",
            "6ed5d4e4cfe588bd",
            "701ec72d3bda62d5",
            "bec92437d668cfa1",
            "ce2709afa8ebd1c9",
        }
        assert set(df.columns) == {
            "x",
            "y",
            "z",
            "graph_projection",
            "layout",
            "component",
            "sample",
            "name",
            "pixel_type",
            "index",
        }

    def test_to_df_filters_columns(self, precomputed_layouts):
        df = precomputed_layouts.to_df(columns=["x", "y", "component"])
        assert isinstance(df, pd.DataFrame)
        assert set(df["component"]) == {
            "2ac2ca983a4b82dd",
            "6ed5d4e4cfe588bd",
            "701ec72d3bda62d5",
            "bec92437d668cfa1",
            "ce2709afa8ebd1c9",
        }
        assert set(df.columns) == {
            "x",
            "y",
            "component",
        }

    def test_get_unique_components(self, precomputed_layouts):
        unique_componentens = precomputed_layouts.unique_components()
        assert unique_componentens == {
            "2ac2ca983a4b82dd",
            "6ed5d4e4cfe588bd",
            "701ec72d3bda62d5",
            "bec92437d668cfa1",
            "ce2709afa8ebd1c9",
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
                "component": ["2ac2ca983a4b82dd", "6ed5d4e4cfe588bd"],
                "graph_projection": ["a-node", "b-node"],
                "layout": ["pmds", "fr"],
            }
        ).lazy()

        precomputed_layouts = PreComputedLayouts(layouts_lazy)
        filtered = precomputed_layouts.filter(
            component_ids=["2ac2ca983a4b82dd"],
            graph_projection="a-node",
            layout_method="pmds",
        )

        assert isinstance(filtered, PreComputedLayouts)

        expected_df = pd.DataFrame(
            {
                "component": ["2ac2ca983a4b82dd"],
                "graph_projection": ["a-node"],
                "layout": ["pmds"],
            }
        )
        assert_frame_equal(filtered.to_df(), expected_df)

    def test_iterator_returns_filtered_dataframes(self):
        layouts_lazy = pl.DataFrame(
            {
                "component": [
                    "2ac2ca983a4b82dd",
                    "2ac2ca983a4b82dd",
                    "6ed5d4e4cfe588bd",
                    "6ed5d4e4cfe588bd",
                ],
                "graph_projection": ["a-node", "a-node", "a-node", "a-node"],
                "layout": ["pmds", "pmds", "fr", "pmds"],
            }
        ).lazy()
        precomputed_layouts = PreComputedLayouts(layouts_lazy)

        result = precomputed_layouts.component_iterator(
            component_ids=["2ac2ca983a4b82dd", "6ed5d4e4cfe588bd"],
            graph_projections="a-node",
            layout_methods="pmds",
        )

        expected_df1 = pd.DataFrame(
            {
                "component": ["2ac2ca983a4b82dd", "2ac2ca983a4b82dd"],
                "graph_projection": ["a-node", "a-node"],
                "layout": ["pmds", "pmds"],
            }
        )
        expected_df2 = pd.DataFrame(
            {
                "component": ["6ed5d4e4cfe588bd"],
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

    def test_iterator_returns_filtered_dataframes_and_requested_columns(self):
        layouts_lazy = pl.DataFrame(
            {
                "component": [
                    "2ac2ca983a4b82dd",
                    "2ac2ca983a4b82dd",
                    "6ed5d4e4cfe588bd",
                    "6ed5d4e4cfe588bd",
                ],
                "graph_projection": ["a-node", "a-node", "a-node", "a-node"],
                "layout": ["pmds", "pmds", "fr", "pmds"],
                "x": [0.5, 0.9, 0.6, 0.3],
                "y": [0.4, 0.1, 0.7, 0.2],
            }
        ).lazy()
        precomputed_layouts = PreComputedLayouts(layouts_lazy)

        result = list(
            precomputed_layouts.component_iterator(
                component_ids=["2ac2ca983a4b82dd"],
                graph_projections="a-node",
                layout_methods="pmds",
                columns=["component", "x", "y"],
            )
        )

        expected_df1 = pd.DataFrame(
            {
                "component": ["2ac2ca983a4b82dd", "2ac2ca983a4b82dd"],
                "x": [0.5, 0.9],
                "y": [0.4, 0.1],
            }
        )

        assert_frame_equal(result[0], expected_df1)

    def test_aggregate_precomputed_layouts(self):
        # Create some sample PreComputedLayouts
        pxl_1 = MockPixelDataset(
            PreComputedLayouts(
                pl.DataFrame(
                    {
                        "x": [1, 2, 3],
                        "y": [4, 5, 6],
                        "component": ["A", "B", "C"],
                        "sample": ["sample1", "sample1", "sample1"],
                    }
                ).lazy()
            )
        )
        pxl_2 = MockPixelDataset(
            PreComputedLayouts(
                pl.DataFrame(
                    {
                        "x": [7, 8, 9],
                        "y": [10, 11, 12],
                        "component": ["A", "B", "C"],
                        "sample": ["sample2", "sample2", "sample2"],
                    }
                ).lazy()
            )
        )

        # Aggregate the layouts
        aggregated_layouts = aggregate_precomputed_layouts(
            [("sample1", pxl_1), ("sample2", pxl_2)],
            all_markers={"x", "y", "component", "sample"},
        )

        # Check the aggregated layout DataFrame
        expected_df = pd.DataFrame(
            {
                "x": [1, 2, 3, 7, 8, 9],
                "y": [4, 5, 6, 10, 11, 12],
                "component": [
                    "A_sample1",
                    "B_sample1",
                    "C_sample1",
                    "A_sample2",
                    "B_sample2",
                    "C_sample2",
                ],
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
        pd.testing.assert_frame_equal(aggregated_layouts.to_df(), expected_df)

    def test_aggregate_precomputed_layouts_one_empty_data_frame(self):
        # Create some sample PreComputedLayouts
        mock_pxl_1 = MockPixelDataset(PreComputedLayouts(None))
        mock_pxl_2 = MockPixelDataset(
            PreComputedLayouts(
                pl.DataFrame(
                    {
                        "x": [7, 8, 9],
                        "y": [10, 11, 12],
                        "component": ["A", "B", "C"],
                        "sample": ["sample2", "sample2", "sample2"],
                    }
                ).lazy()
            )
        )

        # Aggregate the layouts
        aggregated_layouts = aggregate_precomputed_layouts(
            [("sample1", mock_pxl_1), ("sample2", mock_pxl_2)],
            all_markers={"x", "y", "component", "sample"},
        )

        # Check the aggregated layout DataFrame
        expected_df = pd.DataFrame(
            {
                "x": [7, 8, 9],
                "y": [10, 11, 12],
                "component": ["A_sample2", "B_sample2", "C_sample2"],
                "sample": [
                    "sample2",
                    "sample2",
                    "sample2",
                ],
            }
        )
        pd.testing.assert_frame_equal(aggregated_layouts.to_df(), expected_df)

    def test_aggregate_precomputed_layouts_no_layouts_in_data(self):
        # Create some sample PreComputedLayouts
        mock_pxl_1 = MockPixelDataset(PreComputedLayouts(None))
        mock_pxl_2 = MockPixelDataset(PreComputedLayouts(None))

        # Aggregate the layouts
        aggregated_layouts = aggregate_precomputed_layouts(
            [("sample1", mock_pxl_1), ("sample2", mock_pxl_2)],
            all_markers={"x", "y", "component", "sample"},
        )

        assert aggregated_layouts.is_empty


class TestGeneratePrecomputedLayoutsForComponents:
    @pytest.fixture(autouse=True)
    def mock_pool_executor(self):
        """Mock the pool executor to avoid running tests in parallel.

        The overhead of running in separate processes is not worth it here.
        """

        class MockPoolExecutor:
            def __init__(self):
                pass

            def imap(self, func, *args, **kwargs):
                yield from map(func, *args)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with mock.patch(
            "pixelator.pixeldataset.precomputed_layouts.get_pool_executor",
        ) as mock_pool_executor:
            mock_pool_executor.return_value = MockPoolExecutor()
            yield mock_pool_executor

    @pytest.fixture(name="pixel_dataset")
    def pixel_dataset_fixture(self, setup_basic_pixel_dataset):
        (dataset, *_) = setup_basic_pixel_dataset
        yield dataset

    def test_generate_precomputed_layouts_for_components_with_all_components(
        self, pixel_dataset
    ):
        precomputed_layouts = generate_precomputed_layouts_for_components(pixel_dataset)

        assert isinstance(precomputed_layouts, PreComputedLayouts)
        assert not precomputed_layouts.is_empty
        df = precomputed_layouts.to_df()

        assert set(df.columns) == {
            "x",
            "y",
            "z",
            "x_norm",
            "y_norm",
            "z_norm",
            "graph_projection",
            "layout",
            "component",
            "name",
            "pixel_type",
            "index",
        } | set(pixel_dataset.adata.var.index)

        assert set(df["component"]) == set(pixel_dataset.adata.obs.index)
        assert set(df["layout"]) == {"wpmds_3d"}

    def test_generate_precomputed_layouts_for_components_with_specific_components(
        self, pixel_dataset
    ):
        components = {"2ac2ca983a4b82dd", "6ed5d4e4cfe588bd"}
        precomputed_layouts = generate_precomputed_layouts_for_components(
            pixel_dataset, components=components
        )

        assert isinstance(precomputed_layouts, PreComputedLayouts)
        assert not precomputed_layouts.is_empty
        df = precomputed_layouts.to_df()
        assert set(df["component"]) == components

    def test_generate_precomputed_layouts_for_components_without_node_marker_counts(
        self, pixel_dataset
    ):
        precomputed_layouts = generate_precomputed_layouts_for_components(
            pixel_dataset, add_node_marker_counts=False
        )

        assert isinstance(precomputed_layouts, PreComputedLayouts)
        assert not precomputed_layouts.is_empty
        df = precomputed_layouts.to_df()
        assert set(df.columns) == {
            "x",
            "y",
            "z",
            "x_norm",
            "y_norm",
            "z_norm",
            "graph_projection",
            "layout",
            "component",
            "name",
            "pixel_type",
            "index",
        }

    def test_generate_precomputed_layouts_for_components_with_multiple_layout_algorithms(
        self, pixel_dataset
    ):
        layout_algorithms = ["pmds_3d", "pmds"]
        precomputed_layouts = generate_precomputed_layouts_for_components(
            pixel_dataset, layout_algorithms=layout_algorithms
        )

        assert isinstance(precomputed_layouts, PreComputedLayouts)
        assert not precomputed_layouts.is_empty

        df = precomputed_layouts.to_df()
        assert set(df["layout"]) == {"pmds", "pmds_3d"}

        # When we mix 2d and 3d layouts the z-dimension should be NaN
        assert np.all(df[df["layout"] == "pmds"]["z"].isna())
        assert np.all(df[df["layout"] == "pmds"]["z_norm"].isna())

    def test_generate_precomputed_layouts_for_components_with_single_layout_algorithm(
        self, pixel_dataset
    ):
        layout_algorithm = "pmds_3d"
        precomputed_layouts = generate_precomputed_layouts_for_components(
            pixel_dataset, layout_algorithms=layout_algorithm
        )

        assert isinstance(precomputed_layouts, PreComputedLayouts)
        assert not precomputed_layouts.is_empty

        df = precomputed_layouts.to_df()
        assert set(df["layout"]) == {"pmds_3d"}

    @pytest.mark.test_this
    def test_generate_precomputed_layouts_on_to_small_components(self):
        edgelist = pd.DataFrame.from_dict(
            {
                "upia": ["A", "B", "C"],
                "upib": ["B", "C", "A"],
                "umi": ["G", "H", "I"],
                "sequence": ["J", "K", "L"],
                "component": [
                    "2ac2ca983a4b82dd",
                    "2ac2ca983a4b82dd",
                    "2ac2ca983a4b82dd",
                ],
                "marker": ["CD3", "CD3", "CD3"],
                "count": [1, 1, 1],
            }
        )

        class MockAnnData:
            def __init__(self):
                self.n_obs = 10

            def copy(self):
                return self

            @property
            def obs(self):
                return pd.DataFrame(index=edgelist["component"].unique())

            @property
            def var(self):
                return pd.DataFrame(index=edgelist["marker"].unique())

        pixel_dataset = PixelDataset.from_data(MockAnnData(), edgelist=edgelist)
        layout_algorithm = "wpmds_3d"
        with pytest.raises(ValueError):
            generate_precomputed_layouts_for_components(
                pixel_dataset, layout_algorithms=layout_algorithm
            )


@pytest.mark.integration_test
class TestGeneratePrecomputedLayoutsForComponentsIntegrationTest:
    """These tests will include multiprocessing, but cover less than the tests above."""

    @pytest.fixture(name="pixel_dataset")
    def pixel_dataset_fixture(self, setup_basic_pixel_dataset):
        (dataset, *_) = setup_basic_pixel_dataset
        yield dataset

    def test_generate_precomputed_layouts_for_components_with_all_components(
        self, pixel_dataset
    ):
        precomputed_layouts = generate_precomputed_layouts_for_components(
            pixel_dataset,
            add_node_marker_counts=True,
            layout_algorithms=["pmds", "pmds_3d"],
        )

        assert isinstance(precomputed_layouts, PreComputedLayouts)
        assert not precomputed_layouts.is_empty
        df = precomputed_layouts.to_df()

        assert set(df.columns) == {
            "x",
            "y",
            "z",
            "x_norm",
            "y_norm",
            "z_norm",
            "graph_projection",
            "layout",
            "component",
            "name",
            "pixel_type",
            "index",
        } | set(pixel_dataset.adata.var.index)

        assert set(df["component"]) == set(pixel_dataset.adata.obs.index)
        assert set(df["layout"]) == {"pmds_3d", "pmds"}

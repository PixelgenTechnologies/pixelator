"""Tests for on-the-fly Layouts on PNAPixelDataset.

Copyright © 2026 Pixelgen Technologies AB.
"""

from inspect import signature

import pandas as pd
import polars as pl
import pytest

from pixelator.common.graph.backends.protocol import (
    DEFAULT_LAYOUT_ALGORITHM,
    SupportedLayoutAlgorithm,
)
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.layouts import Layouts

# Tiny fixture graphs are too small for spectral_3d; use a force-directed
# algorithm to exercise the public Layouts API quickly.
_FAST_ALGORITHM: SupportedLayoutAlgorithm = "fruchterman_reingold_3d"
_FAST_2D_ALGORITHM: SupportedLayoutAlgorithm = "fruchterman_reingold"

_COORD_COLUMNS = {
    "sample",
    "component",
    "graph_projection",
    "index",
    "layout",
    "pixel_type",
    "x",
    "y",
    "z",
}


def _one_component(pxl_dataset: PNAPixelDataset) -> str:
    return pxl_dataset.adata().obs.index[0]


class TestDefaultLayoutAlgorithm:
    def test_default_is_spectral_3d(self):
        assert DEFAULT_LAYOUT_ALGORITHM == "coarsened_pmds_3d"

    def test_layouts_method_uses_the_shared_default(self):
        assert (
            signature(PNAPixelDataset.layouts).parameters["algorithm"].default
            == DEFAULT_LAYOUT_ALGORITHM
        )

    def test_layout_coordinates_uses_the_shared_default(self):
        assert (
            signature(PNAGraph.layout_coordinates)
            .parameters["layout_algorithm"]
            .default
            == DEFAULT_LAYOUT_ALGORITHM
        )


class TestLayoutsApi:
    def test_to_df_has_precomputed_coordinate_columns(
        self, pxl_dataset: PNAPixelDataset
    ):
        component_id = _one_component(pxl_dataset)
        df = (
            pxl_dataset.filter(components=component_id)
            .layouts(algorithm=_FAST_ALGORITHM, add_marker_counts=False)
            .to_df()
        )
        assert isinstance(df, pd.DataFrame)
        assert _COORD_COLUMNS.issubset(df.columns)
        assert set(df["layout"].unique()) == {_FAST_ALGORITHM}
        assert set(df["component"].unique()) == {component_id}
        assert "A" in set(df["pixel_type"])
        assert "B" in set(df["pixel_type"])

    def test_to_polars_returns_polars(self, pxl_dataset: PNAPixelDataset):
        component_id = _one_component(pxl_dataset)
        df = (
            pxl_dataset.filter(components=component_id)
            .layouts(algorithm=_FAST_ALGORITHM, add_marker_counts=False)
            .to_polars()
        )
        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

    def test_first_is_a_layouts_with_one_component(self, pxl_dataset: PNAPixelDataset):
        layouts = pxl_dataset.layouts(
            algorithm=_FAST_ALGORITHM, add_marker_counts=False
        )
        first = layouts.first()
        assert isinstance(first, Layouts)
        df = first.to_df()
        assert df["component"].nunique() == 1
        assert df["component"].iloc[0] == pxl_dataset.adata().obs.index[0]

    def test_iterator_yields_one_frame_per_component(
        self, pxl_dataset: PNAPixelDataset
    ):
        layouts = pxl_dataset.layouts(
            algorithm=_FAST_ALGORITHM, add_marker_counts=False
        )
        items = list(layouts.iterator())
        assert {component_id for component_id, _ in items} == pxl_dataset.components()
        assert all(isinstance(frame, pd.DataFrame) for _, frame in items)

    def test_always_computes_and_does_not_read_the_layouts_table(
        self, pxl_dataset: PNAPixelDataset
    ):
        component_id = _one_component(pxl_dataset)
        stored = set(
            pxl_dataset.filter(components=component_id)
            .precomputed_layouts(add_marker_counts=False)
            .to_df()["layout"]
            .unique()
        )
        computed = set(
            pxl_dataset.filter(components=component_id)
            .layouts(algorithm=_FAST_ALGORITHM, add_marker_counts=False)
            .to_df()["layout"]
            .unique()
        )
        assert computed == {_FAST_ALGORITHM}
        assert computed != stored

    def test_kwargs_are_forwarded(self, pxl_dataset: PNAPixelDataset):
        component_id = _one_component(pxl_dataset)
        df = (
            pxl_dataset.filter(components=component_id)
            .layouts(
                algorithm=_FAST_ALGORITHM,
                add_marker_counts=False,
                random_seed=1,
            )
            .to_df()
        )
        assert len(df) > 0

    def test_spherical_norm_adds_norm_columns(self, pxl_dataset: PNAPixelDataset):
        component_id = _one_component(pxl_dataset)
        df = (
            pxl_dataset.filter(components=component_id)
            .layouts(
                algorithm=_FAST_ALGORITHM,
                add_marker_counts=False,
                add_spherical_norm=True,
            )
            .to_polars()
        )
        assert {"x_norm", "y_norm", "z_norm"}.issubset(df.columns)

    def test_spherical_norm_adds_norm_columns_for_2d_algorithm(
        self, pxl_dataset: PNAPixelDataset
    ):
        component_id = _one_component(pxl_dataset)
        df = (
            pxl_dataset.filter(components=component_id)
            .layouts(
                algorithm=_FAST_2D_ALGORITHM,
                add_marker_counts=False,
                add_spherical_norm=True,
            )
            .to_polars()
        )
        assert "z" not in df.columns
        assert {"x_norm", "y_norm"}.issubset(df.columns)
        assert "z_norm" not in df.columns


class TestPrecomputedLayoutsDeprecation:
    @pytest.mark.filterwarnings("always::DeprecationWarning")
    def test_precomputed_layouts_emits_deprecation_warning(
        self, pxl_dataset: PNAPixelDataset
    ):
        with pytest.warns(DeprecationWarning, match="layouts\\(\\)"):
            pxl_dataset.precomputed_layouts()

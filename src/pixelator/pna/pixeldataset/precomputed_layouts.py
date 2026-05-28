"""Precomputed layouts wrapper for PNA pixel datasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from pixelator.pna.pixeldataset.io import PixelDataViewer, QueryBuilder
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from pixelator.pna.utils import normalize_input_to_list, normalize_input_to_set


class PreComputedLayouts:
    """Representation of precomputed layouts.

    This contains precomputed layouts for one or more components.
    """

    def __init__(
        self,
        view: PixelDataViewer,
        components: str | Iterable[str] | None = None,
        adata_helper: AnnDataHelper | None = None,
        add_marker_counts: bool = True,
        add_spherical_norm: bool = False,
    ):
        """Create a new instance of PreComputedLayouts.

        Args:
            view: View.
            components: Components.
            adata_helper: Adata helper.
            add_marker_counts: Add marker counts.
            add_spherical_norm: Add spherical norm.
        """
        self._view = view
        self._components = normalize_input_to_set(components)
        self._adata_helper = (
            adata_helper
            if adata_helper is not None
            else AnnDataHelper(self._view, components=self._components)
        )
        self._add_marker_counts = add_marker_counts
        self._add_spherical_norm = add_spherical_norm
        self._query_builder = QueryBuilder()

    @property
    def components(self) -> set[str]:
        """Get the component names."""
        return (
            self._components
            if self._components is not None
            else set(
                self._adata_helper.read_adata(
                    add_clr_transform=False, add_log1p_transform=False
                ).obs.index.to_list()
            )
        )

    def _pivot_marker_table(self, df: pl.DataFrame) -> pl.DataFrame:
        """Pivot the joined marker column into marker count columns.

        Args:
            df: Df.
        """
        return (
            df.select(pl.col("*"), val=pl.lit(1))
            .pivot(
                on="marker",
                index=None,
                values="val",
                aggregate_function=pl.len().cast(pl.UInt8),
            )
            .fill_null(0)
        )

    def __len__(self) -> int:
        """Get the nodes in the layouts."""
        query = self._query_builder.layouts_len_query(
            normalize_input_to_list(self._components)
        )
        with self._view.open() as session:
            try:
                return session.execute_scalar(query)
            except duckdb.CatalogException:
                return 0

    def is_empty(self) -> bool:
        """Check if the precomputed layouts are empty."""
        return len(self) == 0

    def to_df(self) -> pd.DataFrame:
        """Get the precomputed layouts as a pandas DataFrame."""
        return self.to_polars().to_pandas()

    def _handle_backwards_comp(self, df: pl.DataFrame) -> pl.DataFrame:
        # If the norm data is in the output (which it is for some legacy files)
        # drop them.
        return df.drop(["x_norm", "y_norm", "z_norm"], strict=False)

    def _post_process(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self._handle_backwards_comp(df)

        if self._add_spherical_norm:
            coordinates = df.select(["x", "y", "z"]).to_numpy()
            normalized_coordinates = pl.DataFrame(
                coordinates / (1 * np.linalg.norm(coordinates, axis=1))[:, None],
                schema={
                    "x_norm": pl.Float32,
                    "y_norm": pl.Float32,
                    "z_norm": pl.Float32,
                },
            )
            df = df.hstack(normalized_coordinates)

        return df

    def to_polars(self) -> pl.DataFrame:
        """Get the precomputed layouts as a polars DataFrame."""
        query = self._query_builder.layouts_query(
            components=normalize_input_to_list(self._components),
            add_marker_counts=self._add_marker_counts,
        )
        with self._view.open() as session:
            try:
                if self._add_marker_counts:
                    layouts = session.execute_eager(query)
                    layouts = self._pivot_marker_table(layouts)
                    layouts = layouts.drop(["umi", "marker"], strict=False)
                else:
                    layouts = session.execute_eager(query)
            except duckdb.CatalogException:
                layouts = pl.DataFrame()
        return self._post_process(layouts)

    def iterator(
        self, return_polars_df: bool = False
    ) -> Iterable[tuple[str, pd.DataFrame | pl.DataFrame]]:
        """Get a stream of layouts.

        Control the trade-off between memory usage and performance by setting
        the `batch_size` parameter. If you set it to a large number, you will
        use more memory, but the performance should overall be better.

        Args:
            return_polars_df: If True, return polars DataFrames, otherwise return pandas DataFrames
        Returns:
            A stream of layouts names and associated layout dataframes
        """
        with self._view.open() as session:
            for component in self.components:
                query = self._query_builder.layouts_query(
                    components=[component],
                    add_marker_counts=self._add_marker_counts,
                )
                try:
                    if self._add_marker_counts:
                        layouts = session.execute_eager(query)
                        layouts = self._pivot_marker_table(layouts)
                        layouts = layouts.drop(["umi", "marker"], strict=False)
                    else:
                        layouts = session.execute_eager(query)
                except duckdb.CatalogException:
                    layouts = pl.DataFrame()
                component_df = self._post_process(layouts)
                if return_polars_df:
                    yield (component, component_df)
                else:
                    yield (component, component_df.to_pandas())

    def describe(self) -> str:
        """Return a description of the PreComputedLayouts."""
        return f"PreComputedLayouts({len(self.components):,} components, {len(self):,} datapoints)"

    def __str__(self):
        """Get a string representation of the PreComputedLayouts."""
        return f"PreComputedLayouts({len(self.components):,} components)"

    def __repr__(self):
        """Get a string representation of the PreComputedLayouts."""
        return str(self)

    def _ipython_display_(self):
        return print(self.describe())

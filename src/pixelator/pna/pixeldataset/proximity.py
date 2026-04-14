"""Proximity wrapper for PNA pixel datasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData

from pixelator.pna.pixeldataset.io import PixelDataViewer, QueryBuilder
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from pixelator.pna.utils import normalize_input_to_list, normalize_input_to_set


class Proximity:
    """Representation of a proximity data.

    This class allows you to access the proximity data as a pandas or polars
    DataFrame. From there you can analyze the data further using your favorite
    data analysis tools.
    """

    def __init__(
        self,
        view: PixelDataViewer,
        components: str | list[str] | set[str] | None = None,
        markers: str | list[str] | set[str] | None = None,
        adata_helper: AnnDataHelper | None = None,
        add_marker_counts: bool = True,
        add_log2_ratio: bool = True,
    ):
        """Create a new instance of Proximity."""
        self._view = view
        self._components = normalize_input_to_set(components)
        self._markers = normalize_input_to_set(markers)
        self._adata_helper = (
            adata_helper
            if adata_helper is not None
            else AnnDataHelper(view, components=self._components, markers=self._markers)
        )
        self._add_marker_counts = add_marker_counts
        self._add_log2_ratio_col = add_log2_ratio
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

    @property
    def markers(self) -> set[str]:
        """Get the marker names."""
        return (
            self._markers
            if self._markers is not None
            else set(
                self._adata_helper.read_adata(
                    add_clr_transform=False, add_log1p_transform=False
                ).var.index.to_list()
            )
        )

    def __len__(self) -> int:
        """Get the number of proximity scores."""
        query = self._query_builder.proximity_len_query(
            normalize_input_to_list(self._components),
            normalize_input_to_list(self._markers),
        )
        with self._view.open() as session:
            return session.execute_scalar(query)

    def is_empty(self):
        """Check if the proximity data is empty."""
        return len(self) == 0

    def _add_marker_counts_to_proximity_df(
        self, adata: AnnData, proximity_df: pl.DataFrame
    ):
        marker_counts = pl.DataFrame(adata.to_df().reset_index())
        node_counts = (
            marker_counts.drop("component")
            .transpose(column_names=marker_counts["component"])
            .sum()
            .transpose(column_names=["node_counts"])
            .with_columns(component=marker_counts["component"])
        )
        marker_counts = marker_counts.unpivot(
            index="component", variable_name="marker", value_name="marker_1_count"
        )
        marker_counts = marker_counts.with_columns(
            pl.col("marker_1_count").alias("marker_2_count")
        )
        # Cast to uint32 integers to lower memory usage
        marker_counts = marker_counts.with_columns(
            marker_1_count=pl.col("marker_1_count").cast(pl.UInt32),
            marker_2_count=pl.col("marker_2_count").cast(pl.UInt32),
        )
        marker_counts = marker_counts.join(
            node_counts,
            on="component",
            how="left",
        )
        marker_counts = marker_counts.with_columns(
            marker_1_freq=pl.col("marker_1_count") / pl.col("node_counts"),
            marker_2_freq=pl.col("marker_2_count") / pl.col("node_counts"),
        )
        proximity_df = (
            proximity_df.join(
                marker_counts.select(
                    ["component", "marker", "marker_1_count", "marker_1_freq"]
                ),
                left_on=["component", "marker_1"],
                right_on=["component", "marker"],
                how="left",
            )
            .join(
                marker_counts.select(
                    ["component", "marker", "marker_2_count", "marker_2_freq"]
                ),
                left_on=["component", "marker_2"],
                right_on=["component", "marker"],
                how="left",
            )
            .with_columns(
                min_count=pl.min_horizontal("marker_1_count", "marker_2_count")
            )
        )

        return proximity_df

    def _add_log2_ratio(self, proximity):
        return proximity.with_columns(
            log2_ratio=np.log2(
                np.maximum(pl.col("join_count"), 1)
                / np.maximum(pl.col("join_count_expected_mean"), 1)
            )  # setting values <1 to 1 to avoid division by zero
        )

    def _post_process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._add_marker_counts:
            adata = self._adata_helper.read_adata(
                add_clr_transform=False, add_log1p_transform=False
            )
            df = self._add_marker_counts_to_proximity_df(adata, df)

        if self._add_log2_ratio_col:
            df = self._add_log2_ratio(df)

        return df

    def to_df(self) -> pd.DataFrame:
        """Get the edgelist as a pandas DataFrame."""
        return self.to_polars().to_pandas()

    def to_polars(self) -> pl.DataFrame:
        """Get the edgelist as a polars DataFrame."""
        query = self._query_builder.proximity_query(
            normalize_input_to_list(self._components),
            normalize_input_to_list(self._markers),
        )
        with self._view.open() as session:
            df = session.execute_lazy(query).collect()
        return self._post_process(df)

    def __str__(self) -> str:
        """Get a string representation of the Proximity."""
        n_proximity = len(self)
        return ", ".join(
            [
                f"Proximity({n_proximity:,} elements",
                f"add_marker_counts={self._add_marker_counts}",
                f"add_logratio={self._add_log2_ratio_col})",
            ]
        )

    def __repr__(self):
        """Get a string representation of the Proximity."""
        return str(self)

    def _ipython_display_(self):
        return print(str(self))

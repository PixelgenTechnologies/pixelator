"""Edgelist wrapper for PNA pixel datasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import polars as pl
import pyarrow as pa

from pixelator.pna.pixeldataset.io import PixelDataViewer, QueryBuilder
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from pixelator.pna.pixeldataset.types import Component
from pixelator.pna.utils import normalize_input_to_list, normalize_input_to_set


class Edgelist:
    """Representation of an edgelist.

    To get the edgelist as a pandas DataFrame, use the `to_df()` method.
    To get the edgelist as a polars DataFrame, use the `to_polars()` method.
    To access the components one at a time use the `iterator()` method.

    For memory efficient access to the edgelist, use the `to_record_batches()` method.
    This allows you to iterate over the edgelist using py.ArrowRecordBatch's.
    This is useful when you have a large edgelist that does not fit into memory,
    and your algorithm can be implemented in a streaming fashion.
    """

    def __init__(
        self,
        view: PixelDataViewer,
        components: str | Iterable[str] | None = None,
    ):
        """Create a new instance of Edgelist."""
        self._view = view
        self._components = normalize_input_to_set(components)
        self._query_builder = QueryBuilder()

    @property
    def components(self) -> set[str]:
        """Get the component names."""
        if self._components:
            return self._components
        adata = AnnDataHelper(self._view).read_adata(
            add_clr_transform=False, add_log1p_transform=False
        )
        return set(adata.obs.index.to_list())

    def _handle_backwards_compatibility(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Handle legacy marker names
        return df.rename({"marker1": "marker_1", "marker2": "marker_2"}, strict=False)

    def __len__(self) -> int:
        """Get the number of edges in the edgelist."""
        query = self._query_builder.edgelist_len_query(
            normalize_input_to_list(self._components)
        )
        with self._view as connection:
            return self._view.execute_scalar(connection, query)

    def is_empty(self) -> bool:
        """Check if the edgelist is empty."""
        return len(self) == 0

    def to_df(self) -> pd.DataFrame:
        """Get the edgelist as a pandas DataFrame."""
        query = self._query_builder.edgelist_query(
            normalize_input_to_list(self.components)
        )
        with self._view as connection:
            df = (
                self._handle_backwards_compatibility(
                    self._view.execute_lazy(connection, query)
                )
                .collect()
                .to_pandas()
            )
        return df

    def to_polars(self) -> pl.DataFrame:
        """Get the edgelist as a polars DataFrame."""
        query = self._query_builder.edgelist_query(
            normalize_input_to_list(self.components)
        )
        with self._view as connection:
            df = self._handle_backwards_compatibility(
                self._view.execute_lazy(connection, query)
            ).collect()
        return df

    def to_record_batches(
        self, batch_size: int = 1_000_000
    ) -> Iterable[pa.RecordBatch]:
        """Get the edgelist as a stream of pyarrow RecordBatches."""
        query = self._query_builder.edgelist_query(
            normalize_input_to_list(self.components)
        )
        with self._view as connection:
            yield from self._view.execute_arrow_reader(
                connection=connection, query=query, batch_size=batch_size
            )

    def _iterator(self) -> Iterable[tuple[str, pl.LazyFrame]]:
        with self._view as connection:
            for component in self.components:
                query = self._query_builder.edgelist_query([component])
                yield (
                    component,
                    self._view.execute_lazy(connection, query),
                )

    def iterator(self) -> Iterable[Component]:
        """Get a stream of components and their graphs.

        :return: A stream of component names and associated graphs
        """
        for name, df in self._iterator():
            yield Component(
                # TODO Doing collect().lazy() does not really make sense. The reason that it is
                # here is that otherwise the object is not pickable, and thus not handled
                # well by the analysis manager. We should revisit this in the future.
                component_id=name,
                frame=self._handle_backwards_compatibility(df).collect().lazy(),
            )

    def __str__(self) -> str:
        """Get a string representation of the Edgelist."""
        n_edges = len(self)
        n_components = len(self.components)
        if n_components <= 5:
            return f"EdgeList({n_edges:,} edges in component set: {self.components})"
        return f"EdgeList({n_edges:,} edges in {n_components} components)"

    def __repr__(self):
        """Get a string representation of the Edgelist."""
        return str(self)

    def _ipython_display_(self):
        """Display the Edgelist in Jupyter notebooks."""
        return print(str(self))

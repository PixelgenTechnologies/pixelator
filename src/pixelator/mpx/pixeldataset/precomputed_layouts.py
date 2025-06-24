"""Pre-computed layouts for components.

This module contains the functionality to generate and work with pre-computed
graph layouts. The primary purpose of this is to allow components to be
visualized quickly in downstream analysis, since layout computation
is a relatively computationally expensive operation.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Protocol

import pandas as pd
import polars as pl
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pixelator.common.exceptions import PixelatorBaseException
from pixelator.common.graph.backends.protocol import SupportedLayoutAlgorithm
from pixelator.common.marks import experimental
from pixelator.common.utils import batched, flatten, get_pool_executor
from pixelator.mpx.graph import Graph

if TYPE_CHECKING:
    from pixelator.mpx.pixeldataset import PixelDataset

import logging

logger = logging.getLogger(__name__)


def _write_parquet(frame: pl.LazyFrame, path: Path, partitioning: list[str]) -> None:
    table = frame.collect().to_arrow()
    file_options = ds.ParquetFileFormat().make_write_options(
        compression="zstd",
    )
    ds.write_dataset(
        table,
        path,
        format="parquet",
        partitioning_flavor="hive",
        partitioning=partitioning,
        file_options=file_options,
        existing_data_behavior="overwrite_or_ignore",
    )


class PreComputedLayoutsEmpty(PixelatorBaseException):
    """Raised when trying to access an empty PreComputedLayouts instance."""


class _DataProvider(Protocol):
    def is_empty(self) -> bool: ...

    def to_df(self, columns: list[str] | None = None) -> pd.DataFrame: ...

    def lazy(self): ...

    def unique_components(self): ...

    def filter(
        self,
        component_ids: str | set[str] | None = None,
        graph_projections: str | set[str] | None = None,
        layout_methods: str | set[str] | None = None,
    ) -> pl.LazyFrame | list[pl.LazyFrame]: ...

    def write_parquet(self, path: Path, partitioning: list[str]) -> None:
        """Write a parquet file to the provided path."""
        ...


class _EmptyDataProvider(_DataProvider):
    def __init__(self) -> None:
        # This class needs no parameters
        pass

    def write_parquet(self, path: Path, partitioning: list[str]) -> None:
        """Write a parquet file to the provided path."""
        return

    def is_empty(self) -> bool:
        return True

    def to_df(self, columns: list[str] | None = None) -> pd.DataFrame:
        raise PreComputedLayoutsEmpty("No layouts available")

    def lazy(self):
        raise PreComputedLayoutsEmpty("No layouts available")

    def unique_components(self):
        return {}

    def filter(
        self,
        component_ids: str | set[str] | None = None,
        graph_projections: str | set[str] | None = None,
        layout_methods: str | set[str] | None = None,
    ) -> list[pl.LazyFrame]:
        raise PreComputedLayoutsEmpty("No layouts available")


class _SingleFrameDataProvider(_DataProvider):
    def __init__(self, lazy_frame: pl.LazyFrame) -> None:
        self._lazy_frame = lazy_frame

    def is_empty(self) -> bool:
        return self.lazy().select(pl.col("component")).first().collect().is_empty()

    def to_df(self, columns: list[str] | None = None) -> pd.DataFrame:
        if columns:
            return self._lazy_frame.select(columns).collect().to_pandas()
        return self.lazy().collect().to_pandas()

    def write_parquet(self, path: Path, partitioning: list[str]) -> None:
        """Write a parquet file to the provided path."""
        _write_parquet(self.lazy(), path, partitioning)

    def lazy(self):
        return self._lazy_frame

    def unique_components(self):
        return set(
            self.lazy()
            .select(pl.col("component").unique())
            .collect()
            .get_column("component")
        )

    def filter(
        self,
        component_ids: str | set[str] | None = None,
        graph_projections: str | set[str] | None = None,
        layout_methods: str | set[str] | None = None,
    ) -> pl.LazyFrame:
        component_ids = PreComputedLayouts._convert_to_set(component_ids)
        graph_projections = PreComputedLayouts._convert_to_set(graph_projections)
        layout_methods = PreComputedLayouts._convert_to_set(layout_methods)

        layouts_lazy_filtered = self.lazy()

        if component_ids:
            layouts_lazy_filtered = layouts_lazy_filtered.filter(
                pl.col("component").is_in(component_ids)
            )

        if graph_projections:
            layouts_lazy_filtered = layouts_lazy_filtered.filter(
                pl.col("graph_projection").is_in(graph_projections)
            )

        if layout_methods:
            layouts_lazy_filtered = layouts_lazy_filtered.filter(
                pl.col("layout").is_in(layout_methods)
            )

        return layouts_lazy_filtered


class _MultiFrameDataProvider(_DataProvider):
    def __init__(self, lazy_frames: Iterable[pl.LazyFrame]) -> None:
        self._lazy_frames = list(lazy_frames)

    def is_empty(self) -> bool:
        return all(
            frame.select(pl.col("component")).first().collect().is_empty()
            # We don't want to use `lazy()` here to skip
            # concatenating which is an unnecessary operation
            # for this
            for frame in self._lazy_frames
        )

    def to_df(self, columns: list[str] | None = None) -> pd.DataFrame:
        if columns:
            return self.lazy().select(columns).collect().to_pandas()
        return self.lazy().collect().to_pandas()

    def lazy(self):
        try:
            return pl.concat(self._lazy_frames, how="diagonal")
        except ValueError:
            raise PreComputedLayoutsEmpty(f"No layouts available: {self._lazy_frames}")

    def unique_components(self):
        return set(
            flatten(
                frame.select(pl.col("component").unique())
                .collect()
                .get_column("component")
                .to_list()
                for frame in self._lazy_frames
            )
        )

    def filter(
        self,
        component_ids: str | set[str] | None = None,
        graph_projections: str | set[str] | None = None,
        layout_methods: str | set[str] | None = None,
    ) -> list[pl.LazyFrame]:
        component_ids = PreComputedLayouts._convert_to_set(component_ids)
        graph_projections = PreComputedLayouts._convert_to_set(graph_projections)
        layout_methods = PreComputedLayouts._convert_to_set(layout_methods)

        def data():
            for frame in self._lazy_frames:
                if component_ids:
                    frame = frame.filter(pl.col("component").is_in(component_ids))

                if graph_projections:
                    frame = frame.filter(
                        pl.col("graph_projection").is_in(graph_projections)
                    )

                if layout_methods:
                    frame = frame.filter(pl.col("layout").is_in(layout_methods))

                if not frame.select(pl.col("component")).first().collect().is_empty():
                    yield frame

        return list(data())

    def write_parquet(self, path: Path, partitioning: list[str]) -> None:
        """Write a parquet file to the provided path."""
        for frame in self._lazy_frames:
            _write_parquet(frame, path, partitioning)


class PreComputedLayouts:
    """Pre-computed layouts for a set of graphs, per component."""

    DEFAULT_PARTITIONING = [
        "graph_projection",
        "layout",
        "component",
    ]

    def __init__(
        self,
        layouts_lazy: pl.LazyFrame | Iterable[pl.LazyFrame] | None,
        partitioning: Optional[list[str]] = None,
    ) -> None:
        """Initialize the PreComputedLayouts instance."""
        if layouts_lazy is None:
            self._data_provider: _DataProvider = _EmptyDataProvider()
        elif isinstance(layouts_lazy, pl.LazyFrame):
            self._data_provider = _SingleFrameDataProvider(layouts_lazy)
        elif isinstance(layouts_lazy, pl.DataFrame):
            self._data_provider = _SingleFrameDataProvider(layouts_lazy.lazy())
        elif isinstance(layouts_lazy, Iterable):
            self._data_provider = _MultiFrameDataProvider(layouts_lazy)
        else:
            raise ValueError("Must be lazy frame or iterable of lazy frames")

        if not partitioning:
            partitioning = PreComputedLayouts.DEFAULT_PARTITIONING
        self._partitioning = partitioning

    @staticmethod
    def create_empty():
        """Create an empty PreComputedLayouts instance."""
        return PreComputedLayouts(None)

    def __repr__(self) -> str:
        """Return a string representation of the PreComputedLayouts instance."""
        has_data = "empty" if self.is_empty else "with data"
        return f"PreComputedLayouts({has_data})"

    def __str__(self) -> str:
        """Return a string representation of the PreComputedLayouts instance."""
        return self.__repr__()

    @property
    def is_empty(self) -> bool:
        """Return True if the layout is empty."""
        return self._data_provider.is_empty()

    @property
    def partitioning(self) -> list[str]:
        """Return the partitioning of the layouts.

        This is used to determine how to partition the data when
        serializing it to disk, for the cases when this is applicable.
        For example when writing hive style parquet files.
        """
        return self._partitioning

    def write_parquet(self, path: Path, partitioning: list[str]) -> None:
        """Write a parquet file to the provided path."""
        self._data_provider.write_parquet(path, partitioning)

    def unique_components(self) -> set[str]:
        """Return the unique components in the layouts."""
        return self._data_provider.unique_components()

    def to_df(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Return the layouts as a pandas DataFrame.

        :param columns: the columns to return, if `None` all columns will be returned
        :return: A pandas DataFrame with the layout(s)
        """
        return self._data_provider.to_df(columns)

    @property
    def lazy(self) -> pl.LazyFrame:
        """Return the layouts as a polars LazyFrame."""
        return self._data_provider.lazy()

    def filter(
        self,
        component_ids: str | set[str] | None = None,
        graph_projection: str | set[str] | None = None,
        layout_method: str | set[str] | None = None,
    ) -> PreComputedLayouts:
        """Filter the layouts based on the provided criteria.

        :param component_ids: the component ids to filter on
        :param graph_projection: the graph projection to filter on
        :param layout_method: the layout method to filter on
        :return: A new PreComputedLayouts instance with the filtered layouts
        :rtype: PreComputedLayouts
        """
        return PreComputedLayouts(
            self._data_provider.filter(component_ids, graph_projection, layout_method),
            partitioning=self.partitioning,
        )

    def component_iterator(
        self,
        component_ids: Optional[set[str] | str] = None,
        graph_projections: Optional[set[str] | str] = None,
        layout_methods: Optional[set[str] | str] = None,
        columns: Optional[list[str]] = None,
    ) -> Iterable[pd.DataFrame]:
        """Get an iterator over the components provided.

        If `component_ids` is not provided, all components will be returned.

        Providing additional parameters will filter the layouts based on these
        criteria.

        :param component_ids: the component ids to filter on, if `None` all components
                              will be returned.
        :param graph_projections: the graph projections to filter on
        :param layout_methods: the layout methods to filter on
        :param columns: the columns to return, if `None` all columns will be returned
        :yields pd.DataFrame: A generator over the components where each dataframe contains the layout(s)
                  for that component
        """
        if not component_ids:
            unique_components = self._data_provider.unique_components()
        else:
            unique_components = self._convert_to_set(component_ids)  # type: ignore

        # We read in batches since it makes the read operations slightly
        # faster than iterating them one at the time
        for component_ids in batched(unique_components, 20):
            data = self.filter(
                component_ids=component_ids,
                graph_projection=graph_projections,
                layout_method=layout_methods,
            )

            for _, df in data.lazy.collect().group_by("component"):
                yield df.select(columns if columns else pl.all()).to_pandas()

    @staticmethod
    def _convert_to_set(
        value: Optional[str | set[str]] = None,
    ) -> Optional[set[str]]:
        if value is not None:
            value = set([value]) if isinstance(value, str) else set(value)
        return value

    def copy(self) -> PreComputedLayouts:
        """Copy the PreComputedLayouts instance."""
        return PreComputedLayouts(
            self.lazy.clone(), partitioning=self.partitioning.copy()
        )


def aggregate_precomputed_layouts(
    pxl_datasets: Iterable[tuple[str, PixelDataset | None]],
    all_markers: set[str],
) -> PreComputedLayouts:
    """Aggregate precomputed layouts into a single PreComputedLayouts instance."""

    def zero_fill_missing_markers(
        lazyframe: pl.LazyFrame, all_markers: set[str]
    ) -> pl.LazyFrame:
        missing_markers = all_markers - set(lazyframe.collect_schema().names())
        return lazyframe.with_columns(
            **{marker: pl.lit(0) for marker in missing_markers}
        )

    def data():
        for sample_name, pxl_dataset in pxl_datasets:
            layout = pxl_dataset.precomputed_layouts
            if layout is None:
                continue
            if layout.is_empty:
                continue
            layout_with_name = layout.lazy.with_columns(
                sample=pl.lit(sample_name),
                component=pl.concat_str(
                    pl.col("component"), pl.lit(sample_name), separator="_"
                ),
            ).pipe(zero_fill_missing_markers, all_markers=all_markers)
            yield layout_with_name

    try:
        return PreComputedLayouts(
            data(),
            partitioning=["sample"] + PreComputedLayouts.DEFAULT_PARTITIONING,
        )
    except ValueError:
        return PreComputedLayouts.create_empty()


def _zero_fill_missing_markers(dataframe: pd.DataFrame, all_markers) -> pd.DataFrame:
    missing_markers = all_markers - set(dataframe.columns.to_list())
    for marker in missing_markers:
        dataframe[marker] = 0
    return dataframe


def _compute_layouts(
    edgelist: pl.DataFrame,
    add_node_marker_counts: bool,
    layout_algorithms: list[SupportedLayoutAlgorithm] | SupportedLayoutAlgorithm,
    all_markers: list[str],
):
    if isinstance(layout_algorithms, str):
        layout_algorithms = [layout_algorithms]

    def data():
        for component, df in edgelist.group_by("component"):
            logger.debug("Computing for component %s", component)
            graph = Graph.from_edgelist(
                df.lazy(),
                add_marker_counts=add_node_marker_counts,
                simplify=True,
                use_full_bipartite=True,
                convert_indices_to_integers=False,
            )
            for layout_algorithm in layout_algorithms:
                yield (
                    pl.DataFrame(
                        graph.layout_coordinates(
                            layout_algorithm=layout_algorithm,
                            get_node_marker_matrix=add_node_marker_counts,
                            cache=False,
                            only_keep_a_pixels=False,
                        )
                        .pipe(
                            _zero_fill_missing_markers
                            # If we don't want to add the marker counts
                            # we just pass this through an identity function
                            if add_node_marker_counts
                            else lambda df, all_markers: df,
                            all_markers=all_markers,
                        )
                        .sort_index(axis=1)
                    )
                    .with_columns(
                        component=pl.lit(component[0]),
                        graph_projection=pl.lit("bipartite"),
                        layout=pl.lit(layout_algorithm),
                    )
                    .lazy()
                )

    result = pl.concat(data(), how="diagonal")
    return result


def _wrap_get_layouts(d):
    """Deconstruct a tuple of args to their corresponding argument.

    This is only necessary since imap does not support multiple arguments
    """
    (edgelist, add_node_marker_counts, layout_algorithms, all_markers) = d
    return _compute_layouts(
        edgelist, add_node_marker_counts, layout_algorithms, all_markers
    )


def _data(
    pixel_dataset: PixelDataset,
    components,
    add_node_marker_counts,
    layout_algorithms,
    all_markers,
):
    with get_pool_executor() as executor:
        # Batching over the component groups and setting a higher chunksize
        # to speeds up the computation here. I have not extensively
        # benchmarked the exact values here, but they seem to work well
        # enough
        component_group = map(set, batched(components, 20))
        yield from executor.imap(
            _wrap_get_layouts,
            (
                (
                    pixel_dataset.edgelist_lazy.filter(
                        pl.col("component").is_in(components)
                    ).collect(),
                    add_node_marker_counts,
                    layout_algorithms,
                    all_markers,
                )
                for components in component_group
            ),
            chunksize=1,
        )


@experimental
def generate_precomputed_layouts_for_components(
    pixel_dataset: PixelDataset,
    components: set[str] | None = None,
    add_node_marker_counts: bool = True,
    layout_algorithms: SupportedLayoutAlgorithm
    | list[SupportedLayoutAlgorithm] = "wpmds_3d",
) -> PreComputedLayouts:
    """Generate precomputed layouts for the components in the PixelDataset.

    :param pixel_dataset: the PixelDataset to generate the layouts for
    :param components: the components to generate the layouts for, if `None` all components will be used
    :param add_node_marker_counts: whether to add the marker counts to the layout. If you don't need them
                                   you can set this to false and speed up the computations.
    :param layout_algorithm: the layout algorithm to use
    """
    if components is None:
        components = set(pixel_dataset.adata.obs.index)

    # This is a bit of a hack to handle that different
    # graphs might have different markers, but for the end result
    # to be valid we need to have the same markers across all
    # components
    all_markers = set(pixel_dataset.adata.var.index.to_list())

    return PreComputedLayouts(
        _data(
            pixel_dataset=pixel_dataset,
            components=components,
            add_node_marker_counts=add_node_marker_counts,
            layout_algorithms=layout_algorithms,
            all_markers=all_markers,
        )
    )

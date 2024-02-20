"""Pre-computed layouts for components.

This module contains the functionality to generate and work with pre-computed
graph layouts. The primary purpose of this is to allow components to be
visualized quickly in downstream analysis, since layout computation
is a relatively computationally expensive operation.

Copyright (c) 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

from multiprocessing import get_context
from typing import TYPE_CHECKING, Iterable, Optional

import pandas as pd
import polars as pl

from pixelator.graph import Graph
from pixelator.exceptions import PixelatorBaseException
from pixelator.marks import experimental
from pixelator.utils import batched

if TYPE_CHECKING:
    from pixelator.pixeldataset import PixelDataset

import logging

logger = logging.getLogger(__name__)


class PreComputedLayoutsEmpty(PixelatorBaseException):
    """Raised when trying to access an empty PreComputedLayouts instance."""


class PreComputedLayouts:
    """Pre-computed layouts for a set of graphs, typically per component."""

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
        self._layouts_lazy = layouts_lazy
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
        try:
            if self.lazy is None:
                return True
            return self.lazy.select(pl.col("component")).first().collect().is_empty()
        except PreComputedLayoutsEmpty:
            return True

    @property
    def partitioning(self) -> list[str]:
        """Return the partitioning of the layouts.

        This is used to determine how to partition the data when
        serializing it to disk, for the cases when this is applicable.
        For example when writing hive style parquet files.
        """
        return self._partitioning

    @property
    def df(self) -> pd.DataFrame:
        """Return the layouts as a pandas DataFrame."""
        return self.lazy.collect().to_pandas()

    @property
    def lazy(self) -> pl.LazyFrame:
        """Return the layouts as a polars LazyFrame."""
        if self._layouts_lazy is None:
            raise PreComputedLayoutsEmpty("No layouts available")
        if isinstance(self._layouts_lazy, Iterable):
            return pl.concat(self._layouts_lazy, rechunk=False)
        return self._layouts_lazy

    @property
    def raw_iterator(self) -> Iterable[pl.LazyFrame]:
        """Return the raw iterator of the layouts.

        Please note that the raw iterator does not necessarily return
        one component per lazy frame. Instead if will return the lazy frames
        as they were provided to the PreComputedLayouts instance.

        This is useful e.g. to write the data lazily and efficiently to disk
        but in many end-user scenarios it is probably better to work either
        with `filter` or `component_iterator`.
        """
        if not isinstance(self._layouts_lazy, Iterable):
            return [self._layouts_lazy]  # type: ignore
        return self._layouts_lazy

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
            self._filter_layouts(
                self.lazy, component_ids, graph_projection, layout_method
            ),
            partitioning=self.partitioning,
        )

    def component_iterator(
        self,
        component_ids: Optional[set[str] | str] = None,
        graph_projections: Optional[set[str] | str] = None,
        layout_methods: Optional[set[str] | str] = None,
    ) -> Iterable[pd.DataFrame]:
        """Get an iterator over the components provided.

        If `component_ids` is not provided, all components will be returned.

        Providing additional parameters will filter the layouts based on these
        criteria.

        :param component_ids: the component ids to filter on, if `None` all components
                              will be returned.
        :param graph_projections: the graph projections to filter on
        :param layout_methods: the layout methods to filter on
        :yields pd.DataFrame: A generator over the components where each dataframe contains the layout(s)
                  for that component
        """
        if not component_ids:
            unique_components = set(
                self.lazy.select(pl.col("component").unique())
                .collect()
                .get_column("component")
            )
        else:
            unique_components = self._convert_to_set(component_ids)  # type: ignore

        for component in unique_components:
            yield self._filter_layouts(
                self.lazy, component, graph_projections, layout_methods
            ).collect().to_pandas()

    @staticmethod
    def _convert_to_set(
        value: Optional[str | set[str]] = None,
    ) -> Optional[set[str]]:
        if value is not None:
            value = set([value]) if isinstance(value, str) else set(value)
        return value

    @staticmethod
    def _filter_layouts(
        layouts_lazy: pl.LazyFrame,
        component_ids: str | set[str] | None = None,
        graph_projections: str | set[str] | None = None,
        layout_methods: str | set[str] | None = None,
    ) -> pl.LazyFrame:
        component_ids = PreComputedLayouts._convert_to_set(component_ids)
        graph_projections = PreComputedLayouts._convert_to_set(graph_projections)
        layout_methods = PreComputedLayouts._convert_to_set(layout_methods)

        if component_ids:
            layouts_lazy = layouts_lazy.filter(pl.col("component").is_in(component_ids))

        if graph_projections:
            layouts_lazy = layouts_lazy.filter(
                pl.col("graph_projection").is_in(graph_projections)
            )

        if layout_methods:
            layouts_lazy = layouts_lazy.filter(pl.col("layout").is_in(layout_methods))

        return layouts_lazy

    def copy(self) -> PreComputedLayouts:
        """Copy the PreComputedLayouts instance."""
        return PreComputedLayouts(
            self.lazy.clone(), partitioning=self.partitioning.copy()
        )


def aggregate_precomputed_layouts(
    precomputed_layouts: Iterable[tuple[str, PreComputedLayouts | None]],
    all_markers: set[str],
) -> PreComputedLayouts:
    """Aggregate precomputed layouts into a single PreComputedLayouts instance."""

    def zero_fill_missing_markers(
        lazyframe: pl.LazyFrame, all_markers: set[str]
    ) -> pl.LazyFrame:
        missing_markers = all_markers - set(lazyframe.columns)
        return lazyframe.with_columns(
            **{marker: pl.lit(0) for marker in missing_markers}
        )

    def data():
        for sample_name, layout in precomputed_layouts:
            if layout is None:
                continue
            layout_with_name = layout.lazy.with_columns(
                sample=pl.lit(sample_name)
            ).pipe(zero_fill_missing_markers, all_markers=all_markers)
            yield layout_with_name

    return PreComputedLayouts(
        pl.concat(data(), rechunk=False),
        partitioning=["sample"] + PreComputedLayouts.DEFAULT_PARTITIONING,
    )


# TODO The code below this point is not yet tested, and should be considered
# experimental. It is likely to change in the future. It will be exposed as
# public to this method in the future.


def _zero_fill_missing_markers(dataframe: pd.DataFrame, all_markers) -> pd.DataFrame:
    missing_markers = all_markers - set(dataframe.columns.to_list())
    for marker in missing_markers:
        dataframe[marker] = 0
    return dataframe


def _get_layout(
    edgelist: pl.DataFrame,
    add_node_marker_counts,
    layout_algorithm,
    all_markers,
):
    def data():
        for component, df in edgelist.group_by("component"):
            logger.debug("Computing for component %s", component)
            yield (
                pl.DataFrame(
                    Graph.from_edgelist(
                        df.lazy(),
                        add_marker_counts=add_node_marker_counts,
                        simplify=True,
                        use_full_bipartite=True,
                    )
                    .layout_coordinates(
                        layout_algorithm=layout_algorithm,
                        get_node_marker_matrix=add_node_marker_counts,
                        cache=False,
                    )
                    .pipe(_zero_fill_missing_markers, all_markers=all_markers)
                    .sort_index(axis=1)
                ).with_columns(
                    component=pl.lit(component),
                    graph_projection=pl.lit("bipartite"),
                    layout=pl.lit(layout_algorithm),
                )
            )

    result = pl.concat(data())
    return result


def _fn(d):
    (edgelist, add_node_marker_counts, layout_algorithm, all_markers) = d
    return _get_layout(edgelist, add_node_marker_counts, layout_algorithm, all_markers)


def _data(
    pixel_dataset: PixelDataset,
    components,
    add_node_marker_counts,
    layout_algorithm,
    all_markers,
):
    processes = 10

    # with ProcessPoolExecutor(max_workers=processes) as executor:
    with get_context("spawn").Pool(processes=processes) as executor:
        # Batch over the component groups to speed up the computation
        # fine tune this later
        component_group = map(set, batched(components, 10))
        yield from executor.imap(
            _fn,
            (
                (
                    pixel_dataset.edgelist_lazy.filter(
                        pl.col("component").is_in(components)
                    ).collect(),
                    add_node_marker_counts,
                    layout_algorithm,
                    all_markers,
                )
                for components in component_group
            ),
            chunksize=1,
        )


@experimental
def _generate_precomputed_layouts_for_components(
    pixel_dataset: PixelDataset,
    components: set[str] | None = None,
    add_node_marker_counts: bool = False,
    layout_algorithm="pmds_3d",
) -> PreComputedLayouts:
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
            layout_algorithm=layout_algorithm,
            all_markers=all_markers,
        )
    )

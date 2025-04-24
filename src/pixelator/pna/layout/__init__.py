"""Generate layouts for components in a Pixel dataset.

Copyright © 2024 Pixelgen Technologies AB.
"""

import tempfile
from pathlib import Path
from typing import Iterable

import polars as pl

from pixelator.mpx.graph.backends.protocol import SupportedLayoutAlgorithm
from pixelator.pna.analysis_engine import (
    PerComponentTask,
)
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile


class CreateLayout(PerComponentTask):
    """Run one or more layout algorithms on a component."""

    TASK_NAME = "layout"

    def __init__(
        self,
        layout_algorithms: list[SupportedLayoutAlgorithm],
        algorithm_kwargs: dict | None = None,
    ) -> None:
        """Create a new CreateLayout instance.

        :param layout_algorithms: The layout algorithms to run.
        :param algorithm_kwargs: Additional keyword arguments to pass to the layout algorithms.
        """
        self._layout_algorithms = layout_algorithms
        self._algorithm_kwargs = algorithm_kwargs or {}

    def run_on_component_graph(
        self, component: PNAGraph, component_id: str
    ) -> pl.LazyFrame:
        """Run the layout on a component.

        :param component: The component to run the analysis on. Either a Graph or a LazyFrame.
        :param component_id: The id of the component.
        :return: a LazyFrame containing the layout data.
        :raises TypeError: If the component is not a Graph or a LazyFrame.
        """
        results = []
        for algo in self._layout_algorithms:
            # TODO to get things working working with setting a different
            # k for pmgs w need to pass the weights from here, or change
            # the code in pixelator to have that as a parameter
            layout = pl.DataFrame(
                component.layout_coordinates(
                    algo,
                    get_node_marker_matrix=False,
                    **self._algorithm_kwargs,
                )
            )
            layout = layout.with_columns(
                component=pl.lit(component_id),
                graph_projection=pl.lit("full"),
                layout=pl.lit(algo),
            )
            results.append(layout)

        concatenated = pl.concat(results, how="vertical").lazy()
        return concatenated

    def run_on_component_edgelist(
        self, component: pl.LazyFrame, component_id: str
    ) -> pl.LazyFrame:
        """Run the layout on a component.

        :param component: The component to run the analysis on. Either a Graph or a LazyFrame.
        :param component_id: The id of the component.
        :return: a LazyFrame containing the layout data.
        """
        raise NotImplementedError

    def concatenate_data(self, data: Iterable[pl.LazyFrame]) -> pl.LazyFrame:
        """Concatenate the data. Override this if you need custom concatenation behavior."""
        return pl.concat(data, how="vertical_relaxed")

    def add_to_pixel_file(self, data: pl.LazyFrame, pxl_file_target: PxlFile) -> None:
        """Add the data in the right place in the pxl_dataset."""
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
            tmp_file = Path(tmp_file.name)  # type: ignore
            data.sink_parquet(tmp_file)  # type: ignore
            with PixelFileWriter(pxl_file_target.path) as writer:
                writer.write_layouts(tmp_file)

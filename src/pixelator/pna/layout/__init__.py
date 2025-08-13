"""Generate layouts for components in a Pixel dataset.

Copyright © 2024 Pixelgen Technologies AB.
"""

import os
import tempfile
from pathlib import Path
from typing import Iterable

import polars as pl

from pixelator.common.graph.backends.protocol import SupportedLayoutAlgorithm
from pixelator.pna import read
from pixelator.pna.analysis_engine import (
    PerComponentTask,
)
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PNAPixelDataset
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
        super().__init__()
        self._layout_algorithms = layout_algorithms
        self._algorithm_kwargs = algorithm_kwargs or {}
        self.pxl_dataset: PNAPixelDataset | None = None

    def set_dataset(self, pxl_file_path: Path):
        """Specify a dataset to enable analysis being run directly from component IDs."""
        self.pxl_dataset = read(pxl_file_path)

    def run_from_component_id(self, component_id: str):
        """Run the layout on a component specified by its ID.

        This only works when the pxl_dataset is set so that components
        are directly accessible through their IDs.

        :param component_id: The id of the component.
        :return: a LazyFrame containing the layout data.
        """
        edgelist = (
            self.pxl_dataset.filter(components=[component_id])  # type: ignore
            .edgelist()
            .to_polars()
            .lazy()
        )
        res = self.run_on_component_edgelist(edgelist, component_id)
        return res

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

        concatenated = pl.concat(results, how="vertical")
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        concatenated.write_parquet(Path(tmp_file.name))
        return pl.LazyFrame({"filenames": [tmp_file.name]})

    def run_on_component_edgelist(
        self, component: pl.LazyFrame, component_id: str
    ) -> pl.LazyFrame:
        """Run the layout on a component.

        :param component: The component to run the analysis on. Either a Graph or a LazyFrame.
        :param component_id: The id of the component.
        :return: a LazyFrame containing the layout data.
        """
        graph = PNAGraph.from_edgelist(component)
        result = self.run_on_component_graph(graph, component_id)
        return result

    def concatenate_data(self, data: Iterable[pl.LazyFrame]) -> pl.LazyFrame:
        """Concatenate the data. Override this if you need custom concatenation behavior."""
        return pl.concat(data, how="vertical_relaxed")

    def add_to_pixel_file(self, data: pl.LazyFrame, pxl_file_target: PxlFile) -> None:
        """Add the data in the right place in the pxl_dataset."""
        tmp_component_files = data.collect()

        with PixelFileWriter(pxl_file_target.path) as writer:
            for fname in tmp_component_files["filenames"]:
                writer.write_layouts(Path(fname), append=True)
                os.remove(Path(fname))

"""Analysis engine capable of running a list of analysis functions on each component in a pixeldataset.

Copyright © 2024 Pixelgen Technologies AB.
"""

import logging
from collections import defaultdict
from functools import partial
from queue import Queue
from typing import Callable, Iterable, Protocol

import pandas as pd

from pixelator.common.utils import (
    get_process_pool_executor,
)
from pixelator.mpx.graph import Graph
from pixelator.mpx.pixeldataset import PixelDataset

logger = logging.getLogger(__name__)


class PerComponentAnalysis(Protocol):
    """Protocol for analysis functions that are run on each component in a PixelDataset."""

    ANALYSIS_NAME: str

    def run_on_component(self, component: Graph, component_id: str) -> pd.DataFrame:
        """Run the analysis on this component."""
        ...

    def concatenate_data(self, data: Iterable[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate the data. Override this if you need custom concatenation behavior."""
        try:
            scores = pd.concat(data, axis=0)
            return scores
        except ValueError as error:
            logger.error(f"No data was found to compute {self.ANALYSIS_NAME}")
            raise error

    def post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Post process the data (e.g. adjust p-values). Override this if your data needs post processing."""
        return data

    def add_to_pixel_dataset(
        self, data: pd.DataFrame, pxl_dataset: PixelDataset
    ) -> PixelDataset:
        """Add the data in the right place in the pxl_dataset."""
        ...

    def parameters(self) -> dict:
        """Return the parameters of the `PerComponentAnalysis`.

        This is used e.g. to store the metadata the parameters of the analysis
        in the run metadata.
        """
        return {f"{self.ANALYSIS_NAME}": vars(self)}


class _AnalysisManager:
    """Analysis manager that can run a number of analysis across a stream of components.

    The analysis manager is responsible for hooking up the analysis functions and making
    them run on each component in the stream. The main workflow it uses is outlined in the
    `execute` method.
    """

    def __init__(
        self,
        analysis_to_run: Iterable[PerComponentAnalysis],
        component_stream: Iterable[tuple[str, Graph]],
    ):
        self.analysis_to_run = {
            analysis.ANALYSIS_NAME: analysis for analysis in analysis_to_run
        }
        self.component_stream = component_stream

    def _prepare_computation(
        self,
    ) -> Iterable[tuple[str, Callable[[Graph, str], pd.DataFrame]]]:
        for component_id, component_graph in self.component_stream:
            for _analysis_name, analysis in self.analysis_to_run.items():
                yield (
                    _analysis_name,
                    partial(
                        analysis.run_on_component,
                        component=component_graph,
                        component_id=component_id,
                    ),
                )

    def _execute_computations_in_parallel(self, prepared_computations):
        futures = Queue()
        with get_process_pool_executor() as executor:
            for analysis_name, func in prepared_computations:
                logger.debug("Putting %s in the queue for analysis", analysis_name)
                future = executor.submit(func)
                futures.put((analysis_name, future))

            while not futures.empty():
                key, future = futures.get()
                if future.done():
                    logger.debug("Future for %s is done", key)
                    yield (key, future.result())
                else:
                    futures.put((key, future))

    def _post_process(self, per_component_results):
        concatenated_data = defaultdict(list)
        for key, data in per_component_results:
            concatenated_data[key].append(data)

        for key, data_list in concatenated_data.items():
            yield (
                key,
                self.analysis_to_run[key].post_process_data(
                    self.analysis_to_run[key].concatenate_data(data_list)
                ),
            )

    def _add_to_pixel_dataset(self, post_processed_data, pxl_dataset: PixelDataset):
        for key, data in post_processed_data:
            pxl_dataset = self.analysis_to_run[key].add_to_pixel_dataset(
                data, pxl_dataset
            )
        return pxl_dataset

    def execute(self, pixel_dataset) -> PixelDataset:
        """Execute the analysis on the provided pixel dataset."""
        prepared_computations = self._prepare_computation()
        per_component_results = self._execute_computations_in_parallel(
            prepared_computations
        )
        post_processed_data = self._post_process(per_component_results)
        pxl_dataset_with_results = self._add_to_pixel_dataset(
            post_processed_data, pixel_dataset
        )
        return pxl_dataset_with_results


def edgelist_to_component_stream(
    dataset: PixelDataset, use_full_bipartite: bool
) -> Iterable[tuple[str, Graph]]:
    """Convert the edgelist in the dataset to a stream component ids and their component graphs."""
    for component_id, component_df in (
        dataset.edgelist_lazy.collect()
        .partition_by(by="component", as_dict=True)
        .items()
    ):
        yield (
            str(component_id[0]),  # component id is a tuple here, hence the [0]
            Graph.from_edgelist(
                edgelist=component_df.lazy(),
                add_marker_counts=True,
                simplify=True,
                use_full_bipartite=use_full_bipartite,
            ),
        )


def run_analysis(
    pxl_dataset: PixelDataset,
    analysis_to_run: list[PerComponentAnalysis],
    use_full_bipartite: bool = False,
) -> PixelDataset:
    """Run the provided list of `PerComponentAnalysis` on the components in the `pxl_dataset`.

    :param pxl_dataset: The PixelDataset to run the analysis on.
    :param analysis_to_run: A list of `PerComponentAnalysis` to run on the components in the `pxl_dataset`.
    :param use_full_bipartite: Whether to use the full bipartite graph when creating the components.
    :returns: A `PixelDataset` instance with the provided analysis added to it.
    """
    if not analysis_to_run:
        logger.warning("No analysis functions were provided")
        return pxl_dataset

    analysis_manager = _AnalysisManager(
        analysis_to_run,
        component_stream=edgelist_to_component_stream(
            pxl_dataset, use_full_bipartite=use_full_bipartite
        ),
    )
    pxl_dataset = analysis_manager.execute(pxl_dataset)
    return pxl_dataset

import logging
from collections import defaultdict
from functools import partial
from queue import Queue
from threading import get_ident
from typing import Callable, Iterable, Protocol

import pandas as pd

from pixelator.analysis.colocalization import colocalization_from_component_graph
from pixelator.analysis.colocalization.types import TransformationTypes
from pixelator.analysis.polarization import polarization_scores_component
from pixelator.analysis.polarization.types import PolarizationNormalizationTypes
from pixelator.graph import Graph
from pixelator.pixeldataset import PixelDataset
from pixelator.statistics import correct_pvalues
from pixelator.utils import (
    get_process_pool_executor,
)

logger = logging.getLogger(__name__)


class PerComponentAnalysis(Protocol):
    ANALYSIS_NAME: str = ...

    def run_on_component(self, component: Graph, component_id: str) -> pd.DataFrame: ...

    def concatenate_data(self, data: Iterable[pd.DataFrame]) -> pd.DataFrame:
        try:
            scores = pd.concat(data, axis=0)
            return scores
        except ValueError as error:
            logger.error(f"No data was found to compute {self.__name__}")
            raise error

    def post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def add_to_pixel_dataset(
        self, data: pd.DataFrame, pxl_dataset: PixelDataset
    ) -> PixelDataset: ...


class ColocalizationAnalysis(PerComponentAnalysis):
    ANALYSIS_NAME = "colocalization"

    def __init__(
        self,
        transformation_type: TransformationTypes,
        neighbourhood_size: int,
        n_permutations: int,
        min_region_count: int,
    ):
        self.transformation_type = transformation_type
        self.neighbourhood_size = neighbourhood_size
        self.n_permutations = n_permutations
        self.min_region_count = min_region_count

    def run_on_component(self, component: Graph, component_id: str) -> pd.DataFrame:
        logger.debug("Running colocalization analysis on component %s", component_id)
        return colocalization_from_component_graph(
            graph=component,
            component_id=component_id,
            transformation=self.transformation_type,
            neighbourhood_size=self.neighbourhood_size,
            n_permutations=self.n_permutations,
            min_region_count=self.min_region_count,
        )

    def post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Post processing colocalization analysis data")
        p_value_columns = filter(lambda x: "_p" in x, data.columns)
        for p_value_col in p_value_columns:
            data.insert(
                data.columns.get_loc(p_value_col) + 1,
                f"{p_value_col}_adjusted",
                correct_pvalues(data[p_value_col].to_numpy()),
            )
        return data

    def add_to_pixel_dataset(
        self, data: pd.DataFrame, pxl_dataset: PixelDataset
    ) -> PixelDataset:
        logger.debug("Adding colocalization analysis data to PixelDataset")
        pxl_dataset.colocalization = data
        return pxl_dataset


class PolarizationAnalysis(PerComponentAnalysis):
    ANALYSIS_NAME = "polarization"

    def __init__(
        self, normalization: PolarizationNormalizationTypes, permutations: int
    ):
        self.normalization = normalization
        self.permutations = permutations

    def run_on_component(self, component: Graph, component_id: str) -> pd.DataFrame:
        logger.debug("Running polarization analysis on component %s", component_id)
        return polarization_scores_component(
            graph=component,
            component_id=component_id,
            normalization=self.normalization,
            permutations=self.permutations,
        )

    def post_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Post processing polarization analysis data")
        data.insert(
            data.columns.get_loc("morans_p_value") + 1,
            "morans_p_adjusted",
            correct_pvalues(data["morans_p_value"].to_numpy()),
        )
        return data

    def add_to_pixel_dataset(
        self, data: pd.DataFrame, pxl_dataset: PixelDataset
    ) -> PixelDataset:
        logger.debug("Adding polarization analysis data to PixelDataset")
        pxl_dataset.polarization = data
        return pxl_dataset


class _AnalysisManager:
    def __init__(
        self,
        analysis_to_run: list[PerComponentAnalysis],
        component_stream: Iterable[tuple[str, Graph]],
        logging_setup,
    ):
        self.analysis_to_run = {
            analysis.ANALYSIS_NAME: analysis for analysis in analysis_to_run
        }
        self.component_stream = component_stream
        self.logging_setup = logging_setup

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
        with get_process_pool_executor(logging_setup=self.logging_setup) as executor:
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
        prepared_computations = self._prepare_computation()
        per_component_results = self._execute_computations_in_parallel(
            prepared_computations
        )
        post_processed_data = self._post_process(per_component_results)
        pxl_dataset_with_results = self._add_to_pixel_dataset(
            post_processed_data, pixel_dataset
        )
        return pxl_dataset_with_results


def edgelist_to_component_stream(dataset: PixelDataset) -> Iterable[tuple[str, Graph]]:
    for component_id, component_df in (
        dataset.edgelist_lazy.collect()
        .partition_by(by="component", as_dict=True)
        .items()
    ):
        yield (
            component_id,
            Graph.from_edgelist(
                edgelist=component_df.lazy(),
                add_marker_counts=True,
                simplify=True,
                use_full_bipartite=False,
            ),
        )


def run_analysis(
    pxl_dataset: PixelDataset,
    analysis_to_run: list[PerComponentAnalysis],
    logging_setup,
) -> PixelDataset:
    analysis_manager = _AnalysisManager(
        analysis_to_run,
        component_stream=edgelist_to_component_stream(pxl_dataset),
        logging_setup=logging_setup,
    )
    pxl_dataset = analysis_manager.execute(pxl_dataset)
    return pxl_dataset

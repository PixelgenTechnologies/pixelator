"""Analysis engine capable of running a list of analysis functions on each component in a pixeldataset.

Copyright © 2024 Pixelgen Technologies AB.
"""

import itertools
import logging
import multiprocessing
import typing
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from logging.handlers import SocketHandler
from pathlib import Path
from typing import Callable, Generic, Iterable, Protocol, TypeVar

import click
import pandas as pd
import polars as pl
from joblib import Parallel, delayed

from pixelator.pna import read
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import Component, PNAPixelDataset
from pixelator.pna.pixeldataset.io import PxlFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoggingSetup:
    """Dataclass to hold the logging setup for the analysis engine.

    Assumes that the logging setup uses a socket handler on local host
    to manage the logging.
    """

    port: int
    log_level: int

    @staticmethod
    def from_logger(logger: logging.Logger):
        """Create a LoggingSetup from a logger."""
        return LoggingSetup(port=logger.port, log_level=logger.log_level)  # type: ignore


def _add_handlers_to_root_logger(logging_setup):
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        root_logger.setLevel(logging_setup.log_level)
        socket_handler = SocketHandler("localhost", logging_setup.port)
        root_logger.addHandler(socket_handler)


class _ParallelWithLogging(Parallel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _print(self, mgs):
        logger.debug(mgs)


def _get_joblib_executor(nbr_cores=None, **kwargs) -> Parallel:
    """Return a joblib executor with some default settings."""
    current_click_context = click.get_current_context(silent=True)
    click_nbr_cores = None
    if current_click_context:
        click_nbr_cores = current_click_context.obj.get("CORES")

    nbr_cores = (
        nbr_cores if nbr_cores else click_nbr_cores or multiprocessing.cpu_count()
    )
    return _ParallelWithLogging(n_jobs=nbr_cores, **kwargs)


T = TypeVar("T")


def with_logging(f):
    """Add logging to a function with this decorator.

    This is necessary to deal with the fact that for out of process workers
    like the ones used by joblib we need to set up logging in each worker.
    This decorator makes this slightly easier.
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        # Turn this into a dectorator
        logging_setup = kwds.pop("logging_setup", None)
        if logging_setup:
            _add_handlers_to_root_logger(logging_setup)

        return f(*args, **kwds)

    return wrapper


class PerComponentTask(Protocol, Generic[T]):
    """Protocol for tasks that are run on each component in a PixelDataset.

    :var TASK_NAME: The name of the analysis.
    """

    TASK_NAME: typing.ClassVar[str]

    def set_dataset(self, pxl_file_path: Path):
        """Specify a dataset to enable analysis being run directly from component IDs."""
        ...

    @with_logging
    def run_from_component_id(self, component_id: str) -> T:
        """Run the analysis on this component.

        This method assumes that the dataset is stored internally
        and the component is directly accessible from its id.

        :param component_id: The id of the component.
        """
        raise NotImplementedError

    @with_logging
    def run_on_component_graph(self, component: PNAGraph, component_id: str) -> T:
        """Run the analysis on this component.

        :param component: The graph of the component.
        :param component_id: The id of the component.
        """
        raise NotImplementedError

    @with_logging
    def run_on_component_edgelist(
        self, component: pl.LazyFrame, component_id: str
    ) -> T:
        """Run the analysis on this component.

        :param component: The edgelist of the component.
        :param component_id: The id of the component.
        """
        raise NotImplementedError

    @with_logging
    def run_on_component(
        self, component: Component | str, logging_setup: LoggingSetup | None = None
    ) -> T:
        """Run the analysis on a component.

        This method will let the Analysis decide whether to run on a Graph or a LazyFrame.
        The default implementation will first try the graph based analysis and then fall back
        to the edgelist based analysis if the graph based analysis raises NotImplementedError.

        :param component: The component to run the analysis on. Either a Graph, a LazyFrame or the name of a component.
        :param logging_setup: The logging setup to use.
        :raises TypeError: If the component is not a Graph or a LazyFrame.
        """
        if isinstance(component, str):
            return self.run_from_component_id(component)
        try:
            return self.run_on_component_edgelist(
                component.frame, component.component_id
            )
        except NotImplementedError:
            pass
        try:
            return self.run_on_component_graph(component.graph, component.component_id)
        except NotImplementedError as e:
            logger.error(
                "Either `run_on_component_graph` or `run_on_component_edgelist` "
                "must be implemented by a `PerComponentTask`"
            )
            raise e

    def concatenate_data(self, data: Iterable[T]) -> T:
        """Concatenate the data. Override this if you need custom concatenation behavior."""
        try:
            scores = pd.concat(data, axis=0)
            return scores
        except ValueError as error:
            logger.error(f"No data was found to compute {self.TASK_NAME}")
            raise error

    def post_process_data(self, data: T) -> T:
        """Post process the data (e.g. adjust p-values). Override this if your data needs post processing."""
        return data

    def add_to_pixel_file(self, data: T, pxl_file_target: PxlFile) -> None:
        """Add the data in the right place in the pxl_dataset."""
        ...

    def parameters(self) -> dict[str, typing.Any]:
        """Return the parameters of the `PerComponentAnalysis`.

        This is used e.g. to store the metadata the parameters of the analysis
        in the run metadata.
        """
        return {f"{self.TASK_NAME}": vars(self)}


class AnalysisManager:
    """Analysis manager that can run a number of analysis across a stream of components.

    The analysis manager is responsible for hooking up the analysis functions and making
    them run on each component in the stream. The main workflow it uses is outlined in the
    `execute` method.
    """

    def __init__(
        self,
        analysis_to_run: Iterable[PerComponentTask],
        logging_setup: LoggingSetup | None = None,
        n_cores: int | None = None,
        pxl_dataset_builder: Callable[[Iterable[PxlFile]], PNAPixelDataset]
        | None = None,
    ):
        """Initialize the analysis manager.

        :param analysis_to_run: The analysis to run on each component.
        :param logging_setup: The logging setup to use.
        :param n_cores: The number of cores to use for parallel processing (set to 1 to disable parallel processing).
        :param pxl_dataset_builder: A function that can build a PixelDataset from an iterable of PxlFiles.
        """
        self.analysis_to_run = {
            analysis.TASK_NAME: analysis for analysis in analysis_to_run
        }
        self._logging_setup = logging_setup
        if n_cores is not None and n_cores < 1:
            raise ValueError("n_cores must be greater than 0 or None.")
        self._n_cores = n_cores
        if pxl_dataset_builder is None:
            pxl_dataset_builder = PNAPixelDataset.from_pxl_files
        self._pxl_dataset_builder = pxl_dataset_builder

    def _execute_computations_in_parallel(
        self, component_stream: Iterable[Component | str]
    ):
        with _get_joblib_executor(
            nbr_cores=self._n_cores, verbose=100, return_as="generator_unordered"
        ) as parallel:

            def func(component, analysis_to_run):
                results = []
                for analysis_name, analysis in analysis_to_run.items():
                    logger.debug(
                        f"running {analysis_name} on {component.component_id if isinstance(component, Component) else component}"
                    )
                    results.append(
                        (
                            analysis_name,
                            analysis.run_on_component(
                                component, logging_setup=self._logging_setup
                            ),
                        )
                    )

                return results

            yield from itertools.chain.from_iterable(
                parallel(
                    delayed(func)(
                        component,
                        self.analysis_to_run,
                    )
                    for component in component_stream
                )
            )

    def _execute_computations_sequentially(
        self, component_stream: Iterable[Component | str]
    ):
        for component in component_stream:
            results = []
            for analysis_name, analysis in self.analysis_to_run.items():
                logger.debug(
                    f"running {analysis_name} on {component.component_id if isinstance(component, Component) else component}"
                )
                results.append(
                    (
                        analysis_name,
                        analysis.run_on_component(
                            component, logging_setup=self._logging_setup
                        ),
                    )
                )
            yield from results

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

    def _add_to_pixel_dataset(
        self, post_processed_data, pxl_file_target: PxlFile
    ) -> PNAPixelDataset:
        for key, data in post_processed_data:
            self.analysis_to_run[key].add_to_pixel_file(data, pxl_file_target)
        return self._pxl_dataset_builder(pxl_file_target)  # type: ignore

    def _execute_on_iterator(self, iterator, pxl_file_target: PxlFile):
        if self._n_cores is None or self._n_cores > 1:
            per_component_results = self._execute_computations_in_parallel(iterator)
        else:
            per_component_results = self._execute_computations_sequentially(iterator)
        post_processed_data = self._post_process(per_component_results)
        pxl_dataset_with_results = self._add_to_pixel_dataset(
            post_processed_data, pxl_file_target=pxl_file_target
        )
        return pxl_dataset_with_results

    def execute(
        self, pixel_dataset: PNAPixelDataset, pxl_file_target: PxlFile
    ) -> PNAPixelDataset:
        """Execute the analysis on the provided pixel dataset."""
        iterator = pixel_dataset.edgelist().iterator()
        return self._execute_on_iterator(iterator, pxl_file_target)

    def _set_path_to_dataset(self, input_pxl_file_path: Path):
        for key in self.analysis_to_run.keys():
            self.analysis_to_run[key].set_dataset(input_pxl_file_path)

    def execute_from_path(
        self, input_pxl_file_path: Path, pxl_file_target: PxlFile
    ) -> PNAPixelDataset:
        """Execute the analysis on the provided pixel file path."""
        self._set_path_to_dataset(input_pxl_file_path)
        pixel_dataset = read(input_pxl_file_path)
        iterator = pixel_dataset.components()
        return self._execute_on_iterator(iterator, pxl_file_target)

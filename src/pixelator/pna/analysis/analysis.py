"""Analysis classes for the PNA data.

Copyright © 2024 Pixelgen Technologies AB.
"""

import logging
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from pixelator.pna.analysis.proximity import jcs_with_permute_stats
from pixelator.pna.analysis_engine import PerComponentTask
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile

logger = logging.getLogger(__name__)


class ProximityAnalysis(PerComponentTask):
    """Run proximity analysis on each component."""

    TASK_NAME = "proximity"

    def __init__(
        self,
        n_permutations: int = 100,
        min_marker_count: int = 10,
    ) -> None:
        """Initialize a ProximityAnalysis instance.

        :param n_permutations: the number of permutations to use for the method (only used with methods
                               that use permutations).
        """
        self.method = "join_count_statistics"
        self._proxmimity_function = partial(
            jcs_with_permute_stats,
            n_permutations=n_permutations,
            min_marker_count=min_marker_count,
        )

    def run_on_component_edgelist(
        self, component_df: pl.LazyFrame, component_id: str
    ) -> pd.DataFrame:
        """Run proximity analysis on a single component.

        :param component: a Graph for a component to run the analysis on.
        :param component_id: the id of the component.
        :return: a pandas DataFrame containing average_node degrees per type counts.
        """
        result = self._proxmimity_function(component_df.collect())
        result["component"] = component_id
        return result

    def add_to_pixel_file(self, data: pd.DataFrame, pxl_file_target: PxlFile) -> None:
        """Add proximity data for all components to the pixeldataset.

        :param data: a pandas DataFrame containing proximity data for all components.
        :param pxl_dataset: the PixelDataset to add the data to.
        """
        logger.debug("Adding proximity data to PixelDataset")
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
            tmp_file = Path(tmp_file.name)  # type: ignore
            data.to_parquet(tmp_file, index=False)
            with PixelFileWriter(pxl_file_target.path) as writer:
                writer.write_proximity(tmp_file)
        logger.debug("Proximity data added to PixelDataset")

    def run_on_component_graph(
        self, component: PNAGraph, component_id: str
    ) -> pd.DataFrame:
        """Run proximity analysis on a single component."""
        raise NotImplementedError

    def parameters(self) -> dict:
        """Return the parameters of the `PerComponentAnalysis`.

        This is used e.g. to store the metadata the parameters of the analysis
        in the run metadata.
        """
        data = super().parameters()
        data["method"] = self.method
        return data

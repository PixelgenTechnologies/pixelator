"""Build PXL output from edgelist via the graph-step component recovery pipeline.

Copyright © 2025 Pixelgen Technologies AB.
"""

import tempfile
from copy import copy
from pathlib import Path

import numpy as np
from anndata import AnnData

from pixelator.common.annotate.aggregates import call_aggregates
from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.cli.common import logger
from pixelator.pna.config import PNAAntibodyPanel
from pixelator.pna.graph.community_detection import (
    StagedRefinementOptions,
    find_components,
)
from pixelator.pna.graph.report import GraphStatistics
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.io import PixelFileWriter


def update_stats_from_adata(adata: AnnData, stats: GraphStatistics) -> GraphStatistics:
    """Update GraphStatistics from an AnnData object.

    Args:
    adata: Adata.
    stats: Stats.
    """
    component_stats = copy(stats)
    component_stats.reads_output = int(adata.obs["reads_in_component"].sum())

    component_stats.median_reads_per_component = adata.obs[
        "reads_in_component"
    ].median()
    component_stats.median_markers_per_component = adata.obs["n_umi"].median()

    # Add tau_type metrics
    aggregates_mask = adata.obs["tau_type"] != "normal"
    number_of_aggregates = np.sum(aggregates_mask)

    component_stats.aggregate_count = number_of_aggregates
    aggregate_stats = (
        adata[aggregates_mask].obs[["n_edges", "n_umi", "reads_in_component"]].sum()
    )

    component_stats.read_count_in_aggregates = aggregate_stats[
        "reads_in_component"
    ].item()
    component_stats.node_count_in_aggregates = aggregate_stats["n_umi"].item()
    component_stats.edge_count_in_aggregates = aggregate_stats["n_edges"].item()

    return component_stats


def build_pxl_file_with_components(
    parquet_file: Path,
    panel: PNAAntibodyPanel,
    sample_name: str,
    path_output_pxl_file: Path,
    multiplet_recovery: bool,
    edge_cycle_verification: bool,
    min_read_count: int,
    refinement_options: StagedRefinementOptions,
    component_size_threshold: bool | tuple[int, int],
    n_cores: int = 1,
) -> tuple[PNAPixelDataset, GraphStatistics]:
    """Build a new PXL file with components recovered from edgelist.

    Starting from an edgelist parquet file, (collapse step output) this function recovers components
    using community detection with staged refinement, and writes a new PXL file containing the
    edgelist with resolved components. Components are resolved based on the provided community detection options.
    It also computes and returns statistics about the graph.

    Args:
    parquet_file: Path to the input parquet file containing the edgelist. (collapse step output)
    panel: Name of the panel used in the experiment.
    sample_name: Name of the sample.
    path_output_pxl_file: Path to the output PXL file.
    multiplet_recovery: Whether to perform multiplet recovery.
    edge_cycle_verification: Whether to perform edge cycle verification.
    min_read_count: Minimum read count threshold for an edge to be retained.
    refinement_options: Options for staged refinement during community detection.
    component_size_threshold: Min/Max size threshold for components to be retained. Can be a boolean for dynamic sizing or a tuple of two integers (Min, Max).
    n_cores: Number of CPU cores to use for parallel processing.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        component_stats, edgelist_with_components_path = find_components(
            input_edgelist_path=parquet_file,
            working_dir=tmp_dir_path,
            multiplet_recovery=multiplet_recovery,
            edge_cycle_verification=edge_cycle_verification,
            min_read_count=min_read_count,
            refinement_options=refinement_options,
            component_size_threshold=component_size_threshold,
            n_threads=n_cores,
        )

        with PixelFileWriter(path_output_pxl_file) as pxl_file_writer:
            pxl_connection = pxl_file_writer.get_connection()
            pxl_connection.execute(
                """
            CREATE OR REPLACE TABLE edgelist AS
            SELECT * FROM read_parquet($1, hive_partitioning = true)
            ORDER BY component
            """,
                [str(edgelist_with_components_path)],
            )
            logger.debug("Counting molecules")

            sums = pxl_connection.execute(
                "SELECT SUM(uei_count) as uei_count FROM edgelist"
            ).pl()

            component_stats.molecules_output = int(sums["uei_count"][0])

            logger.debug("Building edgelist from anndata")
            adata = pna_edgelist_to_anndata(pxl_connection, panel=panel)
            call_aggregates(adata)

            component_stats = update_stats_from_adata(adata, component_stats)

            # Sort adata on component names for stable output
            adata = adata[adata.obs_names.sort_values(), :]

            # import here to avoid circular imports
            from pixelator import __version__

            metadata = {
                "sample_name": sample_name,
                "version": __version__,
                "technology": "single-cell-pna",
                "panel_name": panel.name,
                "panel_version": panel.version,
            }

            logger.debug("Building pxl file")

            pxl_file_writer.write_metadata(metadata)
            pxl_file_writer.write_adata(adata)

    return PNAPixelDataset.from_pxl_files(path_output_pxl_file), component_stats

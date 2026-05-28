"""Mapping components to samples in sample-hashed datasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

import logging
import re
import tempfile
from itertools import chain
from pathlib import Path
from typing import Tuple

import anndata
import duckdb
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl

from pixelator import __version__
from pixelator.common.annotate.aggregates import call_aggregates
from pixelator.pna.analysis_engine import AnalysisManager, PerComponentTask
from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.io import PixelFileWriter
from pixelator.pna.sample_calling.hash_antibodies import HashedAntibodyMapping
from pixelator.pna.sample_calling.report import SampleCallingTotalReport

logger = logging.getLogger(__name__)


def collect_hash_info(
    input_pxl_file: PNAPixelDataset,
    hashed_antibody_mapping: HashedAntibodyMapping,
    confidence_threshold: float,
    undetermined_sample_name: str = "undetermined",
) -> pl.DataFrame:
    """Map components to samples in an sample-hashed dataset.

    This function processes a pixel dataset and a mapping of components to
    samples based on hashed antibodies. It computes the hashed antibody count
    for each component, determines the most likely sample assignment for each
    component, and calculates a confidence score for the assignment.

    If the most likely assignment is to antibodies not belonging to any sample,
    the component is assigned to the "undetermined" sample.

    Args:
        input_pxl_file: The input pixel dataset.
        hashed_antibody_mapping: Mapping of sample names to hashed antibody names and the full list
            of hashing antibodies (from panel).
        confidence_threshold: Minimum confidence required to assign a component to a sample.
            Defaults to 0.8.
        undetermined_sample_name: Name to use for undetermined components. Defaults to
            "undetermined".

    Returns:
        Polars DataFrame with component id, per-sample hash counts, called sample, and confidence
        score.
    """
    ab_count_data = pl.from_pandas(input_pxl_file.adata().to_df().reset_index()).lazy()
    samples = hashed_antibody_mapping.keys()
    unmapped = hashed_antibody_mapping.unmapped_hashing_antibodies
    undetermined_col = (
        pl.sum_horizontal(list(unmapped)).alias(
            f"{undetermined_sample_name}_hash_count"
        )
        if unmapped
        else pl.lit(0).alias(f"{undetermined_sample_name}_hash_count")
    )

    hash_counts_per_sample = ab_count_data.select(
        ["component"] + list(hashed_antibody_mapping.hashing_antibodies)
    ).with_columns(
        *[
            pl.sum_horizontal(hashed_antibody_mapping[sample]).alias(
                f"{sample}_hash_count"
            )
            for sample in samples
        ],
        undetermined_col,
    )
    unpivot_cols = [f"{sample}_hash_count" for sample in samples] + [
        f"{undetermined_sample_name}_hash_count"
    ]
    called_sample = (
        hash_counts_per_sample.select(["component"] + unpivot_cols)
        .unpivot(unpivot_cols, index="component")
        .group_by("component")
        .agg(
            pl.col("variable")
            .filter(pl.col("value") == pl.col("value").max())
            .first()
            .str.strip_suffix("_hash_count")
            .alias("called_sample"),
            pl.col("value").max().alias("max_value"),
            pl.col("value").sum().alias("total_value"),
        )
        .with_columns(
            pl.when(pl.col("total_value") == 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col("max_value") / pl.col("total_value"))
            .alias("sample_confidence")
        )
        .with_columns(
            pl.when(pl.col("sample_confidence") < confidence_threshold)
            .then(pl.lit(undetermined_sample_name))
            .otherwise(pl.col("called_sample"))
            .alias("called_sample")
        )
    )

    result = hash_counts_per_sample.join(
        called_sample, on="component", how="left"
    ).select(
        ["component"]
        + [pl.col(f"{sample}_hash_count") for sample in samples]
        + [
            f"{undetermined_sample_name}_hash_count",
            "called_sample",
            "sample_confidence",
        ]
    )

    return result.collect()


def _collect_stranded_nodes(edgelist: pl.DataFrame) -> pl.DataFrame:
    graph = nx.from_edgelist(edgelist.select(["umi1", "umi2"]).rows())
    connected_components = sorted(nx.connected_components(graph), key=len, reverse=True)
    stranded_nodes = (
        pl.DataFrame({"umi": list(chain.from_iterable(connected_components[1:]))})
        .cast(pl.UInt64)
        .with_columns(cause=pl.lit("stranded"))
    )

    return stranded_nodes


def _collect_nodes_to_remove(
    edgelist: pl.LazyFrame,
    all_hashing_in_panel: set[str],
    sample_antibodies: list[str],
) -> pl.DataFrame:
    """Find nodes to remove: those touching hashes not belonging to the current sample.

    Incompatible hashes are defined from the panel (all hashing antibodies minus
    the current sample's), so they do not depend on other samples being present
    in the samplesheet.
    """
    incompatible_hashes = all_hashing_in_panel - set(sample_antibodies)
    edgelist_df = edgelist.collect()
    incompatible_umi1 = (
        edgelist_df.filter(pl.col("marker_1").is_in(incompatible_hashes))
        .select("umi1")
        .rename({"umi1": "umi"})
    )
    incompatible_umi2 = (
        edgelist_df.filter(pl.col("marker_2").is_in(incompatible_hashes))
        .select("umi2")
        .rename({"umi2": "umi"})
    )
    incompatible_nodes = (
        pl.concat([incompatible_umi1, incompatible_umi2])
        .cast({"umi": pl.UInt64})
        .with_columns(cause=pl.lit("incompatible_hash"))
    )

    stranded_nodes = _collect_stranded_nodes(
        edgelist_df.filter(
            (~pl.col("umi1").is_in(incompatible_nodes["umi"]))
            & (~pl.col("umi2").is_in(incompatible_nodes["umi"]))
        )
    )
    return pl.concat((incompatible_nodes, stranded_nodes))


def _add_original_hash_counts_to_obs(
    old_adata: anndata.AnnData, antibodies_for_obs: set[str]
) -> None:
    """Add original_hash_counts_* to old_adata.obs for every panel hashing antibody.

    Antibodies present in the adata matrix get their counts; those not present
    (e.g. not in samplesheet or zero counts) get 0 so all panel hashing antibodies
    are represented in obs.
    """
    old_anndata_counts = old_adata.to_df()
    for ab in sorted(list(antibodies_for_obs)):
        if ab in old_anndata_counts.columns:
            old_adata.obs[f"original_hash_counts_{ab}"] = old_anndata_counts[ab].values
        else:
            old_adata.obs[f"original_hash_counts_{ab}"] = 0


def _add_missing_adata_info(new_adata, old_adata):
    missing_obs = set(old_adata.obs.columns) - set(new_adata.obs.columns)
    missing_var = set(old_adata.var.columns) - set(new_adata.var.columns)

    new_adata.obs = new_adata.obs.join(old_adata.obs[list(missing_obs)], how="left")
    new_adata.var = new_adata.var.join(old_adata.var[list(missing_var)], how="left")

    return new_adata


def _build_post_sample_calling_anndata(
    con: duckdb.DuckDBPyConnection,
    old_adata: anndata.AnnData,
    nodes_to_remove: pl.DataFrame,
    hash_info: pl.DataFrame,
    panel: PNAAntibodyPanel,
    hashing_antibody_mapping: HashedAntibodyMapping,
):
    """Build AnnData object after sample calling.

    The new AnnData object has no panel hashing markers in var,
    instead hashing antibodies are added to obs.

    Args:
        con: Con.
        old_adata: Old adata.
        nodes_to_remove: Nodes to remove.
        hash_info: Hash info.
        panel: Panel.
        hashing_antibody_mapping: Information about hashing antibodies, including a mapping from
            sample names to lists of hashed antibody names.
    """
    _add_original_hash_counts_to_obs(
        old_adata, hashing_antibody_mapping.hashing_antibodies
    )

    # Create the anndata object and remove all panel hashing markers from var
    new_adata = pna_edgelist_to_anndata(con, panel)
    non_hashing_markers = [
        marker
        for marker in new_adata.var.index
        if marker not in hashing_antibody_mapping.hashing_antibodies
    ]
    new_adata = new_adata[:, non_hashing_markers].copy()

    call_aggregates(new_adata)
    new_adata = _add_missing_adata_info(new_adata, old_adata)
    new_adata.obs = new_adata.obs.join(
        hash_info.select(["component", "sample_confidence"])
        .to_pandas()
        .set_index("component", drop=True),
        how="left",
    )

    if nodes_to_remove.is_empty():
        new_adata.obs["removed_incompatible_hashes"] = 0
        new_adata.obs["removed_stranded_nodes"] = 0
        new_adata.obs["total_removed_in_sample_calling"] = 0
        return new_adata

    removal_info = (
        nodes_to_remove.group_by(["component", "cause"])
        .len()
        .pivot("cause", index="component", values="len")
    )
    removal_info = removal_info.rename(
        {"incompatible_hash": "removed_incompatible_hashes"}
    )
    if "stranded" in removal_info.columns:
        removal_info = removal_info.rename({"stranded": "removed_stranded_nodes"})
    else:
        removal_info = removal_info.with_columns(removed_stranded_nodes=pl.lit(0))
    removal_info = removal_info.with_columns(
        total_removed_in_sample_calling=pl.col("removed_incompatible_hashes")
        + pl.col("removed_stranded_nodes")
    )
    new_adata.obs = new_adata.obs.join(
        removal_info.to_pandas().set_index("component", drop=True), how="left"
    ).fillna(0)

    return new_adata


def _find_nodes_to_remove(
    hashing_antibody_mapping: HashedAntibodyMapping,
    remove_incompatible: bool,
    sample_name: str,
    sample_data: PNAPixelDataset,
    undetermined_sample_name: str = "undetermined",
):
    if remove_incompatible and (sample_name != undetermined_sample_name):
        task = FindHashedNodesToRemove(  # type: ignore[abstract]
            hashing_antibody_mapping=hashing_antibody_mapping,
            sample_name=sample_name,
        )
        manager = HashedSampleAnalysisManager(analysis_to_run=[task])
        return manager.execute(sample_data)

    return pl.DataFrame({"umi": [], "cause": [], "component": []})


def sample_calling(
    input_pxl: PNAPixelDataset,
    hashing_antibody_mapping: HashedAntibodyMapping,
    output_folder: Path,
    confidence_threshold: float = 0.8,
    remove_incompatible: bool = True,
    undetermined_sample_name: str = "undetermined",
) -> list[Path]:
    """Split components of a pixel dataset by their sample.

    Splits components of a pixel dataset by their sample and writes out
    separate pxl files for each sample. This function processes a
    PNAPixelDataset, assigns components to samples based on hash information
    and confidence thresholds, and writes out pxl files for each determined
    sample. It will also write a file for undetermined components (under the name
    "{undetermined_sample_name}.dehashed.pxl").
    It supports removing incompatible hashes and renaming hash markers in the output.

    Args:
        input_pxl: The input pixel dataset to be split by sample.
        hashing_antibody_mapping: Information about hashing antibodies, including a mapping from
            sample names to lists of hashed antibody names.
        output_folder: Directory where output pxl files will be written.
        confidence_threshold: Minimum confidence required to assign a component to a sample.
            Defaults to 0.8.
        remove_incompatible: Whether to remove hashes incompatible with the current sample from the
            edgelist. Defaults to True.
        undetermined_sample_name: Name to use for undetermined components. Defaults to
            "undetermined".

    Returns:
        List of all output pxl files created.
    """
    hash_info = collect_hash_info(
        input_pxl,
        hashing_antibody_mapping,
        confidence_threshold,
        undetermined_sample_name,
    )
    panel = PNAAntibodyPanel.from_pxl_dataset(input_pxl)

    dehashed = hash_info.group_by("called_sample")
    output_files: list[Path] = []
    for _, group in dehashed:
        sample_name = str(group["called_sample"].first())
        logger.info("Running sample calling for sample: %s", sample_name)
        comps = group["component"].to_list()
        target_path = output_folder / (sample_name + ".dehashed.pxl")
        sample_data = input_pxl.filter(components=comps)

        metadata = {
            "sample_name": sample_name,
            "version": __version__,
            "technology": "single-cell-pna",
            "panel_name": panel.name,
            "panel_version": panel.version,
        }

        nodes_to_remove = _find_nodes_to_remove(
            hashing_antibody_mapping,
            remove_incompatible,
            sample_name,
            sample_data,
            undetermined_sample_name,
        )

        with (
            tempfile.NamedTemporaryFile(
                suffix=".edgelist.tmp.parquet"
            ) as tmp_edgelist_parquet,
            sample_data.view.open() as session,
        ):
            con = session.get_connection()
            con.register("nodes_to_remove_tbl", nodes_to_remove[["umi"]])
            con.register("sample_components", pl.DataFrame({"component": comps}))

            # Dehashing should only strip the `-<hash_index>` suffix for *known*
            # hashing antibody names (e.g. `B2M-1` -> `B2M`), not for other
            # biological marker IDs like `PD-1`.
            hashed_markers = sorted(list(hashing_antibody_mapping.hashing_antibodies))
            hash_marker_map = pl.DataFrame(
                {
                    "hashed_marker": hashed_markers,
                    "base_marker": [
                        re.sub(r"-\d+$", "", marker) for marker in hashed_markers
                    ],
                }
            )
            con.register("hash_marker_map", hash_marker_map)

            marker_1_sql = "COALESCE(hm1.base_marker, e.marker_1) AS marker_1"
            marker_2_sql = "COALESCE(hm2.base_marker, e.marker_2) AS marker_2"
            select_cols = (
                "e.umi1, e.umi2, e.read_count, e.uei_count, "
                f"{marker_1_sql}, {marker_2_sql}, e.component"
            )

            query = f"""COPY (
                            SELECT
                                {select_cols}
                            FROM edgelist e
                            LEFT JOIN hash_marker_map hm1 ON e.marker_1 = hm1.hashed_marker
                            LEFT JOIN hash_marker_map hm2 ON e.marker_2 = hm2.hashed_marker
                            WHERE e.umi1 NOT IN (SELECT umi FROM nodes_to_remove_tbl)
                            AND e.umi2 NOT IN (SELECT umi FROM nodes_to_remove_tbl)
                            AND e.component IN (SELECT component FROM sample_components)
                        ) TO '{tmp_edgelist_parquet.name}' (FORMAT parquet);
                    """
            con.execute(query)
            con.sql(
                f"""CREATE OR REPLACE VIEW edgelist AS SELECT *
                FROM read_parquet('{tmp_edgelist_parquet.name}');"""
            )

            adata = _build_post_sample_calling_anndata(
                con,
                sample_data.adata(),
                nodes_to_remove,
                hash_info,
                panel,
                hashing_antibody_mapping,
            )

            with PixelFileWriter(target_path) as pxl_file_writer:
                pxl_file_writer.write_metadata(metadata)
                pxl_file_writer.write_edgelist(Path(tmp_edgelist_parquet.name))
                pxl_file_writer.write_adata(adata)
            output_files.append(target_path)

    return output_files


class FindHashedNodesToRemove(PerComponentTask):
    """Find nodes to be removed in components with hashed antibodies."""

    TASK_NAME = "find-hashed-nodes-to-remove"

    def __init__(
        self,
        hashing_antibody_mapping: HashedAntibodyMapping,
        sample_name: str,
    ):
        """Initialize the task with hashing antibody mapping and sample name.

        Args:
            hashing_antibody_mapping: Information about hashing antibodies, including a mapping from
                sample names to lists of hashed antibody names.
            sample_name: Sample name.
        """
        self.hashing_antibody_mapping = hashing_antibody_mapping
        self.sample_name = sample_name

    def run_on_component_edgelist(
        self, component: pl.LazyFrame, component_id: str
    ) -> pl.DataFrame:
        """Find nodes to remove for one component."""
        sample_antibodies = self.hashing_antibody_mapping.get(self.sample_name, [])
        nodes_to_remove = _collect_nodes_to_remove(
            component,
            all_hashing_in_panel=self.hashing_antibody_mapping.hashing_antibodies,
            sample_antibodies=sample_antibodies,
        ).with_columns(component=pl.lit(component_id))
        return nodes_to_remove

    def concatenate_data(self, data: Tuple[pl.DataFrame]) -> pl.DataFrame:  # type: ignore[override]
        """Concatenate the results from all components."""
        return pl.concat(data)


class HashedSampleAnalysisManager(AnalysisManager):
    """Manager for sample calling analysis on hashed samples."""

    def execute(self, pixel_dataset: PNAPixelDataset) -> pl.DataFrame:  # type: ignore[override]
        """Execute the analysis on the provided pixel dataset."""
        if self._n_cores is None or self._n_cores > 1:
            per_component_results = self._execute_computations_in_parallel(
                pixel_dataset.edgelist().iterator()
            )
        else:
            per_component_results = self._execute_computations_sequentially(
                pixel_dataset.edgelist().iterator()
            )
        post_processed_data = self._post_process(per_component_results)
        _, result = next(post_processed_data)
        return result


def create_final_report(
    final_dataset: PNAPixelDataset, undetermined_sample_name: str = "undetermined"
) -> SampleCallingTotalReport:
    """Create the final report for the sample calling.

    Args:
        final_dataset: The final dataset after sample calling.
        undetermined_sample_name: Name to use for undetermined components. Defaults to
            "undetermined".

    Returns:
        SampleCallingTotalReport: The final report for the sample calling.
    """
    n_components = len(final_dataset.components())

    if n_components == 0:
        raise ValueError(
            "The final dataset has no components. This is likely due to an error in the samplesheet."
        )

    percentage_of_components_successfully_called = 1.0
    if undetermined_sample_name in final_dataset.sample_names():
        n_undetermined_components = len(
            final_dataset.filter(samples=undetermined_sample_name).components()
        )
        percentage_of_components_successfully_called = 1.0 - (
            n_undetermined_components / n_components
        )

    sample_confidences_per_sample = {
        sample_name: final_dataset.filter(samples=sample_name)
        .adata()
        .obs["sample_confidence"]
        for sample_name in final_dataset.sample_names()
    }

    total_report = SampleCallingTotalReport(
        sample_id="all",
        product_id="single-cell-pna",
        number_of_components=n_components,
        percentage_of_components_successfully_called=percentage_of_components_successfully_called,
        sample_confidences_per_sample={
            sample: confidences.to_list()
            for sample, confidences in sample_confidences_per_sample.items()
        },
    )
    return total_report


def warn_if_undetermined_has_high_confidence(
    undetermined_sample_confidences: pd.Series | np.ndarray,
    confidence_threshold: float,
    undetermined_sample_name: str = "undetermined",
) -> None:
    """Warn if the undetermined sample has a high confidence score.

    Args:
        undetermined_sample_confidences: Undetermined sample confidences.
        confidence_threshold: Minimum confidence required to assign a component to a sample.
            Defaults to 0.8.
        undetermined_sample_name: Name to use for undetermined components. Defaults to
            "undetermined".
    """
    if (
        np.sum(undetermined_sample_confidences > confidence_threshold)
        / len(undetermined_sample_confidences)
        > 0.05
    ):
        logger.warning(
            f"There are more than 5% of components in {undetermined_sample_name} that have a high confidence score. "
            "This may indicate that the samplesheet has the wrong sample indices."
        )

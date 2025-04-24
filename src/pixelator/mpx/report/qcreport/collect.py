"""Helper functions to collect data for report generation.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd
import polars as pl

from pixelator.common.utils.simplification import simplify_line_rdp
from pixelator.mpx.pixeldataset import SIZE_DEFINITION, PixelDataset
from pixelator.mpx.report.qcreport.types import QCReportData

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    from anndata import AnnData

    from pixelator.mpx.report import PixelatorWorkdir


def collect_components_umap_data(adata: AnnData) -> str:
    """Create a csv formatted string with the components umap data for the qc report.

    :param adata: an AnnData object with the umap data (obs)
    :return: a csv formatted string with umap data
    :rtype: str
    """
    empty = np.full(adata.n_obs, np.nan)

    if "X_umap" in adata.obsm:
        umap_df = pd.DataFrame(
            adata.obsm["X_umap"], columns=["umap1", "umap2"], index=adata.obs.index
        )
    else:
        umap_df = pd.DataFrame({"umap1": empty, "umap2": empty}, index=adata.obs.index)

    if "leiden" in adata.obs:
        umap_df["cluster"] = adata.obs["leiden"].to_numpy()
    else:
        umap_df["cluster"] = empty

    if "cluster_cell_class" in adata.obs:
        umap_df["cluster_cell_class"] = adata.obs["cluster_cell_class"].to_numpy()
    else:
        umap_df["cluster_cell_class"] = np.full(adata.n_obs, "unknown")

    umap_df = pd.concat(
        [
            umap_df,
            adata.obs[
                [
                    "reads",
                    "molecules",
                    "mean_b_pixels_per_a_pixel",
                    "mean_molecules_per_a_pixel",
                ]
            ],
        ],
        axis=1,
    )
    return umap_df.to_csv(index=True, index_label="component")


def collect_antibody_percentages_data(adata: AnnData) -> str:
    """Create the antibody percentages histogram data for the qc report.

    This function created a csv formatted string with the antibody name and the
    percentage of the antibody aggregated over all components.

    :param adata: an AnnData object with antibody counts percentages data
    :return: a csv formatted string with antibody percentages data
    :rtype: str
    """
    index = adata.var.index.set_names("antibody", inplace=False)
    df = pd.DataFrame(
        {"count": adata.var["antibody_count"], "percentage": adata.var["antibody_pct"]},
        index=index,
    )
    return df.to_csv()


def collect_antibody_counts_data(adata: AnnData) -> str:
    """Create the antibody counts data for the qc report.

    :param adata: an AnnData object with the antibody counts data
    :return: a csv formatted string with the antibody counts data
    :rtype: str
    """
    return adata.to_df().to_csv(index=True, index_label="component")


def collect_component_ranked_component_size_data(
    components_metrics: pd.DataFrame,
    subsample_non_cell_components: bool = False,
    subsample_epsilon: float = 1e-3,
) -> str:
    """Create data for the `cell calling` and `component size distribution` plot.

    This collects the component size and the number of antibodies per component.
    Components that pass the filters (is_filtered) are marked as selected.

    :param components_metrics: a pd.DataFrame with the components metrics
    :param subsample_non_cell_comoponents: if True, subsample non-cell components
    :param subsample_epsilon: the epsilon value for the subsampling.
        Ignored if subsample_non_cell_comoponents is False
    :return: a csv formatted string with the plotting data
    :rtype: str
    """
    component_sizes = components_metrics[SIZE_DEFINITION].to_numpy()
    df = pd.DataFrame({"component_size": component_sizes})
    df["rank"] = df.rank(method="first", ascending=False).astype(int)
    df["selected"] = components_metrics["is_filtered"].to_numpy()
    df["markers"] = components_metrics["antibodies"].to_numpy()
    df.sort_values(by="rank", inplace=True)

    if subsample_non_cell_components:
        cell_mask = df["selected"].to_numpy()
        coords = np.ascontiguousarray(df[["rank", "component_size"]].to_numpy())
        cell_idx = np.flatnonzero(cell_mask)
        other_coords_mask = ~cell_mask
        other_coords_idx = np.flatnonzero(other_coords_mask)
        other_coords = coords[other_coords_mask]

        if len(other_coords) != 0:
            simplified_coords_idx = simplify_line_rdp(
                other_coords, subsample_epsilon, return_mask=True
            )

            # Check for non empty here since zero length simplified_coords_idx
            # has shape conflicts when concatenating
            global_coords = np.concatenate(
                [cell_idx, other_coords_idx[simplified_coords_idx]]
            )
            global_coords.sort()

            df = df.iloc[global_coords]
            # These should still be sorted since we used sorted indices
            # but lets sort them again by rank just to make sure
            df.sort_values(by="rank", inplace=True)

    df.set_index("rank", inplace=True)
    return df.to_csv(index=True)


def collect_reads_per_molecule_frequency(dataset: PixelDataset) -> str:
    """Create a frequency table for the edge list "count" column.

    :param dataset: The PixelDataset object
    :return: a csv formatted string with the frequency table
    :rtype: str
    """
    freq = (
        dataset.edgelist_lazy.select(
            pl.col("count").alias("reads_per_molecule").value_counts(sort=True)
        )
        .unnest("reads_per_molecule")
        .collect()
    )

    freq = freq.with_columns(
        pl.col("reads_per_molecule"),
        frequency=pl.col("count") / pl.col("count").sum(),
    )
    pd_freq = freq.sort(by="reads_per_molecule").to_pandas()

    return pd_freq.to_csv(index=False)


def collect_report_data(workdir: PixelatorWorkdir, sample_id: str) -> QCReportData:
    """Collect the data needed to generate figures in the qc report.

    The `annotate` folder must be present in `input_path`.

    :param workdir: The PixelatorWorkdir object
    :param sample_id: The sample id
    :returns QCReportData: A QCReportData object
    :raises NotADirectoryError: If the input folder is missing the annotate folder
    :raises FileNotFoundError: If the annotate folder is missing the datasets
    """
    logger.debug("Collecting QC report data for %s in %s", sample_id, workdir.basedir)

    # parse filtered dataset
    dataset_file = workdir.filtered_dataset(sample_id)
    dataset = PixelDataset.from_file(dataset_file)
    adata = dataset.adata
    component_data = collect_components_umap_data(adata)
    antibody_percentages = collect_antibody_percentages_data(adata)
    antibody_counts = collect_antibody_counts_data(adata)

    # parse raw components metrics
    metrics_file = workdir.raw_component_metrics(sample_id)
    raw_components_metrics = pd.read_csv(metrics_file)
    ranked_component_size_data = collect_component_ranked_component_size_data(
        raw_components_metrics, subsample_non_cell_components=True
    )
    reads_per_molecule_frequency_data = collect_reads_per_molecule_frequency(dataset)

    # build the report data
    data = QCReportData(
        component_data=component_data,
        ranked_component_size=ranked_component_size_data,
        antibodies_per_cell=None,
        sequencing_saturation=None,
        antibody_percentages=antibody_percentages,
        antibody_counts=antibody_counts,
        reads_per_molecule_frequency=reads_per_molecule_frequency_data,
    )

    logger.debug("QC report data collected for %s in %s", sample_id, workdir.basedir)
    return data

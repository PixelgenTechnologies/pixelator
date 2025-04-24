"""Helper functions to collect data for report generation.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import itertools
import json
import logging
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import scanpy as sc

from pixelator.common.utils.simplification import simplify_line_rdp
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.report.common import WorkdirOutputNotFound
from pixelator.pna.report.qcreport.types import Metrics, QCReportData

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    from anndata import AnnData

    from pixelator.pna.graph.report import GraphSampleReport
    from pixelator.pna.report.common import PixelatorPNAReporting


def collect_components_umap_data(adata: AnnData) -> str:
    """Create a csv formatted string with the components umap data for the qc report.

    :param adata: an AnnData object with the umap data (obs)
    :return: a csv formatted string with umap data
    :rtype: str
    """
    empty = np.full(adata.n_obs, np.nan)

    sc.pp.neighbors(
        adata,
        use_rep="clr",
        n_neighbors=min(adata.n_obs, 15),
        random_state=1,
    )

    init_pos: typing.Literal["random", "spectral"] = (
        "random" if adata.n_obs - 1 <= 2 else "spectral"
    )
    sc.tl.umap(adata, min_dist=0.5, n_components=2, random_state=1, init_pos=init_pos)

    if "X_umap" in adata.obsm:
        umap_df = pd.DataFrame(
            adata.obsm["X_umap"], columns=["umap1", "umap2"], index=adata.obs.index
        )
    else:
        umap_df = pd.DataFrame({"umap1": empty, "umap2": empty}, index=adata.obs.index)

    umap_df["cluster"] = empty

    umap_df = pd.concat(
        [
            umap_df,
            adata.obs[["reads_in_component", "n_antibodies", "n_umi", "n_edges"]],
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


def collect_ranked_component_size_data(
    adata: AnnData, graph_report: GraphSampleReport, subsample: bool = True
) -> str:
    """Create data for the `cell calling` and `component size distribution` plot.

    This collects the component size and the number of antibodies per component.
    Components that pass the filters (is_filtered) are marked as selected.

    :param adata: The AnnData object with the component metrics
    :param graph_report: The graph report object
    :param subsample_non_cell_comoponents: if True, subsample non-cell components
    :param subsample_epsilon: the epsilon value for the subsampling.
        Ignored if subsample_non_cell_comoponents is False
    :return: a csv formatted string with the plotting data
    :rtype: str
    """
    data = adata.obs

    sorted_keys = sorted(
        list(graph_report.pre_filtering_component_sizes.keys()), reverse=True
    )
    component_sizes = np.fromiter(
        itertools.chain.from_iterable(
            itertools.repeat(k, graph_report.pre_filtering_component_sizes[k])
            for k in sorted_keys
        ),
        dtype=int,
    )
    df = pd.DataFrame({"size": component_sizes})
    df["selected"] = np.logical_and(
        df["size"] >= (graph_report.component_size_min_filtering_threshold or 0),
        df["size"] <= (graph_report.component_size_max_filtering_threshold or np.inf),
    )

    df.sort_values(by="size", ascending=False, inplace=True)
    df["rank"] = df["size"].rank(method="average", ascending=False)

    if subsample:
        cell_mask = df["selected"].to_numpy()
        coords = np.ascontiguousarray(df[["rank", "size"]].to_numpy())
        cell_idx = np.flatnonzero(cell_mask)
        other_coords_mask = ~cell_mask
        other_coords_idx = np.flatnonzero(other_coords_mask)
        other_coords = coords[other_coords_mask]

        if len(other_coords) != 0:
            simplified_coords_idx = simplify_line_rdp(
                other_coords, 1e-2, return_mask=True
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


def _build_symmetric_proximity_matrix(
    df: pl.DataFrame, labels: npt.NDArray
) -> npt.NDArray:
    """Build a symmetric matrix from the pivoted dataframe.

    The dataframe is expected to have a column 'marker_1' and the rest of the columns
    are the markers.

    The matrix is built by filling the missing values with 0, sorting the columns and
    rows according to a fixed order, and then forcing the matrix to be symmetric
    by averaging it with its transpose.

    Params:
        df: The dataframe to build the matrix from
        labels: The labels to use for the fixed order of the matrix

    Returns:
        The symmetric matrix as a numpy array

    """
    missing_cols = set(labels) - set(df.columns[1:])
    missing_rows = set(labels) - set(df["marker_1"])

    # Create a mapping dictionary for the fixed order
    order_mapping = {value: index for index, value in enumerate(labels)}

    # Add missing columns
    for col in missing_cols:
        df = df.with_columns(pl.lit(None).alias(col))

    # Add missing rows
    for row in missing_rows:
        df = df.vstack(
            pl.DataFrame({"marker_1": [row], **{col: None for col in df.columns[1:]}})
        )

    assert df.shape[0] == df.shape[1] - 1, (
        "Unexpected shape after correcting for missing rows/columns"
    )

    # Sort the columns starting from index 1
    sorted_columns = sorted(
        df.columns[1:], key=lambda col: order_mapping.get(col, float("inf"))
    )

    # Reorder columns
    df_sorted = df.select(["marker_1"] + sorted_columns)
    # Reorder rows
    df_sorted = df_sorted.sort(by=pl.col("marker_1").replace_strict(order_mapping))

    assert (df_sorted["marker_1"] == df_sorted.columns[1:]).all(), (
        "Row and column order do not match"
    )

    matrix = df_sorted.drop("marker_1").to_numpy()

    # Replace nan values by the matching transposed values to make a symmetric matrix
    # and fill the rest with 0
    symmetric_matrix = np.where(np.isnan(matrix), matrix.T, matrix)
    symmetric_matrix = np.nan_to_num(symmetric_matrix, copy=False, nan=0.0)

    assert not np.isnan(symmetric_matrix).any(), (
        "Matrix still contains NaN values after filling"
    )

    return symmetric_matrix


def collect_proximity_data(
    pxl_dataset: PNAPixelDataset,
    min_marker_count: int = 50,
    min_join_count_expected_mean=2,
    min_component_count=2,
    max_markers=50,
) -> str | None:
    """Collect the proximity data for the qc report.

    Params:
        pxl_dataset: The PNAPixelDataset object
        min_marker_count: The minimum marker count for a marker pair to be included in the median scores
        min_join_count_expected_mean: The minimum join count expected mean for a marker pair to be included in the median scores
        min_component_count:
            The minimum number of components needed for a marker pair after previous filtering
            to be included in the median scores.
        max_markers:
            The maximum number of markers to include. Markers with the highest stddev are picked.d
    Returns:
        A json formatted string with the proximity data or None of no proximity data is found.
    """
    df = pxl_dataset.proximity().to_polars()
    if df is None:
        return None

    # Filter the scores
    filtered_df = df.filter(
        (pl.col("marker_1_count") >= min_marker_count)
        & (pl.col("marker_2_count") >= min_marker_count)
        & (pl.col("join_count_expected_mean") >= min_join_count_expected_mean)
    )

    # Group by marker_1, marker_2 and count the number of components that
    # have that pair after previously filtering for marker count
    group_counts = filtered_df.group_by(["marker_1", "marker_2"]).len()

    # Filter the dataframe to only include the pairs that have at least min_component_count items
    filtered_df = (
        filtered_df.join(group_counts, on=["marker_1", "marker_2"], how="left")
        .filter(pl.col("len") > min_component_count)
        .drop("len")
    )

    # Group by marker_1, marker_2 and take the median across components for join_count_z and log2_ratio
    result_df = filtered_df.group_by(["marker_1", "marker_2"]).agg(
        [
            pl.median("join_count_z").alias("median_z_score"),
            pl.median("log2_ratio").alias("median_log2ratio"),
        ]
    )

    # Create unique sorted array of markers
    labels = np.unique(group_counts[["marker_1", "marker_2"]].to_numpy())

    join_count_z = result_df.pivot(
        on="marker_2",
        index="marker_1",
        values="median_z_score",
        aggregate_function="median",
        sort_columns=True,
    )

    n_labels = len(labels)
    join_count_z_matrix = _build_symmetric_proximity_matrix(join_count_z, labels)

    log2_ratio = result_df.pivot(
        on="marker_2",
        index="marker_1",
        values="median_log2ratio",
        aggregate_function="median",
        sort_columns=True,
    )

    log2_ratio_matrix = _build_symmetric_proximity_matrix(log2_ratio, labels)

    join_count_z_std = np.std(join_count_z_matrix, axis=1)
    log2_ratio_std = np.std(log2_ratio_matrix, axis=1)

    # This might not make much sense, but it is just to define a total order
    combined_std = join_count_z_std + log2_ratio_std

    # Select the last N element
    marker_subset = np.sort(np.argsort(combined_std)[n_labels - max_markers : n_labels])

    filtered_labels = labels[marker_subset]
    filtered_idx = np.ix_(marker_subset, marker_subset)
    join_count_z_matrix_filtered = join_count_z_matrix[filtered_idx]
    log2_ratio_matrix_filtered = log2_ratio_matrix[filtered_idx]

    return json.dumps(
        {
            "markers": filtered_labels.tolist(),
            "join_count_z": np.round(join_count_z_matrix_filtered, decimals=4).tolist(),
            "log2_ratio": np.round(log2_ratio_matrix_filtered, decimals=4).tolist(),
        }
    )


def collect_metrics_report_data(
    reporting: PixelatorPNAReporting, sample_id: str
) -> Metrics:
    """Collect the data needed for the main metrics in the qc report.

    The `annotate` folder must be present in `input_path`.

    :param reporting: The report object to use for data collection from the workdir
    :param sample_id: The sample id
    :returns Metrics: A Metrics object
    """
    logger.debug(
        "Collecting QC report data for %s in %s", sample_id, reporting.workdir.basedir
    )

    amplicon_metrics = reporting.amplicon_metrics(sample_id)
    demux_metrics = reporting.demux_metrics(sample_id)
    collapse_metrics = reporting.collapse_metrics(sample_id)
    graph_metrics = reporting.graph_metrics(sample_id)
    analysis_metrics = reporting.analysis_metrics(sample_id)
    reads_flow = reporting.reads_flow(sample_id)

    def _fraction_or_zero(x: int | float, y: int | float) -> float:
        return x / y if y > 0.0 else 0.0

    # Global level metrics
    # TODO: Move them into reporting.reads_flow ?

    total_input_reads = amplicon_metrics.input_reads
    fraction_discarded_reads = _fraction_or_zero(
        (amplicon_metrics.input_reads - graph_metrics.reads_output), total_input_reads
    )
    fraction_discarded_reads_amplicon = _fraction_or_zero(
        amplicon_metrics.total_failed_reads, total_input_reads
    )
    fraction_discarded_reads_demux = _fraction_or_zero(
        demux_metrics.failed_reads, total_input_reads
    )
    fraction_discarded_reads_graph = _fraction_or_zero(
        graph_metrics.discarded_reads, total_input_reads
    )
    valid_reads_saturation = 1 - _fraction_or_zero(
        collapse_metrics.output_molecules, collapse_metrics.input_reads
    )
    fraction_valid_reads = _fraction_or_zero(
        collapse_metrics.input_reads, total_input_reads
    )
    fraction_graph_reads = _fraction_or_zero(
        graph_metrics.reads_output, total_input_reads
    )

    # Map pixelator metrics to QC report metrics
    metrics = Metrics(
        # Read level statistics
        number_of_reads=reads_flow.input_read_count,
        # Amplicon
        input_reads_amplicon=amplicon_metrics.input_reads,
        output_reads_amplicon=amplicon_metrics.output_reads,
        discarded_reads_amplicon=amplicon_metrics.total_failed_reads,
        # Demux
        input_reads_demux=demux_metrics.input_reads,
        output_reads_demux=demux_metrics.output_reads,
        corrected_reads_demux=collapse_metrics.corrected_reads,
        discarded_reads_demux=demux_metrics.failed_reads,
        # Collapse
        input_reads_collapse=collapse_metrics.input_reads,
        output_reads_collapse=collapse_metrics.output_reads,
        output_molecules_collapse=collapse_metrics.output_molecules,
        corrected_reads_collapse=collapse_metrics.corrected_reads,
        # Graph
        input_molecules_graph=graph_metrics.molecules_input,
        output_molecules_graph=graph_metrics.molecules_output,
        input_reads_graph=graph_metrics.reads_input,
        output_reads_graph=graph_metrics.reads_output,
        discarded_reads_graph=graph_metrics.discarded_reads,
        discarded_molecules_graph=graph_metrics.discarded_molecules,
        # Discarded reads percentages vs total reads
        fraction_discarded_reads=fraction_discarded_reads,
        fraction_discarded_reads_amplicon=fraction_discarded_reads_amplicon,
        fraction_discarded_reads_demux=fraction_discarded_reads_demux,
        fraction_discarded_reads_graph=fraction_discarded_reads_graph,
        # Saturation
        graph_node_saturation=graph_metrics.node_saturation,
        graph_edge_saturation=graph_metrics.edge_saturation,
        valid_reads_saturation=valid_reads_saturation,
        # Global "read conversion" into useful data
        fraction_valid_reads=fraction_valid_reads,
        fraction_graph_reads=fraction_graph_reads,
        # Global and per region Q30 statistics
        fraction_q30_bases_in_pid1=amplicon_metrics.q30_statistics.pid1,
        fraction_q30_bases_in_pid2=amplicon_metrics.q30_statistics.pid2,
        fraction_q30_bases_in_umi1=amplicon_metrics.q30_statistics.umi1,
        fraction_q30_bases_in_umi2=amplicon_metrics.q30_statistics.umi2,
        fraction_q30_bases_in_lbs1=amplicon_metrics.q30_statistics.lbs1,
        fraction_q30_bases_in_lbs2=amplicon_metrics.q30_statistics.lbs2,
        fraction_q30_bases_in_uei=amplicon_metrics.q30_statistics.uei,
        fraction_q30_bases=amplicon_metrics.q30_statistics.total,
        number_of_cells=graph_metrics.component_count_post_component_size_filtering,
        median_reads_per_cell=graph_metrics.median_reads_per_component,
        median_markers_per_cell=graph_metrics.median_markers_per_component,
        median_average_k_coreness=(
            analysis_metrics.k_cores.median_average_k_core
            if analysis_metrics.k_cores
            else 0.0
        ),
        fraction_of_outlier_cells=graph_metrics.fraction_of_aggregate_components,
    )

    logger.debug(
        "QC report data collected for %s in %s", sample_id, reporting.workdir.basedir
    )
    return metrics


def collect_report_data(
    reporting: PixelatorPNAReporting, sample_id: str
) -> QCReportData:
    """Collect the data needed to generate figures in the qc report.

    :param reporting: The PixelatorPNAReporting object
    :param sample_id: The sample id to collect data for
    :returns QCReportData: A QCReportData object
    """
    logger.debug(
        "Collecting QC report data for %s in %s", sample_id, reporting.workdir.basedir
    )

    workdir = reporting.workdir
    # parse filtered dataset
    dataset_file = workdir.filtered_dataset(sample_id)
    dataset = PNAPixelDataset.from_files(dataset_file)
    adata = dataset.adata()
    graph_report = reporting.graph_metrics(sample_id)
    metric_data = collect_metrics_report_data(reporting, sample_id)

    component_data = collect_components_umap_data(adata)
    antibody_percentages = collect_antibody_percentages_data(adata)
    antibody_counts = collect_antibody_counts_data(adata)
    ranked_component_sizes = collect_ranked_component_size_data(adata, graph_report)

    try:
        analysis_dataset_file = workdir.analysed_dataset(sample_id)
        analysis_dataset = PNAPixelDataset.from_files(analysis_dataset_file)
        proximity_data = collect_proximity_data(analysis_dataset)
    except WorkdirOutputNotFound:
        logger.warning(
            "Analysis stage dataset not found for %s. Skipping proximity data in report.",
            sample_id,
        )
        proximity_data = None

    # build the report data
    data = QCReportData(
        metrics=metric_data,
        component_data=component_data,
        ranked_component_size=ranked_component_sizes,
        antibody_percentages=antibody_percentages,
        antibody_counts=antibody_counts,
        proximity_data=proximity_data,
    )

    logger.debug("QC report data collected for %s in %s", sample_id, workdir.basedir)
    return data

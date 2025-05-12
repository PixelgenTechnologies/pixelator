"""Copyright © 2024 Pixelgen Technologies AB."""

import logging

import pandas as pd
import polars as pl
from anndata import AnnData

from pixelator.pna.config import PNAAntibodyPanel

logger = logging.getLogger(__name__)


def calculate_antibody_metrics(counts_df):
    """Calculate antibody metrics from a count matrix, can be used to set/update adata.vars."""
    total_antibody = pd.Series(counts_df.sum(axis=0), name="antibody_count")
    relative_antibody = pd.Series(
        total_antibody / total_antibody.sum(), name="antibody_pct"
    )
    components_detected = pd.Series((counts_df > 0).sum(axis=0), name="components")

    return pd.concat([total_antibody, relative_antibody, components_detected], axis=1)


def add_panel_information(adata, panel):
    """Add extra panel variables to var."""
    panel_meta_data = panel.df[["marker_id", "control", "nuclear"]]
    panel_meta_data = panel_meta_data.set_index("marker_id")
    panel_meta_data.index = panel_meta_data.index.astype(str)
    panel_meta_data = panel_meta_data.fillna("no")

    adata.var = adata.var.join(panel_meta_data, how="left")
    return adata


def pna_edgelist_to_anndata(edgelist: pl.LazyFrame, panel: PNAAntibodyPanel) -> AnnData:
    """Build an AnnData object from a PNA edgelist and a panel object."""

    def construct_marker_count_matrix(edgelist):
        marker_1_counts = (
            edgelist.unique(["umi1", "marker_1"])
            .select(["marker_1", "component"])
            .group_by(["component", "marker_1"])
            .agg(pl.len().alias("marker_1_count"))
            .with_columns(marker="marker_1")
            .select("component", "marker", "marker_1_count")
        )

        marker_2_counts = (
            edgelist.unique(["umi2", "marker_2"])
            .select(["marker_2", "component"])
            .group_by(["component", "marker_2"])
            .agg(pl.len().alias("marker_2_count"))
            .with_columns(marker="marker_2")
            .select("component", "marker", "marker_2_count")
        )
        marker_count = (
            marker_1_counts.join(
                marker_2_counts, on=["component", "marker"], how="full"
            )
            .with_columns(
                marker_fixed=pl.when(pl.col("marker").is_null())
                .then(pl.col("marker_right"))
                .otherwise(pl.col("marker")),
                component_fixed=pl.when(pl.col("component").is_null())
                .then(pl.col("component_right"))
                .otherwise(pl.col("component")),
                count=pl.col("marker_1_count").fill_null(0)
                + pl.col("marker_2_count").fill_null(0),
            )
            .select(
                [
                    pl.col("component_fixed").alias("component"),
                    pl.col("marker_fixed").alias("marker"),
                    pl.col("count"),
                ]
            )
            .collect()
            .pivot(on="marker", index="component", values="count")
            .fill_null(0)
        )
        return marker_count

    def component_metrics(edgelist, counts_df):
        grouped_by_component = edgelist.group_by("component")
        a_markers = grouped_by_component.agg(pl.col("umi1").n_unique().alias("n_umi1"))
        b_markers = grouped_by_component.agg(pl.col("umi2").n_unique().alias("n_umi2"))
        edges = grouped_by_component.agg(pl.len().alias("n_edges"))
        reads_in_component = grouped_by_component.agg(
            pl.col("read_count").sum().alias("reads_in_component")
        )
        info_agg = (
            a_markers.join(b_markers, on="component")
            .join(edges, on="component")
            .join(reads_in_component, on="component")
            .collect()
            .to_pandas()
            .set_index("component")
            .astype("uint64")
        )
        markers = pd.DataFrame(
            pd.Series((counts_df > 0).sum(axis=1), name="n_antibodies", dtype="uint64")
        )
        df = pd.concat([info_agg, markers], axis=1)
        df["n_umi"] = df["n_umi1"] + df["n_umi2"]

        return df

    logger.debug("Constructing counts matrix.")
    counts_df = construct_marker_count_matrix(edgelist).to_pandas()
    counts_df = counts_df.set_index("component")
    counts_df = counts_df.reindex(columns=panel.markers, fill_value=0)
    counts_df.columns = counts_df.columns.astype(str)

    # compute components metrics (obs) and re-index
    logger.debug("Computing component metrics.")
    components_metrics_df = component_metrics(edgelist, counts_df)
    components_metrics_df = components_metrics_df.reindex(index=counts_df.index)
    components_metrics_df.index = components_metrics_df.index.astype(str)

    # compute antibody metrics (var) and re-index
    logger.debug("Computing antibody metrics.")
    antibody_metrics_df = calculate_antibody_metrics(counts_df=counts_df)
    antibody_metrics_df = antibody_metrics_df.reindex(index=panel.markers, fill_value=0)
    # Do a dtype conversion of the columns here since AnnData cannot handle
    # a pyarrow arrays.
    antibody_metrics_df = antibody_metrics_df.astype(
        {"antibody_count": "int64", "antibody_pct": "float32"}
    )
    antibody_metrics_df.index = antibody_metrics_df.index.astype(str)

    # create AnnData object
    counts_df.index = counts_df.index.astype(
        str
    )  # anndata requires indexes to be strings
    logger.debug("Building AnnData instance.")
    adata = AnnData(
        X=counts_df,
        obs=components_metrics_df,
        var=antibody_metrics_df,
    )

    adata = add_panel_information(adata, panel)

    # find fraction of isotype markers in cell
    total_marker_counts = counts_df.sum(axis=1)
    isotype_markers = adata.var[adata.var["control"] == "yes"].index
    isotype_counts = counts_df[isotype_markers].sum(axis=1)
    adata.obs["isotype_fraction"] = isotype_counts / total_marker_counts

    intracellular_markers = adata.var[adata.var["nuclear"] == "yes"].index
    adata.obs["intracellular_fraction"] = (
        counts_df[intracellular_markers].sum(axis=1) / total_marker_counts
    )

    return adata

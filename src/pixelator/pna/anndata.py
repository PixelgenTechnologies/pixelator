"""Copyright © 2024 Pixelgen Technologies AB."""

import logging
import tempfile

import duckdb
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
    """Add panel data to var."""
    panel_copy = panel.df.copy()
    panel_columns = list(panel_copy.columns)
    panel_copy = panel_copy.set_index("marker_id")
    panel_copy.index = panel_copy.index.astype(str)
    panel_copy = panel_copy.fillna("no")

    adata.var = adata.var.join(panel_copy, how="left")

    adata.uns["panel_metadata"] = {
        "name": panel.name,
        "aliases": panel.aliases,
        "description": panel.description,
        "version": panel.version,
        "panel_columns": panel_columns,
    }

    return adata


def pna_edgelist_to_anndata(
    pixel_connection: duckdb.DuckDBPyConnection, panel: PNAAntibodyPanel
) -> AnnData:
    """
    Build an AnnData object from a DuckDB connection to a pixel file and a panel object.

    Parameters
    ----------
    pixel_connection : duckdb.DuckDBPyConnection
        A DuckDB connection to a pixel file. The connection must contain an 'edgelist' table
        with the required columns (e.g., component, marker_1, marker_2, umi1, umi2, read_count).
    panel : PNAAntibodyPanel
        The antibody panel object containing marker metadata.

    Returns
    -------
    AnnData
        An AnnData object with counts and panel information.

    Notes
    -----
    Assumes that the 'edgelist' table exists in the DuckDB connection and contains the necessary columns.
    """
    def calculate_count_info(pixel_connection: duckdb.DuckDBPyConnection):
        with tempfile.TemporaryDirectory() as temp_dir:
            pixel_connection.execute(f"""
                COPY (
                    SELECT component, marker_1 AS marker, COUNT(DISTINCT umi1) AS marker_1_count
                    FROM edgelist
                    GROUP BY component, marker_1
                ) TO '{temp_dir + "/marker_1_counts.parquet"}' (FORMAT PARQUET)
            """)
            pixel_connection.execute(f"""
                COPY (
                    SELECT component, marker_2 AS marker, COUNT(DISTINCT umi2) AS marker_2_count
                    FROM edgelist
                    GROUP BY component, marker_2
                ) TO '{temp_dir + "/marker_2_counts.parquet"}' (FORMAT PARQUET)
            """)
            pixel_connection.execute(f"""
                COPY (
                    SELECT component, COUNT(*) AS n_edges, SUM(read_count) AS reads_in_component
                    FROM edgelist
                    GROUP BY component
                ) TO '{temp_dir + "/edge_counts.parquet"}' (FORMAT PARQUET)
            """)
            m1_counts_df = pl.read_parquet(temp_dir + "/marker_1_counts.parquet")
            m2_counts_df = pl.read_parquet(temp_dir + "/marker_2_counts.parquet")
            edge_counts_df = pl.read_parquet(temp_dir + "/edge_counts.parquet")

        return m1_counts_df, m2_counts_df, edge_counts_df

    def mix_counts(m1_counts_df, m2_counts_df):
        counts_df = (
            m1_counts_df.join(
                m2_counts_df, on=["component", "marker"], how="full", coalesce=True
            )
            .fill_null(0)
            .with_columns(
                (pl.col("marker_1_count") + pl.col("marker_2_count")).alias("count")
            )
            .pivot(values="count", index="component", on="marker")
            .fill_null(0)
            .to_pandas()
            .set_index("component")
            .astype("uint32")
        )
        return counts_df

    def component_metrics(
        m1_counts_df: pl.DataFrame,
        m2_counts_df: pl.DataFrame,
        edge_counts_df: pl.DataFrame,
    ) -> pd.DataFrame:
        a_markers = m1_counts_df.group_by("component").agg(
            pl.col("marker_1_count").sum().alias("n_umi1")
        )
        b_markers = m2_counts_df.group_by("component").agg(
            pl.col("marker_2_count").sum().alias("n_umi2")
        )
        info_agg = (
            a_markers.join(b_markers, on="component")
            .join(edge_counts_df, on="component")
            .to_pandas()
            .set_index("component")
            .astype("uint64")
        )
        node_counts_df = mix_counts(m1_counts_df, m2_counts_df)
        markers = pd.DataFrame(
            pd.Series(
                (node_counts_df > 0).sum(axis=1), name="n_antibodies", dtype="uint64"
            )
        )
        df = pd.concat([info_agg, markers], axis=1)
        df["n_umi"] = df["n_umi1"] + df["n_umi2"]

        return df

    logger.debug("Constructing counts matrix.")
    m1_counts_df, m2_counts_df, edge_counts_df = calculate_count_info(pixel_connection)
    node_counts_df = mix_counts(m1_counts_df, m2_counts_df)
    node_counts_df = node_counts_df.reindex(columns=panel.markers, fill_value=0)
    # compute components metrics (obs) and re-index
    logger.debug("Computing component metrics.")
    components_metrics_df = component_metrics(
        m1_counts_df, m2_counts_df, edge_counts_df
    )
    components_metrics_df = components_metrics_df.reindex(index=node_counts_df.index)
    components_metrics_df.index = components_metrics_df.index.astype(str)

    # compute antibody metrics (var) and re-index
    logger.debug("Computing antibody metrics.")
    antibody_metrics_df = calculate_antibody_metrics(counts_df=node_counts_df)
    antibody_metrics_df = antibody_metrics_df.reindex(index=panel.markers, fill_value=0)
    # Do a dtype conversion of the columns here since AnnData cannot handle
    # a pyarrow arrays.
    antibody_metrics_df = antibody_metrics_df.astype(
        {"antibody_count": "int64", "antibody_pct": "float32"}
    )
    antibody_metrics_df.index = antibody_metrics_df.index.astype(str)

    # create AnnData object
    node_counts_df.index = node_counts_df.index.astype(
        str
    )  # anndata requires indexes to be strings
    logger.debug("Building AnnData instance.")
    adata = AnnData(
        X=node_counts_df,
        obs=components_metrics_df,
        var=antibody_metrics_df,
    )

    adata = add_panel_information(adata, panel)

    # find fraction of isotype markers in cell
    total_marker_counts = node_counts_df.sum(axis=1)
    isotype_markers = adata.var[adata.var["control"] == "yes"].index
    isotype_counts = node_counts_df[isotype_markers].sum(axis=1)
    adata.obs["isotype_fraction"] = isotype_counts / total_marker_counts

    intracellular_markers = adata.var[adata.var["nuclear"] == "yes"].index
    adata.obs["intracellular_fraction"] = (
        node_counts_df[intracellular_markers].sum(axis=1) / total_marker_counts
    )

    return adata

"""Copyright © 2024 Pixelgen Technologies AB."""

import logging

import duckdb
import numpy as np
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
    """Build an AnnData object from a DuckDB connection to a pixel file and a panel object.

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
    logger.debug("Constructing counts matrix.")

    marker_names = [f"'{m}'" for m in panel.markers]
    marker_names_sql = ", ".join(marker_names)
    node_counts_df = (
        pixel_connection.execute(f"""
        SELECT *
        FROM (
            WITH counts_df_long AS (
                WITH
                    marker_1_counts AS (
                        SELECT component, marker_1 AS marker, COUNT(DISTINCT umi1) AS marker_1_count
                        FROM edgelist
                        GROUP BY component, marker_1),
                    marker_2_counts AS (
                        SELECT component, marker_2 AS marker, COUNT(DISTINCT umi2) AS marker_2_count
                        FROM edgelist
                        GROUP BY component, marker_2
                    )
                SELECT
                    COALESCE(a.component, b.component) AS component,
                    COALESCE(a.marker, b.marker) AS marker,
                    COALESCE(a.marker_1_count, 0) AS marker_1_count,
                    COALESCE(b.marker_2_count, 0) AS marker_2_count,
                    COALESCE(a.marker_1_count, 0) + COALESCE(b.marker_2_count, 0) AS count
                FROM marker_1_counts a
                FULL OUTER JOIN marker_2_counts b
                    ON a.component = b.component AND a.marker = b.marker
            )
            PIVOT counts_df_long
            ON marker IN ({marker_names_sql})
            USING SUM(count)
            GROUP BY component
        )
    """)
        .df()
        .fillna(0)
    )

    node_counts_df.set_index("component", inplace=True)
    node_counts_df = node_counts_df.reindex(columns=panel.markers, fill_value=0)
    node_counts_df = node_counts_df.astype("uint32")

    # compute components metrics (obs) and re-index
    logger.debug("Computing component metrics.")

    components_metrics_df = pixel_connection.execute(f"""
            WITH
                marker_1_counts AS (
                    SELECT component, marker_1 AS marker, COUNT(DISTINCT umi1) AS marker_1_count
                    FROM edgelist
                    GROUP BY component, marker_1),
                marker_2_counts AS (
                    SELECT component, marker_2 AS marker, COUNT(DISTINCT umi2) AS marker_2_count
                    FROM edgelist
                    GROUP BY component, marker_2
                ),
                component_marker_counts AS (
                    SELECT
                        COALESCE(a.component, b.component) AS component,
                        COALESCE(a.marker_1_count, 0) AS marker_1_count,
                        COALESCE(b.marker_2_count, 0) AS marker_2_count
                    FROM marker_1_counts a
                    FULL OUTER JOIN marker_2_counts b
                        ON a.component = b.component AND a.marker = b.marker
                    ),
                component_umi AS (
                    SELECT
                        component,
                        SUM(marker_1_count) AS n_umi1,
                        SUM(marker_2_count) AS n_umi2
                    FROM component_marker_counts
                    GROUP BY component
                ),
                edge_counts AS (
                    SELECT component, COUNT(*) AS n_edges, SUM(read_count) AS reads_in_component
                    FROM edgelist
                    GROUP BY component
                )
            SELECT
                u.component,
                n_umi1,
                n_umi2,
                e.n_edges,
                e.reads_in_component,
                (n_umi1 + n_umi2) AS n_umi
            FROM component_umi u
            LEFT JOIN edge_counts e ON u.component = e.component
            ORDER BY u.component
    """).df()

    n_antibodies = pd.Series(
        (node_counts_df != 0).sum(axis=1),
        index=node_counts_df.index,
        name="n_antibodies",
        dtype=np.uint32,
    )
    components_metrics_df.set_index("component", inplace=True)
    components_metrics_df = components_metrics_df.join(n_antibodies)

    components_metrics_df = components_metrics_df.astype(
        {
            "n_umi": np.uint64,
            "n_umi1": np.uint64,
            "n_umi2": np.uint64,
            "n_edges": np.uint64,
            "n_antibodies": np.uint32,
            "reads_in_component": np.uint64,
        }
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

    # This is set to preserve backwards compatibility with downstream reports that may expect it.
    # Eventually we should be able to remove this.
    adata.obs["intracellular_fraction"] = 0.0

    return adata

"""Copyright © 2024 Pixelgen Technologies AB."""

import logging

import duckdb
import numpy as np
import pandas as pd
from anndata import AnnData

from pixelator.pna.config import PNAAntibodyPanel

logger = logging.getLogger(__name__)


# Number of components aggregated in a single DuckDB query inside
# :func:`pna_edgelist_to_anndata`. Larger values amortise per-query overhead;
# smaller values bound the peak ``COUNT(DISTINCT umi*)`` hash table size. The
# default keeps peak memory roughly linear in this constant.
_COMPONENT_BATCH_SIZE = 512


def calculate_antibody_metrics(counts_df):
    """Calculate antibody metrics from a count matrix, can be used to set/update adata.vars."""
    total_antibody = pd.Series(counts_df.sum(axis=0), name="antibody_count")
    relative_antibody = pd.Series(
        total_antibody / total_antibody.sum(), name="antibody_pct"
    )
    components_detected = pd.Series((counts_df > 0).sum(axis=0), name="components")

    return pd.concat([total_antibody, relative_antibody, components_detected], axis=1)


def add_panel_information(adata: AnnData, panel: PNAAntibodyPanel) -> AnnData:
    """Add panel data to var."""
    adata.var = adata.var.join(panel.df, how="left")

    adata.uns["panel_metadata"] = panel.metadata.model_dump()
    adata.uns["panel_metadata"]["panel_columns"] = list(panel.df.columns)

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

    Returns:
    -------
    AnnData
        An AnnData object with counts and panel information.

    Notes:
    -----
    Assumes that the 'edgelist' table exists in the DuckDB connection and contains the necessary columns.

    The aggregations are computed in fixed-size batches of components. Each
    ``COUNT(DISTINCT umi*)`` operator therefore only ever builds a hash set
    over one batch's worth of rows instead of a single global distinct hash
    table per worker thread. The batch size (:data:`_COMPONENT_BATCH_SIZE`) is
    tuned to amortise DuckDB's per-query overhead while keeping peak hash
    table memory bounded. ``WHERE component IN (...)`` benefits from DuckDB's
    row-group zone maps when the edgelist is clustered by component (the
    typical case for PNA pipelines that emit edges component-by-component).

    """
    components = (
        pixel_connection.execute(
            "SELECT DISTINCT component FROM edgelist ORDER BY component"
        )
        .df()["component"]
        .tolist()
    )

    n_components = len(components)
    n_markers = len(panel.markers)
    component_to_idx = {c: i for i, c in enumerate(components)}
    marker_to_idx = {m: i for i, m in enumerate(panel.markers)}

    X = np.zeros((n_components, n_markers), dtype=np.uint32)
    n_umi1_arr = np.zeros(n_components, dtype=np.uint64)
    n_umi2_arr = np.zeros(n_components, dtype=np.uint64)
    n_edges_arr = np.zeros(n_components, dtype=np.uint64)
    reads_arr = np.zeros(n_components, dtype=np.uint64)

    logger.debug(
        "Aggregating per-component metrics over %d components in batches of %d.",
        n_components,
        _COMPONENT_BATCH_SIZE,
    )
    for batch_start in range(0, n_components, _COMPONENT_BATCH_SIZE):
        batch = components[batch_start : batch_start + _COMPONENT_BATCH_SIZE]
        placeholders = ", ".join(["?"] * len(batch))

        # Per-marker counts: combine A-side and B-side distinct UMI counts
        # for each (component, marker). The CTE shape plus outer ``SUM``
        # produces a single hash aggregation in DuckDB's plan that scales
        # better than a flat ``UNION ALL`` of two ``COUNT(DISTINCT)`` queries.
        per_marker_sql = f"""
            WITH
                a AS (
                    SELECT component, marker_1 AS marker, COUNT(DISTINCT umi1) AS c
                    FROM edgelist
                    WHERE component IN ({placeholders})
                    GROUP BY component, marker_1
                ),
                b AS (
                    SELECT component, marker_2 AS marker, COUNT(DISTINCT umi2) AS c
                    FROM edgelist
                    WHERE component IN ({placeholders})
                    GROUP BY component, marker_2
                )
            SELECT component, marker, SUM(c) AS total
            FROM (SELECT * FROM a UNION ALL SELECT * FROM b)
            GROUP BY component, marker
        """
        for comp, marker, total in pixel_connection.execute(
            per_marker_sql, batch + batch
        ).fetchall():
            j = marker_to_idx.get(marker)
            if j is not None:
                X[component_to_idx[comp], j] = total

        per_component_metrics_sql = f"""
            SELECT
                component,
                COUNT(DISTINCT umi1) AS n_umi1,
                COUNT(DISTINCT umi2) AS n_umi2,
                COUNT(*) AS n_edges,
                SUM(read_count) AS reads_in_component
            FROM edgelist
            WHERE component IN ({placeholders})
            GROUP BY component
        """
        for (
            comp,
            n_umi1_val,
            n_umi2_val,
            n_edges_val,
            reads_val,
        ) in pixel_connection.execute(per_component_metrics_sql, batch).fetchall():
            i = component_to_idx[comp]
            n_umi1_arr[i] = n_umi1_val
            n_umi2_arr[i] = n_umi2_val
            n_edges_arr[i] = n_edges_val
            reads_arr[i] = reads_val

    components_str = [str(c) for c in components]
    component_index = pd.Index(components_str, name="component")

    node_counts_df = pd.DataFrame(
        X,
        index=component_index,
        columns=pd.Index(panel.markers, name="marker_id"),
    )

    logger.debug("Computing component metrics.")
    n_antibodies = (X > 0).sum(axis=1)

    components_metrics_df = pd.DataFrame(
        {
            "n_umi1": n_umi1_arr,
            "n_umi2": n_umi2_arr,
            "n_edges": n_edges_arr,
            "reads_in_component": reads_arr,
            "n_umi": n_umi1_arr + n_umi2_arr,
            "n_antibodies": n_antibodies,
        },
        index=component_index,
    )
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

    logger.debug("Computing antibody metrics.")
    antibody_metrics_df = calculate_antibody_metrics(counts_df=node_counts_df)
    antibody_metrics_df = antibody_metrics_df.reindex(index=panel.markers, fill_value=0)
    antibody_metrics_df.index.name = "marker_id"
    # Do a dtype conversion of the columns here since AnnData cannot handle
    # a pyarrow arrays.
    antibody_metrics_df = antibody_metrics_df.astype(
        {"antibody_count": "int64", "antibody_pct": "float32"}
    )
    antibody_metrics_df.index = antibody_metrics_df.index.astype(str)

    logger.debug("Building AnnData instance.")
    adata = AnnData(
        X=node_counts_df,
        obs=components_metrics_df,
        var=antibody_metrics_df,
    )

    adata = add_panel_information(adata, panel)

    total_marker_counts = node_counts_df.sum(axis=1)
    isotype_markers = adata.var[adata.var["control"]].index
    isotype_counts = node_counts_df[isotype_markers].sum(axis=1)
    adata.obs["isotype_fraction"] = isotype_counts / total_marker_counts

    # This is set to preserve backwards compatibility with downstream reports that may expect it.
    # Eventually we should be able to remove this.
    adata.obs["intracellular_fraction"] = 0.0

    return adata


def add_missing_adata_info(new_adata: AnnData, old_adata: AnnData) -> AnnData:
    """Add missing obs and var columns from old_adata to new_adata."""
    missing_obs = set(old_adata.obs.columns) - set(new_adata.obs.columns)
    missing_var = set(old_adata.var.columns) - set(new_adata.var.columns)

    new_adata.obs = new_adata.obs.join(old_adata.obs[list(missing_obs)], how="left")
    new_adata.var = new_adata.var.join(old_adata.var[list(missing_var)], how="left")

    return new_adata

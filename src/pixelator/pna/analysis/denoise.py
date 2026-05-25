"""Denoise module.

Functions and classes relating to denoising PNA components.

Copyright © 2025 Pixelgen Technologies AB
"""

import logging
import random
import tempfile
from itertools import chain
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr

from pixelator.common.annotate.aggregates import call_aggregates
from pixelator.pna.analysis_engine import PerComponentTask
from pixelator.pna.anndata import add_missing_adata_info, pna_edgelist_to_anndata
from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import PNAAntibodyPanel, load_antibody_panel
from pixelator.pna.graph import PNAGraph
from pixelator.pna.graph.adaptive_core_expansion import adaptive_core_expansion
from pixelator.pna.graph.node_pls import (
    GraphNormalizationOptions,
    create_node_neighborhood_abundance_matrix,
    node_pls,
)
from pixelator.pna.pixeldataset import PNAPixelDataset, read
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile

logger = logging.getLogger(__name__)


def _calculate_core_marker_counts(
    node_marker_counts: pd.DataFrame, node_core_numbers: pd.Series
) -> pd.DataFrame:
    """Calculate marker counts for one-core and higher-core nodes.

    Args:
        node_marker_counts: Node marker counts.
        node_core_numbers: Node core numbers.
    """
    one_core_counts = node_marker_counts[
        (node_core_numbers[node_marker_counts.index] == 1).values
    ].sum()
    higher_core_counts = node_marker_counts[
        (node_core_numbers[node_marker_counts.index] > 1).values
    ].sum()

    marker_counts = (
        one_core_counts.to_frame(name="one_core")
        .join(higher_core_counts.to_frame(name="higher_core"), how="outer")
        .fillna(0)
    )
    return marker_counts


def _perform_fishers_exact_test(
    marker_counts: pd.DataFrame,
    pval_significance_threshold: float,
) -> list:
    """Perform Fisher's exact test to identify over-expressed markers.

    Args:
        marker_counts: Marker counts.
        pval_significance_threshold: Pval significance threshold.
    """
    overexpressed_markers = []
    total_one_core_markers = marker_counts["one_core"].sum()
    total_higher_core_markers = marker_counts["higher_core"].sum()
    for marker in marker_counts.index:
        contingency_mat = [
            [
                marker_counts.loc[marker, "one_core"],
                total_one_core_markers - marker_counts.loc[marker, "one_core"],
            ],
            [
                marker_counts.loc[marker, "higher_core"],
                total_higher_core_markers - marker_counts.loc[marker, "higher_core"],
            ],
        ]
        pval = fisher_exact(contingency_mat, alternative="greater").pvalue
        if pval < pval_significance_threshold:
            overexpressed_markers.append(marker)
    return overexpressed_markers


def _calculate_excess_counts(
    marker_counts: pd.DataFrame,
    overexpressed_markers: list,
    inflate_factor: float,
) -> list:
    """Calculate inflated excess counts for over-expressed markers.

    Args:
        marker_counts: Marker counts.
        overexpressed_markers: Overexpressed markers.
        inflate_factor: Inflate factor.
    """
    results = []
    total_one_core_markers = marker_counts["one_core"].sum()
    total_higher_core_markers = marker_counts["higher_core"].sum()
    for marker in overexpressed_markers:
        expected_count = np.round(
            total_one_core_markers
            * marker_counts.loc[marker, "higher_core"]
            / total_higher_core_markers
        )
        excess = int(
            np.ceil(
                inflate_factor
                * (marker_counts.loc[marker, "one_core"] - expected_count)
            )
        )
        results.append((marker, excess))
    return results


def get_overexpressed_markers_in_one_core(
    node_marker_counts: pd.DataFrame,
    node_core_numbers: pd.Series,
    pval_significance_threshold: float = 0.05,
    inflate_factor: float = 1.5,
) -> pd.DataFrame:
    """Identify over-expressed markers in nodes belonging to core 1.

    This function calculates the over-expression of markers in nodes that belong
    to core 1 by comparing their marker counts to nodes in higher cores. It
    uses Fisher's exact test to determine statistical significance and inflates
    the excess count of markers based on the provided inflate factor.
    The inflate factor increases the number of nodes with a given over-expressed
    marker to be removed from the one-core layer.

    Args:
        node_marker_counts: A DataFrame where rows represent nodes and columns represent markers, with values indicating the count of each marker in each node.
        node_core_numbers: A Series where the index corresponds to the nodes and the values indicate the core number each node belongs to.
        pval_significance_threshold: The p-value threshold for statistical significance in Fisher's exact test. Defaults to 0.05.
        inflate_factor: A factor used to inflate the excess count of markers identified as overexpressed. Defaults to 1.5.

    Returns:
        DataFrame with columns ``name`` (overexpressed marker) and ``count`` (inflated excess count).
    """
    marker_counts = _calculate_core_marker_counts(node_marker_counts, node_core_numbers)

    overexpressed_markers = _perform_fishers_exact_test(
        marker_counts,
        pval_significance_threshold,
    )

    results = _calculate_excess_counts(
        marker_counts,
        overexpressed_markers,
        inflate_factor,
    )

    return pd.DataFrame(results, columns=["name", "count"])


def _sample_nodes_to_be_removed(
    node_marker_counts: pd.DataFrame, markers_to_remove: pd.DataFrame
) -> list:
    to_be_removed = []
    for _, marker in markers_to_remove.iterrows():
        marker_nodes = list(
            node_marker_counts[node_marker_counts[marker["name"]] > 0].index
        )
        marker_count_available = len(marker_nodes)
        n_to_be_removed = min(marker_count_available, marker["count"])
        rand_gen = random.Random(0)
        to_be_removed += list(rand_gen.sample(marker_nodes, n_to_be_removed))
    return to_be_removed


def get_stranded_nodes(component: PNAGraph, nodes_to_remove: list = []) -> list:
    """Identify nodes that become stranded after removing nodes_to_remove.

    Args:
        component: The graph component from which nodes will be removed.
        nodes_to_remove: A list of nodes to be removed from the graph.

    Returns:
        list: A list of stranded nodes that are disconnected from the largest connected component after nodes_to_remove are removed.
    """
    graph = component.raw.copy()
    graph.remove_nodes_from(nodes_to_remove)
    connected_components = sorted(nx.connected_components(graph), key=len, reverse=True)
    stranded_nodes = list(chain.from_iterable(connected_components[1:]))
    return stranded_nodes


def denoise_ace(
    component: PNAGraph,
    k: int = 3,
    max_k_core: int = 4,
    max_iter: int = 200,
    min_seed_pct: float = 0.1,
    nodes_to_move_threshold: int = 10,
    select_lcc: bool = True,
) -> list:
    """Partition the graph with ACE and return nodes in the peripheral-like ("low") partition.

    Nodes in the ``low`` partition are candidates for removal to retain a denser core-like graph.

    Args:
        component: The graph component to process.
        k: Neighborhood radius (steps) for the transition matrix in ACE.
        max_k_core: Maximum k-core layer used for seeding in ACE.
        max_iter: Maximum expansion iterations per binding threshold in ACE.
        min_seed_pct: Minimum fraction of nodes required for the initial ACE seed.
        nodes_to_move_threshold: ACE convergence threshold (nodes moved per iteration).
        select_lcc: If True, restrict the initial ACE seed to the largest connected component.

    Returns:
        List of node identifiers to remove.
    """
    try:
        adaptive_core_expansion(
            component,
            k=k,
            max_k_core=max_k_core,
            max_iter=max_iter,
            min_seed_pct=min_seed_pct,
            nodes_to_move_threshold=nodes_to_move_threshold,
            select_lcc=select_lcc,
        )
    except ValueError as exc:
        logger.debug("ACE denoising skipped for component: %s", exc)
        return []

    partitions = nx.get_node_attributes(component.raw, "partition")
    low_nodes = [n for n, part in partitions.items() if part == "low"]
    return low_nodes


def _pixel_type_design_matrix(component: PNAGraph, node_index: pd.Index) -> np.ndarray:
    """Build a 2-column design matrix [intercept, B-side dummy] matching R treatment coding.

    PNA graphs store bipartite side in the ``pixel_type`` node attribute (``A`` / ``B``).

    Args:
        component: Component.
        node_index: Node index.
    """
    pixel_type = nx.get_node_attributes(component.raw, "pixel_type")
    n = len(node_index)
    mat = np.zeros((n, 2), dtype=np.float64)
    mat[:, 0] = 1.0
    for i, node_id in enumerate(node_index):
        mat[i, 1] = 1.0 if pixel_type.get(node_id) == "B" else 0.0
    return mat


def _nodes_outside_largest_cc_after_pls_scores(
    raw: nx.Graph,
    node_index: pd.Index,
    passing_mask: np.ndarray,
) -> list:
    """Return nodes to drop: fail score filter or lie outside the largest CC among passers.

    Args:
        raw: Raw.
        node_index: Node index.
        passing_mask: Passing mask.
    """
    keep_candidates = [node_index[i] for i in range(len(node_index)) if passing_mask[i]]
    if not keep_candidates:
        return list(raw.nodes)

    sub = raw.subgraph(keep_candidates)
    largest = max(nx.connected_components(sub), key=len)
    largest_set = set(largest)
    return [n for n in raw.nodes if n not in largest_set]


def denoise_pls(
    component: PNAGraph,
    *,
    ncomp: int = 5,
    model_k: int = 2,
    pred_k: int = 1,
    use_weights: bool = True,
    normalization: GraphNormalizationOptions = "L1",
    residualize: bool = False,
    pls_component_p_threshold: float = 0.01,
    min_pls_coreness_correlation: float = 0.0,
    pls_score_threshold: float = -3.0,
) -> list:
    """PLS-on-coreness denoise: significant components, score gate, then largest connected set.

    Fits PLS with neighborhood radius ``model_k`` and scores nodes
    using radius ``pred_k``.

    Filtering uses **all** nodes (not only ACE ``high``); removals
    are returned for merging with other denoise methods.

    Args:
        component: Component graph (mutates ``coreness`` on nodes temporarily).
        ncomp: Requested PLS components (capped by sample size and feature count).
        model_k: Neighborhood steps for fitting X.
        pred_k: Neighborhood steps for prediction / scores.
        use_weights: Use edge weights in neighborhood expansion.
        normalization: Neighborhood matrix normalization.
        residualize: If True, residualize X against a ``pixel_type`` design matrix.
        pls_component_p_threshold: Per-component Pearson test vs coreness.
        min_pls_coreness_correlation: Minimum positive correlation.
        pls_score_threshold: All selected score columns must exceed this.

    Returns:
        Nodes to remove, ``[]`` if no PLS-based removal applies.
    """
    node_marker_counts = component.node_marker_counts
    idx = node_marker_counts.index
    n_samples = len(idx)
    n_features = node_marker_counts.shape[1]
    max_comp = min(ncomp, n_features, max(1, n_samples - 1))
    if max_comp < 1:
        logger.debug("PLS denoising skipped: insufficient samples or features.")
        return []

    coreness_series = pd.Series(nx.core_number(component.raw)).reindex(idx)
    if coreness_series.isna().any():
        logger.debug("PLS denoising skipped: missing coreness for some nodes.")
        return []

    nx.set_node_attributes(component.raw, coreness_series.to_dict(), "coreness")
    model_mat: Optional[np.ndarray] = None
    if residualize:
        model_mat = _pixel_type_design_matrix(component, idx)

    def _cleanup_coreness() -> None:
        for n in list(component.raw.nodes):
            data = component.raw.nodes[n]
            if isinstance(data, dict) and "coreness" in data:
                del data["coreness"]

    try:
        pls_model = node_pls(
            component,
            y_vars="coreness",
            k=model_k,
            use_weights=use_weights,
            ncomp=max_comp,
            normalization=normalization,
            model_mat=model_mat,
        )
        X_pred = create_node_neighborhood_abundance_matrix(
            component,
            k=pred_k,
            use_weights=use_weights,
            normalization=normalization,
            model_mat=model_mat,
        )
        # Save model scores for correlation testing and potential score flipping
        # to match R PLS orientation (positive correlation with coreness = more core-like).
        # Use the model scores to avoid discrepancies from different neighborhood expansion
        # in prediction vs model fitting.
        scores_model = pls_model.x_scores_.copy()
        scores = pls_model.transform(X_pred.values)
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.debug("PLS denoising skipped for component: %s", exc)
        _cleanup_coreness()
        return []

    _cleanup_coreness()

    n_scores = scores.shape[1]
    coreness_arr = coreness_series.astype(float).values
    comps_to_use: list[int] = []
    for j in range(n_scores):
        r, p = pearsonr(coreness_arr, scores[:, j])
        r_model, p_model = pearsonr(coreness_arr, scores_model[:, j])
        if np.isnan(p) or np.isnan(r):
            continue
        # If r_model is negative, flip the score direction for consistent interpretation
        # (higher score = more core-like). This is required to match the R implementation
        # (rpls) where scores are oriented by positive correlation with coreness.
        if r_model < 0:
            r = -r
            scores[:, j] = -scores[:, j]
        if r > min_pls_coreness_correlation and p < pls_component_p_threshold:
            comps_to_use.append(j)

    if not comps_to_use:
        return []

    passing = np.ones(n_samples, dtype=bool)
    for j in comps_to_use:
        passing &= scores[:, j] > pls_score_threshold

    return _nodes_outside_largest_cc_after_pls_scores(component.raw, idx, passing)


def denoise_one_core_layer(
    component: PNAGraph,
    pval_significance_threshold: float = 0.05,
    inflate_factor: float = 1.5,
    one_core_ratio_threshold: float = 0.9,
) -> list:
    """Identify markers over-expressed in the one-core layer and sample nodes to remove.

    This function analyzes the one-core layer of a graph, identifies markers
    that are over-expressed using a statistical significance threshold, and
    samples nodes associated with those markers for removal (bleed-over candidates).

    Args:
        component: The graph component to process, containing node marker counts and raw graph data.
        pval_significance_threshold: The p-value threshold for determining marker overexpression significance. Defaults to 0.05.
        inflate_factor: A factor used for inflating certain calculations (not explicitly used in the provided code). Defaults to 1.5.
        one_core_ratio_threshold: Components with higher nodes in their one-core layer are not denoised.

    Returns:
        list: Node ids sampled for removal from the one-core layer (bleed-over candidates). Does not include stranded nodes; callers merge with other denoise steps and then call ``get_stranded_nodes`` once on the combined set.
    """
    node_marker_counts = component.node_marker_counts
    node_core_numbers = pd.Series(nx.core_number(component.raw))
    if (node_core_numbers <= 1).mean() >= one_core_ratio_threshold:
        logger.debug(
            "Too many low core number nodes. Skipping denoising for this component."
        )
        return []  # Marking component as unqualified for denoising
    markers_to_remove = get_overexpressed_markers_in_one_core(
        node_marker_counts=node_marker_counts,
        node_core_numbers=node_core_numbers,
        pval_significance_threshold=pval_significance_threshold,
    )
    one_core_layer = node_marker_counts[
        ((node_core_numbers[node_marker_counts.index] == 1).values)
    ]
    nodes_to_remove = _sample_nodes_to_be_removed(one_core_layer, markers_to_remove)
    return nodes_to_remove


def write_denoised_edgelist(
    pxl: PNAPixelDataset, umis_to_remove: list, output_edgelist_path: str
):
    """Write a filtered edgelist (with specified UMIs removed) to a Parquet file.

    This function takes an existing PNAPixelDataset, removes edges associated
    with the specified UMIs, and writes the resulting edgelist to a Parquet file
    at the specified path. It does not create a new pixel (.pxl) file or update
    any AnnData object.

    Args:
        pxl: The original pixel dataset containing the edgelist.
        umis_to_remove: A list of UMIs (nodes) to be removed from the edgelist.
        output_edgelist_path: The file path where the filtered edgelist Parquet file will be saved.
    """
    with pxl.view.open() as session:
        session.get_connection().execute(
            f"""
            COPY(
                SELECT *
                FROM edgelist
                WHERE umi1 NOT IN (SELECT UNNEST(?))
                AND umi2 NOT IN (SELECT UNNEST(?))
            ) TO '{output_edgelist_path}' (FORMAT PARQUET)
            """,
            [umis_to_remove, umis_to_remove],
        )


class DenoiseGraph(PerComponentTask):
    """Graph denoising: one-core, ACE, and/or PLS on the full graph, merged removal plus stranding."""

    TASK_NAME = "denoise-graph"

    def __init__(
        self,
        *,
        run_one_core: bool = False,
        run_ace: bool = False,
        run_pls: bool = False,
        pval_significance_threshold: float = 0.05,
        inflate_factor: float = 1.5,
        one_core_ratio_threshold: float = 0.9,
        k: int = 3,
        max_k_core: int = 4,
        max_iter: int = 200,
        min_seed_pct: float = 0.1,
        nodes_to_move_threshold: int = 10,
        ace_select_lcc: bool = True,
        pls_ncomp: int = 5,
        pls_model_k: int = 2,
        pls_pred_k: int = 1,
        pls_use_weights: bool = True,
        pls_normalization: GraphNormalizationOptions = "L1",
        pls_residualize: bool = False,
        pls_component_p_threshold: float = 0.01,
        min_pls_coreness_correlation: float = 0.0,
        pls_score_threshold: float = -3.0,
    ):
        """Configure which denoise steps run and their hyperparameters.

        Args:
            run_one_core: Run one core.
            run_ace: Run ace.
            run_pls: Run pls.
            pval_significance_threshold: Pval significance threshold.
            inflate_factor: Inflate factor.
            one_core_ratio_threshold: One core ratio threshold.
            k: K.
            max_k_core: Max k core.
            max_iter: Max iter.
            min_seed_pct: Min seed pct.
            nodes_to_move_threshold: Nodes to move threshold.
            ace_select_lcc: Ace select lcc.
            pls_ncomp: Pls ncomp.
            pls_model_k: Pls model k.
            pls_pred_k: Pls pred k.
            pls_use_weights: Pls use weights.
            pls_normalization: Pls normalization.
            pls_residualize: Pls residualize.
            pls_component_p_threshold: Pls component p threshold.
            min_pls_coreness_correlation: Min pls coreness correlation.
            pls_score_threshold: Pls score threshold.
        """
        if not run_one_core and not run_ace and not run_pls:
            raise ValueError(
                "At least one of run_one_core, run_ace, or run_pls must be True."
            )
        self.run_one_core = run_one_core
        self.run_ace = run_ace
        self.run_pls = run_pls
        self.pval_significance_threshold = pval_significance_threshold
        self.inflate_factor = inflate_factor
        self.one_core_ratio_threshold = one_core_ratio_threshold
        self.k = k
        self.max_k_core = max_k_core
        self.max_iter = max_iter
        self.min_seed_pct = min_seed_pct
        self.nodes_to_move_threshold = nodes_to_move_threshold
        self.ace_select_lcc = ace_select_lcc
        self.pls_ncomp = pls_ncomp
        self.pls_model_k = pls_model_k
        self.pls_pred_k = pls_pred_k
        self.pls_use_weights = pls_use_weights
        self.pls_normalization = pls_normalization
        self.pls_residualize = pls_residualize
        self.pls_component_p_threshold = pls_component_p_threshold
        self.min_pls_coreness_correlation = min_pls_coreness_correlation
        self.pls_score_threshold = pls_score_threshold

    def run_on_component_graph(
        self, component: PNAGraph, component_id: str
    ) -> pd.DataFrame:
        """Return nodes to remove (including stranded) and per-method metadata columns.

        One-core, ACE, and PLS each see the same full ``component`` graph; removals are
        merged afterward so no step is conditioned on another's output.

        Args:
            component: Component.
            component_id: Component id.
        """
        one_core_marked: list = []
        ace_marked: list = []
        pls_marked: list = []

        nodes_to_remove = pd.DataFrame(
            columns=[
                "umi",
                "component",
                "marked_by",
            ]
        )

        def _append_marked_count(
            old_nodes_to_remove: pd.DataFrame, new_nodes: list, method_name: str
        ) -> pd.DataFrame:
            n_new = len(new_nodes) if new_nodes and new_nodes != [None] else 0
            new_marked_df = pd.DataFrame(
                {
                    "umi": new_nodes if new_nodes and new_nodes != [None] else [],
                    "component": [component_id] * n_new,
                    "marked_by": [method_name] * n_new,
                }
            )
            return pd.concat([old_nodes_to_remove, new_marked_df], ignore_index=True)

        if self.run_one_core:
            logger.debug("Running one-core denoising on component %s", component_id)
            one_core_marked = denoise_one_core_layer(
                component,
                pval_significance_threshold=self.pval_significance_threshold,
                inflate_factor=self.inflate_factor,
                one_core_ratio_threshold=self.one_core_ratio_threshold,
            )
            nodes_to_remove = _append_marked_count(
                nodes_to_remove, one_core_marked, "one_core"
            )

        if self.run_ace:
            logger.debug("Running ACE denoising on component %s", component_id)
            ace_marked = denoise_ace(
                component,
                k=self.k,
                max_k_core=self.max_k_core,
                max_iter=self.max_iter,
                min_seed_pct=self.min_seed_pct,
                nodes_to_move_threshold=self.nodes_to_move_threshold,
                select_lcc=self.ace_select_lcc,
            )
            nodes_to_remove = _append_marked_count(nodes_to_remove, ace_marked, "ace")

        if self.run_pls:
            logger.debug("Running PLS denoising on component %s", component_id)
            pls_marked = denoise_pls(
                component,
                ncomp=self.pls_ncomp,
                model_k=self.pls_model_k,
                pred_k=self.pls_pred_k,
                use_weights=self.pls_use_weights,
                normalization=self.pls_normalization,
                residualize=self.pls_residualize,
                pls_component_p_threshold=self.pls_component_p_threshold,
                min_pls_coreness_correlation=self.min_pls_coreness_correlation,
                pls_score_threshold=self.pls_score_threshold,
            )
            nodes_to_remove = _append_marked_count(nodes_to_remove, pls_marked, "pls")

        combined = list(set(one_core_marked + ace_marked + pls_marked))
        stranded = get_stranded_nodes(component, combined)
        nodes_to_remove = _append_marked_count(nodes_to_remove, stranded, "stranded")

        return nodes_to_remove

    def add_to_pixel_file(self, data: pd.DataFrame, pxl_file_target: PxlFile):
        """Filter edgelist by removed nodes and write denoise metrics to AnnData.

        Args:
            data: Data.
            pxl_file_target: Pxl file target.
        """
        pxl = PNAPixelDataset.from_files(pxl_file_target)
        old_adata = pxl.adata()
        try:
            panel = PNAAntibodyPanel.from_pxl_dataset(read(pxl_file_target.path))
        except KeyError:
            # If pxl file does not contain panel data, try to load it from
            # the panel name.
            # This will happen when old pxl files generated before v0.22.0
            # are used.
            panel_name = pxl.metadata().popitem()[1]["panel_name"]
            panel = load_antibody_panel(pna_config, panel_name)
        nodes_to_remove = (
            data.loc[~data["umi"].isna(), "umi"].astype(np.uint64).tolist()
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            denoised_edgelist_path = temp_dir + "/denoised_edgelist.parquet"
            write_denoised_edgelist(pxl, nodes_to_remove, denoised_edgelist_path)
            with PixelFileWriter(pxl_file_target.path) as writer:
                writer.write_edgelist(Path(denoised_edgelist_path))
                adata = pna_edgelist_to_anndata(writer.get_connection(), panel)
                call_aggregates(adata)
                old_adata.obs.rename(
                    columns={"isotype_fraction": "pre_denoise_isotype_fraction"},
                    inplace=True,
                )
                adata = add_missing_adata_info(adata, old_adata)

                denoise_info = _collect_denoise_summary_info(
                    data, comp_index=adata.obs.index
                )

                for col in denoise_info.columns:
                    if col in adata.obs.columns:
                        adata.obs = adata.obs.drop(columns=[col])
                adata.obs = adata.obs.join(denoise_info, how="left")

                writer.write_adata(adata)


def _collect_denoise_summary_info(
    data: pd.DataFrame, comp_index: pd.Index
) -> pd.DataFrame:
    """Collect summary information about the denoising process for each component.

    Args:
        data: Data.
        comp_index: Comp index.
    """
    denoise_info = pd.DataFrame(index=comp_index)

    n_total_removed = (
        data[["component", "umi"]].drop_duplicates().groupby("component").size()
    )
    denoise_info["number_of_nodes_removed_in_denoise"] = (
        pd.to_numeric(n_total_removed.reindex(denoise_info.index), errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    summary_cols = [
        "denoised_nodes_marked_only_by_ace",
        "denoised_nodes_marked_only_by_pls",
        "denoised_nodes_marked_only_by_one_core",
        "denoised_nodes_marked_stranded",
        "denoised_nodes_marked_ace_and_pls",
        "denoised_nodes_marked_ace_and_one_core",
        "denoised_nodes_marked_pls_and_one_core",
        "denoised_nodes_marked_ace_pls_and_one_core",
    ]
    mark_summary = (
        data.groupby(["component", "umi"])["marked_by"].apply(list).reset_index()
    )
    mark_summary["denoised_nodes_marked_only_by_ace"] = mark_summary["marked_by"].apply(
        lambda x: x == ["ace"]
    )
    mark_summary["denoised_nodes_marked_only_by_pls"] = mark_summary["marked_by"].apply(
        lambda x: x == ["pls"]
    )
    mark_summary["denoised_nodes_marked_only_by_one_core"] = mark_summary[
        "marked_by"
    ].apply(lambda x: x == ["one_core"])
    mark_summary["denoised_nodes_marked_stranded"] = mark_summary["marked_by"].apply(
        lambda x: "stranded" in x
    )
    mark_summary["denoised_nodes_marked_ace_and_pls"] = mark_summary["marked_by"].apply(
        lambda x: set(x) == {"ace", "pls"}
    )
    mark_summary["denoised_nodes_marked_ace_and_one_core"] = mark_summary[
        "marked_by"
    ].apply(lambda x: set(x) == {"ace", "one_core"})
    mark_summary["denoised_nodes_marked_pls_and_one_core"] = mark_summary[
        "marked_by"
    ].apply(lambda x: set(x) == {"pls", "one_core"})
    mark_summary["denoised_nodes_marked_ace_pls_and_one_core"] = mark_summary[
        "marked_by"
    ].apply(lambda x: set(x) == {"ace", "pls", "one_core"})
    for col in summary_cols:
        col_summary = (
            mark_summary.groupby("component")[col]
            .sum()
            .reindex(comp_index)
            .fillna(0)
            .astype("int64")
        )
        denoise_info[col] = col_summary
    return denoise_info

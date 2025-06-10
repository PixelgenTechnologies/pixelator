"""Denoise module.

Functions and classes relating to denoising PNA components.

Copyright © 2025 Pixelgen Technologies AB
"""

import logging
import random
from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import fisher_exact

from pixelator.common.annotate.aggregates import call_aggregates
from pixelator.pna.analysis_engine import PerComponentTask
from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import load_antibody_panel
from pixelator.pna.graph import PNAGraph
from pixelator.pna.pixeldataset import PNAPixelDataset, PxlFile
from pixelator.pna.pixeldataset.io import PixelFileWriter

logger = logging.getLogger(__name__)


def _calculate_core_marker_counts(
    node_marker_counts: pd.DataFrame, node_core_numbers: pd.Series
) -> pd.DataFrame:
    """Calculate marker counts for one-core and higher-core nodes."""
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
    """Perform Fisher's exact test to identify over-expressed markers."""
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
    """Calculate inflated excess counts for over-expressed markers."""
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
        node_marker_counts (pd.DataFrame): A DataFrame where rows represent
            nodes and columns represent markers, with values indicating the
            count of each marker in each node.
        node_core_numbers (pd.Series): A Series where the index corresponds to
            the nodes and the values indicate the core number each node belongs to.
        pval_significance_threshold (float, optional): The p-value threshold
            for statistical significance in Fisher's exact test. Defaults to 0.05.
        inflate_factor (float, optional): A factor used to inflate the excess
            count of markers identified as overexpressed. Defaults to 1.5.
        one_core_ratio_threshold(float, optional): Components with higher nodes in
                                                   their one-core layer are not denoised.

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - "name": The names of the overexpressed markers.
            - "count": The inflated excess count of each overexpressed marker.

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
        component (PNAGraph): The graph component from which nodes will be removed.
        nodes_to_remove (list): A list of nodes to be removed from the graph.

    Returns:
        list: A list of stranded nodes that are disconnected from the largest
        connected component after nodes_to_remove are removed.

    """
    graph = component.raw.copy()
    graph.remove_nodes_from(nodes_to_remove)
    connected_components = sorted(nx.connected_components(graph), key=len, reverse=True)
    stranded_nodes = list(chain.from_iterable(connected_components[1:]))
    return stranded_nodes


def denoise_one_core_layer(
    component: PNAGraph,
    pval_significance_threshold: float = 0.05,
    inflate_factor: float = 1.5,
    one_core_ratio_threshold: float = 0.9,
) -> list:
    """Identify and remove markers over-expressed in the one-core layer of a graph.

    This function analyzes the one-core layer of a graph, identifies markers
    that are over-expressed using a statistical significance threshold, and
    removes nodes associated with those markers. Additionally, it ensures
    that stranded nodes (nodes disconnected from the graph due to removal)
    are also removed.

    Args:
        component (PNAGraph): The graph component to process, containing
            node marker counts and raw graph data.
        pval_significance_threshold (float, optional): The p-value threshold
            for determining marker overexpression significance. Defaults to 0.05.
        inflate_factor (float, optional): A factor used for inflating certain
            calculations (not explicitly used in the provided code). Defaults to 1.5.
        one_core_ratio_threshold(float, optional): Components with higher nodes in
                                                   their one-core layer are not denoised.

    Returns:
        list: A list of nodes to be removed from the one-core layer of the graph.

    """
    node_marker_counts = component.node_marker_counts
    node_core_numbers = pd.Series(nx.core_number(component.raw))
    if (node_core_numbers <= 1).mean() >= one_core_ratio_threshold:
        logger.debug(
            "Too many low core number nodes. Skipping denoising for this component."
        )
        return [None]  # Marking component as unqualified for denoising
    markers_to_remove = get_overexpressed_markers_in_one_core(
        node_marker_counts=node_marker_counts,
        node_core_numbers=node_core_numbers,
        pval_significance_threshold=pval_significance_threshold,
    )
    one_core_layer = node_marker_counts[
        ((node_core_numbers[node_marker_counts.index] == 1).values)
    ]
    nodes_to_remove = _sample_nodes_to_be_removed(one_core_layer, markers_to_remove)
    stranded_nodes = get_stranded_nodes(component, nodes_to_remove)
    nodes_to_remove += stranded_nodes
    return nodes_to_remove


class DenoiseOneCore(PerComponentTask):
    """Denoise bleed-over markers in parts of the component with low coreness."""

    TASK_NAME = "denoise-one-core"

    def __init__(
        self,
        pval_significance_threshold: float = 0.05,
        inflate_factor: float = 1.5,
        one_core_ratio_threshold: float = 0.9,
    ):
        """Initialize a DenoiseOneCore instance.

        Args:
            pval_significance_threshold (float): The p-value threshold for considering
            a marker over-expressed in the one-core layer.
            inflate_factor: A factor used to inflate the excess count of markers.
            one_core_ratio_threshold: Components with higher nodes in their one-core
                                      layer are not denoised.

        """
        self.pval_significance_threshold = pval_significance_threshold
        self.inflate_factor = inflate_factor
        self.one_core_ratio_threshold = one_core_ratio_threshold

    def run_on_component_graph(
        self, component: PNAGraph, component_id: str
    ) -> pd.DataFrame:
        """Execute one-core denoising on a given component graph and return nodes to remove.

        This function performs denoising on the provided PNAGraph component by
        identifying nodes to be removed based on a single core layer denoising
        process. The resulting nodes are returned in a DataFrame along with
        their associated component ID.

        Args:
            component (PNAGraph): The graph component to be denoised.
            component_id (str): The identifier for the graph component.

        Returns:
            pd.DataFrame: A DataFrame containing the nodes to be removed with the following columns:
                - "umi": The unique identifier of the node to be removed.
                - "component": The ID of the component the node belongs to.

        """
        logger.debug(f"Running low-core denoising on component {component_id}")
        nodes_to_remove = pd.DataFrame(
            denoise_one_core_layer(
                component,
                pval_significance_threshold=self.pval_significance_threshold,
                inflate_factor=self.inflate_factor,
                one_core_ratio_threshold=self.one_core_ratio_threshold,
            ),
            columns=["umi"],
        )
        nodes_to_remove["component"] = component_id
        return nodes_to_remove

    def add_to_pixel_file(self, data: pd.DataFrame, pxl_file_target: PxlFile):
        """Add denoised component to a pixel file by updating the edgelist and adata.

        This function reads an existing pixel file, filters its edgelist
        based on the provided data, and updates the file with the modified
        edgelist and corresponding AnnData object.

        Args:
            data (pd.DataFrame): A DataFrame containing the data to be used
                for filtering.
            pxl_file_target (PxlFile): The target pixel file to which
                the data will be added.

        """
        pxl = PNAPixelDataset.from_files(pxl_file_target)
        panel_name = pxl.metadata().popitem()[1]["panel_name"]
        panel = load_antibody_panel(pna_config, panel_name)
        nodes_to_remove = pl.Series(
            data.loc[~data["umi"].isna(), "umi"], dtype=pl.UInt64
        )

        edgelist = (
            pxl.edgelist()
            .to_polars()
            .drop("sample", strict=False)
            .filter(
                pl.min_horizontal(
                    ~pl.col("umi1").is_in(nodes_to_remove),
                    ~pl.col("umi2").is_in(nodes_to_remove),
                )
            )
        )

        adata = pna_edgelist_to_anndata(edgelist.lazy(), panel)
        call_aggregates(adata)
        denoise_info = pd.DataFrame(index=adata.obs.index)
        denoise_info["disqualified_for_denoising"] = False
        denoise_info.loc[
            data.loc[data["umi"].isna(), "component"], "disqualified_for_denoising"
        ] = True
        n_umis_removed = data.loc[~data["umi"].isna(), :].groupby("component").size()
        denoise_info["number_of_nodes_removed_in_denoise"] = n_umis_removed
        denoise_info["number_of_nodes_removed_in_denoise"] = denoise_info[
            "number_of_nodes_removed_in_denoise"
        ].fillna(0)
        adata.obs = adata.obs.join(denoise_info, how="left")

        with PixelFileWriter(pxl_file_target.path) as writer:
            writer.write_edgelist(edgelist)
            writer.write_adata(adata)

"""Connected components module.

Functions and classes relating to building connected components in the graph step.

Copyright © 2024 Pixelgen Technologies AB
"""

import logging
import typing
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Literal

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import xxhash
from graspologic_native import leiden

from pixelator.common.annotate.aggregates import call_aggregates
from pixelator.common.annotate.cell_calling import find_component_size_limits
from pixelator.common.exceptions import PixelatorBaseException
from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config import PNAAntibodyPanel
from pixelator.pna.graph.report import (
    GraphStatistics,
)
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.io import PixelFileWriter

LEIDEN_RANDOM_SEED = 1
MIN_PNA_COMPONENT_SIZE = 8000

logger = logging.getLogger(__name__)


class ConnectedComponentException(PixelatorBaseException):
    """Exception raised when there is an issue with connected components."""


def _calculate_component_stats(
    connected_components: Iterable[set[int]],
) -> tuple[int, float]:
    if not isinstance(connected_components, list):
        connected_components = list(connected_components)
    max_component_size = 0
    total_graph_size = 0
    nbr_of_components = 0
    for component in connected_components:
        nbr_of_components += 1
        total_graph_size += len(component)
        max_component_size = max(max_component_size, len(component))
    return (
        nbr_of_components,
        max_component_size / total_graph_size if total_graph_size else 0,
    )


def _filter_edgelist(
    edgelist: pl.LazyFrame,
    min_read_count: int,
    component_stats: GraphStatistics,
):
    """Filter the edgelist by read count."""
    edgelist = edgelist.filter(pl.col("read_count") >= min_read_count)
    post_filter_stat = edgelist.select(
        [
            pl.col("read_count").sum(),
            pl.col("uei_count").sum(),
            pl.col("umi1").n_unique().alias("umi1_count"),
            pl.col("umi2").n_unique().alias("umi2_count"),
            pl.len().alias("edge_count"),
        ]
    ).collect()
    component_stats.molecules_post_read_count_filtering = post_filter_stat["uei_count"][
        0
    ]
    component_stats.reads_post_read_count_filtering = post_filter_stat["read_count"][0]
    component_stats.node_count_pre_recovery = (
        post_filter_stat["umi1_count"] + post_filter_stat["umi2_count"]
    )[0]
    component_stats.edge_count_pre_recovery = post_filter_stat["edge_count"][0]
    return edgelist


def _label_connected_components(
    edgelist: pl.LazyFrame,
    component_stats: GraphStatistics,
):
    """Build the graph and calculate some statistics."""
    logger.debug("Finding connected components")

    graph = nx.from_edgelist(edgelist.select(["umi1", "umi2"]).collect().rows())
    connected_components = list(nx.connected_components(graph))
    umi_component_map = dict()
    for i, cc in enumerate(connected_components):
        umi_component_map.update(dict.fromkeys(cc, i))
    del graph

    (
        component_count,
        fraction_nodes_in_largest_component,
    ) = _calculate_component_stats(connected_components)
    del connected_components

    if component_count == 0:
        raise ConnectedComponentException(
            "No connected components found in the graph. Will not continue with the rest of the computations."
        )

    component_stats.component_count_pre_recovery = component_count
    component_stats.fraction_nodes_in_largest_component_pre_recovery = (
        fraction_nodes_in_largest_component
    )

    return umi_component_map, component_stats


@dataclass
class RefinementOptions:
    """Options for refining components."""

    min_component_size: int = 10

    max_edges_to_remove: int | None = 20
    max_edges_to_remove_relative: float | None = None
    leiden_resolution: float = 1.0
    min_component_size_to_prune: int = 100


@dataclass
class StagedRefinementOptions:
    """Options for refining components at each stage."""

    inital_stage_options: RefinementOptions = field(
        default_factory=lambda: RefinementOptions(
            max_edges_to_remove=None,
            max_edges_to_remove_relative=None,
            leiden_resolution=1.0,
        )
    )
    refinement_stage_options: RefinementOptions = field(
        default_factory=lambda: RefinementOptions(
            max_edges_to_remove=4,
            max_edges_to_remove_relative=None,
            leiden_resolution=0.01,
        )
    )
    max_component_refinement_depth: int = 1


@dataclass(slots=True)
class MultipletRecoveryStats:
    """Statistics from multiplet recovery."""

    crossing_edges_removed: int = 0
    crossing_edges_removed_in_initial_stage: int = 0
    max_recursion_depth: int = 0


def merge_communities_with_many_crossing_edges(
    edgelist: pl.DataFrame,
    node_community_dict: dict,
    max_edges_to_remove: int | None = None,
    max_edges_to_remove_relative: float | None = None,
) -> pd.Series:
    """Merge communities what have many crossing edges between them in an edge list.

    This function takes an edge list and a dictionary with the community mapping
    for each node. It then computes the number of edges between communities and
    if they are higher than the given a threshold. It assigns the same community
    id to the nodes in the connected strongly connected communities. The
    threshold is determined by an absolute count and/or relative to the number
    of nodes in the smaller of the two communities, whichever is higher.
    If one of them is None, only the other one is considered and if they are
    both None, the split communities are not considered for merging.

    :param edgelist: The edge list to process
    :param node_community_dict: A dictionary mapping each node to a community
    :param n_edges: The threshold for the number of edges to be found between
    communities to merge or None to avoid merging
    :returns: The updated community mapping
    """
    community_serie = pd.Series(node_community_dict)
    if max_edges_to_remove is None and max_edges_to_remove_relative is None:
        return community_serie
    if community_serie.nunique() == 1:
        return community_serie

    community_sizes = community_serie.value_counts().to_dict()
    # This will be an edgelist where every node is a community
    # and there is an edge when there are more than maximum_edges to
    # remove between the communities.
    community_edgelist = (
        edgelist.with_columns(
            community1=pl.col("umi1").replace_strict(node_community_dict),
            community2=pl.col("umi2").replace_strict(node_community_dict),
        )
        .with_columns(
            community1=pl.min_horizontal(pl.col("community1"), pl.col("community2")),
            community2=pl.max_horizontal(pl.col("community1"), pl.col("community2")),
        )
        .filter(pl.col("community1") != pl.col("community2"))
        .group_by(["community1", "community2"])
        .len()
        .with_columns(
            community1_size=pl.col("community1").replace_strict(community_sizes),
            community2_size=pl.col("community2").replace_strict(community_sizes),
        )
        .with_columns(
            threshold=pl.max_horizontal(
                max_edges_to_remove,
                max_edges_to_remove_relative
                * pl.min_horizontal(
                    pl.col("community1_size"), pl.col("community2_size")
                ),
            ),
        )
        .filter(pl.col("len") >= pl.col("threshold"))
        .select(["community1", "community2"])
        .rows()
    )

    communities_graph = nx.from_edgelist(community_edgelist)
    for cc in nx.connected_components(communities_graph):
        new_tag = min(cc)
        community_serie[community_serie.isin(cc)] = new_tag

    return community_serie


def _get_umi_component_map_from_edgelist(edgelist: pl.LazyFrame) -> dict:
    """Get a umi component map from an edgelist."""
    umi_component_map = dict()
    if "component1" in edgelist.columns:
        component_umis = (
            edgelist.group_by("component1")
            .agg(pl.col("umi1").unique(), pl.col("umi2").unique())
            .collect()
        )
        for comp, umi1, umi2 in component_umis.rows():
            umi_component_map.update(dict.fromkeys(umi1 + umi2, comp))

    else:
        umis = set(edgelist.select("umi1").unique().collect()["umi1"]).union(
            set(edgelist.select("umi2").unique().collect()["umi2"])
        )
        umi_component_map.update(dict.fromkeys(umis, 0))
    return umi_component_map


def _update_components_column(
    edgelist: pl.LazyFrame, umi_component_map: dict
) -> pl.LazyFrame:
    """Update the component column in an edgelist."""
    return edgelist.with_columns(
        component1=pl.col("umi1").replace_strict(umi_component_map),
        component2=pl.col("umi2").replace_strict(umi_component_map),
    )


def make_edgelits_with_component_column(
    edgelist: pl.LazyFrame, umi_component_map: dict
) -> pl.LazyFrame:
    """Add a component column to an edgelist using a node map and remove crossing edges.

    :param edgelist: The edgelist to add the component column to.
    :param umi_component_map: A dictionary mapping nodes to components.
    :returns: The edgelist with crossing edges removed and the component column
    added.
    """
    return (
        _update_components_column(edgelist, umi_component_map)
        .filter(pl.col("component1") == pl.col("component2"))
        .drop("component2")
        .rename({"component1": "component"})
    )


def _find_initial_components_to_be_refined(
    edgelist: pl.LazyFrame, min_component_size: int
) -> list[str]:
    return list(
        edgelist.group_by("component1")
        .agg(
            pl.col("umi1").n_unique().alias("n_umi1"),
            pl.col("umi2").n_unique().alias("n_umi2"),
        )
        .filter((pl.col("n_umi1") + pl.col("n_umi2")) > min_component_size)
        .select("component1")
        .unique()
        .collect()["component1"]
    )


def _refine_components(
    umi_component_map: dict,
    clusters_to_be_refined: pl.dataframe.group_by.GroupBy,
    leiden_iterations: int,
    refinement_options: RefinementOptions,
) -> tuple[dict, list]:
    def id_generator(start=0):
        next_id = start
        while True:
            yield next_id
            next_id += 1

    id_gen = id_generator(max(umi_component_map.values()) + 1)

    refinement_candidates = []
    for _, cluster_edges in clusters_to_be_refined:
        number_of_nodes = cluster_edges.n_unique("umi1") + cluster_edges.n_unique(
            "umi2"
        )
        if number_of_nodes < refinement_options.min_component_size:
            continue

        _, leiden_communities = leiden(
            cluster_edges.select(
                pl.col("umi1").cast(pl.Utf8), pl.col("umi2").cast(pl.Utf8)
            )
            .with_columns(weight=pl.lit(1))
            .rows(),
            seed=LEIDEN_RANDOM_SEED,
            use_modularity=True,
            resolution=refinement_options.leiden_resolution,
            # These parameters are used to sync up the native implementation with
            # the python implementation we originally used.
            iterations=leiden_iterations + 1,
            randomness=0.001,
            trials=1,
            starting_communities=None,
        )
        # Map the communites back from strings to integers
        leiden_communities = {int(k): v for k, v in leiden_communities.items()}  # type: ignore
        community_serie = merge_communities_with_many_crossing_edges(
            cluster_edges,
            leiden_communities,
            refinement_options.max_edges_to_remove,
            refinement_options.max_edges_to_remove_relative,
        )
        if community_serie.nunique() > 1:
            for _, nodes in community_serie.groupby(community_serie):
                community_id = next(id_gen)
                umi_component_map.update(dict.fromkeys(nodes.index, community_id))
                if len(nodes) > refinement_options.min_component_size:
                    refinement_candidates.append(community_id)
    return umi_component_map, refinement_candidates


def recover_multiplets(
    edgelist: pl.LazyFrame,
    umi_component_map: dict | None = None,
    leiden_iterations: int = 1,
    refinement_options: StagedRefinementOptions | None = None,
) -> tuple[nx.Graph, MultipletRecoveryStats]:
    """Recovery multiplets by leiden community detection, removing crossing edges between communities."""
    if umi_component_map is None:
        umi_component_map = _get_umi_component_map_from_edgelist(edgelist)

    if refinement_options is None:
        refinement_options = StagedRefinementOptions(
            inital_stage_options=RefinementOptions(),
            refinement_stage_options=RefinementOptions(),
        )

    edgelist_with_components = _update_components_column(edgelist, umi_component_map)
    to_be_refined = _find_initial_components_to_be_refined(
        edgelist_with_components,
        refinement_options.inital_stage_options.min_component_size,
    )

    max_recursion_depth_observed = 0
    for recursion_level in range(refinement_options.max_component_refinement_depth):
        logger.debug(
            "Running Leiden community detection. At recursion depth %s", recursion_level
        )

        clusters_to_be_refined = (
            edgelist_with_components.filter(pl.col("component1").is_in(to_be_refined))
            # Remove the crossing edges, will do nothing on the first iteration
            .filter(pl.col("component1") == pl.col("component2"))
            .collect()
            .group_by("component1")
        )
        umi_component_map, to_be_refined = _refine_components(
            umi_component_map=umi_component_map,  # type: ignore
            clusters_to_be_refined=clusters_to_be_refined,
            leiden_iterations=leiden_iterations,
            refinement_options=(
                refinement_options.inital_stage_options
                if recursion_level == 0
                else refinement_options.refinement_stage_options
            ),
        )
        edgelist_with_components = _update_components_column(
            edgelist,
            umi_component_map,  # type: ignore
        )
        if recursion_level == 0:
            nbr_of_edges_removed_in_initial_stage = (
                edgelist_with_components.filter(
                    pl.col("component1") != pl.col("component2")
                )
                .select(pl.len())
                .collect()
            )["len"][0]

        if not to_be_refined:
            logger.debug(
                "No more components to refine, will break the refinement loop."
            )
            max_recursion_depth_observed = recursion_level
            break

    total_nbr_crossing_edges = (
        edgelist_with_components.filter(pl.col("component1") != pl.col("component2"))
        .select(pl.len())
        .collect()
    )["len"][0]
    return umi_component_map, MultipletRecoveryStats(
        crossing_edges_removed=total_nbr_crossing_edges,
        crossing_edges_removed_in_initial_stage=nbr_of_edges_removed_in_initial_stage,
        max_recursion_depth=max_recursion_depth_observed,
    )


def filter_components_by_size_dynamic(
    component_sizes: pd.DataFrame,
    lowest_passable_bound: int | None = MIN_PNA_COMPONENT_SIZE,
) -> tuple[pd.DataFrame, int | None]:
    """Filter components by size using dynamic thresholds.

    :param component_sizes: A DataFrame with columns `component` and `n_umi`.
    :returns: only the 'component' column of with the components that pass the filter, and the lower bound.
    """
    if lowest_passable_bound is None:
        lowest_passable_bound = MIN_PNA_COMPONENT_SIZE

    lower_bound = find_component_size_limits(
        component_sizes=component_sizes["n_umi"].to_numpy(), direction="lower"
    )
    if lower_bound is None or lower_bound < lowest_passable_bound:
        lower_bound = lowest_passable_bound
        logger.warning(
            "Could not find a lower bound for component size filtering, will "
            "set the lower bound to " + str(lowest_passable_bound),
        )
    return (
        component_sizes.filter(pl.col("n_umi") >= lower_bound)["component"],
        lower_bound,
    )


def filter_components_by_size_hard_thresholds(
    component_sizes: pd.DataFrame, lower_bound: int | None, higher_bound: int | None
) -> pd.DataFrame:
    """Filter components by size using hard thresholds.

    :param component_sizes: A DataFrame with columns `component` and `n_umi`.
    :param lower_bound: The lower bound for the component size.
    :param higher_bound: The higher bound for the component size.
    :returns: only the 'component' column of with the components that pass the filter.
    """
    if lower_bound is None:
        lower_bound = 0
    if higher_bound is None:
        higher_bound = np.iinfo(np.uint64).max
    return component_sizes.filter(
        (pl.col("n_umi") >= lower_bound) & (pl.col("n_umi") <= higher_bound)
    )["component"]


def hash_component(component: set[int]) -> str:
    """Hash a component. Should yield the same hash if the components consists of the same nodes."""
    hasher = xxhash.xxh3_64()
    for node in sorted(component):
        hasher.update(node.to_bytes(length=8, byteorder="little"))
    return hasher.hexdigest()


def _find_clashing_umis(
    molecules_lazy_frame: pl.LazyFrame,
) -> tuple[pl.Series, pl.Series]:
    # Find umis that are associated with multiple markers
    # indicating a clash.
    umi1_clashes = (
        molecules_lazy_frame.select(["umi1", "marker_1"])
        .unique()
        .group_by("umi1")
        .len()
        .filter(pl.col("len") > 1)
        .collect()["umi1"]
    )
    umi2_clashes = (
        molecules_lazy_frame.select(["umi2", "marker_2"])
        .unique()
        .group_by("umi2")
        .len()
        .filter(pl.col("len") > 1)
        .collect()["umi2"]
    )

    # Find umis that are present in both umi1 and umi2
    # which also indicates a clash
    umi1_umi2_clashes = (
        pl.concat(
            (
                molecules_lazy_frame.select("umi1").unique().rename({"umi1": "umi"}),
                molecules_lazy_frame.select("umi2").unique().rename({"umi2": "umi"}),
            )
        )
        .group_by("umi")
        .len()
        .filter(pl.col("len") > 1)
        .collect()["umi"]
    )
    umi1_clashes = umi1_clashes.extend(umi1_umi2_clashes)
    umi2_clashes = umi2_clashes.extend(umi1_umi2_clashes)
    return umi1_clashes, umi2_clashes


def build_pxl_file_with_components(
    molecules_lazy_frame: pl.LazyFrame,
    panel: PNAAntibodyPanel,
    sample_name: str,
    path_output_pxl_file: Path,
    multiplet_recovery: bool,
    leiden_iterations: int,
    min_count: int,
    refinement_options: StagedRefinementOptions | None = None,
    component_size_threshold: tuple[int, int] | bool = True,
) -> tuple[PNAPixelDataset, GraphStatistics]:
    """Create a pxl file after having created components and removed crossing edges."""
    with TemporaryDirectory(prefix="pixelator-") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        edgelist_with_components, component_statics = find_components(  # type: ignore
            input_edgelist=molecules_lazy_frame,
            multiplet_recovery=multiplet_recovery,
            leiden_iterations=leiden_iterations,
            min_read_count=min_count,
            return_component_statistics=True,
            refinement_options=refinement_options,
            component_size_threshold=component_size_threshold,
        )

        logger.debug("Sorting edgelist by component")
        edgelist_with_components_sorted = edgelist_with_components.sort("component")
        tmp_edgelist_file_sorted = (
            tmp_dir_path / f"{sample_name}.edgelist.sorted.parquet"
        )
        edgelist_with_components_sorted.collect().write_parquet(
            tmp_edgelist_file_sorted
        )
        logger.debug("Counting molecules")

        sums = (
            edgelist_with_components_sorted.select(["uei_count", "read_count"])
            .sum()
            .collect()
        )
        component_statics.molecules_output = sums.item(0, "uei_count")

        logger.debug("Building edgelist from anndata")
        adata = pna_edgelist_to_anndata(edgelist_with_components_sorted, panel=panel)
        call_aggregates(adata)

        component_statics.reads_output = adata.obs["reads_in_component"].sum()

        component_statics.median_reads_per_component = adata.obs[
            "reads_in_component"
        ].median()
        component_statics.median_markers_per_component = adata.obs["n_umi"].median()

        # Add tau_type metrics
        aggregates_mask = adata.obs["tau_type"] != "normal"
        number_of_aggregates = np.sum(aggregates_mask)

        component_statics.aggregate_count = number_of_aggregates
        aggregate_stats = (
            adata[aggregates_mask].obs[["n_edges", "n_umi", "reads_in_component"]].sum()
        )

        component_statics.read_count_in_aggregates = aggregate_stats[
            "reads_in_component"
        ].item()
        component_statics.node_count_in_aggregates = aggregate_stats["n_umi"].item()
        component_statics.edge_count_in_aggregates = aggregate_stats["n_edges"].item()

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

        with PixelFileWriter(path_output_pxl_file) as pxl_file_writer:
            pxl_file_writer.write_metadata(metadata)
            pxl_file_writer.write_adata(adata)
            pxl_file_writer.write_edgelist(tmp_edgelist_file_sorted)

        return PNAPixelDataset.from_pxl_files(path_output_pxl_file), component_statics


def _get_working_edgelist(
    input_edgelist: pl.LazyFrame,
) -> tuple[pl.LazyFrame, pl.DataFrame]:
    """Get a working edgelist and a map from original to working node names."""
    node_map = (
        pl.concat(
            (
                input_edgelist.select("umi1").rename({"umi1": "original_name"}),
                input_edgelist.select("umi2").rename({"umi2": "original_name"}),
            )
        )
        .unique()
        .sort("original_name")
        .with_row_index()
        .rename({"index": "working_name"})
        .collect()
    )

    working_edgelist = input_edgelist.with_columns(
        pl.col("umi1").replace_strict(
            old=node_map["original_name"], new=node_map["working_name"]
        ),
        pl.col("umi2").replace_strict(
            old=node_map["original_name"], new=node_map["working_name"]
        ),
    )
    return working_edgelist, node_map


def _add_post_recovery_stats(edgelist: pl.LazyFrame, component_stats: GraphStatistics):
    """Add post recovery statistics to the component statistics object."""
    post_recovery_stats = edgelist.select(
        [
            pl.col("umi1").n_unique().alias("umi1_count"),
            pl.col("umi2").n_unique().alias("umi2_count"),
            pl.len().alias("edge_count"),
        ]
    ).collect()
    component_stats.node_count_post_recovery = (
        post_recovery_stats["umi1_count"] + post_recovery_stats["umi2_count"]
    )[0]
    component_stats.edge_count_post_recovery = post_recovery_stats["edge_count"][0]

    connected_component_sizes = (
        edgelist.group_by("component")
        .agg(
            pl.col("umi1").n_unique().alias("n_umi1"),
            pl.col("umi2").n_unique().alias("n_umi2"),
        )
        .select(component_size=pl.col("n_umi1") + pl.col("n_umi2"))
        .collect()
    )
    component_stats.component_count_post_recovery = connected_component_sizes.shape[0]
    component_stats.fraction_nodes_in_largest_component_post_recovery = (
        connected_component_sizes["component_size"].max()  # type: ignore
        / connected_component_sizes["component_size"].sum()  # type: ignore
    )
    return component_stats


def _name_components_with_umi_hashes(edgelist):
    comp_umis = (
        edgelist.group_by("component")
        .agg(pl.col("umi1").unique(), pl.col("umi2").unique())
        .collect()
    )
    comp_hashes = dict()
    for comp, umi1, umi2 in comp_umis.rows():
        comp_hashes[comp] = hash_component(set(umi1 + umi2))

    return edgelist.with_columns(pl.col("component").replace_strict(comp_hashes))


def _remove_umi_clashes_and_get_stats(input_edgelist, component_stats):
    raw_stat = input_edgelist.select(["read_count", "uei_count"]).sum().collect()
    component_stats.molecules_input = raw_stat["uei_count"][0]
    component_stats.reads_input = raw_stat["read_count"][0]
    umi1_clashes, umi2_clashes = _find_clashing_umis(input_edgelist)
    no_clash_edgelist = input_edgelist.filter(
        ~pl.col("umi1").is_in(umi1_clashes) & ~pl.col("umi2").is_in(umi2_clashes)
    )
    post_collision_stat = (
        no_clash_edgelist.select(["read_count", "uei_count"]).sum().collect()
    )
    component_stats.molecules_post_umi_collision_removal = post_collision_stat[
        "uei_count"
    ][0]
    component_stats.reads_post_umi_collision_removal = post_collision_stat[
        "read_count"
    ][0]

    return no_clash_edgelist, component_stats


@typing.overload
def find_components(
    input_edgelist: pl.LazyFrame,
    multiplet_recovery: bool,
    leiden_iterations: int = 10,
    min_read_count: int = 2,
    component_size_threshold: tuple[int, int] | bool = True,
    refinement_options: StagedRefinementOptions | None = None,
    return_component_statistics: Literal[True] = True,
    dynamic_lowest_passable_bound: int | None = None,
) -> tuple[pl.LazyFrame, GraphStatistics]: ...


@typing.overload
def find_components(
    input_edgelist: pl.LazyFrame,
    multiplet_recovery: bool,
    leiden_iterations: int = 10,
    min_read_count: int = 2,
    component_size_threshold: tuple[int, int] | bool = True,
    refinement_options: StagedRefinementOptions | None = None,
    return_component_statistics: Literal[False] = False,
    dynamic_lowest_passable_bound: int | None = None,
) -> pl.LazyFrame: ...


def find_components(
    input_edgelist: pl.LazyFrame,
    multiplet_recovery: bool,
    leiden_iterations: int = 10,
    min_read_count: int = 2,
    component_size_threshold: tuple[int, int] | bool = True,
    refinement_options: StagedRefinementOptions | None = None,
    return_component_statistics: bool = False,
    dynamic_lowest_passable_bound: int | None = None,
) -> pl.LazyFrame | tuple[pl.LazyFrame, GraphStatistics]:
    """Retrieve all connected components from an edgelist.

    This function takes as input an edge list in `parquet` format that has
    been generated with `pixelator collapse`. The function filters the
    edge list by count (`min_count`) and then adds a `component` column to the
    edge list with the respective connected components ids obtained from the graph.

    If `multiplet_recovery` is True the edge list is then processed to recover
    components from technical multiplets. The recovery is done using community
    detection to detect and remove problematic edges using the Leiden [1]_ community
    detection algorithm.

    .. [1] Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing
        well-connected communities. Sci Rep 9, 5233 (2019).
        https://doi.org/10.1038/s4q:598-019-41695-z

    :param input_edgelist: The input edgelist
    :param multiplet_recovery: if True run multiplet recovery, otherwise skip it
    :param leiden_iterations: number of Leiden iterations to run
    :param min_read_count: minimum number of supporting reads for an edge to be considered part of the graph
    :param component_size_threshold: if True, filter components by dynamic size thresholds, if a tuple, filter by hard thresholds as (min, max)
    :param refinement_options: options for component refinement
    :param return_component_statistics: if True, return a component statistics object

    :returns: an edgelist with components added to it, and a component statistics object if return_component_statistics is True
    """
    component_stats = GraphStatistics()
    no_clash_edgelist, component_stats = _remove_umi_clashes_and_get_stats(
        input_edgelist, component_stats
    )
    working_edgelist, node_map = _get_working_edgelist(no_clash_edgelist)
    working_edgelist = _filter_edgelist(
        working_edgelist, min_read_count, component_stats
    )
    logger.debug("Labelling connected components")
    umi_component_map, component_stats = _label_connected_components(
        working_edgelist, component_stats
    )

    if multiplet_recovery:
        logger.debug("Initiating multiplet recovery")
        umi_component_map, recover_stats = recover_multiplets(
            working_edgelist,
            umi_component_map,
            leiden_iterations,
            refinement_options=refinement_options,
        )
        component_stats.crossing_edges_removed = recover_stats.crossing_edges_removed
        component_stats.crossing_edges_removed_initial_stage = (
            recover_stats.crossing_edges_removed_in_initial_stage
        )
        component_stats.max_recursion_depth = recover_stats.max_recursion_depth

    edgelist_with_components = make_edgelits_with_component_column(
        working_edgelist, umi_component_map
    )
    if multiplet_recovery:
        component_stats = _add_post_recovery_stats(
            edgelist_with_components, component_stats
        )
        component_stats.component_count_pre_component_size_filtering = (
            component_stats.component_count_post_recovery
        )
    else:
        component_stats.component_count_pre_component_size_filtering = (
            component_stats.component_count_pre_recovery
        )

    # Restore original umi names
    edgelist_with_components = edgelist_with_components.with_columns(
        pl.col("umi1").replace_strict(
            old=node_map["working_name"], new=node_map["original_name"]
        ),
        pl.col("umi2").replace_strict(
            old=node_map["working_name"], new=node_map["original_name"]
        ),
    )

    edgelist_with_components = _filter_connected_components_by_size(
        edgelist_with_components,
        component_size_threshold,
        component_stats,
        dynamic_lowest_passable_bound,
    )
    edgelist_with_components = _name_components_with_umi_hashes(
        edgelist_with_components
    )
    if return_component_statistics:
        return edgelist_with_components, component_stats
    return edgelist_with_components


def _filter_connected_components_by_size(
    edgelist: pl.LazyFrame,
    component_size_threshold,
    component_stats: GraphStatistics,
    dynamic_lowest_passable_bound=None,
):
    """Filter connected components by size and get statistics."""
    component_sizes = (
        edgelist.group_by("component")
        .agg(
            pl.col("umi1").n_unique().alias("n_umi1"),
            pl.col("umi2").n_unique().alias("n_umi2"),
        )
        .select(pl.col("component"), n_umi=pl.col("n_umi1") + pl.col("n_umi2"))
        .collect()
    )
    unique, counts = np.unique(
        component_sizes["n_umi"].cast(pl.Int32), return_counts=True
    )
    component_stats.pre_filtering_component_sizes = dict(zip(unique, counts))
    if isinstance(component_size_threshold, bool) and component_size_threshold:
        logger.debug("Filtering components by dynamic size thresholds")
        try:
            passing_components, min_size = filter_components_by_size_dynamic(
                component_sizes,
                lowest_passable_bound=dynamic_lowest_passable_bound,
            )
            component_stats.component_size_min_filtering_threshold = min_size
        except Exception as e:
            # This is a hack since the exception is not properly
            # by passed the spline interpolation package.
            if str(type(e)) == "<class 'dfitpack.error'>":
                msg = (
                    "Could not find component size filters, this probably means that no components (i.e. cells) were formed. "
                    "If you are running on the command line you can try to use --component-size-max-threshold and "
                    " --component-size-min-threshold to set hard thresholds, but most likely the problem is with the input data."
                )
                raise ConnectedComponentException(msg)
            else:
                raise e
    else:
        if component_size_threshold is False:
            min_size = 0
            max_size = np.iinfo(np.uint64).max
        else:
            min_size, max_size = component_size_threshold
        component_stats.component_size_max_filtering_threshold = max_size
        component_stats.component_size_min_filtering_threshold = min_size
        logger.debug(
            "Filtering components by hard size thresholds, min: %s, max: %s",
            min_size,
            max_size,
        )
        passing_components = filter_components_by_size_hard_thresholds(
            component_sizes, min_size, max_size
        )

    if len(passing_components) == 0:
        msg = (
            "No connected components found in the graph. Likely they were all filtered away for being to small. "
            "This indicates some serious issue with the data. Will not continue with the rest of the computations."
        )
        raise ConnectedComponentException(msg)

    component_stats.component_count_post_component_size_filtering = len(
        passing_components
    )

    edgelist = edgelist.filter(pl.col("component").is_in(passing_components))

    return edgelist

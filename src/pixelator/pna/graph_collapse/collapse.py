"""Graph-collapse: rename likely UMI sequencing errors on low-degree nodes, per component.

A UMI sequencing error is unlikely to recur, so an erroneous UMI typically shows up as a
leaf in its component's graph. The Rust binding ``pixelator_core.graph_collapse``
searches every node whose degree is at most ``max_degree`` (leaves by default) and finds,
for each, the nearest UMI on the same side within a Hamming-distance cutoff,
and returns ``(component, original, corrected)`` rows. It does not know markers. Three
SQL steps finish the job: drop pairs whose markers differ, rename the remaining
original UMIs, and merge edges that are identical after that rename.

The steps of a batch pass data through temp objects on the DuckDB connection, so
their order in ``_collapse_batch`` matters:

- ``select_batch`` creates the view ``_batch_edges`` and the table ``_batch_components``.
- ``store_renames`` creates ``_renames`` from the binding's output.
- ``drop_renames_with_different_markers`` narrows ``_renames`` and creates ``_umi_marker``.
- ``collapsed_sequences`` reads ``_renames`` and ``_umi_marker``.
- ``write_collapsed_edges`` reads ``_batch_edges`` and ``_renames``.

The batching and input access live in ``batches``, and the binding and SQL steps in
``renames``.

The input edgelist is processed in batches of whole components and written to a new
pxl file as it goes, so memory follows the batch size rather than the file size. Every
step works within one component: a rename only applies to the component it was found
in, and the marker check only looks at markers recorded in that component.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
from anndata import AnnData

from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.graph_collapse.batches import (
    SOURCE,
    attach_input,
    component_sizes,
    detach_input,
    plan_component_batches,
    select_batch,
)
from pixelator.pna.graph_collapse.constants import DEFAULT_BATCH_EDGES
from pixelator.pna.graph_collapse.renames import (
    CollapseParams,
    collapsed_sequences,
    drop_renames_with_different_markers,
    fetch_batch_edges,
    search_renames,
    store_renames,
    write_collapsed_edges,
)
from pixelator.pna.pixeldataset import read
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile

logger = logging.getLogger(__name__)


def _merged_counts(first: dict[int, int], second: dict[int, int]) -> dict[int, int]:
    """Add two ``{value: count}`` distributions, keeping the values in ascending order."""
    merged = Counter(first)
    merged.update(second)
    return dict(sorted(merged.items()))


@dataclass
class GraphCollapseResult:
    """Counts of a collapse, for one batch or for the whole run.

    ``component_count``, ``nodes_considered``, ``targets_found`` and the two
    distributions come from the binding. ``targets_found`` and the
    distributions describe the matches the search kept, before pairs with
    different markers are dropped, so ``nodes_collapsed`` can be smaller.
    A new instance is an empty result, which ``merge`` adds batches to.
    """

    component_count: int = 0
    nodes_considered: int = 0
    targets_found: int = 0
    bfs_distance_distribution: dict[int, int] = field(default_factory=dict)
    hamming_distance_distribution: dict[int, int] = field(default_factory=dict)
    nodes_collapsed: int = 0
    collapsed_sequences: list[dict] = field(default_factory=list)

    def merge(self, other: GraphCollapseResult) -> None:
        """Add the counts of ``other`` to this result, and append its sequences."""
        self.component_count += other.component_count
        self.nodes_considered += other.nodes_considered
        self.targets_found += other.targets_found
        self.bfs_distance_distribution = _merged_counts(
            self.bfs_distance_distribution, other.bfs_distance_distribution
        )
        self.hamming_distance_distribution = _merged_counts(
            self.hamming_distance_distribution, other.hamming_distance_distribution
        )
        self.nodes_collapsed += other.nodes_collapsed
        self.collapsed_sequences.extend(other.collapsed_sequences)


def _collapse_batch(
    con: duckdb.DuckDBPyConnection,
    components: list[str],
    params: CollapseParams,
    *,
    umi1_length: int,
    umi2_length: int,
    report_collapsed_sequences: bool,
) -> GraphCollapseResult:
    """Collapse one batch of components and append its edges to the output.

    Args:
        con: Connection with the input attached as ``SOURCE``.
        components: Consecutive components, in component order.
        params: The binding settings.
        umi1_length: Nucleotide length used to decode a renamed UMI seen as umi1.
        umi2_length: Nucleotide length used to decode a renamed UMI seen only as umi2.
        report_collapsed_sequences: Decode each kept rename into nucleotide sequences.

    Returns:
        The batch's counts and, when requested, collapsed-sequence rows.

    """
    select_batch(con, f"{SOURCE}.edgelist", components)
    renames, stats = search_renames(fetch_batch_edges(con), params)
    store_renames(con, renames)
    drop_renames_with_different_markers(con)
    # The binding maps each original UMI to one corrected UMI per
    # component, so every kept row is a distinct collapsed node.
    nodes_collapsed = con.execute("SELECT COUNT(*) FROM _renames").fetchone()[0]  # type: ignore[index]
    sequences = (
        collapsed_sequences(con, umi1_length, umi2_length)
        if report_collapsed_sequences
        else []
    )
    write_collapsed_edges(con)
    return GraphCollapseResult(
        component_count=stats["total_components"],
        nodes_considered=stats["nodes_considered"],
        targets_found=stats["targets_found"],
        bfs_distance_distribution=stats["bfs_distance_distribution"],
        hamming_distance_distribution=stats["hamming_distance_distribution"],
        nodes_collapsed=nodes_collapsed,
        collapsed_sequences=sequences,
    )


def _rebuild_adata(con: duckdb.DuckDBPyConnection, panel: PNAAntibodyPanel) -> AnnData:
    """Build the AnnData from the collapsed edgelist, sorted by component."""
    logger.info("Building AnnData from the collapsed edgelist")
    adata = pna_edgelist_to_anndata(con, panel)
    return adata[adata.obs_names.sort_values(), :]


def run_graph_collapse(
    input_pxl_file: PxlFile,
    output_path: Path,
    *,
    n_threads: int,
    bfs_distance: int,
    hamming_threshold: int,
    umi1_length: int,
    umi2_length: int,
    max_degree: int = 1,
    report_collapsed_sequences: bool = False,
    memory_limit: int | None = None,
    temp_directory: str | Path | None = None,
    batch_edges: int = DEFAULT_BATCH_EDGES,
) -> GraphCollapseResult:
    """Collapse the edgelist of ``input_pxl_file`` into a new pxl file at ``output_path``.

    The output starts empty, with the input's metadata and panel. The input's
    components are processed in batches of about ``batch_edges`` edges, in
    component order: the binding finds the renames, pairs with different
    markers are dropped, and the renamed and merged edges are appended to the
    output edgelist. Finally the output's AnnData is rebuilt from the collapsed
    edgelist. Tables other than these (e.g. layouts) are not carried over, as
    the collapse makes them stale.

    Args:
        input_pxl_file: The pxl file to collapse. It is not modified.
        output_path: Where the collapsed pxl file is written. An existing file
            is replaced.
        n_threads: Passed to the binding as ``n_threads``, and DuckDB's thread count.
        bfs_distance: Passed to the binding as ``bfs_threshold``.
        hamming_threshold: Passed to the binding as ``hamming_threshold``.
        umi1_length: Nucleotide length used to decode a renamed UMI seen as umi1.
        umi2_length: Nucleotide length used to decode a renamed UMI seen only as umi2.
        max_degree: Passed to the binding as ``max_degree``. Nodes with a higher
            degree are not searched. The default, 1, investigates only leaves.
        report_collapsed_sequences: Decode each kept rename into nucleotide
            sequences. Off unless the CLI is writing the collapsed-sequence TSV.
        memory_limit: Soft memory limit in bytes for DuckDB. When set, DuckDB
            spills to disk once the limit is reached. The native search is not
            covered by it; its memory follows ``batch_edges``.
        temp_directory: DuckDB spill directory. Defaults to
            ``PIXELATOR_DUCKDB_TEMP_DIR`` or ``/tmp``.
        batch_edges: Number of edges a batch of components is filled to.

    Returns:
        Counts and, when requested, the collapsed-sequence rows.

    """
    # The panel comes from the file only.
    input_dataset = read(input_pxl_file.path)
    panel = PNAAntibodyPanel.from_pxl_dataset(input_dataset)

    params = CollapseParams(
        n_threads=n_threads,
        bfs_distance=bfs_distance,
        hamming_threshold=hamming_threshold,
        max_degree=max_degree,
    )
    result = GraphCollapseResult()

    with PixelFileWriter(
        output_path,
        exits_ok=True,
        temp_dir=temp_directory,
        memory_limit=memory_limit,
        threads=n_threads,
    ) as writer:
        con = attach_input(writer, input_pxl_file)
        writer.write_metadata(input_pxl_file.metadata())
        writer.write_panel(panel)
        sizes = component_sizes(con)
        batches = plan_component_batches(sizes, batch_edges)
        logger.info(
            "Collapsing %d components in %d batches",
            len(sizes),
            len(batches),
        )

        for index, components in enumerate(batches, start=1):
            result.merge(
                _collapse_batch(
                    con,
                    components,
                    params,
                    umi1_length=umi1_length,
                    umi2_length=umi2_length,
                    report_collapsed_sequences=report_collapsed_sequences,
                )
            )
            logger.info(
                "Batch %d/%d done: %d components processed",
                index,
                len(batches),
                result.component_count,
            )

        detach_input(con)
        writer.write_adata(_rebuild_adata(con, panel))

    result.collapsed_sequences.sort(
        key=lambda record: (record["component"], record["original_sequence"])
    )
    return result

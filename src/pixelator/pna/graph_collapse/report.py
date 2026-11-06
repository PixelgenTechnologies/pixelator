"""Reporting metadata for the graph-collapse command.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import pydantic

from pixelator.pna.report.models import SampleReport


class GraphCollapseSampleReport(SampleReport):
    """Model for report data returned by the graph-collapse stage."""

    report_type: str = "graph-collapse"

    component_count: int = pydantic.Field(
        ..., description="The number of components processed."
    )

    umis_input: int = pydantic.Field(
        ..., description="The number of distinct UMIs in the input edgelist."
    )
    umis_output: int = pydantic.Field(
        ..., description="The number of distinct UMIs in the output edgelist."
    )

    reads_input: int = pydantic.Field(
        ..., description="The number of reads in the input edgelist."
    )
    reads_output: int = pydantic.Field(
        ...,
        description="The number of reads in the output edgelist. Collapsing nodes never "
        "discards reads, only re-attributes them, so this should equal reads_input.",
    )

    nodes_considered: int = pydantic.Field(
        ...,
        description="The number of nodes searched for a collapse target, across components. "
        "These are the nodes whose degree is at most --max-degree, so leaves by default.",
    )
    targets_found: int = pydantic.Field(
        ...,
        description="The number of searched nodes for which a collapse target was found. A match "
        "dropped because it would close a loop is not counted. Pairs whose markers differ are "
        "counted here and dropped later, so this can exceed nodes_collapsed.",
    )
    bfs_distance_distribution: dict[int, int] = pydantic.Field(
        ...,
        description="Number of targets found by graph distance, in hops, from the node to its "
        "target. Only distances that occurred are listed, in ascending order. The distance is "
        "that of the match as found, not of the end of a chain of matches.",
    )
    hamming_distance_distribution: dict[int, int] = pydantic.Field(
        ...,
        description="Number of targets found by Hamming distance between the node's UMI and its "
        "target's UMI. Only distances that occurred are listed, in ascending order. Counted over "
        "the same targets as bfs_distance_distribution.",
    )
    nodes_collapsed: int = pydantic.Field(
        ...,
        description="The number of nodes that were renamed to their collapse target. Pairs "
        "whose markers differ are not renamed.",
    )
    collapsed_sequences_file: str | None = pydantic.Field(
        default=None,
        description="Path to a TSV file listing, for every collapsed node, its component, "
        "original UMI sequence, and corrected UMI sequence. Only written when "
        "--report-collapsed-sequences is passed.",
    )

    @pydantic.computed_field(  # type: ignore
        description="The fraction of all nodes in the input graph that were collapsed "
        "(nodes_collapsed over umis_input).",
        return_type=float,
    )
    @property
    def nodes_collapsed_fraction(self) -> float:
        """Return the fraction of all input nodes that were collapsed."""
        if self.umis_input == 0:
            return 0.0
        return self.nodes_collapsed / self.umis_input

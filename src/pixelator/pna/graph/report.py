"""Model for report data returned by the single-cell-pna graph stage.

Copyright © 2024 Pixelgen Technologies AB.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pydantic

from pixelator.pna.report.models.base import SampleReport


@dataclass
class GraphStatistics:
    """Graph statistics for a sample."""

    # Raw input from collapse to graph
    molecules_input: int = 0
    reads_input: int = 0

    # Molecules and reads after removing UMI collisions
    molecules_post_umi_collision_removal: int = 0
    reads_post_umi_collision_removal: int = 0

    # Input into to initial graph after read_count filtering
    molecules_post_read_count_filtering: int = 0
    reads_post_read_count_filtering: int = 0

    # Final output after recovery and component size filtering
    molecules_output: int = 0
    reads_output: int = 0

    component_count_pre_recovery: int = 0
    component_count_post_recovery: int = 0

    component_count_pre_component_size_filtering: int = 0
    component_count_post_component_size_filtering: int = 0

    fraction_nodes_in_largest_component_pre_recovery: float = 0.0
    fraction_nodes_in_largest_component_post_recovery: float = 0.0
    component_size_min_filtering_threshold: int | None = 0
    component_size_max_filtering_threshold: int | None = None
    crossing_edges_removed: int = 0
    crossing_edges_removed_initial_stage: int = 0
    max_recursion_depth: int = 0

    node_count_pre_recovery: int = 0
    edge_count_pre_recovery: int = 0

    node_count_post_recovery: int = 0
    edge_count_post_recovery: int = 0

    pre_filtering_component_sizes: dict[int, int] | None = None

    median_reads_per_component: float = 0

    median_markers_per_component: float = 0

    aggregate_count: int = 0
    read_count_in_aggregates: int = 0
    edge_count_in_aggregates: int = 0
    node_count_in_aggregates: int = 0

    def to_dict(self):
        """Convert the object to a dictionary."""
        return asdict(self)


class GraphSampleReport(SampleReport):
    """Model for report data returned by the graph stage."""

    report_type: str = "graph"

    molecules_input: int = pydantic.Field(
        ..., description="The number of molecules in the input edgelist."
    )

    reads_input: int = pydantic.Field(
        ..., description="The number of reads in the input edgelist."
    )

    molecules_post_umi_collision_removal: int = pydantic.Field(
        ..., description="The number of molecules after removing UMI collisions."
    )

    reads_post_umi_collision_removal: int = pydantic.Field(
        ..., description="The number of reads after removing UMI collisions."
    )

    molecules_post_read_count_filtering: int = pydantic.Field(
        ..., description="The number of molecules after read count filtering."
    )

    reads_post_read_count_filtering: int = pydantic.Field(
        ..., description="The number of reads after read count filtering."
    )

    molecules_output: int = pydantic.Field(
        ..., description="The number of molecules in the output graph."
    )

    reads_output: int = pydantic.Field(
        ..., description="The number of reads in the output graph."
    )

    component_count_pre_recovery: int = pydantic.Field(
        ...,
        description="The total number of components in the graph before mega-cluster recovery.",
    )

    component_count_post_recovery: int = pydantic.Field(
        ...,
        description="The total number of components in the graph after mega-cluster recovery.",
    )

    component_count_pre_component_size_filtering: int = pydantic.Field(
        ...,
        description="The total number of components in the graph before filtering for size.",
    )

    component_count_post_component_size_filtering: int = pydantic.Field(
        ...,
        description="The total number of components in the graph after filtering for size.",
    )

    fraction_nodes_in_largest_component_pre_recovery: float = pydantic.Field(
        ...,
        ge=0,
        le=1,
        description="The fraction of all molecules that are located in the largest component before multiplet recovery.",
    )

    fraction_nodes_in_largest_component_post_recovery: float = pydantic.Field(
        ...,
        ge=0,
        le=1,
        description="The fraction of all molecules that are located in the largest component after multiplet recovery.",
    )

    crossing_edges_removed: int = pydantic.Field(
        ...,
        description="The total number of crossing edges removed. Will be 0 if multiplet recovery was not enabled.",
    )

    crossing_edges_removed_initial_stage: int = pydantic.Field(
        ...,
        description="The number of crossing edges removed in the initial step. Will be 0 if multiplet recovery was not enabled.",
    )

    component_size_min_filtering_threshold: int | None = pydantic.Field(
        ..., description="The minimum component size threshold used for filtering."
    )
    component_size_max_filtering_threshold: int | None = pydantic.Field(
        ..., description="The max component size threshold used for filtering."
    )

    node_count_pre_recovery: int = pydantic.Field(
        ..., description="The number of nodes before multiplet recovery."
    )
    edge_count_pre_recovery: int = pydantic.Field(
        ..., description="The number of edges before multiplet recovery."
    )

    node_count_post_recovery: int = pydantic.Field(
        ..., description="The number of nodes after multiplet recovery."
    )
    edge_count_post_recovery: int = pydantic.Field(
        ..., description="The number of edges after multiplet recovery."
    )

    pre_filtering_component_sizes: dict[int, int] = pydantic.Field(
        ..., description="The sizes of all components before filtering."
    )

    median_reads_per_component: float = pydantic.Field(
        ..., description="The median number of reads per cell component."
    )

    median_markers_per_component: float = pydantic.Field(
        ..., description="The median number of markers per cell component."
    )

    aggregate_count: int = pydantic.Field(
        ..., description="The number of aggregates in the graph."
    )

    read_count_in_aggregates: int = pydantic.Field(
        ..., description="The number of reads in aggregates."
    )

    edge_count_in_aggregates: int = pydantic.Field(
        ..., description="The number of edges in aggregates."
    )

    @pydantic.computed_field(  # type: ignore
        description="The fraction of components discarded by filtering.",
        return_type=float,
    )
    @property
    def fraction_of_discarded_components(self) -> float:
        """Calculate the fraction of discarded components."""
        if self.component_count_pre_component_size_filtering == 0:
            return 0.0
        return (
            1
            - self.component_count_post_component_size_filtering
            / self.component_count_pre_component_size_filtering
        )

    @pydantic.computed_field(  # type: ignore
        description="The fraction of components marked as aggregates by the tau_type outlier detection. ",
        return_type=float,
    )  # type: ignore
    @property
    def fraction_of_aggregate_components(self) -> float:
        """Calculate the fraction of aggregate components."""
        if self.component_count_post_component_size_filtering == 0:
            return 0.0
        return self.aggregate_count / self.component_count_post_component_size_filtering

    @pydantic.computed_field(  # type: ignore
        description="The redundancy of edge detection. The saturation is calculated as: `1 - # graph edges / # graph reads`",
        return_type=float,
    )
    @property
    def edge_saturation(self):
        """Return the edge saturation."""
        return 1 - self.edge_count_post_recovery / self.reads_output

    @pydantic.computed_field(  # type: ignore
        description="The redundancy of node detection. The saturation is calculated as: `1 - # graph node / # graph reads`",
        return_type=float,
    )
    @property
    def node_saturation(self):
        """Return the node saturation."""
        return 1 - self.node_count_post_recovery / self.reads_output

    @pydantic.computed_field(  # type: ignore
        description="The number of reads discarded due to cell size filtering and multiplet recovery.",
        return_type=int,
    )
    @property
    def discarded_reads(self) -> int:
        """Return the total number of discarded reads in the graph stage."""
        return self.reads_input - self.reads_output

    @pydantic.computed_field(  # type: ignore
        description="The number of molecules discarded due to cell size filtering and multiplet recovery.",
        return_type=int,
    )
    @property
    def discarded_molecules(self) -> int:
        """Return the total number of discarded reads in the graph stage."""
        return self.molecules_input - self.molecules_output

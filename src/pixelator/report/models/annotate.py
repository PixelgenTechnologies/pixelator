"""Copyright © 2023 Pixelgen Technologies AB."""

from __future__ import annotations

from typing import Optional

import pydantic

from pixelator.report.models.base import SampleReport
from pixelator.report.models.summary_statistics import SummaryStatistics


class AnnotateSampleReport(SampleReport):
    """Model for report data returned by the annotate stage."""

    components_modularity: float
    fraction_molecules_in_largest_component: float
    fraction_pixels_in_largest_component: float

    # ------------------------------------------------------------------------------- #
    #   Annotate metrics
    # ------------------------------------------------------------------------------- #
    input_cell_count: int = pydantic.Field(
        ...,
        description="The total number of cell components in the input before filtering.",
    )

    input_read_count: int = pydantic.Field(
        ...,
        description="The total number of reads in the input before filtering.",
    )

    # cells_filtered: int
    cell_count: int = pydantic.Field(
        ..., description="The total number of cells after filtering for component size."
    )

    marker_count: int = pydantic.Field(
        ...,
        description="The total number of detected antibodies after filtering for component size.",
    )

    total_marker_count: int = pydantic.Field(
        ...,
        description="The total number of antibodies defined by the panel.",
    )

    molecule_count: int = pydantic.Field(
        ...,
        description="The total number of unique molecules in cell or aggregate components.",
    )

    a_pixel_count: int = pydantic.Field(
        ...,
        description="The number of unique A-pixels in the graph.",
    )

    b_pixel_count: int = pydantic.Field(
        ...,
        description="The number of unique B-pixels in the graph.",
    )

    @pydantic.computed_field(
        return_type=float,
        description="The total number of unique pixels in the graph.",
    )
    def pixel_count(self) -> int:  # noqa: D102
        return self.a_pixel_count + self.b_pixel_count

    read_count: int = pydantic.Field(
        ...,
        description="The total number of reads for all unique molecules in cell or aggregate components.",
    )

    molecule_count_per_cell_stats: SummaryStatistics = pydantic.Field(
        ...,
        description="Summary statistics for the number of molecules per cell component.",
    )

    read_count_per_cell_stats: SummaryStatistics = pydantic.Field(
        ...,
        description="Summary statistics for the number of reads per cell component.",
    )

    a_pixel_count_per_cell_stats: SummaryStatistics = pydantic.Field(
        ...,
        description="Summary statistics for the number of A-pixels per cell component.",
    )

    b_pixel_count_per_cell_stats: SummaryStatistics = pydantic.Field(
        ...,
        description="Summary statistics for the number of B-pixels per cell component.",
    )

    marker_count_per_cell_stats: SummaryStatistics = pydantic.Field(
        ...,
        description="Summary statistics for the number of markers per cell component.",
    )

    a_pixel_b_pixel_ratio_per_cell_stats: SummaryStatistics = pydantic.Field(
        ...,
        description="Summary statistics for the number of B-pixels per A-pixel in cell components.",
    )

    molecule_count_per_a_pixel_per_cell_stats: SummaryStatistics = pydantic.Field(
        ...,
        description="Summary statistics for the number of molecules per A-pixel in cell components.",
    )

    b_pixel_count_per_a_pixel_per_cell_stats: SummaryStatistics = pydantic.Field(
        ...,
        description="Summary statistics for the number of B-pixels per A-pixel in cell components.",
    )

    # ------------------------------------------------------------------------------- #
    #   Aggregate filtering
    # ------------------------------------------------------------------------------- #

    #: The number of aggregates called
    aggregate_count: int | None = pydantic.Field(
        description="The number of components identified as aggregates and removed from results."
    )

    #: The total number of molecules in aggregates
    molecules_in_aggregates_count: Optional[int]

    #: The total number of reads for unique molecules in aggregates
    reads_in_aggregates_count: Optional[int]

    @pydantic.computed_field(return_type=float)
    def fraction_reads_in_aggregates(self) -> float | None:
        """Return the fraction of molecules (edges) in the aggregate.

        Returns None if no aggregate recovery was disabled during analysis.
        """
        if self.reads_in_aggregates_count is not None and self.read_count > 0:
            return self.reads_in_aggregates_count / self.read_count
        return None

    @pydantic.computed_field(return_type=float)
    def fraction_molecules_in_aggregates(self) -> float | None:
        """Return the fraction of molecules (edges) in the aggregate.

        Returns None if no aggregate recovery was disabled during analysis.
        """
        if self.molecules_in_aggregates_count is not None and self.read_count > 0:
            return self.molecules_in_aggregates_count / self.molecule_count
        return None

    # ------------------------------------------------------------------------------- #
    #   Component size filtering
    # ------------------------------------------------------------------------------- #

    min_size_threshold: Optional[int]
    max_size_threshold: Optional[int]
    doublet_size_threshold: Optional[int]

    size_filter_fail_cell_count: int
    size_filter_fail_molecule_count: int
    size_filter_fail_read_count: int

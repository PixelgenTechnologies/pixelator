"""Model for report data returned by the annotate stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import textwrap
from typing import Optional

import pydantic

from pixelator.common.report.models.summary_statistics import SummaryStatistics
from pixelator.mpx.report.models.base import SampleReport


class AnnotateSampleReport(SampleReport):
    """Model for report data returned by the annotate stage."""

    fraction_molecules_in_largest_component: float = pydantic.Field(
        description="The fraction of molecules in the largest component.",
    )

    fraction_pixels_in_largest_component: float = pydantic.Field(
        description="The fraction of pixels (A and B pixels) in the largest component.",
    )

    fraction_potential_doublets: float | None = pydantic.Field(
        description=(
            "The fraction of components that appear to consist of multiple "
            "parts by the community detection algorithm."
        ),
    )

    n_edges_to_split_potential_doublets: int | None = pydantic.Field(
        description=(
            "The total number of edges that need to be removed to split the "
            "potential doublets into their sub-communities."
        ),
    )

    # ------------------------------------------------------------------------------- #
    #   Annotate metrics
    # ------------------------------------------------------------------------------- #
    input_cell_count: int = pydantic.Field(
        description="The total number of cell components in the input before filtering.",
    )

    input_read_count: int = pydantic.Field(
        description="The total number of reads in the input before filtering.",
    )

    # cells_filtered: int
    cell_count: int = pydantic.Field(
        description="The total number of cells after filtering for component size."
    )

    marker_count: int = pydantic.Field(
        description="The total number of detected antibodies after filtering for component size.",
    )

    total_marker_count: int = pydantic.Field(
        description="The total number of antibodies defined by the panel.",
    )

    molecule_count: int = pydantic.Field(
        description="The total number of unique molecules in cell or aggregate components.",
    )

    a_pixel_count: int = pydantic.Field(
        description="The number of unique A-pixels in the graph.",
    )

    b_pixel_count: int = pydantic.Field(
        description="The number of unique B-pixels in the graph.",
    )

    @pydantic.computed_field(
        return_type=int,
        description="The total number of unique pixels in the graph.",
    )
    def pixel_count(self) -> int:  # noqa: D102
        return self.a_pixel_count + self.b_pixel_count

    read_count: int = pydantic.Field(
        description="The total number of reads for all unique molecules in cell or aggregate components.",
    )

    molecule_count_per_cell_stats: SummaryStatistics = pydantic.Field(
        description="Summary statistics for the number of molecules per cell component.",
    )

    read_count_per_cell_stats: SummaryStatistics = pydantic.Field(
        description="Summary statistics for the number of reads per cell component.",
    )

    a_pixel_count_per_cell_stats: SummaryStatistics = pydantic.Field(
        description="Summary statistics for the number of A-pixels per cell component.",
    )

    b_pixel_count_per_cell_stats: SummaryStatistics = pydantic.Field(
        description="Summary statistics for the number of B-pixels per cell component.",
    )

    marker_count_per_cell_stats: SummaryStatistics = pydantic.Field(
        description="Summary statistics for the number of markers per cell component.",
    )

    a_pixel_b_pixel_ratio_per_cell_stats: SummaryStatistics = pydantic.Field(
        description="Summary statistics for the ratio of A-pixels to B-pixel in cell components.",
    )

    molecule_count_per_a_pixel_per_cell_stats: SummaryStatistics = pydantic.Field(
        description="Summary statistics for the number of molecules per A-pixel in cell components.",
    )

    a_pixel_count_per_b_pixel_per_cell_stats: SummaryStatistics = pydantic.Field(
        description="Summary statistics for the mean number of A-pixels per B-pixel in cell components.",
    )

    b_pixel_count_per_a_pixel_per_cell_stats: SummaryStatistics = pydantic.Field(
        description="Summary statistics for the mean number of B-pixels per A-pixel in cell components.",
    )

    # ------------------------------------------------------------------------------- #
    #   Aggregate filtering
    # ------------------------------------------------------------------------------- #

    aggregate_count: int | None = pydantic.Field(
        description="The number of components identified as aggregates."
    )

    molecules_in_aggregates_count: int | None = pydantic.Field(
        description="The total number of unique molecules in aggregates."
    )

    reads_in_aggregates_count: int | None = pydantic.Field(
        description="The total number of reads for unique molecules in aggregates."
    )

    @pydantic.computed_field(
        return_type=float,
        description=textwrap.dedent(
            """The fraction of components identified as aggregates or
            None if aggregate recovery was disabled.
            """
        ),
    )
    def fraction_aggregate_components(self) -> float | None:
        """Return the fraction of the total cell count that is marked as aggregates.

        Returns None if aggregate recovery was disabled during analysis.
        """
        if self.aggregate_count is not None and self.cell_count > 0:
            return self.aggregate_count / self.cell_count
        return None

    @pydantic.computed_field(
        return_type=float,
        description=textwrap.dedent(
            """The fraction of reads in the aggregate outliers or
            None if aggregate recovery was disabled.
            """
        ),
    )
    def fraction_reads_in_aggregates(self) -> float | None:
        """Return the fraction of reads in the aggregate outliers.

        Returns None if no aggregate recovery was disabled during analysis.
        """
        if self.reads_in_aggregates_count is not None and self.read_count > 0:
            return self.reads_in_aggregates_count / self.read_count
        return None

    @pydantic.computed_field(
        return_type=float,
        description=textwrap.dedent(
            """The fraction of molecules in the aggregate outliers or
            None if aggregate recovery was disabled.
            """
        ),
    )
    def fraction_molecules_in_aggregates(self) -> float | None:
        """Return the fraction of molecules (edges) in the aggregate outliers.

        Returns None if no aggregate recovery was disabled during analysis.
        """
        if self.molecules_in_aggregates_count is not None and self.read_count > 0:
            return self.molecules_in_aggregates_count / self.molecule_count
        return None

    # ------------------------------------------------------------------------------- #
    #   Component size filtering
    # ------------------------------------------------------------------------------- #

    min_size_threshold: Optional[int] = pydantic.Field(
        description="The minimum size threshold used for filtering components."
    )

    max_size_threshold: Optional[int] = pydantic.Field(
        description="The maximum size threshold used for filtering components."
    )

    doublet_size_threshold: Optional[int] = pydantic.Field(
        description="The doublet size threshold used for filtering components."
    )

    size_filter_fail_cell_count: int = pydantic.Field(
        description="The number of cells that do NOT pass the component size filters."
    )

    size_filter_fail_molecule_count: int = pydantic.Field(
        description="The number of molecules in components that do NOT pass the component size filters."
    )

    size_filter_fail_read_count: int = pydantic.Field(
        description="The number of reads in components that do NOT pass the component size filters."
    )

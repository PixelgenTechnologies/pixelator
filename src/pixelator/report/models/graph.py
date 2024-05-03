"""Model for report data returned by the single-cell graph stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import pydantic

from pixelator.report.models.base import SampleReport
from pixelator.report.models.summary_statistics import SummaryStatistics


class GraphSampleReport(SampleReport):
    """Model for report data returned by the graph stage."""

    component_count: int = pydantic.Field(
        ...,
        description="The total number of components in the graph.",
    )

    molecule_count: int = pydantic.Field(
        ...,
        description="The total number of unique molecules in the graph.",
    )

    read_count: int = pydantic.Field(
        ...,
        description="The total number of reads in the graph.",
    )

    marker_count: int = pydantic.Field(
        ...,
        description="The total number of unique markers in the graph.",
    )

    read_count_per_molecule_stats: SummaryStatistics = pydantic.Field(
        ...,
        description="Summary statistics of the reads per component.",
    )

    a_pixel_count: int = pydantic.Field(
        ...,
        description="The number of unique A-pixels in the graph.",
    )

    b_pixel_count: int = pydantic.Field(
        ...,
        description="The number of unique B-pixels in the graph.",
    )

    components_modularity: float = pydantic.Field(
        ..., description="The modularity of the components."
    )

    fraction_molecules_in_largest_component: float = pydantic.Field(
        ...,
        ge=0,
        le=1,
        description="The fraction of all molecules that are located in the largest component.",
    )

    fraction_pixels_in_largest_component: float = pydantic.Field(
        ...,
        ge=0,
        le=1,
        description="The fraction of all pixels that are located in the largest component.",
    )

    @pydantic.computed_field(
        return_type=float,
        description="The ratio of the total number of A-pixels and the total number of B-pixels in the graph.",
    )
    def a_pixel_b_pixel_ratio(self):  # noqa: D102
        return self.a_pixel_count / self.b_pixel_count

    @pydantic.computed_field(  # type: ignore
        return_type=int,
        description="The total number of unique pixels in the graph.",
    )
    @property
    def pixel_count(self) -> int:  # noqa: D102
        return self.a_pixel_count + self.b_pixel_count

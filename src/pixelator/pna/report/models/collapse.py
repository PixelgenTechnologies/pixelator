"""Model for report data returned by the single-cell collapse stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import pydantic

from pixelator.common.report.models.summary_statistics import SummaryStatistics
from pixelator.pna.report.models.base import SampleReport


class CollapseSampleReport(SampleReport):
    """Model for data returned by the demux stage."""

    input_read_count: int = pydantic.Field(
        ...,
        description="The total number of input reads before collapsing.",
    )

    output_read_count: int = pydantic.Field(
        ...,
        description="The total number of reads from unique molecules after collapsing.",
    )

    molecule_count: int = pydantic.Field(
        ...,
        description="The total number of unique molecules in the graph.",
    )

    collapsed_molecule_count_stats: SummaryStatistics = pydantic.Field(
        ...,
        description="Summary statistics for the number of unique molecules that are collapsed into a single molecule.",
    )

    @pydantic.computed_field(  # type: ignore
        return_type=float,
        description="The fraction of reads (after preprocessing) that are PCR duplicates.",
    )
    @property
    def fraction_duplicate_reads(self) -> float:  # noqa: D102
        return 1.0 - (self.molecule_count / self.output_read_count)

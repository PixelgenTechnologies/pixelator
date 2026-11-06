"""Reporting metadata for the aggregate command.

Copyright © 2026 Pixelgen Technologies AB.
"""

import pydantic

from pixelator.pna.report.models import SampleReport


class AggregateSampleReport(SampleReport):
    """Model for report data returned by the aggregate stage."""

    report_type: str = "aggregate"

    input_reads: int = pydantic.Field(
        ..., description="The number of input reads processed."
    )

    output_molecules: int = pydantic.Field(
        ..., description="The total number of unique molecules after aggregation."
    )

    umi1_degree_distribution: dict[int, int] = pydantic.Field(
        ..., description="The degree distribution of the umi1."
    )

    umi2_degree_distribution: dict[int, int] = pydantic.Field(
        ..., description="The degree distribution of the umi2."
    )

    @pydantic.computed_field(  # type: ignore
        description="The number of output reads.", return_type=int
    )
    @property
    def output_reads(self) -> int:
        """The total number of output reads.

        This is an alias for the input reads as no reads are removed during
        the aggregate step.
        """
        return self.input_reads

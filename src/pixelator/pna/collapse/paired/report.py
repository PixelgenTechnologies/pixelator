"""Functions and classes related to reporting for the collapse stage.

Copyright © 2024 Pixelgen Technologies AB
"""

import pydantic

from pixelator.pna.collapse.paired.statistics import (
    CollapseInputFile,
    MarkerLinkGroupStats,
)
from pixelator.pna.report.models.base import SampleReport


class CollapseSampleReport(SampleReport):
    """Model for report data returned by the collapse stage."""

    report_type: str = "collapse"
    input_reads: int = pydantic.Field(
        ..., description="The number of input reads processed."
    )

    input_molecules: int = pydantic.Field(
        ...,
        description="The number of unique molecules (unique input reads) processed.",
    )

    @pydantic.computed_field(  # type: ignore
        description="The number of output reads.", return_type=int
    )
    @property
    def output_reads(self) -> int:
        """The total number of error-corrected output reads.

        This is an alias for the input reads as no reads are removed during the collapse step.
        """
        return self.input_reads

    output_molecules: int = pydantic.Field(
        ..., description="The total number of error-corrected molecules detected."
    )

    unique_marker_links: int = pydantic.Field(
        ...,
        description="The total number of unique marker links (edges) in the output molecules.",
    )

    processed_files: list[CollapseInputFile] = pydantic.Field(
        ..., description="The files processed during the collapse step."
    )
    markers: list[MarkerLinkGroupStats] = pydantic.Field(
        ..., description="The statistics for each marker pair."
    )

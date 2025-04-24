"""Functions and classes related to reporting for the collapse stage.

Copyright © 2024 Pixelgen Technologies AB
"""

import pydantic

from pixelator.pna.collapse.statistics import (
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

    output_molecules: int = pydantic.Field(
        ..., description="The total number of error-corrected molecules detected."
    )

    corrected_reads: int = pydantic.Field(
        ...,
        description="The total number of passed reads that were error-corrected and reassigned to a different molecule.",
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

    @pydantic.computed_field(  # type: ignore
        description="The total number of output reads.",
        return_type=int,
    )
    @property
    def output_reads(self) -> int:
        """Return the number of output reads."""
        # This is defined as an alias to input reads since collapse does not filter anything.
        # Corrected reads are can just be assigned to different molecules but the total number of reads is the same.
        return self.input_reads

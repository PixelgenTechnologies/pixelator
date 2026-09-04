"""Functions and classes related to reporting for the collapse stage.

Copyright © 2024 Pixelgen Technologies AB
"""

import typing

import pydantic

from pixelator.pna.collapse.independent.collapser import MarkerCorrectionStats
from pixelator.pna.report.models.base import SampleReport


class IndependentCollapseSampleReport(SampleReport):
    """Model for report data returned by the collapse stage."""

    report_type: str = "collapse-independent"

    input_reads: int = pydantic.Field(
        ..., description="The number of input reads processed."
    )

    input_molecules: int = pydantic.Field(
        ...,
        description="The number of molecules (unique input reads) processed.",
    )

    output_molecules: int = pydantic.Field(
        ..., description="The total number of error-corrected molecules detected."
    )

    @pydantic.computed_field(  # type: ignore
        description="The number of UMI1 reads that had error and were corrected.",
        return_type=int,
    )
    @property
    def corrected_umi1_reads(self) -> int:
        """The number of UMI1 reads that had an error and were corrected."""
        return sum([m.corrected_reads for m in self.markers if m.region_id == "umi-1"])

    @pydantic.computed_field(  # type: ignore
        description="The number of UMI2 reads that had error and were corrected.",
        return_type=int,
    )
    @property
    def corrected_umi2_reads(self) -> int:
        """The number of UMI2 reads that had an error and were corrected."""
        return sum([m.corrected_reads for m in self.markers if m.region_id == "umi-2"])

    corrected_reads: int = pydantic.Field(
        ...,
        description="The number of reads that had an error in either UMI1, UMI2 or both and were corrected.",
    )

    markers: list[MarkerCorrectionStats] = pydantic.Field(
        ..., description="The statistics for each marker pair."
    )

    umi1_degree_distribution: dict[int, int] = pydantic.Field(
        ..., description="The degree distribution of the umi1."
    )

    umi2_degree_distribution: dict[int, int] = pydantic.Field(
        ..., description="The degree distribution of the umi2."
    )

    output_reads: int = pydantic.Field(
        default=0,
        description="The number of reads written after uint16 overflow filtering.",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _default_output_reads(cls, data: typing.Any) -> typing.Any:
        """Mirror input reads when output reads are omitted (legacy reports)."""
        if isinstance(data, dict) and data.get("output_reads") is None:
            data["output_reads"] = data.get("input_reads", 0)
        return data

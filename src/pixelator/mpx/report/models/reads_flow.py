"""Model for flow of input/output of read counts through processing stages.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import pydantic

from pixelator.mpx.report.models.base import SampleReport


class ReadsDataflowReport(SampleReport):
    """Model for tracking the read counts through all processing stages."""

    input_read_count: int = pydantic.Field(
        ...,
        description="The number of raw input reads from the input fastq files.",
    )

    qc_filtered_read_count: int = pydantic.Field(
        ...,
        description="The number of input reads after basic QC filtering.",
    )

    valid_pbs_read_count: int = pydantic.Field(
        ...,
        description="The number of input reads after QC filtering and with valid PBS1/2 regions.",
    )

    valid_antibody_read_count: int = pydantic.Field(
        ...,
        description=(
            "The number of input reads after QC filtering and with valid PBS1/2 regions and with a valid "
            "antibody barcode."
        ),
    )

    raw_molecule_read_count: int = pydantic.Field(
        ...,
        description=(
            "The number of reads that are attributed to a unique molecule in the sample "
            "after deduplication of reads with close UPIA and UMI sequences. "
            "Note that this should be equal to valid_antibody_read_count."
        ),
    )

    size_filter_fail_molecule_read_count: int = pydantic.Field(
        ...,
        description=(
            "The number of reads in components that do NOT pass "
            "the component size filters."
        ),
    )

    aggregate_molecule_read_count: int | None = pydantic.Field(
        ...,
        description="The number of reads in components identified as aggregates.",
    )

    cell_molecule_read_count: int = pydantic.Field(
        ..., description="The number of reads in cell or aggregate components."
    )

    @pydantic.computed_field(return_type=float)  # type: ignore
    @property
    def fraction_reads_in_molecules(self):
        """Return the fraction of raw input reads in unique molecules."""
        return self.raw_molecule_read_count / self.input_read_count

    @pydantic.computed_field(return_type=float)  # type: ignore
    @property
    def fraction_reads_in_cells(self) -> float:
        """Return the fraction of molecule reads in cells."""
        return self.cell_molecule_read_count / self.input_read_count

    @pydantic.computed_field(return_type=float)  # type: ignore
    @property
    def fraction_valid_pbs_reads(self) -> float:
        """Return the fraction of QC filtered input reads that has a valid PBS region."""
        return self.valid_pbs_read_count / self.input_read_count

    @pydantic.computed_field(return_type=float)  # type: ignore
    @property
    def fraction_valid_antibody_reads(self) -> float:
        """Return the fraction of reads with a valid PBS that have a valid antibody barcode."""
        return self.valid_antibody_read_count / self.input_read_count

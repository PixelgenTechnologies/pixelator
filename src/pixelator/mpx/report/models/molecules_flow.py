"""Model for flow of input/output of molecule counts through processing stages.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import textwrap

import pydantic

from pixelator.mpx.report.models.base import SampleReport


class MoleculesDataflowReport(SampleReport):
    """Model for flow of input/output counts through processing stages."""

    raw_molecule_count: int = pydantic.Field(
        ...,
        description=textwrap.dedent(
            """The number of raw unique molecules in the sample.
            This is the number of unique molecules after deduplication of reads with
            close UPIA UPIA and UMI sequences by the collapse stage.
            """
        ),
    )

    size_filter_fail_molecule_count: int = pydantic.Field(
        ...,
        description=(
            "The number of molecules in components that do NOT pass "
            "the component size filters."
        ),
    )

    aggregate_molecule_count: int | None = pydantic.Field(
        ...,
        description="The number of molecules in components identified as aggregates.",
    )

    cell_molecule_count: int = pydantic.Field(
        ...,
        description=textwrap.dedent(
            """
            The number of molecules in the graph after annotate.
            Note that this includes the molecules in aggregate components.
            """
        ),
    )

    @pydantic.computed_field(  # type: ignore
        return_type=float,
        description="Return the fraction of raw input reads in unique molecules.",
    )
    @property
    def fraction_molecules_in_cells(self) -> float:
        """Return the fraction of raw input reads in unique molecules."""
        return self.cell_molecule_count / self.raw_molecule_count

    @pydantic.computed_field(  # type: ignore
        return_type=float,
        description=textwrap.dedent(
            """Return the fraction of raw molecules that were discarded.
            This is equal to: 1 - `fraction_molecules_in_cells`.
            """
        ),
    )
    @property
    def fraction_molecules_discarded(self) -> float:
        """Return the fraction of raw molecules that were discarded.

        This is 1 - `fraction_molecules_in_cells`.
        """
        return 1 - self.cell_molecule_count / self.raw_molecule_count

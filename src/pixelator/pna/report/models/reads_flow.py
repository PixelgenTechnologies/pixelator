"""Model for flow of input/output of read counts through processing stages.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import typing

import pydantic

from pixelator.pna.report.models.base import SampleReport


class ReadsDataflowReport(SampleReport):
    """Model for tracking the read counts through all processing stages."""

    report_type: typing.Literal["reads_dataflow"] = "reads_dataflow"

    input_read_count: int = pydantic.Field(
        ...,
        description="The number of raw input reads from the input fastq files.",
    )

    amplicon_output_read_count: int = pydantic.Field(
        ...,
        description="The number of valid reads after QC filtering and amplicon assembly.",
    )

    demux_output_read_count: int = pydantic.Field(
        ...,
        description="The number of input reads after QC filtering that could be demultiplexed.",
    )

    collapse_output_molecule_count: int = pydantic.Field(
        ...,
        description="The number of reads remaining after error-correction to known antibody markers.",
    )

    graph_output_molecule_count: int = pydantic.Field(
        ...,
        description="The number of molecules in the proximity graph.",
    )

"""Reporting metadata for the native edgelist command.

Copyright © 2026 Pixelgen Technologies AB.
"""

import pydantic

from pixelator.pna.report.models import SampleReport


class Q30Statistics(pydantic.BaseModel):
    """Fraction of bases at Phred 30 or better, per region of interest.

    Measured over every read the stage consumed, including the ones it went on
    to discard, since quality is profiled ahead of the keep/drop decision. Each
    fraction pools base counts across reads rather than averaging per-read
    rates, so a short read does not weigh as much as a long one.

    A region that no read was long enough to reach reports 0.0.
    """

    total: float = pydantic.Field(
        ...,
        description="Over the whole read: R1 and R2 together for paired-end input, the single read otherwise.",
    )
    umi1: float = pydantic.Field(..., description="Over the umi-1 region.")
    pid1: float = pydantic.Field(..., description="Over the pid-1 region.")
    pid2: float = pydantic.Field(..., description="Over the pid-2 region.")
    umi2: float = pydantic.Field(..., description="Over the umi-2 region.")


class EdgelistSampleReport(SampleReport):
    """Model for a sample report from the native edgelist stage."""

    report_type: str = "edgelist"

    input_reads: int = pydantic.Field(
        ..., description="The number of input reads processed."
    )
    output_reads: int = pydantic.Field(
        ...,
        description="The number of reads that yielded an edgelist row, i.e. the row count of the parquet.",
    )
    q30_statistics: Q30Statistics = pydantic.Field(
        ..., description="Q30 fractions over the whole read and per region."
    )
    dropped_by: dict[str, int] = pydantic.Field(
        ...,
        description=(
            "Discarded reads per reason, most frequent first. Every reason is "
            "listed, including those that never fired, so the key set is the "
            "same on every run."
        ),
    )
    reads_per_marker_1: dict[str, int] = pydantic.Field(
        ...,
        description=(
            "Kept reads per marker on the pid-1 side (the parquet's marker_1 "
            "column), most frequent first. Every panel marker is listed, "
            "including those never seen."
        ),
    )
    reads_per_marker_2: dict[str, int] = pydantic.Field(
        ...,
        description=(
            "Kept reads per marker on the pid-2 side (the parquet's marker_2 "
            "column), most frequent first. Every panel marker is listed, "
            "including those never seen."
        ),
    )

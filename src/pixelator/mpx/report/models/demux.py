"""Model for report data returned by the single-cell demux stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pydantic

from pixelator.common.utils import get_sample_name
from pixelator.mpx.report.models.base import SampleReport


class DemuxSampleReport(SampleReport):
    """Model for data returned by the demux stage."""

    input_read_count: int = pydantic.Field(
        ...,
        description="The total number of input reads in the demux stage.",
    )

    output_read_count: int = pydantic.Field(
        ...,
        description="The total number of reads that passed the demux stage.",
    )

    per_antibody_read_counts: dict[str, int] = pydantic.Field(
        ...,
        description="The number of reads per antibody after demultiplexing.",
    )

    @pydantic.computed_field(  # type: ignore
        return_type=dict[str, float],
        description="The fraction of reads per antibody after demultiplexing.",
    )
    @property
    def per_antibody_read_count_fractions(self) -> dict[str, float]:  # noqa: D102
        return {
            antibody: count / self.output_read_count
            for antibody, count in self.per_antibody_read_counts.items()
        }

    @pydantic.computed_field(  # type: ignore
        return_type=int,
        description="Number of reads without a recognized antibody barcode.",
    )
    @property
    def unrecognised_antibody_read_count(self) -> int:  # noqa: D102
        return self.input_read_count - self.output_read_count

    @pydantic.computed_field(  # type: ignore
        return_type=int,
        description="Fraction of reads without a recognized antibody barcode.",
    )
    @property
    def fraction_unrecognised_antibody_reads(self) -> float:  # noqa: D102
        return self.unrecognised_antibody_read_count / self.input_read_count

    @classmethod
    def from_json(cls, p: Path) -> DemuxSampleReport:
        """Initialize an :class:`DemuxSampleReport` from a cutadapt report file."""
        sample_name = get_sample_name(p)

        with open(p) as fp:
            json_data = json.load(fp)

        input_read_count = json_data["read_counts"]["input"]
        output_read_count = json_data["read_counts"]["read1_with_adapter"]

        per_antibody_read_counts = dict()

        for antibody_json in sorted(
            json_data["adapters_read1"], key=lambda x: x["name"]
        ):
            antibody_name = antibody_json["name"]
            per_antibody_read_counts[antibody_name] = antibody_json["total_matches"]

        return cls(
            sample_id=sample_name,
            input_read_count=input_read_count,
            output_read_count=output_read_count,
            per_antibody_read_counts=per_antibody_read_counts,
        )

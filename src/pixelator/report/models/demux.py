from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pydantic

from pixelator.report.models.base import StageReport
from pixelator.utils import get_sample_name

try:
    from typing import Self, Tuple
except ImportError:
    from typing_extensions import Self


class DemuxStageReport(StageReport):
    """Model for data returned by the demux stage."""

    #: The total number of input reads in the demux stage
    input_read_count: int

    #: The total number of reads that passed the demux stage
    output_read_count: int

    #: The number of reads per antibody after demultiplexing
    per_antibody_read_counts: dict[str, int]

    @property
    def per_antibody_read_fractions(self) -> dict[str, float]:
        """The fraction of reads per antibody after demultiplexing."""
        return {
            antibody: count / self.output_read_count
            for antibody, count in self.per_antibody_read_counts.items()
        }

    @property
    def unrecognised_antibody_read_count(self) -> int:
        """Number of reads without a recognized antibody barcode."""
        return self.input_read_count - self.output_read_count

    @classmethod
    def from_json(cls, p: Path) -> Self:
        """Initialize an :class:`DemuxStageReport` from a cutadapt report file."""
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

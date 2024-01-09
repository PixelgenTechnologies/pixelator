from __future__ import annotations

import json
from pathlib import Path

import pydantic

from pixelator.report.models.base import StageReport
from pixelator.utils import get_sample_name

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class PreQCStageReport(StageReport):
    """Model for data returned by the preqc stage."""

    #: The total number of input reads in the preqc stage
    total_read_count: int

    #: The number of reads that passed the filter
    passed_filter_read_count: int

    #: The number of low quality reads
    low_quality_read_count: int

    #: The number of reads discarded because of too many Ns
    too_many_n_read_count: int

    #: The number of reads discarded because they are too short
    too_short_read_count: int

    #: The number of reads discarded because they are too long
    too_long_read_count: int

    @pydantic.computed_field(return_type=float)
    @property
    def discarded(self) -> float:
        return 1 - (self.passed_filter_read_count / self.total_read_count)

    @classmethod
    def from_json(cls, p: Path) -> Self:
        """Initialize an :class:`AmpliconStageReport` from a report file."""
        sample_name = get_sample_name(p)

        with open(p) as fp:
            json_data = json.load(fp)

        filtering_results = json_data["filtering_result"]

        data = {
            "total_read_count": json_data["summary"]["before_filtering"]["total_reads"],
            "passed_filter_read_count": filtering_results["passed_filter_reads"],
            "low_quality_read_count": filtering_results["low_quality_reads"],
            "too_many_n_read_count": filtering_results["too_many_N_reads"],
            "too_short_read_count": filtering_results["too_short_reads"],
            "too_long_read_count": filtering_results["too_long_reads"],
        }

        return cls(sample_id=sample_name, **data)

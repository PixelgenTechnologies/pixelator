"""Model for report data returned by the single-cell preqc stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pydantic

from pixelator.common.utils import get_sample_name
from pixelator.mpx.report.models.base import SampleReport


class PreQCSampleReport(SampleReport):
    """Model for data returned by the preqc stage."""

    total_read_count: int = pydantic.Field(
        ..., description="The total number of input reads in the preqc stage."
    )

    passed_filter_read_count: int = pydantic.Field(
        ...,
        description="The number of reads that passed the filters in the preqc stage.",
    )

    low_quality_read_count: int = pydantic.Field(
        ..., description="The number of low quality reads."
    )

    too_many_n_read_count: int = pydantic.Field(
        ..., description="The number of reads discarded because of too many Ns."
    )

    too_short_read_count: int = pydantic.Field(
        ..., description="The number of reads discarded because they are too short."
    )

    too_long_read_count: int = pydantic.Field(
        ..., description="The number of reads discarded because they are too long."
    )

    @pydantic.computed_field(  # type: ignore
        return_type=float,
        description="The fraction of reads that was discarded in this stage.",
    )
    @property
    def discarded(self) -> float:  # noqa: D102
        return 1 - (self.passed_filter_read_count / self.total_read_count)

    @classmethod
    def from_json(cls, p: Path) -> PreQCSampleReport:
        """Initialize an :class:`PreQCSampleReport` from a report file."""
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

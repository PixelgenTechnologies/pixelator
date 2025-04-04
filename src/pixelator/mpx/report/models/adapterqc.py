"""Model for report data returned by the single-cell adapterqc stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pydantic

from pixelator.common.utils import get_sample_name
from pixelator.mpx.report.models.base import SampleReport


class AdapterQCSampleReport(SampleReport):
    """Model for data returned by the adapterqc stage."""

    total_read_count: int = pydantic.Field(
        ...,
        description="The total number of input reads in the adapterqc stage.",
    )

    passed_filter_read_count: int = pydantic.Field(
        ...,
        description="The number of reads that passed the filter in the adapterqc stage.",
    )

    @pydantic.computed_field(  # type: ignore
        return_type=float,
        description="The fraction of reads that was discarded in this stage.",
    )
    @property
    def discarded(self) -> float:  # noqa: D102
        return 1 - (self.passed_filter_read_count / self.total_read_count)

    @classmethod
    def from_json(cls, p: Path) -> AdapterQCSampleReport:
        """Initialize an :class:`AdapterQCSampleReport` from a report file."""
        sample_name = get_sample_name(p)

        with open(p) as fp:
            json_data = json.load(fp)

        data = {
            "total_read_count": json_data["read_counts"]["input"],
            "passed_filter_read_count": json_data["read_counts"]["output"],
        }

        return cls(sample_id=sample_name, **data)

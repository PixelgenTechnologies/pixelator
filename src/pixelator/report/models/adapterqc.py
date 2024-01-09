from __future__ import annotations

import json
from pathlib import Path

from polars.dependencies import pydantic

from pixelator.report.models.base import StageReport
from pixelator.utils import get_sample_name

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class AdapterQCStageReport(StageReport):
    """Model for data returned by the adapterqc stage."""

    #: The total number of input reads in the preqc stage
    total_read_count: int

    #: The number of reads that passed the filter
    passed_filter_read_count: int

    @pydantic.computed_field(return_type=float)
    def discarded(self) -> float:
        return 1 - (self.passed_filter_read_count / self.total_read_count)

    @classmethod
    def from_json(cls, p: Path) -> Self:
        """Initialize an :class:`AdapterQCStageReport` from a report file."""
        sample_name = get_sample_name(p)

        with open(p) as fp:
            json_data = json.load(fp)

        data = {
            "total_read_count": json_data["read_counts"]["input"],
            "passed_filter_read_count": json_data["read_counts"]["output"],
        }

        return cls(sample_id=sample_name, **data)

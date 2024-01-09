"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
from pathlib import Path

import pydantic

from pixelator.report.models.base import StageReport
from pixelator.utils import get_sample_name

try:
    from typing import Self, Tuple
except ImportError:
    from typing_extensions import Self


class CollapseStageReport(StageReport):
    """Model for data returned by the demux stage."""

    #: The total number of reads assigned to unique molecules after collapsing
    output_read_count: int

    #: The number of unique molecules (based on UMI UPIA)
    unique_molecule_count: int

    #: The total number of edges after collapsing
    edge_count: int

    @pydantic.computed_field(return_type=float)
    def fraction_duplicate_reads(self) -> float:
        """The fraction of reads (after preprocessing) that are PCR duplicates."""
        return 1.0 - (self.unique_molecule_count / self.output_read_count)

    @classmethod
    def from_json(cls, p: Path) -> Self:
        """Initialize an :class:`CollapseStageReport` from a json report file."""
        sample_name = get_sample_name(p)

        with open(p) as fp:
            json_data = json.load(fp)

        output_read_count = json_data["total_count"]
        unique_read_count = json_data["total_unique_umi"]
        edge_count = json_data["total_pixels"]

        return cls(
            sample_id=sample_name,
            output_read_count=output_read_count,
            unique_molecule_count=unique_read_count,
            edge_count=edge_count,
        )

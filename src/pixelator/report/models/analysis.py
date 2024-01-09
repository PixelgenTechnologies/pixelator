"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
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


class AnalysisStageReport(StageReport):
    """Model for report data returned by the single-cell analysis stage."""

    antibody_control: list[str]

    @classmethod
    def from_json(cls, p: Path) -> Self:
        """Initialize an :class:`CollapseStageReport` from a json report file."""
        sample_name = get_sample_name(p)

        with open(p) as fp:
            json_data = json.load(fp)

        antibody_control = json_data["antibody_control"]

        return cls(sample_id=sample_name, antibody_control=antibody_control)

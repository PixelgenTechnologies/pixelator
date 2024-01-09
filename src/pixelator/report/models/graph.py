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


class GraphStageReport(StageReport):
    """Model for report data returned by the graph stage."""

    total_upia: int
    total_upib: int
    total_umi: int
    total_upi: int
    frac_upib_upia: float
    marker_count: int
    edge_count: int
    mean_count: float
    upia_degree_mean: float
    upia_degree_median: float
    vertices: int
    components: int
    components_modularity: float
    frac_largest_edges: float
    frac_largest_vertices: float

    @classmethod
    def from_json(cls, p: Path) -> Self:
        """Initialize an :class:`CollapseStageReport` from a json report file."""
        sample_name = get_sample_name(p)

        with open(p) as fp:
            json_data = json.load(fp)

        # TODO: Review names of these variables

        total_upia = json_data["total_upia"]
        total_upib = json_data["total_upib"]
        total_umi = json_data["total_umi"]
        total_upi = json_data["total_upi"]
        frac_upib_upia = json_data["frac_upib_upia"]
        marker_count = json_data["markers"]
        edge_count = json_data["edges"]
        mean_count = json_data["mean_count"]
        upia_degree_mean = json_data["upia_degree_mean"]
        upia_degree_median = json_data["upia_degree_median"]
        vertices = json_data["vertices"]
        components = json_data["components"]
        components_modularity = json_data["components_modularity"]
        frac_largest_edges = json_data["frac_largest_edges"]
        frac_largest_vertices = json_data["frac_largest_vertices"]

        return cls(
            sample_id=sample_name,
            total_upia=total_upia,
            total_upib=total_upib,
            total_umi=total_umi,
            total_upi=total_upi,
            frac_upib_upia=frac_upib_upia,
            marker_count=marker_count,
            edge_count=edge_count,
            mean_count=mean_count,
            upia_degree_mean=upia_degree_mean,
            upia_degree_median=upia_degree_median,
            vertices=vertices,
            components=components,
            components_modularity=components_modularity,
            frac_largest_edges=frac_largest_edges,
            frac_largest_vertices=frac_largest_vertices,
        )

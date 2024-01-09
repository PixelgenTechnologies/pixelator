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
    from typing import Optional, Self, Tuple
except ImportError:
    from typing_extensions import Self


class AnnotateStageReport(StageReport):
    """Model for report data returned by the annotate stage."""

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

    cells_filtered: int
    total_markers: int
    total_umis: int
    total_reads_cell: int
    median_reads_cell: int
    mean_reads_cell: float
    median_upi_cell: int
    mean_upi_cell: float
    median_upia_cell: int
    mean_upia_cell: float
    median_umi_cell: int
    mean_umi_cell: float
    median_umi_upia_cell: float
    mean_umi_upia_cell: float
    median_upia_degree_cell: float
    mean_upia_degree_cell: float
    median_markers_cell: int
    mean_markers_cell: float
    upib_per_upia: float

    # Aggregate calling metrics
    number_of_aggregates: Optional[int]
    fraction_of_aggregates: Optional[float]
    reads_of_aggregates: Optional[float]
    umis_of_aggregates: Optional[float]

    min_size_threshold: Optional[int]
    max_size_threshold: Optional[int]
    doublet_size_threshold: Optional[int]

    @pydantic.computed_field(return_type=float)
    def fraction_umis_in_non_cell_components(self) -> float:
        """The fraction of UMIs discarded during the annotation stage."""
        if self.umis_of_aggregates is not None:
            return self.umis_of_aggregates / self.total_umi
        return 0.0

    @classmethod
    def from_json(cls, p: Path) -> Self:
        """Initialize a :py:class:`CollapseStageReport` from a json report file."""
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

        cells_filtered = json_data["cells_filtered"]
        total_markers = json_data["total_markers"]
        total_umis = json_data["total_umis"]
        total_reads_cell = json_data["total_reads_cell"]
        median_reads_cell = json_data["median_reads_cell"]
        mean_reads_cell = json_data["mean_reads_cell"]
        median_upi_cell = json_data["median_upi_cell"]
        mean_upi_cell = json_data["mean_upi_cell"]
        median_upia_cell = json_data["median_upia_cell"]
        mean_upia_cell = json_data["mean_upia_cell"]
        median_umi_cell = json_data["median_umi_cell"]
        mean_umi_cell = json_data["mean_umi_cell"]
        median_umi_upia_cell = json_data["median_umi_upia_cell"]
        mean_umi_upia_cell = json_data["mean_umi_upia_cell"]
        median_upia_degree_cell = json_data["median_upia_degree_cell"]
        mean_upia_degree_cell = json_data["mean_upia_degree_cell"]
        median_markers_cell = json_data["median_markers_cell"]
        mean_markers_cell = json_data["mean_markers_cell"]
        upib_per_upia = json_data["upib_per_upia"]
        number_of_aggregates = json_data["number_of_aggregates"]
        fraction_of_aggregates = json_data["fraction_of_aggregates"]
        reads_of_aggregates = json_data["reads_of_aggregates"]
        umis_of_aggregates = json_data["umis_of_aggregates"]
        min_size_threshold = json_data["min_size_threshold"]
        max_size_threshold = json_data["max_size_threshold"]
        doublet_size_threshold = json_data["doublet_size_threshold"]

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
            cells_filtered=cells_filtered,
            total_markers=total_markers,
            total_umis=total_umis,
            total_reads_cell=total_reads_cell,
            median_reads_cell=median_reads_cell,
            mean_reads_cell=mean_reads_cell,
            median_upi_cell=median_upi_cell,
            mean_upi_cell=mean_upi_cell,
            median_upia_cell=median_upia_cell,
            mean_upia_cell=mean_upia_cell,
            median_umi_cell=median_umi_cell,
            mean_umi_cell=mean_umi_cell,
            median_umi_upia_cell=median_umi_upia_cell,
            mean_umi_upia_cell=mean_umi_upia_cell,
            median_upia_degree_cell=median_upia_degree_cell,
            mean_upia_degree_cell=mean_upia_degree_cell,
            median_markers_cell=median_markers_cell,
            mean_markers_cell=mean_markers_cell,
            upib_per_upia=upib_per_upia,
            number_of_aggregates=number_of_aggregates,
            fraction_of_aggregates=fraction_of_aggregates,
            reads_of_aggregates=reads_of_aggregates,
            umis_of_aggregates=umis_of_aggregates,
            min_size_threshold=min_size_threshold,
            max_size_threshold=max_size_threshold,
            doublet_size_threshold=doublet_size_threshold,
        )

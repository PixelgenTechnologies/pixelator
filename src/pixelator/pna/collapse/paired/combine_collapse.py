"""Combine collapse reports and parquet files into a single report and parquet file.

Copyright © 2025 Pixelgen Technologies AB
"""

import json
import logging
from pathlib import Path, PurePath
from typing import Iterable

import pyarrow.parquet as pq

from pixelator.pna.collapse.paired.statistics import (
    CollapseStatistics,
    MarkerLinkGroupStats,
)

logger = logging.getLogger(__name__)


def combine_parquet_files(input_files: Iterable[Path], output_file: Path) -> Path:
    """Scan a directory for parquet files and stream them into a single parquet file.

    Args:
        input_files: The folder containing the parquet files.
        output_file: The path to the output parquet file.

    Returns:
        The path to the combined parquet file. This is the same path as `output_file`.

    """
    files = list(input_files)
    first_file = pq.read_table(files.pop(0))

    with pq.ParquetWriter(output_file, first_file.schema) as writer:
        writer.write_table(first_file)

        for f in files:
            table = pq.read_table(f)
            writer.write_table(table)

    return output_file


def combine_report_files(input_files: Iterable[Path]) -> CollapseStatistics:
    """Combine a list of JSON collapse report files into a single report.

    :param input_files: The folder containing the parquet files.
    :returns: A new CollapseStatistics instance containing the combined report data.
    """
    files = list(input_files)
    combined_stats = CollapseStatistics()

    for report_file in files:
        with open(report_file, "r") as f:
            json_data = json.load(f)

            per_marker_stats = json_data.pop("markers")
            collapse_data = json_data

            # TODO: Remove this path once the unnecessary nesting is removed from the JSON
            if (
                isinstance(per_marker_stats, list)
                and len(per_marker_stats) == 1
                and isinstance(per_marker_stats[0], list)
            ):
                per_marker_stats = per_marker_stats[0]

            # Add markers
            for marker_group in per_marker_stats:
                m = MarkerLinkGroupStats.model_validate(marker_group)
                combined_stats.add_marker_stats(
                    m.marker_1,
                    m.marker_2,
                    cluster_stats=m,
                )

            for file_data in collapse_data["processed_files"]:
                combined_stats.add_input_file(
                    PurePath(file_data["path"]),
                    molecule_count=file_data["molecule_count"],
                    file_size=file_data["file_size"],
                )

    return combined_stats

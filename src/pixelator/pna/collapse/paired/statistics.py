"""Collect statistics about the collapse process.

Copyright © 2024 Pixelgen Technologies AB.
"""

import dataclasses
import typing
from pathlib import Path, PurePath
from typing import Any

import polars as pl
import pydantic

from pixelator.common.report.models import SummaryStatistics


@dataclasses.dataclass(slots=True)
class CollapseInputFile:
    """Keep track of the input file to collapse.

    :param path: Path to the input file.
    :param file_size: The total size of the input file.
    :param molecule_count: The number of rows in the dataframe.
    """

    path: str
    file_size: int
    molecule_count: int


class MarkerLinkGroupStats(pydantic.BaseModel):
    """Collect statistics of groups of molecules that will be collapsed.

    Attributes
    ----------
    corrected_reads_count: int
        The total number of reads in unique molecules that were error corrected to another "close" molecule.
    cluster_size_distribution: 1D array of ints
        The result of binning the size of each group of close molecules.
    collapsed_molecules_count: int
        The total number of unique molecules (UMI1+UMI2+UEI) after error correction.

    """

    marker_1: str
    marker_2: str

    input_reads_count: int = 0
    input_molecules_count: int = 0

    corrected_reads_count: int = 0
    cluster_size_distribution: list = pydantic.Field(
        ...,
        default_factory=list,
        description="The result of binning the size of each group of close molecules.",
    )
    collapsed_molecules_count: int = 0
    unique_marker_links_count: int = 0

    read_count_per_collapsed_molecule_stats: SummaryStatistics | None = None
    read_count_per_unique_marker_link_stats: SummaryStatistics | None = None
    uei_count_per_unique_marker_link_stats: SummaryStatistics | None = None


class CollapseSummaryStatistics(pydantic.BaseModel):
    """Collect summary statistics on reads and ueis."""

    read_counts_stats: SummaryStatistics | None = None
    uei_stats: SummaryStatistics | None = None

    @staticmethod
    def from_lazy_frame(collapsed_lz_df: pl.LazyFrame):
        """Create a CollapseSummaryStatistics from a collapsed LazyFrame."""
        df = collapsed_lz_df.select("uei_count", "read_count").collect()
        uei_stats = SummaryStatistics.from_series(df.get_column("uei_count"))
        read_stats = SummaryStatistics.from_series(df.get_column("read_count"))

        return CollapseSummaryStatistics(
            read_counts_stats=read_stats,
            uei_stats=uei_stats,
        )


class CollapseStatistics:
    """Collect statistics about the collapse process."""

    def __init__(self) -> None:
        """Initialize the statistics collector."""
        self._processed_files: list[CollapseInputFile] = []
        self._marker_pair_data: dict[tuple[str, str], dict[str, Any]] = {}
        self._summary_statistics: CollapseSummaryStatistics | None = None

    @typing.overload
    def add_input_file(
        self, input_file: PurePath, molecule_count: int, file_size: int
    ) -> None: ...

    @typing.overload
    def add_input_file(
        self, input_file: Path, molecule_count: int, file_size: int | None = None
    ) -> None: ...

    def add_input_file(
        self,
        input_file: Path | PurePath,
        molecule_count: int,
        file_size: int | None = None,
    ) -> None:
        """Collect file statistics for an input file to the MoleculeCollapser.

        :param input_file: The input file to collapse.
        :param molecule_count: The number of molecules in the input file
        :raise TypeError: If file_size is not provided when input_file is a PurePath.
        """
        if file_size is None and isinstance(input_file, Path):
            file_size = input_file.stat(follow_symlinks=True).st_size
        elif file_size is None:
            raise TypeError("file_size must be provided if input_file is a PurePath")

        self._processed_files.append(
            CollapseInputFile(
                path=str(input_file),
                file_size=file_size,
                molecule_count=molecule_count,
            )
        )

    def add_marker_stats(
        self,
        marker1: str,
        marker2: str,
        *,
        cluster_stats: MarkerLinkGroupStats,
        elapsed_time: float | None = None,
    ) -> None:
        """Add statistics for a marker pair.

        :param marker1: The first marker in the pair.
        :param marker2: The second marker in the pair.
        :param input_molecules_count: The number of unique molecules (UMI1+UMI2+UEI) before error correction.
        :param input_reads_count: The number of reads in unique molecules before error correction.
        :param cluster_stats: The statistics for the marker pair.
        :param elapsed_time: The time taken to process the marker pair.
        """
        key = (marker1, marker2)
        if key in self._marker_pair_data:
            raise KeyError("Marker pair already exists")

        self._marker_pair_data[key] = {
            **cluster_stats.model_dump(),
        }

        if elapsed_time is not None:
            self._marker_pair_data[key]["elapsed_real_time"] = elapsed_time

    def add_summary_statistics(self, collapsed_lz_df: pl.LazyFrame) -> None:
        """Add summary statistics for the entire collapse process.

        :param collapsed_lz_df: The collapsed dataframe.
        """
        self._summary_statistics = CollapseSummaryStatistics.from_lazy_frame(
            collapsed_lz_df
        )

    def get_high_level_statistics(self) -> dict:
        """Compute high level summary statistics from all the marker pairs."""
        reads_input = sum(
            data["input_reads_count"] for data in self._marker_pair_data.values()
        )
        molecules_input = sum(
            data["input_molecules_count"] for data in self._marker_pair_data.values()
        )
        molecules_output = sum(
            data["collapsed_molecules_count"]
            for data in self._marker_pair_data.values()
        )
        corrected_reads = sum(
            data["corrected_reads_count"] for data in self._marker_pair_data.values()
        )
        unique_marker_links_count = sum(
            data["unique_marker_links_count"]
            for data in self._marker_pair_data.values()
        )

        stats = {
            "input_reads": reads_input,
            "input_molecules": molecules_input,
            "output_molecules": molecules_output,
            "corrected_reads": corrected_reads,
            "unique_marker_links": unique_marker_links_count,
        }

        if self._summary_statistics:
            stats["summary_statistics"] = self._summary_statistics.dict()

        return stats

    def to_dict(self) -> dict:
        """Return the statistics as a dictionary."""
        flat_markers_data = list(self._marker_pair_data.values())
        return {
            **self.get_high_level_statistics(),
            "processed_files": [dataclasses.asdict(f) for f in self._processed_files],
            "markers": flat_markers_data,
        }

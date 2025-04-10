"""Module for combining intermediate collapsed results into a final edgelist.

These routines are for combining data using the "independent" demuxing and collapsing strategy.

Copyright © 2025 Pixelgen Technologies AB
"""

import dataclasses
from collections import Counter
from pathlib import Path
from typing import Iterable, MutableMapping

import duckdb as dd
from duckdb import DuckDBPyConnection

from pixelator.pna.collapse.independent import (
    IndependentCollapseSampleReport,
    MarkerCorrectionStats,
    SingleUMICollapseSampleReport,
)
from pixelator.pna.collapse.paired.combine_collapse import logger


def _merge_sort_parquet(
    conn: DuckDBPyConnection, parquet: Iterable[Path], output_file: Path
) -> Path:
    """Combine and sort a partitioned collapsed dataset.

    DuckDB sorting can spill out to disk when memory space is limited.

    Args:
        conn: A DuckDB connection
        parquet: An iterable of input file paths
        output_file: The path to write the combined and sorted output to

    Returns:
        The path to the combined and sorted output file.

    """
    conn.sql(
        f"""
        CREATE OR REPLACE TEMP VIEW combined_sorted_data AS
        SELECT *
        FROM read_parquet({str([str(i) for i in parquet])})
        ORDER BY original_umi1, original_umi2, uei;
        """
    )
    conn.execute(
        f"""
        COPY combined_sorted_data TO '{output_file}' ( FORMAT 'parquet', CODEC 'zstd', COMPRESSION_LEVEL 6 );
        """
    )

    return output_file


@dataclasses.dataclass()
class CombineCollapseIndependentStats:
    """Some combine collapse statistics.

    Attributes:
        output_molecules: Total number of unique output molecules
        corrected_reads: Total number of corrected reads

    """

    output_molecules: int
    corrected_reads: int


def combine_independent_parquet_files(
    umi1_files: Iterable[Path],
    umi2_files: Iterable[Path],
    output_file: Path,
    memory_limit: str | None = None,
) -> CombineCollapseIndependentStats:
    """Scan a directory for parquet files with corrected UMI1s and UMI2s and join them.

    Use DuckDB for their support of larger than memory joins.

    The independently corrected partitioned umi1 and umi2 output files are first combined and sorted.
    This result in two sorted parquet files, one for with UMI1 correction info and one with the UMI2 correction.
    The sorting is done on the original_umi1,original_umi2 and uei columns.
    The umi regions before correction are common in the m1 and m2 parquet files,
    so we guaranteed that the row id of each molecule in the merged m1 file will match that
    of the row id in the m2 file.

    Combining the correction for the UMI1 and UMI2 can now be done using a POSITIONAL JOIN.

    The sorting and merging approach works well with DuckDB larger-than-memory processing.
    Simply merging and then using an INNER JOIN used a prohibitive amount of memory for our data
    which would try to join two tables with each in 100M+ rows.

    Note that we are also splitting the original molecule embedding into three separate integers
    (umi1, umi2, uei) in the demux step to optimize for the sorting operations here.

    This multi-key sorting helps with cache-locality in the sorting process and the use of integers
    enables DuckDB to use an efficient radix sort implementation.
    Sorting on the full molecule turned out to be too slow.

    See this conference paper for more details on DuckDB's sorting engine:
    https://hannes.muehleisen.org/publications/ICDE2023-sorting.pdf

    Args:
        umi1_files: The list of UMI1 parquet files.
        umi2_files: The list of UMI2 parquet files.
        output_file: The path to the output parquet file.
        memory_limit: The memory limit to use for DuckDB. eg: '16GB'
            If None, the default is used (80% of the system memory).

    Returns:
        A dictionary with additional statistics calculated on the combined parquet data.

    """
    conn = dd.connect(":memory:")

    if memory_limit is not None:
        conn.execute(f"SET memory_limit = ?", memory_limit)

    logger.info("Combining and sorting UMI1 parquet files")
    sorted_umi1_file = _merge_sort_parquet(
        conn, umi1_files, output_file.with_suffix(".m1.parquet")
    )

    logger.info("Combining and sorting UMI2 parquet files")
    sorted_umi2_file = _merge_sort_parquet(
        conn, umi2_files, output_file.with_suffix(".m2.parquet")
    )

    logger.info("Combining UMI1 and UMI2 data.")

    # UMI1 data
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW umi1_data AS
            SELECT marker_1, marker_2, read_count,
                   corrected_umi1 as umi1,
                   original_umi1,
                   corrected_umi1,
                   corrected_umi1 != original_umi1 as umi1_is_corrected,
                   uei
            FROM read_parquet('{str(sorted_umi1_file)}');
        """
    )

    # UMI2 data
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW umi2_data AS
            SELECT corrected_umi2 as umi2,
                   original_umi2,
                   corrected_umi2 != original_umi2 as umi2_is_corrected
            FROM read_parquet('{str(sorted_umi2_file)}');
        """
    )

    conn.execute(
        """
        CREATE OR REPLACE TEMP VIEW combined_data AS
            SELECT marker_1, marker_2, umi1, umi2,
                sum(read_count)::UINTEGER as read_count,
                count(DISTINCT uei)::USMALLINT as uei_count,
                ifnull(sum("read_count") FILTER ( umi1_data.umi1_is_corrected OR umi2_data.umi2_is_corrected ), 0)::UINTEGER as corrected_read_count
            FROM umi1_data POSITIONAL JOIN umi2_data
            GROUP BY marker_1, marker_2, umi1, umi2
        """
    )

    conn.execute(
        f"""
        COPY combined_data TO '{output_file}' ( FORMAT 'parquet', CODEC 'zstd', COMPRESSION_LEVEL 6 );
        """
    )

    # Another pass to collect some stats needed for the reports
    res = conn.sql(
        f"""
        SELECT sum(uei_count)::UINTEGER as output_molecules,
               sum(corrected_read_count)::UINTEGER as corrected_reads
        FROM read_parquet('{output_file}')
        """
    ).pl()

    output_molecules, corrected_reads = res.row(0)

    stats = CombineCollapseIndependentStats(
        output_molecules=int(output_molecules), corrected_reads=int(corrected_reads)
    )

    sorted_umi1_file.unlink()
    sorted_umi2_file.unlink()

    return stats


def combine_independent_report_files(
    marker1_reports: list[Path],
    marker2_reports: list[Path],
    sample_id: str,
    stats,
    output_file,
) -> None:
    """Combine collapse reports per marker into a single combined report.

    Args:
        marker1_reports: The list of marker1 reports.
        marker2_reports: The list of marker2 reports.
        sample_id: The sample ID to use for the combined report.
        stats: Additional statistics calculated on the combined parquet data.
            These statistics cannot be derived from the individual reports.
        output_file: The output file to write the combined report to.

    Returns:
        None

    """
    regions_to_combine = frozenset(
        (
            "input_reads",
            "input_molecules",
            "input_unique_umis",
            "output_unique_umis",
            "corrected_unique_umis",
            "corrected_reads",
        )
    )

    umi1_markers: list[MarkerCorrectionStats] = []
    umi1_summary_stats: MutableMapping[str, int] = Counter()

    for f in marker1_reports:
        umi1_report = SingleUMICollapseSampleReport.from_json(f)
        umi1_markers.extend(umi1_report.markers)
        umi1_summary_stats.update(
            {
                k: v
                for k, v in umi1_report.model_dump().items()
                if k in regions_to_combine
            }
        )

    umi2_markers: list[MarkerCorrectionStats] = []
    umi2_summary_stats: MutableMapping[str, int] = Counter()

    for f in marker2_reports:
        umi2_report = SingleUMICollapseSampleReport.from_json(f)
        umi2_markers.extend(umi2_report.markers)
        umi2_summary_stats.update(
            {
                k: v
                for k, v in umi2_report.model_dump().items()
                if k in regions_to_combine
            }
        )

    report = IndependentCollapseSampleReport(
        product_id="single-cell-pna",
        sample_id=sample_id,
        markers=umi1_markers + umi2_markers,
        input_reads=umi1_summary_stats["input_reads"],
        input_molecules=umi1_summary_stats["input_molecules"],
        corrected_reads=stats.corrected_reads,
        output_molecules=stats.output_molecules,
    )

    report.write_json_file(output_file, indent=4)
    return

"""Correction and demultiplexing of marker barcodes in a FASTQ file.

Copyright © 2024 Pixelgen Technologies AB.
"""

import logging
import sys
import typing
from pathlib import Path, PurePath
from typing import Literal

import duckdb as dd
import pyarrow as pa
import pyarrow.parquet
from cutadapt.files import FileOpener, InputPaths, OutputFiles
from cutadapt.pipeline import SingleEndPipeline
from cutadapt.steps import SingleEndSink
from cutadapt.utils import DummyProgress, Progress

from pixelator.common.exceptions import PixelatorBaseException
from pixelator.common.utils import get_sample_name
from pixelator.pna.config import PNAAntibodyPanel, PNAAssay
from pixelator.pna.demux.barcode_demuxer import (
    BarcodeDemuxer,
    IndependentBarcodeDemuxer,
    PairedBarcodeDemuxer,
    create_barcode_group_to_batch_mapping,
    independent_marker_groups_mapping,
)
from pixelator.pna.demux.barcode_identifier import BarcodeIdentifier
from pixelator.pna.demux.pipeline import (
    DemuxFilenamePolicy,
    DemuxPipeline,
    IndependentMarkersFilenamePolicy,
    ParallelDemuxPipelineRunner,
    PartsFilenamePolicy,
)
from pixelator.pna.demux.report import BarcodeCorrectionStatistics
from pixelator.pna.read_processing.runners import ParallelPipelineRunner
from pixelator.pna.utils import clean_suffixes

logger = logging.getLogger("demux")

import multiprocessing as mp


def correct_marker_barcodes(
    input: Path,
    assay: PNAAssay,
    panel: PNAAntibodyPanel,
    output: Path,
    save_failed: bool = True,
    mismatches: int = 1,
    threads: int = -1,
) -> tuple[BarcodeCorrectionStatistics, Path, Path | None]:
    """Correct the marker barcodes in a FASTQ file.

    Errors in the sequence barcodes within `mismatches` distance are
    corrected to the list of known markers in the panel.
    The original reads are left unmodified but the name of the
    marker is added to the header.

    :param input: The path to the input FASTQ file.
    :param assay: The assay design.
    :param panel: The antibody panel.
    :param output: The output directory.
    :param save_failed: Save failed reads to a separate file.
    :param mismatches: The number of mismatches allowed when correcting towards known markers.
    :param threads: The number of threads to use for processing. By default all available cores are used.
    """
    threads = threads if threads > 0 else mp.cpu_count()

    # Open file handles for input files
    input_files = InputPaths(str(input))

    # Open file handles for output files
    # The compression threads will be capped at 1
    compression_threads = 1
    file_opener = FileOpener(threads=compression_threads)

    logger.info(
        "Opening output file writer with %s compression threads", compression_threads
    )

    # Open file handles for output files
    # When writing from multiple cores the `proxied` argument will take care
    # of the necessary reordering.
    output_files = OutputFiles(
        proxied=True,
        qualities=True,
        file_opener=file_opener,
        interleaved=False,
    )

    # Open file handles for corrected and failed reads
    sample_name = get_sample_name(input)
    corrected_reads_path = Path(output / f"{sample_name}.demux.passed.fq.zst")
    failed_reads_path = None
    if save_failed:
        failed_reads_path = Path(output / f"{sample_name}.demux.failed.fq.zst")

    sink = SingleEndSink(output_files.open_record_writer(corrected_reads_path))

    failed_writer = None
    if save_failed:
        failed_writer = output_files.open_record_writer(failed_reads_path)

    barcodes_id = BarcodeIdentifier(
        assay=assay, panel=panel, mismatches=mismatches, writer=failed_writer
    )

    # Construct the pipeline
    pipeline = SingleEndPipeline(modifiers=[], steps=[barcodes_id, sink])

    # Progress bar for the pipeline
    if sys.stderr.isatty():
        progress = Progress()
    else:
        progress = DummyProgress()

    n_workers = max(1, threads - compression_threads)
    logger.info("Correcting barcodes using %s worker threads", n_workers)

    # Run the pipeline on a parallel runner
    runner = ParallelPipelineRunner(
        inpaths=input_files,
        n_workers=n_workers,
        statistics_class=BarcodeCorrectionStatistics,
    )

    with runner as r:
        stats = r.run(pipeline, progress, output_files)
        output_files.close()

    return (stats, corrected_reads_path, failed_reads_path)


def demux_barcode_groups(
    corrected_reads: Path,
    assay: PNAAssay,
    panel: PNAAntibodyPanel,
    stats: BarcodeCorrectionStatistics,
    output_dir: Path,
    reads_per_chunk: int = 50_000_000,
    max_chunks: int = 8,
    threads: int = -1,
    stategy: Literal["independent", "paired"] = "paired",
):
    """Demux a FASTQ file containing marker information in the header.

    Since there are many combinations of markers, the barcode groups are
    not demultiplexed in individual (pid1, pid1) combinations but in batches.
    Sequence records will be streamed into Arrow IPC files given a target batch size.

    Args:
        corrected_reads (Path): The path to the corrected FASTQ file.
        assay (PNAAssay): The assay design.
        panel (PNAAntibodyPanel): The antibody panel.
        stats (BarcodeCorrectionStatistics):
            The statistics from the barcode correction.
        output_dir (Path):
            The output directory.
        reads_per_chunk (int, optional):
            The target number of molecules in each batch.
        max_chunks (int, optional):
            The maximum number of batches.
        threads (int, optional):
            The number of threads to use for processing. The default of `-1` will use all available cores.
        stategy (Literal["independent", "paired"], optional):
            The demultiplexing strategy to use. Defaults to "paired".

    """
    # Open file handles for input files

    threads = threads if threads > 0 else max(mp.cpu_count(), 1)

    input_files = InputPaths(str(corrected_reads))

    # 1 Reader and 1 Writer thread but these do not require much CPU
    worker_threads = max(threads - 1, 1)
    prefix = clean_suffixes(PurePath(input_files.paths[0])).name
    prefix = prefix.replace(".demux.passed", ".demux")
    filename_policy: DemuxFilenamePolicy

    if stategy == "independent":
        m1_group_map, m2_group_map = independent_marker_groups_mapping(
            stats.pid_pair_counter,
            reads_per_chunk=reads_per_chunk,
            max_chunks=max_chunks,
        )
        barcode_demuxer = typing.cast(
            BarcodeDemuxer,
            IndependentBarcodeDemuxer(
                assay=assay,
                panel=panel,
                marker1_groups=m1_group_map,
                marker2_groups=m2_group_map,
            ),
        )
        filename_policy = typing.cast(
            DemuxFilenamePolicy,
            IndependentMarkersFilenamePolicy(
                prefix=prefix,
                m1_map=m1_group_map,
                m2_map=m2_group_map,
            ),
        )

    else:
        batch_mapping = create_barcode_group_to_batch_mapping(
            stats.pid_pair_counter,
            reads_per_chunk=reads_per_chunk,
            max_chunks=max_chunks,
        )
        barcode_demuxer = typing.cast(
            BarcodeDemuxer,
            PairedBarcodeDemuxer(assay=assay, panel=panel, supergroups=batch_mapping),
        )
        filename_policy = typing.cast(
            DemuxFilenamePolicy, PartsFilenamePolicy(prefix=prefix)
        )

    pipeline = DemuxPipeline(barcode_demuxer)
    runner = ParallelDemuxPipelineRunner(
        inpaths=input_files,
        n_workers=worker_threads,
        output_directory=output_dir,
        filename_policy=filename_policy,
    )
    progress = Progress()

    logger.info("Demuxing with %s worker threads", worker_threads)

    with runner as r:
        r.run(pipeline, progress)

    return output_dir


def finalize_batched_groups(
    input_dir: Path,
    output_dir: Path,
    remove_intermediates: bool = True,
    strategy: Literal["paired", "independent"] = "independent",
    memory: int | None = None,
):
    """Post-process the demuxed data by sorting and writing to Parquet.

    The demuxed data is written to Arrow IPC file format (v2) in the work directory.
    These are then sorted and written to Parquet format with ZSTD compression.

    The sorting is done on the `marker_1` and `marker_2` columns and written to parquet metadata
    which allows for fast contiguous reads [marker_1, marker_2] groups.

    Params:
        input_dir: the path to the work directory containing the demuxed data
        output_dir: the path to the output directory
        remove_intermediates: whether to remove the intermediate Arrow files after writing to parquet
        strategy: the demultiplexing strategy to use. Can be "paired" or "independent"
        memory: maximum amount of memory to use in bytes
    """
    if strategy == "independent":
        return _finalize_batched_groups_independent(
            input_dir,
            output_dir,
            remove_intermediates=remove_intermediates,
            memory=memory,
        )
    elif strategy == "paired":
        return _finalize_batched_groups_paired(
            input_dir,
            output_dir,
            remove_intermediates=remove_intermediates,
            memory=memory,
        )
    else:
        raise ValueError("Unknown strategy")


def _finalize_batched_groups_paired(
    input_dir: Path,
    output_dir,
    remove_intermediates: bool = True,
    memory: int | None = None,
):
    """Post-process the demuxed data by sorting and writing to Parquet.

    The demuxed data is written as intermediate parquet files in the work directory.
    These are then sorted and deduplicated.

    The sorting is done on the `marker_1` and `marker_2` columns and written to parquet metadata
    which allows for fast contiguous reads of [marker_1, marker_2] groups.

    Params:
        input_dir:
            the path to the work directory containing the demuxed data
        output_dir:
            the path to the output directory where the final parquet files are written
        remove_intermediates:
            Whether to remove the intermediate parquet files after sorting and deduplication
        memory:
            Maximum amount of memory to use. Use None to disable memory limits.

    Returns:
        A list of paths to the Parquet files

    Raises:
        ValueError: If no marker identifier (m1 or m2) is found in the Arrow IPC file name.

    """
    parquet_files = []
    tmp_parquet_files = list(input_dir.glob("*.parquet"))

    conn = dd.connect(":memory:")
    if memory:
        val = f"{memory / 10**6}MB"
        conn.execute(f"SET memory_limit = '{val}'")

    for f in tmp_parquet_files:
        output_name = str(clean_suffixes(f).name)
        output_name = output_name.removesuffix(".parquet")

        output_path = Path(output_dir) / f"{output_name}.parquet"
        parquet_files.append(output_path)

        conn.execute(
            f"""
            COPY (
               SELECT *, count(*) as read_count
               FROM read_parquet('{f}')
               GROUP BY ALL
               ORDER BY marker_1, marker_2
           ) TO '{output_path}' (FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 6);
           """
        )

        if remove_intermediates:
            f.unlink()

    return parquet_files


def _finalize_batched_groups_independent(
    input_dir: Path,
    output_dir: Path,
    remove_intermediates: bool = True,
    memory: int | None = None,
):
    """Post-process the demuxed data by sorting and writing to Parquet.

    The demuxed data is written to Arrow IPC file format (v2) in the work directory.
    These are then sorted and written to Parquet format with ZSTD compression.

    The sorting is done on the `marker_1` and `marker_2` columns and written to parquet metadata
    which allows for fast contiguous reads [marker_1, marker_2] groups.

    Params:
        work_dir:
            the path to the work directory containing the demuxed data
      remove_intermediates:
            Whether to remove the intermediate parquet files after sorting and deduplication
        memory:
            Maximum amount of memory to use. Use None to disable memory limits.

    Returns:
        A list of paths to the Parquet files

    Raises:
        ValueError: If no marker identifier (m1 or m2) is found in the Arrow IPC file name.

    """
    parquet_files = []
    tmp_files = list(input_dir.glob("*.parquet"))

    conn = dd.connect(":memory:")
    if memory:
        val = f"{memory / 10**6}MB"
        conn.execute(f"SET memory_limit = '{val}'")

    for f in tmp_files:
        sorting_order: tuple[str, str]

        if ".m1." in str(f):
            sorting_order = ("marker_1", "marker_2")
        elif ".m2." in str(f):
            sorting_order = ("marker_2", "marker_1")
        else:
            raise PixelatorBaseException(
                f"Unrecognised marker suffix. Could not determine sorting order"
            )

        output_name = str(clean_suffixes(f).name)
        output_name = output_name.removesuffix(".parquet")

        output_path = Path(output_dir) / f"{output_name}.parquet"
        parquet_files.append(output_path)

        # Params not supported in ORDER BY clause
        conn.execute(f"""
            COPY (
                SELECT *, count(*) as read_count
                FROM read_parquet('{f}')
                GROUP BY ALL
                ORDER BY {sorting_order[0]}, {sorting_order[1]}
            ) TO '{output_path}' (FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 6);
        """)

        if remove_intermediates:
            f.unlink()

    return parquet_files

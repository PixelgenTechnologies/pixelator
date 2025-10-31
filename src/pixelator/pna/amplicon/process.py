"""Copyright © 2024 Pixelgen Technologies AB."""

import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Sequence

from cutadapt.files import FileOpener, InputPaths, OutputFiles
from cutadapt.modifiers import (
    NextseqQualityTrimmer,
    PairedEndModifier,
    PairedEndModifierWrapper,
    QualityTrimmer,
    SingleEndModifier,
)
from cutadapt.steps import PairedEndStep, SingleEndFilter, SingleEndSink, SingleEndStep
from cutadapt.utils import DummyProgress, Progress

from pixelator.pna.amplicon.build_amplicon import (
    PairedEndAmpliconBuilder,
    SingleEndAmpliconBuilder,
)
from pixelator.pna.amplicon.filters import (
    LBSDetectedInUMI,
    LowComplexityUMI,
    SingleEndFilterWithFailureReason,
    TooManyN,
)
from pixelator.pna.amplicon.quality import QualityProfileStep
from pixelator.pna.amplicon.report import AmpliconStatistics
from pixelator.pna.config.assay import PNAAssay
from pixelator.pna.read_processing.pipeline import AmpliconPipeline
from pixelator.pna.read_processing.runners import ParallelPipelineRunner
from pixelator.pna.utils import clean_suffixes

logger = logging.getLogger(__name__)


def amplicon_fastq(
    inputs: Sequence[Path],
    assay: PNAAssay,
    output: Path,
    mismatches: float = 0.1,
    poly_g_trimming: bool = False,
    quality_cutoff: int = 20,
    low_complexity_filter: bool = True,
    low_complexity_threshold: float = 0.8,
    lbs_filter: bool = True,
    lbs_filter_min_overlap: int = 8,
    lbs_filter_error_rate: float = 0.1,
    threads: int = -1,
    save_failed: bool = False,
) -> AmpliconStatistics:
    """Output a FASTQ file with amplicon sequences.

    This function will take paired-end reads and combine them into the proper amplicon
    sequences based on the assay design. The reads will be quality trimmed and filtered.

    Args:
        inputs: The input files to process
        assay: The assay design to use
        output: The output file to write
        mismatches: The number of mismatches to allow
        poly_g_trimming: Whether to perform poly-G trimming
        quality_cutoff: The quality cutoff to use for quality trimming read tails
        low_complexity_filter: Whether to filter reads with low complexity UMIs
        low_complexity_threshold: The percentage of the UMI that must consists of a single base to be considered of low complexity.
        lbs_filter: Whether to filter reads with LBS detected in the UMI regions
        lbs_filter_min_overlap: The minimum overlap to use for LBS detection in UMI regions
        lbs_filter_error_rate: The maximum error rate to allow when determining overlap with the LBS sequence in UMI regions,
        threads: The number of cores to use. -1 will use all available cores
        save_failed: Whether to save reads that fail during amplicon combining to a separate file

    Returns:
        An `AmpliconStatistics` instance.

    """
    threads = threads if threads > 0 else mp.cpu_count()

    if len(inputs) not in [1, 2]:
        raise ValueError("Expected one or two input files, got %s" % len(inputs))

    # Open file handles for input files
    input_files = InputPaths(*(str(i) for i in inputs))

    # Open file handles for output files
    # The compression threads will be capped at 2
    write_threads = max(0, min(threads - 1, 2))
    file_opener = FileOpener(threads=write_threads)

    logging.info(
        "Opening output file writer with %s compression threads", write_threads
    )

    # Open file handles for output files
    # When writing from multiple cores the `proxied` argument will take care
    # of the necessary reordering.
    output_files = OutputFiles(
        proxied=threads > 1,
        qualities=True,
        file_opener=file_opener,
        interleaved=False,
    )

    # Open file handles for failed reads
    # TODO: Should this be configurable?
    output_filename = clean_suffixes(Path(inputs[0])).name
    failed_1 = Path(
        output.parent / f"{clean_suffixes(Path(inputs[0])).name}.failed.fq.zst"
    )
    if len(inputs) == 2:
        failed_2 = Path(
            output.parent / f"{clean_suffixes(Path(inputs[1])).name}.failed.fq.zst"
        )

    amplicon_failed_writer = None
    if save_failed:
        amplicon_failed_writer = output_files.open_record_writer(failed_1, failed_2)

    is_paired_end = len(inputs) == 2

    # Construct an amplicon builder class that will be used to combine the reads using the assay design

    builder_factory = (
        PairedEndAmpliconBuilder if is_paired_end else SingleEndAmpliconBuilder
    )
    builder = builder_factory(
        assay=assay, mismatches=mismatches, writer=amplicon_failed_writer
    )

    # ----------------------------------------
    # Configure pre amplicon filtering and trimming steps
    # ----------------------------------------

    pre_steps: list[PairedEndStep | SingleEndStep] = []
    pre_modifiers: list[PairedEndModifier | SingleEndModifier] = []

    if is_paired_end:
        pre_modifiers.append(
            PairedEndModifierWrapper(
                QualityTrimmer(cutoff_front=0, cutoff_back=quality_cutoff),
                QualityTrimmer(cutoff_front=0, cutoff_back=quality_cutoff),
            )
        )

        if poly_g_trimming:
            pre_modifiers.append(
                PairedEndModifierWrapper(
                    NextseqQualityTrimmer(cutoff=20), NextseqQualityTrimmer(cutoff=20)
                )
            )
    else:
        pre_modifiers.append(
            QualityTrimmer(cutoff_front=0, cutoff_back=quality_cutoff),
        )

        if poly_g_trimming:
            pre_modifiers.append(NextseqQualityTrimmer(cutoff=20))

    sink = SingleEndSink(output_files.open_record_writer(output))

    # ----------------------------------------
    # Configure post amplicon filtering steps
    # ----------------------------------------

    # Configure if failed reads should be saved to a separate output file
    post_failed_writer = None
    if save_failed:
        post_failed_writer = output_files.open_record_writer(
            output.parent / f"{output_filename}.post_failed.fq.zst"
        )

    post_filters: list[SingleEndStep] = []

    # Always run the TooManyN filter
    post_filters.append(
        SingleEndFilterWithFailureReason(
            predicate=TooManyN(0, assay), writer=post_failed_writer
        )
    )

    # Configure low complexity UMI filter
    if low_complexity_filter:
        post_filters.append(
            SingleEndFilterWithFailureReason(
                predicate=LowComplexityUMI(assay, proportion=low_complexity_threshold),
                writer=post_failed_writer,
            )
        )

    # Configure LBS in UMI filter
    if lbs_filter:
        post_filters.append(
            SingleEndFilterWithFailureReason(
                predicate=LBSDetectedInUMI(
                    assay,
                    min_overlap=lbs_filter_min_overlap,
                    max_error_rate=lbs_filter_error_rate,
                ),
                writer=post_failed_writer,
            )
        )

    # Construct the pipeline
    pipeline = AmpliconPipeline(
        combiner=builder,
        pre_modifiers=pre_modifiers,
        pre_steps=pre_steps,
        post_steps=post_filters
        + [
            QualityProfileStep(assay=assay),
            sink,
        ],
    )

    # Progress bar for the pipeline
    if sys.stderr.isatty():
        progress = Progress()
    else:
        progress = DummyProgress()

    n_workers = max(1, threads - write_threads)
    logging.info("Running pipeline with %s worker threads", n_workers)

    # Run the pipeline on a parallel runner
    runner = ParallelPipelineRunner(
        inpaths=input_files, n_workers=n_workers, statistics_class=AmpliconStatistics
    )

    with runner as r:
        stats = r.run(pipeline, progress, output_files)

    return stats

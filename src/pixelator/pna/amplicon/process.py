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
from pixelator.pna.amplicon.filters import TooManyN
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
    threads: int = -1,
    save_failed: bool = False,
) -> AmpliconStatistics:
    """Output a FASTQ file with amplicon sequences.

    This function will take paired-end reads and combine them into the proper amplicon
    sequences based on the assay design. The reads will be quality trimmed and filtered.

    :param inputs: The input files to process
    :param assay: The assay design to use
    :param output: The output file to write
    :param mismatches: The number of mismatches to allow
    :param poly_g_trimming: Whether to perform poly-G trimming
    :param quality_cutoff: The quality cutoff to use for quality trimming read tails
    :param threads: The number of cores to use. -1 will use all available cores
    :param save_failed: Whether to save reads that fail during amplicon combining to a separate file
    :return: A `AmpliconStatistics` instance
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

    # Construct the pipeline
    pipeline = AmpliconPipeline(
        combiner=builder,
        pre_modifiers=pre_modifiers,
        pre_steps=pre_steps,
        post_steps=[
            SingleEndFilter(predicate=TooManyN(0, assay)),
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

"""Console script for the native (Rust) edgelist step.

Build an edgelist Parquet file straight from raw FASTQ reads using the
``pixelator_core.run_edgelist`` binding. Marker lookup and UMI encoding
happen inline, so this skips the separate amplicon / demux / collapse stages.

Copyright © 2025 Pixelgen Technologies AB.
"""

import logging
from pathlib import Path, PurePath

import click
from pixelator_core import run_edgelist

from pixelator.common.utils import (
    create_output_stage_dir,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.pna.cli.common import (
    design_option,
    output_option,
    panel_option,
    threads_option,
)
from pixelator.pna.config import load_antibody_panels, pna_config
from pixelator.pna.edgelist.report import EdgelistSampleReport
from pixelator.pna.utils import clean_suffixes, get_read_sample_name

logger = logging.getLogger(__name__)


@click.command(
    "edgelist",
    short_help="Build an edgelist directly from raw FASTQ reads.",
    help=(
        "Build an edgelist Parquet file straight from raw FASTQ reads using the "
        "native Rust extractor. Provide both --read-forward and --read-reverse for "
        "paired-end data, or exactly one of them for single-end data."
    ),
    options_metavar="<options>",
)
@click.option(
    "--read-forward",
    "read_forward",
    default=None,
    type=click.Path(exists=True),
    help="Path to the gzipped forward-strand FASTQ file (R1).",
)
@click.option(
    "--read-reverse",
    "read_reverse",
    default=None,
    type=click.Path(exists=True),
    help="Path to the gzipped reverse-strand FASTQ file (R2).",
)
@click.option(
    "--sample-name",
    default=None,
    show_default=False,
    type=click.STRING,
    help=(
        "Sample id written into the report JSON. Default is derived from the "
        "input FASTQ basename. Output filenames always follow the input FASTQ "
        "(minus extension, and minus the R1/R2 suffix for paired reads)."
    ),
)
@click.option(
    "--mismatch-threshold",
    default=0,
    show_default=True,
    type=click.IntRange(0, 3),
    help="Allow this many mismatches when matching marker barcodes.",
)
@click.option(
    "--min-avg-quality",
    default=20,
    show_default=True,
    type=click.IntRange(0, 93),
    help="Drop reads whose average quality in marker and UMI regions is below this.",
)
@click.option(
    "--max-umi-base-freq",
    default=0.8,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help=(
        "Drop low-complexity UMIs where one base makes up more than this "
        "fraction of the UMI."
    ),
)
@click.option(
    "--lbs-overlap-max-edits",
    default=6,
    show_default=True,
    type=click.IntRange(0, 32),
    help=(
        "Drop reads whose UMI aligns to any substring of an assay LBS (or its "
        "reverse complement) within this many edits (in Levenshtein distance)."
    ),
)
@click.option(
    "--save-dropped-reads",
    is_flag=True,
    default=False,
    help=(
        "Also write every read that does not make it into the edgelist, with the "
        "reason it was dropped, to <sample>.dropped_reads.parquet."
    ),
)
@design_option
@panel_option
@output_option
@threads_option
@click.pass_context
@timer
def edgelist(
    ctx,
    read_forward: str | None,
    read_reverse: str | None,
    sample_name: str | None,
    mismatch_threshold: int,
    min_avg_quality: int,
    max_umi_base_freq: float,
    lbs_overlap_max_edits: int,
    save_dropped_reads: bool,
    design: str,
    panel: tuple[str, ...],
    output: str,
    threads: int,
):
    """Build an edgelist Parquet file directly from raw FASTQ reads."""
    if read_forward is not None:
        primary_read = read_forward
    elif read_reverse is not None:
        primary_read = read_reverse
    else:
        raise click.UsageError(
            "At least one of --read-forward / --read-reverse must be provided."
        )

    # Respect the global --cores when --threads is left at its default.
    threads = threads if threads and threads > 0 else ctx.obj["CORES"]

    log_step_start(
        "edgelist",
        read_forward=read_forward,
        read_reverse=read_reverse,
        output=output,
        design=design,
        panel=panel,
        mismatch_threshold=mismatch_threshold,
        min_avg_quality=min_avg_quality,
        max_umi_base_freq=max_umi_base_freq,
        lbs_overlap_max_edits=lbs_overlap_max_edits,
        save_dropped_reads=save_dropped_reads,
    )

    inputs = [Path(p) for p in (read_forward, read_reverse) if p is not None]
    sanity_check_inputs(inputs, allowed_extensions=("fastq.gz", "fq.gz"))

    edgelist_output = create_output_stage_dir(output, "edgelist")

    paired = read_forward and read_reverse
    output_stem = (
        get_read_sample_name(primary_read)
        if paired
        else clean_suffixes(PurePath(primary_read)).name
    )
    sample_name = sample_name or output_stem

    write_parameters_file(
        ctx,
        edgelist_output / f"{output_stem}.meta.json",
        command_path="pixelator single-cell-pna edgelist",
    )

    logger.info(
        "Building edgelist for %s with %s threads",
        ", ".join(str(p) for p in inputs),
        threads,
    )

    assay = pna_config.get_assay(design)
    if assay is None:
        raise click.UsageError(f"Could not resolve an assay for design '{design}'.")

    panel_obj = load_antibody_panels(pna_config, panel)

    stats = run_edgelist(
        str(read_forward) if read_forward is not None else None,
        str(read_reverse) if read_reverse is not None else None,
        output_stem,
        assay,
        panel_obj,
        str(edgelist_output),
        threads,
        mismatch_threshold=mismatch_threshold,
        min_avg_quality=min_avg_quality,
        max_umi_base_freq=max_umi_base_freq,
        lbs_overlap_max_edits=lbs_overlap_max_edits,
        save_dropped_reads=save_dropped_reads,
    )

    logger.info(
        "Wrote edgelist to %s",
        edgelist_output / f"{output_stem}.parquet",
    )
    if save_dropped_reads:
        logger.info(
            "Wrote dropped reads to %s",
            edgelist_output / f"{output_stem}.dropped_reads.parquet",
        )

    report = EdgelistSampleReport(
        product_id="single-cell-pna",
        report_type="edgelist",
        sample_id=sample_name,
        **stats,
    )

    report.write_json_file(
        edgelist_output / f"{output_stem}.report.json",
        indent=4,
    )

    if report.output_reads < report.input_reads / 2:
        raise click.ClickException(">50% of reads were discarded. Check panel file.")

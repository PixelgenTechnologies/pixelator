"""Console script for pixelator pna amplicon.

Copyright © 2024 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from pixelator.common.utils import (
    create_output_stage_dir,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.pna.amplicon import amplicon_fastq
from pixelator.pna.amplicon.report import AmpliconSampleReport
from pixelator.pna.cli.common import (
    design_option,
    logger,
    output_option,
    threads_option,
)
from pixelator.pna.config import pna_config
from pixelator.pna.utils import get_read_sample_name, is_read_file


@click.command(
    "amplicon",
    short_help=("process diverse raw pixel data (FASTQ) formats into common amplicon"),
    options_metavar="<options>",
)
@click.argument(
    "fastq_1",
    nargs=1,
    required=True,
    type=click.Path(exists=True),
    metavar="FASTQ_1",
)
@click.argument(
    "fastq_2",
    nargs=1,
    required=False,
    type=click.Path(exists=True),
    metavar="FASTQ_2",
)
@output_option
@design_option
@click.option(
    "--sample-name",
    default=None,
    show_default=False,
    type=click.STRING,
    help=(
        "Override the basename of the output fastq file. "
        "Default is the basename of the first input file "
        "without extension and read 1 identifier."
    ),
)
@click.option(
    "--mismatches",
    default=0.1,
    required=False,
    type=click.FloatRange(0.0, 0.9),
    show_default=True,
    help="The number of mismatches allowed (given as fraction)",
)
@click.option(
    "--remove-polyg",
    required=False,
    is_flag=True,
    help="Remove PolyG sequences (length of 10 or more)",
)
@click.option(
    "--quality-cutoff",
    required=False,
    default=20,
    help="Remove bases from the tail with a Phred score lower then the cutoff value.",
)
@click.option(
    "--skip-input-checks",
    default=False,
    is_flag=True,
    type=click.BOOL,
    help="Skip all check on the filename of input fastq files.",
)
@click.option(
    "--force-run",
    default=False,
    is_flag=True,
    type=click.BOOL,
    help=(
        "If more than 50% of reads are filtered, this indicates some serious problem with the input data and the amplicon command"
        "will fail with an exit status of 1. Setting this flag will force an exit status of 0."
    ),
)
@threads_option
@click.pass_context
@timer
def amplicon(
    ctx,
    fastq_1: Path,
    fastq_2: Path | None,
    sample_name: str | None,
    output: str,
    design: str,
    mismatches: int,
    remove_polyg: bool,
    quality_cutoff: int,
    skip_input_checks: bool,
    force_run: bool,
    threads: int,
):
    """Process diverse raw pixel data (FASTQ) formats into common amplicon."""
    # log input parameters

    error_level = logging.WARNING if skip_input_checks else logging.ERROR

    log_step_start(
        "amplicon",
        fastq1=fastq_1,
        fastq_2=fastq_2,
        output=output,
        design=design,
        remove_polyg=remove_polyg,
        quality_cutoff=quality_cutoff,
        skip_input_checks=skip_input_checks,
        force_run=force_run,
    )

    fastq_inputs = [Path(fastq_1)]
    if fastq_2:
        fastq_inputs.append(Path(fastq_2))
    # some basic sanity check on the input files
    sanity_check_inputs(
        fastq_inputs,
        allowed_extensions=("fastq.gz", "fq.gz", "fastq", "fq", "fastq.zst", "fq.zst"),
    )

    # create output folder if it does not exist
    amplicon_output = create_output_stage_dir(output, "amplicon")
    r1_sample_name = get_read_sample_name(fastq_1)

    # Some checks on the input files
    # - check if there are read 1 and read2 identifiers in the filename
    # - check if the sample name is the same for read1 and read2
    # no need to check this if only one fastq file is given.
    if fastq_2 is not None:
        if not is_read_file(fastq_1, "r1"):
            msg = "Read 1 file does not contain a recognised read 1 suffix."
            logger.log(level=error_level, msg=msg)
            if not skip_input_checks:
                ctx.exit(1)

        if fastq_2 and not is_read_file(fastq_2, "r2"):
            msg = "Read 2 file does not contain a recognised read 2 suffix."
            logger.log(level=error_level, msg=msg)
            if not skip_input_checks:
                ctx.exit(1)

        r2_sample_name = get_read_sample_name(fastq_2) if fastq_2 else None

        if r1_sample_name != r2_sample_name:
            msg = (
                f"The sample name for read1 and read2 is different:\n"
                f'"{r1_sample_name}" vs "{r2_sample_name}"\n'
                "Did you pass the correct files?"
            )
            logger.log(level=error_level, msg=msg)
            if not skip_input_checks:
                ctx.exit(1)

    sample_name = sample_name or r1_sample_name
    output_file = amplicon_output / f"{sample_name}.amplicon.fq.zst"
    json_file = amplicon_output / f"{sample_name}.report.json"

    write_parameters_file(
        ctx,
        amplicon_output / f"{sample_name}.meta.json",
        command_path="pixelator single-cell-pna amplicon",
    )

    msg = f"Creating amplicon for {','.join(str(p) for p in fastq_inputs)}"
    logger.info(msg)

    assay = pna_config.get_assay(design)
    if assay is None:
        msg = f"Could not find assay design with name '{design}'"
        logger.error(msg)
        return ctx.exit(1)

    stats = amplicon_fastq(
        inputs=fastq_inputs,
        assay=assay,
        output=output_file,
        mismatches=mismatches,
        poly_g_trimming=remove_polyg,
        quality_cutoff=quality_cutoff,
        threads=threads,
    )

    report = AmpliconSampleReport(
        sample_id=sample_name, product_id="single-cell-pna", **stats.as_dict()
    )

    report.write_json_file(json_file, indent=4)

    if report.fraction_discarded_reads > 0.5:
        logger.error(
            "The number of reads in the output file is less than half of the input file. "
            "This may indicate a problem with the amplicon design."
        )
        if not force_run:
            ctx.exit(1)

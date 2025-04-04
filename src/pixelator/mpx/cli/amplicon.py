"""Console script for pixelator (amplicon)

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging

import click

from pixelator.common.utils import (
    create_output_stage_dir,
    get_extension,
    get_read_sample_name,
    is_read_file,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.mpx.amplicon import amplicon_fastq
from pixelator.mpx.cli.common import (
    design_option,
    logger,
    output_option,
)


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
    "--skip-input-checks",
    default=False,
    is_flag=True,
    type=click.BOOL,
    help="Skip all check on the filename of input fastq files.",
)
@output_option
@design_option
@click.pass_context
@timer
def amplicon(
    ctx,
    fastq_1: str,
    fastq_2: str | None,
    sample_name: str | None,
    skip_input_checks: bool,
    output: str,
    design: str,
):
    """
    Process diverse raw pixel data (FASTQ) formats into common amplicon
    """
    # log input parameters

    error_level = logging.WARNING if skip_input_checks else logging.ERROR

    log_step_start(
        "amplicon",
        fastq1=fastq_1,
        fastq_2=fastq_2,
        output=output,
        design=design,
    )

    # some basic sanity check on the input files
    fastq_inputs = [fastq_1] + [fastq_2] if fastq_2 else []
    sanity_check_inputs(fastq_inputs, allowed_extensions=("fastq.gz", "fq.gz"))

    # create output folder if it does not exist
    amplicon_output = create_output_stage_dir(output, "amplicon")

    # Some checks on the input files
    # - check if there are read 1 and read2 identifiers in the filename
    # - check if the sample name is the same for read1 and read2
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

    r1_sample_name = get_read_sample_name(fastq_1)
    r2_sample_name = get_read_sample_name(fastq_2) if fastq_2 else None

    if fastq_2 and r1_sample_name != r2_sample_name:
        msg = (
            f"The sample name for read1 and read2 is different:\n"
            f'"{r1_sample_name}" vs "{r2_sample_name}"\n'
            "Did you pass the correct files?"
        )
        logger.log(level=error_level, msg=msg)
        if not skip_input_checks:
            ctx.exit(1)

    sample_name = sample_name or r1_sample_name
    extension = get_extension(fastq_1)
    output_file = amplicon_output / f"{sample_name}.merged.{extension}"
    json_file = amplicon_output / f"{sample_name}.report.json"

    write_parameters_file(
        ctx,
        amplicon_output / f"{sample_name}.meta.json",
        command_path="pixelator single-cell-mpx amplicon",
    )

    msg = f"Creating amplicon for {','.join(str(p) for p in fastq_inputs)}"
    logger.info(msg)

    amplicon_fastq(
        inputs=fastq_inputs,
        design=design,
        metrics=json_file,
        sample_id=sample_name,
        output=str(output_file),
    )

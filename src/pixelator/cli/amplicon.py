"""
Console script for pixelator (amplicon)

Copyright (c) 2022 Pixelgen Technologies AB.
"""
from concurrent import futures
from typing import List

import click

from pixelator.cli.common import design_option, logger, output_option
from pixelator.amplicon import amplicon_fastq
from pixelator.utils import (
    click_echo,
    create_output_stage_dir,
    get_extension,
    group_input_reads,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)


@click.command(
    "amplicon",
    short_help=("process diverse raw pixel data (FASTQ) formats into common amplicon"),
    options_metavar="<options>",
)
@click.argument(
    "input_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True),
    metavar="FASTQ_FILES",
)
@click.option(
    "--input1-pattern",
    default="_R1",
    required=False,
    type=click.STRING,
    show_default=True,
    help="The string pattern to use to identify forward (R1) files",
)
@click.option(
    "--input2-pattern",
    default="_R2",
    required=False,
    type=click.STRING,
    show_default=True,
    help="The string pattern to use to identify reverse (R2) files",
)
@output_option
@design_option
@click.pass_context
@timer
def amplicon(
    ctx,
    input_files: List[str],
    input1_pattern: str,
    input2_pattern: str,
    output: str,
    design: str,
):
    """
    Process diverse raw pixel data (FASTQ) formats into common amplicon
    """
    # log input parameters
    log_step_start(
        "amplicon",
        input_files=input_files,
        output=output,
        input1_pattern=input1_pattern,
        input2_pattern=input2_pattern,
        design=design,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files, allowed_extensions=("fastq.gz", "fq.gz"))

    # create output folder if it does not exist
    amplicon_output = create_output_stage_dir(output, "amplicon")

    # group input file by sample id and order reads by R1 and R2
    grouped_sorted_inputs = group_input_reads(
        input_files, input1_pattern, input2_pattern
    )

    # run amplicon using parallel processing
    with futures.ProcessPoolExecutor(max_workers=ctx.obj["CORES"]) as executor:
        jobs = []
        for k, v in grouped_sorted_inputs.items():
            extension = get_extension(v[0])
            output_file = amplicon_output / f"{k}.merged.{extension}"
            json_file = amplicon_output / f"{k}.report.json"

            write_parameters_file(
                ctx,
                amplicon_output / f"{k}.meta.json",
                command_path="pixelator single-cell amplicon",
            )

            if len(v) > 2:
                msg = "Found more files than needed for concatenating fastq files"
                logger.error(msg)
                raise RuntimeError(msg)

            msg = f"Concatenating {','.join(str(p) for p in v)}"
            click_echo(msg, multiline=False)
            logger.info(msg)

            jobs.append(
                executor.submit(
                    amplicon_fastq,
                    inputs=v,
                    design=design,
                    metrics=json_file,
                    output=str(output_file),
                )
            )

        for job in futures.as_completed(jobs):
            exc = job.exception()
            if exc is not None:
                raise exc

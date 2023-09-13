"""
Console script for pixelator (preqc)

Copyright (c) 2022 Pixelgen Technologies AB.
"""

from concurrent import futures
from shutil import which

import click

from pixelator.cli.common import (
    design_option,
    logger,
    output_option,
)
from pixelator.config import config
from pixelator.qc import qc_fastq
from pixelator.utils import (
    click_echo,
    create_output_stage_dir,
    get_extension,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)


@click.command(
    "preqc",
    short_help=(
        "process raw pixel data (FASTQ) to perform QC,"
        " filtering, trimming and remove duplicates"
    ),
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
    "--trim-front",
    default=0,
    required=False,
    type=click.INT,
    show_default=True,
    help="Trim N bases from the front of the reads",
)
@click.option(
    "--trim-tail",
    default=0,
    required=False,
    type=click.INT,
    show_default=True,
    help="Trim N bases from the tail of the reads",
)
@click.option(
    "--max-length",
    default=None,
    required=False,
    type=click.INT,
    show_default=False,
    help=(
        "The maximum length (bases) of a read (longer reads will be trimmed off).\n"
        "If you set this argument it will overrule the value from the chosen design"
    ),
)
@click.option(
    "--min-length",
    default=None,
    required=False,
    type=click.INT,
    show_default=False,
    help=(
        "The minimum length (bases) of a read (shorter reads will be discarded).\n"
        "If you set this argument it will overrule the value from the chosen design"
    ),
)
@click.option(
    "--max-n-bases",
    default=0,
    required=False,
    type=click.INT,
    show_default=True,
    help=(
        "The maximum number of Ns allowed in a read (default of 0 means "
        "any reads with N in it will be filtered out)"
    ),
)
@click.option(
    "--avg-qual",
    default=20,
    required=False,
    type=click.INT,
    show_default=True,
    help="Minimum avg. quality a read must have (0 will disable the filter)",
)
@click.option(
    "--dedup",
    required=False,
    is_flag=True,
    help="Remove duplicated reads (exact same sequence)",
)
@click.option(
    "--remove-polyg",
    required=False,
    is_flag=True,
    help="Remove PolyG sequences (length of 10 or more)",
)
@output_option
@design_option
@click.pass_context
@timer
def preqc(
    ctx,
    input_files,
    trim_front,
    trim_tail,
    max_length,
    min_length,
    max_n_bases,
    avg_qual,
    dedup,
    remove_polyg,
    output,
    design,
):
    """
    Process raw Molecular Pixelation data (FASTQ) to perform QC, filtering,
    trimming and removal of duplicates
    """
    # log input parameters
    log_step_start(
        "fastp",
        input_files=input_files,
        output=output,
        trim_front=trim_front,
        trim_tail=trim_tail,
        max_length=max_length,
        min_length=min_length,
        max_n_bases=max_n_bases,
        avg_qual=avg_qual,
        dedup=dedup,
        remove_polyg=remove_polyg,
        design=design,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(
        input_files=input_files, allowed_extensions=("fastq.gz", "fq.gz")
    )

    # check that fastp is installed
    if which("fastp") is None:
        click.ClickException("Fastp is not installed in this system")

    # update arguments from config file (CLI arguments have priority)
    try:
        amplicon = config.get_assay(design).get_region_by_id("amplicon")
        conf_min_length, conf_max_length = amplicon.get_len()
        min_length = conf_min_length if min_length is None else min_length
        max_length = conf_max_length if max_length is None else max_length
    except KeyError as exc:
        raise click.ClickException(f"Parsing attribute from config file {exc}")

    # create output folder if it does not exist
    preqc_output = create_output_stage_dir(output, "preqc")

    # run fastq (pre QC and filtering) using parallel processing
    with futures.ProcessPoolExecutor(max_workers=ctx.obj["CORES"]) as executor:
        jobs = []
        for fastq_file in input_files:
            msg = f"Processing {fastq_file} with fastp"
            click_echo(msg, multiline=False)
            logger.info(msg)

            clean_name = get_sample_name(fastq_file)
            extension = get_extension(fastq_file)
            output_file = preqc_output / f"{clean_name}.processed.{extension}"
            failed_file = preqc_output / f"{clean_name}.failed.{extension}"
            html_file = preqc_output / f"{clean_name}.report.html"
            json_file = preqc_output / f"{clean_name}.report.json"

            write_parameters_file(
                ctx,
                preqc_output / f"{clean_name}.meta.json",
                command_path="pixelator single-cell preqc",
            )

            jobs.append(
                executor.submit(
                    qc_fastq,
                    input=str(fastq_file),
                    output=str(output_file),
                    failed=str(failed_file),
                    report=str(html_file),
                    metrics=str(json_file),
                    design=design,
                    n_limit=max_n_bases,
                    trim_front=trim_front,
                    trim_tail=trim_tail,
                    min_length=min_length,
                    max_length=max_length,
                    # fastp developers recommend to keep this low to avoid I/O overhead
                    threads=2,
                    avg_qual=avg_qual,
                    dedup=dedup,
                    remove_polyg=remove_polyg,
                    verbose=ctx.obj["VERBOSE"],
                )
            )

        for job in futures.as_completed(jobs):
            if job.exception() is not None:
                raise job.exception()

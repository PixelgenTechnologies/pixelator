"""
Console script for pixelator (adapterqc)

Copyright (c) 2022 Pixelgen Technologies AB.
"""

from concurrent import futures

import click

from pixelator.cli.common import design_option, logger, output_option
from pixelator.qc import adapter_qc_fastq
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
    "adapterqc",
    short_help=(
        "process pixel data (FASTQ) to check for the presence of PBS1/2 sequences"
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
    "--mismatches",
    default=0.1,
    required=False,
    type=click.FloatRange(0.0, 0.9),
    show_default=True,
    help="The number of mismatches allowed (in percentage)",
)
@click.option(
    "--pbs1",
    default=None,
    required=False,
    type=click.STRING,
    show_default=False,
    help=(
        "The PBS1 sequence that must be present in the reads.\n"
        "If you set this argument it will overrule the value from the chosen design"
    ),
)
@click.option(
    "--pbs2",
    default=None,
    required=False,
    type=click.STRING,
    show_default=False,
    help=(
        "The PBS2 sequence that must be present in the reads.\n"
        "If you set this argument it will overrule the value from the chosen design"
    ),
)
@output_option
@design_option
@click.pass_context
@timer
def adapterqc(
    ctx,
    input_files,
    mismatches,
    pbs1,
    pbs2,
    output,
    design,
):
    """
    Process Molecular Pixelation data (FASTQ) to check for the presence of
    PBS1/2 sequences
    """
    from pixelator.config import config

    # log input parameters
    log_step_start(
        "adapterqc",
        input_files=input_files,
        output=output,
        mismatches=mismatches,
        pbs1=pbs1,
        pbs2=pbs2,
        design=design,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files, allowed_extensions=("fastq.gz", "fq.gz"))

    # update arguments from config file (CLI arguments have priority)
    try:
        amplicon = config.get_assay(design).get_region_by_id("amplicon")
        if pbs1 is None:
            pbs1 = amplicon.get_region_by_id("pbs-1").get_sequence()
        if pbs2 is None:
            pbs2 = amplicon.get_region_by_id("pbs-2").get_sequence()
    except KeyError as exc:
        raise click.ClickException(f"Parsing attribute from config file {exc}")

    # create output folder if it does not exist
    adapterqc_output = create_output_stage_dir(output, "adapterqc")

    # run cutadapt (adapter mode) using parallel processing
    with futures.ProcessPoolExecutor(max_workers=ctx.obj["CORES"]) as executor:
        jobs = []
        for fastq_file in input_files:
            msg = f"Processing {fastq_file} with cutadapt (adapter mode)"
            click_echo(msg, multiline=False)
            logger.info(msg)

            clean_name = get_sample_name(fastq_file)
            extension = get_extension(fastq_file)
            output_file = adapterqc_output / f"{clean_name}.processed.{extension}"
            failed_file = adapterqc_output / f"{clean_name}.failed.{extension}"
            json_file = adapterqc_output / f"{clean_name}.report.json"

            write_parameters_file(
                ctx,
                adapterqc_output / f"{clean_name}.meta.json",
                command_path="pixelator single-cell adapterqc",
            )

            jobs.append(
                executor.submit(
                    adapter_qc_fastq,
                    input=fastq_file,
                    output=str(output_file),
                    failed=str(failed_file),
                    report=str(json_file),
                    mismatches=mismatches,
                    pbs1=pbs1,
                    pbs2=pbs2,
                    cores=0,  # cutadapt will choose the optimal number
                    verbose=ctx.obj["VERBOSE"],
                )
            )

        for job in futures.as_completed(jobs):
            if job.exception() is not None:
                raise job.exception()

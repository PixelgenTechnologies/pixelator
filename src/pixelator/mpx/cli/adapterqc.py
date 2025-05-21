"""Console script for pixelator (adapterqc).

Copyright © 2022 Pixelgen Technologies AB.
"""

import click

from pixelator.common.utils import (
    create_output_stage_dir,
    get_extension,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.mpx.cli.common import design_option, logger, output_option
from pixelator.mpx.qc import adapter_qc_fastq


@click.command(
    "adapterqc",
    short_help=(
        "process pixel data (FASTQ) to check for the presence of PBS1/2 sequences"
    ),
    options_metavar="<options>",
)
@click.argument(
    "fastq_file",
    nargs=1,
    required=True,
    type=click.Path(exists=True),
    metavar="FASTQ",
)
@click.option(
    "--mismatches",
    default=0.1,
    required=False,
    type=click.FloatRange(0.0, 0.9),
    show_default=True,
    help="The number of mismatches allowed (in percentage)",
)
@output_option
@design_option
@click.pass_context
@timer
def adapterqc(
    ctx,
    fastq_file,
    mismatches,
    output,
    design,
):
    """Check for the presence of PBS1/2 sequences in FASTQ input files."""
    from pixelator.mpx.config.config_instance import config

    input_files = [fastq_file]
    # log input parameters
    log_step_start(
        "adapterqc",
        input_files=input_files,
        output=output,
        mismatches=mismatches,
        design=design,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files, allowed_extensions=("fastq.gz", "fq.gz"))

    # update arguments from config file (CLI arguments have priority)
    try:
        amplicon = config.get_assay(design).get_region_by_id("amplicon")
        pbs1 = amplicon.get_region_by_id("pbs-1").get_sequence()
        pbs2 = amplicon.get_region_by_id("pbs-2").get_sequence()
    except KeyError as exc:
        raise click.ClickException(f"Parsing attribute from config file {exc}")

    # create output folder if it does not exist
    adapterqc_output = create_output_stage_dir(output, "adapterqc")

    # run cutadapt (adapter mode) using parallel processing
    msg = f"Processing {fastq_file} with cutadapt (adapter mode)"
    logger.info(msg)

    clean_name = get_sample_name(fastq_file)
    extension = get_extension(fastq_file)
    output_file = adapterqc_output / f"{clean_name}.processed.{extension}"
    failed_file = adapterqc_output / f"{clean_name}.failed.{extension}"
    json_file = adapterqc_output / f"{clean_name}.report.json"

    write_parameters_file(
        ctx,
        adapterqc_output / f"{clean_name}.meta.json",
        command_path="pixelator single-cell-mpx adapterqc",
    )

    adapter_qc_fastq(
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

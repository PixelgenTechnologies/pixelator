"""
Console script for pixelator (demux)

Copyright (c) 2022 Pixelgen Technologies AB.
"""

from concurrent import futures

import click

from pixelator.cli.common import design_option, logger, output_option
from pixelator.config import config
from pixelator.config.panel import load_antibody_panel
from pixelator.demux import demux_fastq
from pixelator.utils import (
    build_barcodes_file,
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
    "demux",
    short_help="demultiplex pixel data (FASTQ) to generate one file per antibody",
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
    "--min-length",
    default=None,
    required=False,
    type=click.INT,
    show_default=False,
    help=(
        "The minimum length of the barcode that must overlap when matching.\n"
        "If you set this argument it will overrule the value from the chosen design"
    ),
)
@click.option(
    "--panel",
    required=True,
    type=str,
    help=(
        "A key of a panel file in the config,"
        " or a csv file with the antibody panel conjugations"
    ),
)
@click.option(
    "--anchored",
    default=None,
    type=click.BOOL,
    help=(
        "Enforce the barcodes to be anchored (at the end of the read).\n "
        "(default: use value determined by --design)."
    ),
)
@click.option(
    "--rev-complement",
    default=None,
    type=click.BOOL,
    help=(
        "Use the reverse complement of the barcodes sequences.\n "
        "(default: use value determined by --design)."
    ),
)
@output_option
@design_option
@click.pass_context
@timer
def demux(
    ctx,
    input_files,
    mismatches,
    min_length,
    panel,
    output,
    design,
    anchored,
    rev_complement,
):
    """
    Demultiplex Molecular Pixelation data (FASTQ) to generate one file per antibody
    """
    # log input parameters
    log_step_start(
        "demux",
        input_files=input_files,
        output=output,
        mismatches=mismatches,
        panel=panel,
        min_length=min_length,
        anchored=anchored,
        rev_complement=rev_complement,
        design=design,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files, allowed_extensions=("fastq.gz", "fq.gz"))

    # update arguments from config file (CLI arguments have priority)
    try:
        amplicon = config.get_assay(design).get_region_by_id("amplicon")
        barcode = amplicon.get_region_by_id("bc")

        # Load default options from the design
        if min_length is None:
            min_length = barcode.min_len
        if anchored is None:
            anchored = barcode.data.get("anchored")
        if rev_complement is None:
            rev_complement = barcode.data.get("reverse_complement")

    except KeyError as err:
        raise click.ClickException(f"Parsing attribute from config file {str(err)}")

    # create output folder if it does not exist
    demux_output = create_output_stage_dir(output, "demux")

    # load marker panel
    panel = load_antibody_panel(config, panel)

    # build barcodes (fasta) on-the-fly
    barcodes = build_barcodes_file(
        panel=panel, anchored=anchored, rev_complement=rev_complement
    )

    # run cutadapt (demux mode) using parallel processing
    with futures.ProcessPoolExecutor(max_workers=ctx.obj["CORES"]) as executor:
        jobs = []
        for fastq_file in input_files:
            msg = f"Processing {fastq_file} with cutadapt (demux mode)"
            click_echo(msg, multiline=False)
            logger.info(msg)

            name = get_sample_name(fastq_file)
            extension = get_extension(fastq_file)
            output_file = demux_output / f"{name}.processed-{{name}}.{extension}"
            failed_file = demux_output / f"{name}.failed.{extension}"
            json_file = demux_output / f"{name}.report.json"

            write_parameters_file(
                ctx,
                demux_output / f"{name}.meta.json",
                command_path="pixelator single-cell demux",
            )

            jobs.append(
                executor.submit(
                    demux_fastq,
                    input=str(fastq_file),
                    output=str(output_file),
                    failed=str(failed_file),
                    report=str(json_file),
                    panel=panel,
                    mismatches=mismatches,
                    barcodes=barcodes,
                    min_length=min_length,
                    cores=0,  # cutadapt will choose the optimal number
                    verbose=ctx.obj["VERBOSE"],
                )
            )

        for job in futures.as_completed(jobs):
            if job.exception() is not None:
                raise job.exception()

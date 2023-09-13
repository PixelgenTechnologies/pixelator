"""
Console script for pixelator

Copyright (c) 2022 Pixelgen Technologies AB.
"""

import logging
from pathlib import Path

import click

from pixelator.cli.common import output_option
from pixelator.report import make_report
from pixelator.utils import click_echo, create_output_stage_dir, log_step_start, timer

logger = logging.getLogger(__name__)


@click.command(
    "report",
    short_help=("create a summary web report for all the samples"),
    options_metavar="<options>",
)
@click.argument(
    "input_folder",
    required=True,
    type=click.Path(exists=True),
    metavar="FOLDER",
)
@click.option(
    "--panel",
    required=False,
    type=str,
    help=(
        "A key of a panel file in the config,"
        " or a csv file with the antibody panel conjugations"
    ),
)
@click.option(
    "--metadata",
    required=False,
    default=None,
    type=click.Path(exists=True),
    help="Path to file with the samples metadata",
)
@output_option
@click.pass_context
@timer
def report(
    ctx,
    input_folder,
    panel,
    metadata,
    output,
):
    """
    Create a summary web report for all the samples (must complete all steps first)
    """
    # check that the input folder contains all the steps
    # (sub-folders preqc, adapterqc, demux, collapse, cluster and annotate)
    required_sub_folders = [
        "preqc",
        "adapterqc",
        "demux",
        "collapse",
        "graph",
        "annotate",
    ]
    if not all(
        map(
            lambda dir: (Path(input_folder) / dir).is_dir(),
            required_sub_folders,
        )
    ):
        raise click.ClickException(f"The structure of {input_folder} is not valid")

    # create output folder if it does not exist
    report_output = create_output_stage_dir(output, "report")

    # log input parameters
    log_step_start(
        "report",
        input_folder=input_folder,
        output=str(report_output),
        metadata=metadata,
    )

    # create html reports
    click_echo(f"Creating report for data present in {input_folder}")

    make_report(
        input_path=input_folder,
        output_path=str(report_output),
        panel=panel,
        metadata=metadata,
        verbose=ctx.obj["VERBOSE"],
    )

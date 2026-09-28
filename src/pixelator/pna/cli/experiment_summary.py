"""Console script for the single-cell-pna experiment summary.

Copyright © 2026 Pixelgen Technologies AB.
"""

import logging
from pathlib import Path

import click

from pixelator.common.utils import create_output_stage_dir, log_step_start, timer
from pixelator.pna.cli.common import output_option
from pixelator.pna.report.experiment_summary import write_experiment_summary

logger = logging.getLogger(__name__)


@click.command(
    "experiment-summary",
    short_help="summarize which pipeline steps passed or failed for each sample",
    options_metavar="<options>",
)
@click.argument(
    "input_folder",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    metavar="FOLDER",
)
@output_option
@timer
def experiment_summary(input_folder: Path, output: str):
    """Write an experiment summary for every sample in a pixelator output folder.

    Samples that failed for a data reason are included, together with the
    statistics gathered before the failure and the null-file reason.
    """
    summary_output = create_output_stage_dir(output, "experiment_summary")
    log_step_start(
        "experiment-summary",
        input_folder=str(input_folder),
        output=str(summary_output),
    )
    destination = summary_output / "experiment_summary.json"
    summary = write_experiment_summary(input_folder, destination)
    logger.info(
        "Wrote experiment summary for %d sample(s) to %s",
        len(summary.samples),
        destination,
    )

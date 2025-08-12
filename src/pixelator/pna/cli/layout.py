"""Console script for pixelator layout creation.

Copyright © 2024 Pixelgen Technologies AB.
"""

import logging
import sys
from pathlib import Path

import click

from pixelator.common.utils import (
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.pna.analysis_engine import (
    AnalysisManager,
    LoggingSetup,
)
from pixelator.pna.cli.common import output_option
from pixelator.pna.layout import CreateLayout
from pixelator.pna.pixeldataset import PxlFile, read
from pixelator.pna.report.common import PixelatorPNAWorkdir
from pixelator.pna.report.models.layout import LayoutSampleReport

logger = logging.getLogger(__name__)


@click.command(
    "layout",
    short_help="compute graph layouts that can be used to visualize components",
    options_metavar="<options>",
)
@click.argument(
    "pxl_file",
    nargs=1,
    required=True,
    type=click.Path(exists=True),
    metavar="PXL",
)
@click.option(
    "--layout-algorithm",
    required=False,
    multiple=True,
    default=["wpmds_3d"],
    help="Select a layout algorithm to use. This can be specified multiple times to compute multiple layouts. Default: pmds_3d",
    type=click.Choice(["pmds_3d", "wpmds_3d"]),
)
@click.option(
    "--pmds-pivots",
    default=50,
    required=False,
    type=click.IntRange(50, None),
    show_default=True,
    help="Number of pivots to use for the PMDS layout algorithm. Default: 50. More give better results, but increase computation times.",
)
@click.option(
    "--wpmds-k",
    default=3,
    required=False,
    type=click.IntRange(1, 10),
    show_default=True,
    help="The window size used when computing probability weights to the wpmds layout method. Only used when the wpmds layout method is selected. Default: 3.",
)
@output_option
@click.pass_context
@timer
def layout(
    ctx,
    pxl_file,
    layout_algorithm,
    pmds_pivots,
    wpmds_k,
    output,
):
    """Compute graph layouts that can be used to visualize components."""
    log_step_start(
        "layout",
        input_files=pxl_file,
        layout_algorithm=layout_algorithm,
        pmds_pivots=pmds_pivots,
        wpmds_k=wpmds_k,
        output=output,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(pxl_file, allowed_extensions="pxl")

    # create output folder if it does not exist
    workdir = PixelatorPNAWorkdir(output)
    layout_output_dir = workdir.stage_dir("layout")

    logger.info(f"Computing layout(s) for file {pxl_file}")

    clean_name = get_sample_name(pxl_file)
    write_parameters_file(
        ctx,
        layout_output_dir / f"{clean_name}.meta.json",
        command_path="pixelator single-cell-pna layout",
    )

    analysis_to_run = [
        CreateLayout(
            layout_algorithm,
        )
    ]

    pxl_file = PxlFile(Path(pxl_file))
    pxl_dataset = read(pxl_file.path)

    logging_setup = LoggingSetup.from_logger(ctx.obj.get("LOGGER"))
    analysis_manager = AnalysisManager(analysis_to_run, logging_setup=logging_setup)
    pxl_file_target = PxlFile.copy_pxl_file(
        pxl_file, layout_output_dir / f"{clean_name}.layout.pxl"
    )
    pxl_dataset = analysis_manager.execute_from_path(
        input_pxl_file_path=pxl_file.path, pxl_file_target=pxl_file_target
    )

    metrics_file = layout_output_dir / f"{clean_name}.report.json"
    report = LayoutSampleReport(
        sample_id=clean_name,
        product_id="single-cell-pna",
        report_type="layout",
    )
    report.write_json_file(metrics_file, indent=4)


if __name__ == "__main__":
    sys.exit(layout())

"""
Console script for pixelator (layout)

Copyright © 2024 Pixelgen Technologies AB.
"""

from typing import get_args

import click

from pixelator.common.graph.backends.protocol import SupportedLayoutAlgorithm
from pixelator.common.utils import (
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.mpx import read
from pixelator.mpx.cli.common import logger, output_option
from pixelator.mpx.pixeldataset.precomputed_layouts import (
    generate_precomputed_layouts_for_components,
)
from pixelator.mpx.report.common import PixelatorWorkdir
from pixelator.mpx.report.models.layout import LayoutSampleReport


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
    "--no-node-marker-counts",
    required=False,
    is_flag=True,
    default=False,
    help="Skip adding marker counts to the layout. Default: False.",
)
@click.option(
    "--layout-algorithm",
    required=False,
    multiple=True,
    default=["wpmds_3d"],
    help="Select a layout algorithm to use. This can be specified multiple times to compute multiple layouts. Default: pmds_3d",
    type=click.Choice(get_args(SupportedLayoutAlgorithm)),
)
@output_option
@click.pass_context
@timer
def layout(
    ctx,
    pxl_file,
    no_node_marker_counts,
    layout_algorithm,
    output,
):
    """
    Compute graph layouts that can be used to visualize components
    """
    log_step_start(
        "layout",
        input_files=pxl_file,
        no_node_marker_counts=no_node_marker_counts,
        layout_algorithm=layout_algorithm,
        output=output,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(pxl_file, allowed_extensions="pxl")

    # create output folder if it does not exist
    workdir = PixelatorWorkdir(output)
    layout_output_dir = workdir.stage_dir("layout")

    logger.info(f"Computing layout(s) for file {pxl_file}")

    clean_name = get_sample_name(pxl_file)
    write_parameters_file(
        ctx,
        layout_output_dir / f"{clean_name}.meta.json",
        command_path="pixelator single-cell-mpx layout",
    )

    pxl_dataset = read(pxl_file)
    pxl_dataset.precomputed_layouts = generate_precomputed_layouts_for_components(
        pxl_dataset,
        add_node_marker_counts=not no_node_marker_counts,
        layout_algorithms=layout_algorithm,
    )
    pxl_dataset.save(
        layout_output_dir / f"{clean_name}.layout.dataset.pxl", force_overwrite=True
    )

    metrics_file = layout_output_dir / f"{clean_name}.report.json"
    report = LayoutSampleReport(
        sample_id=clean_name,
    )
    report.write_json_file(metrics_file, indent=4)

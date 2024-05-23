"""Graph console script for pixelator.

Copyright © 2022 Pixelgen Technologies AB.
"""

import click

from pixelator.cli.common import logger, output_option
from pixelator.graph import connect_components
from pixelator.utils import (
    create_output_stage_dir,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)


@click.command(
    "graph",
    short_help=("compute graph, components and metrics from an edge list"),
    options_metavar="<options>",
)
@click.argument(
    "parquet_file",
    nargs=1,
    required=True,
    type=click.Path(exists=True),
    metavar="parquet",
)
@click.option(
    "--multiplet-recovery",
    required=False,
    is_flag=True,
    default=False,
    type=click.BOOL,
    help=("Activate the multiplet recovery using leiden community detection"),
)
@click.option(
    "--leiden-iterations",
    default=10,
    required=False,
    type=click.IntRange(1, 100),
    show_default=True,
    help=(
        "Number of iterations for the leiden algorithm, high values will decrease "
        "the variance of the results but increase the runtime"
    ),
)
@click.option(
    "--min-count",
    default=2,
    required=False,
    type=click.IntRange(1, 50),
    show_default=True,
    help=(
        "Discard edges (pixels) with with a count (reads) below this (use 1 to disable)"
    ),
)
@output_option
@click.pass_context
@timer
def graph(
    ctx,
    parquet_file,
    multiplet_recovery,
    leiden_iterations,
    min_count,
    output,
):
    """Compute graph, components and other metrics from an edge list."""
    # log input parameters
    input_files = [parquet_file]
    log_step_start(
        "graph",
        input_files=input_files,
        output=output,
        multiplet_recovery=multiplet_recovery,
        leiden_iterations=leiden_iterations,
        min_count=min_count,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files, allowed_extensions="parquet")

    # create output folder if it does not exist
    graph_output = create_output_stage_dir(output, "graph")

    # compute graph/components using parallel processing
    logger.info(f"Computing clusters for file {parquet_file}")

    clean_name = get_sample_name(parquet_file)
    metrics_file = graph_output / f"{clean_name}.report.json"

    write_parameters_file(
        ctx,
        graph_output / f"{clean_name}.meta.json",
        command_path="pixelator single-cell graph",
    )

    connect_components(
        input=str(parquet_file),
        output=str(graph_output),
        sample_name=clean_name,
        metrics_file=str(metrics_file),
        multiplet_recovery=multiplet_recovery,
        leiden_iterations=leiden_iterations,
        min_count=min_count,
    )

"""Graph console script for pixelator.

Copyright © 2022 Pixelgen Technologies AB.
"""

import click

from pixelator.common.utils import (
    create_output_stage_dir,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.mpx.cli.common import logger, output_option
from pixelator.mpx.graph import connect_components


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
    "--max-refinement-recursion-depth",
    default=5,
    required=False,
    type=click.IntRange(1, 100),
    show_default=True,
    help=(
        "The number of times a component can be broken down into "
        "smaller components during the multiplet recovery process."
    ),
)
@click.option(
    "--max-edges-to-split",
    default=5,
    required=False,
    type=click.IntRange(1, 100),
    show_default=True,
    help=(
        "Maximum number of edges  between the product components "
        "as a result of a component split operation during "
        "the multiplet recovery process."
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
    max_refinement_recursion_depth,
    max_edges_to_split,
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
        max_refinement_recursion_depth=max_refinement_recursion_depth,
        max_edges_to_split=max_edges_to_split,
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
        command_path="pixelator single-cell-mpx graph",
    )

    connect_components(
        input=str(parquet_file),
        output=str(graph_output),
        sample_name=clean_name,
        metrics_file=str(metrics_file),
        multiplet_recovery=multiplet_recovery,
        max_refinement_recursion_depth=max_refinement_recursion_depth,
        max_edges_to_split=max_edges_to_split,
        min_count=min_count,
    )

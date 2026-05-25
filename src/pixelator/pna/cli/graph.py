"""Graph console script for pixelator.

Copyright © 2025 Pixelgen Technologies AB.
"""

import sys
from pathlib import Path

import click

from pixelator.common.utils import (
    create_output_stage_dir,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.pna.cli.common import logger, output_option, panel_option
from pixelator.pna.config import load_antibody_panel, pna_config
from pixelator.pna.graph.community_detection import (
    RefinementOptions,
    StagedRefinementOptions,
)
from pixelator.pna.graph.component_recovery import build_pxl_file_with_components
from pixelator.pna.graph.component_recovery_utils import ConnectedComponentException
from pixelator.pna.graph.constants import MIN_PNA_COMPONENT_SIZE
from pixelator.pna.graph.report import GraphSampleReport


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
    "--edge-cycle-verification",
    required=False,
    is_flag=True,
    default=False,
    type=click.BOOL,
    help=(
        "Activate edge cycle verification to remove edges from well connected regions that are not part of cycles in the graph."
    ),
)
@click.option(
    "--initial-stage-leiden-resolution",
    default=1.0,
    required=False,
    type=click.FloatRange(0, None),
    show_default=True,
    help=(
        "The resolution parameter for the leiden algorithm at the initial stage. This should "
        "typically be set higher than the refinement stage resolution."
    ),
)
@click.option(
    "--refinement-stage-leiden-resolution",
    default=0.01,
    required=False,
    type=click.FloatRange(0, None),
    show_default=True,
    help=(
        "The resolution parameter for the leiden algorithm at the refinement stage. This should "
        "typically be set lower than the initial stage resolution."
    ),
)
@click.option(
    "--min-count",
    default=1,
    required=False,
    type=click.IntRange(1, 50),
    show_default=True,
    help=(
        "Discard edges with a read count below given value. Set to 1 to disable filtering."
    ),
)
@click.option(
    "--max-refinement-recursion-depth",
    default=5,
    required=False,
    type=click.IntRange(min=1, max=20),
    show_default=True,
    help=(
        "The maximum recursion depth for the refinement algorithm. Set to 1 to disable refinement."
    ),
)
@click.option(
    "--initial-stage-max-edges-to-remove",
    default=None,
    required=False,
    type=click.IntRange(min=1, max=None),
    show_default=True,
    help=(
        "The maximum number of edges to remove between components during the "
        "initial stage (iteration == 0) of multiplet recovery."
    ),
)
@click.option(
    "--refinement-stage-max-edges-to-remove",
    default=4,
    required=False,
    type=click.IntRange(min=1, max=None),
    show_default=True,
    help=(
        "The maximum number of edges to remove between components during the "
        "refinement stage (iteration > 0) of multiplet recovery."
    ),
)
@click.option(
    "--initial-stage-max-edges-to-remove-relative",
    default=None,
    required=False,
    type=click.FloatRange(min=1e-6, max=None),
    show_default=True,
    help=(
        "The maximum number of edges to remove between two components relative "
        "to the number of nodes in the smaller of the two when during the "
        "initial stage (iteration == 0) of multiplet recovery."
    ),
)
@click.option(
    "--refinement-stage-max-edges-to-remove-relative",
    default=None,
    required=False,
    type=click.FloatRange(min=1e-6, max=None),
    show_default=True,
    help=(
        "The maximum number of edges to remove between two components relative "
        "to the number of nodes in the smaller of the two when during the "
        "refinement stage (iteration > 0) of multiplet recovery."
    ),
)
@click.option(
    "--component-size-max-threshold",
    default=None,
    required=False,
    type=click.IntRange(min=1, max=None),
    show_default=True,
    help=(
        "Components with more nodes than this will be filtered from the output data. "
        "This is typically not needed. Setting this will disable the automatic size filtering."
    ),
)
@click.option(
    "--component-size-min-threshold",
    default=None,
    required=False,
    type=click.IntRange(min=1, max=None),
    show_default=True,
    help=(
        "Components with fewer nodes than this will be filtered from the output data. "
        "This is typically not needed. Setting this will disable the automatic size filtering."
    ),
)
@panel_option
@output_option
@click.pass_context
@timer
def graph(
    ctx,
    parquet_file,
    multiplet_recovery,
    edge_cycle_verification,
    initial_stage_leiden_resolution,
    refinement_stage_leiden_resolution,
    min_count,
    max_refinement_recursion_depth,
    initial_stage_max_edges_to_remove,
    refinement_stage_max_edges_to_remove,
    initial_stage_max_edges_to_remove_relative,
    refinement_stage_max_edges_to_remove_relative,
    component_size_max_threshold,
    component_size_min_threshold,
    panel,
    output,
):
    """Find connected components from the input molecules.

    The graph stage will attempt to identify connected components from the input molecules.
    When `--multiplet-recovery` is active we will try to break up components that are likely
    not single cells. We do so in two main stages and one optional stage.

    Main stages:
    1) Fast label propagation: Used as a graph coarsening step to reduce the size of the graph.
    2) Leiden community detection
    2.1) Initial stage: Attempt to break up the mega cluster that is formed due to random edges between cells.
    2.2) Refinement stage: Refine the connected components found in the initial stage.

    Optional stage:
    - Edge cycle verification: Remove edges, connecting highly connected nodes in the graph
    (k-core number > 1) that are not part of any short cycles in the graph. Such nodes are likely
    crossing edges connecting different components.

    After the connected components have been identified we will create a pxl file that contains
    data for all of there putative cells.

    Args:
        ctx: Ctx.
        parquet_file: Parquet file.
        multiplet_recovery: Multiplet recovery.
        edge_cycle_verification: Edge cycle verification.
        initial_stage_leiden_resolution: Initial stage leiden resolution.
        refinement_stage_leiden_resolution: Refinement stage leiden resolution.
        min_count: Min count.
        max_refinement_recursion_depth: Max refinement recursion depth.
        initial_stage_max_edges_to_remove: Initial stage max edges to remove.
        refinement_stage_max_edges_to_remove: Refinement stage max edges to remove.
        initial_stage_max_edges_to_remove_relative: Initial stage max edges to remove relative.
        refinement_stage_max_edges_to_remove_relative: Refinement stage max edges to remove relative.
        component_size_max_threshold: Component size max threshold.
        component_size_min_threshold: Component size min threshold.
        panel: Panel.
        output: Output.
    """
    # log input parameters
    input_files = [parquet_file]
    log_step_start(
        "graph",
        input_files=input_files,
        output=output,
        multiplet_recovery=multiplet_recovery,
        edge_cycle_verification=edge_cycle_verification,
        initial_stage_leiden_resolution=initial_stage_leiden_resolution,
        refinement_stage_leiden_resolution=refinement_stage_leiden_resolution,
        min_count=min_count,
        max_refinement_recursion_depth=max_refinement_recursion_depth,
        initial_stage_max_edges_to_remove=initial_stage_max_edges_to_remove,
        refinement_stage_max_edges_to_remove=refinement_stage_max_edges_to_remove,
        initial_stage_max_edges_to_remove_relative=initial_stage_max_edges_to_remove_relative,
        refinement_stage_max_edges_to_remove_relative=refinement_stage_max_edges_to_remove_relative,
        component_size_max_threshold=component_size_max_threshold,
        component_size_min_threshold=component_size_min_threshold,
        panel=panel,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files, allowed_extensions="parquet")

    # create output folder if it does not exist
    graph_output = create_output_stage_dir(output, "graph")

    logger.info(f"Trying to identify components from file {parquet_file}")

    sample_name = get_sample_name(parquet_file)

    write_parameters_file(
        ctx,
        graph_output / f"{sample_name}.meta.json",
        command_path="pixelator single-cell-pna graph",
    )
    output_path = graph_output / f"{sample_name}.graph.pxl"

    panel = load_antibody_panel(pna_config, panel)
    initial_stage_refinement_options = RefinementOptions(
        leiden_resolution=initial_stage_leiden_resolution,
        max_edges_to_remove=initial_stage_max_edges_to_remove,
        max_edges_to_remove_relative=initial_stage_max_edges_to_remove_relative,
        min_component_size_to_prune=component_size_min_threshold
        or MIN_PNA_COMPONENT_SIZE,
    )
    refinement_stage_refinement_options = RefinementOptions(
        leiden_resolution=refinement_stage_leiden_resolution,
        max_edges_to_remove=refinement_stage_max_edges_to_remove,
        max_edges_to_remove_relative=refinement_stage_max_edges_to_remove_relative,
        min_component_size_to_prune=component_size_min_threshold
        or MIN_PNA_COMPONENT_SIZE,
    )
    refinement_options = StagedRefinementOptions(
        max_component_refinement_depth=max_refinement_recursion_depth,
        initial_stage_options=initial_stage_refinement_options,
        refinement_stage_options=refinement_stage_refinement_options,
    )

    component_size_threshold = True
    if any([component_size_min_threshold, component_size_max_threshold]):
        component_size_threshold = (
            component_size_min_threshold,
            component_size_max_threshold,
        )

    n_cores = ctx.obj.get("CORES")
    try:
        _, component_statics = build_pxl_file_with_components(
            parquet_file=Path(parquet_file),
            panel=panel,
            sample_name=sample_name,
            path_output_pxl_file=output_path,
            multiplet_recovery=multiplet_recovery,
            edge_cycle_verification=edge_cycle_verification,
            min_read_count=min_count,
            refinement_options=refinement_options,
            component_size_threshold=component_size_threshold,
            n_cores=n_cores,
        )
    except ConnectedComponentException as e:
        logger.error(e)
        sys.exit(1)

    metrics_file = graph_output / f"{sample_name}.report.json"
    report = GraphSampleReport(
        sample_id=sample_name,
        product_id="single-cell-pna",
        **component_statics.to_dict(),
    )
    report.write_json_file(metrics_file, indent=4)

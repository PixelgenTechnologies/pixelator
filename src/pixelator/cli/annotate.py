"""Console script for pixelator (annotate).

Copyright © 2022 Pixelgen Technologies AB.
"""

import click

from pixelator.annotate import annotate_components
from pixelator.cli.common import logger, output_option
from pixelator.config import config, load_antibody_panel
from pixelator.utils import (
    create_output_stage_dir,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)


@click.command(
    "annotate",
    short_help="filter, annotate and call cells from an edge list",
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
    "--panel",
    required=True,
    type=str,
    help=(
        "A key of a panel file in the config, "
        "or a csv file with the antibody panel conjugations"
    ),
)
@click.option(
    "--min-size",
    default=None,
    required=False,
    type=click.INT,
    show_default=False,
    help="The minimum size (edges) a component must have (default is disabled)",
)
@click.option(
    "--max-size",
    default=None,
    required=False,
    type=click.INT,
    show_default=False,
    help="The maximum size (edges) a component must have (default is disabled)",
)
@click.option(
    "--dynamic-filter",
    required=False,
    default=None,
    type=click.Choice(["both", "min", "max"]),
    help=(
        "Enable the estimation of dynamic size filters using a log-rank approach\n"
        "\t both: estimate both min and max size"
        "\t min: estimate min size (--min-size)"
        "\t max: estimate max size (--max-size)"
    ),
)
@click.option(
    "--aggregate-calling",
    default=False,
    is_flag=True,
    help=(
        "Enable aggregate calling, information on "
        "potential aggregates will be added to the output data"
    ),
)
@output_option
@click.pass_context
@timer
def annotate(
    ctx,
    parquet_file,
    panel,
    min_size,
    max_size,
    dynamic_filter,
    aggregate_calling,
    output,
):
    """
    Filter, annotate and call cells from an edge list
    """
    input_files = [parquet_file]
    # log input parameters
    log_step_start(
        "annotate",
        input_files=input_files,
        panel=panel,
        output=output,
        min_size=min_size,
        max_size=max_size,
        dynamic_filter=dynamic_filter,
        aggregate_calling=aggregate_calling,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files, allowed_extensions="parquet")

    # sanity check on thresholds and input parameters
    if min_size is not None and min_size < 0:
        click.ClickException("--min-size cannot be less than 0")
    if max_size is not None and max_size < 0:
        click.ClickException("--max-size cannot be less than 0")
    if max_size is not None and min_size is not None and max_size < min_size:
        click.ClickException("--max-size cannot be less than --min-size")

    # warn if both --dynamic-filter and hard-coded sizes are input
    if min_size is not None and dynamic_filter in ["min", "both"]:
        logger.warning(
            "--dynamic-filter will overrule the value introduced in --min-size"
        )

    if max_size is not None and dynamic_filter in ["max", "both"]:
        logger.warning(
            "--dynamic-filter will overrule the value introduced in --max-size"
        )

    # create output folder if it does not exist
    annotate_output = create_output_stage_dir(output, "annotate")

    # load marker panel
    panel = load_antibody_panel(config, panel)

    # compute graph/components using parallel processing
    logger.info(f"Computing annotation for file {parquet_file}")

    clean_name = get_sample_name(parquet_file)
    metrics_file = annotate_output / f"{clean_name}.report.json"

    write_parameters_file(
        ctx,
        annotate_output / f"{clean_name}.meta.json",
        command_path="pixelator single-cell annotate",
    )

    annotate_components(
        input=str(parquet_file),
        panel=panel,
        output=str(annotate_output),
        output_prefix=clean_name,
        metrics_file=str(metrics_file),
        min_size=min_size,
        max_size=max_size,
        dynamic_filter=dynamic_filter,
        aggregate_calling=aggregate_calling,
        verbose=ctx.obj["VERBOSE"],
    )

"""Console script for the aggregate step.

Collapse the per-read edgelist parquet from ``edgelist`` into unique molecules
with read counts.

Copyright © 2026 Pixelgen Technologies AB.
"""

import logging
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
from pixelator.pna.aggregate import InvalidEdgelistSchemaError, aggregate_edgelist
from pixelator.pna.aggregate.report import AggregateSampleReport
from pixelator.pna.cli.common import memory_option, output_option, threads_option

logger = logging.getLogger(__name__)


@click.command(
    "aggregate",
    short_help="Collapse per-read edgelists into unique molecules.",
    help=(
        "Collapse the per-read edgelist Parquet produced by the edgelist command "
        "into unique molecules, counting reads per molecule. Multiple input "
        "files are merged into a single output."
    ),
    options_metavar="<options>",
)
@click.argument(
    "edgelist",
    nargs=-1,
    required=True,
    type=click.Path(exists=True),
    metavar="<EDGELIST...>",
)
@click.option(
    "--sample-name",
    default=None,
    show_default=False,
    type=click.STRING,
    help=(
        "Override the basename of the output parquet file. "
        "Default is the basename of the first input file up to the first dot."
    ),
)
@output_option
@threads_option
@memory_option
@click.pass_context
@timer
def aggregate(
    ctx,
    edgelist: tuple[str, ...],
    sample_name: str | None,
    output: str,
    threads: int,
    memory: int | None,
):
    """Collapse per-read edgelists into unique molecules with read counts."""
    # Respect the global --cores when --threads is left at its default.
    threads = threads if threads and threads > 0 else ctx.obj["CORES"]

    log_step_start(
        "aggregate",
        edgelist=edgelist,
        output=output,
        sample_name=sample_name,
        threads=threads,
        memory=memory,
    )

    edgelist_paths = [Path(path) for path in edgelist]
    sanity_check_inputs(edgelist_paths, allowed_extensions=("parquet",))

    aggregate_output = create_output_stage_dir(output, "aggregate")
    sample_name = sample_name or get_sample_name(edgelist_paths[0])

    write_parameters_file(
        ctx,
        aggregate_output / f"{sample_name}.meta.json",
        command_path="pixelator single-cell-pna aggregate",
    )

    output_path = aggregate_output / f"{sample_name}.parquet"
    logger.info(
        "Aggregating %s edgelist file(s) into unique molecules with %s threads",
        len(edgelist_paths),
        threads,
    )

    try:
        stats = aggregate_edgelist(
            edgelist_paths,
            output_path,
            threads=threads,
            memory_limit=memory,
            temp_directory=ctx.obj.get("DUCKDB_TEMP_DIR"),
        )
    except InvalidEdgelistSchemaError as error:
        raise click.ClickException(str(error)) from error

    report = AggregateSampleReport(
        product_id="single-cell-pna",
        report_type="aggregate",
        sample_id=sample_name,
        input_reads=stats.input_reads,
        output_molecules=stats.output_molecules,
        umi1_degree_distribution=stats.umi1_degree_distribution,
        umi2_degree_distribution=stats.umi2_degree_distribution,
    )
    report.write_json_file(aggregate_output / f"{sample_name}.report.json", indent=4)

    logger.info("Wrote aggregated edgelist to %s", output_path)

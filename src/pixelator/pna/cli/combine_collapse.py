"""CLI for combining parquet and json files from partitioned collapse results.

Copyright © 2024 Pixelgen Technologies AB
"""

import glob

import click
import polars as pl

from pixelator.common.utils import (
    create_output_stage_dir,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    write_parameters_file,
)
from pixelator.pna.cli.common import (
    logger,
    output_option,
)
from pixelator.pna.collapse.independent.combine_collapse import (
    combine_independent_parquet_files,
    combine_independent_report_files,
)
from pixelator.pna.collapse.paired.combine_collapse import (
    combine_parquet_files,
    combine_report_files,
)
from pixelator.pna.collapse.report import CollapseSampleReport
from pixelator.pna.collapse.utilities import check_collapse_strategy_inputs
from pixelator.pna.utils import timer


def validate_mismatches(ctx, param, value):
    """Validate the --mismatches commandline option.

    :param ctx: The click context
    :param param: The click parameter name
    :param value: The click value
    """
    try:
        value = int(value)
        if value >= 1:
            return value
    except ValueError:
        pass

    try:
        value = float(value)
        if 0 <= value < 1:
            return value
    except ValueError:
        pass

    raise click.BadParameter("Must be an integer >= 1 or a float in range [0, 1)")


@click.command(
    "combine-collapse",
    short_help=("Combine parquet and json files from partitioned collapse results."),
    options_metavar="<options>",
)
@click.option(
    "--parquet",
    multiple=True,
    type=click.Path(exists=True),
    metavar="PARQUET",
)
@click.option(
    "--report",
    "reports",
    multiple=True,
    type=click.Path(exists=True),
    metavar="JSON",
)
@click.option(
    "--parquet-pattern",
    type=str,
    help="The pattern to match parquet files.",
)
@click.option(
    "--report-pattern",
    type=str,
    help="The pattern to match report files.",
)
@output_option
@click.pass_context
@timer(command_name="combine-collapse")
def combine_collapse(ctx, parquet, reports, output, parquet_pattern, report_pattern):
    """Collapse Molecular Pixelation data (FASTQ) by umi-upi to remove duplicates and perform error correction."""  # noqa
    # log input parameters
    log_step_start(
        "combine-collapse",
        parquet=parquet,
        report=reports,
        parquet_pattern=parquet_pattern,
        report_pattern=report_pattern,
    )

    if parquet_pattern is not None:
        additional_parquet = glob.glob(parquet_pattern)
        parquet = set(parquet) if parquet else set()
        parquet.update(additional_parquet)
        parquet = list(parquet)

    if report_pattern is not None:
        additional_reports = glob.glob(report_pattern)
        reports = set(reports) if reports else set()
        reports.update(additional_reports)
        reports = list(reports)

    # some basic sanity check on the input files
    sanity_check_inputs(input_files=parquet, allowed_extensions=("parquet",))
    sanity_check_inputs(input_files=reports, allowed_extensions=("json",))

    checked_inputs = check_collapse_strategy_inputs(parquet, reports)
    collapse_output = create_output_stage_dir(output, "collapse")

    if isinstance(checked_inputs.parquet, tuple):
        umi1_files, umi2_files = checked_inputs.parquet

        # Log out all input files
        for f in umi1_files:
            logger.info(f"Using UMI1 parquet file: {str(f)}")
        for f in umi2_files:
            logger.info(f"Using UMI2 parquet file: {str(f)}")

        sample_name = get_sample_name(umi1_files[0])

        # Parameters meta.json
        write_parameters_file(
            ctx,
            collapse_output / f"{sample_name}.meta.json",
            command_path="pixelator single-cell-pna combine-collapse",
        )

        logger.info(f"Combining parquet files.")

        combined_parquet_output = collapse_output / f"{sample_name}.collapse.parquet"
        stats = combine_independent_parquet_files(
            umi1_files, umi2_files, combined_parquet_output
        )

        logger.info(f"Combining report files.")

        report1_files, report2_files = checked_inputs.reports
        combined_report_output = collapse_output / f"{sample_name}.report.json"

        combine_independent_report_files(
            report1_files,
            report2_files,
            sample_id=sample_name,
            stats=stats,
            output_file=combined_report_output,
        )

        return (
            umi1_files,
            umi2_files,
            output,
            ctx,
        )

    # create the output directory
    sample_name = get_sample_name(parquet[0])

    combined_output = collapse_output / f"{sample_name}.parquet"
    metrics_output = collapse_output / f"{sample_name}.report.json"

    # Parameters meta.json
    write_parameters_file(
        ctx,
        collapse_output / f"{sample_name}.meta.json",
        command_path="pixelator single-cell-pna combine-collapse",
    )

    logger.info(f"Combining parquet files.")

    # Merge parquet files
    combine_parquet_files(parquet, combined_output)

    logger.info(f"Combining report files.")

    # Merge report files
    combined_report = combine_report_files(reports)
    combined_report.add_summary_statistics(pl.scan_parquet(combined_output))

    report = CollapseSampleReport(
        product_id="single-cell-pna",
        report_type="collapse",
        sample_id=sample_name,
        **combined_report.to_dict(),
    )

    report.write_json_file(metrics_output, indent=4)

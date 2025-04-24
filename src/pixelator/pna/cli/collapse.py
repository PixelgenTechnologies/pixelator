"""Console script for pixelator (collapse).

Copyright © 2022 Pixelgen Technologies AB.
"""

import json
import logging
from pathlib import Path

import click
import polars as pl

from pixelator.common.utils import (
    create_output_stage_dir,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.pna.cli.common import (
    design_option,
    output_option,
    panel_option,
    threads_option,
)
from pixelator.pna.collapse.independent.collapser import (
    RegionCollapser,
    SingleUMICollapseSampleReport,
)
from pixelator.pna.collapse.paired.collapser import MoleculeCollapser
from pixelator.pna.collapse.utilities import split_collapse_inputs
from pixelator.pna.config import load_antibody_panel, pna_config
from pixelator.pna.utils import get_demux_filename_info

logger = logging.getLogger("collapse")


def validate_mismatches(ctx, param, value):
    """Validate the --mismatches commandline option.

    :param ctx: The click context
    :param param: The click parameter name
    :param value: The click value
    :returns: The validated value
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
    "collapse",
    short_help=(
        "Detect duplicates and perform error correction on demultiplexed PNA data (parquet)."
    ),
    options_metavar="<options>",
)
@click.argument(
    "input_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True),
    metavar="PARQUET_FILES",
)
@click.option(
    "--mismatches",
    default=2,
    required=False,
    type=click.UNPROCESSED,
    callback=validate_mismatches,
    show_default=True,
    help=("The number of mismatches allowed when collapsing."),
)
@click.option(
    "--algorithm",
    required=False,
    type=click.Choice(["directional", "cluster"]),
    default="directional",
    show_default=True,
    help=("The network based algorithm to use for collapsing."),
)
@threads_option
@design_option
@panel_option
@output_option
@click.pass_context
@timer
def collapse(
    ctx,
    input_files,
    design,
    panel,
    output,
    mismatches,
    algorithm,
    threads,
):
    """Collapse Molecular Pixelation data (FASTQ) to remove duplicates and perform error correction."""  # noqa
    # log input parameters
    log_step_start(
        "collapse",
        panel=panel,
        design=design,
        output=output,
        mismatches=mismatches,
        algorithm=algorithm,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files=input_files, allowed_extensions=("parquet",))

    assay = pna_config.get_assay(design)
    panel = load_antibody_panel(pna_config, panel)

    # create the output directory
    collapse_output = create_output_stage_dir(output, "collapse")

    checked_parquet = split_collapse_inputs(input_files)
    if isinstance(checked_parquet, tuple):
        logger.info("Detected independent UMI1 and UMI2 demuxed input.")

        umi1_files, umi2_files = checked_parquet
        return process_independent_files(
            umi1_files,
            umi2_files,
            assay=assay,
            panel=panel,
            collapse_output=collapse_output,
            mismatches=mismatches,
            algorithm=algorithm,
            threads=threads,
            ctx=ctx,
        )

    logger.info("Detected paired UMI1 and UMI2 demuxed input.")
    return process_paired_input(
        ctx, input_files, panel, mismatches, algorithm, threads, assay, collapse_output
    )


def process_paired_input(
    ctx, input_files, panel, mismatches, algorithm, threads, assay, collapse_output
) -> None:
    """Process paired UMI1 and UMI2 demuxed files.

    This will run the collapse process on the files together

    Args:
        ctx: The click context.
        input_files: The UM1 and UMI2 demuxed files.
        panel: The panel configuration.
        mismatches: The number of mismatches allowed when error-correcting.
        algorithm: The collapse strategy to use.
        threads: The number of threads to use.
        assay: The assay configuration.
        collapse_output: The output filename for the collapsed data.

    """
    if len(input_files) == 1:
        logger.info("Detected single input file.")
        input_file = input_files[0]
        sample_name, part_number = get_demux_filename_info(input_file)

        if part_number:
            logger.info(
                "Found part number in input, will use that part number in output files."
            )
            output_file = (
                collapse_output
                / f"{sample_name}.part_{part_number:03d}.collapsed.parquet"
            )
            metrics_output = (
                collapse_output / f"{sample_name}.part_{part_number:03d}.report.json"
            )
            param_file = (
                collapse_output / f"{sample_name}.part_{part_number:03d}.meta.json"
            )
        else:
            output_file = collapse_output / f"{sample_name}.collapsed.parquet"
            metrics_output = collapse_output / f"{sample_name}.report.json"
            param_file = collapse_output / f"{sample_name}.meta.json"
    else:
        logger.info("Detected multiple input files.")
        sample_name, part_number = get_demux_filename_info(input_files[0])

        output_file = collapse_output / f"{sample_name}.collapsed.parquet"
        metrics_output = collapse_output / f"{sample_name}.report.json"
        param_file = collapse_output / f"{sample_name}.meta.json"

    write_parameters_file(
        ctx,
        param_file,
        command_path="pixelator single-cell-pna collapse",
    )

    collapser = MoleculeCollapser(
        assay=assay,
        panel=panel,
        output=output_file,
        max_mismatches=mismatches,
        algorithm=algorithm,
        threads=threads,
    )

    with collapser as c:
        for f in input_files:
            c.process_file(Path(f))

    stats = collapser.statistics()
    stats.add_summary_statistics(pl.scan_parquet(output_file))

    encoder = json.JSONEncoder(indent=4)
    with open(metrics_output, "w") as f:
        f.write(encoder.encode(stats.to_dict()))


def process_independent_files(
    umi1_files,
    umi2_files,
    *,
    assay,
    panel,
    collapse_output,
    mismatches,
    algorithm,
    threads=-1,
    ctx=None,
):
    """Process independent UMI1 and UMI2 demuxed files.

    This will run the collapse process on all UMI1 files and UMI2 files successively.

    Args:
        umi1_files: The UMI1 demuxed files.
        umi2_files: The UMI2 demuxed files.
        assay: The assay configuration.
        panel: The panel configuration.
        collapse_output: The output filename for the collapsed data.
        mismatches: The number of mismatches allowed when error-correcting.
        algorithm: The collapse strategy to use.
        threads: The number of threads to use.
        ctx: The click context.

    Returns:
        A tuple with umi1 and umi2 output file paths

    """
    umi1_outputs = []
    umi2_outputs = []

    min_parallel_chunk_size = 500

    umi1_collapser = RegionCollapser(
        assay=assay,
        panel=panel,
        region_id="umi-1",
        threads=threads,
        algorithm=algorithm,
        min_parallel_chunk_size=min_parallel_chunk_size,
        max_mismatches=mismatches,
    )

    with umi1_collapser as c:
        for umi1_file in umi1_files:
            res = get_demux_filename_info(umi1_file)

            if not res:
                raise click.ClickException(
                    f"Could not extract sample name and part number from file name: {umi1_file}"
                )

            sample_name, part_number = res
            base_output_name = f"{sample_name}.collapse.m1.part_{part_number:03d}"
            output_file = collapse_output / f"{base_output_name}.parquet"
            metrics_output = collapse_output / f"{base_output_name}.report.json"

            # Allow skipping writing the parameters file when not running from a click context
            if ctx:
                write_parameters_file(
                    ctx,
                    collapse_output / f"{base_output_name}.meta.json",
                    command_path="pixelator single-cell-pna collapse",
                )

            c.process_file(umi1_file, output=output_file)
            umi1_outputs.append(output_file)

            stats = umi1_collapser.statistics()
            report = SingleUMICollapseSampleReport(
                sample_id=sample_name, product_id="single-cell-pna", **stats.to_dict()
            )
            report.write_json_file(metrics_output, indent=4)

    del umi1_collapser

    umi2_collapser = RegionCollapser(
        assay=assay,
        panel=panel,
        region_id="umi-2",
        threads=threads,
        algorithm=algorithm,
        min_parallel_chunk_size=min_parallel_chunk_size,
        max_mismatches=mismatches,
    )

    with umi2_collapser as c:
        for umi2_file in umi2_files:
            res = get_demux_filename_info(umi2_file)

            if not res:
                raise click.ClickException(
                    f"Could not extract sample name and part number from file name: {umi2_file}"
                )

            sample_name, part_number = res
            base_output_name = f"{sample_name}.collapse.m2.part_{part_number:03d}"
            output_file = collapse_output / f"{base_output_name}.parquet"
            metrics_output = collapse_output / f"{base_output_name}.report.json"

            # Allow skipping writing the parameters file when not running from a click context
            if ctx:
                write_parameters_file(
                    ctx,
                    collapse_output / f"{base_output_name}.meta.json",
                    command_path="pixelator single-cell-pna collapse",
                )

            c.process_file(umi2_file, output=output_file)
            umi2_outputs.append(output_file)

            stats = umi2_collapser.statistics().to_dict()
            report = SingleUMICollapseSampleReport(
                sample_id=sample_name, product_id="single-cell-pna", **stats
            )
            report.write_json_file(metrics_output, indent=4)

    del umi2_collapser

    return (umi1_outputs, umi2_outputs)

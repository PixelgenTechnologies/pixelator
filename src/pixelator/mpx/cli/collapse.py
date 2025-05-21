"""Console script for pixelator (collapse).

Copyright © 2022 Pixelgen Technologies AB.
"""

import sys
from collections import defaultdict
from concurrent import futures
from pathlib import Path

import click
import polars as pl

from pixelator.common.report.models import SummaryStatistics
from pixelator.common.utils import (
    create_output_stage_dir,
    get_process_pool_executor,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.mpx.cli.common import design_option, logger, output_option
from pixelator.mpx.collapse import collapse_fastq
from pixelator.mpx.config import config, get_position_in_parent, load_antibody_panel
from pixelator.mpx.report.models import CollapseSampleReport


def _handle_errors(jobs, executor):
    for job in jobs:
        exception = job.exception()
        if exception is None:
            continue

        logger.error(
            "Found an issue in the process pool. Trying to determine what went wrong and set the correct exit code. Exception was: %s",
            exception,
        )
        process_map = executor._processes
        for pid in process_map.keys():
            exit_code = process_map[pid].exitcode
            if exit_code is not None and exit_code != 0:
                logger.error(
                    "The child process in the process pool returned a non-zero exit code: %s.",
                    exit_code,
                )
                # If we have an out of memory exception, make sure we exit with that.
                if abs(exit_code) == 9:
                    logger.error(
                        "One of the child processes was killed (exit code: 9). "
                        "Usually this is caused by the out-of-memory killer terminating the process. "
                        "The parent process will return an exit code of 137 to indicate that it terminated because of a kill signal in the child process."
                    )
                    sys.exit(137)
        logger.error(
            "Was unable to determine what when wrong in process pool. Will raise original exception."
        )
        raise exception


@click.command(
    "collapse",
    short_help=(
        "collapse pixel data (FASTQ) by UMI-UPI to remove duplicates"
        " and perform error correction"
    ),
    options_metavar="<options>",
)
@click.argument(
    "input_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True),
    metavar="FASTQ_FILES",
)
@click.option(
    "--markers-ignore",
    required=False,
    default=None,
    type=click.STRING,
    help="A list of comma separated antibodies to ignore (discard)",
)
@click.option(
    "--algorithm",
    required=False,
    default="adjacency",
    type=click.Choice(["adjacency", "unique"]),
    show_default=True,
    help=(
        "The algorithm to use for collapsing (adjacency will perform error correction"
        " using the number of mismatches given)"
    ),
)
@click.option(
    "--max-neighbours",
    default=60,
    required=False,
    type=click.IntRange(1, 250),
    show_default=True,
    help=(
        "The maximum number of neighbors to use when searching for similar sequences."
        " This number depends on the sequence depth and the ratio of"
        " erronous molecules expected. A high value can make the algorithm slower."
        " This is only used when algorithm is set to 'adjacency'"
    ),
)
@click.option(
    "--mismatches",
    default=2,
    required=False,
    type=click.IntRange(0, 5),
    show_default=True,
    help=(
        "The number of mismatches allowed when collapsing. This is only used when the"
        " algorithm is set to 'adjacency'."
    ),
)
@click.option(
    "--min-count",
    default=1,
    required=False,
    type=click.IntRange(1, 10),
    show_default=True,
    help=(
        "Discard molecules with with a count (reads) lower than this (set to 1 to"
        " disable)"
    ),
)
@click.option(
    "--panel",
    required=True,
    type=str,
    help=(
        "A key of a panel file in the config,"
        " or a csv file with the antibody panel conjugations"
    ),
)
@output_option
@design_option
@click.pass_context
@timer
def collapse(
    ctx,
    input_files,
    markers_ignore,
    algorithm,
    max_neighbours,
    mismatches,
    min_count,
    panel,
    output,
    design,
):
    """Collapse Molecular Pixelation data (FASTQ) by umi-upi to remove duplicates and perform error correction."""  # noqa
    # log input parameters
    log_step_start(
        "collapse",
        output=output,
        markers_ignore=markers_ignore,
        algorithm=algorithm,
        max_neighbours=max_neighbours,
        mismatches=mismatches,
        min_count=min_count,
        panel=panel,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(
        input_files=input_files, allowed_extensions=("fastq.gz", "fq.gz")
    )

    # obtain a list of markers to ignore if given by the user
    if markers_ignore is not None:
        markers_ignore = markers_ignore.split(",")

    # update arguments from config file (CLI arguments have priority)
    try:
        assay = config.get_assay(design)

        if "umi-a" in assay.region_ids:
            umia_start, umia_end = get_position_in_parent(assay, "umi-a")
        else:
            umia_start, umia_end = (None, None)

        upia_start, upia_end = get_position_in_parent(assay, "upi-a")
        upib_start, upib_end = get_position_in_parent(assay, "upi-b")
        umib_start, umib_end = get_position_in_parent(assay, "umi-b")

    except KeyError as exc:
        raise click.ClickException(f"Parsing attribute from config file {exc}")

    # sanity check on upi and umi arguments
    if upia_end <= upia_start:
        click.ClickException("UPIA end or start positions seems to be incorrect")
    if upib_end <= upib_start:
        click.ClickException("UPIB end or start positions seems to be incorrect")
    if umia_start is None and umia_end is not None:
        click.ClickException("You must specify both the start and end position in UMIA")
    if umib_start is None and umib_end is not None:
        click.ClickException("You must specify both the start and end position in UMIB")
    if umia_start is None and umia_end is None:
        logger.info("UMIA will be ignored in the collapsing")
    elif umia_end <= umia_start:
        click.ClickException("UMIA end or start positions seems to be incorrect")
    if umib_start is None and umib_end is None:
        logger.info("UMIB will be ignored in the collapsing")
    elif umib_end <= umib_start:
        click.ClickException("UMIB end or start positions seems to be incorrect")

    # create output folder if it does not exist
    collapse_output = create_output_stage_dir(output, "collapse")

    # load marker panel
    panel = load_antibody_panel(config, panel)

    # group the input files by sample name
    input_samples = defaultdict(list)
    for fastq_file in input_files:
        sample = get_sample_name(fastq_file)
        input_samples[sample].append(fastq_file)

    if len(input_samples) == 0:
        click.ClickException("No input file could be found with a valid sample name")

    for sample, files in input_samples.items():
        logger.info(f"Processing {len(files)} files for sample {sample}")

        write_parameters_file(
            ctx,
            collapse_output / f"{sample}.meta.json",
            command_path="pixelator single-cell-mpx collapse",
        )

        # run cutadapt (demux mode) using parallel processing
        with get_process_pool_executor(
            nbr_cores=ctx.obj["CORES"], logging_setup=ctx.obj["LOGGER"]
        ) as executor:
            jobs = []
            for fastq_file in files:
                # get the marker from the file name
                sequence = fastq_file.split(".processed-")[1].split(".")[0]
                marker = panel.get_marker_id(sequence)
                if marker is None:
                    raise click.ClickException(
                        f"Retrieving marker from file {fastq_file}"
                    )

                # ignore the marker if it is in the ignore list
                if markers_ignore is not None and marker in markers_ignore:
                    logger.debug("Ignoring %s with antibody %s", fastq_file, marker)
                else:
                    logger.debug("Processing %s with antibody %s", fastq_file, marker)

                    jobs.append(
                        executor.submit(
                            collapse_fastq,
                            input_file=fastq_file,
                            algorithm=algorithm,
                            marker=marker,
                            sequence=sequence,
                            upia_start=upia_start,
                            upia_end=upia_end,
                            upib_start=upib_start,
                            upib_end=upib_end,
                            umia_start=umia_start,
                            umia_end=umia_end,
                            umib_start=umib_start,
                            umib_end=umib_end,
                            max_neighbours=max_neighbours,
                            mismatches=mismatches,
                            min_count=min_count,
                        )
                    )
            jobs = list(futures.as_completed(jobs))
            _handle_errors(jobs, executor)

        total_input_reads = 0
        tmp_files = []
        for job in jobs:
            # the worker returns a path to a file (temp antibody edge list)
            tmp_file, input_reads_count = job.result()
            if tmp_file is not None:
                tmp_files.append(tmp_file)
            if input_reads_count is not None:
                total_input_reads += input_reads_count

        # create the final edge list from all the temporary ones (antibodies)
        logger.debug("Creating edge list for sample %s", sample)
        df = pl.concat(
            (pl.read_ipc(f, memory_map=False) for f in tmp_files), how="vertical"
        )

        # Collect some stats for reporting
        collapsed_molecule_count_stats = SummaryStatistics.from_series(
            df["unique_molecules_count"]
        )

        output_file = collapse_output / f"{sample}.collapsed.parquet"
        # Remove the unique_molecules_count column before saving
        df.drop("unique_molecules_count")
        df.write_parquet(output_file, compression="zstd")

        # remove temporary edge list files
        for f in tmp_files:
            logger.debug("Removing temporary edge list %s", f)
            Path(f).unlink(missing_ok=True)

        output_read_count = int(df["count"].sum())
        molecule_count = int(df.shape[0])

        report = CollapseSampleReport(
            sample_id=sample,
            input_read_count=total_input_reads,
            output_read_count=output_read_count,
            molecule_count=molecule_count,
            collapsed_molecule_count_stats=collapsed_molecule_count_stats,
        )

        report.write_json_file(collapse_output / f"{sample}.report.json", indent=4)

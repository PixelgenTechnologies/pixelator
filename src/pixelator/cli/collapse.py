"""Console script for pixelator (collapse).

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import json
from collections import defaultdict
from concurrent import futures
from pathlib import Path

import click
import pandas as pd

from pixelator.cli.common import design_option, logger, output_option
from pixelator.collapse import collapse_fastq
from pixelator.config import config, get_position_in_parent, load_antibody_panel
from pixelator.utils import (
    click_echo,
    create_output_stage_dir,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)


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
        click_echo("UMIA will be ignored in the collapsing", multiline=False)
    elif umia_end <= umia_start:
        click.ClickException("UMIA end or start positions seems to be incorrect")
    if umib_start is None and umib_end is None:
        click_echo("UMIB will be ignored in the collapsing", multiline=False)
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

        output_file = collapse_output / f"{sample}.collapsed.csv.gz"
        json_file = collapse_output / f"{sample}.report.json"

        write_parameters_file(
            ctx,
            collapse_output / f"{sample}.meta.json",
            command_path="pixelator single-cell collapse",
        )

        # run cutadapt (demux mode) using parallel processing
        with futures.ProcessPoolExecutor(max_workers=ctx.obj["CORES"]) as executor:
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

        tmp_files = []
        for job in futures.as_completed(jobs):
            if job.exception() is not None:
                raise job.exception()
            # the worker returns a path to a file (temp antibody edge list)
            tmp_file = job.result()
            if tmp_file is not None:
                tmp_files.append(tmp_file)

        # create the final edge list from all the temporary ones (antibodies)
        logger.debug("Creating edge list for sample %s", sample)
        df = pd.concat(
            (pd.read_feather(f, use_threads=True) for f in tmp_files), axis=0
        )
        df.to_csv(output_file, header=True, index=False, compression="gzip")

        # remove temporary edge list files
        for f in tmp_files:
            logger.debug("Removing temporary edge list %s", f)
            Path(f).unlink(missing_ok=True)

        # write metrics
        metrics = {
            "total_pixels": int(df.shape[0]),
            "total_count": int(df["count"].sum()),
            "total_unique_umi": int(df["umi_unique_count"].sum()),
            "total_unique_upi": int(df["upi_unique_count"].sum()),
        }
        with open(json_file, "w") as outfile:
            json.dump(metrics, outfile)

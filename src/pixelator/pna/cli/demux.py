"""Console script for pixelator (demux).

Copyright © 2022 Pixelgen Technologies AB.
"""

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
from pixelator.common.utils.units import parse_size
from pixelator.pna.cli.common import (
    design_option,
    logger,
    memory_option,
    output_option,
    panel_option,
    threads_option,
)
from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import load_antibody_panel
from pixelator.pna.demux import (
    correct_marker_barcodes,
    demux_barcode_groups,
    finalize_batched_groups,
)
from pixelator.pna.demux.report import DemuxSampleReport


def _chunk_size_validator(ctx, param, value):
    try:
        return int(parse_size(value))
    except ValueError as exc:
        raise click.BadParameter(
            "chunk size must be a positive integer, optionally with a unit suffix [K, M, G]"
        )


@click.command(
    "demux",
    short_help="demultiplex pixel data (FASTQ) to generate one file per antibody",
    options_metavar="<options>",
)
@click.argument(
    "fastq_file",
    nargs=1,
    required=True,
    type=click.Path(exists=True),
    metavar="FASTQ",
)
@click.option(
    "--mismatches",
    default=1,
    required=False,
    show_default=True,
    help="The number of mismatches allowed in marker barcodes",
)
@click.option(
    "--output-chunk-reads",
    default="10M",
    type=click.STRING,
    callback=_chunk_size_validator,
    required=False,
    show_default=True,
    help="The target number of molecules in each output parquet file",
)
@click.option(
    "--output-max-chunks",
    default=8,
    type=click.INT,
    required=False,
    show_default=True,
    help="The maximum number of marker parts to split the demuxed data into",
)
@click.option(
    "--strategy",
    type=click.Choice(["paired", "independent"]),
    default="independent",
    required=False,
    help=(
        "The strategy for splitting demuxed files. 'paired' will split files by batches of PID pairs"
        "'independent' will create two partitions of files split by marker1 and marker2 independently."
    ),
)
@threads_option
@memory_option
@design_option
@panel_option
@output_option
@click.pass_context
@timer
def demux(
    ctx,
    fastq_file,
    mismatches,
    output_chunk_reads,
    output_max_chunks,
    panel,
    output,
    design,
    threads,
    memory,
    strategy,
):
    """Demultiplex Molecular Pixelation data (FASTQ) to generate one file per antibody."""
    # log input parameters
    input_files = [fastq_file]
    log_step_start(
        "demux",
        input_files=input_files,
        output=output,
        mismatches=mismatches,
        output_chunk_reads=output_chunk_reads,
        output_max_chunks=output_max_chunks,
        panel=panel,
        design=design,
        strategy=strategy,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(
        input_files,
        allowed_extensions=("fastq.gz", "fq.gz", "fastq", "fq", "fastq.zst", "fq.zst"),
    )

    fastq_file = Path(fastq_file)

    # create output folder if it does not exist
    demux_output = create_output_stage_dir(output, "demux")

    # load assay design
    assay = pna_config.get_assay(design)
    # load marker panel
    panel = load_antibody_panel(pna_config, panel)

    logger.info(f"Correcting marker barcodes for input: {fastq_file}")

    stats, corrected, failed = correct_marker_barcodes(
        input=input_files[0],
        assay=assay,
        panel=panel,
        output=demux_output,
        save_failed=True,
        threads=threads,
    )

    # Store intermediate parquet files before deduplication and sorting
    tmp_output_dir = demux_output / "tmp"
    tmp_output_dir.mkdir()

    demux_barcode_groups(
        corrected_reads=corrected,
        assay=assay,
        panel=panel,
        stats=stats,
        output_dir=tmp_output_dir,
        threads=threads,
        reads_per_chunk=output_chunk_reads,
        max_chunks=output_max_chunks,
        stategy=strategy,
    )

    finalize_batched_groups(
        input_dir=tmp_output_dir,
        output_dir=demux_output,
        strategy=strategy,
        memory=memory,
    )

    # remove results in the temporary output directory
    for file in tmp_output_dir.iterdir():
        file.unlink()
    tmp_output_dir.rmdir()

    sample_name = get_sample_name(fastq_file)
    report_json = demux_output / f"{sample_name}.report.json"

    write_parameters_file(
        ctx,
        demux_output / f"{sample_name}.meta.json",
        command_path="pixelator single-cell-pna demux",
    )

    report = DemuxSampleReport(
        product_id="single-cell-pna",
        report_type="demux",
        sample_id=sample_name,
        **stats.as_json(),
    )
    report.write_json_file(report_json, indent=4)

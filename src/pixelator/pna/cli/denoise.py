"""Pipeline console script for denoising pna data.

Copyright © 2025 Pixelgen Technologies AB.
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
from pixelator.pna import read
from pixelator.pna.analysis.denoise import DenoiseOneCore
from pixelator.pna.analysis.report import DenoiseSampleReport
from pixelator.pna.analysis_engine import AnalysisManager, LoggingSetup
from pixelator.pna.cli.common import output_option
from pixelator.pna.pixeldataset.io import PxlFile

logger = logging.getLogger(__name__)


@click.command(
    "denoise",
    short_help=("Denoise a pxl file by applying node filtering techniques."),
    options_metavar="<options>",
)
@click.argument(
    "pxl_file",
    required=True,
    type=click.Path(exists=True),
    metavar="<PIXELFILE>",
)
@click.option(
    "--run-one-core-graph-denoising",
    required=False,
    is_flag=True,
    help="Run the denoise step to remove markers that are over-expressed in the one-core layer of a component.",
)
@click.option(
    "--one-core-ratio-threshold",
    default=0.9,
    required=False,
    type=click.FloatRange(
        0,
        1,
    ),
    show_default=True,
    help=(
        "ratio of the number of nodes in the one-core layer to the total number of nodes in a component. "
        "If the ratio is above this threshold, the component is marked as disqualified for denoising."
    ),
)
@click.option(
    "--pval-threshold",
    default=0.05,
    required=False,
    type=click.FloatRange(
        0,
        1,
    ),
    show_default=True,
    help="pvalue threshold for an over-expression in the one-core layer to be considered significant.",
)
@click.option(
    "--inflate-factor",
    default=1.5,
    required=False,
    type=click.FloatRange(
        1,
        10,
    ),
    show_default=True,
    help="How much to inflate number of noise markers in the one-core layer to remove.",
)
@output_option
@click.pass_context
@timer
def denoise(
    ctx,
    pxl_file,
    run_one_core_graph_denoising,
    one_core_ratio_threshold,
    pval_threshold,
    inflate_factor,
    output,
):
    """Denoise components of a PXL file."""
    input_files = [pxl_file]
    log_step_start(
        "denoise",
        input_files=input_files,
        run_one_core_graph_denoising=run_one_core_graph_denoising,
        one_core_ratio_threshold=one_core_ratio_threshold,
        pval_threshold=pval_threshold,
        inflate_factor=inflate_factor,
        output=output,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files=input_files, allowed_extensions=("pxl",))

    sample_name = get_sample_name(pxl_file)
    pxl_file = PxlFile(Path(pxl_file))
    denoise_output = create_output_stage_dir(output, "denoise")
    output_file = denoise_output / f"{sample_name}.denoised_graph.pxl"
    output_pxl_file_target = PxlFile.copy_pxl_file(pxl_file, output_file)

    write_parameters_file(
        ctx,
        denoise_output / f"{sample_name}.meta.json",
        command_path="pixelator single-cell-pna denoise",
    )
    metrics = denoise_output / f"{sample_name}.report.json"

    if not run_one_core_graph_denoising:
        report = DenoiseSampleReport(
            sample_id=sample_name,
            product_id="single-cell-pna",
            number_of_umis_removed=None,
            ratio_of_umis_removed=None,
            number_of_disqualified_components=None,
            ratio_of_disqualified_components=None,
        )
        report.write_json_file(metrics, indent=4)
        return

    analysis_to_run = [
        DenoiseOneCore(pval_threshold, inflate_factor, one_core_ratio_threshold)
    ]
    logging_setup = LoggingSetup.from_logger(ctx.obj.get("LOGGER"))
    manager = AnalysisManager(analysis_to_run, logging_setup=logging_setup)
    pxl_dataset = read(pxl_file.path)

    pxl_dataset_denoised = manager.execute(pxl_dataset, output_pxl_file_target)

    number_of_umis_removed = int(
        pxl_dataset_denoised.adata().obs["number_of_nodes_removed_in_denoise"].sum()
    )
    ratio_of_umis_removed = float(
        number_of_umis_removed
        / (pxl_dataset.adata().obs["n_umi"].sum() + number_of_umis_removed)
    )
    number_of_disqualified_components = int(
        pxl_dataset_denoised.adata().obs["disqualified_for_denoising"].sum()
    )
    ratio_of_disqualified_components = float(
        pxl_dataset_denoised.adata().obs["disqualified_for_denoising"].mean()
    )
    report = DenoiseSampleReport(
        sample_id=sample_name,
        product_id="single-cell-pna",
        number_of_umis_removed=number_of_umis_removed,
        ratio_of_umis_removed=ratio_of_umis_removed,
        number_of_disqualified_components=number_of_disqualified_components,
        ratio_of_disqualified_components=ratio_of_disqualified_components,
    )

    report.write_json_file(metrics, indent=4)

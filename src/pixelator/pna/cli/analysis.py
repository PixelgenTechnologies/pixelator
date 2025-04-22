"""Pipeline console script for spatial analysis of pna data.

Copyright © 2024 Pixelgen Technologies AB.
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
from pixelator.pna.analysis.analysis import (
    ProximityAnalysis,
)
from pixelator.pna.analysis.graph_noise_analysis import KcoreAnalysis, SvdAnalysis
from pixelator.pna.analysis.report import (
    AnalysisSampleReport,
    KCoreReport,
    ProximityReport,
    SvdReport,
)
from pixelator.pna.analysis_engine import AnalysisManager, LoggingSetup
from pixelator.pna.cli.common import output_option
from pixelator.pna.pixeldataset.io import PxlFile

logger = logging.getLogger(__name__)


@click.command(
    "analysis",
    short_help=("Add analysis results to a pxl file."),
    options_metavar="<options>",
)
@click.argument(
    "pxl_file",
    required=True,
    type=click.Path(exists=True),
    metavar="<PIXELFILE>",
)
@click.option(
    "--compute-proximity",
    required=False,
    is_flag=True,
    help="Compute proximity scores matrix",
)
@click.option(
    "--proximity-nbr-of-permutations",
    default=100,
    required=False,
    type=click.IntRange(
        50,
    ),
    show_default=True,
    help="Number of permutations to use when calculating the expected proximity scores",
)
@click.option(
    "--compute-k-cores",
    required=False,
    is_flag=True,
    help="Compute k-core summary tables for each component",
)
@click.option(
    "--compute-svd-var-explained",
    required=False,
    is_flag=True,
    help="Compute variance explained for top 3 singular vectors for each component",
)
@click.option(
    "--svd-nbr-of-pivots",
    default=50,
    required=False,
    type=click.IntRange(10, 200),
    show_default=True,
    help="Number of pivots to use for SVD variance explained computation",
)
@output_option
@click.pass_context
@timer
def analysis(
    ctx,
    pxl_file,
    compute_proximity,
    proximity_nbr_of_permutations,
    compute_k_cores,
    compute_svd_var_explained,
    svd_nbr_of_pivots,
    output,
):
    """Add analysis results to a PXL file."""
    input_files = [pxl_file]
    log_step_start(
        "analysis",
        input_files=input_files,
        output=output,
        compute_proximity=compute_proximity,
        proximity_nbr_of_permutations=proximity_nbr_of_permutations,
        compute_k_cores=compute_k_cores,
        compute_svd_var_explained=compute_svd_var_explained,
        svd_nbr_of_pivots=svd_nbr_of_pivots,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files=input_files, allowed_extensions=("pxl",))

    # sanity check on the analyses
    if not any(
        [
            compute_proximity,
            compute_k_cores,
            compute_svd_var_explained,
        ]
    ):
        logger.warning("All the analysis are disabled, no scores will be computed")

    analysis_to_run = []

    if compute_proximity:
        logger.info("Proximity score computation is activated")
        analysis_to_run.append(
            ProximityAnalysis(n_permutations=proximity_nbr_of_permutations)
        )

    if compute_k_cores:
        logger.info("K-core computation is activated")
        analysis_to_run.append(KcoreAnalysis())

    if compute_svd_var_explained:
        logger.info("SVD variance explained computation is activated")
        analysis_to_run.append(SvdAnalysis(pivots=svd_nbr_of_pivots))

    sample_name = get_sample_name(pxl_file)
    pxl_file = PxlFile(Path(pxl_file))
    analysis_output = create_output_stage_dir(output, "analysis")
    output_file = analysis_output / f"{sample_name}.analysis.pxl"
    metrics = analysis_output / f"{sample_name}.report.json"

    logging_setup = LoggingSetup.from_logger(ctx.obj.get("LOGGER"))
    manager = AnalysisManager(analysis_to_run, logging_setup=logging_setup)
    output_pxl_file_target = PxlFile.copy_pxl_file(pxl_file, output_file)
    pxl_dataset = read(pxl_file.path)
    pxl_dataset_with_analysis = manager.execute(pxl_dataset, output_pxl_file_target)

    write_parameters_file(
        ctx,
        analysis_output / f"{sample_name}.meta.json",
        command_path="pixelator single-cell-pna analysis",
    )

    proximity_report = ProximityReport() if compute_proximity else None

    k_cores_report = None
    if compute_k_cores:
        k_cores_report = KCoreReport(
            median_average_k_core=pxl_dataset_with_analysis.adata()
            .obs["average_k_core"]
            .median()
        )

    svd_report = None
    if compute_svd_var_explained:
        svd_report = SvdReport(
            median_variance_explained_3d=pxl_dataset_with_analysis.adata()
            .obs.filter(regex="svd_var_expl_s3")
            .median()
        )

    report = AnalysisSampleReport(
        sample_id=sample_name,
        product_id="single-cell-pna",
        proximity=proximity_report,
        k_cores=k_cores_report,
        svd=svd_report,
    )

    report.write_json_file(metrics, indent=4)

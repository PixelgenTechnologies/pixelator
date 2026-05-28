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
from pixelator.pna.analysis.denoise import DenoiseGraph
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
    "--run-ace-denoising",
    required=False,
    is_flag=True,
    help=(
        "Remove nodes in the peripheral-like partition from adaptive core expansion (ACE). "
        "Can be combined with --run-one-core-graph-denoising: both use the full component graph; "
        "marked nodes are merged and stranded nodes are removed once at the end."
    ),
)
@click.option(
    "--ace-k",
    default=3,
    type=click.IntRange(1, 6),
    show_default=True,
    help="Neighborhood radius (steps) for ACE when --run-ace-denoising is set.",
)
@click.option(
    "--ace-max-k-core",
    default=4,
    type=click.IntRange(2, 10),
    show_default=True,
    help="Maximum k-core layer used for ACE seeding when --run-ace-denoising is set.",
)
@click.option(
    "--ace-no-select-lcc",
    is_flag=True,
    default=False,
    help=(
        "When --run-ace-denoising is set, do not restrict the ACE initial seed to the "
        "largest connected component (default: seed uses the LCC)."
    ),
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
        "If the ratio is above this threshold, one-core denoising is skipped for that component."
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
@click.option(
    "--run-pls-denoising",
    required=False,
    is_flag=True,
    help=(
        "PLS-on-coreness denoise: retain the largest connected component after filtering nodes "
        "by significant PLS score components. "
        "Uses the full graph like ACE and one-core; removals are merged with other methods."
    ),
)
@click.option(
    "--pls-ncomp",
    default=5,
    type=click.IntRange(1, 50),
    show_default=True,
    help="Requested PLS components when --run-pls-denoising is set (capped per component).",
)
@click.option(
    "--pls-model-k",
    default=2,
    type=click.IntRange(0, 6),
    show_default=True,
    help="Neighborhood steps for fitting the PLS X matrix when --run-pls-denoising is set.",
)
@click.option(
    "--pls-pred-k",
    default=1,
    type=click.IntRange(0, 6),
    show_default=True,
    help="Neighborhood steps for PLS scores (prediction X) when --run-pls-denoising is set.",
)
@click.option(
    "--pls-use-weights/--pls-no-weights",
    default=True,
    show_default=True,
    help="Use edge weights in PLS neighborhood expansion when --run-pls-denoising is set.",
)
@click.option(
    "--pls-normalization",
    default="L1",
    type=click.Choice(["L1", "CLR", "LogNormalize", None], case_sensitive=True),
    show_default=True,
    help="Normalization for the PLS neighborhood matrix when --run-pls-denoising is set.",
)
@click.option(
    "--pls-residualize",
    is_flag=True,
    help=(
        "Residualize the PLS neighborhood matrix against pixel_type (A/B) when "
        "--run-pls-denoising is set."
    ),
)
@click.option(
    "--pls-component-p-threshold",
    default=0.01,
    type=click.FloatRange(0.0, 1.0, min_open=True),
    show_default=True,
    help="Pearson vs coreness: keep PLS components with p-value below this when --run-pls-denoising is set.",
)
@click.option(
    "--pls-min-coreness-correlation",
    default=0.0,
    type=click.FloatRange(-1.0, 1.0),
    show_default=True,
    help="Keep PLS components whose Pearson correlation with coreness exceeds this.",
)
@click.option(
    "--pls-score-threshold",
    default=-3.0,
    type=float,
    show_default=True,
    help="All retained PLS score columns must exceed this value when --run-pls-denoising is set.",
)
@output_option
@click.pass_context
@timer
def denoise(
    ctx,
    pxl_file,
    run_one_core_graph_denoising,
    run_ace_denoising,
    ace_k,
    ace_max_k_core,
    ace_no_select_lcc,
    one_core_ratio_threshold,
    pval_threshold,
    inflate_factor,
    run_pls_denoising,
    pls_ncomp,
    pls_model_k,
    pls_pred_k,
    pls_use_weights,
    pls_normalization,
    pls_residualize,
    pls_component_p_threshold,
    pls_min_coreness_correlation,
    pls_score_threshold,
    output,
):
    """Denoise components of a PXL file.

    Args:
        ctx: Click context from the command decorator.
        pxl_file: Pxl file.
        run_one_core_graph_denoising: Run the denoise step to remove markers that are over-expressed in the one-core layer of a component.
        run_ace_denoising: Run ace denoising.
        ace_k: Ace k.
        ace_max_k_core: Ace max k core.
        ace_no_select_lcc: Ace no select lcc.
        one_core_ratio_threshold: One core ratio threshold.
        pval_threshold: Pval threshold.
        inflate_factor: Inflate factor.
        run_pls_denoising: Run pls denoising.
        pls_ncomp: Pls ncomp.
        pls_model_k: Pls model k.
        pls_pred_k: Pls pred k.
        pls_use_weights: Pls use weights.
        pls_normalization: Pls normalization.
        pls_residualize: Pls residualize.
        pls_component_p_threshold: Pls component p threshold.
        pls_min_coreness_correlation: Pls min coreness correlation.
        pls_score_threshold: All retained PLS score columns must exceed this value when --run-pls-denoising is set.
        output: Output.
    """
    input_files = [pxl_file]
    log_step_start(
        "denoise",
        input_files=input_files,
        run_one_core_graph_denoising=run_one_core_graph_denoising,
        run_ace_denoising=run_ace_denoising,
        run_pls_denoising=run_pls_denoising,
        ace_k=ace_k,
        ace_max_k_core=ace_max_k_core,
        ace_select_lcc=not ace_no_select_lcc,
        one_core_ratio_threshold=one_core_ratio_threshold,
        pval_threshold=pval_threshold,
        inflate_factor=inflate_factor,
        pls_ncomp=pls_ncomp,
        pls_model_k=pls_model_k,
        pls_pred_k=pls_pred_k,
        pls_use_weights=pls_use_weights,
        pls_normalization=pls_normalization,
        pls_residualize=pls_residualize,
        pls_component_p_threshold=pls_component_p_threshold,
        pls_min_coreness_correlation=pls_min_coreness_correlation,
        pls_score_threshold=pls_score_threshold,
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

    if (
        not run_one_core_graph_denoising
        and not run_ace_denoising
        and not run_pls_denoising
    ):
        report = DenoiseSampleReport(
            sample_id=sample_name,
            product_id="single-cell-pna",
            number_of_umis_removed=None,
            ratio_of_umis_removed=None,
        )
        report.write_json_file(metrics, indent=4)
        return

    analysis_to_run = [
        DenoiseGraph(
            run_one_core=run_one_core_graph_denoising,
            run_ace=run_ace_denoising,
            run_pls=run_pls_denoising,
            pval_significance_threshold=pval_threshold,
            inflate_factor=inflate_factor,
            one_core_ratio_threshold=one_core_ratio_threshold,
            k=ace_k,
            max_k_core=ace_max_k_core,
            ace_select_lcc=not ace_no_select_lcc,
            pls_ncomp=pls_ncomp,
            pls_model_k=pls_model_k,
            pls_pred_k=pls_pred_k,
            pls_use_weights=pls_use_weights,
            pls_normalization=pls_normalization,
            pls_residualize=pls_residualize,
            pls_component_p_threshold=pls_component_p_threshold,
            min_pls_coreness_correlation=pls_min_coreness_correlation,
            pls_score_threshold=pls_score_threshold,
        )
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
    report = DenoiseSampleReport(
        sample_id=sample_name,
        product_id="single-cell-pna",
        number_of_umis_removed=number_of_umis_removed,
        ratio_of_umis_removed=ratio_of_umis_removed,
    )

    report.write_json_file(metrics, indent=4)

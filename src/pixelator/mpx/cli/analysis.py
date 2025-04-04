"""
Console script for pixelator (analysis)

Copyright © 2022 Pixelgen Technologies AB.
"""

from typing import get_args

import click

from pixelator.common.utils import (
    create_output_stage_dir,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.mpx.analysis import analyse_pixels
from pixelator.mpx.analysis.colocalization import ColocalizationAnalysis
from pixelator.mpx.analysis.colocalization.types import TransformationTypes
from pixelator.mpx.analysis.polarization import PolarizationAnalysis
from pixelator.mpx.cli.common import logger, output_option


@click.command(
    "analysis",
    short_help="compute different downstream analyses on annotated samples",
    options_metavar="<options>",
)
@click.argument(
    "pxl_file",
    nargs=1,
    required=True,
    type=click.Path(exists=True),
    metavar="PXL",
)
@click.option(
    "--compute-polarization",
    required=False,
    is_flag=True,
    help="Compute polarization scores matrix (components by markers)",
)
@click.option(
    "--compute-colocalization",
    required=False,
    is_flag=True,
    help="Compute colocalization scores (marker by marker) for each component",
)
@click.option(
    "--use-full-bipartite",
    required=False,
    is_flag=True,
    help=(
        "Use the bipartite graph instead of the one-node projection (UPIA)"
        " when computing polarization, coabundance and colocalization scores"
    ),
)
@click.option(
    "--polarization-transformation",
    default="log1p",
    required=False,
    type=click.Choice(["raw", "log1p"]),
    show_default=True,
    help=(
        "Which approach to use to normalize the antibody counts:"
        " \n\traw will use the raw counts"
        " \n\tlog1p will use the log(x+1) transformed counts"
    ),
)
@click.option(
    "--polarization-n-permutations",
    default=50,
    required=False,
    type=click.IntRange(min=5),
    show_default=True,
    help=(
        "Set the number of permutations use to compute the empirical"
        " z-score and p-value for the polarization score."
    ),
)
@click.option(
    "--polarization-min-marker-count",
    default=5,
    required=False,
    type=click.IntRange(min=2),
    show_default=True,
    help=(
        "Set the minimum number of counts of a marker to calculate"
        " the polarization score in a component."
    ),
)
@click.option(
    "--colocalization-transformation",
    default="rate-diff",
    required=False,
    type=click.Choice(get_args(TransformationTypes)),
    show_default=True,
    help=(
        "Select the type of transformation to use on the "
        "node by antibody counts matrix when computing colocalization"
    ),
)
@click.option(
    "--colocalization-neighbourhood-size",
    default=1,
    required=False,
    type=click.IntRange(min=0),
    show_default=True,
    help=(
        "Select the size of the neighborhood to use when computing "
        "colocalization metrics on each component"
    ),
)
@click.option(
    "--colocalization-n-permutations",
    default=50,
    required=False,
    type=click.IntRange(min=5),
    show_default=True,
    help=(
        "Set the number of permutations use to compute the empirical "
        "p-value for the colocalization score"
    ),
)
@click.option(
    "--colocalization-min-region-count",
    default=5,
    required=False,
    type=click.IntRange(min=0),
    show_default=True,
    help=(
        "The minimum number of counts in a region for it to be considered "
        "valid for computing colocalization"
    ),
)
@click.option(
    "--colocalization-min-marker-count",
    default=5,
    required=False,
    type=click.IntRange(min=0),
    show_default=True,
    help=("The minimum number of marker counts in component for colocalization"),
)
@output_option
@click.pass_context
@timer
def analysis(
    ctx,
    pxl_file,
    compute_polarization,
    compute_colocalization,
    use_full_bipartite,
    polarization_transformation,
    polarization_n_permutations,
    polarization_min_marker_count,
    colocalization_transformation,
    colocalization_neighbourhood_size,
    colocalization_n_permutations,
    colocalization_min_region_count,
    colocalization_min_marker_count,
    output,
):
    """
    Perform different analyses on a PixelDataset from pixelator annotate
    """
    # log input parameters
    input_files = [pxl_file]

    log_step_start(
        "analysis",
        input_files=input_files,
        output=output,
        compute_polarization=compute_polarization,
        compute_colocalization=compute_colocalization,
        polarization_transformation=polarization_transformation,
        polarization_n_permutations=polarization_n_permutations,
        polarization_min_marker_count=polarization_min_marker_count,
        colocalization_transformation=colocalization_transformation,
        colocalization_neighbourhood_size=colocalization_neighbourhood_size,
        colocalization_n_permutations=colocalization_n_permutations,
        colocalization_min_region_count=colocalization_min_region_count,
        colocalization_min_marker_count=colocalization_min_marker_count,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files, allowed_extensions="pxl")

    # sanity check on the anlyses
    if not any([compute_polarization, compute_colocalization]):
        logger.warning("All the analysis are disabled, no scores will be computed")

    # create output folder if it does not exist
    analysis_output = create_output_stage_dir(output, "analysis")

    # compute graph/clusters using parallel processing
    logger.info(f"Computing analysis for file {pxl_file}")

    clean_name = get_sample_name(pxl_file)
    metrics_file = analysis_output / f"{clean_name}.report.json"

    write_parameters_file(
        ctx,
        analysis_output / f"{clean_name}.meta.json",
        command_path="pixelator single-cell-mpx analysis",
    )

    analysis_to_run = []
    if compute_polarization:
        logger.info("Polarization score computation is activated")
        analysis_to_run.append(
            PolarizationAnalysis(
                transformation_type=polarization_transformation,
                n_permutations=polarization_n_permutations,
                min_marker_count=polarization_min_marker_count,
            )
        )

    if compute_colocalization:
        logger.info("Colocalization score computation is activated")
        analysis_to_run.append(
            ColocalizationAnalysis(
                transformation_type=colocalization_transformation,
                neighbourhood_size=colocalization_neighbourhood_size,
                n_permutations=colocalization_n_permutations,
                min_region_count=colocalization_min_region_count,
                min_marker_count=colocalization_min_marker_count,
            )
        )

    analyse_pixels(
        input=pxl_file,
        output=str(analysis_output),
        output_prefix=clean_name,
        metrics_file=str(metrics_file),
        use_full_bipartite=use_full_bipartite,
        analysis_to_run=analysis_to_run,
    )

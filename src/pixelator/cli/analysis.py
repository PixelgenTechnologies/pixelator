"""
Console script for pixelator (analysis)

Copyright (c) 2022 Pixelgen Technologies AB.
"""

from concurrent import futures
from typing import get_args

import click

from pixelator.analysis import analyse_pixels
from pixelator.analysis.colocalization.types import TransformationTypes
from pixelator.cli.common import logger, output_option
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
    "analysis",
    short_help="compute different downstream analyses on annotated samples",
    options_metavar="<options>",
)
@click.argument(
    "input_files",
    nargs=-1,
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
    "--polarization-normalization",
    default="clr",
    required=False,
    type=click.Choice(["raw", "clr", "denoise"]),
    show_default=True,
    help=(
        "Which approach to use to normalize the antibody counts:"
        " \n\traw will use the raw counts\n\tclr will use the"
        " CLR transformed counts\n\tdenoise will use"
        " CLR transformed counts and subtract the counts of the"
        " control antibodies"
    ),
)
@click.option(
    "--polarization-binarization",
    required=False,
    is_flag=True,
    help=(
        "Transform the antibody counts to 0-1 (binarized) when computing"
        " polarization scores"
    ),
)
@click.option(
    "--colocalization-transformation",
    default="log1p",
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
        "Select the size of the neighborhood to use when computing"
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
        "Set the number of permutations use to compute the empirical"
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
        "The minimum number of counts in a region for it to be considered"
        "valid for computing colocalization"
    ),
)
@output_option
@click.pass_context
@timer
def analysis(
    ctx,
    input_files,
    compute_polarization,
    compute_colocalization,
    use_full_bipartite,
    polarization_normalization,
    polarization_binarization,
    colocalization_transformation,
    colocalization_neighbourhood_size,
    colocalization_n_permutations,
    colocalization_min_region_count,
    output,
):
    """
    Perform different analyses on a PixelDataset from pixelator annotate
    """
    # log input parameters
    log_step_start(
        "analysis",
        input_files=input_files,
        output=output,
        compute_polarization=compute_polarization,
        compute_colocalization=compute_colocalization,
        normalization=polarization_normalization,
        binarization=polarization_binarization,
        colocalization_transformation=colocalization_transformation,
        colocalization_neighbourhood_size=colocalization_neighbourhood_size,
        colocalization_n_permutations=colocalization_n_permutations,
        colocalization_min_region_count=colocalization_min_region_count,
    )

    # some basic sanity check on the input files
    sanity_check_inputs(input_files, allowed_extensions="pxl")

    # sanity check on the anlyses
    if not any([compute_polarization, compute_colocalization]):
        msg = "All the analysis are disabled, no scores will be computed"
        click_echo(msg)
        logger.warning(msg)

    # create output folder if it does not exist
    analysis_output = create_output_stage_dir(output, "analysis")

    # compute graph/clusters using parallel processing
    with futures.ProcessPoolExecutor(max_workers=ctx.obj["CORES"]) as executor:
        jobs = []
        for zip_file in input_files:
            msg = f"Computing analysis for file {zip_file}"
            click_echo(msg)
            logger.info(msg)

            clean_name = get_sample_name(zip_file)
            metrics_file = analysis_output / f"{clean_name}.report.json"

            write_parameters_file(
                ctx,
                analysis_output / f"{clean_name}.meta.json",
                command_path="pixelator single-cell analysis",
            )

            jobs.append(
                executor.submit(
                    analyse_pixels,
                    input=zip_file,
                    output=str(analysis_output),
                    output_prefix=clean_name,
                    metrics_file=str(metrics_file),
                    compute_polarization=compute_polarization,
                    compute_colocalization=compute_colocalization,
                    use_full_bipartite=use_full_bipartite,
                    polarization_normalization=polarization_normalization,
                    polarization_binarization=polarization_binarization,
                    colocalization_transformation=colocalization_transformation,
                    colocalization_neighbourhood_size=colocalization_neighbourhood_size,
                    colocalization_n_permutations=colocalization_n_permutations,
                    colocalization_min_region_count=colocalization_min_region_count,
                    verbose=ctx.obj["VERBOSE"],
                )
            )

        for job in futures.as_completed(jobs):
            if job.exception() is not None:
                raise job.exception()

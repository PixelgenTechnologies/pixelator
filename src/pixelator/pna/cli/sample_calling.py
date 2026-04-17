"""Console script for sample calling in antibody-hashed datasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from pathlib import Path

import click
import polars as pl

from pixelator.common.utils import (
    create_output_stage_dir,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.pna import read
from pixelator.pna.cli.common import output_option
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.sample_calling import (
    create_final_report,
    sample_calling,
    warn_if_undetermined_has_high_confidence,
)
from pixelator.pna.sample_calling.hash_antibodies import HashedAntibodyMapping
from pixelator.pna.sample_calling.report import (
    SampleCallingSampleReport,
)


@click.command(
    "sample-calling",
    short_help=("Map components to samples in antibody-hashed datasets."),
    options_metavar="<options>",
)
@click.argument(
    "input_pxl_file",
    required=True,
    type=click.Path(exists=True),
    metavar="INPUT_PXL_FILE",
)
@click.option(
    "--samplesheet",
    required=True,
    type=click.Path(),
    help="Path to a samplesheet file with a hash_index column.",
)
@click.option(
    "--remove-incompatible",
    is_flag=True,
    default=False,
    help="Remove antibodies that are incompatible with their component's called sample.",
)
@click.option(
    "--save-undetermined",
    is_flag=True,
    default=False,
    help="Save components that could not be confidently assigned to any sample.",
)
@click.option(
    "--confidence-threshold",
    required=False,
    type=float,
    default=0.9,
    help="Confidence threshold for sample calling. "
    "Components with a sample confidence below this threshold will be considered undetermined. "
    "Default is 0.9.",
)
@output_option
@click.pass_context
@timer
def sample_calling_cli(
    ctx,
    input_pxl_file: str,
    samplesheet: str,
    remove_incompatible: bool,
    save_undetermined: bool,
    confidence_threshold: float,
    output,
):
    """Map components to samples in sample-hashed datasets."""
    log_step_start(
        "sample-calling",
        input_files=input_pxl_file,
        samplesheet=samplesheet,
        output=output,
        remove_incompatible=remove_incompatible,
        save_undetermined=save_undetermined,
        confidence_threshold=confidence_threshold,
    )
    # some basic sanity check on the input files
    sanity_check_inputs(input_files=input_pxl_file, allowed_extensions=("pxl",))

    sample_calling_output = create_output_stage_dir(output, "sample_calling")

    pool_name = Path(input_pxl_file).name.split(".")[0]
    undetermined_sample_name = f"{pool_name}_undetermined"

    panel_info = PNAAntibodyPanel.from_pxl_file(input_pxl_file)
    hashing_antibodies_in_panel = set(
        panel_info.df[panel_info.df["sample_hashing"] == "yes"].index.to_list()
    )
    samplesheet_df = pl.read_csv(samplesheet)
    if undetermined_sample_name in samplesheet_df["sample"].to_list():
        raise ValueError(
            f"The sample '{undetermined_sample_name}' is not allowed in the samplesheet as it "
            "is reserved for undetermined components. Please edit your "
            "samplesheet to use a different sample name."
        )

    hashed_antibodies = HashedAntibodyMapping.from_samplesheet(
        samplesheet_df,
        all_hashing_antibodies=hashing_antibodies_in_panel,
        pool_name=pool_name,
    )

    pxl = read(Path(input_pxl_file))
    output_files = sample_calling(
        input_pxl=pxl,
        hashing_antibody_mapping=hashed_antibodies,
        output_folder=sample_calling_output,
        remove_incompatible=remove_incompatible,
        confidence_threshold=confidence_threshold,
        undetermined_sample_name=undetermined_sample_name,
    )

    for pxl_file in output_files:
        sample_name = get_sample_name(pxl_file)
        pxl = read(pxl_file)
        write_parameters_file(
            ctx,
            sample_calling_output / f"{sample_name}.meta.json",
            command_path="pixelator single-cell-pna sample-calling",
        )
        metrics = sample_calling_output / f"{sample_name}.report.json"
        report = SampleCallingSampleReport(
            sample_id=sample_name,
            product_id="single-cell-pna",
            number_of_components=len(pxl.components()),
            number_of_incompatible_hashes_removed=(
                pxl.adata().obs["removed_incompatible_hashes"].sum()
            ),
        )
        report.write_json_file(metrics, indent=4)

    # Create a report with information from all samples (including undetermined)
    final_dataset = read(output_files)
    total_report = create_final_report(
        final_dataset=final_dataset,
    )
    total_report.write_json_file(
        sample_calling_output / "sample_calling.report.json", indent=4
    )

    if undetermined_sample_name in final_dataset.sample_names():
        warn_if_undetermined_has_high_confidence(
            undetermined_sample_confidences=final_dataset.filter(
                samples=undetermined_sample_name
            )
            .adata()
            .obs["sample_confidence"],
            confidence_threshold=confidence_threshold,
        )

    if not save_undetermined:
        undetermined_pxl = (
            sample_calling_output / f"{undetermined_sample_name}.dehashed.pxl"
        )
        undetermined_pxl.unlink(missing_ok=True)

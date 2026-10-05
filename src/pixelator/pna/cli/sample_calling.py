"""Console script for sample calling in antibody-hashed datasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

import logging
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
from pixelator.pna.pixeldataset import NullPxlFileError
from pixelator.pna.pixeldataset.io import PxlFile, read_dataset_panel, write_null_pxl
from pixelator.pna.sample_calling import (
    create_final_report,
    sample_calling,
    warn_if_undetermined_has_high_enrichment,
)
from pixelator.pna.sample_calling.hash_antibodies import HashedAntibodyMapping
from pixelator.pna.sample_calling.report import (
    SampleCallingSampleReport,
    SampleCallingTotalReport,
)

logger = logging.getLogger(__name__)


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
    "--enrichment-threshold",
    required=False,
    type=float,
    default=10.0,
    help="Hash enrichment threshold for sample calling. "
    "Components with a hash enrichment factor below this threshold will be considered undetermined. "
    "Hash enrichment is calculated as the ration between highest hash count and the second highest hash count."
    "Default is 10.0.",
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
    enrichment_threshold: float,
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
        enrichment_threshold=enrichment_threshold,
    )
    # some basic sanity check on the input files
    sanity_check_inputs(input_files=input_pxl_file, allowed_extensions=("pxl",))

    sample_calling_output = create_output_stage_dir(output, "sample_calling")

    pool_name = Path(input_pxl_file).name.split(".")[0]
    undetermined_sample_name = f"{pool_name}_undetermined"

    try:
        panel_info = read_dataset_panel(read(input_pxl_file))
    except NullPxlFileError as exc:
        logger.warning("%s", exc)
        _pass_through_null_sample_calling(
            ctx,
            samplesheet=samplesheet,
            pool_name=pool_name,
            null_reason=exc.reason,
            sample_calling_output=sample_calling_output,
            pool_metadata=PxlFile(Path(input_pxl_file)).metadata(),
        )
        return
    if "sample_hashing" not in panel_info.df.columns:
        raise ValueError(
            "Sample calling requires a sample_hashing column on the panel "
            "so hashing markers can be identified. This panel has no "
            "sample_hashing column."
        )
    hashing_antibodies_in_panel = panel_info.hashing_marker_ids
    samplesheet_df = pl.read_csv(samplesheet)
    _reject_reserved_samplesheet_names(
        samplesheet_df["sample"].to_list(),
        undetermined_sample_name,
    )

    hashed_antibodies = HashedAntibodyMapping.from_samplesheet(
        samplesheet_df,
        all_hashing_antibodies=hashing_antibodies_in_panel,
        pool_name=pool_name,
    )

    input_pxl_dataset = read(Path(input_pxl_file))
    output_files = sample_calling(
        input_pxl=input_pxl_dataset,
        hashing_antibody_mapping=hashed_antibodies,
        output_folder=sample_calling_output,
        remove_incompatible=remove_incompatible,
        enrichment_threshold=enrichment_threshold,
        undetermined_sample_name=undetermined_sample_name,
    )

    for pxl_file in output_files:
        sample_name = get_sample_name(pxl_file)
        sample_pxl = read(pxl_file)
        write_parameters_file(
            ctx,
            sample_calling_output / f"{sample_name}.meta.json",
            command_path="pixelator single-cell-pna sample-calling",
        )
        metrics = sample_calling_output / f"{sample_name}.report.json"
        output_reads = int(sample_pxl.adata().obs["reads_in_component"].sum())
        input_reads = int(
            input_pxl_dataset.filter(components=sample_pxl.adata().obs.index.tolist())
            .adata()
            .obs["reads_in_component"]
            .sum()
        )
        report = SampleCallingSampleReport(
            sample_id=sample_name,
            product_id="single-cell-pna",
            number_of_components=len(sample_pxl.components()),
            number_of_incompatible_hashes_removed=(
                sample_pxl.adata().obs["removed_incompatible_hashes"].sum()
            ),
            input_reads=input_reads,
            output_reads=output_reads,
        )
        report.write_json_file(metrics, indent=4)

    # Create a report with information from all samples (including undetermined)
    final_dataset = read(output_files)
    total_report = create_final_report(
        final_dataset=final_dataset,
        input_reads=int(input_pxl_dataset.adata().obs["reads_in_component"].sum()),
        undetermined_sample_name=undetermined_sample_name,
    )
    total_report.write_json_file(
        sample_calling_output / f"{pool_name}.sample_calling.report.json", indent=4
    )

    if undetermined_sample_name in final_dataset.sample_names():
        warn_if_undetermined_has_high_enrichment(
            undetermined_enrichment_factors=final_dataset.filter(
                samples=undetermined_sample_name
            )
            .adata()
            .obs["hash_enrichment_factor"],
            enrichment_threshold=enrichment_threshold,
            undetermined_sample_name=undetermined_sample_name,
        )

    if not save_undetermined:
        undetermined_pxl = (
            sample_calling_output / f"{undetermined_sample_name}.dehashed.pxl"
        )
        undetermined_pxl.unlink(missing_ok=True)


def _reject_reserved_samplesheet_names(
    sample_names: list, undetermined_sample_name: str
) -> None:
    """Reject samplesheet names reserved for components that were not called."""
    if "undetermined" in sample_names:
        raise ValueError(
            "The sample 'undetermined' is not allowed in the samplesheet as it "
            "is reserved for undetermined components. Please edit your "
            "samplesheet to use a different sample name."
        )
    if undetermined_sample_name in sample_names:
        raise ValueError(
            f"The sample '{undetermined_sample_name}' is not allowed in the samplesheet as it "
            "is reserved for undetermined components. Please edit your "
            "samplesheet to use a different sample name."
        )


def _pass_through_null_sample_calling(
    ctx,
    *,
    samplesheet: str,
    pool_name: str,
    null_reason: str,
    sample_calling_output: Path,
    pool_metadata: dict | None = None,
) -> None:
    """Write a null pxl for every samplesheet sample in this pool.

    A missing or unmatched samplesheet is a configuration error and still
    raises. Samples that the sheet names are kept so they show up downstream.
    Reserved sample names are rejected the same way as a successful run.
    Panel metadata from the pool file is copied onto each null file.
    """
    samplesheet_df = pl.read_csv(samplesheet)
    if "pool" not in samplesheet_df.columns or "sample" not in samplesheet_df.columns:
        raise ValueError(
            "The samplesheet must contain 'pool' and 'sample' columns to "
            "pass a null pxl file through sample calling."
        )
    _reject_reserved_samplesheet_names(
        samplesheet_df["sample"].to_list(),
        f"{pool_name}_undetermined",
    )
    sample_names = samplesheet_df.filter(pl.col("pool") == pool_name)[
        "sample"
    ].to_list()
    if not sample_names:
        raise ValueError(
            f"No matching entries found in samplesheet for pool '{pool_name}'."
        )

    for sample_name in sample_names:
        target = sample_calling_output / f"{sample_name}.dehashed.pxl"
        write_null_pxl(
            target,
            sample_name=str(sample_name),
            reason=null_reason,
            source_metadata=pool_metadata,
        )
        write_parameters_file(
            ctx,
            sample_calling_output / f"{sample_name}.meta.json",
            command_path="pixelator single-cell-pna sample-calling",
        )
        report = SampleCallingSampleReport(
            sample_id=str(sample_name),
            product_id="single-cell-pna",
            number_of_components=0,
            number_of_incompatible_hashes_removed=0,
            input_reads=0,
            output_reads=0,
            status="failed",
            null_reason=null_reason,
        )
        report.write_json_file(
            sample_calling_output / f"{sample_name}.report.json", indent=4
        )

    total_report = SampleCallingTotalReport(
        sample_id="all",
        product_id="single-cell-pna",
        number_of_components=0,
        percentage_of_components_successfully_called=0.0,
        hash_enrichment_factors_per_sample={},
        input_reads=0,
        output_reads=0,
        status="failed",
        null_reason=null_reason,
    )
    total_report.write_json_file(
        sample_calling_output / f"{pool_name}.sample_calling.report.json",
        indent=4,
    )

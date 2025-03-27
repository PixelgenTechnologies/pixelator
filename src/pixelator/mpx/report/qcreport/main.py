"""Functions to create a summary interactive web report.

Copyright © 2022 Pixelgen Technologies AB.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from pixelator import __version__
from pixelator.mpx.report.common import PixelatorReporting
from pixelator.mpx.report.models.report_metadata import SampleMetadata
from pixelator.mpx.report.qcreport.builder import QCReportBuilder
from pixelator.mpx.report.qcreport.collect import collect_report_data
from pixelator.mpx.report.qcreport.types import Metrics, SampleInfo

logger = logging.getLogger("pixelator.report")


def create_dynamic_report(
    reporting: PixelatorReporting,
    sample_name: str,
    info: SampleInfo,
    output_path: Path,
) -> None:
    """Create a dynamic customer QC report for a single sample.

    A helper function to create a dynamic web report of a single sample.
    The function uses a template stored in 'report/qcreport/template.html'
    as a base template which is filled with the different metrics and stats.

    :param reporting: A PixelatorReporting object
    :param sample_name: the name of the sample
    :param info: A dictionary containing meta information about the sample
    :param output_path: the path to the output folder
    """
    sample_id = info.sample_id
    logger.info("Creating QC report for sample %s", sample_id)

    amplicon_metrics = reporting.amplicon_metrics(sample_name)
    preqc_metrics = reporting.preqc_metrics(sample_name)
    demux_metrics = reporting.demux_metrics(sample_name)
    collapse_metrics = reporting.collapse_metrics(sample_name)
    graph_metrics = reporting.graph_metrics(sample_name)
    annotate_metrics = reporting.annotate_metrics(sample_name)
    reads_flow = reporting.reads_flow(sample_name)
    molecules_flow = reporting.molecules_flow(sample_name)

    # Collect antibody metrics
    antibodies_data_values = {
        "antibody_reads": reads_flow.valid_antibody_read_count,
        "antibody_reads_usable_per_cell": annotate_metrics.read_count_per_cell_stats.mean,
        "antibody_reads_in_outliers": annotate_metrics.reads_in_aggregates_count,
        "unrecognized_antibodies": demux_metrics.unrecognised_antibody_read_count,
    }

    total_reads_per_cell = reads_flow.input_read_count / annotate_metrics.cell_count
    antibodies_data_fractions = {
        "fraction_antibody_reads": reads_flow.fraction_valid_antibody_reads,
        "fraction_antibody_reads_usable_per_cell": (
            annotate_metrics.read_count_per_cell_stats.mean / total_reads_per_cell
        ),
        "fraction_antibody_reads_in_outliers": annotate_metrics.fraction_reads_in_aggregates,
        "fraction_unrecognized_antibodies": demux_metrics.fraction_unrecognised_antibody_reads,
    }

    placeholder_cell_predictions = {
        "predicted_cell_type_b_cells": None,
        "fraction_predicted_cell_type_b_cells": None,
        "predicted_cell_type_cd4p_cells": None,
        "fraction_predicted_cell_type_cd4p_cells": None,
        "predicted_cell_type_cd8p_cells": None,
        "fraction_predicted_cell_type_cd8p_cells": None,
        "predicted_cell_type_monocytes": None,
        "fraction_predicted_cell_type_monocytes": None,
        "predicted_cell_type_nk_cells": None,
        "fraction_predicted_cell_type_nk_cells": None,
        "predicted_cell_type_unknown": None,
        "fraction_predicted_cell_type_unknown": None,
    }

    # The total number of input reads per cell.
    # Note that this is not the number of actual reads that were assigned to the cell.
    # This gives an idea of how many reads are lost in the pipeline averaged over
    # the number of cells
    average_input_read_per_cell = (
        reads_flow.input_read_count / annotate_metrics.cell_count
    )

    # Map pixelator metrics to QC report metrics
    metrics = Metrics(
        number_of_cells=annotate_metrics.cell_count,
        fraction_outlier_cells=annotate_metrics.fraction_aggregate_components,
        average_reads_usable_per_cell=int(
            annotate_metrics.read_count_per_cell_stats.mean
        ),
        average_reads_per_cell=int(average_input_read_per_cell),
        average_antibody_molecules_per_cell=annotate_metrics.molecule_count_per_cell_stats.mean,
        average_upias_per_cell=round(
            annotate_metrics.a_pixel_count_per_cell_stats.mean
        ),
        average_umis_per_upia=round(
            annotate_metrics.molecule_count_per_a_pixel_per_cell_stats.mean
        ),
        fraction_reads_in_cells=reads_flow.fraction_reads_in_cells,
        fraction_discarded_umis=molecules_flow.fraction_molecules_discarded,
        total_unique_antibodies_detected=annotate_metrics.marker_count,
        number_of_reads=reads_flow.input_read_count,
        number_of_short_reads_skipped=preqc_metrics.too_short_read_count,
        fraction_valid_pbs=reads_flow.fraction_valid_pbs_reads,
        fraction_valid_umis=reads_flow.fraction_reads_in_molecules,
        average_reads_per_molecule=graph_metrics.read_count_per_molecule_stats.mean,
        sequencing_saturation=collapse_metrics.fraction_duplicate_reads,
        fraction_q30_bases_in_antibody_barcode=amplicon_metrics.fraction_q30_bc,
        fraction_q30_bases_in_umi=amplicon_metrics.fraction_q30_umi,
        fraction_q30_bases_in_upia=amplicon_metrics.fraction_q30_upia,
        fraction_q30_bases_in_upib=amplicon_metrics.fraction_q30_upib,
        fraction_q30_bases_in_pbs1=amplicon_metrics.fraction_q30_pbs1,
        fraction_q30_bases_in_pbs2=amplicon_metrics.fraction_q30_pbs2,
        fraction_q30_bases_in_read=amplicon_metrics.fraction_q30,
        **antibodies_data_values,  # type: ignore
        **antibodies_data_fractions,  # type: ignore
        **placeholder_cell_predictions,  # type: ignore
    )

    data = collect_report_data(reporting.workdir, sample_id)

    # uses the default template
    builder = QCReportBuilder()

    report_path = str(Path(output_path) / f"{sample_id}.qc-report.html")
    with open(report_path, "wb") as f:
        builder.write(f, sample_info=info, metrics=metrics, data=data)

    logger.debug("QC report created in %s", output_path)


def make_report(
    input_path: Path,
    output_path: Path,
    panel: Optional[str],
    metadata: Optional[str],
    verbose: Optional[bool],
) -> None:
    """Parse every stage metrics and interactive plots to a single HTML file.

    This function iterates the different steps (preqc, adapterqc, demux, collapse,
    graph and annotate) for a given processed dataset (folder). The function will parse
    the metrics of all the samples present in each subfolder (step). These metrics will
    then be converted to tables. The function will also generate interactive plotly
    figures using the output data present in the graph/annotate steps. All the tables
    and figures will be saved to files and used to generate individual html report for
    each sample (web reports).

    :param input_path: the path to results folder containing all the steps
    :param output_path: the path to the output folder
    :param panel: a path to a panel file or a panel name to load from the config
    :param metadata: a path to a metadata file
    :param verbose: run if verbose mode when true
    :returns: None
    :rtype: None
    :raises AssertionError: when the input is not valid
    :raises RuntimeError: when the metadata is not valid
    """
    from pixelator.mpx.config import config, load_antibody_panel

    logger.info("Generating QC reports from %s", input_path)

    # collect and save metrics
    if panel is not None:
        panel_obj = load_antibody_panel(config, panel)
    else:
        panel_obj = None

    reporting = PixelatorReporting(Path(input_path))

    sample_metadata = SampleMetadata.from_csv(metadata) if metadata else None

    # create the dynamic report (one per sample)

    for sample in reporting.samples():
        this_sample_metadata = (
            sample_metadata.get_by_id(sample) if sample_metadata else None
        )
        sample_description = (
            this_sample_metadata.description if this_sample_metadata else ""
        )

        invocation_info = reporting.cli_invocation_info(sample)
        pixel_version = ""

        try:
            collapse_design_option = invocation_info.get_option("collapse", "--design")
            pixel_version = collapse_design_option.value
        except KeyError:
            logger.warning("No design info found for sample %s", sample)

        webreport_panel_name = None
        webreport_panel_version = None

        if panel_obj is not None:
            webreport_panel_name = (
                panel_obj.name if panel_obj.name else str(panel_obj.filename)
            )
            webreport_panel_version = panel_obj.version

        # create SampleInfo object
        sample_info = SampleInfo(
            pixelator_version=__version__,
            generation_date=datetime.utcnow().isoformat(),
            sample_id=sample,
            sample_description=sample_description,
            pixel_version=pixel_version,
            panel_name=webreport_panel_name,
            panel_version=webreport_panel_version,
            parameters=list(invocation_info),
        )

        create_dynamic_report(
            reporting,
            sample_name=sample,
            info=sample_info,
            output_path=output_path,
        )

    logger.info("Finished generating QC reports")
